"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random
import heapq
from scipy.spatial.distance import pdist
from tensorflow.python.ops import math_ops
import pandas as pd
import os.path
from os import path
import math
import re
import time
import numpy as np
import tensorflow as tf
import skimage.color
import skimage.io
import skimage.transform

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import mmid model
sys.path.append(ROOT_DIR)  # To find local version of the library
from mmid.config import Config
from mmid import model as modellib, utils

# Path to trained weights file
PRETRAINED_WEIGHTS_PATH = os.path.join(ROOT_DIR, "cascaded_mask_rcnn_retrain1.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "/opt/ml/model"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class MMIDConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if 
    IMAGES_PER_GPU = 2
    NAME = 'inference'

catalog_path = "/workspace/Cascade_RCNN/data/train_data/catalog/"
instance_path = "/workspace/Cascade_RCNN/data/train_data/instance/"
val_catalog_path = "/workspace/Cascade_RCNN/data/test_data/catalog/"
val_instance_path = "/workspace/Cascade_RCNN/data/test_data/instance/"
FC_test_path = "/workspace/Cascade_RCNN/data/train_data/instance/"
WB_test_path = "/workspace/Cascade_RCNN/data/train_data/catalog/"
virtue_tote = '/workspace/Cascade_RCNN/virtue_tr_tote.csv'

def create_virtue_tote(config):
    FC_test_list = os.listdir(FC_test_path)
    WB_test_list = os.listdir(WB_test_path)

    prefix = "vt_16_"
    test_totes = []
    vt_list = []
    tt_index = 1
    length = config.GPU_COUNT * config.IMAGES_PER_GPU
    while (len(WB_test_list) > 12 ):
        #create a virtue tote
        virtue_tote = []
        for index in range(length):
            asin = random.choice(WB_test_list)
            virtue_tote.append(asin)
            WB_test_list.remove(asin)
        test_totes.append(virtue_tote)
        vt = prefix + str(tt_index)
        vt_list.append(vt)
        tt_index = tt_index + 1
    print(vt_list)
    print(len(test_totes))

    import pandas as pd
    df = pd.DataFrame(test_totes, index=vt_list)
    df.transpose()
    df.to_csv('/workspace/Cascade_RCNN/virtue_tr_tote.csv', index=True)

def inference(mmid_model, vt):
    total_compare_count = 0
    correct_count = 0
    correct_count_3 = 0
    tfconfig = tf.ConfigProto()
    vt_list = pd.read_csv(vt)

    with tf.Session(config=tfconfig).as_default() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i, row in enumerate(vt_list.values):
            #create a virtue tote
            virtue_tote = row[1:]
            index = 0
            vt_len = len(virtue_tote)
            for i, anchor_asin in enumerate(virtue_tote):
                catalog_image = modellib.load_mmid_image(WB_test_path, config, anchor_asin, 'val', augment=False, augmentation=None)
                instance_image = modellib.load_mmid_image(FC_test_path, config, anchor_asin, 'val', augment=False, augmentation=None)
                # Init batch arrays
                if i == 0:
                    catalog_batch_images = np.zeros(
                        (len(virtue_tote),) + catalog_image.shape, dtype=np.float32)

                    instance_batch_images = np.zeros(
                        (len(virtue_tote),) + instance_image.shape, dtype=np.float32)

                # Add to batch
                catalog_batch_images[i] = catalog_image
                instance_batch_images[i] = instance_image
            embeddings = mmid_model.mmid_detect([catalog_batch_images, instance_batch_images], verbose=0)
            print(embeddings)    
            pred_matrix = math_ops.matmul(embeddings[0], embeddings[1], transpose_a=False, transpose_b=True)
            pred = tf.math.argmax(input = pred_matrix, axis=1)
            labels = tf.range(len(virtue_tote), dtype=tf.int32)
            print(sess.run(pred), sess.run(labels))
            if labels.dtype != pred.dtype:
                pred = math_ops.cast(pred, labels.dtype)
            is_correct = math_ops.cast(math_ops.equal(pred, labels), tf.float32)
            print(sess.run(tf.reduce_mean(is_correct)))
            print("=========================")

############################################################
#  Training
############################################################

if __name__ == '__main__':

    class InferenceConfig(MMIDConfig):
         # Set batch size to 1 since we'll be running inference on
         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
         IMAGES_PER_GPU = 2
    config = InferenceConfig()
    config.display()

    mmid_model = modellib.MMID_vec(mode="training", config=config,
                              model_dir='/workspace/Cascade_RCNN/')

    # Load weights
    weights_path = "/opt/ml/model/resnet101_enloss_lr0.05_avg20190725T2036/mmid_resnet101_enloss_lr0.05_avg_0014.h5"
    print("Loading weights ", weights_path)
    
    mmid_model.load_weights(weights_path, by_name=True)

    print("finshed weights loading") 
    print(mmid_model.summary())
    #evaluate
    if not path.exists(virtue_tote):
       create_virtue_tote(config)

    inference(mmid_model, '/workspace/Cascade_RCNN/virtue_tr_tote.csv')
