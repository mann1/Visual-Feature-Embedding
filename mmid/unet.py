import pandas as pd
import numpy as np
import re
import glob
import spacy
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import pickle
import csv
import string
import itertools
import os.path
from tqdm import tqdm
from scipy.sparse import load_npz
from keras.models import load_model
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import TFOptimizer, Adam
from keras.models import model_from_json
from keras import regularizers
from keras import layers as ll
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.layers.core import Flatten, RepeatVector, Reshape, Dropout, Masking
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot, multiply, concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.optimizers import TFOptimizer, Adam
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import plot_model, normalize

def cbr(x, out_layer, kernel, stride, dilation):
    x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def se_block(x_in, layer_n):
    x = GlobalAveragePooling1D()(x_in)
    x = Dense(layer_n//8, activation="relu")(x)
    x = Dense(layer_n, activation="sigmoid")(x)
    x_out=Multiply()([x_in, x])
    return x_out

def resblock(x_in, layer_n, kernel, dilation, use_se=False):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = Add()([x_in, x])
    return x

def Unet(input_shape=(None,1)):
    layer_n = 64
    kernel_size = 7
    depth = 2

    input_layer = Input(input_shape)    
    input_layer_1 = AveragePooling1D(5)(input_layer)
    input_layer_2 = AveragePooling1D(25)(input_layer)
    
    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)#1000
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n*2, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*2, kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])    
    x = cbr(x, layer_n*3, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*3, kernel_size, 1)
    out_2 = x

    x = Concatenate()([x, input_layer_2])    
    x = cbr(x, layer_n*4, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n*4, kernel_size, 1)

    model = Model(input_layer, x)
    
    return model
