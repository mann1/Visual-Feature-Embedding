"""
Ideas and a small code snippets from these sources:
https://github.com/fchollet/keras/issues/2436
https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
"""

import tensorflow as tf
import keras
from keras import backend as K
import keras.layers as KL
import keras.models as KM
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import nn

############################################################
#  Loss Functions
############################################################
def top_1_accuracy(config, embeddings_positive, embeddings_anchor):
  pred_matrix = math_ops.matmul(embeddings_anchor, embeddings_positive, transpose_a=False, transpose_b=True)
  pred = tf.math.argmax(input = pred_matrix, axis=1)

  labels = tf.range(config.BATCH_SIZE, dtype=tf.int32)
  if labels.dtype != pred.dtype:
    pred = math_ops.cast(pred, labels.dtype)
  is_correct = math_ops.cast(
      math_ops.equal(pred, labels), tf.float32)
  is_correct = logging_ops.Print(is_correct, ['acc:', tf.reduce_mean(is_correct)])
  return tf.reduce_mean(is_correct)


def mmid_Npair_loss_graph(config, reg_lambda, embeddings_positive, embeddings_anchor):
  """Uses npairs_loss in both directions.
  Args:
    pregrasp_embedding: Batch of embeddings of the pregrasp image
    goal_embedding: Batch of embeddings of the goal image
    postgrasp_embedding: Batch of embeddings of the postgrasp image
    params: Parameters for loss. Currently unused.
  Returns:
    A scalar loss
  """
  pair_a = embeddings_positive
  pair_b = embeddings_anchor
  labels = tf.range(config.BATCH_SIZE, dtype=tf.int32)

  pair_a = logging_ops.Print(
          pair_a, ['mean_embedding:', math_ops.reduce_mean(math_ops.reduce_sum(pair_a, 1))])

  loss_1 = tf.contrib.losses.metric_learning.npairs_loss(
      labels, pair_a, pair_b, reg_lambda=reg_lambda, print_losses=True)
  loss_2 = tf.contrib.losses.metric_learning.npairs_loss(
      labels, pair_b, pair_a, reg_lambda=reg_lambda, print_losses=True)
  tf.summary.scalar('npairs_loss1', loss_1)
  tf.summary.scalar('npairs_loss2', loss_2)
  return loss_1+loss_2

############################################################
#  Parallel Model Constructor
############################################################


class ParallelModel(KM.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """

    def __init__(self, keras_model, config):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        super(ParallelModel, self).__init__()
        self.inner_model = keras_model
        self.gpu_count = config.GPU_COUNT
        self.config = config
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        KL.Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
               if name == "MMID_fc":
                  fc_tensor = KL.Concatenate(axis=0, name=name)(outputs)
                  #fc_tensor = KL.Lambda(lambda x: tf.Print(x,["MMID_fc:",tf.shape(x)]))(fc_tensor)
                  continue
               if name == "MMID_wb":
                  wb_tensor = KL.Concatenate(axis=0, name=name)(outputs)
                  #wb_tensor = KL.Lambda(lambda x: tf.Print(x,["MMID_wb:",tf.shape(x)]))(wb_tensor)
                  continue
            merged.append(fc_tensor)
            merged.append(wb_tensor)
            loss = KL.Lambda(lambda x: mmid_Npair_loss_graph(self.config,
                                                             0.02,
                                                             *x), name="MMID_loss")([fc_tensor, wb_tensor])
            acc = KL.Lambda(lambda x: top_1_accuracy(self.config, *x), name="MMID_acc")([fc_tensor, wb_tensor])
            if self.config.MODE == 'training':
               merged.append(loss)
               merged.append(acc)
        return merged

