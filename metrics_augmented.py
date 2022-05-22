# encoding: utf-8

# Tensorflow
import tensorflow as tf
print(tf.__version__)
# import tensorflow_probability as tfp
# tfd = tfp.distributions
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

# Customized loss function 
from tensorflow.python.ops import clip_ops, math_ops # constant_op

# Misc
import sys, os
import numpy as np
# from numpy import linalg as LA
from typing import Dict, Text
# from collections import namedtuple


# For seq2seq model when `y_true` is augmented (first feature dimension: mask values, second feature dimension: ratings + class label)
def recall(y_true, y_pred):

    # `y_true` has shape (batch size, sequence_length, n_features) or (bsize, nt, nf) 
    #  - y_true[:, :, 0] is the main supervising signal
    #  - y_true[:, :, 1] is a supplementary signal that helps to construct the loss function with desirable properties
    
    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)

    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    true_positives = K.sum(K.round(K.clip(y_true_mask * y_pred_mask, 0, 1))) # K.clip ensures that tensor elements are bounded within [0, 1]
    possible_positives = K.sum(K.round(K.clip(y_true_mask, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras

def precision(y_true, y_pred):

    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)
    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    true_positives = K.sum(K.round(K.clip(y_true_mask * y_pred_mask, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_mask, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

def specificity(y_true, y_pred):

    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)
    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    tn = K.sum(K.round(K.clip((1 - y_true_mask) * (1 - y_pred_mask), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true_mask) * y_pred_mask, 0, 1)))
    return tn / (tn + fp + K.epsilon())

def negative_predictive_value(y_true, y_pred):

    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)
    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    tn = K.sum(K.round(K.clip((1 - y_true_mask) * (1 - y_pred_mask), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true_mask * (1 - y_pred_mask), 0, 1)))
    return tn / (tn + fn + K.epsilon())

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def fbeta(y_true, y_pred, beta=2):
    """

    Reference 
    ---------
    1. https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
    """
    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)
    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    y_pred_mask = K.clip(y_pred_mask, 0, 1)

    tp = K.sum(K.round(K.clip(y_true_mask * y_pred_mask, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred_mask - y_true_mask, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true_mask - y_pred_mask, 0, 1)), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    num = (1.0 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + K.epsilon())
    return K.mean(num / den)

def accuracy(y_true, y_pred): 
    from keras.utils import metrics_utils

    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)

    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    tp = K.sum(K.round(K.clip(y_true_mask * y_pred_mask, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true_mask) * (1 - y_pred_mask), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true_mask) * y_pred_mask, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true_mask * (1 - y_pred_mask), 0, 1)))

    return (tp+tn)/(tp+tn+fp+fn)

def matthews_correlation_coefficient(y_true, y_pred):

    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)

    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    tp = K.sum(K.round(K.clip(y_true_mask * y_pred_mask, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true_mask) * (1 - y_pred_mask), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true_mask) * y_pred_mask, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true_mask * (1 - y_pred_mask), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def equal_error_rate(y_true, y_pred):
    mask_seq = y_true[:, :, 0] 
    y_true_mask = mask_values = mask_seq[:, :-1] # shape: (bsize, nt-1)
    predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
    y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)

    n_imp = tf.count_nonzero(tf.equal(y_true_mask, 0), dtype=tf.float32) + tf.constant(K.epsilon())
    n_gen = tf.count_nonzero(tf.equal(y_true_mask, 1), dtype=tf.float32) + tf.constant(K.epsilon())

    scores_imp = tf.boolean_mask(y_pred_mask, tf.equal(y_true, 0))
    scores_gen = tf.boolean_mask(y_pred_mask, tf.equal(y_true, 1))

    loop_vars = (tf.constant(0.0), tf.constant(1.0), tf.constant(0.0))
    cond = lambda t, fpr, fnr: tf.greater_equal(fpr, fnr)
    body = lambda t, fpr, fnr: (
        t + 0.001,
        tf.divide(tf.count_nonzero(tf.greater_equal(scores_imp, t), dtype=tf.float32), n_imp),
        tf.divide(tf.count_nonzero(tf.less(scores_gen, t), dtype=tf.float32), n_gen)
    )
    t, fpr, fnr = tf.while_loop(cond, body, loop_vars, back_prop=False)
    eer = (fpr + fnr) / 2

    return eer