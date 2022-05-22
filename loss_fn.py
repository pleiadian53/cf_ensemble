# encoding: utf-8

# Tensorflow
import tensorflow as tf
print(tf.__version__)
# import tensorflow_probability as tfp
# tfd = tfp.distributions
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding
# from tensorflow.keras.layers import Concatenate
# from tensorflow.keras.optimizers import RMSprop
# from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

# Customized loss function 
from tensorflow.python.ops import clip_ops, math_ops # constant_op

# Misc
import sys, os
import numpy as np
# from numpy import linalg as LA
from typing import Dict, Text
# from collections import namedtuple


# [todo]
def generalized_bce_loss(multiplier=1.0): 
    """
    ratio: the ratio between the size of majority class to that of the minority class
    """
    pass

# [todo]
def filter_predict_loss(alpha=0.5, weight_multiplier=1.0, r_th=0.5, from_logits=False):
    def filter_predict_loss_core(y_true, y_pred): 
        """

        Dependency
        ----------
        from tensorflow.python.ops import clip_ops, math_ops, constant_op

        Memo
        ----
        # 'x' <- [[1, 1, 1]
        #         [1, 1, 1]]
        tf.reduce_sum(x) ==> 6
        tf.reduce_sum(x, 0) ==> [2, 2, 2]
        tf.reduce_sum(x, 1) ==> [3, 3]
        tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
        tf.reduce_sum(x, [0, 1]) ==> 6

        2. binary cross entropy (different options)
    
           bce = tf.keras.losses.BinaryCrossentropy(from_logits=False) # f(y_true, y_pred) => shape:(), scalar
           
           By default, the following BCE functions will not apply reduce mean: 
           
           bce = tf.keras.metrics.binary_crossentropy
           bce = K.binary_crossentropy

        """
        seq_len = tf.size(y_pred.shape[1]) # shape(y_pred): bsize, n_timesteps (nt), n_features (nf)

        # 1. `y_true` has shape (batch size, sequence_length, n_features) or (bsize, nt, nf) 
        # 
        # 2. `y_true` consists of two parts: 
        # 
        #     first feature dimension: y_true[:, :, 0], which represents the reliabilty/polarity sequence (+ zero padding)
        #     second feature dimension: y_true[:, :, 1], which represents the BP probability scores + class label
        # 
        #     -  y_true[:, :, 0] is the main supervising signal, that is, what we aim to predict given x 
        #        i.e. given BP probability scores, predict their reliability (polarity) scores, with 0 being unreliable 1 being reliable
        #        and these true reliabilty scores are packed into the first feature dimension y_true[:, :, 0]
        # 
        #     -  y_true[:, :, 1] is a supplementary signal that helps to construct the loss function with desirable properties
        # 
        #        In this loss function, we want the mask prediction (which will not be exactly 0s or 1s but a continous value in [0, 1]) 
        #        to not only approximate the 0-1 mask values but also help to predict the class label (which is packed into the last
        #        element of this feature dimension). How does these predicted mask values help to predict the class label?
        #        Well, one possible criterion is that when we softmax-convert them into "weights", their weighted average 
        #        should be as close to the class label as possible. Ratings (BP probabilities) with higher reliabilty are given 
        #        higher weights while ratings with lower reliabilty are given less weights, which is captured by the softmax. 
        #        Why using softmax? Because the ratings are probability scores and therefore we want the weighted ratings remain a 
        #        valid probability; that is, when we take weighted average on the predicted reliability scores, 
        #        the result should still be a valid probability. 
     
        mask_seq = y_true[:, :, 0] # this is what a seq2seq model aims to predict
        # ... shape: (bsize, nt), where each input sequence is a row vector

        # Any training instance is a sequence, which consists of two parts: 1) mask values (reliability of BP predictions) 2) class label
        y_true_mask = mask_values = mask_seq[:, :-1] # (bsize, nt) -> (bsize, nt-1), slices s.t. all but last column is retained
        # ... shape: (bsize, nt-1)

        # dummy = mask_seq[:, -1] # the last element of y_true[:, :, 0] is just a zero padding
        # ... shape: (bsize, )

        # BP probability scores (aka ratings)
        rating_seq = y_true[:, :, 1] # the corresponding BP probability scores, a tagged-on information
        # ... shape: (bsize, nt)

        y_true_rating = ratings = rating_seq[:, :-1] # the rating part of the rating sequence
        # ... shape: (bsize, nt-1)
        labels = rating_seq[:, -1]
        # ... shape: (bsize, )

        # ---------------------------------------------------
        predicted_mask_seq = y_pred[:, :, 0] # shape (bsize, nt)
        y_pred_mask = predicted_mask_seq[:, :-1] # shape (bsize, nt-1)
        
        # --- Loss criterion #1 --- 
        # The sequence of mask values (mask_seq[:, :-1]) and the predicted values (y_pred[:, :, 0]) should combine 
        # the probability scores (ratings) in a similar fashion, such that their results are as close as possible

        # y_pred_mask = tf.where(tf.math.greater_equal(y_pred_mask, r_th), 1, 0)
        # y_pred_mask = tf.dtypes.cast(y_pred_mask, dtype = tf.float64) # cast to float o.w. type mismatch in sum-product operation 

        # mask_sum = K.sum(y_pred_mask, axis=-1) # shape: (bsize,), number of reliable entries (where mask value is 1) for each instance
        # mask_sum_product = K.sum(y_pred_mask * y_true_rating, axis=-1) # shape: (bsize, )
        
        # Take the "masked average" of the probabilities as the final probability prediction, if, however, the mask is degenerative 
        # (all zeros or masked), then take average of all probabilities
        # label_scores = tf.where(K.equal(mask_sum, 0),    # if number of reliable entries is zero
        #                             K.mean(y_true_rating, axis=-1), # take the mean of user ratings (BP probability scores)
        #                                 mask_sum_product/mask_sum )  # otherwise, take the masked average (essentially 0-1 weighted average)
        # [criterion] ref_score ~ label_score? MSE loss 

        # --- Loss criterion #2 --- 
        # the softmax-weighted average of the predicted mask values should be as close as possible to the class label

        # y_pred_dummy = predicted_mask_seq[:, -1] # the last element is just zero padding; its predicted value should be close to 0
        mask_weights = tf.nn.softmax(y_pred_mask, axis=-1) # convert to weights via softmax along the mask-value dimension
        # ... softmax helps to ensure that predicted mask values are summed to 1, resulting in valid weighted average of the ratings 
        # ... that remains a valid probaiblity score 
        # shape: (bsize, nt-1)

        label_scores2 = K.sum(mask_weights * ratings, axis=-1) # the weighted average of the ratings should help predict the label
        # shape: (bsize, )
        
        # ---------------------------------------------------
        # Finally pass the target and output/y_pred to BCE loss 

        # Compute class weights
        weights = tf.where(K.equal(labels, 1), weight_multiplier, 1.0) # minority has a proportionally larger weight (inversely proportional to class sample sizes)
        weights = tf.dtypes.cast(weights, dtype = tf.float64)

        # BCE between true mask and predicted mask
        bce_mask = tf.keras.metrics.binary_crossentropy(y_true_mask, y_pred_mask, axis=-1, from_logits=from_logits) # shape: (bsize, )
        bce_mask = tf.dtypes.cast(bce_mask, dtype = tf.float64)

        # BCE between true labels and mask-predicted labels
        bce_label_pred = K.binary_crossentropy(labels, label_scores2) # shape: (bsize, )
        # tf.keras.metrics.binary_crossentropy(labels, label_scores) # this will take the mean of individual BCE losses and results in a scalar

        bce_mask_weighted = K.sum(bce_mask * weights)/K.sum(weights)
        bce_label_pred_weighted =  K.sum(bce_label_pred * weights)/K.sum(weights)

        return alpha * bce_mask_weighted + (1.0-alpha) * bce_label_pred_weighted
        # return bce_mask_weighted

    return filter_predict_loss_core


def c_squared_loss(y_true, y_pred):

    y_label = y_true[:, 0] # R[i][j] is the "label", which is a probability score in [0, 1]
    weights = y_true[:, 1]
    colors =  y_true[:, 2]
    thresholds = y_true[:, 3]

    mask_tp = tf.dtypes.cast( K.equal(colors, 2), dtype = tf.float64 ) # TP
    mask_tn = tf.dtypes.cast( K.equal(colors, 1), dtype = tf.float64 ) # TN 
    mask_fp = tf.dtypes.cast( K.equal(colors,-2), dtype = tf.float64 ) # FP
    mask_fn = tf.dtypes.cast( K.equal(colors,-1), dtype = tf.float64 ) # FN

    # if TP, want y_pred >= y_true, i.e. the larger (the closer to 1), the better
    loss_tp = weights * K.square(K.maximum(y_label-y_pred, 0)) # if y_pred > y_label => y_label-y_pred < 0 => no loss ...
    # loss_tp = weights * K.square(y_label-y_pred) 
    # ... otherwise, the smaller the y_pred, the higher the penalty (quadratic)
    
    # if TN, want y_pred < y_true, i.e. the smaller (the closer to 0), the better
    loss_tn = weights * K.square(K.maximum(y_pred-y_label, 0)) # if y_label > y_pred => y_pred-y_true < 0 => no loss
    # loss_tn = weights * K.square(y_label-y_pred) 

    # if FP, y_pred must've been too large, want y_pred smaller, but how much smaller? well, it'd better be 
    # smaller than the probability threshold (associated with the corresponding base classifier)
    loss_fp = weights * K.square(K.maximum(y_pred-thresholds, 0)) # if y_pred < p_threshold => no loss
    # loss_fp = weights * K.pow(K.maximum(y_true-y_pred, 0) # may need to be 'a lot' smaller => could penalize error cubically instead

    # if FN, y_pred must've been too small, want y_pred larger; but it needs to be larger than the threshold to be helpful
    loss_fn = weights * K.square(K.maximum(thresholds-y_pred, 0)) # if y_pred > p_threshold => no loss
    # loss_fn = weights * K.pow(K.maximum(y_pred-y_true, 0), 3) # penalize cubically or any exponent > 2

    wmse = K.mean(mask_tp * loss_tp + mask_tn * loss_tn + mask_fp * loss_fp + mask_fn * loss_fn)
    return wmse


def confidence_weighted_loss(y_true, y_pred): # this has to be used with .add_loss() with greater flexibility
    # from tensorflow.keras import backend as K    
    
    y_label = y_true[:, 0]
    weights = y_true[:, 1]
    colors =  y_true[:, 2]
  
    # Conditions
    mask_tp = tf.dtypes.cast( K.equal(colors, 2), dtype = tf.float64 )
    mask_tn = tf.dtypes.cast( K.equal(colors, 1), dtype = tf.float64 )
    mask_fp = tf.dtypes.cast( K.equal(colors,-2), dtype = tf.float64 )
    mask_fn = tf.dtypes.cast( K.equal(colors,-1), dtype = tf.float64 )
    # Note: We also need to convert these masks to numeric type so that we can use them to make the weighted sum

    # if TP, want y_pred >= y_true, i.e. the larger (the closer to 1), the better
    loss_tp = weights * K.square(K.maximum(y_label-y_pred, 0)) # if y_pred > y_label => y_label-y_pred < 0 => no loss ...
    # ... otherwise, the smaller the y_pred, the higher the penalty (quadratic)
    
    # if TN, want y_pred < y_true, i.e. the smaller (the closer to 0), the better
    loss_tn = weights * K.square(K.maximum(y_pred-y_label, 0)) # if y_label > y_pred => y_pred-y_true < 0 => no loss

    # if FP, y_pred must've been too large, want y_pred smaller 
    loss_fp = loss_tn 
    # loss_fp = weights * K.pow(K.maximum(y_true-y_pred, 0) # may need to be 'a lot' smaller => could penalize error cubically instead

    # if FN, y_pred must've been too small, want y_pred larger
    loss_fn = loss_tp 
    # loss_fn = weights * K.pow(K.maximum(y_pred-y_true, 0), 3) # penalize cubically or any exponent > 2

    wmse = K.mean(mask_tp * loss_tp + mask_tn * loss_tn + mask_fp * loss_fp + mask_fn * loss_fn)
    return wmse

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # K: tensorflow backend
    
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

