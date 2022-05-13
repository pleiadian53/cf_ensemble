# encoding: utf-8
import os, sys, re, math, random, time
import numpy as np
import scipy.sparse as sparse

# CF-ensemble-specific libraries
import utils_stacking as ustk
import utils_classifier as uclf
import utils_sys as usys
import utils_cf as uc
import polarity_models as pmodel
from polarity_models import Polarity

# from utilities import softmax

# Sklearn
from sklearn import metrics


def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)

def aggregate(X, aggregate_func=np.mean, axis=0, **kargs):

    if isinstance(aggregate_func, str):
        if aggregate_func.startswith( ('mean', 'ave' ) ):
            return np.mean(X, axis=axis)
        elif aggregate_func.startswith( 'med' ):
            return np.median(X, axis=axis)
        elif aggregate_func.startswith( 'sum' ):
            return np.sum(X, axis=axis)

        elif aggregate_func.startswith( 'maj' ): # majority vote

            # Parameters
            if 'p_threshold' not in kargs:
                raise ValueError("`p_threshold` must be provided to use majority vote.")
            p_threshold = kargs['p_threshold']
            pos_label = kargs.get('pos_label', 1)

            W = pmodel.filter_by_majority_vote(X, p_threshold, pos_label=pos_label, dtype='float')
            # [condition] W should be a binary matrix

            # Note: In this case, it's not possible that any column j of W[:, j] sums to zero, which reuslts in undefined average
            #       => no need to set `fallback_on_low_weight` here

            # Take the mean of Th[i,j] where the corresponding label prediction matches the majority vote
            return uc.predict_by_importance_weights(X, W, aggregate_func='mean', fallback_on_low_weight=False)
        else:
            raise NotImplementedError
    else:
        msg = f"`aggregate_func` should be either a string or a callalble function but received: {aggregate_func} of type {type(aggregate_func)}"
        if not hasattr(aggregate_func, '__call__'):
            raise ValueError(msg)

        return aggregate_func(X, axis=axis)  # e.g. mean prediction of users/classifiers

    return np.mean(X, axis=axis) # return mean aggregate by default

def mask_given_filter(X, P, mask_value=0, replace_by='mean', axis=0, exception=True, **kargs):
    """
    Create a new rating matrix X' with all masked entries (indicated by 0s in P)
    replaced by a new value(s) as indicated by `replace_by` with the following options:

    - 'mean': masked values is to be replaced by column mean (i.e. np.mean(X[:, j])) if axis=0 or row mean if axis=1
    - 'zero'

    """
    import collections

    if X.shape != P.shape:
        raise ValueError(f"shape(X): {X.shape} != shape(P): {P.shape}")

    # P must be a hard filter with 0s representing unwanted values
    if not pmodel.is_hard_filter(P):
        if exception:
            sample_values = np.random.choice(np.ravel(P), 10)
            raise ValueError(f"Input filter must be a hard filter. Example filter values:\n{sample_values}\n")
        else: # Convert to hard filter
            r_th = kargs.get('r_th', 0.5)

            P = pmodel.to_hard_filter(P, r_th, n_codes_ref=None, dtype='int', inplace=False)
            # P = softmax(P, axis=axis) # softmax helps to ensure that P no degenerative cases and that all weights sum to 1

    # [condition] pmodel.is_hard_filter(P) == True

    n_masked = collections.Counter(np.unique(P)).get(mask_value, 0)

    if n_masked > 0:
        if replace_by in (0, 'zero'):
            X[P==mask_value] = 0.0

        elif replace_by.startswith('mean'): # replace masked entries by non-masked means
            if axis == 0:
                for j in range(P.shape[1]):
                    x = X[:, j] # column vector
                    p = P[:, j]
                    v = np.mean(x[p != mask_value]) # take the mean where X isn't masked
                    x[p == mask_value] = v # replace idiom: array[condition]=value
            else:
                for i in range(P.shape[0]):
                    x = X[i, :] # row vector
                    p = P[i, :]
                    v = np.mean(x[p != mask_value])
                    x[p == mask_value] = v # replace idiom: array[condition]=value
        else:
            raise NotImplementedError
    else:
        # no-op
        pass

    return X # default: no-op


def combine_given_filter(X, P, aggregate_func='mean', axis=0, **kargs):
    """
    A specialized combine function.

    For a more general combine function, see `combine()`.
    """
    if X.shape != P.shape:
        raise ValueError(f"shape(X): {X.shape} != shape(P): {P.shape}")
    if sparse.issparse(P): P = P.A

    predictions = np.zeros(X.shape[1])
    if pmodel.is_hard_filter(P):
        predictions = uc.predict_by_importance_weights(X, P, aggregate_func=aggregate_func, fallback_on_low_weight=False)
        # Note: it's okay for `P` to be degenerative (having column- or row-wise filter values equal to 0)
    else: # P is a soft filter, where P[i,j] is a continous value between [0, 1]
        # if np.max(P) > 1.0:
        #       raise ValueError(f"The filter P's entries must be in [0, 1]; observed min(P): {np.min(P)}, max(P): {np.max(P)}")

        # Normalize P such that X * P can be used to compute weighted average with higher X[i, j] having higher weights
        P = softmax(P, axis=axis) # softmax helps to ensure that P no degenerative cases and that all weights sum to 1
        predictions = aggregate(X * P, aggregate_func='sum', axis=axis) # weighted average

    return predictions

def weighted_average(X, W, axis=0):
    W = softmax(W, axis=axis) # softmax helps to ensure that P no degenerative cases and that all weights sum to 1
    return np.sum(X * W, axis=axis) # weighted average

def combine(Th, weights=None, aggregate_func=np.mean, axis=0, **kargs):
    """
    Combine the probabilities in test set to form the final prediction.

    Use
    ---
    1. If `Th` contains re-estimated probabilities

    2. If `Th` contains preference scores (or other scores)

       combiner(Th, aggregate_func='pref', T=T)
    """
    # two cases: Th holds the predictive labels OR Th is a 'rating matrix' (users/classifiers vs items/data)
    nrow = 1
    try:
        nrow, ncol = Th.shape[0], Th.shape[1]  # Th is 2D
    except:
        nrow = 1
        ncol = len(Th)
        Th = Th[np.newaxis, :]

    # W = weights
    predictions = np.zeros(Th.shape[1])
    if isinstance(aggregate_func, str):
        if aggregate_func.startswith('pref'): # `Th` is interpreted as a preference matrix (a binary matrix where 0: not preferred, 1: preferred)
            if ('T' not in kargs) or (kargs['T'] is None): raise ValueError("Missing T")
            T = kargs['T']
            predictions =  combiner_pref(Th, T)  # return predictive scores

        elif aggregate_func.startswith(('mean', 'av')): # mean or average
            if weights is not None:
                print('(combiner) aggregate_func: mean | using predict_by_importance_weights() | n(zeros):{}'.format(np.sum(weights==0)))
                predictions = predict_by_importance_weights(Th, weights, aggregate_func='mean', fallback_on_low_weight=True, min_weight=0.1)
            else:
                predictions = np.mean(Th, axis=axis)  # e.g. mean prediction of users/classifiers

        elif aggregate_func.startswith('med'): # median
            if weights is not None:
                predictions = predict_by_importance_weights(Th, weights, aggregate_func='median', fallback_on_low_weight=True, min_weight=0.1)
            else:
                predictions = np.median(Th, axis=axis)

        elif aggregate_func.startswith('maj'): # majority vote
            # Note that in this mode "weights" is not used but is determined via majority vote
            # print(f"[combine] aggregation method: {aggregate_func}")

            # Take the mean of Th[i,j] where the corresponding label prediction matches the majority vote
            predictions = aggregate(Th, aggregate_func='majority_vote', axis=axis, **kargs)
        else:
            raise NotImplementedError(f"Unknown aggregation function: {aggregate_func}")
    else:
        msg = f"`aggregate_func` should be either a string or a callalble function but received: {aggregate_func} of type {type(aggregate_func)}"
        if not hasattr(aggregate_func, '__call__'):
            raise ValueError(msg)

        if weights is not None:
            predictions = aggregate_func(Th, weights, axis=axis) # A special aggregate function that takes two matrices E.g. weighted_average()
        else:
            predictions = aggregate_func(Th, axis=axis)

    return predictions

def combine_pref(Th, T):
    predictions = []

    n_zero_pref = 0
    for i in range(Th.shape[1]):
        pref = Th[:, i]
        prob = T[:, i]

        s = np.sum(pref)
        assert s <= pref.size

        if s > 0:
            y_score = np.dot(pref, prob) / s
        else:
            print('(combiner_pref) None of the BP prediction are reliable? sum(pref)=0 at data #%d' % i)
            y_score = np.mean(prob)
            n_zero_pref += 1

        predictions.append(y_score)
    if n_zero_pref > 0: print('(combiner_pref) Found %d instances with preference score = 0!' % n_zero_pref)
    return np.array(predictions)

def test():

    return


if __name__ == "__main__":
    test()
