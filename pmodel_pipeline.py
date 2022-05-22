# CF-ensemble-specific libraries
import utils_stacking as ustk
import utils_classifier as uclf
import utils_sys as us
import utils_cf as uc

# Primary dependency
import cf_models as cm
import polarity_models as pmodel

#################################################################

# Plotting
# import matplotlib.pylab as plt
# %matplotlib inline

# Misc
import sys, os
from pandas import DataFrame
import numpy as np
from numpy import linalg as LA

import scipy.sparse as sparse

# import pprint
import tempfile
from typing import Dict, Text
from collections import namedtuple

# Progress
from tqdm import tqdm


# Data & predictive supportive modules for polarity modelings
# E.g., mask prediction, polarity prediction, probability filters
#################################################################

def make_seq2seq_training_data(R, Po=None, L=None, *, include_label=True, **kargs):
    """
    Create mask prediction training data set.

    Parameters
    ----------
    R:  Rating matrix where rows correspond to classifers/users and columns correspond to data points
    Po: A Probability filter where 1 represents reliable entries and 0 represents unreliable entries), whose
        column vectors serve as "labels" for R

        That is Po[:, i] is the label for the corresponding column vector R[:, i] for all i in [0, R.shape[1]]

    L: class labels associated with R

    filter_type: 'mask', 'polarity', 'color'
       - mask (probability filter): uses 0-1 encoding, where 0 represents incorrect predictions (FPs, FNs) and 1 represents correct predictions (TPs, TNs)
       - preference: uses 0-1 encoding but, unlike a mask (or a probability filter), 0 represents "not preferred" whereas 1 "preferred"

         Although 'mask' and 'preference matrix' both uses 0-1 encoding, they are derived differently

       - polarity: similar to "mask" but uses {-1, 1}-encoding

                   if we consider "neutral" particles (e.g. ratings with high uncertainty),
                   then 0 can be a valid value as well, which generalizes to a {-1, 0, 1}-encoding


       - color: color matrix that generalizes polarity matrix and further distingues between:
                TPs (2), TNs (1), FPs (-2), FNs (-1)

                neutral particles are also possible and represented by 0s


    Returns
    -------
    A 2-tuple (X, Y), where
    X:
    Y:
    """
    # import utils_cf as uc

    verbose = kargs.get('verbose', 0)
    filter_type = kargs.get('filter_type', 'mask') # Options: 'mask' ({0, 1}-encoding), 'polarity', ''

    p_threshold = kargs.get('p_threshold', [])
    if Po is None: # If the probability filter is not provided externally, we need to estimate using a default method
        assert len(p_threshold) > 0 and L is not None
        assert (len(p_threshold), len(L)) == R.shape

        if filter_type.startswith( ("mask", "proba", "filter") ): # 0-1 encoding
            Po, _ = pmodel.probability_filter(R, L, p_threshold)
        elif filter_type.startswith( "po" ):
            Po, _ = pmodel.polarity_matrix(R, L, p_threshold) # {-1,1} encoding, possibly including 0
        elif filter_type == "color":
            Po, _ = pmodel.color_matrix(R, L, p_threshold) # {-2, -1, 1, 2} encoding; possibly including 0 and more

        elif filter_type.startswith("pref"):
            raise NotImplementedError("For preference filter, please provide a pre-computed `Po`")
        else:
            raise ValueError(f"Unrecognized filter type: {filter_type}")

    if L is None:
        if include_label: raise ValueError("Missing class labels (L)")
    else:
        if len(L) != R.shape[1]: raise ValueError(f"Size of the labels ({len(L)}) must match number of columns in R ({R.shape[1]}).")

    if R.shape != Po.shape: raise ValueError(f"shape(R): {R.shape} does not match shape(Po): {Po.shape}")

    if sparse.issparse(Po): Po = Po.A

    X = [] # input sequences (one training instance consists of `n_users` ratings per data point, where `n_users` = R.shape[0])
    Y = [] # labels or output sequences

    if not include_label:
        for j in range(R.shape[1]):

            X.append( R[:, j].reshape(-1, 1) ) # number of "timesteps" is the number of classifiers/users, where ...
            # ... each object/rating is represented by 1 feature value, which is the rating itself

            Y.append(Po[:, j].reshape(-1, 1) ) # `X -> Y` maps a sequence of ratings to a sequence of polarities
    else:
        assert len(p_threshold) > 0 and L is not None

        # Either use zero-padding or compute a label place older for the training instances
        # (LSTM requires that the input and output sequence have fixed, pre-specified lengths)

        # Why do we use a placeholder (or a heuristic label) instead of the true label?
        L_heuristic = kargs.get('L_heuristic', "majority_vote")

        # Indeed we know `L_train` but we may not want to rely on it too much. After the model is trained at the prediction/test time,
        # we will no longer have the labeling information (i.e. when calling model.predict(X_test));
        # this means that we cannot make the test instances with the label in it (because ultimately the class label for T is what
        # we are aiming to predict)
        #
        # What we could do instead is to infer the labeling using the ratings and probability thresholds, which are equally accessible
        # during the prediction/test time

        # Heusristic #1: Majority vote
        if isinstance(L_heuristic, str):
            if L_heuristic.startswith( ('maj', 'max') ): # majority vote
                L_heuristic = uc.estimateLabels(R, p_th=p_threshold) # this heuristically guessed labeling by default uses majority vote
            elif L_heuristic.startswith( 'zero' ): # zero-padding
                L_heuristic = np.zeros(R.shape[1]) # [design] we could instead just assign 0 in the loop
            else:
                raise NotImplementedError(f"Unrecognized label heuristic: {L_heuristic}")
        else:
            msg = f"Invalid `L_heuristic` (hint: L_heuristic should be a list/ndarray of same size as R.shape[1]): {L_heuristic}"
            assert isinstance(L_heuristic, (list, np.ndarray)) and len(L_heuristic) == R.shape[1], msg

        for j in range(R.shape[1]):
            X.append( np.vstack( (R[:, j].reshape(-1, 1), L_heuristic[j]) )) # Append either a heustically guessed label or zero-pad at the end of the rating sequence ...
            # Note:
            # Why don't we add class label to X? Because at the prediction time, we will not know the class label (which is to be predicted)

            Y.append( np.vstack( (Po[:, j].reshape(-1, 1), L[j]) )) # Also append the class label L[j] and assume positive class as having positive polarity

    X = np.array(X).astype('float')
    Y = np.array(Y).astype('float')
    if verbose:
        print(f"[info] shape(X): {X.shape}, shape(Y): {Y.shape}")

    return (X, Y) # X and Y should be in 3D, which is required by TF's LSTM
def predict_filter(model_seq, X, batch_size):
    """
    Given trained a sequence model, predict the test sequence data (X) having the following shape:

    (batch_size, len_seq, n_features)

    Using numpy's notation, we denote the i-th sequence of the batch in X as X[i, :, 0].
    The sequence is composed of two parts (where i is an index into the batch or i-th data instance):

    1) BP-generated probability scores (aka ratings); there are (len_seq - 1) probabilities as there are this many BPs.
    2) initial guess of the class label: e.g. labeling produced by majority vote

    Parameters
    ----------
    model_seq: A trained sequence model that provides a predict() method


    Returns
    -------
    Y: output sequence prediction with the same dimensioinality as X: (batch_size, len_seq, n_features)
    """
    Y = model_seq.predict(X, batch_size=batch_size)
    return Y

def make_seq2seq_training_data2(R, Po=None, L=None, include_label=True, **kargs):
    import utils_cf as uc

    verbose = kargs.get('verbose', 0)
    filter_type = kargs.get('filter_type', 'mask') # Options: 'mask' ({0, 1}-encoding), 'polarity', ''
    augmented = kargs.get('augmented', False) # if True, pack additional information to `Y` which help determine the loss

    p_threshold = kargs.get('p_threshold', [])
    if Po is None: # If the probability filter is not provided externally, we need to estimate using a default method
        assert len(p_threshold) > 0 and L is not None
        assert (len(p_threshold), len(L)) == R.shape

        if filter_type.startswith( ("mask", "proba", "filter") ): # 0-1 encoding
            Po, _ = pmodel.probability_filter(R, L, p_threshold)
        elif filter_type.startswith( "po" ):
            Po, _ = pmodel.polarity_matrix(R, L, p_threshold) # {-1,1} encoding, possibly including 0
        elif filter_type == "color":
            Po, _ = pmodel.color_matrix(R, L, p_threshold) # {-2, -1, 1, 2} encoding; possibly including 0 and more

        elif filter_type.startswith("pref"):
            raise NotImplementedError("For preference filter, please provide a pre-computed `Po`")
        else:
            raise ValueError(f"Unrecognized filter type: {filter_type}")

    if L is not None: assert len(L) == R.shape[1]

    assert R.shape == Po.shape

    if sparse.issparse(Po): Po = Po.A

    X = [] # input sequences (one training instance consists of `n_users` ratings per data point, where `n_users` = R.shape[0])
    Y = [] # labels or output sequences

    if not include_label:
        if not augmented:
            for j in range(R.shape[1]):

                X.append( R[:, j].reshape(-1, 1) ) # number of "timesteps" is the number of classifiers/users, where ...
                # ... each object/rating is represented by 1 feature value, which is the rating itself
                Y.append(Po[:, j].reshape(-1, 1) ) # `X -> Y` maps a sequence of ratings to a sequence of polarities
        else:
            for j in range(R.shape[1]):

                X.append( R[:, j].reshape(-1, 1) ) # number of "timesteps" is the number of classifiers/users, where ...
                # ... each object/rating is represented by 1 feature value, which is the rating itself

                y_aug = np.hstack( (Po[:, j].reshape(-1, 1), R[:, j].reshape(-1, 1)) )
                Y.append( y_aug )


    else:
        if not augmented:
            for j in range(R.shape[1]):
                X.append( np.vstack( (R[:, j].reshape(-1, 1), L[j]) )) # L[j] is the ground-truth label for j-th instance

                Y.append( np.vstack( (Po[:, j].reshape(-1, 1), 0)   )) # pad a zero as the element of the output sequence
                # NOTE: padded zeros have no meaning but just to keep the input and output lengths equal
        else:
            for j in range(R.shape[1]):
                X.append( np.vstack( (R[:, j].reshape(-1, 1), L[j])) ) # L[j] is the ground-truth label for j-th instance

                po = np.vstack( (Po[:, j].reshape(-1, 1), 0)) # main supervised signal
                r = np.vstack( (R[:, j].reshape(-1, 1), L[j])) # supplementary info
                y_aug = np.hstack((po, r))
                Y.append( y_aug )

    X = np.array(X).astype('float')
    Y = np.array(Y).astype('float')
    if verbose:
        print(f"[info] shape(X): {X.shape}, shape(Y): {Y.shape}")

    return (X, Y) # X and Y should be in 3D, which is required by TF's LSTM

# [todo]
def predict_filter2(model_seq, X, batch_size, **kargs):
    """
    Given a (trained) sequence model, predict the test sequence data (X) having the following shape:

    (batch_size, len_seq, n_features)


    This version of prediction `predict_filter2()` represents a function that maps the sequences in X (X[:, :, 0]) to
    the output sequence in Y (Y[:, :-1, 0]) that consists of only the target mask values in 0-1 encoding representing reliabliity

    Notice the numpy indexing notation in Y's second dimension `:-1`, which represents all but the last element
    (not ':' which represents the entire sequence)

    The last element (Y[:, -1, 0]) is assumed to be just a zero padding, meaning that Y does not carry the class label.
    This mapping from X to Y is a "mask-only output representation" generated by `make_seq2seq_training_data2()`


    Similar to the input X in predict_filter(), X[i, :, 0] is a also sequence comprising two parts (where i is an index into the batch or i-th data instance):

    1) BP-generated probability scores (aka ratings); there are (len_seq - 1) probabilities as there are this many BPs.
    2) Initial guess of the class label: e.g. labeling produced by majority vote

    Rather than describing this function's behavior as making prediction (as in .predict() operation in sklearn and TF),
    perhaps calling it a "label hypothsis testing" is more appropriate. This sequence model does not predict class label
    directly due to the input sequence representation X -> Y where Y does not carry labeling information as mentioned above.

    """
    import combiner

    n_users = X.shape[1]-1
    n_items = X.shape[0]

    # Parameters
    pos_label = kargs.get("pos_label", 1)
    verbose = kargs.get("verbose", 0)
    mask_aggregate = kargs.get("mask_aggregate", False)
    predict_labels = kargs.get("predict_labels", False)
    predict_proba = False if predict_labels else kargs.get("predict_proba", False)
    p_threshold = kargs.get("p_threshold", None)
    if predict_proba:
        if p_threshold is None:
            raise ValueError("Probability thresholds needed to reduce rating matrix to a prediction vector.")
        else:
            if len(p_threshold) != n_users:
                raise ValueError(f"Length of `p_threshold`: {len(p_threshold)} does not match number of users: {len(n_users)}")

    ##########################################

    # X holds the origina label guess
    y_original = X[:, n_users, 0]

    Y = model_seq.predict(X, batch_size=batch_size)

    # Create alternative labeling hypothesis
    ##########################################
    neg_label = 1 - pos_label
    pos_idx = np.where(y_original==pos_label)[0]
    neg_idx = np.where(y_original==neg_label)[0]

    # Alternative hypothesis by flipping the original label guess
    y_flipped = y_original.copy()
    y_flipped[pos_idx] = neg_label
    y_flipped[neg_idx] = pos_label
    ##########################################

    Xa = X.copy()
    Xa[:, n_users, 0] = y_flipped # the same test-set rating matrix (T) but with the alternative label hypothesis
    Ya = model_seq.predict(Xa, batch_size=batch_size)

    # Which hypothesis is more likely? y_original or y_flipped?

    # "Null" hypothesis
    X0 = X.squeeze().T  # e.g. (1000, 6, 1) -> (1000, 6) -> (6, 1000)
    X0 = X0[:n_users, :]
    Y0 = Y.squeeze().T   # predicted fitler given `X` with labels: y_original
    Y0 = Y0[:n_users, :] # the last element of Y is just zero padding

    assert X0.shape == Y0.shape, f"shape(X0): {X0.shape} != shape(Y0): {Y0.shape}"
    if mask_aggregate: Y0 = pmodel.to_hard_filter(Y0, r_th=0.5, inplace=False)

    # "Alternative" hypothesis
    X1 = Xa.squeeze().T #
    X1 = X1[:n_users, :]
    Y1 = Ya.squeeze().T  # predicted filter given `Xa` with labels: y_flipped
    Y1 = Y1[:n_users, :]
    assert X1.shape == Y1.shape
    if mask_aggregate: Y1 = pmodel.to_hard_filter(Y1, r_th=0.5, inplace=False)

    # print(f"X0: {X0.shape}, Y0: {Y0.shape}, X1: {X1.shape}, Y1: {Y1.shape} ")

    y_adjusted = []
    n_flipped = 0
    for i in range(n_items):

        score0 = score1 = 0
        # The original/null hypothesis
        p = Y0[:, i]
        x = X0[:, i]
        y_pred_i = combiner.combine_given_filter(x[:, None], p[:, None], aggregate_func='mean', axis=0)
        y_orig_i = y_original[i]

        # Is the prediction and the original guess consistent?
        score0 = (y_orig_i - y_pred_i)**2

        # The alternative hypothesis
        p = Y1[:, i]
        x = X1[:, i]
        y_pred_i = combiner.combine_given_filter(x[:, None], p[:, None], aggregate_func='mean', axis=0)
        y_alter_i = y_flipped[i]
        assert (y_orig_i + y_alter_i) == 1

        score1 = (y_alter_i - y_pred_i)**2

        # The more consistent label is probably more accurate (?)
        if score0 <= score1:
            y_adjusted.append(y_orig_i)
        else:
            y_adjusted.append(y_alter_i)
            n_flipped += 1

    if verbose:
        print(f"[info] Found n={n_flipped} different labeling results wrt the original guess")

    # Insert the new guesstimated labels
    y_adjusted = np.array(y_adjusted)
    Xa[:, n_users, 0] = y_adjusted

    # Now make final prediction
    Ya = model_seq.predict(Xa, batch_size=batch_size)

    if predict_labels:
        return (Ya, y_adjusted)

    if predict_proba:
        T = Xa[:, :-1, 0].T  # take transpose such that the rating matrix follows the shape convention: n_users x n_items
        assert T.shape[0] == n_users
        P_adj, _ = pmodel.probability_filter(T, y_adjusted, p_threshold)
        y_proba = combiner.combine_given_filter(T, P_adj, aggregate_func='mean', axis=0)
        return (Ya, y_proba)

    return Ya
