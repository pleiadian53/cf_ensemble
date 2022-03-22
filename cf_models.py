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

#################################################################

# Scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
#################################################################

# CF-ensemble-specific libraries
import utils_stacking as ustk
import utils_classifier as uclf
import utils_sys as us
import utils_cf as uc 
import utils_knn as uknn
import polarity_models as pmodel

from analyzer import is_sparse # or use sparse.issparse()
import scipy.sparse as sparse
from utils_sys import highlight
#################################################################

# Plotting
import matplotlib.pylab as plt
# %matplotlib inline

# Misc
from pandas import DataFrame
import numpy as np
from numpy import linalg as LA
import pprint
import tempfile
from typing import Dict, Text
from collections import namedtuple

# Progress
from tqdm import tqdm

# np.set_printoptions(precision=3, edgeitems=5, suppress=True)


class CFNet(keras.Model):
    def __init__(self, n_users, n_items, embedding_size, **kwargs):
        super(CFNet, self).__init__(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.user_embedding = Embedding(
            n_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(0.01),
        )
        self.user_bias = Embedding(n_users, 1)

        self.item_embedding = Embedding(
            n_items,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(0.01),
        )
        self.item_bias = Embedding(n_items, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])

        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])

        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

def get_cfnet_uncompiled(n_users, n_items, n_factors): 
    return CFNet(n_users, n_items, n_factors)
def get_cfnet_compiled(n_users, n_items, n_factors, loss=None, lr=0.001):
    if loss is None: loss = tf.keras.losses.MeanSquaredError()
    model = CFNet(n_users, n_items, n_factors)
    model.compile(
        loss=loss, optimizer=keras.optimizers.Adam(lr=lr)
    )
    return model

def get_cfnet_approximating_labels(n_users, n_items, n_factors, lr=0.001): 
    loss_fn = bce = tf.keras.losses.BinaryCrossentropy()
    model = CFNet(n_users, n_items, n_factors)
    model.compile(
        loss=loss_fn, optimizer=keras.optimizers.Adam(lr=lr)
    )
    return model

def get_cfnet_approximating_scores(n_users, n_items, n_factors, lr=0.001): 
    loss_fn = mse = tf.keras.losses.MeanSquaredError()
    model = CFNet(n_users, n_items, n_factors)
    model.compile(
        loss=loss_fn, optimizer=keras.optimizers.Adam(lr=lr)
    )
    return model



class CFNet2(keras.Model):
    pass

# kNN utilities
##########################################################################

def predict_by_knn(model, model_knn, R, T, L_train, L_test, C, Pc, codes={}, pos_label=1, verbose=1): 
    """
    Given an instance of CFNet (model) and an instance of Faiss KNN model (model_knn) along with 
    the training data (R, L_train, C, Pc), reestimate the test data (T). 
    
    Parameters 
    ----------
    model: An instance of CFNet that has been pre-trained 
    model_knn: An instance of Faiss KNN model that has been pre-trained

    R:  probability/rating matrix of the training split
    T: 
    L_train:
    L_test: 
    C: 
    Pc: color matrix of the training split  

    """ 
    if verbose: np.set_printoptions(precision=3, edgeitems=5, suppress=True)

    # Convert rating matrices back to typical ML training set format
    X_train = R.T
    X_test = T.T

    # Find kNNs for each test instances in T
    distances, knn_indices = model_knn.search(X_test)

    N, k = knn_indices.shape # `knn_indices` is a k-by-N matrix, where k as in kNN and N is the sample size 
    n_users = T.shape[0]

    assert N == T.shape[1], f"Size of test set: {T.shape[1]} inconsistent what's inferred from knn indices: {N}"
    assert R.shape == Pc.shape

    if len(codes) == 0: codes = Polarity.codes

    if is_sparse(Pc): Pc = Pc.A #

    # Infer true labels (L_train) from color matrix
    L_train = pmodel.color_matrix_to_labels(Pc, codes=codes) # True labels for R
    n_unreliable_knn_cases = 0
    col_user, col_item, col_value = 'user', 'item', 'rating'

    Th = np.zeros_like(T, dtype='float32') # Initialize the re-estimated test set (Th) for T
    T_knn_best = np.zeros_like(T, dtype='float32')
    T_avg = np.zeros_like(T, dtype='float32')
    T_masked_avg = np.zeros_like(T, dtype='float32')
    Th_reliable = np.zeros_like(T, dtype='float32') # unreliable entries are marked by special number (e.g. 0)

    T_pred = {} # keep track of various predictied outputs according to different strategies
    T_pred['knn_max'] = []

    # kNN top of the top (rank kNNs further by their entropy values, the smaller the better)
    # L_knn, top_indices = uknn.estimate_labels_by_rank(model_knn, T, Pc, topn=min(3, k), 
    #                                                    rank_fn=uknn.compute_entropy, 
    #                                                    larger_is_better=False, 
    #                                                    verbose=0)
    msg = ''
    test_points = np.random.choice(range(N), 10)
    for i in tqdm(range(N)):  # foreach position in the test split (T)
        knn_idx = knn_indices[i] # test point (i)'s k nearest neighbors in R (in terms of their indices)
        # knn_idx = top_indices[i]

        Pc_i = Pc[:, knn_idx].astype(int) # subset the color matrix at kNN indices

        # Method #1 Majority vote: Use the label determined by majority vote within kNNs
        L_knn_i = pmodel.color_matrix_to_labels(Pc_i, codes=codes) # kNN's labels
        ti_knn_max = np.argmax( np.bincount(L_knn_i) ) # kNN-predicted label by majority vote
        # ti_knn_max = L_knn[i]
        T_pred['knn_max'].append(ti_knn_max)

        # Gather statistics
        ni = Pc_i.size # ~ T.size
        ntp = np.sum(Pc_i == codes['tp'])
        ntn = np.sum(Pc_i == codes['tn'])
        nfp = np.sum(Pc_i == codes['fp'])
        nfn = np.sum(Pc_i == codes['fn'])

        if (ntp+ntn)==0: # None of the base classifiers (users) made any correct predictions within these kNNs
            n_unreliable_knn_cases += 1

        # [Test]
        if verbose > 1: 
            msg += f"[info] test point index: {i}\n" + '#' * 50 + '\n'
            msg += f"> T({i}):\n{T[:, i]}\n"
            msg += f"> R({i}):\n{R[:, knn_idx[0]]}\n" # point in R closest to the current test point T[:, i]
            msg += f"> Pc_i(shape={Pc_i.shape}):\n{Pc_i}\n"
            msg += f"> L_knn(size={len(L_knn)}):\n{L_knn}\n"
            msg += f"> label prediction (knn) => {ti_knn_max}\n"

        # Method #2 Best uses: foreach base classifier predictio in ti, use the "best" among these kNNs (majority vote followed by restiamte)
        max_colors, max_indices = [], []
        for u in range(n_users): 
            color, pos = uknn.most_common_element_and_position(Pc_i[u, :], pos_key_only=True)
            max_colors.append(color)
            max_indices.append(knn_idx[pos]) # we want the knn index
        X_knn_best = dp.zip_user_item_pairs(T, item_ids=max_indices)
        y_knn_best = model.predict(X_knn_best)
        T_knn_best[:, i] = np.squeeze(y_knn_best, axis=-1)
        
        # Compute the mask within these kNN part of the training data
        M = np.zeros_like(Pc_i) # np.repeat(Li, Pc_i.size).reshape(Pc_i.shape)
        M[Pc_i > 0] = 1 # polarity > 0 => correct predictions (either TP or TN) => keep their re-estimated values by setting these entries to 1s
        
        # ... polarity < 0 => incorrect predictions => discard by setting them to 0s

        # Get re-estimated values for the kNN (of test instance)
        X_knn = dp.make_user_item_pairs(T, item_ids=knn_idx) # structure k-NN in user-item-pair format for CFNet-based models
        assert X_knn.shape[0] == Pc_i.size
        y_knn = model.predict(X_knn)
        T_knn = y_knn.reshape((n_users, len(knn_idx))) # use len(knn_idx) instead of `k` for the flexibility of selecting fewer candidates
        
        # if i == 10: print(f"[test] knn_idx: {knn_idx}"); print(f"[test] X_knn:\n{X_knn}\n"); print(f"[test] T_knn:\n{T_knn}\n") 
        assert T_knn.shape[1] <= k, f"T_knn[1] == k(NN): {k} but got {T_knn.shape[1]}"
        assert T_knn.shape == Pc_i.shape, f"T_knn is a n_users-by-k matrix but got shape: {T_knn.shape}"
        
        # Method #3 Column Average: Use the average across the re-estimated kNNs
        ti_knn_avg = np.mean(T_knn, axis=1) # take column-wise average (i.e. for each user, take the average among kNNs)
        T_avg[:, i] = ti_knn_avg

        # Method #4 Masked Average: Use the reestimated values w.r.t ONLY those with positive polarity (i.e. averaging from TPs or TNs)
        eps = 1e-4
        ti_knn_masked_avg = (M*T_knn).sum(1)/(M.sum(1)+eps) # average from non-zero entries only
        T_masked_avg[:, i] = ti_knn_masked_avg

        # Method #5 Adjusted Masked Average: Consider degenerative cases in which, for a given base classifier, 
        #           NONE of its predictions in these kNNs are correct
        #           - It's possible that some classifiers never made correct predictions in the context of these kNNs
        #           - Set a default value if that's the case (e.g. average)
        Th[:, i] = np.where(ti_knn_masked_avg == 0, ti_knn_avg, ti_knn_masked_avg)
        
        # Method #6: Mark unreliable entries by -1 (and apply a post-hoc method to Th); post-hoc method is yet to be defined
        Th_reliable[:, i] =  np.where(ti_knn_masked_avg == 0, -1, ti_knn_masked_avg)

    T_pred['T_knn_best'] = T_knn_best # best users
    T_pred['T_avg'] = T_avg # average
    T_pred['T_masked_avg'] = T_masked_avg # masked average
    T_pred['Th'] = Th # adjusted masked average
    T_pred['Th_reliable'] = Th_reliable # -1

    if verbose: 
        print(f"[info] Number of unreliable kNN cases: {n_unreliable_knn_cases}") 
   
    return T_pred



# Loss functions
##########################################################################
def c_squared_loss(y_true, y_pred):

    y_label = y_true[:, 0] # R[i][j] is the "label", which is a probability score in [0, 1]
    weights = y_true[:, 1]
    colors =  y_true[:, 2]
    thresholds = y_true[:, 3]

    mask_tp = tf.dtypes.cast( K.equal(colors, 2), dtype = tf.float32 )
    mask_tn = tf.dtypes.cast( K.equal(colors, 1), dtype = tf.float32 )
    mask_fp = tf.dtypes.cast( K.equal(colors,-2), dtype = tf.float32 )
    mask_fn = tf.dtypes.cast( K.equal(colors,-1), dtype = tf.float32 )

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
    mask_tp = tf.dtypes.cast( K.equal(colors, 2), dtype = tf.float32 )
    mask_tn = tf.dtypes.cast( K.equal(colors, 1), dtype = tf.float32 )
    mask_fp = tf.dtypes.cast( K.equal(colors,-2), dtype = tf.float32 )
    mask_fn = tf.dtypes.cast( K.equal(colors,-1), dtype = tf.float32 )
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


# CFNet model utilities
##########################################################################

def reestimate(model, X, n_train=None, **kargs):
    """
    Use the (TF-trained) model to reconstruct the probaility rating matrix. 

    If `n_train` is not None, re-estiamte the whole matrix X with new probabilities
    using the learned latent factors given by `model` 

    If, however, `n_train` is a number, then re-estimate only the test split of X 
    i.e. X[n_train: ] (typically denoted by `T`). 
    """
    import data_pipeline as dp

    # from pandas import DataFrame
    kargs['shuffle'] = False
    kargs['normalize'] = False

    is_cascade = True # i.e. X = [R|T]
    if n_train is None: # i.e. X = R
        is_cascade = False # if the split point is smaller than sample size, then we know that X consists of two parts
        n_train = X.shape[1]

    # Convert the rating matrix into (X, y)-format that the model accepts
    R, T = X[:,:n_train], X[:,n_train:]    
    X_train, X_test, y_train, y_test = dp.matrix_to_train_test_split(X, n_train, **kargs)
    if not is_cascade: assert X_test.size == 0 and y_test.size == 0, f"X=R but got X_test.size: {X_test.size}, y_test.size: {y_test.size}"

    ####################################
    y_train_pred = model.predict(X_train) # Note: X_train is a dataframe
    y_test_pred =  model.predict(X_test) if is_cascade else np.array([]) # Note: X_test, if available, is a also dataframe  
    ####################################

    # Put the prediction back into the rating-matrix format
    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')
    
    df_train = DataFrame(X_train, columns=[col_user, col_item])
    df_train[col_value] = y_train_pred
    Rh = df_train.pivot(col_user, col_item, col_value).values
    assert Rh.shape == R.shape

    # Th = DataFrame(columns=[col_user, col_item, col_value]) # dummy
    df_test = DataFrame(X_test, columns=[col_user, col_item])
    df_test[col_value] = y_test_pred
    Th = df_test.pivot(col_user, col_item, col_value).values
    assert not is_cascade or (Th.shape == T.shape)
    if not is_cascade: assert Th.size == 0

    return (Rh, Th)

def reestimate_unreliable_only(model, X, *, Pc, C=None, n_train=None, **kargs):
    L = kargs.pop('L', [])
    p_threshold = kargs.pop('p_threshold', [])
    use_confidence_weights = kargs.pop('use_confidence_weights', False) 
    verbose = kargs.pop('verbose', 0)
    assert X.shape == Pc.shape, f"Inconsistent dimensionality: shape(X)={X.shape} <> {Pc.shape}=shape(Pc)"

    Rh, Th = reestimate(model, X, n_train, **kargs)
    Xh = np.hstack((Rh, Th)) if Th.size > 0 else Rh 
    Xhi = interpolate(X, Xh, Pc, C=C, L=L, p_threshold=p_threshold, 
              use_confidence_weights=use_confidence_weights, verbose=verbose)

    return (Xhi[:,:n_train], Xhi[:,n_train:])

def interpolate(X, Xh, Pc=None, C=None, L=[], p_threshold=[], use_confidence_weights=False, verbose=0): 
    # from analyzer import is_sparse
    import utils_cf as uc

    # W = Pc
    if use_confidence_weights: # Use confidence scores as weights 
        assert C is not None
        if is_sparse(C): C = C.toarray()
        W = uc.softmax(C, axis=0)
    else: 
        if Pc is None: 
            W, _ = uc.probability_filter(X, L, p_threshold)
        else: 
            W = Pc.A if is_sparse(Pc) else Pc
            # Note: Why converting to dense? Subtracting a sparse matrix from a nonzero scalar is not supported 
            #       E.g. can't do 1.0-W later if W is sparse

            if verbose > 1: 
                print('(reconstruct) Converting color matrix to a standard probability filter (aka preference matrix) ...')
            W = uc.to_preference(W) # Note: this operation won't affect Pc

    wmin, wmax = np.min(W), np.max(W)
    assert wmin >= 0 and wmax <= 1, "W is not a probability filter | values: [{}, {}]".format(wmin, wmax)

    # Note: 
    # W as a weight matrix: the higher the W[i,j], the more weight on X[i,j]
    # W as a preference matrix: W[i,j] = 1 => use X[i,j] (original value), if W[i,j] = 0, use Xh[i,j] (re-est value)

    # Xh = uc.replace(P, Q, X=(W, X), canonicalize=True, 
    #         fill=null_marker, predict_func=ua.predict_by_factors, name=name)

    # uc.interpolate(X1, X2, W1, W2)
    #   if W1[i,j]=1, then use X1[i, j]
    #   if W1[i,j]=0 => W2[i,j]=1 => use X2[i,j]; effectively replacing X1[i,j] by X2[i, j]
    Xh_partial = uc.interpolate(X, Xh, W, 1.0-W) # replace X by Xs selectively according to W (and 1-W)

    return Xh_partial

def analyze_reestimated_matrices(train, test, meta, **kargs): 
    """
    Analyze and compare performance measures under different algorithmic settings. 
    Similar to analyze_reconstruction() but here we instead assume that the restimated rating matrices 
    have been pre-computed externally and are provided as inputs. 

    """
    # `train`, `test` and `meta` are namedtuples:
    # train: R, L_train 
    # test:  T, L_test 
    # p_policy: p_threshold, policy_threshold 
    # conf_policy: 
    import data_pipeline as dp
    import utils_cf as uc
    import utils_classifier as uclf
    from scipy.spatial import distance

    # Optional Parameters 
    # -------------------
    verbose = kargs.get('verbose', 1)

    reestimated = {}
    scores = []

    # Original data
    R, Rh, L_train = train.X, train.Xh, train.L # [add] train.Pc
    T, Th, L_test = test.X, test.Xh, test.L # [add] test.Pc
    policy_threshold = meta.policy_threshold
    conf_measure = meta.conf_measure
    alpha = meta.alpha
    n_factors = meta.n_factors

    # Probability thresholds associated with the original training data (R)
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy=policy_threshold)

    # Re-estimate the p_threshold as well 
    reestimated['p_threshold2'] = p_threshold_new = uc.estimateProbThresholds(Rh, L=L_train, pos_label=1, policy=policy_threshold)

    if verbose: 
        print(f"[info] From R to Rh, delta(Frobenius norm)= {LA.norm(Rh-R, ord='fro')}")
        print(f"[info] From T to Th, delta(Frobenius norm)= {LA.norm(Th-T, ord='fro')}")
        print(f"[info] From `p_threshold(R)` to `p_threshold(Rh)`, delta(2-norm)= {LA.norm(p_threshold_new-p_threshold, 2)}")
        print(f"...    Original p_threshold:\n{p_threshold}\n")
        print(f"...    New p_threshold:\n{p_threshold_new}\n")

    # Evaluation
    ###################################################
    msg = ""

    # [todo] Try different strategies of reducing T to label predictions
    reestimated['lh_maxvote'] = lh = uc.estimateLabels(T, L=[], p_th=p_threshold, pos_label=1) # "majority vote given proba thresholds" is the default strategy
    reestimated['lh2_maxvote_pth_unadjusted'] = lh_new_orig_pth = \
            uc.estimateLabels(Th, L=[], p_th=p_threshold, pos_label=1) # Use the re-estimated T and original p_th to predict labels
    reestimated['lh2_maxvote_pth_adjusted'] = lh_new = \
            uc.estimateLabels(Th, L=[], p_th=p_threshold_new, pos_label=1) # Use the re-estimated T to predict labels
    msg += f"[info] How different are lh and lh_new? {distance.hamming(lh, lh_new)}\n"


    # 1. Prediction: By majority vote
    ####################################

    # Evaluate using a given performance score (since CF ensemble is primarily targeting imbalance class distributions, 
    # by defeaut, we will use F1 score)
    reestimated['score_lh_maxvote'] = reestimated['score_baseline'] = perf_score = f1_score(L_test, lh)
    scores.append((perf_score , {'lh': lh, 'p_threshold': p_threshold, 'name': 'lh_maxvote'}))
    msg += f'[result] Majority vote: F1 score with the original T:  {perf_score}\n'

    reestimated['score_lh2_maxvote_pth_unadjusted'] = perf_score = f1_score(L_test, lh_new_orig_pth)
    scores.append((perf_score , {'lh': lh_new_orig_pth, 'p_threshold': p_threshold, 'name': 'lh2_maxvote_pth_unadjusted'}))
    msg += f'[result] Majority vote: F1 score with re-estimated Th using original p_threshold: {perf_score}\n'

    reestimated['score_lh2_maxvote_pth_adjusted'] = perf_score = f1_score(L_test, lh_new) 
    scores.append((perf_score , {'lh': lh_new, 'p_threshold': p_threshold_new, 'name': 'lh2_maxvote_pth_adjusted'}))
    msg += f'[result] Majority vote: F1 score with re-estimated Th: {perf_score}\n'

    if verbose: print(msg)

    # 2. Prediction: By stacking
    ####################################
    # Parameters: 
    #   - include_stacking
    #   - stacker 
    #   - grid
    msg = ""
    include_stacking = kargs.get('include_stacking', False) 
    if include_stacking:
        stacker = kargs.get('stacker', None)
        grid = kargs.get('grid', {})
        if stacker is None: 
            stacker = LogisticRegression() 
            grid = uclf.hyperparameter_template('logistic')
        else: 
            assert callable(stacker), f"Invalid meta-classifier: {stacker}"

        lh = uclf.tune_model(stacker, grid, scoring='f1', verbose=0)(R.T, L_train).predict(T.T)
        reestimated['score_lh_stacker'] = f1_score(L_test, lh) 
        msg += f"[result] Stacking: F1 score with the original T:  {reestimated['score_lh_stacker']}\n"

        lh_new = uclf.tune_model(stacker, grid, scoring='f1', verbose=0)(Rh.T, L_train).predict(Th.T)
        reestimated['score_lh2_stacker_pth_adjusted'] = f1_score(L_test, lh_new)
        msg += f"[result] Stacking: F1 score with re-estimated Th: {reestimated['score_lh2_stacker_pth_adjusted']}\n" 

        if verbose: print(msg)

    # 3. Rank parameter settings
    ####################################
    msg = ""
    # Choose the best settings (excluding stacking on re-estiamted matrices)
    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True) 
    if verbose > 1: msg += f"[result] Methods ranked:\n{[(s, d['name']) for s, d in scores_sorted]}\n\n"
    reestimated['best_params'] = scores_sorted[0][1] # select the best setting according to the performance measure

    if verbose: 
        # mode = 'unreliable only' if unreliable_only else 'complete' # 'complete' reestimation or reestimating the entire rating matrix
        msg += f"[result] Best settings: {reestimated['best_params']['name']}, score: {scores_sorted[0][0]}\n"
        print(msg)
    if verbose > 1: 
        print("[help] Reestiamted quantities are available through the following keys:")
        for k, v in reestimated.items(): 
            print(f'  - {k}')

    return reestimated

def analyzer_pipeline(model, X, L, Pc, *, p_threshold=[], policy_threshold='fmax', **kargs):

    # Optional Parameters
    # -------------------
    verbose = kargs.get('verbose', 1)
    L_test = kargs.get('L_test', None) # A validation set to select the best algorithmic settings

    analyzer = analyze_reconstruction(model, 
                                      X=X,  # e.g. (R, T),
                                      L=L,  # e.g. (L_train, L_estimated), 
                                      Pc=Pc, p_threshold=p_threshold, 
                                      policy_threshold=policy_threshold, n_train=n_train)

    # Add conditions here: 
    # 1. Re-estimating the unreliable entries only or re-estimate the entire rating matrix

    highlight("Reestimate the entire rating matrix (X) with learned latent factors/embeddings")
    reestimated = analyzer(L_test, unreliable_only=False, verbose=verbose)
    highlight("Reestimate ONLY the unreliable entries in X with learned latent factors/embeddings")
    reestimated = analyzer(L_test, unreliable_only=True, verbose=2) # use verbose level 2 to show keys of the return value

    # [todo]

    return

def analyze_reconstruction(model, X, L, Pc, n_train=None, p_threshold=[], policy_threshold='fmax'): 
    """

    Parameters 
    ----------
    model: A trained instance of TF-based CF model (e.g. CFNet)
    X:     Rating marix (in user-by-item format) that combines the training split (R) and 
           the test split (T)
    L:     A Label vector that combines true labels from the training split (L_train) and 
           estimated labels (lh) from the test split; note that lh is not the same as L_test. 
           `lh` is used to faciliate the derivation of latent factors whereas `L_test` is 
           what the model aims to predict. 
    Pc:    Color matrix 
    n_train: number of training instances; this parameter is used to split X into R and T

    Returns 
    -------
    A dicitonary with keys representing reestimated quantities: 

    Rh: 
    Th:  
    ratings: 
    p_threshold: 

    """
    import data_pipeline as dp
    import utils_cf as uc
    import utils_classifier as uclf
    from scipy.spatial import distance

    # Original rating matrices 
    # - (R, T, X)
    if isinstance(X, (tuple, list)): 
        R, T = X
        n_train = R.shape[1] # infer training set size
        X = np.hstack([R, T]) 
    else: 
        assert n_train is not None
        R, T = X[:,:n_train], X[:,n_train:]

    # Labels: (L_train, lh, L)
    if isinstance(L, (tuple, list)): 
        L_train, lh = L
        n_train = len(L_train) # infer training set size
        L = np.hstack([L_train, lh]) 
    else: 
        L_train, lh = L[:n_train], L[n_train:] # split L into L_train and lh; note that lh is NOT L_test 
    
    # Color matrix (Pc_train, Pc_test, Pc)
    # if isinstance(Pc, (tuple, list)): 
    #     Pc_train, Pc_test = Pc
    #     Pc = np.hstack([Pc_train, Pc_test])

    def analyze_reconstruction_core(L_test, unreliable_only=True, **kargs): 

        nonlocal p_threshold # X, L, n_train
        reestimated = {} # keeps track of reestimated quantities (e.g. Rh, Th)
        scores = [] # keeps track of performance score and parameter settings
        settings = {} # keeps track of the algorithmic settings (used with `scores` to rank settings)
        # [todo] Use Hyper or other more organized methods to keep track of parameter settings

        verbose = kargs.get('verbose', 1)

        # 1. New rating matrices (level-1) with probabilties reestimated
        ####################################
        if not unreliable_only: 
            # A. Reestimate entire matrix
            Rh, Th = reestimate(model, X, n_train=n_train)
        else: 
            # B. Reestimate only unreliable entries (better)
            Rh, Th = reestimate_unreliable_only(model, X, Pc=Pc, n_train=n_train)
        ####################################
        reestimated['ratings'] = (Rh, Th)

        # Probability thresholds associated with the original training data (R)
        if len(p_threshold) == 0: p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy=policy_threshold)
        # Re-estimate the p_threshold as well 
        reestimated['p_threshold2'] = p_threshold_new = uc.estimateProbThresholds(Rh, L=L_train, pos_label=1, policy=policy_threshold)

        if verbose: 
            print(f"[info] From R to Rh, delta(Frobenius norm)= {LA.norm(Rh-R, ord='fro')}")
            print(f"[info] From T to Th, delta(Frobenius norm)= {LA.norm(Th-T, ord='fro')}")

        # 2. Prediction: By majority vote
        ####################################
        msg = ""
        reestimated['lh_maxvote'] = lh = uc.estimateLabels(T, L=[], p_th=p_threshold, pos_label=1) # "majority vote given proba thresholds" is the default strategy
        reestimated['lh2_maxvote_pth_unadjusted'] = lh_new_orig_pth = \
                uc.estimateLabels(Th, L=[], p_th=p_threshold, pos_label=1) # Use the re-estimated T and original p_th to predict labels
        reestimated['lh2_maxvote_pth_adjusted'] = lh_new = \
                uc.estimateLabels(Th, L=[], p_th=p_threshold_new, pos_label=1) # Use the re-estimated T to predict labels
        msg += f"[info] How different are lh and lh_new? {distance.hamming(lh, lh_new)}\n"

        # Evaluate using a given performance score (since CF ensemble is primarily targeting imbalance class distributions, 
        # by defeaut, we will use F1 score)
        reestimated['score_lh_maxvote'] = reestimated['score_baseline'] = perf_score = f1_score(L_test, lh)
        scores.append((perf_score , {'lh': lh, 'p_threshold': p_threshold, 'name': 'lh_maxvote'}))
        msg += f'[result] Majority vote: F1 score with the original T:  {perf_score}\n'

        reestimated['score_lh2_maxvote_pth_unadjusted'] = perf_score = f1_score(L_test, lh_new_orig_pth)
        scores.append((perf_score , {'lh': lh_new_orig_pth, 'p_threshold': p_threshold, 'name': 'lh2_maxvote_pth_unadjusted'}))
        msg += f'[result] Majority vote: F1 score with re-estimated Th using original p_threshold: {perf_score}\n'

        reestimated['score_lh2_maxvote_pth_adjusted'] = perf_score = f1_score(L_test, lh_new) 
        scores.append((perf_score , {'lh': lh_new, 'p_threshold': p_threshold_new, 'name': 'lh2_maxvote_pth_adjusted'}))
        msg += f'[result] Majority vote: F1 score with re-estimated Th: {perf_score}\n'

        if verbose: print(msg)

        # [todo] Use a hyperparameter tracker to organize different parameter settings

        # 3. Prediction: By stacking
        ####################################
        msg = ""
        stacker = kargs.get('stacker', None)
        grid = kargs.get('grid', {})
        if stacker is None: 
            stacker = LogisticRegression() 
            grid = uclf.hyperparameter_template('logistic')
        else: 
            assert callable(stacker), f"Invalid meta-classifier: {stacker}"

        # X_train, y_train = R.T, L_train
        lh = uclf.tune_model(stacker, grid, scoring='f1', verbose=0)(R.T, L_train).predict(T.T)
        reestimated['score_lh_stacker'] = perf_score = f1_score(L_test, lh) 
        msg += f'[result] Stacking: F1 score with the original T:  {perf_score}\n'

        # X_train, y_train = Rh.T, L_train
        lh_new = uclf.tune_model(stacker, grid, scoring='f1', verbose=0)(Rh.T, L_train).predict(Th.T)
        reestimated['score_lh2_stacker_pth_adjusted'] = perf_score = f1_score(L_test, lh_new)
        msg += f'[result] Stacking: F1 score with re-estimated Th: {perf_score}\n'   
        
        if verbose: print(msg)  

        # 4. Rank parameter settings
        ####################################
        msg = ""
        # Choose the best settings (excluding stacking on re-estiamted matrices)
        # Note: somehow `key` is necessary (shouldn't be by default) otherwise error: "The truth value of an array with more than one element"
        scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True) 
        if verbose > 1: msg += f"[result] Methods ranked:\n{[(s, d['name']) for s, d in scores_sorted]}\n\n"
        reestimated['best_params'] = scores_sorted[0][1] # select the best setting according to the performance measure
        reestimated['best_params_score'] = scores_sorted[0][0]

        if verbose: 
            mode = 'unreliable only' if unreliable_only else 'complete' # 'complete' reestimation or reestimating the entire rating matrix
            msg += f"[result] Best settings ({mode}): {reestimated['best_params']['name']}, score: {scores_sorted[0][0]}\n"
            print(msg)
        if verbose > 1: 
            print("[help] Reestiamted quantities are available through the following keys:")
            for k, v in reestimated.items(): 
                print(f'  - {k}')
 
        # Returns
        # -------
        # A dicitonary with reestimated quantities: `ratings`, `p_threshold_new`, ...
        return reestimated 
    return analyze_reconstruction_core

# Training pipeline utilities
###########################################################################

def prepare_training_data(X, C, Pc, p_threshold, target_type='generic'):
    """
    Convert training data in matrix format (X, C, Pc) to user-item-pair format 
    in order to feed them into CFNet. 
    """

    import utils_cf as uc
    import data_pipeline as dp

    sample_weights = np.array([])
    if target_type.startswith(('label', )):
        Lh = uc.estimateLabelMatrix(X, p_th=p_threshold)
        Xc, yc, weights, colors = dp.matrix_to_augmented_training_data(Lh, C, Pc)
        # `weights`, `colors` are not used here

        sample_weights = dp.unravel(C, normalize=False) # Cn is a masked and balanced version of C0
        
    elif target_type.startswith(('prob', 'rating')):
        Xc, yc, weights, colors = dp.matrix_to_augmented_training_data(X, C, Pc) 
        # `weights`, `colors` are not used here
        
        sample_weights = dp.unravel(C, normalize=False) # Cn is a masked and balanced version of C0
    else: # a more general case where `y_true` carries more than just the labels
        Xc, yc = dp.matrix_to_augmented_training_data2(X, C, Pc, p_threshold=p_threshold)
        assert yc.shape[1] >= 3, f"Got n={yc.shape[1]} columns but `yc` should carry at least 3 attributes: (y_true, weight, color) and may include additional attributes"
    
    return (Xc, yc, sample_weights)

def training_loop(input_model, input_data, **kargs): 
    # import matplotlib.pylab as plt

    # Optional Parameters 
    # -------------------
    verbose = kargs.get('verbose', 1)
    test_size = kargs.get('test_size', 0.1)
    batch_size = kargs.get('batch_size', 64) 
    epochs = kargs.get('epochs', 120)
    use_sample_weights = kargs.get('use_sample_weights', True)
   
    model, loss_fn = input_model
    
    X, y, sample_weights, *rest = input_data
    assert len(sample_weights) == 0 or (len(sample_weights) == X.shape[0])

    split_pt = int((1-test_size) * X.shape[0])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    if use_sample_weights and (len(sample_weights) > 0): 
        X_train, X_val, y_train, y_val, W_train, W_val = (
            X[:split_pt],
            X[split_pt:],
            y[:split_pt],
            y[split_pt:],
            sample_weights[:split_pt], 
            sample_weights[split_pt:]
        )

        history = model.fit(
            x=X_train,
            y=y_train,
            sample_weight=W_train, 
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, y_val, W_val), # test how the model predict unseen ratings
            callbacks=[tensorboard_callback]
        )
    else: # no sample weights
        X_train, X_val, y_train, y_val = (
            X[:split_pt],
            X[split_pt:],
            y[:split_pt],
            y[split_pt:])
        
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, y_val), # test how the model predict unseen ratings
            callbacks=[tensorboard_callback]
        )

    if verbose: 
        # %matplotlib inline
        f, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(20,8))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # %load_ext tensorboard
        # %tensorboard --logdir logs

        # analyzer = analyze_reconstruction(model, X, L, Pc, n_train, p_threshold=p_threshold, policy_threshold=policy_threshold)
        # highlight(f"Reestimate the entire rating matrix (X) with learned latent factors/embeddings")
        # analyzer(L_test, unreliable_only=False)
        # highlight(f"Reestimate ONLY the unreliable entries in X with learned latent factors/embeddings")
        # analyzer(L_test, unreliable_only=True)

    return model 

def get_loss_function_name(loss_fn):
    # Utility function for `training_pipeline()`
    try: 
        return loss_fn.__name__ # works only if loss_fn is a plain funciton, not wrapped in a class
    except: 
        pass 
    return loss_fn.__class__.__name__ # E.g. tf.keras.losses.MeanSquaredError() ~> `MeanSquaredError`
def determine_target_type(loss_fn):
    # Uitlity function for `training_pipeline()`
    loss_fn_name = get_loss_function_name(loss_fn)
    target_type = 'generic'
    if loss_fn_name.lower().find('entropy') > 0: 
        return 'label'
    if loss_fn_name.lower().startswith('meansquared') == 0: 
        return 'rating'
    return target_type
def training_pipeline(input_model, input_data, **kargs):
    """
    Load and transform input training data (from rating matrix format to user-item-pair format for CFNet), 
    followed by iteratively training the model (i.e. training loop). 


    Paramters 
    ---------
    input_model: A 2-tuple (model, loss function)
    input_data: A dictionary or a tuple with training data in the form of rating matrices. 

    Optional Parameters
    -------------------
    verbose 

    is_cascade 

    alpha: Scaling factor for confidence matrix 

    conf_measure: A confidence measure
    conf_type: The type of confidence matrix (e.g. `Cn`, `Cw`, `C0`)
    
    policy_threshold: The policy/strategy for determining optimial probability threshold (e.g. 'fmax')
    
    lh: Pre-computed estiamted labels for the test split (T)
    fold_number: 

    target_type: The type of training data; this determines how training data is structureed

                How do we structure the training data? This depends on two factors: 

                1. The quantitiy that the model attempts to approximate E.g. 'rating', 'label'

                   - Set `target_type` to 'rating' if the goal is to approximate the rating (e.g. the 
                    probability score in entries of X, that is, X[i][j]). A suggested loss function 
                    in this case is the mean square loss (or MSE)
       
                   - Set `target_type` to 'label' if the goal is to approximate the labeling associated 
                     with the rating matrix (X); also see utils_cf.estimateLabelMatrix()
                     This is useful for entropy-based function such as binary entropy loss or BCE loss

                2. The type of loss function E.g. the label (i.e. `y_true` by sklearn's convention). 

                   Simple loss functions like MSE and BCE have the shape (n_instances, ), where 
                   we've used `n_instances` refer to the training sample size. 

                   More complex loss functions can take into account not only the true labels (e.g. ratings) but also 
                   confidence matrix, colors and probability thresholds, among others, and as a result will 
                   on a more complex shape (n_instances, >=2) where the "label" of the training data becomes a matrix 

                   - Set `target_type` to 'generic' if this is the case. 

    test_size: 
    epochs: 
    batch_size: 

    """
    import data_pipeline as dp 
    import utils_cf as uc
    from utils_sys import highlight
    from analyzer import is_sparse
    import matplotlib.pylab as plt

    # General Paramters 
    # -----------------
    verbose = kargs.get("verbose", 1)
    ########################################
    is_cascade = kargs.get('is_cascade', False) # if True, merge training split (R) and test split (T) to get `X`... 
    # ... and also merge training set labels (L_train) and estimated test set labels (lh) to get `L`, and train the entire model ... 
    # ... based on (X, L)
    # In other words, when `is_cascade` is on, then we train a model that predicts ratings in both training split and test split 
    # but when `is_cascade` is off, we only train a model that predicts ratings in the training split (R); the test split's ratings will 
    # then be based on the patterns learned from the training split such as those learnd from kNN-based models
    ########################################

    # Algorithm Parameters 
    #---------------------
    alpha = kargs.get('alpha', 10.0)  # A scaling factor for the "implicit feedback," which in this case is the confidence scores 
    conf_measure = kargs.get('conf_measure', 'brier')  # measure of confidence of base predictors' probabilistic predictions
    policy_threshold = kargs.get('policy_threshold', 'fmax') # method for optimizing the probability threshold 
    estimated_labels = kargs.get('lh', None) # pre-computed estimated labels for `T`
    fold_number = kargs.get('fold_number', 0) # dataset identifier used in cross validation or multiple runs of subsampling


    # Load pre-trained level-1 data (associated with a given fold number)
    ####################################################################################
    # a. Basic quantifies
    L_test = None # Test data is optional; evaluation is conducted outside this training loop
    # input data is a 5-tuple 
    if isinstance(input_data, dict): 
        R, T, U, L_train = input_data['R'], input_data['T'], input_data['U'], input_data['L_train']
        L_test = input_data.get('L_test', None) # Test data is optional
    elif isinstance(input_data, (tuple, list)): 
        assert len(input_data) >= 4
        R, T, U, L_train, *rest = input_data
        if len(rest) > 0: 
            L_test = rest[0]
 
    n_train = R.shape[1]
    assert len(U) == R.shape[0]

    # b. Derived quantities
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy=policy_threshold)
    
    # Compute various types of confidence matrices (using only training split `R` and `L_train`)
    ####################################################################################
    
    msg = ''
    if is_cascade: 
        #####################
        lh = estimated_labels
        if lh is None: 
            msg += f"[labeling] Use 'majority vote' by default for the estimated labels for T\n"
            lh = uc.estimateLabels(T, p_th=p_threshold) # NOTE: We of course cannot use `L_test`
        L = np.hstack((L_train, lh)) # Remember to use "estimated labels" (lh) for the test set; not the true label (L_test)
        msg += f"[merge] Merging 'L_train' and 'lh': len(L_train): {len(L_train)} || len(lh): {len(lh)} => len(L): {len(L)}\n"
        #####################

        X = np.hstack((R, T))
        msg += f"[merge] Merging 'R' and 'T': shape(R):{R.shape} || shape(T): {T.shape} => shape(X): {X.shape}\n"
        assert len(L) == X.shape[1]
    else: 
        L = L_train
        X = R
    if verbose: print(msg)

    Pc, C0, Cw, Cn, *rest = \
        uc.evalConfidenceMatrices(X, L, alpha=alpha, 
                                        p_threshold=p_threshold, 
                                        conf_measure=conf_measure, policy_threshold=policy_threshold, 
                                        
                                        # Optional debug/test parameters 
                                        U=U, n_train=n_train, 
                                        fold_number=fold_number, 
                                        is_cascade=True,
                                        verbose=0)
    assert C0.shape == X.shape
    y_colors = pmodel.verify_colors(Pc)  # [log] status: ok

    # Prepare training data
    ####################################################################################
    conf_type = kargs.get("conf_type", 'Cn') # confidence matrix type 

    # Choose confidence matrix type
    # Note: This is only relevant when we use the "color-aware" loss function like C-square loss (see `demo_cfnet_with_csqr_loss()`)
    if conf_type in ('Cn', 'sparse'): # "sparse" because FP- and FN- weights are zeroed out
        # Use `Cn` (masked confidence matrix) instead of `Cw` (a dense matrix that includes weights for FPs, FNs) 
        C = Cn
    elif conf_type in ('Cw', 'dense'): 
        C = Cw
    else: # "Raw" confidence matrix
        C = C0 
    # ... Now we have (X, C, Pc, p_threshold)

    cf_model, loss_fn = input_model

    # How do we structure the training data? This depends on two factors: 
    # 1. The quantitiy that the model attempts to approximate E.g. `rating`, `label`
    # 2. The type of loss function E.g. the label (i.e. `y_true` by sklearn's convention) 
    #    for the MSE and BCE losses has shape (n_instances, ), where n_instances refer to the training sample size
    #    More complex loss functions can take into account not only the true labels (e.g. ratings) but also 
    #    confidence matrix, colors and probability thresholds, among others, and as a result will 
    #    on a more complex shape (n_instances, >=2) where the "label" of the training data becomes a matrix 
    target_type = kargs.get('target_type', determine_target_type(loss_fn)) # 'label', 'rating', 'generic'
    if verbose: print(f"[info] Confidence matrix type: {conf_type}, target data type: {target_type}")

    Xc, yc, sample_weights = prepare_training_data(X, C, Pc, p_threshold, target_type=target_type)

    # SGD training paramters
    #-----------------------
    epochs = kargs.get('epochs', 100)
    batch_size = kargs.get('batch_size', 64)
    test_size = kargs.get('test_size', 0.1)
    use_sample_weights = kargs.get('use_sample_weights', True) # if True, then use sample weights whenever they are available
    #-----------------------

    # Training Parameters
    # -------------------
    # test_size 
    # batch_size 
    # epochs
    # lr: learning rate => this goes into model definition
    
    # model, loss_fn = input_model
    model = training_loop(input_model=input_model, input_data=(Xc, yc, sample_weights), 
                               test_size=test_size, batch_size=batch_size, epochs=epochs, 
                               use_sample_weights=use_sample_weights) # 

    return model

# Demo 
#######################################################

def demo_cfnet_with_csqr_loss(loss_fn=None, ctype='Cn', n_factors=50, alpha=10.0, 
                              conf_measure='brier', policy_threshold='fmax', data_dir='./data'): 
    import data_pipeline as dp 
    import utils_cf as uc
    from utils_sys import highlight
    from analyzer import is_sparse

    import matplotlib.pylab as plt
    # from matplotlib.pyplot import figure
    # import seaborn as sns

    # Algorithmic parameters 
    #-----------------------
    # n_factors = 50
    # alpha = 10.0              # A scaling factor for the "implicit feedback," which in this case is the confidence scores 
    # conf_measure = 'brier'    # measure of confidence of base predictors' probabilistic predictions
    # policy_threshold = 'fmax' # method for optimizing the probability threshold 
    fold_number = 0
    test_size = 0.1
    #-----------------------

    # Load pre-trained level-1 data (associated with a given fold number)
    ####################################################################################
    # a. Basic quantifies
    R, T, U, L_train, L_test = dp.load_pretrained_level1_data(fold_number=fold_number, verbose=1, data_dir=data_dir) 
    n_train = R.shape[1]

    # b. Derived quantities
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy=policy_threshold)
    lh = uc.estimateLabels(T, p_th=p_threshold) # We cannot use L_test (cheating), but we have to guesstimate [1]
    L = np.hstack((L_train, lh)) 
    X = np.hstack((R, T))
    # Note: Remember to use "estimated labels" (lh) for the test set; not the true label (L_test)

    assert len(U) == X.shape[0]
    print(f"> shape(R):{R.shape} || shape(T): {T.shape} => shape(X): {X.shape}")

    # Compute various types of confidence matrices
    ####################################################################################

    Pc, C0, Cw, Cn, *rest = \
        uc.evalConfidenceMatrices(X, L, alpha=alpha, 
                                     p_threshold=p_threshold, 
                                        conf_measure=conf_measure, policy_threshold=policy_threshold, 
                                        
                                        # Optional debug/test parameters 
                                        U=U, n_train=n_train, fold_number=fold_number, 
                                        is_cascade=True,
                                        verbose=0)

    # Prepare training data
    ####################################################################################

    # Choose confidence matrix type
    if ctype == 'Cn': # "sparse" because FP- and FN- weights are zeroed out
        # Use `Cn` (masked confidence matrix) instead of `Cw` (a dense matrix that includes weights for FPs, FNs) 
        C = Cn
    elif ctype == 'Cw': # "dense"
        C = Cw
    else: 
        C = C0

    # Xc, yc, weights, colors = dp.matrix_to_augmented_training_data(X, C, Pc) # NOTE: Don't overwrite X (`Xc` is not the same as `X`, which is a rating matrix)
    # yc = np.column_stack([yc, weights, colors])

    Xc, yc = dp.matrix_to_augmented_training_data2(X, C, Pc, p_threshold=p_threshold)
    assert yc.shape[1] >= 3, f"Got n={yc.shape[1]} columns but `yc` should carry at least 3 attributes: (y_true, weight, color) and may include additional attributes"

    #----------------
    # test_size = 0.1
    #----------------
    split_pt = int((1-test_size) * Xc.shape[0])
    X_train, X_val, y_train, y_val = (
        Xc[:split_pt],
        Xc[split_pt:],
        yc[:split_pt],
        yc[split_pt:])

    if loss_fn is None:
        loss_fn = confidence_weighted_loss # Options: confidence_weighted_loss, c_squared_loss
    
    n_users, n_items = X.shape
    model = get_cfnet_uncompiled(n_users, n_items, n_factors)
    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=0.001))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    history = model.fit(
        x=X_train,
        y=y_train,
        # sample_weight=W_train, # not using sample weight in this case
        batch_size=64,
        epochs=100,
        verbose=1,
        validation_data=(X_val, y_val), # test how the model predict unseen ratings
        callbacks=[tensorboard_callback]
    )

    # %matplotlib inline
    f, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(20,8))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    # %load_ext tensorboard
    # %tensorboard --logdir logs

    analyzer = analyze_reconstruction(model, X, L, Pc, n_train, p_threshold=p_threshold, policy_threshold=policy_threshold)
    highlight(f"(C-Sqr with {ctype}) Reestimate the entire rating matrix (X) with learned latent factors/embeddings")
    analyzer(L_test, unreliable_only=False)
    highlight(f"(C-Sqr with {ctype}) Reestimate ONLY the unreliable entries in X with learned latent factors/embeddings")
    analyzer(L_test, unreliable_only=True)

    return model


def demo_cf_stacking(input_data=None, input_dir='./data', n_iter=5, base_learners=None, verbose=1): 
    from sklearn.metrics import f1_score
    import data_pipeline as dp
    from tqdm import tqdm

    # Create base predictors
    if base_learners is None: 
        base_learners = [
                         ('RF', RandomForestClassifier(n_estimators= 200, 
                                                           oob_score = True, 
                                                           class_weight = "balanced", 
                                                           random_state = 20, 
                                                           ccp_alpha = 0.1)), 
                         ('KNNC', KNeighborsClassifier(n_neighbors = len(np.unique(y))
                                                             , weights = 'distance')),
                         ('SVC', SVC(kernel = 'linear', probability=True,
                                           class_weight = 'balanced'
                                          , break_ties = True)), 

                         ('GNB', GaussianNB()), 
                         ('QDA',  QuadraticDiscriminantAnalysis()), 
                         ('MLPClassifier', MLPClassifier(alpha=1, max_iter=1000)), 
                         ('DT', DecisionTreeClassifier(max_depth=5)),
                         ('GPC', GaussianProcessClassifier(1.0 * RBF(1.0))),
                        ]
    
    # Generate training data if not provided
    if input_data is None: 
        X, y = dp.generate_imbalanced_data(class_ratio=0.95, verbose=1)
    else: 
        assert len(input_data) >= 2
        X, y, *rest = input_data

    # Run CF Stacking with `n_iter` iterations
    ################################################################
    cf_stackers = []
    for i in tqdm(range(n_iter)): 

        # Initialize CF Stacker
        if verbose > 1: print(f"[loop] Instantiate CFStacker #[{i+1}] ...")
        clf = ustk.CFStacker(estimators=base_learners, 
                                final_estimator=LogisticRegression(),  # [todo] Replace this with a completed CF method
                                work_dir = input_dir,
                                fold_number = i, # use this to index traing and test data 
                                verbose=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        clf.fit(X_train, y_train)

        X_meta_test = clf.transform(X_test)
        if verbose > 1: print(f"[info] shape(X_meta_test): {X_meta_test.shape}")

        y_pred = clf.predict(X_test)
        perf_score = f1_score(y_test, y_pred)  # clf.score(X_test, y_test)
        if verbose: print('[result]', perf_score)

        # Add test label for the convenience of future evaluation after applying a CF ensemble method
        clf.cf_write(dtype='test', y=y_test)

        # keep track of all the stackers (trained on differet parts of the same data as in CV or resampling)
        cf_stackers.append(clf)

    ###################################################################################
    meta_set = clf.cf_fetch(fold_number=0) # Take the first dataset as an example
    X_train, y_train = meta_set['train']['X'], meta_set['train']['y'] 
    n_train = X_train.shape[1]
    X_test = meta_set['test']['X']
    y_test = None
    try: 
        y_test = meta_set['test']['y']
    except: 
        print("[warning] test label is not available yet. Run the previous code block first.")

    # Names of the base classifiers/predictors/estimators
    U = meta_set['train']['U']
    print(f"[info] list of base classifiers:\n{U}\n")

    # Structure the rating/probability matrix
    highlight("R: Rating/probability matrix for the TRAIN set")
    R = X_train.T # transpose because we need users by items (or classifiers x data) for CF
    n_train = R.shape[1]
    L_train = y_train

    T = X_test.T
    L_test = y_test

    # The following are all derived quantities
    # p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy='fmax')

    # lh = uc.estimateLabels(T, p_th=p_threshold) # We cannot use L_test (cheating), but we have to guesstimate
    # L = np.hstack((L_train, lh)) 
    # NOTE: Remember to use "estimated labels" for the test set; not the true label `L_test` that we are trying to predict

    # X = np.hstack((R, T))
    # assert len(U) == X.shape[0]
    # print(f"> shape(R):{R.shape} || shape(T): {T.shape} => shape(X): {X.shape}")

    return (R, T, U, L_train, L_test)

def test(): 
    demo_cfnet_with_csqr_loss(ctype='Cn')

    return

if __name__ == "__main__": 
    test()