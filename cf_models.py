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
import scipy.sparse as sparse
from utils_sys import highlight
#################################################################

# Misc
import numpy as np
from numpy import linalg as LA
import pprint
import tempfile
from typing import Dict, Text

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

def get_cfnet_approximating_labels(n_users, n_items, n_factors): 
    loss_fn = bce = tf.keras.losses.BinaryCrossentropy()
    model = CFNet(n_users, n_items, n_factors)
    model.compile(
        loss=loss_fn, optimizer=keras.optimizers.Adam(lr=0.001)
    )
    return model

def get_cfnet_approximating_scores(n_users, n_items, n_factors): 
    loss_fn = mse = tf.keras.losses.MeanSquaredError()
    model = CFNet(n_users, n_items, n_factors)
    model.compile(
        loss=loss_fn, optimizer=keras.optimizers.Adam(lr=0.001)
    )
    return model

# Loss functions
##########################################################################
def confidence_weighted_loss(y_true, y_pred): # this has to be used with .add_loss() with greater flexibility
    # from tensorflow.keras import backend as K    
    
    y_label = y_true[:, 0]
    weights = y_true[:, 1]
    colors =  y_true[:, 2]
  
    # condition
    mask_tp = tf.dtypes.cast( K.equal(colors, 2), dtype = tf.float32 )
    mask_tn = tf.dtypes.cast( K.equal(colors, 1), dtype = tf.float32 )
    mask_fp = tf.dtypes.cast( K.equal(colors,-2), dtype = tf.float32 )
    mask_fn = tf.dtypes.cast( K.equal(colors,-1), dtype = tf.float32 )
    # Note: We also need to convert these masks to integer type so that we can use them to make the weighted sum

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

    wmse = mask_tp * loss_tp + mask_tn * loss_tn + mask_fp * loss_fp + mask_fn * loss_fn
    return wmse

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # K: tensorflow backend
    
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


##########################################################################

def analyze_reconstruction(model, X, L, Pc, n_train, p_threshold=[], policy_threshold='fmax'): 
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

    """
    import data_pipeline as dp
    import utils_cf as uc
    from scipy.spatial import distance

    def analyze_reconstruction_core(L_test, reestimate_unreliable_only=True): 

        nonlocal p_threshold 

        # Original data
        R, T = X[:,:n_train], X[:,n_train:]
        L_train, lh = L[:n_train], L[n_train:] # split L into L_train and lh; note that lh is NOT L_test

        # New (level-1) data with probabilties reestimated
        ####################################
        if not reestimate_unreliable_only: 
            # A. Reestimate entire matrix
            Rh, Th = dp.reestimate(model, X, n_train=n_train)
        else: 
            # B. Reestimate only unreliable entries (better)
            Rh, Th = dp.reestimate_unreliable_only(model, X, n_train=n_train, Pc=Pc)
        ####################################

        # Probability thresholds associated with the original training data (R)
        if len(p_threshold) == 0: p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy=policy_threshold)

        # Re-estimate the p_threshold as well 
        p_threshold_new = uc.estimateProbThresholds(Rh, L=L_train, pos_label=1, policy=policy_threshold)

        print(f"[info] From R to Rh, delta(Frobenius norm)= {LA.norm(Rh-R, ord='fro')}")
        print(f"[info] From T to Th, delta(Frobenius norm)= {LA.norm(Th-T, ord='fro')}")

        lh = uc.estimateLabels(T, L=[], p_th=p_threshold, pos_label=1) # "majority vote given proba thresholds" is the default strategy
        lh_new = uc.estimateLabels(Th, L=[], p_th=p_threshold_new, pos_label=1) # Use the re-estimated T to predict labels
        print(f"[info] How different are lh and lh_new? {distance.hamming(lh, lh_new)}")
  
        perf_score = f1_score(L_test, lh)
        print(f'[result] F1 score with the original T:  {perf_score}')

        perf_score = f1_score(L_test, lh_new) 
        print(f'[result] F1 score with re-estimated Th: {perf_score}')

        # model = LogisticRegression()
        # # solvers = ['newton-cg', 'lbfgs', 'liblinear']
        # penalty = ['l1', 'l2']
        # c_values = np.logspace(-2, 2, 5)

        # grid = dict(penalty=penalty,C=c_values)
        # tuner = get_tuned_classifier(model, grid, n_splits=5, n_repeats=3)
        # lh2 = tuner(R.T, L_train).predict(L_test)

        return 
    return analyze_reconstruction_core


def demo_cfnet_with_csqr_loss(ctype='Cn', n_factors=50, alpha=10.0, conf_measure='brier', policy_threshold='fmax', data_dir='./data'): 
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

    CX = uc.evalConfidenceMatrix(X, L=L, U=U, 
                                 p_threshold=p_threshold, # not needed if L is given (suggested use: estimate L outside of this call)
                                 policy_threshold=policy_threshold,
                                 conf_measure=conf_measure, 
                                 fill=0, is_cascade=True, n_train=n_train, 
                                 fold=fold_number, # for debugging only
                                 verbose=0) 
    C0, Pc, p_threshold, *CX_res = CX

    # Cw: A re-weighted (dense) confidence matrix in which confidence scores are adjusted to take into account 
    #     the disparity in sample sizes (e.g. the size of TPs is usually much smaller than that of TNs in class-imbalanced data)
    Cw = uc.balance_and_scale(C0, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, 
                        alpha=alpha, conf_measure=conf_measure, n_train=n_train, verbose=0)

    # Cn: A masked confidence matrix where the confidence scores associated with FPs and FNs are set to 0
    Cn = uc.mask_neutral_and_negative(C0, Pc, is_unweighted=False, weight_negative=0.0, sparsify=True)
    Cn = uc.balance_and_scale(Cn, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, 
                        alpha=alpha, conf_measure=conf_measure, n_train=n_train, verbose=0)
    
    # Test: Wherever Pc is negative, the corresponding entries in Cn must be 0 (By constrast, C is a full/dense confidence matrix)
    assert np.all(Cn[Pc < 0]==0)
    assert np.all(Cn[Pc > 0]>0)

    # Color matrix should have 4 distinct values
    uniq_colors = np.unique(Pc.A if is_sparse(Pc) else Pc)
    assert len(uniq_colors) == 4, f"n_colors: {uniq_colors}"

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

    Xc, yc, weights, colors = dp.matrix_to_augmented_training_data(X, C, Pc) # NOTE: Don't overwrite X (`Xc` is not the same as `X`, which is a rating matrix)
    yc = np.column_stack([yc, weights, colors])

    #----------------
    # test_size = 0.1
    #----------------
    split_pt = int((1-test_size) * Xc.shape[0])
    X_train, X_val, y_train, y_val = (
        Xc[:split_pt],
        Xc[split_pt:],
        yc[:split_pt],
        yc[split_pt:])

    loss_fn = confidence_weighted_loss
    
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
    analyzer(L_test, reestimate_unreliable_only=False)
    highlight(f"(C-Sqr with {ctype}) Reestimate ONLY the unreliable entries in X with learned latent factors/embeddings")
    analyzer(L_test, reestimate_unreliable_only=True)

    return


def demo_stacking(n_iter=5, base_learners=None): 
    from sklearn.metrics import f1_score

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

    cf_stackers = []
    for i in range(n_iter): 
        # Initialize CF Stacker
        print(f"[demo] Instantiate CFStacker #[{i+1}] ...")
        clf = ustk.CFStacker(estimators=base_learners, 
                                final_estimator=LogisticRegression(), 
                                work_dir = input_dir,
                                fold_number = i, # use this to index traing and test data 
                                verbose=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        clf.fit(X_train, y_train)

        X_meta_test = clf.transform(X_test)
        print(f"[info] shape(X_meta_test): {X_meta_test.shape}")

        y_pred = clf.predict(X_test)
        perf_score = f1_score(y_test, y_pred)  # clf.score(X_test, y_test)
        print('[result]', perf_score)

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
    highlight("R: Rataing/probability matrix for the TRAIN set")
    R = X_train.T # transpose because we need users by items (or classifiers x data) for CF
    n_train = R.shape[1]
    L_train = y_train

    T = X_test.T
    L_test = y_test

    # Remember to use "estimated labels" for the test set; not the true label `L_test` that we are trying to predict
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy='fmax')
    lh = uc.estimateLabels(T, p_th=p_threshold) # We cannot use L_test (cheating), but we have to guesstimate
    L = np.hstack((L_train, lh)) 
    X = np.hstack((R, T))

    assert len(U) == X.shape[0]
    print(f"> shape(R):{R.shape} || shape(T): {T.shape} => shape(X): {X.shape}")

    return (X, L, n_train)

def test(): 
    demo_cfnet_with_csqr_loss(ctype='Cn')

    return

if __name__ == "__main__": 
    test()