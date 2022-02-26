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
def confidence_weighted_loss(y_true, y_pred, weights=None, colors=None): # this has to be used with .add_loss() with greater flexibility
    # from tensorflow.keras import backend as K    

    if weights is None: 
        weights = K.ones_like(y_pred)
    
    if colors is None: 
        # confidence-weighted sum of squares 
        wmse = weights * K.square(y_pred - y_true, axis=-1) # difference between predicted and "true" probability
    else: 
        # condition
        mask_tp = K.equal(colors, 2)
        mask_tn = K.equal(colors, 1)
        mask_fp = K.equal(colors, -2)
        mask_fn = K.equal(colors, -1)

        # if TP, want y_pred >= y_true, i.e. the larger (the closer to 1), the better
        loss_tp = weights * K.square(K.maximum(y_true-y_pred, 0)) # if y_pred > y_true => y_true-y_pred < 0 => no loss, if ow, then the smaller, the higher the penalty (quadratically)
        
        # if TN, want y_pred < y_true, i.e. the smaller (the closer to 0), the better
        loss_tn = weights * K.square(K.maximum(y_pred-y_true, 0)) # if y_true>y_pred => y_pred-y_true < 0 => no loss

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


    return

def test(): 

    return

if __name__ == "__main__": 
    test()