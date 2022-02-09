#!/usr/bin/env python
# encoding: utf-8

### configurations
import os, math, sys, gc
from sys import argv
import utils_sys

# cluster_module_path = os.path.join(os.getcwd(), 'cluster')
# sys.path.insert(0,cluster_module_path) 

Domain = 'pf2'
ProjectPath = utils_sys.getProjectPath(domain=Domain, verify_=False)  # default
try: 
    ProjectPath = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    Domain = os.path.basename(ProjectPath)
except: 
    pass 
import cf_spec
cf_spec.config(project_path=ProjectPath, domain=Domain)  # to be shared by all relavant modules 
from cf_spec import MFEnsemble
# assert os.path.exists(ProjectPath)
# condition: definition of Domain and ProjectPath needs to precede evaluate, utils_cf

import numpy as np
import math
from pandas import DataFrame, Series
import timeit

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

from nnls import NNLS

### Plotting 

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('ggplot')  # values: {'seaborn', 'ggplot', }
from utils_plot import saveFig, plot_path

### supporting modules
from utils_sys import div

### datasink 
import common, utilities
Properties = common.load_properties(ProjectPath, config_file='config.txt')  # parse config.txt (instead of weka.properties)
cf_spec.FoldCount = FoldCount = int(Properties['foldCount'])
cf_spec.BagCount = BagCount = int(Properties['bagCount']) if 'bagCount' in Properties else int(Properties['bags']) 

### CF dependent modules
import utils_cf
import evaluate
from evaluate import PerformanceMetrics, Metrics # as perfm
from evaluate import visualizeCoeffs, plot_roc

import logging

### logging
# PerformanceMetrics.init_log()

# evaluate.Metrics
# MetricsTracked = ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', ]


"""
  
  Use 
  ---
  Run the first 2 steps first

  1. python step1a_generate.py /Users/chiup04/Documents/work/data/diabetes_cf
  2. python combine.py /Users/chiup04/Documents/work/data/diabetes_cf 
  3. 

  Reference
  ----------
  1. http://sdsawtelle.github.io/blog/output/week9-recommender-andrew-ng-machine-learning-with-python.html


"""

class MC(BaseEstimator):
    """ An estimator for latent factor collaborative filtering models in Recommender Systems.
    """
    def __init__(self,  n_u, n_m, n_factors=10, n_epochs=250, lmbda=10, gamma=9e-5, solver="sgd"):
        self.n_u = n_u
        self.n_m = n_m
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lmbda = lmbda
        self.gamma = gamma
        self.solver = solver
        self.missing_value = -1

    def fit(self, X, y):
        """Fits all the latent factors for users and items and saves the resulting matrix representations.
        """
        X, y = check_X_y(X, y)
        
        
        # Create training matrix
        R = np.zeros((self.n_u, self.n_m))
        for idx, row in enumerate(X):
            R[row[0]-1, row[1]-1] = y[idx]  

        # Initialize latent factors
        P = 3 * np.random.rand(self.n_u, self.n_factors) # Latent factors for users
        Q = 3 * np.random.rand(self.n_m, self.n_factors) # Latent factors for movies
                    
        def rmse_score(R, Q, P, missing_value=-1): # [todo]
            I = R != missing_value  # Indicator function which is zero for missing data
            ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
            MSE = ME**2  
            return np.sqrt(np.sum(MSE)/np.sum(I))  # sum of squared errors

        # Fit with stochastic or batch gradient descent
        train_errors = []
        if self.solver == "sgd":
            # Stochastic GD
            users,items = R.nonzero()      
            for epoch in range(self.n_epochs):
                for u, i in zip(users,items):
                    e = R[u, i] - np.dot(P[u, :], Q[i, :].T)  # Error for this observation
                    P[u, :] += self.gamma * ( e * Q[i, :] - self.lmbda * P[u, :]) # Update this user's features
                    Q[i, :] += self.gamma * ( e * P[u, :] - self.lmbda * Q[i, :])  # Update this movie's features
                train_errors.append(rmse_score(R,Q,P)) # Training RMSE for this pass
        elif self.solver == "batch_gd":
            # Batch GD
            for epoch in range(self.n_epochs): 
                ERR = np.multiply(R != 0, R - np.dot(P, Q.T))  # compute error with present values of Q, P, ZERO if no rating   
                P += self.gamma*(np.dot(Q.T, ERR.T).T - self.lmbda*P)  # update rule
                Q += self.gamma*(np.dot(P.T, ERR).T - self.lmbda*Q)  # update rule
                train_errors.append(rmse_score(R,Q,P)) # Training RMSE for this pass
        else:
            print("I'm sorry, we don't recognize that solver.")

#         print("Completed %i epochs, final RMSE = %.2f" %(self.n_epochs, train_errors[-1]))
        self.Q = Q
        self.P = P
        self.train_errors = train_errors
        
        # Return the estimator
        return self

    def predict(self, X):
        """ Predicts a vector of ratings from a matrix of user and item ids.
        """
        X = check_array(X)
        
        y = np.zeros(len(X))
        PRED = np.dot(self.P, self.Q.T)
        for idx, row in enumerate(X):
            y[idx] = PRED[row[0]-1, row[1]-1]
        
        
        return y

    def score(self, X, y):
        """ Element-wise root mean squared error.
        """
        yp = self.predict(X)
        err = y - yp
        mse = np.sum(np.multiply(err, err))/len(err)
        return np.sqrt(mse)

def sgd(R, T, **kargs): 
    """

    Memo
    ----
    1. 

    Ref
    ---
    1. http://sdsawtelle.github.io/blog/output/week9-recommender-andrew-ng-machine-learning-with-python.html


    """

    # Scoring Function: Root Mean Squared Error
    def rmse_score(A, Q, P):
        I = A != missing_value  # Indicator function which is zero for missing data
        ME = I * (A - np.dot(P, Q.T))  # Errors between real and predicted ratings
        MSE = ME**2  
        return np.sqrt(np.sum(MSE)/np.sum(I))  # sum of squared errors

    # [test]
    start_time = timeit.default_timer()

    n_users, n_items = R.shape[0], R.shape[1]

    # Set parameters and initialize latent factors
    # params: n_features, lambda, gamma, n_epochs, mssing_value (default missing value)
    missing_value = kargs.get('missing_value', -1)

    nf = kargs.get('n_features', 20)  # Number of latent factor pairs
    lmbda = kargs.get('lambda', 0.5)  # Regularisation strength
    gamma = kargs.get('gamma', 0.01) # Learning rate
    n_epochs = kargs.get('n_epochs', 50)  # Number of loops through training data
    P = 3 * np.random.rand(n_users, nf) # Latent factors for users
    Q = 3 * np.random.rand(n_items, nf) # Latent factors for items

    # Stochastic GD
    train_errors = []
    test_errors = []
    users,items = R.nonzero()      
    for epoch in range(n_epochs):
        for u, i in zip(users,items):

            # fit toward R: the classifier's prediction (minus FP and FN)
            e = R[u, i] - np.dot(P[u, :], Q[i, :].T)  # Error for this observation
            P[u, :] += gamma * ( e * Q[i, :] - lmbda * P[u, :]) # Update this user's features
            Q[i, :] += gamma * ( e * P[u, :] - lmbda * Q[i, :])  # Update this item's features

        train_errors.append(rmse_score(R,Q,P)) # Training RMSE for this pass
        test_errors.append(rmse_score(T,Q,P)) # Test RMSE for this pass (but consider training split as missing thus not included in RMSE)

    # Print how long it took
    print("Run took %.2f seconds" % (timeit.default_timer() - start_time))
    
    # Check performance by plotting train and test errors
    tPlotTrainTestError = kargs.get('analyze_training', True)
    plotName = kargs.get('plot_name', 'cf_sgd_rmse-%s' % Domain)
    if tPlotTrainTestError: 
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(train_errors, color="g", label='Training RMSE')
        ax.plot(test_errors, color="r", label='Test RMSE')

        plt.title("Error During Stochastic GD")
        plt.xlabel("Number of Epochs")
        plt.ylabel("RMSE")
        # snp.labs("Number of Epochs", "RMSE", "Error During Stochastic GD")

        ax.legend()

        saveFig(plt, plot_path(name=plotName), message='Training in SGD: n_epochs vs rmse error.')

    return (P, Q)

# evaluate
def evalTestTset0(P, Q, T, **kargs): 
    
    Rhat = np.dot(P, Q.T)  # an estimate of ratings

    ### plot 
    plotName = kargs.get('plot_name', 'demo-cf_sgd_rmse')

    plt.clf()
    fig, axs = plt.subplots(figsize=[5, 10], nrows=5, ncols=1, sharex=True)
    fig.suptitle("Stochastic GD Test Performance")
    for idx, ax in enumerate(axs.ravel()):
        vals = Rhat[T == idx+1]
        ax.hist(vals, bins=20, normed=True, label="Ground Truth Rating = %i" %(idx+1))
        ax.legend()
        ax.set_xlim([0, 6])

    saveFig(plt, plot_path(name=plotName), message='test')

    return

# evaluate
def evalTestSet0(P, Q, T, **kargs): 
    """

    labels: true labels in the test tset 
    
    Memo
    ----
    1. offset: only consider the test partition
    """
    import common 

    n_features = P.shape[1]
    assert n_features == Q.shape[1]
    n_users, n_items = P.shape[0], Q.shape[0]

    # take subblocks of P, Q and T?    

    # See how well we did on Test Set Predictions
    Rhat = np.dot(P, Q.T)
    print('... Rhat:\n%s\n' % Rhat)

    plt.clf()

    # if n_labels is None: 
    #     # deduce number of "ratings"
    #     # minL, maxL = np.min(T), np.max(T)
    #     n_labels = np.unique(T)
    #     print('evalTestSet> Found %d unique ratings' % n_labels)
    labels = kargs.get('labels', [])  # only used in the CF-based ensemble learning setting 
    n_ratings = kargs.get('n_ratings', -1) # the numeber of possible ratings (e.g. 1-5)
    if n_ratings == -1:  # -1: unknown 
        if len(labels) > 0: 
            n_ratings = len(np.unique(labels))
        else: 
            n_ratings = len(np.unique(T)) # less reliable
        print('evalTestSet0> Estimated number of ratings/labels: %d' % n_ratings)

    nrows = n_ratings 
    fig, axs = plt.subplots(figsize=[5, 10], nrows=nrows, ncols=1, sharex=True)
    fig.suptitle("Stochastic GD Test Performance")
    
    for r, ax in enumerate(axs.ravel()):  # idx: rating value (e.g. 1-5) 
        rating = r+1

        # Rhat constains estimated "ratings" for T (containing ground truth ratings)
        vals = Rhat[T == rating]  # estimated ratings (e.g. 3.8) for the true rating (e.g. 4)

        # estimated rating is a distribution but hopefully near the true rating
        ax.hist(vals, bins=20, normed=True, label="Ground Truth Rating = %i" %(r+1))
        ax.legend()
        ax.set_xlim([0, nrows+1])

    plot_name = kargs.get('plot_name', 'cf_sgd_test')
    saveFig(plt, plot_path(name=plot_name))

    # if ground truth labels are given (in which case, the prediction for each item/datum is given by combining user/classifier's prediction)
    # this is only useful in the ensemble learning setting 
    if len(labels) > 0: 
        mean_predictions = np.mean(Rhat, axis=0)  # mean prediction of users/classifiers
        print('evalTestSet0> labels: %s' % labels[:100])
        print('...              scores: %s' % mean_predictions[:100])
        auc = common.score(labels, mean_predictions)
        fmax = common.fmax_score(labels, mean_predictions, beta = 1.0, pos_label = 1)
        print('... auc: %f, fmax: %f' % (auc, fmax))

    return

# evaluate
def evalTestSet(labels, Th, **kargs): # labels: true labels, Th: estimates, T: 'true' rating matrix
    # return metrics  # a dictionary: metric -> score
    return evaluate.evalTestSet(labels, Th, **kargs)

# utils_cf
def toUserItem(fold, split='train', save_=False): 
    """
    Convert level-1 training data to user-item dataframe format consisting of the following attributes (columns): 

    ['user_id', 'item_id', 'prediction', 'label'], 

    where user_id corresponds to classifiers 
          item_id corresponds to data points 

    """
    train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
    print('(toUserItem) dim(train_df):%s' % str(train_df.shape))
    
    # get all data IDs 
    users = train_df.columns.values
    ts = train_df if split.startswith('tr') else test_df

    ts = ts.reset_index() # convert multilevel index to flat index
    idx = ts['id'].values
    assert len(idx) == len(set(idx)), "Data IDs are not unique!"
    labels = ts['label'].values

    # split = 'train'
    nU = nUsers = len(users) # number of users/classifiers
    nI = nItems = len(idx)  # number of items/data points

    # treat classifiers as users, data points as items 
    # dataframe format
    #   user_id, item_id, prediction, label
    header = ['user_id', 'item_id', 'prediction', 'label']
    adict = {h: [] for h in header}

    for i, user in enumerate(users): 
        predictions = ts[user].values
        if i == 0: assert len(idx) == len(predictions)

        adict['user_id'].extend([user] * len(idx)) # repeated
        adict['item_id'].extend(idx)
        adict['prediction'].extend(predictions)
        adict['label'].extend(labels) # repeated
    
    ts = DataFrame(adict, columns=header)   # level 1 training data
    print('(toUserItem) sample set=%s | n_ids: %d, n_users: %d, dim(ts): %s' % (split, len(idx), nU, str(ts.shape)))
    print('... ts(n=5):\n%s\n' % ts.head(5))
    if save_: 
        fpath = os.path.join(l1_data_path, 'cf-%s-f%d-b%d.csv' % (split, fold, BagCount))  # naming: test-b3-f1-s1.csv.gz
        print('(toUserItem) Saving level-1 CF %s set (dim=%s) to .csv: %s' % (split, str(ts.shape), fpath))
        ts.to_csv(fpath, sep=',', index=False, header=True)

    return (ts, nUsers, nItems)

def bestBP(ts, labels, scoring_func=None):
    # ts: train_df, test_df
    if scoring_func is None: scoring_func = common.score

    # a Series (index={classifiers}, value: performance score)
    return ts.apply(lambda x: scoring_func(labels, x)).sort_values(ascending = not common.greater_is_better) 

# utils_cf
def toPredictiveScores(fold, **kargs):
    """

    Memo
    ----
    1. analogous to toRatings()
    """
    return utils_cf.toPredictiveScores(fold, **kargs) # a dictionary of 5 entries: ['train', 'test', 'train_labels', 'test_labels', 'users']

# [factor] utils_cf
def _dfToLabelMatrix(ts, n_users, n_items, p_threshold=0.5, users=[], soft_label=False, missing_value=-1):
    """
    Convert rating dataframe (transformed via toUserItem()) to label matrix. 
    Each entry in the label matrix corresponds to the True label. 


    Memo
    ----
    1. Assuming that all classifiers/users predict items/data correctly and perfectly (positive: 1, negative: 0). 

    2. Mask all FP and FN by considering them as missing. 

    3. Call chain: 
        to_rating_matrix_train -> toUserItem -> _toRatingMatrix -> R 
        to_rating_matrix_test -> toUserItem -> _toLabelMatrix -> L

    """
    import pandas as pd

    # get all unique users/classifiers
    if not users: 
        users = np.unique(ts['user_id'].values)

    N = ts.shape[0]

    # Create training and test matrix
    L = np.zeros((n_users, n_items)) 
    print('(toLabelMatrix) input(ts): %s, n_users/classifiers: %d, n_items/data per user: %d' % (str(ts.shape), len(users), n_items))

    if soft_label: 
        pos = ts.loc[ts['label']==1]
        neg = ts.loc[ts['label']==0]
        assert pos.shape[0] > 0 and neg.shape[0] > 0
        print('(toLabelMatrix) n_pos: %d, n_neg: %d' % (pos.shape[0], neg.shape[0])) # n_pos: 6360, n_neg: 12060
        
        # tsp = tsp.loc[tsp['prediction'] >= p_threshold]
        is_pos = ts['label']==1
        is_neg = ts['label']==0
        pred_pos = ts['prediction'] >= p_threshold 
        pred_neg = ts['prediction'] < p_threshold

        nTP = ts.loc[is_pos & pred_pos].shape[0]
        nTN = ts.loc[is_neg & pred_neg].shape[0]
        nFP = ts.loc[is_pos & pred_neg].shape[0]
        nFN = ts.loc[is_neg & pred_pos].shape[0]
        assert nTP > 0, "No true positives found!"

        # treat incorrectly predicted values as missing data
        ts.loc[is_pos & pred_neg, 'prediction'] = missing_value
        ts.loc[is_neg & pred_pos, 'prediction'] = missing_value  # 0 

        # use ground truths for all TP and TN
        # ts.loc[is_pos & pred_pos, 'prediction'] = 1
        # ts.loc[is_neg & pred_neg, 'prediction'] = 0       

        # assert pos.shape[0] > 0 and neg.shape[0] > 0
        # ts_ref = pd.concat([pos, neg], ignore_index=True)
        
        Nc = nTP + nTN # ts_ref.shape[0]
        assert Nc > 0

        accuracy = Nc/(N+0.0)
        print('(toLabelMatrix) dim(ts_ref): %s, precision: %f' % (str(ts.shape), accuracy))
        print(' ...  nTP: %d, nTN: %d, nFP: %d, nFN: %d' % (nTP, nTN, nFP, nFN))  # [log] 
    ### end "masking" wrongly predictive values as missing values

    for i, user in enumerate(users): 
        labels = ts[ts['user_id']==user]['label'] # only look at TP, TN
        # print('... n(scores): %d, n_items: %d' % (len(scores), n_items))
        assert len(labels) == n_items
        L[i,:] = labels

    return L

# [factor] utils_cf
def _dfToRatingMatrix(ts, n_users, n_items, p_threshold=0.5, users=[], select_all=False, missing_value=-1): # I: indicator, R: ratings
    """
    Convert rating dataframe format (transformed via toUserItem()) to the rating matrix format: users/classifiers vs items/data.  

    Memo
    ----
    1. Call chain: 
            to_rating_matrix_train -> toUserItem -> _dfToRatingMatrix -> R 
            to_rating_matrix_test -> toUserItem -> _dfToLabelMatrix -> L

    """
    import pandas as pd

    # get all unique users/classifiers
    if not users: 
        users = np.unique(ts['user_id'].values)

    N = ts.shape[0]
    # Create training and test matrix
    R = np.zeros((n_users, n_items))
    # for i, line in enumerate(ts.itertuples()): # [note] itertuples returns each row as a namedtuple
    #     print line
    #     if i > 5: break
    #     R[line[1]-1, line[2]-1] = line[3]  
    
    # only look at TP, TN
    # ts_ref = ts
    print('(toRatingMatrix) input(ts): %s, n_users/classifiers: %d, n_items/data per user: %d' % (str(ts.shape), len(users), n_items))
    if not select_all: 
        # cond_tp = (ts['prediction'] >= p_threshold) & (ts['label']==1)
        # cond_tn = (ts['prediction']< p_threshold) & (ts['label']==0)

        pos = ts.loc[ts['label']==1]
        neg = ts.loc[ts['label']==0]
        assert pos.shape[0] > 0 and neg.shape[0] > 0
        print('(toRatingMatrix) n_pos: %d, n_neg: %d' % (pos.shape[0], neg.shape[0])) # n_pos: 6360, n_neg: 12060
        
        # tsp = tsp.loc[tsp['prediction'] >= p_threshold]
        is_pos = ts['label']==1
        is_neg = ts['label']==0
        pred_pos = ts['prediction'] >= p_threshold 
        pred_neg = ts['prediction'] < p_threshold

        nTP = ts.loc[is_pos & pred_pos].shape[0]
        nTN = ts.loc[is_neg & pred_neg].shape[0]
        nFP = ts.loc[is_pos & pred_neg].shape[0]
        nFN = ts.loc[is_neg & pred_pos].shape[0]
        assert nTP > 0, "No true positives found!"

        # treat incorrectly predicted values as missing data
        ts.loc[is_pos & pred_neg, 'prediction'] = missing_value
        ts.loc[is_neg & pred_pos, 'prediction'] = missing_value  # 0        

        # assert pos.shape[0] > 0 and neg.shape[0] > 0
        # ts_ref = pd.concat([pos, neg], ignore_index=True)
        
        Nc = nTP + nTN # ts_ref.shape[0]
        assert Nc > 0

        accuracy = Nc/(N+0.0)
        print('(toRatingMatrix) dim(ts_ref): %s, precision: %f' % (str(ts.shape), accuracy))
        print(' ...  nTP: %d, nTN: %d, nFP: %d, nFN: %d' % (nTP, nTN, nFP, nFN))  # [log] nTP: 4689, nTN: 9011, nFP: 1671, nFN: 3049
    ### end "masking" wrongly predictive values as missing values
    
    for i, user in enumerate(users): 
        scores = ts[ts['user_id']==user]['prediction'] # only look at TP, TN
        # print('... n(scores): %d, n_items: %d' % (len(scores), n_items))
        assert len(scores) == n_items
        R[i,:] = scores

    return R

# [factor] utils_cf
def to_rating_matrix_test(fold, p_threshold=0.5, save_=False, missing_value=-1, merge_=False):
    """
    Read training data (and test data) coming from a CV fold and convert them (either by 
    only considering the training data or both) to the rating matrix format (i.e. users/classifiers
    vs items/data)

    Memo
    ----
    1. training and testing matrices need to have the same dimensionality in the current implementation. 
       ... 01.03.19

    2. in testing, consider the training partition as "missing" so that we do not count the prediction error 
       in the training parition

    """ 
    ts_train, n_users, n_items = toUserItem(fold, split='train', save_=save_)

    L = _dfToLabelMatrix(ts_train, n_users, n_items, missing_value=missing_value)

    ts_test, n_users, n_items = toUserItem(fold, split='test', save_=save_)
    Lt = _dfToLabelMatrix(ts_test, n_users, n_items, missing_value=missing_value)

    # merge 
    print('... dim(L): %s, dim(Lt): %s' % (str(L.shape), str(Lt.shape)))
    if merge_: 
        L = np.hstack((L, Lt))
        print('... dim(L_combined): %s' % str(L.shape))

    return (L, Lt)

# [factor] utils_cf
def to_rating_matrix_train(fold, p_threshold=0.5, save_=False, missing_value=-1, merge_=False): 
    def verify(A):
        n_total = A.shape[0] * A.shape[1]
        n_missing = n_total - np.count_nonzero(A)
        r_missing = n_missing/(n_total + 0.0)

        # print('... A[:10]:\n%s\n' % R[:10])
        print('to_rating_matrix_train> n_missing: %d, n_total: %d => ratio: %f' % (n_missing, n_total, r_missing)) 
        return

    print('to_rating_matrix_train> 1. mask FP and FN in the training partition ...')
    ts_train, n_users, n_items = toUserItem(fold, split='train', save_=save_)
    R = _dfToRatingMatrix(ts_train, n_users, n_items, p_threshold=p_threshold, select_all=False, missing_value=missing_value)
    verify(R)

    # test set needs to come in because we need to know all the items at test time ... [todo]
    ts_test, nu_test, ni_test = toUserItem(fold, split='test', save_=save_)
    assert nu_test == n_users, "n_classifiers @ training: %d, in test: %d" % (n_users, nu_test)

    # set select_all to True, because we cannot use label info to mask FP and FN
    print('to_rating_matrix_train> 2. include all test data but without using the ground truth information ...')
    Rt = _dfToRatingMatrix(ts_test, nu_test, ni_test, p_threshold=0.5, select_all=True, missing_value=missing_value)
    verify(Rt)

    # merge
    print('... dim(R): %s, dim(Rt): %s' % (str(R.shape), str(Rt.shape)))
    if merge_:  
        R = np.hstack((R, Rt))
        print('... dim(R_combined): %s' % str(R.shape))

    return (R, Rt)  # R: combines train and test split, Rt: test only

def demo(): 
    ### modified array entries when the condition is met 
    a = np.random.randint(0, 5, size=(5, 4))
    b = a < 3  # ~> (True, False)-valued array
    c = b.astype(int)  # ~> (1, 0)-valued array
    a[b] = 10 # set entries to 10 where condition b is True 

    ### dataframe: create table, named index 
    df = pd.DataFrame({"a": [1,2,3], "b": [3,4,5], "c": [5,6,7]})
    df.index = ["x", "y", "z"]
    df.loc[["x", "y"]]  # select rows by name

    return

# utils_cf
def toRatingMatrix(fold, **kargs): 
    """

    kargs
        p_threshold: 0.5
        missing_value: 0, 
        verbose: 0
        mask: True

    Return
        (R, T, L_train, L_test, U)  # U: users/classifiers

    Memo
    ----
    1. diabetes
       nTP: 4689, nTN: 9011, nFP: 1671, nFN: 3049 (F: 4720)

    """
    return utils_cf.toRatingMatrix(fold, **kargs) # (R, T, L_train, L_test, U)  # U: users/classifiers

def t_cf_ensemble0(**kargs): 

    ### configurations
    # project_path = os.path.abspath('../data/diabetes_cf')
    # try: 
    #     project_path = path = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    # except: 
    #     pass 
    # assert os.path.exists(path)

    ### data format 
    fold = 1
    missing_value = -1 # marker for missing data
    # ts, nU, nI = toUserItem(fold, save_=False)
    # print('(test) n_classifiers: %d, n_data: %d' % (nU, nI))

    # to 'rating' matrix
    # toR0(ts, nU, nI, threshold=0.5)
    # R, Rt = toRatingMatrix0(fold, p_threshold=0.5, merge_=False) # training matrix
    R, Rt = to_rating_matrix_train(fold, p_threshold=0.5, missing_value=0, verbose=True)
    
    # assume all Rt is missing? but this will not allow us to derive hidden features for these data points
    R = np.hstack((R, Rt))

    T, Tt  = to_rating_matrix_test(fold, p_threshold=0.5, merge_=False)

    # assuming training set is missing, so that we do not include rmse for the training part
    # this is ok because we are not trying to fit P and Q for T
    Tm = np.full((T.shape[0], T.shape[1]), missing_value)
    T = np.hstack((Tm, Tt))
    print('... T:\n%s\n' % T[:10])

    assert R.shape[0] == T.shape[0] and R.shape[1] == T.shape[1]

    # CF
    P, Q = sgd(R, T, n_features=20, n_epochs=1000, plot_name='cf_sgd_rmse-%s' % Domain)
    print('... dim(P):%s, dim(Q):%s' % (str(P.shape), str(Q.shape)))

    # evaluate only on the test partition 
    offset = Tm.shape[1]  # Tm: users/classifiers vs items/data
    # P, no change, use all 
    Pt = P
    # Q, starts from test offset 
    Qt = Q[offset:,:]

    nt_users, nt_items = Tt.shape[0], Tt.shape[1]
    assert Pt.shape[0] == nt_users
    assert Qt.shape[0] == nt_items, "nt_items: %d but dim(Qt): %s" % (nt_items, str(Qt.shape))

    train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
    evalTestSet0(Pt, Qt, Tt, labels = test_labels)

    return


        

### end Hashbag

# class Metrics(object): 
#     def __init__(self): 
#         self.adict = {}
#     def add(self, adict): 
#         for k, v in adict.items(): 
#             if not k in self.adict: self.adict[k] = []
#             self.adict[k].append(v)
#     def do(self, op=np.mean):  # perform an operation
#         mx = {}
#         for k, v in self.adict.items(): 
#             mx[k] = op(v)
#         return mx

#     def show(self, op=np.mean, message=''): 
#         assert hasattr(op, '__call__')
#         if not message: 
#             for metric, val in self.do(op).items(): 
#                 print('... metric=%s, applied %s => %s' % (metric, op.__name__, val))
#         else: 
#             message = ''
#             for metric, val in self.do(op).items(): 
#                 message += '... metric=%s, applied %s => %s\n' % (metric, op.__name__, val)
#             div(message=message, symbol='*', border=2)

def run_baseline_stacker(topk=-1, metric='fmax', parallelize=True): 
    
    target_metric = metric # kargs.pop('metric', 'fmax')
    # topk = kargs.get('topk', -1)
    perf = run_stacker(dataset='bp', parallelize=parallelize)  # dataset='bp'
    perf = PerformanceMetrics.getTopK(perf, metric=target_metric, k=topk, reverse=True, verbose=True)
    return perf
### alias 
base_stackers = run_baseline_stacker

def run_pref_stacker_suite(**kargs): 
    """
    Run stacking over the CF-produced training data containing preference information (see cf_als.ImplicitMF.iter_preference())

    """
    import stacking, common 
    import utils_cf as uc
    # from evaluate import Metrics # evalTestSet spoke examples for losing if you going to my phone directory which I think you might be

    ret = {}  # output
    # perf = PerformanceMetrics()   # rows: metrics, cols: bps

    perfMetrics = [] # performance metrics object for each method
    dataset = kargs.get('dataset', 'bp')

    # probably always want to compare with base predictors 
    topk_bps = kargs.get('topk_bps', -1)
    base_perf = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', True) else None 
    if base_perf is not None: perfMetrics.append(base_perf)

    n_fold = 5

    ######### WMF parameters #########
    params = {}
    params['n_factors'] = args.get('n_factors', MFEnsemble.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)
    params['alpha'] = kargs.get('alpha', MFEnsemble.alpha)
    
    # parameters for confidence matrix
    params['mode'] = kargs.get('mode', 'brier')  # confidence matrix
    params['masked'] = kargs.get('masked', True) # mask FP and FN? this makes Cui 'sparse'

    # parameters for ALS methods 
    params['policy'] = kargs.get('policy', 'rating')  # options: {'rating', 'preference', 'label'}
    ######################################################


    perfMetrics = []
    # example method name:wmf_F10_A100_Xbrier_preference, which corresponds to: wmf_F10_A100_Xbrier_preference-validation-3.csv.gz
    tsetID = MFEnsemble.get_tset_id(params)

    # datasets 
    datasets = common.match_exact(path=ProjectPath, method=tsetID, file_type='validation', ext='csv.gz', verify=True) # exception_=False
    
    for dataset in datasets: 
        # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
        assert dataset.find('pref') > 0, "dataset {0} does not contain preference scores ...".format(dataset)
        for method in ['mean', ]:
            # load preference scores

            perf_per_fold = []
            for fold in range(n_fold): 
                # load preference scores
                train_pref, train_labels, test_pref, test_labels = common.read(fold, path=ProjectPath, dataset=dataset, reconstructed_testset=True) # common.read_fold(project_path, fold)        

                # load original data 
                train_df, train_labels, test_df, test_labels = common.read(fold, path=ProjectPath, dataset='bp', reconstructed_testset=True)
                assert test_pref.columns.size == 2 * test_df.columns.size, "dim(test_pref): {0}, dim(test_df): {1}".format(test_pref.shape, test_df.shape)

                if method == 'mean': 
                    # test_pref is a augmentd data set, therefore we need to subset the columns that pertain to classifiers but not the indicators
                    T_pref = test_pref[test_df.columns].T   # transpose to get users-by-items format
                    T = test_df.values.T

                    predictions = uc.combiner(T_pref, aggregate_func='pref', T=T)

                    perf = analyzePerf(test_labels, predictions, method='{combiner_type}_pref'.format(combiner_type=method))
                    perf_per_fold.append(perf)

            perfMetrics.append( PerformanceMetrics.consolidate(perf_per_fold, test_= False) ) # foreach metric, take average over CV folds

        ### end foreach method 
        div('(run_pref_stacker_suite) Completed method {0} #'.format(method))
    ### end foreach dataset 

    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    if base_perf is not None: 
        docs = {'method': 'preference_stacking', 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return perfAll

def run_preference_combiner(dataset, **kargs): 
    from cf_spec import System 
    # pref scores in one dataset and original BP score in another
    n_fold = System.foldCount 
    method = kargs.get('method', 'mean') 
    
    # the algorithm behind the dataset + combiner method
    full_method = '{prefix}_C{combiner}_pref'.format(prefix=dataset, combiner=method)

    perf_per_fold = []
    for fold in range(n_fold): 
        
        # preference score 
        train_pref, train_labels, test_pref, test_labels = common.read(fold, path=ProjectPath, dataset=dataset, reconstructed_testset=True) # common.read_fold(project_path, fold)        
        
        # original data 
        train_df, train_labels, test_df, test_labels = common.read(fold, path=ProjectPath, dataset='bp', reconstructed_testset=True)
        assert test_pref.columns.size == 2 * test_df.columns.size, "dim(test_pref): {0}, dim(test_df): {1}".format(test_pref.shape, test_df.shape)

        if method == 'mean': 
            T_pref = test_pref[test_df.columns].values.T   # transpose to get users-by-items format
            T = test_df.values.T

            predictions = combiner(T_pref, aggregate_func='pref', T=T)
            perf_per_fold.append( analyzePerf(test_labels, predictions, method=full_method) )
    div('(run_preference_combiner) Completed method %s #' % full_method)

    return PerformanceMetrics.consolidate(perf_per_fold, test_= False)  # foreach metric, take average over CV folds

def run_simple_combiner(dataset, **kargs):
    from cf_spec import System 

    perf_per_fold = []
    method = kargs.get('method', 'mean')
    n_fold = System.foldCount

    # the algorithm behind the dataset + combiner method
    full_method = '{prefix}_C{combiner}'.format(prefix=dataset, combiner=method)
    for fold in range(n_fold): 
        # original data 
        train_df, train_labels, test_df, test_labels = common.read(fold, path=ProjectPath, dataset=dataset, reconstructed_testset=True)
        Th = test_df.values.T

        if method == 'mean': 
            predictions = combiner(Th, aggregate_func=method)
        elif method == 'median': 
            predictions = combiner(Th, aggregate_func=np.median)
        else:  # user provide a customize aggregation function (e.g. weighted average)
            customized_func = kargs['aggregate_func'] 
            assert hasattr(customized_func, '__call__')
            method = customized_func.__name__
            predictions = combiner(Th, aggregate_func=customized_func)

        perf_per_fold.append( analyzePerf(test_labels, predictions, method=full_method) )
    div('(run_simple_combiner) Completed method %s' % full_method)

    return PerformanceMetrics.consolidate(perf_per_fold, test_=kargs.get('test', False))  # foreach metric, take average over CV folds

def run_combiner(dataset, **kargs):
    """
    
    Memo
    ----
    1. Two types of training data 
       - regular 
       - augmented by preference scores

    """

    perf = PerformanceMetrics()
    if MFEnsemble.is_preference_data(dataset): 
        if kargs.get('pref_aug', False): 
            perf = run_pref_stacker(dataset)  # stacker on the augmented dataset
        else: 
            perf = run_preference_combiner(dataset, method=kargs.get('method', 'mean'))  # simple combining rule using preference scores as heuristics
    else: 
        # kargs: method, aggregation_func, (test, )
        perf = run_simple_combiner(dataset, **kargs)

    return perf

def test_combiner(datasets=[], param_grid={}, **kargs):
    """
    Counterpart of test_stacker()
    A combiner is a simple stacker. 

    Input
    -----
    datasets 
    param_grid
    
    either the named datasets or the parameter grid has to be given

    """
    import stacking, common 
    from sklearn.model_selection import ParameterGrid
    # from evaluate import Metrics # evalTestSet

    ret = {}  # output
    # perf = PerformanceMetrics()   # rows: metrics, cols: bps

    perfMetrics = [] # performance metrics object for each method
    # dataset = kargs.get('dataset', 'bp')

    # probably always want to compare with base predictors 
    topk_bps = kargs.get('topk_bps', -1)
    base_perf = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', False) else None 
    if base_perf is not None: perfMetrics.append(base_perf)

    n_fold = 5 # System.foldCount
    if len(datasets) == 0: 
        assert len(param_grid) > 0, "Both the named datasets and parameter grid are missing!"
        datasets = MFEnsemble.name_tsets(param_grid, meta_params=kargs.get('meta_params', {'mode': 'brier', 'policy': 'rating', }))

    # Now, given a set of named datasets ... 
    for dataset in datasets: 
        # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
        perfMetrics.append( run_combiner(dataset, **kargs) ) # kargs: method, aggregation_func, (test, )

    ### end foreach dataset 

    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    if base_perf is not None: 
        docs = {'method': kargs.get('method', 'mean'), 
                'target_method': kargs.get('target_method', 'wmf'), 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    post_analysis(perfAll, context=kargs.get('context', 'test_combiner'), highlight=[])

    return perfAll

def run_pref_stacker(dataset, **kargs): 
    """
    Given specific data set augmented with preference scores, generate the final prediction
    based on simple combiner methods using preference scores and the BP probability scores. 
    """
    import stacking, common 
    # from evaluate import Metrics # evalTestSet

    # ret = {}  # output
    perfMetrics = [] # performance metrics object for each method
    # dataset = kargs.get('dataset', 'bp')

    # probably always want to compare with base predictors 
    topk_bps = kargs.get('topk_bps', -1)
    base_perf = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', False) else None 
    if base_perf is not None: perfMetrics.append(base_perf)

    n_fold = kargs.get('n_fold', 5)
    for dataset in [dataset, ]: 
        # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
        assert dataset.find('pref') > 0, "dataset {0} does not contain preference scores ...".format(dataset)
        for method in ['mean', ]:
            # load preference scores

            full_method = '{prefix}_C{combiner}_pref'.format(prefix=dataset, combiner=method)
            perf_per_fold = []
            for fold in range(n_fold): 
                train_pref, train_labels, test_pref, test_labels = common.read(fold, path=ProjectPath, dataset=dataset, reconstructed_testset=True) # common.read_fold(project_path, fold)        
                
                # load original data 
                train_df, train_labels, test_df, test_labels = common.read(fold, path=ProjectPath, dataset='bp', reconstructed_testset=True)
                assert test_pref.columns.size == 2 * test_df.columns.size, "dim(test_pref): {0}, dim(test_df): {1}".format(test_pref.shape, test_df.shape)

                if method == 'mean': 
                    T_pref = test_pref[test_df.columns].values.T   # transpose to get users-by-items format
                    T = test_df.values.T

                    predictions = combiner(T_pref, aggregate_func='pref', T=T)

                    perf = analyzePerf(test_labels, predictions, method=full_method)
                    perf_per_fold.append(perf)

            perfMetrics.append( PerformanceMetrics.consolidate(perf_per_fold, test_= False) ) # foreach metric, take average over CV folds

        ### end foreach method 
        div('(run_pref_stacker) Completed method %s #' % full_method)
    ### end foreach dataset 

    perfAll = PerformanceMetrics.merge(perfMetrics)
    # for metric in PerformanceMetrics.tracked: 
    #     ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    if base_perf is not None: 
        docs = {'method': 'preference_stacking', 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return perfAll

def run_stacker(**kargs): 
    """
    Run stacker routines. 
    Not part of this module but put here for the convenience of comparison. 

    **kargs
        topk_bps

        parallelize

    """
    import stacking 
    # from evaluate import Metrics # evalTestSet

    ret = {}  # output
    # perf = PerformanceMetrics()   # rows: metrics, cols: bps

    perfMetrics = [] # performance metrics object for each method
    dataset = kargs.get('dataset', 'bp')

    # probably always want to compare with base predictors 
    topk_bps = kargs.get('topk_bps', -1)
    base_perf = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', False) else None 
    if base_perf is not None: perfMetrics.append(base_perf)

    for method in ['lasso', 'enet', 'rf', 'gb', ]:  # rf: random forest, 'gb': gradient boosting tree
        predictions_df = stacking.run(name=method, dataset=dataset, parallelize=kargs.pop('parallelize', True))  # stacker=

        ### apply a scoring function to each fold and then take the average
        # predictions_df.groupby('fold').apply(lambda x: common.score(x.label, x.prediction)).mean()
        
        # stackerMetrics = Metrics()
        method_id = '{base}_{dataset}_stacker'.format(base=method, dataset=dataset)
        if dataset != 'bp': print('(run_stacker) Stacking on transformed dataset: {0} via stacker: {1}'.format(dataset, method_id))

        # foreach CV fold
        perf_per_fold = []
        for name, group in predictions_df.groupby('fold'): 
            # mdict = evalTestSet(group['label'], group['prediction'], aggregate_func=np.mean, fold=name)
            # stackerMetrics.add(mdict)

            # perf.add(scores=mdict, method=method_id)  # use a dataframe to keep tracck of performance scores, which then faciliates plotting
            perf = analyzePerf(group['label'], group['prediction'], method=method_id)
            perf_per_fold.append(perf)
 
        # stackerMetrics.report(op=np.mean, message='stacking via %s' % method)

        # each perf has only 1 column (represeting a particular kind of stacker)
        perfMetrics.append( PerformanceMetrics.consolidate(perf_per_fold, test_= False) ) # foreach metric, take average over CV folds
    
    # merge all performance metrics into one big table 
    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    if base_perf is not None: 
        docs = {'method': 'stacking', 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
    return perfAll

def run_stacker_suite(**kargs):
    import stacking

    ret = {}
    perfMetrics = [] # performance metrics object for each method

    ### Baseline Methods: base predictors (BPs), stackers over BP outputs (or BP stackers)

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    if kargs.get('run_bp', True): 
        base_perf = base_predictors(topk=kargs.get('topk_bps', -1), metric=kargs.get('metric', 'fmax'))
        perfMetrics.append(base_perf) 

    ## how does it compare to stacking on BP outputs?
    if kargs.pop('run_bp_stacker', True):
        perfStacker = base_stackers(topk=kargs.get('topk_bp_stackers', -1), metric=kargs.get('metric', 'fmax'))  # dataset='bp'
        perfMetrics.append(perfStacker)

    method = kargs.get('method', '')  # if given, focus only on the methods containing this method keyword (e.g. 'nmf'); if '', then consider all methods
    keywords = kargs.get('keywords', [])  # e.g. to focus on those with n_factors = 20 => put 'F20' in the keyword

    ### Condition: Baseline methods established

    if kargs.get('exact', False): 
        # method has to match exactly with the prefixed part of the file name 
        # e.g. Say method <- 'nmf', then we are only looking at { 'nmf-validation-{fold}.csv.gz' | fold = 0, 1, 2...}
        #      nmf_item_kmeans_sim_F10-validation-0.csv.gz for instance, is not considered a match
        datasets = common.match_exact(path=ProjectPath, method=method, file_type='validation', ext='csv.gz', verify=True) # exception_=False
    else: 
        # search relevant training data whose names contain the given method (as a keyword); use keywords argument as additional constraints to narrow down the scope 
        datasets = common.match(path=ProjectPath, method=method, keywords=keywords, file_type='validation', ext='csv.gz', verify=True)

    div('(run_stacker_suite) Found %d matched datasets' % len(datasets), symbol='=', border=1)

    for dataset in datasets:
        if not kargs.get('pref_aug', True) and dataset.find('pref') > 0:  
            print('(run_stacker_suite) Not using the preference-augmented dataset ...')
            perfMetrics.append(run_pref_stacker(dataset=dataset))  
        else: 
            perfMetrics.append(run_stacker(dataset=dataset, parallelize=kargs.get('parallelize', True), 
                reconstructed_testset=kargs.get('reconstructed_testset', True) ))

    # merge all performance metrics into one big table 
    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)

    if kargs.get('evaluation', True): 
        for metric in PerformanceMetrics.tracked: 
            ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

        post_analysis(perfAll, context=kargs.get('context', 'stacker_suite'), highlight=['stacker', ])

    # context = kargs.get('context', 'stacker_suite')
    # if base_perf is not None: 
    #     docs = {'method': 'stacking', 'dataset': dataset}
    #     PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    #     # set operation (after collecting the performance metrics of multiple methods)
    #     post_analysis(perfMetrics, context=kargs.get('context', 'stacker_suite'), highlight=['wmf', 'nmf', ])

    return perfAll

def mfb_ensemble():   # basic matrix factorization
    """
    Matrix factorization via SGD. 

    Memo
    ----
    1. evaluate CF-based ensemble learning in two ways
    """
    from evaluate import Metrics, plot_roc # evalTestSet

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    mfMetrics = Metrics() # matrix factorization metrics
    userCV = []
    offset = n_users_train = n_items_train = 0
    n_users_test = n_items_test = 0
    for fold in range(n_fold): 

        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        R, T, L_train, L_test, U = utils_cf.to_rating_matrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        # => R: users vs items
    
        # [note] need to combine the test split (T), albeit not using its true labels, in order to derive 
        # their latent factors
        Ra = np.hstack((R, T))  # Ra: R augmented (by the test split as we need to know their latent features)

        # test data Ta (i.e. adjusted T)
        #    assuming training set is missing, so that we do not include rmse for the training part
        #    this is ok because we are not trying to fit P and Q for T
        Tt = np.full((R.shape[0], R.shape[1]), missing_value)
        Ta = np.hstack((Tt, T))

        assert Ta.shape == Ra.shape
        
        ### matrix factorization
        # [note] T is only needed for testing
        P, Q = sgd(Ra, Ta, n_features=20, n_epochs=100, plot_name='cf_sgd_rmse-%s' % Domain)
        print('... dim(P):%s, dim(Q):%s' % (str(P.shape), str(Q.shape)))

        # Now predict test data
        offset = R.shape[1]
        Qt = Q[offset:, :]

        Th = np.dot(P, Qt.T) 

        # now we need to combinine rule for the final ensemble prediction
        metrics = evalTestSet(L_test, Th, aggregate_func=np.mean, fold=fold)
        mfMetrics.add(metrics)

    # op: a combiner function for performance scores across CV folds
    mfMetrics.report(op=np.mean, message='MF ensemble based on SGD on matrix factorization')

    return

# evaluate
def analyzeBasePerf(L_test, T, U=None, **kargs): 
    # import evaluate

    # output PerformanceMetrics object  # metrics (rows) vs classifiers (columns)
    return evaluate.analyzeBasePerf(L_test, T, U, **kargs)

# evaluate
def analyzePerf(L_test, Th, method, **kargs):     
    return evaluate.analyzePerf(L_test, Th, method, **kargs)

# evaluate 
def analyzePerfStacker(fold, Rh, Th, **kargs): 
    return evaluate.analyzePerfStacker(fold, Rh, Th, **kargs)

# evaluate
def compareEstimates(L_test, Ts, **kargs):
    """
    Compare the probability estimates between T and Th (possibly multple sets), where T comes from base predictors and 
    Th comes from a CF algorithm (specifiied by - method -). 

    Params
    ------
    Ts: a set of matrices representing different probability estimates
        T, Th1, Th2, ... 

    **kargs

    U: users
    method: 
    """
    # import evaluate
    return evaluate.compareEstimates(L_test, Ts, **kargs)

# evaluate
def compareEstimates0(T, L_test, Th=None, method='cf', **kargs): 
    """

    Params
    ------

    **kargs

    users: classifier names would useful if availabel

    Result
    ------
    1. Under which metrics do CF-based methods have advantage over BPs? 
       

    """
    def alert(hashtable, message=''):
        if not message: message="... metric: {0:s} | CF: {1:5.3f} > (best) BP: {2:5.3f}"
        for metric, values in hashtable.items(): 
            target_score, best_bp_score = values
            print(message.format(metric, target_score, best_bp_score))

    from evaluate import perf_measures, evalTestSet # (y_true, y_score)
    from evaluate import Metrics, PerformanceMetrics, plot_roc
    # import common
    
    ### I. Compare max bp performance with the ensemble performance
 
    # test set
    n_users, n_items = T.shape[0], T.shape[1]

    bpMetrics = Metrics()

    metrics = Metrics.tracked  # ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', ] 
    perf_scores = {m:[] for m in metrics}

    # use argmax to find which classifier returns the desired result (e.g. max performance)

    # BP performance on the test split (which is to be compared to Th, re-estimated by CF-based methods)
    for i in range(n_users):  # foreach ith user
        predictions = T[i, :]   # a particular user/classifier
        labels = L_test 

        perf_scores['auc'].append(common.score(labels, predictions))
        perf_scores['fmax'].append(common.fmax_score(labels, predictions, beta = 1.0, pos_label = 1))
        perf_scores['fmax_negative'].append(common.fmax_score(labels, predictions, beta = 1.0, pos_label = 0))

        # bpMetrics.add_value('auc', common.score(labels, predictions))
        # bpMetrics.add_value('fmax', common.fmax_score(labels, predictions, beta = 1.0, pos_label = 1))
        # bpMetrics.add_value('fmax_negative', common.fmax_score(labels, predictions, beta = 1.0, pos_label = 0))

        # [todo] Classification metrics can't handle a mix of binary and continuous targets
        metrics2 = perf_measures(labels, predictions)
        # bpMetrics.add_value('sensitivity', metrics2['sensitivity'])  # recall, TPR
        # bpMetrics.add_value('specificity', metrics2['specificity'])  # TNR
        perf_scores['sensitivity'].append(metrics2['sensitivity'])
        perf_scores['specificity'].append(metrics2['specificity'])

    # T estimate from MF method
    cf_perf_scores = {}

    # prepare return value
    target_method = kargs.get('target_method', 'cf')
    methods = ['bp', target_method, ]   # base predictor (bp), collaborative filtering (cf)
    ret = {method: {} for method in methods}
    perf = PerformanceMetrics()
    if Th is not None: 
        assert Th.shape[0] == T.shape[0] and Th.shape[1] == T.shape[1]

        # [result]
        metrics_cf_ko_bp = {}  # which metrics, CF-based methods are better than BP? 

        # CF scores
        cf_perf_scores = evalTestSet(L_test, Th, aggregate_func=np.mean)  # mean is not the ideal combining rule
        for metric, score in cf_perf_scores.items(): 
            if not metric in metrics: continue
            ret[method][metric] = score
        perf.add(scores=ret['cf'], method=method)  # use a dataframe to keep tracck of performance scores, which then faciliates plotting 
        
        # BP scores
        for metric, scores in perf_scores.items(): 
            if not metric in metrics: continue
            ret['bp'][metric] = best_bp_score = max(scores)

            tCF_KO_BP = cf_perf_scores[metric] > best_bp_score
            print('(result) metric=%s, max(bp): %f, MF: %f, >bp? %s' % (metric, best_bp_score, cf_perf_scores[metric], tCF_KO_BP))
            if tCF_KO_BP: metrics_cf_ko_bp[metric] = [cf_perf_scores[metric], best_bp_score]
        perf.add(scores=ret['bp'], method='bp')

        # [result]
        alert(metrics_cf_ko_bp)

    else: 
        # base predictors only
        for metric, scores in perf_scores.items(): 
            if not metric in metrics: continue
            ret['bp'][metric] = best_bp_score = max(scores)
            print('... metric=%s, max(bp): %f' % (metric, best_bp_score))
        perf.add(scores=ret['bp'], method='bp')
    return ret # a two-level dictionary ret: method -> metric -> score

def base_predictors(topk=-1, metric='fmax'): 
    """

    Input
    -----
        metric: the metric used to select the top K model (when topk > 0)
    """
    from evaluate import Metrics, plot_roc
    from evaluate import PerformanceMetrics # as perfm

    ret = {}
    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    perfx = []
    for fold in range(n_fold): 
     
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        R, T, L_train, L_test, U = utils_cf.to_rating_matrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        n_users, n_items = R.shape[0], R.shape[1]
        print('[base_predictors] dim(T): %s, n_test: %d' % (str(T.shape), len(L_test)))

        df_perf = analyzeBasePerf(L_test, T, U, unbag=True, bag_count=BagCount)  # tracked_metrics = PerformanceMetrics.tracked_metrics
        perfx.append(df_perf)

    # consolidate CV folds
    perf = PerformanceMetrics.consolidate(perfx, unbag=True)  # foreach metric, take average over CV folds

    # add average, median, etc. 
    # perf.aggregate(np.mean, new_col='bp_mean')  # this adds a new column
    perf.aggregate(np.median, new_col='bp_median')

    # only see the top K 
    target_metric = metric
    if topk > 0: perf = PerformanceMetrics.getTopK(perf, metric=target_metric, k=topk, reverse=True, verbose=True)

    # add meta data 
    # perf.add_doc({})

    ret['metrics'] = perf
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perf, metric=metric, verbose=False) 

    # merge bags

    return ret['metrics']   

# configure WMF-based ensemble learning paramters here
# > cf_spec. 
# class MFEnsemble(object)

# solution 1
def nmf_ensemble(**kargs): 
    """

    Memo
    ----
    1. related modules: 
        selection 


    """
    from evaluate import Metrics, plot_roc
    from evaluate import analyzePerfStacker
    import utils_cf as uc

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    
    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)
    params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)

    ret = {}

    perfMetrics = []

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', True) else None

    # topk = 30
    method = 'nmf'
    kinds = ['mean_aggregate', ] 
    
    # stackers = ['enet', 'rf', ]  # 'cf_stacker_{0}'.format(kind)
    # kinds = kinds + stackers

    nmfMetrics = {k: [] for k in kinds}
    nmfCV = {k: [] for k in kinds}

    # also keep track of reproduced probabilities
    # > this is delegated to stacking

    for fold in range(n_fold): 
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        R, T, L_train, L_test, U = uc.to_rating_matrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        n_users, n_items = R.shape[0], R.shape[1]
        # print('[nmf_ensemble] dim(T): %s, n_test: %d' % (str(T.shape), len(L_test)))

        # [note] Surprise does not take inputs in the form of R (rating matrix)
        P, Q = uc.applyMF(fold, n_factors=params['n_factors'], n_epochs=params['n_epochs'], fill=missing_value)  # P and Q
        print('(nmf_ensemble) Fold: {0} | dim(P): {1}, dim(Q): {2}'.format(fold, P.shape, Q.shape))

        Rh, Th = uc.predict_by_factors(P, Q, test_offset=len(L_train), test_set_only=False) # Rh <- None if test_set_only: True
        if kargs.get('save', True): 
            uc.save_reconstructed_training_data(Rh, L_train, fold, method, verify=True, U=U)
            # uc.save_reconstructed_test_data(Th, L_test, fold, method, verify=True, U=U)

        for kind in kinds: 

            # metrics = compareEstimates0(T, L_test, Th=Th, R=None, L_train=None)

            # optinal params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
            method_specific = MFEnsemble.name_method(method, kind, params=params)  # '{method}_{kind}'.format(method=method, kind=kind)
            
            # # also consider stacking on top of the reproduced probabilities
            # if kind in stackers: # is a kind of stacker => need special performance Handler
                
            #     # **kargs: classifier hyperparams
            #     perf, df_prediction = analyzePerfStacker(fold, Rh, Th, method=method_specific)  # run stacker on top of the reproduced probabilities
            #     y_true, y_score = df_prediction['label'], df_prediction['prediction']
            #     assert all(L_test == y_true)
            #     nmfMetrics[kind].append(perf)
            #     nmfCV[kind].append((L_test, y_score))
            
            nmfMetrics[kind].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean, T=T, fold=fold))  # analyzePerf -> { compare* } where compare* is a set of analysis functions (e.g.  compareEstimates_
            nmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )

        # bpMetrics.add(metrics['bp'])
        # mfMetrics.add(metrics['cf'])

    ## evaluation 
    for kind in kinds: 
        if kind in nmfCV: 
            method_specific = MFEnsemble.name_method(method, kind, params=params)
            plot_roc(nmfCV[kind], file_name='roc-{method}-{project}'.format(method=method_specific, project=Domain))  # an import from evaluate
    
    # Q1: does the reconstructed prob "better? 
    # perfMetrics.extend([nmfMetrics[kind] for kind in kinds])
    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(nmfMetrics[kind]) for kind in kinds]) # merge all CV-consolidated PerformanceMetrics objects

    # ret: output dictionary
    #      key: {methods}, {metrics}
    ret['metrics'] = perfAll  # foreach metric, take average over CV folds
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    if base_perf is not None: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return perfAll

# solution 1a 
def nmf_similarity_ensemble(**kargs):
    """
    Use NMF factorized latent matrices to compute similarity
    """ 
    def save_factors(P, Q, fold=-1, method='nmf', cols_users=[], cols_items=[]):
        if not fold in (-1, 1): return # do nothing

        if len(cols_users) == 0: cols_users = U # U in function closure 
        MFEnsemble.save_factors(P, cols=cols_users, file_name='{method}_P.csv'.format(method=method))
        MFEnsemble.save_factors(Q, cols_items=cols_items, file_name='{method}_Q.csv'.format(method=method)) 
        return
    def save_array(S, kind, fold, method='nmf'): 
        if fold != 1: return # do nothing
        MFEnsemble.save_array(S, file_name='{method}_{kind}_S.csv'.format(method=method, kind=kind))

    import math
    from evaluate import Metrics, plot_roc
    import utils_cf as uc
    from itertools import product

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 

    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)
    params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)
    topk = 30

    # output
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', True) else None


    topk = 30
    # kinds = ['user', 'item', 'user_topk', 'item_topk', 'user_kmeans', 'item_kmeans', 'user_latent', 'item_latent', 'user_spectral', 'item_spectral', ]
    clusterings = ['kmeans', 'latent', 'spectral']  # product(*[kind[:2], clusterings])
    kinds = ['user', 'item', 'user_topk', 'item_topk', ]
    if kargs.get('run_clustering', False): 
        kinds += ['_'.join(pair) for pair in product(*[kinds[:2], clusterings])]

    nmfMetrics = {k: [] for k in kinds}
    nmfCV = {k: [] for k in kinds}
    
    # base, method = 'nmf', 'nmf_sim'
    method = 'nmf'
    Pe = Qe = 0.
    for fold in range(n_fold): 
     
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        R, T, L_train, L_test, U = utils_cf.to_rating_matrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        n_users, n_items, n_items_total = R.shape[0], R.shape[1], len(L_train)+len(L_test)
        print('(nmf_ensemble) dim(R):{0}, dim(T): {1}, n_train: {2}, n_test: {3}'.format(R.shape, T.shape, len(L_train), len(L_test)))

        # [note] Surprise does not take inputs in the form of R (rating matrix) and T (test matrix)
        P, Q = uc.applyMF(fold, n_factors=params['n_factors'], n_epochs=params['n_epochs'], fill=missing_value)  # P and Q

        # but we only need the Q from the test split 
        # Qt = Q[R.shape[1]:, :]   # 30 * 10, (768-x) * 10
        # print('... dim(P): %s, dim(Qt): %s' % (str(P.shape), str(Qt.shape)))
        # Th = np.dot(P, Qt.T)
        Pe += P
        Qe += Q

        ### now, we will use the learned latent represntation to build similarity matrix, which is then used to make predictions on T

        ## A. NMF + neighborhood

        # use P (user latent features) to construct Su 
        # [todo] MF => similarity matrix
        for kind in kinds[:2]: 
            
            factors = P if kind == 'user' else Q  # P: all users vs factors, Q: all items vs factors
            S = uc.evalSimilarityByLatentFeatures(factors, epsilon=1e-9)
            
            # axis = 0 if kind == 'user' else 1
            dimS = n_users if kind == 'user' else n_items_total
            assert (S.shape[0] == S.shape[1] == dimS), "kind={0}, dim(S)={1} but expecting: {2}".format(kind, S.shape, dimS)
            print('(test) kind={0} | dim(S): {1} (S[i,j] in [0, 1]?):\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # [note] R, T are to be merged prior to calling predict_nobias or predict_topk
            Rh, Th = uc.predict(R, T, S=S, kind=kind, canonicalize=True)
            assert T.shape == Th.shape, "dim(T): {0} but dim(Th): {1}".format(T.shape, Th.shape)

            method_specific = MFEnsemble.name_sim_method(method, kind=kind, params=params) # '{base}_{kind}_sim'.format(base=base, kind=kind)
            nmfMetrics[kind].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean))
            nmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )  # plot K-Fold CV
            if kargs.get('save', True): uc.save_reconstructed_training_data(Rh, L_train, fold, method=method_specific, verify=True, U=U)

            ## use top K only 
            topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
            Rh, Th = uc.predict(R, T, S=S, kind=kind, topk=topk, canonicalize=True)
        
            kind_topk = '%s_topk' % kind
            method_specific = MFEnsemble.name_sim_method(method, kind=kind_topk, params=params) # '{base}_{kind}'.format(base=base, kind=kind_topk)
            nmfMetrics[kind_topk].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean) )
            nmfCV[kind_topk].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )

            ## NMF + Clustering
            if kargs.get('run_clustering', False):
                for clustering in clusterings: 
                    kind_cluster = '%s_%s' % (kind, clustering)
                    method_specific = MFEnsemble.name_sim_method(method, kind=kind_cluster, params=params)
                    user_labels, item_labels = uc.nmfCluster(P, Q, n_clusters=params['n_factors'], method=clustering, evaluate=True)
                    
                    cluster_labels = user_labels if kind == 'user' else item_labels
                    Rh, Th = uc.predict_by_cluster(R, T, similarity=S, kind=kind, C=cluster_labels)

                    nmfMetrics[kind_cluster].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean) )
                    nmfCV[kind_cluster].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )
                    if kargs.get('save', True): uc.save_reconstructed_training_data(Rh, L_train, fold, method=method_specific, verify=True, U=U)

                    # maybe not using mean aggregate, use smarter ways 
                     
 
        ### end user-item loop
    ### end foreach CV fold 
    Pe = Pe/n_fold
    Qe = Qe/n_fold
    save_factors(Pe, Qe)
    
    for kind in kinds: 
        # nmfCV[kind] is a sequence of (lable, prediction)
        if kind in nmfCV: 
            method_specific = MFEnsemble.name_sim_method(method, kind=kind, params=params)
            plot_roc(nmfCV[kind], file_name='roc-{method}-{project}'.format(method=method_specific, project=Domain))  # an import from evaluate

    ## Compare with baseline methods
    
    # perf_unfold = [PerformanceMetrics.consolidate(nmfMetrics[kind]) for kind in kinds]
    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(nmfMetrics[kind]) for kind in kinds])

    ret['metrics'] = perfAll # PerformanceMetrics.consolidate(fullMetrics)  # foreach metric, take average over CV folds
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    if base_perf is not None: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
    return perfAll

# utils_cf
def combiner(Th, aggregate_func='mean', axis=0, **kargs): 
    # return predictions
    return utils_cf.combiner(Th, aggregate_func, axis=axis, **kargs)

def combiner_sim(Rh, Th, similarity):

    # S = uc.evalSimilarityByLatentFeatures(factors, epsilon=1e-9)
    Ra = np.hstack((Rh, Th))
    for j in range(Th.shape[1]): 
        # top_k_items = tuple([np.argsort(similarity[:,j])[:-k-1:-1]])  # top k most similar items
        similarity[:,j]

    return

def t_neighborhood_ensemble(**kargs): 
    import utils_cf as uc
    from evaluate import Metrics, plot_roc, evalTestSet, analyzePerf
    div(message='Running memory-based approach ...', symbol='#', border=1)
    
    n_fold = 5
    p_th = 0.5
    missing_value = 0 # marker for missing data
    topk = 30

    ret = {}  # output
    perfMetrics = [] # method-wise performance metrics including (cosine and pearson correlation-based methods)
    # options: target_metric is the metric used to select the topK models
    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', True) else None
    
    params = {}  

    kinds = ['user', 'item', 'user_topk', 'item_topk', ]
    clustering = ['kmeans', 'spectral', ]

    simMetrics = {k: [] for k in kinds}  # a list of PerformanceMetrics objects; [old] Metrics()
    simCV = {k: [] for k in kinds}
    base, method = 'cosine', 'sim'
    for fold in range(n_fold): 
        print('(t_neighborhood_ensemble) User-user similarity: Fold=%d | user-user based' % fold)

        R, T, L_train, L_test, U = uc.to_rating_matrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        n_users, n_items = R.shape[0], R.shape[1]

        ### memory-based method
        #   user-user 
        Ra = np.hstack((R, T))  # augmented rating matrix by combining R (from train split) and T (from test split)
        for kind in kinds[:2]: 

            Rc= uc.demean(Ra, kind=kind) # Rc: c, centered
            S = uc.evalSimilarity(Rc, kind=kind) # cosine similairty

            axis = 0 if kind == 'user' else 1
            assert S.shape[0] == Rc.shape[axis]
            print('(test) kind={0} | dim(S):{1} Sim:\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # full vs topk 
            Rh, Th = uc.predict(R, T, S=S, kind=kind, topk=None) # uc.predict_biased(T, S, kind=kind) # full similarity 
            
            method_specific = '{base}_{kind}_sim'.format(base=base, kind=kind)
            simMetrics[kind].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean))
            simCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )
        
            topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
            Rh_tokK, Th_topK = uc.predict(R, T, S=S, kind=kind, topk=topk) # uc.predict_topk(T, S, kind=kind, k=topk)  # S(top k)

            kind_topk = '%s_topk' % kind
            method_specific = '{base}_{kind}'.format(base=base, kind=kind_topk)           
            simMetrics[kind_topk].append(analyzePerf(L_test, Th_topK, method=method_specific, aggregate_func=np.mean))
            simCV[kind_topk].append( (L_test, uc.combiner(Th_topK, aggregate_func=np.mean)) )

    # show the combined performance from the CV
    # userMetrics.report(op=np.mean, message='Memory-based ensemble based on user similarity')
    # topKUserMetrics.report(op=np.mean, message='Memory-based ensemble based on top %d user similarity' % topk)
    
    # evalutation 
    for kind in kinds: 
        if kind in simCV: 
            method_specific = MFEnsemble.name_method(method='cosine', kind=kind, params=params)
            plot_roc(simCV[kind], file_name='roc-{method}-{project}'.format(method=method_specific, project=Domain))  # an import from evaluate
    perfMetrics.extend([PerformanceMetrics.consolidate(simMetrics[kind]) for kind in simMetrics.keys()])  # ~> a list of CV-consolidated PerformanceMetrics objects

    ### Pearson correlation (pcorr) with the true labels
    topk = 30
    # kinds = ['user', 'item', 'user_topk', 'item_topk', ]
    kinds = ['user', 'label', 'user_topk', 'label_topk', ]  # classifier prediction, prediction-label 
    corrMetrics = {k: [] for k in kinds}
    corrCV = {k: [] for k in kinds}
    base, method = 'corr', 'sim'
    for fold in range(n_fold): 
        R, T, L_train, L_test, U = uc.to_rating_matrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        # n_users, n_items = R.shape[0], R.shape[1]

        ### memory-based method
        #  user-user
        Ra = np.hstack((R, T))  # augmented rating matrix by combining R (from train split) and T (from test split)
        n_users, n_items = Ra.shape[0], Ra.shape[1]
        print('(test) n_users: %d, n_items_train: %d, n_items_total: %d' % (n_users, R.shape[1], Ra.shape[1]))

        # [note] no valid item-item predictions: y_score:[nan nan nan ... ] 
        for kind in kinds[:2]: # 'item'
            print('(t_neighborhood_ensemble) Pearson correlation similarity: Fold={0} | {1}-{1} correlation based'.format(fold, kind))
        
            # Use pcorr as weights to predict T 
            if kind.startswith('u'): 
                S = uc.evalCorrelation(Ra, kind=kind, epsilon=1e-9, to_distance=False)
                Rh, Th = uc.predict(R, T, S=S, kind=kind, topk=None) # uc.predictNewItemsByCorr(T, R, L_train)
            elif kind.startswith('l'):  # predictions vs true labels 
                Rh = None # undefined
                Th = uc.predictByCorrWithLabels(T, R, labels=L_train, topk=None)

            # [log] Input contains NaN, infinity or a value too large for dtype('float64')
            method_specific = '{base}_{kind}_sim'.format(base=base, kind=kind)
            corrMetrics[kind].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean))
            corrCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )
            # metrics = evalTestSet(L_test, L_pred, fold=fold)  # [log]
            # corrMetrics.add(metrics)

            # consider only top k most correlated classifiers' predictions and take their weighted average
            topk = int(math.floor(n_users/2)) # if kind.startswith('user') else int(math.floor(n_items/10))
            if kind.startswith('u'): 
                Rh_topK, Th_topK = uc.predict(R, T, S=S, kind=kind, topk=topk) # uc.predictNewItemsByCorr(T, R, L_train, topk=topk)
            elif kind.startswith('l'):
                Rh_topK = None # undefined
                Th_topK = uc.predictByCorrWithLabels(T, R, labels=L_train, topk=topk) 

            kind_topk = '%s_topk' % kind
            method_specific = '{base}_{kind}'.format(base=base, kind=kind_topk)
            corrMetrics[kind_topk].append(analyzePerf(L_test, Th_topK, method=method_specific, aggregate_func=np.mean))
            corrCV[kind_topk].append( (L_test, uc.combiner(Th_topK, aggregate_func=np.mean)) )
            # metrics = evalTestSet(L_test, L_pred, fold=fold)  # [log]
            # topKCorrMetrics.add(metrics)

    # corrMetrics.report(op=np.mean, message='Memory-based ensemble based on Pearson correlation as classifier weights.')
    # topKCorrMetrics.report(op=np.mean, message='Memory-based ensemble based on top %d Pearson correlation as classifier weights.' % topk)
    for kind in kinds: 
        if kind in corrCV: 
            method_specific = MFEnsemble.name_method(method='corr', kind=kind, params=params)
            plot_roc(corrCV[kind], file_name='roc-{method}-{project}'.format(method=method_specific, project=Domain))  # an import from evaluate

    perfMetrics.extend([PerformanceMetrics.consolidate(corrMetrics[kind]) for kind in corrMetrics.keys()])  # ~> a list of CV-consolidated perf

    ### instead of comparing with all the data points, find the subset of most similar data points to 
    # the target data points and use their similarity weights 
    # need to pick the k

    div(message='Todo: Other similarity metrics go here ... ', symbol='-', border=1)
    # for fold in range(n_fold): 
    #     R, T, L_train, L_test, U = uc.toRatingMatrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
    
    perfAll = PerformanceMetrics.merge(perfMetrics)  # merge all methods
    print('(test) n_methods: %d' % perfAll.n_methods()); assert perfAll.n_methods() == 8

    # so far, we have 2+2+4 = 8 methods
    method = 'sim' # off-the-shelf, ready-made
    ret['metrics'] = perfAll # PerformanceMetrics.consolidate(fullMetrics)  # foreach metric, take average over CV folds

    # ret['nmf_sim_top%d' % topk] = PerformanceMetrics.consolidate(topKMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    if base_perf is not None: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
    return perfAll

def als_cv_routine(fold, **kargs): 
    """
    
    **kargs
       n_factors 
       n_epochs 
       alpha
       mode

    """
    from utils_als import implicit_als_cg, implicit_als_johnson

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    
    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)
    params['alpha'] = alpha_val = kargs.get('alpha', MFEnsemble.alpha)
    
    # parameters for confidence matrix
    params['mode'] = kargs.get('mode', 'brier')  # confidence matrix

    Cui, R, T, L_train, L_test, U = uc.toConfidenceMatrix(fold, p_threshold=p_th, fill=missing_value, verbose=True, is_augmented=True, mode=params['mode'])
    n_users, n_items = Cui.shape[0], Cui.shape[1]

    # n_users, n_items = R.shape[0], R.shape[1]
    print('(als_cv_routine) Fold: %d, n_factors: %d | dim(Cui): %s, L_train: %d, n_test: %d' % (fold, params['n_factors'], str(Cui.shape), len(L_train), len(L_test)))

    conf_data = (Cui * params['alpha']).astype('double')
    n_nonzeros = sparse.csr_matrix.count_nonzero(conf_data)
    n_zeros = n_users * n_items - n_nonzeros
    print('... n_zeros: %d, n_nonzeros: %d, ratio: %f' % (n_zeros, n_nonzeros, n_zeros/(n_zeros+n_nonzeros+0.0)))  # [log] 104902189710

    ### change ALS algorithm here ### 
    P, Q = implicit_als_johnson(conf_data, R=np.hstack((R, T)), iterations=params['n_iter'], features=params['n_factors'])

    return (P, Q)

def wmf_ensemble(**kargs):

    # MFEnsemble
    def save_meta_tset(fold, params): # module: tset
        # MFEnsemble.meta_keys: ['mode', 'policy', 'supervised', 'masked']
        meta_params = {k: params[k] for k in MFEnsemble.meta_keys}  # meta_params are a subset of params that do not go into model selection loop
        tset_id = MFEnsemble.name_tset(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data
        
        # how to utilize preference scores? 
        # [note] tset_id provides a parameterized file ID that precedes {validation, prediction} in the training data file nameing
        #        tset: training set
        #        e.g. wmf_F10_A100_Xbrier_preference-validation-4.csv.gz     
        #                prefix = wmf_F10_A100_Xbrier_preference
        #                validation: combined 'test data' in the inner loop of the nested CV 
        #                4: fold count 
        #             general form: 
        #             <tset_id>_{validation, prediction}_<fold>.csv.gz
        print('(test) dim(Rh): {0}, dim(Th): {1}, n_train: {2}, n_test: {3}'.format(Rh.shape, Th.shape, L_train.size, L_test.size))
        uc.save_reconstructed_probs((Rh, Th), labels=(L_train, L_test), fold=fold, method=tset_id, verify=True, U=U)
        return

    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from utils_als import implicit_als_cg, implicit_als_johnson
    import utils_cf as uc
    import utils_als as ua

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    
    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)
    params['alpha'] = alpha_val = kargs.get('alpha', MFEnsemble.alpha)
    
    # parameters for confidence matrix
    params['mode'] = kargs.get('mode', 'brier')  # confidence matrix
    params['masked'] = kargs.get('masked', True) # mask FP and FN? this makes Cui 'sparse'
    params['mask_T'] = kargs.get('mask_T', True)

    params['topk'] = kargs.get('topk', 0.1)  # select top k entries of R (through Cui) to approximate, only relevant when policy in {'user', 'item'}
    params['supervised'] = kargs.get('supervised', False)
    # params['p_threshold'] = p_th  # 'ratio'-based confidence matrix depends on threshould
    # params['delta'] = 0

    # parameters for ALS methods 
    params['policy'] = kargs.get('policy', 'rating')  # options: {'rating', 'preference', 'label'}
    print('(wmf_ensemble) n_factors: {0}, alpha: {1} | mode: {2}, policy: {3}'.format(params['n_factors'], params['alpha'], 
        params['mode'], params['policy']))
    # ... say we use n_factors=20 and alpha=100 as default 

    ret = {}
    perfMetrics = []

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models

    # BP performance metrics are prefereable factored out of this routine, compute BP performance in _suite() routine instead
    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', False) else None
    # perfMetrics.append(base_perf)
    # ALS_Routine = kargs.get('als_routine', implicit_als_johnson)

    method = 'wmf'
    kinds = ['mean_aggregate', ]
    wmfMetrics = {k: [] for k in kinds}
    wmfCV = {k: [] for k in kinds}

    Pe = Qe = 0.
    for fold in range(n_fold): 
        
        # 
        Cui, Cui_bar, R, T, L_train, L_test, U = uc.evalConfidenceMatrix(fold, topk=params['topk'], 
                policy=params['policy'], # determines the subroutine for computing Cui
                    supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                        mode=params['mode'], alpha=params['alpha'], masked=params['masked']) 

        n_users, n_items = Cui.shape[0], Cui.shape[1]
        # assert Cui.shape == Cui_bar.shape

        # n_users, n_items = R.shape[0], R.shape[1]
        print('(wmf_ensemble) Fold: %d, n_factors: %d | dim(Cui): %s, L_train: %d, n_test: %d' % \
            (fold, params['n_factors'], str(Cui.shape), len(L_train), len(L_test)))

        # note: do the scaling from within the call of toConfidenceMatrix
        # conf_data = (C * params['alpha']).astype('double')

        # n_nonzeros = sparse.csr_matrix.count_nonzero(Cui)
        # n_zeros = n_users * n_items - n_nonzeros
        # assert n_zeros > 0
        # print('... n_zeros: %d, n_nonzeros: %d, ratio: %f' % (n_zeros, n_nonzeros, n_zeros/(n_zeros+n_nonzeros+0.0)))  # [log] 104902189710
        # n_nonzeros = sparse.csr_matrix.count_nonzero(Cui_bar)
        # n_zeros2 = n_users * n_items - n_nonzeros
        # assert n_zeros + n_zeros2 == n_users * n_items, "n_zeros(Cui + Cui_bar): %d =?= total: %d" % (n_zeros+n_zeros2, n_users * n_items)

        ### Very ALS algorithms here ###

        # conf_matrix must be alpha * R for this call (instead of 1 + alpha * R) => minimal confidence at 0
        if not params['policy'].startswith('t'):
            P, Q = implicit_als_johnson(Cui, ratings=np.hstack((R, T)), labels=np.hstack((L_train, L_test)),
                        iterations=params['n_iter'], features=params['n_factors'], 
                            policy=params['policy'])
        else: # tradeoff between R and L
            # Cui, Cui_bar are not in sparse format
            P, Q = implicit_als_johnson(Cui, label_confidence=Cui_bar, ratings=np.hstack((R, T)), labels=np.hstack((L_train, L_test)),
                            iterations=params['n_iter'], features=params['n_factors'], 
                                policy=params['policy'])
            
        # P, Q = implicit_als_cg(conf_data, R=R, iterations=params['n_iter'], features=params['n_factors']) 

        ##################################

        # P = P.todense()
        # Q = Q.todense()
        Pe += P
        Qe += Q
        print('(wmf_ensemble) Fold: {0} | dim(P): {1}, dim(Q): {2} | type: {3}'.format(fold, P.shape, Q.shape, type(P)) )

        # but we only need the Q from the test split ... 
        # set canonicalize=True because the reconstructed probabilities could be not in [0.0, 1.0]
        if params['policy'].startswith('pref'):
            Rh, Th = uc.predict_by_preference(P, Q, len(L_train), test_set_only=False, canonicalize=True) 
            # Rh, Th are preference scores
        else: 
            Rh, Th = ua.predict_by_factors(P, Q, len(L_train), test_set_only=False, canonicalize=True) # Rh <- None if test_set_only: True
        
        # prediction via preference {0, 1}
        # ua.predict_by_preference(P, Q, len(L_train), test_set_only=False, canonicalize=True)
        
        for kind in kinds:
            if kargs.get('save', True): save_meta_tset(fold=fold, params=params)
                
            # metrics = comparePerfMetrics0(T, L_test, Th=Th, R=None, L_train=None)
            method_specific = MFEnsemble.name_method(method, kind, params=params)  # base method + specific kind
            
            # evaluation
            aggregate_func = 'preference' if params['policy'].startswith('pref') else 'mean'
            wmfMetrics[kind].append( analyzePerf(L_test, Th, method=method_specific, aggregate_func=aggregate_func, T=T, fold=fold) ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
            wmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=aggregate_func, T=T)) )
    ### end CV loop 
    
    Pe = Pe/n_fold
    Qe = Qe/n_fold
    if kargs.get('save_factors', False): save_factors(Pe, Qe) # use these to compute the similarity matrix

    ## evaluation 
    for kind in kinds: 
        if kind in wmfCV: 
            method_specific = MFEnsemble.name_method(method, kind, params=params) 
            plot_roc(wmfCV[kind], file_name='roc-{method}-{project}'.format(method=method_specific, project=Domain))  # an import from evaluate

    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(wmfMetrics[kind]) for kind in kinds])

    # ret: output dictionary
    #      key: {methods}, {metrics}
    ret['metrics'] = perfAll
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    if base_perf is not None: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return perfAll  # CV-fold combined PerformanceMetrics instance

def wmf_similarity_ensemble(**kargs):
    """
    Use NMF factorized latent matrices to compute similarity
    """ 
    def convert(P, Q):
        if not isinstance(P, np.ndarray):
            P = np.array(P.todense())
        if not isinstance(Q, np.ndarray):
            Q = np.array(Q.todense()) 
        return P, Q
    def save_factors(P, Q, fold=-1, method='wmf', cols_users=[], cols_items=[]):
        if not fold in (-1, 1): return # do nothing

        if len(cols_users) == 0: cols_users = U # U in function closure 
        MFEnsemble.save_factors(P, cols=cols_users, file_name='{method}_P.csv'.format(method=method.upper()))
        MFEnsemble.save_factors(Q, cols_items=cols_items, file_name='{method}_Q.csv'.format(method=method.upper())) 
        return
    def save_array(S, kind, fold, method='wmf'): 
        if fold != 1: return # do nothing
        MFEnsemble.save_array(S, file_name='{method}_{kind}_S.csv'.format(method=method, kind=kind))
        
    import math
    from utils_als import implicit_als_cg, implicit_als_johnson
    from evaluate import plot_roc, analyzePerf, Metrics, PerformanceMetrics
    import utils_cf as uc
    from itertools import product

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 

    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)
    params['alpha'] = alpha_val = kargs.get('alpha', MFEnsemble.alpha)

    # parameters for confidence matrix 
    params['mode'] = kargs.get('mode', 'brier') # {'ratio', }
    print('(wmf_similarity_ensemble) Confidence matrix based on mode={0}, alpha={1}'.format(params['mode'], params['alpha']))

    topk = 30

    # output
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', False) else None

    # bpMetrics, fullMetrics, topKMetrics = Metrics(), Metrics(), Metrics() # matrix factorization metrics
    perfMetrics = []
    fullMetrics, topKMetrics = [], []
    wmfCV, topKWMFCV = [], []

    clusterings = ['kmeans', 'spectral']  # product(*[kind[:2], clusterings])
    kinds = ['user', 'item', 'user_topk', 'item_topk', ]
    if kargs.get('run_clustering', False): 
        kinds += ['_'.join(pair) for pair in product(*[kinds[:2], clusterings])]
    # kinds = ['user', 'item', 'user_topk', 'item_topk', 'user_kmeans', 'item_kmeans', 'user_spectral', 'item_spectral', ]

    wmfMetrics = {k: [] for k in kinds}  # a list of PerformanceMetrics objects; [old] Metrics()
    wmfCV = {k: [] for k in kinds}

    method = 'wmf' # 'wmf_sim'
    Pe = Qe = 0.
    for fold in range(n_fold): 
     
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        Cui, R, T, L_train, L_test, U = uc.toConfidenceMatrix(fold, p_threshold=p_th, 
                                            fill=missing_value, verbose=True, is_augmented=True, mode=params['mode'])
        n_users, n_items, n_items_total = Cui.shape[0], Cui.shape[1], len(L_train)+len(L_test)
        print('(wmf_similarity_ensemble) dim(Cui/R):{0}, dim(T): {1}, n_train: {2}, n_test: {3}'.format(
            Cui.shape, T.shape, len(L_train), len(L_test)))

        conf_data = (Cui * params['alpha']).astype('double')
        
        P, Q = implicit_als_johnson(conf_data, R=np.hstack((R, T)), iterations=params['n_iter'], features=params['n_factors'])
        # P, Q = implicit_als_cg(conf_data, iterations=params['n_iter'], features=params['n_factors']) 
        
        if not isinstance(P, np.ndarray):
            P = np.array(P.todense())
        if not isinstance(Q, np.ndarray):
            Q = np.array(Q.todense())
        # save latent factors?
        # save_factors(P, Q, fold)
        Pe += P
        Qe += Q

        # given latent factors
        for kind in kinds[:2]:  # foreach user or item
            factors = P if kind == 'user' else Q
            S = uc.evalSimilarityByLatentFeatures(factors, epsilon=1e-9)
            save_array(S, kind, fold)
            
            # axis = 0 if kind == 'user' else 1
            dimS = n_users if kind == 'user' else n_items_total
            assert S.shape[0] == S.shape[1] == dimS
            print('(test) kind={0} | dim(S): {1} (S[i,j] in [0, 1]?):\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # [note] R, T are to be merged prior to calling predict_nobias or predict_topk
            Rh, Th = uc.predict(R, T, S=S, kind=kind)
            assert T.shape == Th.shape, "dim(T): {0} but dim(Th): {1}".format(T.shape, Th.shape)

            method_specific = MFEnsemble.name_sim_method(method, kind=kind, params=params) # '{base}_{kind}_sim'.format(base=base, kind=kind)
            wmfMetrics[kind].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean))
            wmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )  # plot K-Fold CV
            if kargs.get('save', False): uc.save_reconstructed_probs((Rh, Th), labels=(L_train, L_test), fold=fold, method=method, verify=True, U=U)

            # use top K only 
            topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
            Rh_topK, Th_topK = uc.predict(R, T, S=S, kind=kind, topk=topk)
        
            kind_topk = '%s_topk' % kind
            method_specific = MFEnsemble.name_sim_method(method, kind=kind_topk, params=params) # '{base}_{kind}'.format(base=base, kind=kind_topk)
            wmfMetrics[kind_topk].append(analyzePerf(L_test, Th_topK, method=method_specific, aggregate_func=np.mean) )
            wmfCV[kind_topk].append( (L_test, uc.combiner(Th_topK, aggregate_func=np.mean)) )
            
            ## WMF + Clustering
            if kargs.get('run_clustering', False):
                for clustering in clusterings:  # foreach clustering method
                    kind_cluster = '%s_%s' % (kind, clustering)
                    method_specific = MFEnsemble.name_sim_method(method, kind=kind_cluster, params=params)

                    # factors could be user-based or item-based (depending on kind)
                    cluster_labels = uc.runClustering(factors, n_clusters=params['n_factors'], method=clustering) 
                    Rh, Th = uc.predict_by_cluster(R, T, similarity=S, kind=kind, C=cluster_labels)

                    wmfMetrics[kind_cluster].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean) )
                    wmfCV[kind_cluster].append( (L_test, uc.combiner(Th, aggregate_func=np.mean))) 
                    if kargs.get('save', False): uc.save_reconstructed_probs((Rh, Th), labels=(L_train, L_test), fold=fold, method=method, verify=True, U=U)

        ### end user-item loop
    
    ### end foreach CV fold 
    Pe = Pe/n_fold
    Qe = Qe/n_fold
    if kargs.get('save_factors', False): save_factors(Pe, Qe)

    ## evaluation
    for kind in kinds: 
        # wmfCV[kind] is a sequence of (lable, prediction)
        if kind in wmfCV: 
            method_specific = MFEnsemble.name_sim_method(method, kind, params=params)

            # plot only top 5
            plot_roc(wmfCV[kind], file_name='roc-{method}-{project}'.format(method=method_specific, project=Domain))  # an import from evaluate

    ## Compare with baseline methods
    
    # perf_unfold = [PerformanceMetrics.consolidate(nmfMetrics[kind]) for kind in kinds]
    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(wmfMetrics[kind]) for kind in kinds])
    ret['metrics'] = perfAll # PerformanceMetrics.consolidate(fullMetrics)  # foreach metric, take average over CV folds

    # ret['nmf_sim_top%d' % topk] = PerformanceMetrics.consolidate(topKMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    if base_perf is not None: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return perfAll

def wmf_clustering(**kargs):
    """
    Use NMF factorized latent matrices to compute similarity
    """ 
    def convert(P, Q):
        if not isinstance(P, np.ndarray):
            P = np.array(P.todense())
        if not isinstance(Q, np.ndarray):
            Q = np.array(Q.todense()) 
        return P, Q
        
    import math
    from utils_als import implicit_als_cg, implicit_als_johnson
    from evaluate import plot_roc, analyzePerf, Metrics, PerformanceMetrics
    import utils_cf as uc
    from itertools import product

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 

    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)
    params['alpha'] = alpha_val = kargs.get('alpha', MFEnsemble.alpha)

    # parameters for confidence matrix 
    params['mode'] = kargs.get('mode', 'brier') # {'ratio', }

    topk = 30

    # output
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', True) else None

    # bpMetrics, fullMetrics, topKMetrics = Metrics(), Metrics(), Metrics() # matrix factorization metrics
    perfMetrics = []
    fullMetrics, topKMetrics = [], []
    wmfCV, topKWMFCV = [], []

    clusterings = ['kmeans', 'spectral']  # product(*[kind[:2], clusterings])
    kinds = ['user', 'item', 'user_topk', 'item_topk', ]
    if kargs.get('run_clustering', False): 
        kinds += ['_'.join(pair) for pair in product(*[kinds[:2], clusterings])]
    # kinds = ['user', 'item', 'user_topk', 'item_topk', 'user_kmeans', 'item_kmeans', 'user_spectral', 'item_spectral', ]

    wmfMetrics = {k: [] for k in kinds}  # a list of PerformanceMetrics objects; [old] Metrics()
    wmfCV = {k: [] for k in kinds}

    method = 'wmf' # 'wmf_sim'
    for fold in range(n_fold): 
     
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        Cui, R, T, L_train, L_test, U = uc.toConfidenceMatrix(fold, p_threshold=p_th, 
                                            fill=missing_value, verbose=True, is_augmented=True, mode=params['mode'])
        n_users, n_items, n_items_total = Cui.shape[0], Cui.shape[1], len(L_train)+len(L_test)
        print('(wmf_similarity_ensemble) dim(Cui/R):{0}, dim(T): {1}, n_train: {2}, n_test: {3}'.format(
            Cui.shape, T.shape, len(L_train), len(L_test)))

        conf_data = (Cui * params['alpha']).astype('double')
        
        P, Q = implicit_als_johnson(conf_data, R=np.hstack((R, T)), iterations=params['n_iter'], features=params['n_factors'])
        # P, Q = implicit_als_cg(conf_data, iterations=params['n_iter'], features=params['n_factors']) 
        
        if not isinstance(P, np.ndarray):
            P = np.array(P.todense())
        if not isinstance(Q, np.ndarray):
            Q = np.array(Q.todense())

        # save latent factors?

        # given latent factors
        for kind in kinds[:2]:  # foreach user or item
            factors = P if kind == 'user' else Q
            S = uc.evalSimilarityByLatentFeatures(factors, epsilon=1e-9)
            
            # axis = 0 if kind == 'user' else 1
            dimS = n_users if kind == 'user' else n_items_total
            assert S.shape[0] == S.shape[1] == dimS
            print('(test) kind={0} | dim(S): {1} (S[i,j] in [0, 1]?):\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # [note] R, T are to be merged prior to calling predict_nobias or predict_topk
            Rh, Th = uc.predict(R, T, S=S, kind=kind)
            assert T.shape == Th.shape, "dim(T): {0} but dim(Th): {1}".format(T.shape, Th.shape)

            # use top K only 
            topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
            Rh_topK, Th_topK = uc.predict(R, T, S=S, kind=kind, topk=topk)
            
            ## WMF + Clustering
            if kargs.get('run_clustering', False):
                for clustering in clusterings:  # foreach clustering method
                    kind_cluster = '%s_%s' % (kind, clustering)
                    method_specific = MFEnsemble.name_sim_method(method, kind=kind_cluster, params=params)

                    # factors could be user-based or item-based (depending on kind)
                    cluster_labels = uc.runClustering(factors, n_clusters=params['n_factors'], method=clustering) 
                    Rh, Th = uc.predict_by_cluster(R, T, similarity=S, kind=kind, C=cluster_labels)

                    wmfMetrics[kind_cluster].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean) )
                    wmfCV[kind_cluster].append( (L_test, uc.combiner(Th, aggregate_func=np.mean))) 
                    if kargs.get('save', True): uc.save_reconstructed_training_data(Rh, L_train, fold, method=method_specific, verify=True, U=U)

        ### end user-item loop

    ### end foreach CV fold 
    return perfAll

def nmf_ensemble_suite(**kargs): 

    # basic matrix factorization using gradient descent
    # div(message='Running model-based approach (ideally, masking FPs and FNs) ...', symbol='#', border=1) 
    # mfb_ensemble()   # sgd, batch GD

    perfMetrics = []
    # base predictors 
    if kargs.get('run_bp', True):  
        topk_bps = kargs.get('topk_bps', -1)
        base_perf = base_predictors(topk=topk_bps, metric='fmax')
        perfMetrics.append(base_perf)

    ## how does it compare to stacking methods?
    if kargs.pop('run_bp_stacker', True):
        topk_bp_stackers = kargs.get('topk_bp_stackers', -1)
        perf_stacker = run_baseline_stacker(topk=topk_bp_stackers, metric='fmax')  # dataset='bp'
        perfMetrics.append(perf_stacker)

    ### matrix factorization methods based on Surprise 
    if kargs.get('run_nmf_ensemble', True): 
        div(message='Running NMF approach ...', symbol='#', border=1)
        pm1 = nmf_ensemble(base_perf, **kargs)  # metrics -> perf, fmax -> sorted methods
        perfMetrics.append(pm1)

    if kargs.get('run_nmf_stacker', True): 
        div(message='Running Stackers on top of NMF-reproduced trainining data ...', symbol='#', border=1)
        perf_mf_stacker = run_stacker(dataset='nmf')
        perfMetrics.append(perf_mf_stacker)

    if kargs.get('run_nmf_similarity', True): 
        div(message='Running NMF-induced similarity ensemble ...', symbol='#', border=1)
        pm2 = nmf_similarity_ensemble(base_perf, **kargs)
        perfMetrics.append(pm2)

    perfAll = PerformanceMetrics.merge(perfMetrics)
    # print('(mf_ensemble_suite) how many methods in total? %d' % perfAll.n_methods())

    post_analysis(perfAll, context='nmf_ensemble_suite', highlight=['nmf', ])

    return perfAll

def wmf_ensemble_suite(**kargs):
    perfMetrics = []

    # base predictors 
    if kargs.get('run_bp', True):  
        base_perf = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
        perfMetrics.append(base_perf)

    ## how does it compare to stacking methods?
    if kargs.pop('run_bp_stacker', True):
        perf_stacker = base_stackers(topk=kargs.get('topk_bp_stackers', -1), metric='fmax')
        perfMetrics.append(perf_stacker)
    
    # try different variations
    # param_grid = kargs.get('param_grid', {'n_factors':[10, 20], 'alpha':[100, ]})
    
    # very hyperparameters 
    if kargs.get('run_wmf_ensemble', True): 
        perf_wmf = wmf_ensemble(**kargs)
        perfMetrics.append(perf_wmf)

    # stacker on top of wmf
    if kargs.get('run_wmf_stacker', False): 
        perf_mf_stacker = run_stacker(dataset='wmf')
        perfMetrics.append(perf_mf_stacker)

    ## WMF-derived neighborhood methods
    if kargs.get('run_wmf_similarity', False): 
        perf_wmf_sim = wmf_similarity_ensemble(**kargs)
        perfMetrics.append(perf_wmf_sim)

    perfAll = PerformanceMetrics.merge(perfMetrics)
    # print('(wmf_ensemble_suite) how many methods in total? %d' % perfAll.n_methods())

    post_analysis(perfAll, context=kargs.get('context', 'wmf_ensemble_suite'), highlight=['wmf', ])

    return perfAll

def wmf_ensemble_suite_multimodel(**kargs):
    """
    
    Replace wmf_ensemble_suite(**kargs) when attempting multiple parameter settings (assuming that 
    multiple settings all potentially lead to good ensemble & analytics solutions)

    Memo
    ----
    1. to be used directly in suite() but considering multiple parameter settings 
    """
    import evaluate 
    from sklearn.model_selection import ParameterGrid
    from cf_spec import MFEnsemble

    div(message='(wmf_ensemble_suite_multimodel) Multimodel mode (project: %s) ...' % ProjectPath, symbol='#', border=1)

    perfMetrics = []

    # base predictors 
    # if kargs.get('run_bp', True):  
    #     topk_bps = kargs.get('topk_bps', -1)
    #     base_perf = base_predictors(topk=topk_bps, metric='fmax')
    #     perfMetrics.append(base_perf)

    ## weighted matrix factorization
    param_grid = kargs.get('param_grid', {'n_factors':[10, 20, ], 'alpha':[100, ]})

    # very hyperparameters 
    n_target_methods = 0
    for seti, params in enumerate(list(ParameterGrid(param_grid))):  # a list of dictionaries containing target (hyper)parameters
        perfSuite = wmf_ensemble_suite(topk_bps=3, topk_bp_stackers=3, 
                        n_factors=params.get('n_factors', 10), alpha=params.get('alpha', 100))
        perfMetrics.append(perfSuite)
        div('Param set #{0}: {1} completed --'.format(seti+1, params), symbol='#', border=2)
        n_target_methods += 1

    perfAll = PerformanceMetrics.merge(perfMetrics)
    print('(wmf_ensemble_suite_multimodel) how many methods in total {0}'.format(n_target_methods))

    post_analysis(perfAll, context=kargs.get('context', 'wmf_multimodel'), highlight=['wmf', ], metrics=['fmax', 'auc', ])

    return perfAll

def wmf_ensemble_modelselect(**kargs):
    """

    Related
    -------
    1. test_wmb_probs(): this test suite attempts to find out if the reconstructed probabilities via WMF can help label predictions


    """
    import evaluate 
    from sklearn.model_selection import ParameterGrid

    div(message='(wmf_ensemble_modelselect) Model selection (project: %s) ...' % ProjectPath, symbol='#', border=1)

    perfMetrics = []

    base_perf = kargs.get('base_perf', None) 
    
    # base predictors 
    if kargs.get('run_bp', True):  
        topk_bps = kargs.get('topk_bps', -1)
        base_perf = base_predictors(topk=topk_bps, metric='fmax')
        perfMetrics.append(base_perf)

    ## how does it compare to stacking methods?
    ## how does it compare to stacking methods?
    if kargs.pop('run_bp_stacker', False):
        perf_stacker = run_stacker()
        perfMetrics.append(perf_stacker)

    ## weighted matrix factorization

    # default parameters
    n_factors = 20
    n_epochs = 30
    alpha = 100
    param_grid = kargs.get('param_grid', {'n_factors':[5, 10, 20, 30], 'alpha':[10, 50, 100, 200, ]})

    # very hyperparameters 
    n_target_methods = 0
    for params in list(ParameterGrid(param_grid)):  # a list of dictionaries containing target (hyper)parameters
        perfSuite = wmf_ensemble_suite(n_factors=params.get('n_factors', n_factors), alpha=params.get('alpha', alpha))
        perfMetrics.append(perfSuite)

        n_target_methods +=1
        div('Param set #{0}: {1} completed --'.format(n_target_methods, params), symbol='#', border=2)

    perfAll = PerformanceMetrics.merge(perfMetrics)
    print('(wmf_ensemble_suite_multimodel) how many methods in total? {0}'.format(n_target_methods))

    post_analysis(perfAll, context=kargs.get('context', 'wmf_model_selection'), highlight=['wmf', ], metrics=['fmax', 'auc', ])

    return perfAll 

def suite(**kargs):
    import evaluate
    # a set of methods aggregated to be compared with each other 

    ### foundataion: recommender system 
    div(message='(suite) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    perfMetrics = []

    # t_recommender()
    topk_bps = kargs.get('topk_bps', -1)
    base_perf = base_predictors(topk=topk_bps, metric='fmax')
    perfMetrics.append(base_perf)

    ## how does it compare to stacking methods?
    topk_bp_stackers = kargs.get('topk_bp_stackers', -1)
    perf_stacker = base_stackers(topk=topk_bp_stackers, metric='fmax')
    perfMetrics.append(perf_stacker)

    ## Model-based ensemble learning, Basics MF-based methods (compare this with more advanced methods e.g. weighted MF)
    perf_mf = nmf_ensemble_suite()  # (basic) model-based
    perfMetrics.append(perf_mf)    
    
    ## Neighborhood ensemble, memory-based ensemble
    perf_neighborhood = t_neighborhood_ensemble()  # memory-based
    perfMetrics.append(perf_neighborhood)

    ### Weighted MF via ALS
    # perf_wmf = wmf_ensemble_suite(base_perf, n_factors=params.get('n_factors', 10), alpha=params.get('alpha', 100))
    
    default_grid = {'n_factors':[10, ], 'alpha':[50, 100, 500, ]}
    perf_wmf = wmf_ensemble_suite_multimodel(param_grid=default_grid, plot=True)  # wrappr of wmf_ensemble_suite()
    perfMetrics.append(perf_wmf)

    perfGrand = PerformanceMetrics.merge(perfMetrics)
    div(message='(suite) How many methods in total? %d' % perfGrand.n_methods())

    # 'performance_metrics-{kind}-{domain}'.format(kind='suite', domain=Domain)
    perfGrand.save(file_name=perfGrand.my_shortname(context='suite', domain=Domain))  # only saves the table for now ... 02.05.19

    # total ranking
    indent_level = 2
    greater_is_better = True
    category = 'suite'
    for metric in ['fmax', 'auc', 'fmax_negative', ]: 

        # sorted_pairs = PerformanceMetrics.sort2(perfGrand, metric=metric, verbose=True, sorted_pairs=True)
        sorted_pairs = perfGrand.sort(metric=metric, reverse=greater_is_better, verbose=False)
        
        # target_set: if specified, methods that match the keywords will be highlighted
        s = evaluate.format_ranked_list(sorted_pairs, metric=metric, topk=None, verbose=False, highlight=['wmf', 'nmf', ])
        print(s.rjust(len(s)+indent_level, ' '))

        # plot 
        file_name = '{method}_{metric}_comparison-N{size}-D{domain}'.format(method=category, 
            metric=metric, size=perfGrand.n_methods(), domain=Domain)
        evaluate.plot_performance(perfGrand, metric=metric, ascending=True, domain=Domain, file_name=file_name)

    return perfGrand

def post_analysis(perfMetrics, context='post_analysis', highlight=['wmf', 'nmf', ], metrics=[]): 
    # after collecting all the performance metrics, make comparisons and analyses

    if not perfMetrics: 
        # do nothing 
        return
    # precondition: after all the performance metrics are collected for all objects 
    # operations: merge, save, sort, plot
    
    if isinstance(perfMetrics, list): 
        perfGrand = PerformanceMetrics.merge(perfMetrics)
    else: 
        perfGrand = perfMetrics
    div(message='(post_analysis) How many methods in total? %d' % perfGrand.n_methods())

    # 'performance_metrics-{kind}-{domain}'.format(kind='suite', domain=Domain)
    perfGrand.save(file_name=perfGrand.my_shortname(context=context, domain=Domain))  # only saves the table for now ... 02.05.19

    # total ranking
    indent_level = 0
    greater_is_better = True
    category = context

    if not metrics: metrics = ['fmax', 'auc', 'fmax_negative', ]
    for metric in metrics: 

        # sorted_pairs = PerformanceMetrics.sort2(perfGrand, metric=metric, verbose=True, sorted_pairs=True)
        sorted_pairs = perfGrand.sort(metric=metric, reverse=greater_is_better, verbose=False)
        
        # target_set: if specified, methods that match the keywords will be highlighted
        div(evaluate.format_ranked_list2(sorted_pairs, metric=metric, topk=None, verbose=False, highlight=highlight), symbol='=', border=2)
        # print(s.rjust(len(s)+indent_level, ' '))

        # plot 
        file_name = '{method}_{metric}_comparison-N{size}-D{domain}'.format(method=category, 
            metric=metric, size=perfGrand.n_methods(), domain=Domain)
        evaluate.plot_performance(perfGrand, metric=metric, ascending=True, domain=Domain, file_name=file_name)
    return perfGrand

def plot(**kargs):
    import evaluate
    from cf_spec import MFEnsemble

    # e.g. ./data/performance_metrics-model-select-pf2.csv
    input_file = kargs.get('file_name', 'performance_metrics-model-select-pf2.csv')

    perf = PerformanceMetrics(load=True, file_name=input_file)  # note: will throw exception if file doesn't exist
    print('(test) perf:\n%s\n ... index:\n%s\n' % (perf.table.head(5), perf.table.index))

    print('(plot) Plotting performance metrics with %d methods' % perf.n_methods())
    category = kargs.get('method', 'wmf')
    greater_is_better = True
    for metric in ['fmax', 'auc', 'fmax_negative', ]:

        s = evaluate.format_ranked_list2(perf.sort(metric=metric, reverse=greater_is_better, verbose=False), 
                metric=metric, topk=50, verbose=False, highlight=['stacker', ], inverse_highlight=['wmf', 'nmf', ]) # highlight=['wmf', 'nmf', ]
        # print(s.rjust(len(s)+indent_level, ' '))
        print(s)

        file_name = MFEnsemble.name_performance_plot(method=category, metric=metric, 
            size=perf.n_methods(), aspect='comparison', domain=Domain)
        evaluate.plot_performance(perf, metric=metric, ascending=True, domain=Domain, file_name=file_name)

    return

def test_stacker(**kargs):
    div(message='(test) Comparison of stackers (project: %s)' % ProjectPath, symbol='#', border=1) # stackers on MF-reproduced probabilities vs stackers on BP predictions
    exact_match = kargs.get('exact', True)
    
    # perf = PerformanceMetrics()
    perfMetrics = []
    if exact_match: 
        for dataset in kargs.get('datasets', ['wmf', 'nmf', ]):  # these are the name of the training data 
            # 1. dataset is named via the MF/CF method and its parameter setting
            # 2. if pref_aug is set to True, then don't do stacking on top of preference-augmented dataset
            # 3. set 'evaluation' to False to disable post_analysis(); in doing so, we get the performance metrics as a return value 
            #    then later on, pass this performance metrics to the caller in order to compare with other methods
            perf = run_stacker_suite(context='{mf_method}_stackers'.format(mf_method=dataset), method=dataset, exact=True, 
                        run_bp_stacker=True, 
                        topk_bps=3, topk_bp_stackers=3,  # only comparing with the top 3 baseline methods 
                        parallelize=True, reconstructed_testset=True, 
                        pref_aug=kargs.get('pref_aug', True), evaluation=kargs.get('evaluation', False))  
            perfMetrics.append(perf)
    else: # user-based methods vs item-based methods
        # set 'evaluation' to True to run analysis function (e.g. post_analysis())
        for kind in ['user', 'item', ]: # separate user models and item models
            perf_wmf = run_stacker_suite(context='wmf_{kind}_stackers'.format(kind=kind), run_bp_stacker=True, parallelize=True, 
                            method='wmf', keywords=[kind, ], evaluation=kargs.get('evaluation', False)) 
            perf_nmf = run_stacker_suite(context='nmf_{kind}_stackers'.format(kind=kind), run_bp_stacker=True, parallelize=True, 
                            method='nmf', keywords=[kind, ], evaluation=kargs.get('evaluation', False)) 
            perfMetrics.extend([perf_wmf, perf_nmf])
    
    return PerformanceMetrics.merge(perfMetrics)

def test_wmf_probs(**kargs): 
    import evaluate 
    from evaluate import PerformanceMetrics
    from sklearn.model_selection import ParameterGrid
    div(message="(test_wmf_probs) Do the reconstructed probabilities have higher predictive strength? (domain: {0})".format(Domain), symbol='#', border=1)

    np.set_printoptions(precision=3)

    target_metric = kargs.get('metric', 'fmax')
    topk_bps = 3  # only show top K BPs on the performance chart
    topk_bp_stackers = 3

    perfMetrics = []

    # set reverse to True if target_metric is of the-greater-the-better kind (e.g. auc)
    ### base predictors
    perfBase = base_predictors(topk=topk_bps, metric='fmax')
    nBP = perfBase.n_methods()
    assert nBP > 0 and nBP <= topk_bps, "Found %d base predictors while requesting %d" % (perfBase.n_methods(), topk_bps)
    perfMetrics.append(perfBase)

    ### baseline stackers
    perfStacker = base_stackers(topk=topk_bp_stackers, metric='fmax', parallelize=True)
    perfMetrics.append(perfStacker)

    ### weighted matrix factorization

    # default parameters
    n_epochs = 30
    tTrainModel = True

    ### Section I: Model selection

    # parameter settings
    ## a. fix (nf, alpha), vary mode
    #     {'n_factors':[10, ], 'alpha':[100, ], 'mode': ['brier', 'ratio']}
    ## b. fix mode, vary (nf, alpha)

    # meta_params do not go into the model selection loop for the following reasons: 
    #     1) if included, there'll be too many parameters to tune 
    #     2) these parameters may not play a key role in the performance or other research questions we care about
    #     3) we wish to analyze these parameter settings separately and independently
    meta_params = {'mode': 'brier', 
                   'policy': 'item-match',  # I. {'rating', 'preference', 'label', 'tradeoff'} | II. {'user', 'item', 'item-match'}
                   'topk': -1,   # relvant to policy group II; -1 to use class prior proportion
                   'supervised': True,  # policy group II
                   'mask_T': False,  # policy group II: item, item-match
                   'masked': True, 
                   'pref_aug': True}  # options: {'rating', 'preference', 'label', 'tradeoff'}
    param_grid = kargs.get('param_grid', {'n_factors':[10, ], 'alpha':[100, ]})
    
    # very hyperparameters 
    if tTrainModel:
        n_target_methods = 0
        for params in list(ParameterGrid(param_grid)):  # a list of dictionaries containing target (hyper)parameters
            perf = wmf_ensemble_suite(run_bp=False, run_bp_stacker=False,  
                        run_wmf_ensemble=True, run_wmf_stacker=False, run_wmf_similarity=False, 
                            n_factors=params['n_factors'], alpha=params['alpha'], 
                            mode=meta_params['mode'], 
                            supervised=meta_params['supervised'], 
                            # conf_user=meta_params.get('conf_user', True), 
                            mask_T=meta_params['mask_T'], 
                            policy=meta_params['policy'], masked=meta_params['masked'], topk=meta_params['topk'])
            perfMetrics.append(perf)
            
            n_target_methods+=1
            div('(test_wmf_probs) Param set #{0}: {1} completed --'.format(n_target_methods, params), symbol='#', border=2)

        perfAll = PerformanceMetrics.merge(perfMetrics)
        n_baseline = perfAll.n_methods()-n_target_methods
        print('(test_wmf_probs) how many WMF methods in total? {0} vs n_baseline: {1}'.format(n_target_methods, n_baseline))

        post_analysis(perfAll, context='test_wmf_probs_modeling', highlight=['wmf', ], metrics=['fmax', 'auc', ])

    ## Setting II. After obtaining the MF-produced training data

    perfMetrics = [perfBase, perfStacker, ]
    ## now test the quality using stacking methodology 
    for params in list(ParameterGrid(param_grid)): 
        tset_id = MFEnsemble.name_tset(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data
        print('(test) tset_id: %s' % tset_id)
        perf = test_stacker(context='test_wmf_probs_via_stackers-{0}'.format(tset_id), 
                                datasets=[tset_id, ], exact=True, pref_aug=meta_params['pref_aug'])
        perfMetrics.append(perf)
        
    # test via simple aggregation method
    for params in list(ParameterGrid(param_grid)): 
        tset_id = MFEnsemble.name_tset(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data    
        
        # note: can use this routine to aggregate preference scores as well
        perf = test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(tset_id), datasets=[tset_id, ], method='mean') # params: aggregate_func, pref_aug, test  
        perfMetrics.append(perf)

    # Alternatively, put parameter grid iteration inside the test_<ensemble method> function
    # test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(tset_id), 
    #     param_grid=param_grid, meta=meta, method='mean', target_method='wmf') # method refers to the combining method, specifiy specific MF method via target method
    post_analysis(PerformanceMetrics.merge(perfMetrics), context='test_wmf_probs_stacker_combiner', highlight=['wmf', ], metrics=['fmax', 'auc', ])
 
    return

def test_nmf_vs_wmf(**kargs):
    perfMetrics = []

    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        perfWMF = wmf_ensemble_suite(n_factors=nf, save=True, 
                    run_stacker=False, run_wmf_ensemble=True, run_wmf_stacker=False, run_wmf_similarity=False, 
                        run_clustering=False, context='wmf_ensemble') # set save to True to save R' and T' (reproduced probabilities)
        perfMetrics.append(perfWMF)
        perfNMF = nmf_ensemble_suite(n_factors=nf, save=True, 
                    run_stacker=True, run_nmf_ensemble=True, run_nmf_stacker=False, run_nmf_similarity=False, 
                        run_clustering=False, context='wmf_ensemble')
        perfMetrics.append(perfNMF)

    perfAll = post_analysis(perfMetrics, context='wmf_vs_nmf_ensemble')

    return perfAll

def test_similarity_nmf_vs_wmf(**kargs):
    perfMetrics = []

    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append(base_perf)

    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        perfWMF = wmf_ensemble_suite(base_perf=base_perf, n_factors=nf, save=True, 
                    run_stacker=False, run_wmf_ensemble=False, run_wmf_stacker=False, run_wmf_similarity=True, 
                        run_clustering=False, context='wmf_similarity') # set save to True to save R' and T' (reproduced probabilities)
        perfMetrics.append(perfWMF)
        perfNMF = nmf_ensemble_suite(base_perf=base_perf, n_factors=nf, save=True, 
                    run_stacker=True, run_nmf_ensemble=False, run_nmf_stacker=False, run_nmf_similarity=True, 
                        run_clustering=False, context='nmf_similarity')
        perfMetrics.append(perfNMF)

    perfAll = post_analysis(perfMetrics, context='wmf_vs_nmf_similarity')

    return perfAll

def test_wmf_clustering(**kargs):
    perfMetrics = []

    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append(base_perf)

    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        perfWMF = wmf_ensemble_suite(base_perf=base_perf, n_factors=nf, save=True, 
                    run_stacker=False, run_wmf_ensemble=False, run_wmf_stacker=False, run_wmf_similarity=True, 
                        run_clustering=True, context='wmf_similarity') # set save to True to save R' and T' (reproduced probabilities)
        perfMetrics.append(perfWMF)

    perfAll = post_analysis(perfMetrics, context='wmf_clustering')

    return perfAll 

def test_confidence(*kargs):
    perfMetrics = []

    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append(base_perf)

    div(message='(test) Compare different confidence matrix (project: %s)' % ProjectPath, symbol='#', border=1)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        for mode in ['brier', 'ratio', ]: 
            for alpha in [1, 50, 100, 1000, ]: 
                perfWMF = wmf_ensemble_suite(base_perf=base_perf, n_factors=nf, mode=mode, alpha=alpha, save=True, 
                            run_stacker=False, run_wmf_ensemble=False, run_wmf_stacker=False, run_wmf_similarity=True, 
                                run_clustering=False, context='wmf_confidence_matrices-C{mode}-A{alpha}'.format(mode=mode, alpha=alpha)) # set save to True to save R' and T' (reproduced probabilities)
                perfMetrics.append(perfWMF)

    perfAll = post_analysis(perfMetrics, context='wmf_confidence_matrices')

    return 

def test_similarity(**kargs):

    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    perfMetrics = []

    # t_recommender()
    base_perf = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append(base_perf)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        perfWMF = wmf_ensemble_suite(n_factors=nf, save=True, 
                    run_bp=False, run_stacker=False, run_wmf_ensemble=False, run_wmf_stacker=False, run_wmf_similarity=True, 
                        run_clustering=False, context='wmf_similarity') # set save to True to save R' and T' (reproduced probabilities)
        perfMetrics.append(perfWMF)
        perfNMF = nmf_ensemble_suite(n_factors=nf, save=True, 
                    run_bp=False, run_stacker=False, run_nmf_ensemble=False, run_nmf_stacker=False, run_nmf_similarity=True, 
                        run_clustering=False, context='nmf_similarity')
        perfMetrics.append(perfNMF)

        ## Neighborhood ensemble, memory-based ensemble
        perf_neighborhood = t_neighborhood_ensemble()  # memory-based
        perfMetrics.append(perf_neighborhood)

    perfAll = post_analysis(perfMetrics, context='similarity_comparison')

    return perfAll

def sysConfig(domain='pf2'):
    import utils_sys, common
    from cf_spec import System, MFEnsemble
    from evaluate import PerformanceMetrics
    
    System.projectPath = utils_sys.getProjectPath(domain=domain, verify_=False)  # default
    System.domain = domain

    # all directories depend on this prefix including data_dir, log_dir, plot_dir
    PerformanceMetrics.set_path(prefix=System.projectPath) 
    MFEnsemble.set_path(prefix=System.projectPath)

    p = common.load_properties(System.projectPath, config_file=System.configFile)  # parse config.txt (instead of weka.properties)
    System.foldCount = int(p['foldCount'])
    System.bagCount = int(p['bagCount']) if 'bagCount' in p else int(p['bags']) 

    return

def sysConfig0(**kargs):
    import cf_spec, common
    from cf_spec import System, MFEnsemble
    from evaluate import PerformanceMetrics
    
    cf_spec.ProjectPath = ProjectPath
    cf_spec.Domain = Domain

    # all directories depend on this prefix including data_dir, log_dir, plot_dir
    PerformanceMetrics.set_path(prefix=cf_spec.ProjectPath) 
    MFEnsemble.set_path(prefix=cf_spec.ProjectPath)

    # print('(sysConfig) log_dir: %s, plot_dir: %s' % (PerformanceMetrics.log_dir, PerformanceMetrics.plot_dir))

    p = common.load_properties(ProjectPath, config_file=System.configFile)  # parse config.txt (instead of weka.properties)
    System.foldCount = int(p['foldCount'])
    System.bagCount = int(p['bagCount']) if 'bagCount' in p else int(p['bags']) 

    return 

def test(**kargs): 
    import utils_cf as uc
    ### basic operations 
    # fold = 1
    # R, T, L_train, L_test, U = utils_cf.toRatingMatrix(fold, p_threshold=0.5, missing_value=0, verbose=True)
    # print('(test) dim(R): %s, dim(T): %s' % (str(R.shape), str(T.shape)))

    # label matrix doesn't quite make sense because all "users" i.e. classifiers will share the same ground truths 
    # L, Lt = to_rating_matrix_test(fold, p_threshold=0.5, save_=False, missing_value=-1, merge_=False)
    # print('(test) dim(Lt): %s, dim(L): %s' % (str(L.shape), str(Lt.shape)))
    # print('... Lt:\n%s\n' % Lt[:5, :5])

    # uc.t_cluster()

    ### foundataion: recommender system 
    perfAll = PerformanceMetrics()  # intended to hold merged performance metrics objects
    perfMetrics = []
    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    ## experiment on confidence weights
    # t_confidence_weights()

    ## slow ALS 
    # t_als()

    # suite(**kargs)  # comparison of groups of algorithms

    ### individual tests 

    # WMF vs NMF 
    # test_nmf_vs_wmf()

    ## test predictive strength of ME methods 
    test_wmf_probs(**kargs)

    # similairty metrics comparison (off-the-shelf vs MF-induced)
    # test_similarity()

    # test_similarity_nmf_vs_wmf()  # no clustering

    # test_wmf_clustering() # wmf + clustering, which clustering algorithms seem better?

    # test_confidence()
     
    ## stacker comparisons: stackers on MF-reproduced probabilities vs stackers on BP predictions
    # test_stacker()

    return perfAll

def run(**kargs): 
    # 1. tune model 
    # 2. run suite() using the hyperparams setting obtained from step 1

    ## model selection 
    # wmf_ensemble_modelselect(**kargs)

    ### MF-based ensemble learning
    suite(**kargs)  # comparison of groups of algorithms

    ### stacking on MF-reproduced dataset 
    # test_stacker()

    ### plot 
    # plot(file_name='performance_metrics-model-select-pf2.csv')

    return 

def main(**kargs): 
    # run()

    test()
    
    return 

if __name__ == "__main__": 
    sysConfig0()
    main()

