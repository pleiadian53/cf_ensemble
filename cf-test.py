#!/usr/bin/env python
# encoding: utf-8

### configurations
import os, math, sys, gc
from sys import argv
import utils_sys

# cluster_module_path = os.path.join(os.getcwd(), 'cluster')
# sys.path.insert(0,cluster_module_path) 

### Global Variables
Domain = '?'
ProjectPath = utils_sys.getProjectPath(domain=Domain, verify_=False)  # default
FoldCount = nestedFoldCount = 5 
BagCount = 10 

# try: 
#     ProjectPath = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
#     Domain = os.path.basename(ProjectPath)
# except: 
#     pass 
import cf_spec
# cf_spec.config(project_path=ProjectPath, domain=Domain)  # to be shared by all relavant modules 
from cf_spec import System
from utils_cf import MFEnsemble # derived from cf_spec(MFEnsemble)

# assert os.path.exists(ProjectPath)
# condition: definition of Domain and ProjectPath needs to precede evaluate, utils_cf

import numpy as np
import pandas as pd
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
from utilities import load_properties, cluster_cmd

### CF dependent modules
import utils_cf
import evaluate
from evaluate import PerformanceMetrics, Metrics # as perfm
from evaluate import visualizeCoeffs, plot_roc

import logging
from optparse import OptionParser

import warnings
warnings.filterwarnings("ignore")

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

class System(cf_spec.System): 
    # np.logspace(0, 3, num=4, base=10)
    param_grid = {'n_factors': [5, 10, 50, 100, 250, 500], 'alpha': np.logspace(0, 3, num=4, base=10, dtype=int)}  # {'n_factors':[100, ], 'alpha':[100, ]}
    policy_iter = 'subsampling'  # options: {'cv', 'seq', 'subsampling', }
    n_epochs = 30
    n_epochs_foldin = 30  # used in the "reduced" ALS to LS at test time
    lambda_val = 0.8
    n_runs = 1
    n_runs_modelselect = 1
    run_baseline = True

    unbag = False

    descriptions = {0: 'baseline',

                    1: 'item_centered', 2: 'user-centered', 
                    3: 'item-centered-unsupervised', 4: 'user-centered-unsupervised', 

                    5: 'item-centered-low-support', 6: 'preference-masked', 

                    7: 'item-centered-reconstruct', 8: 'user-centered-reconstruct', 
                    9: 'item-centered-mask-test-unsupervised', 10: 'user-centered-mask-test-unsupervised', 
                    11: 'item-centered-tradeoff', 12: 'user-centered-tradeoff', 
                    17: 'item-centered-tradeoff-reconstruct', 18: 'user-centered-tradeoff-reconstruct', 

                    21: 'item-centered-transfer', 22: 'user-centered-transfer', 

                    # proba threshold policy
                    31: 'item-centered-fmax', 32: 'user-centered-fmax', 

                    # ALS, optimization
                    41: 'item-centered-seed', 42: 'user-centered-seed', 
                    43: 'item-centered-long-iter', 44: 'user-centered-long-iter', 
                    45: 'item-centered-low-reg', 46: 'user-centered-low-reg',

                    # uniform confidence scores 
                    51: 'item-centered-uniform', 52: 'user-centered-uniform',

                    63: 'meta-users-filter-user-item',

                    # algorithmic control group 
                    100: 'uniform', 

                    }

    @staticmethod
    def display(padding=4):
        indent = ' ' * padding
        setting = System.options.setting
        msg = ''
        msg += '(System) >>> (verify)\n'
        msg += "{s}project path (domain={d}):{path}\n".format(s=indent, d=System.domain, path=System.projectPath)
        msg += "{s}n_factors: {nf}, alpha: {a} when NOT in model selection mode\n".format(s=indent, nf=System.n_factors, a=System.alpha)
        msg += "{s}iteration policy: {p}\n".format(s=indent, p=System.policy_iter)
        msg += "{s}n_epochs (MF algorithm): {ne}, n_runs (ensemble learning): {n}, n_runs_modelselect: {nm}\n".format(s=indent, ne=System.n_epochs, n=System.n_runs, nm=System.n_runs_modelselect)
        msg += "{s}outer foldout: {ocv}, inner cv foldcount:{icv}\n{s}... should be consistent with how BPs were trained!\n".format(s=indent, ocv=System.foldCount, icv=System.nestedFoldCount)
        msg += "{s}param_grid: {g}\n".format(s=indent, g=System.param_grid)
        msg += "{s}setting: {setting}: {descrp}".format(s=indent, setting=setting, descrp=System.descriptions.get(setting, 'generic'))
        div(message=msg, symbol="#", border=2)
    @staticmethod
    def get_domain():
        if System.domain in ['?', '', ]: 
            return os.path.basename(System.projectPath)
        return System.domain 
    

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

    @staticmethod
    def rmse_score(R, Q, P, missing_value=-1): # [todo]
        I = R != missing_value  # Indicator function which is zero for missing data
        ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
        MSE = ME**2  
        return np.sqrt(np.sum(MSE)/np.sum(I))  # sum of squared errors

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
                train_errors.append(MC.rmse_score(R,Q,P)) # Training RMSE for this pass
        elif self.solver == "batch_gd":
            # Batch GD
            for epoch in range(self.n_epochs): 
                ERR = np.multiply(R != 0, R - np.dot(P, Q.T))  # compute error with present values of Q, P, ZERO if no rating   
                P += self.gamma*(np.dot(Q.T, ERR.T).T - self.lmbda*P)  # update rule
                Q += self.gamma*(np.dot(P.T, ERR).T - self.lmbda*Q)  # update rule
                train_errors.append(MC.rmse_score(R,Q,P)) # Training RMSE for this pass
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
        fpath = os.path.join(l1_data_path, 'cf-%s-f%d-b%d.csv' % (split, fold, System.bagCount))  # naming: test-b3-f1-s1.csv.gz
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

def base_stackers(topk=-1, metric='fmax', parallelize=True): 
    
    ret = {}
    target_metric = metric # kargs.pop('metric', 'fmax')
    # topk = kargs.get('topk', -1)

    ret = run_stacker(dataset='bp', parallelize=parallelize)  # dataset='bp'
    perf = ret['metrics']
    if topk > 0: perf = PerformanceMetrics.getTopK(perf, metric=target_metric, k=topk, reverse=True, verbose=True)
    return ret
### alias 
run_baseline_stacker = base_stackers

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
    baseline = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', True) else {}
    if baseline: perfMetrics.append(baseline['metrics'])

    # n_fold = System.foldCount

    ######### WMF parameters #########
    params = {}
    params['n_factors'] = args.get('n_factors', MFEnsemble.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)
    params['alpha'] = kargs.get('alpha', MFEnsemble.alpha)
    
    # parameters for confidence matrix
    params['conf_measure'] = kargs.get('conf_measure', 'brier')  # confidence matrix
    params['masked'] = kargs.get('masked', True) # mask FP and FN? this makes Cui 'sparse'

    # parameters for ALS methods 
    params['policy'] = kargs.get('policy', 'rating')  # options: {'rating', 'preference', 'label'}
    ######################################################


    perfMetrics = []
    # example method name:wmf_F10_A100_Xbrier_preference, which corresponds to: wmf_F10_A100_Xbrier_preference-validation-3.csv.gz
    tsetID = MFEnsemble.get_dset_id(method='wms', params=params)

    # datasets 
    datasets = common.match_exact(path=ProjectPath, method=tsetID, file_type='validation', ext='csv.gz', verify=True) # exception_=False
    
    for dataset, indices in datasets.items(): 
        # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
        assert dataset.find('pref') > 0, "dataset {0} does not contain preference scores ...".format(dataset)
        for method in ['mean', ]:
            # load preference scores

            perf_per_fold = []
            for index in indices:  # note: range(n_fold) does not work with 'random subsampling' 
                # load preference scores
                train_pref, train_labels, test_pref, test_labels = common.read(index, path=ProjectPath, dataset=dataset, reconstructed_testset=True) # common.read_fold(project_path, index)        

                # load original data 
                train_df, train_labels, test_df, test_labels = common.read(index, path=ProjectPath, dataset='bp', reconstructed_testset=True)
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

    if baseline: 
        docs = {'method': 'preference_stacking', 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return perfAll

def run_preference_combiner(dataset, **kargs): 
    from cf_spec import System 
    ret = {}

    # pref scores in one dataset and original BP score in another
    # n_fold = System.foldCount 
    aggregate_method = kargs.get('method', 'mean') 

    method_id, indices = common.resolve(path, dataset=dataset, file_type='predictions', reconstructed_testset=True, ext='csv.gz', exception_=True)
    assert len(datasets) > 0, "(run_preference_combiner) Could not find matching datasets using method_id: %s" % dataset

    perf_per_fold = []
    for index in indices: 
        # preference score 
        train_pref, train_labels, test_pref, test_labels = common.read(index, path=ProjectPath, dataset=method_id, reconstructed_testset=True) # common.read_fold(project_path, index)        
        
        # original data 
        train_df, train_labels, test_df, test_labels = common.read(index, path=ProjectPath, dataset='bp', reconstructed_testset=True)
        assert test_pref.columns.size == 2 * test_df.columns.size, "dim(test_pref): {0}, dim(test_df): {1}".format(test_pref.shape, test_df.shape)

        if aggregate_method == 'mean': 
            T_pref = test_pref[test_df.columns].values.T   # transpose to get users-by-items format
            T = test_df.values.T

            predictions = combiner(T_pref, aggregate_func='pref', T=T)

            # the algorithm behind the dataset + combiner method
            full_method = '{prefix}_C{combiner}_pref'.format(prefix=method_id, combiner=aggregate_method)
            perf_per_fold.append( analyzePerf(test_labels, predictions, method=full_method) )
    
    div('(run_preference_combiner) Completed method %s #' % full_method)
    ret['metrics'] = PerformanceMetrics.consolidate(perf_per_fold, test_= False)  # foreach metric, take average over CV folds
    return ret

def make_prediction_dataframe(y_pred, y_label, method, index):
    return DataFrame({'prediction':y_pred,'label':y_label, 'method': method, 'fold': index}, index=range(len(y_pred))) 

def run_simple_combiner(dataset, aggregation_func='mean', file_type='', n_runs=-1, **kargs):
    from cf_spec import System 
    import pandas as pd
    import utils_sys as us
    from tabulate import tabulate

    ret = {}   # output

    aggregate_method = aggregation_func  # a string or a function
    method_id = dataset
    # n_fold = System.foldCount

    # tset, labels = common.readAll(ProjectPath, dataset=method_id, file_type='predictions', exception_=True) # note: set exception_ to True to preclude multiple matches
    perf_per_fold = []
    dfs = []  # save data

    full_method = '?'
    index = 0
    if file_type.startswith(('pri', 'post')): 

        if aggregate_method.find('latent') >= 0: # then the predictions are already available 
            sep = kargs.get('sep', ',')
            dset_type = file_type if file_type.startswith(('prior', 'post')) else 'prediction'
            fpath = '{path}/analysis/{stacker}.S-{dataset}-{suffix}.csv'.format(path=ProjectPath, stacker=aggregate_method, dataset=method_id, suffix=dset_type)
            assert os.path.exists(fpath), "predictions not found | dtype: {dtype}, method_id: {id} | path: {path}".format(dtype=dset_type, id=method_id, path=fpath)
            df = pd.read_csv(fpath, sep=sep, header=0, index_col=False) # error_bad_lines=True 

            predictions = df.prediction
            labels = df.label
            full_method = '{prefix}_{dataset}_combiner'.format(prefix=aggregate_method, dataset=dataset)
            perf_per_fold.append( analyzePerf(labels, predictions, method=full_method) )

        else: 
            if n_runs <= 0: n_runs = System.n_runs
            # readAllIter() will attempt to resolve the indices by itself
            for train_df, train_labels, test_df, test_labels in common.readAllIter(path=ProjectPath, dataset=method_id, file_type=file_type, n_runs=n_runs): 
                
                # simple combiner very often do not even look at the training split 
                Th = test_df.values.T
                labels = test_labels # test_df.index.get_level_values('label').values

                if aggregate_method == 'mean': 
                    predictions = combiner(Th, aggregate_func=np.mean)
                elif aggregate_method == 'median': 
                    predictions = combiner(Th, aggregate_func=np.median)
                else:  # user provide a customize aggregation function (e.g. weighted average)
                    customized_func = aggregation_func
                    assert hasattr(customized_func, '__call__')
                    aggregate_method = customized_func.__name__
                    predictions = combiner(Th, aggregate_func=customized_func)

                # the algorithm behind the dset_id + combiner method
                # note: stacker naming format: '{prefix}_{dataset}_stacker'.format(prefix=method, dataset=dataset)
                full_method = '{prefix}_{dataset}_combiner'.format(prefix=aggregate_method, dataset=dataset)
                if file_type: 
                    full_method = '{prefix}_{dataset}_combiner_{dtype}'.format(prefix=aggregate_method, dataset=dataset, dtype=file_type)
                
                perf_per_fold.append( analyzePerf(labels, predictions, method=full_method) )

                dfe = DataFrame({'prediction':predictions,'label':labels, 'method': aggregate_method, 'fold': index}, index=range(len(predictions)))
                dfe['label'] = dfe['label'].astype(int)
                dfe['fold'] = dfe['fold'].astype(int)
                dfs.append(dfe)
           
                index += 1 

    else:
        # if validation, predictions sets are already separated
        for tset, labels in common.readAllIter(ProjectPath, dataset=method_id, file_type='predictions', reconstructed_testset=True, ext='csv.gz', exception_=True): 
            Th = tset.values.T

            if aggregate_method == 'mean': 
                predictions = combiner(Th, aggregate_func=np.mean)
            elif aggregate_method == 'median': 
                predictions = combiner(Th, aggregate_func=np.median)
            else:  # user provide a customize aggregation function (e.g. weighted average)
                customized_func = aggregation_func
                assert hasattr(customized_func, '__call__')
                aggregate_method = customized_func.__name__
                predictions = combiner(Th, aggregate_func=customized_func)

            # the algorithm behind the dset_id + combiner method
            full_method = '{prefix}_C{combiner}'.format(prefix=method_id, combiner=aggregate_method)
            perf_per_fold.append( analyzePerf(labels, predictions, method=full_method) )

            # columns: ['fold',  'label', 'prediction', 'method']  # other candidate columns: 'id', 'diversity',
            # naming convention: 
            #  e.g. <aggregate_method>.S-prediction.csv for the regular data set (generated by base predictors)
            #       <aggregate_method>.S-wmf_F10_A100_Xbrier_rating-prediction.csv (generated by CF reconstruction)

            dfe = DataFrame({'prediction':predictions,'label':labels, 'method': aggregate_method, 'fold': index}, index=range(len(predictions)))
            dfe['label'] = dfe['label'].astype(int)
            dfe['fold'] = dfe['fold'].astype(int)
            dfs.append(dfe)

            index += 1

    # save predictions
    if len(dfs) > 0: 
        df_prediction = pd.concat(dfs, ignore_index=True)
        dset_type = file_type if file_type.startswith(('prior', 'post')) else 'prediction'
        if method_id in ('', None, 'bp'): 
            df_prediction.to_csv('{path}/analysis/{stacker}.S-{suffix}.csv'.format(path=ProjectPath, stacker=aggregate_method, suffix=dset_type), index = False)
        else: 
            print('(verify) saving aggregation result from {name} on reconstructed data {data}'.format(name=aggregate_method, data=method_id))
            df_prediction.to_csv('{path}/analysis/{stacker}.S-{dataset}-{suffix}.csv'.format(path=ProjectPath, 
                stacker=aggregate_method, dataset=method_id, suffix=dset_type), index = False)
    
    div('(run_simple_combiner) Completed method %s' % full_method)
    ret['metrics'] = perf = PerformanceMetrics.consolidate(perf_per_fold, test_=kargs.get('test', False))  # foreach metric, take average over CV folds
    
    # [test]
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perf, metric=metric, verbose=True if metric in ['fmax', ] else False) 
    
    print('(run_simple_combiner) sorted methods by fmax score')
    print(tabulate(perf.table.head(10), headers='keys', tablefmt='psql'))
    print(ret['fmax'])
    # print(us.format_sort_dict(ret['fmax'], key='default', reverse=True, padding=0, title='', symbol='#', border=1, verbose=False))
    return ret

def run_combiner(dataset, **kargs):
    """
    
    Memo
    ----
    1. Two types of training data 
       - regular 
       - augmented by preference scores

    """
    ret = {}
    perf = PerformanceMetrics()
    if MFEnsemble.is_preference_data(dataset): 
        # if kargs.get('aug_data', False): 
        # perf = run_pref_stacker(dataset)  # use preference scores to combine the result in the test set
        ret = run_preference_combiner(dataset, **kargs)  # simple combining rule using preference scores as heuristics
    else: 
        # kargs: method, aggregation_func, (test, )
        ret = run_simple_combiner(dataset, **kargs)

    return ret 

def test_combiner(datasets=[], **kargs):
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
    baseline = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', False) else {}
    if baseline: perfMetrics.append(baseline['metrics'])

    # n_fold = System.foldCount
    if len(datasets) == 0: 
        param_grid = kargs.get('param_grid', {})
        assert len(param_grid) > 0, "Both the named datasets and parameter grid are missing!"
        datasets = MFEnsemble.name_tsets(param_grid, meta_params=kargs.get('meta_params', {'conf_measure': 'brier', 'policy': 'rating', }))
    
    # run simple aggregation (e.g. mean) on the original data as well 
    if not 'bp' in datasets: datasets.append('bp')

    # Now, given a set of named datasets ... 
    for dataset in datasets: 
        # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
        combiner = run_combiner(dataset, **kargs)
        perfMetrics.append( combiner['metrics'] ) # kargs: method, aggregation_func, (test, )

    ### end foreach dataset 

    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    if baseline: 
        docs = {'method': kargs.get('method', 'mean'), 
                'target_method': kargs.get('target_method', 'wmf'), 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

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
    baseline = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', False) else {} 
    if baseline: perfMetrics.append(baseline['metrics'])

    indices = kargs.get('indices', range(System.foldCount) if System.policy_iter == 'cv' else range(System.n_runs))
    for dataset in [dataset, ]: 
        # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
        assert dataset.find('pref') > 0, "dataset {0} does not contain preference scores ...".format(dataset)
        for method in ['mean', ]:
            # load preference scores

            full_method = '{prefix}_C{combiner}_pref'.format(prefix=dataset, combiner=method)
            perf_per_fold = []
            for index in indices: 
                train_pref, train_labels, test_pref, test_labels = common.read(index, path=ProjectPath, dataset=dataset, reconstructed_testset=True) # common.read_fold(project_path, index)        
                
                # load original data 
                train_df, train_labels, test_df, test_labels = common.read(index, path=ProjectPath, dataset='bp', reconstructed_testset=True)
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

    ret = {}
    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)
    # for metric in PerformanceMetrics.tracked: 
    #     ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    if baseline: 
        docs = {'method': 'preference_stacking', 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return ret

def run_stacker(**kargs): 
    """
    Run stacker routines. 
    Not part of this module but put here for the convenience of comparison. 

    **kargs
        topk_bps

        parallelize

    
    Memo
    ----
    1. Candidate stackers 

       Naive Bayes: sklearn.naive_bayes.GaussianNB
       AdaBoost:    sklearn.ensemble.AdaBoostpredictor
       Decision Tree: sklearn.tree.DecisionTreepredictor

       LogitBoost:   sklearn.ensemble.GradientBoostingpredictor
       KNN:          sklearn.neighbors.KNeighborspredictor

       logistic:     sklearn.linear_model.LogisticRegression
       SGD:          sklearn.linear_model.SGDpredictor
       RF:           sklearn.ensemble.RandomForestpredictor

    """
    import stacking 
    import utils_sys as us
    from tabulate import tabulate
    # from evaluate import Metrics # evalTestSet

    ret = {}  # output
    # perf = PerformanceMetrics()   # rows: metrics, cols: bps

    perfMetrics = [] # performance metrics object for each method
    dataset = kargs.get('dataset', 'bp')

    # probably always want to compare with base predictors 
    topk_bps = kargs.get('topk_bps', -1)
    baseline = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', False) else {} 
    if baseline: perfMetrics.append(baseline['metrics'])

    # candidate default stackers 
    # a. ['lasso', 'enet', 'rf', 'gb',  ]
    # b. ['log', 'qda', 'enet', 'svm', 'naive', 'rf', 'ada', 'knn', ]  # rf: random forest, 'gb': gradient boosting tree
    policy_iter = kargs.get('policy_iter', System.policy_iter)
    file_type = kargs.get('file_type', '')

    for method in ['log', 'qda', 'enet', 'svm', 'naive', 'rf', 'ada', 'knn',  ]:  # special stackers:  
        print('(run_stacker) Running stacker {model} ...'.format(model=method))
        indices = kargs.get('indices', range(System.foldCount) if policy_iter == 'cv' else range(System.n_runs))

        ################################################################################################
        predictions_df = stacking.run(name=method, dataset=dataset, parallelize=kargs.pop('parallelize', True), 
            indices=indices, policy_iter=policy_iter, file_type=file_type)  # stacking.run() -> stacked_generalization()
        ################################################################################################

        ### apply a scoring function to each fold and then take the average
        # predictions_df.groupby('fold').apply(lambda x: common.score(x.label, x.prediction)).mean()
        
        ### method naming
        method_id = '{prefix}_{id}_stacker'.format(prefix=method, id=dataset)  # e.g. rf_bp_stacker
        if file_type.startswith('pri'):  
            method_id = '{prefix}_stacker_{dtype}'.format(prefix=method, dtype=file_type) # omit 'dataset' for simplicity
        elif file_type.startswith('post'):
            print("(run_stacker) Stacking '{dtype}' dataset: {0} using stacker: {1}".format(dataset, method_id, dtype=file_type)) 
            method_id = '{prefix}_{id}_stacker_{dtype}'.format(prefix=method, id=dataset, dtype=file_type) 

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
    
    # [test]
    print('(run_stacker) sorted methods by fmax score:\n%s' % ('-' * 100))
    print(tabulate(us.display_dataframe(perfAll.table), headers='keys', tablefmt='psql')) 
    print(ret['fmax'])

    if baseline: 
        docs = {'method': 'stacking', 'dataset': dataset}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
    return ret

def run_stacker_suite(**kargs):
    import stacking

    ret = {}
    perfMetrics = [] # performance metrics object for each method

    ### Baseline Methods: base predictors (BPs), stackers over BP outputs (or BP stackers)

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    if kargs.get('run_bp', True): 
        baseline = base_predictors(topk=kargs.get('topk_bps', -1), metric=kargs.get('metric', 'fmax'))
        perfMetrics.append(baseline['metrics']) 

    ## how does it compare to stacking on BP outputs?
    if kargs.pop('run_bp_stacker', True):
        basestacker = base_stackers(topk=kargs.get('topk_bp_stackers', -1), metric=kargs.get('metric', 'fmax'))  # dataset='bp'
        perfMetrics.append(basestacker['metrics'])

    method = kargs.get('method', '')  # if given, focus only on the methods containing this method keyword (e.g. 'nmf'); if '', then consider all methods
    keywords = kargs.get('keywords', [])  # e.g. to focus on those with n_factors = 20 => put 'F20' in the keyword

    ### Condition: Baseline methods established
    
    # first, locate the training data
    file_type = kargs.get('file_type', 'validation')
    policy_iter = kargs.get('policy_iter', 'subsampling')

    if kargs.get('exact', False): 
        # method has to match exactly with the prefixed part of the file name 
        # e.g. Say method <- 'nmf', then we are only looking at { 'nmf-validation-{fold}.csv.gz' | fold = 0, 1, 2...}
        #      nmf_item_kmeans_sim_F10-validation-0.csv.gz for instance, is not considered a match
        datasets = common.match_exact(path=ProjectPath, method=method, file_type=file_type, ext='csv.gz', verify=True, policy_iter=policy_iter) # exception_=False
    else: 
        # search relevant training data whose names contain the given method (as a keyword); use keywords argument as additional constraints to narrow down the scope 
        datasets = common.match(path=ProjectPath, method=method, keywords=keywords, file_type=file_type, ext='csv.gz', verify=True)

    div('(run_stacker_suite) Found %d matched datasets (method ID: %s)' % (len(datasets), method), symbol='=', border=1)

    for dataset, indices in datasets.items():
        # in this case, preference scores are only used to select probabilities in the test data to make final predictions
        print('... (verify) dataset: {name}, indices: {idx}'.format(name=dataset, idx=indices))

        # preference data are now obsolete (too complicated yet no apparent performance benefits)        
        # if not kargs.get('aug_data', True) and dataset.find('pref') > 0:  
        #     print('(run_stacker_suite) Not using the preference-augmented dataset ...')
        #     ret = run_pref_stacker(dataset=dataset, indices=indices)
        #     perfMetrics.append(ret['metrics'])  
        # else: 

        # if aug_data <- True and dataset has the keyword 'pref', then just treat the dataset as a normal training data (but with augmented features)
        ret = run_stacker(dataset=dataset, parallelize=kargs.get('parallelize', True), reconstructed_testset=kargs.get('reconstructed_testset', True), indices=indices)
        perfMetrics.append(ret['metrics'])

    # merge all performance metrics into one big table 
    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)

    if kargs.get('evaluation', True): 
        for metric in PerformanceMetrics.tracked: 
            ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

        post_analysis(perfAll, context=kargs.get('context', 'stacker_suite'), highlight=['stacker', ])

    # context = kargs.get('context', 'stacker_suite')
    # if baseline: 
    #     docs = {'method': 'stacking', 'dataset': dataset}
    #     PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

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

    n_fold = System.foldCount
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

def base_predictors(topk=-1, metric='fmax', **kargs): 
    """

    Input
    -----
        metric: the metric used to select the top K model (when topk > 0)
    """
    from evaluate import Metrics, plot_roc
    from evaluate import PerformanceMetrics # as perfm
    import utils_cf as uc

    ret = {}
    n_fold = System.foldCount
    missing_value = 0 # marker for missing data
    p_th = 0.5

    perfx = []
    policy_iter = kargs.get('policy_iter', System.policy_iter)  # train_dev_test
    
    n_runs = kargs.get('n_runs', System.n_runs)  # facilitates external call without init System (e.g. analyze_performance.base_predictors())
    fold_count = kargs.get('fold_count', System.foldCount)
    unbag = kargs.get('unbag', System.unbag)
    bag_count = kargs.get('bag_count', System.bagCount)

    indices = kargs.get('indices', range(fold_count) if policy_iter == 'cv' else range(n_runs))

    for index in indices: 
     
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        if policy_iter == 'cv': 
            R, T, L_train, L_test, U = uc.to_rating_matrix(index, verbose=True, unbag=unbag, bag_count=bag_count)
        else: 
            R, T, L_train, L_test, U = uc.to_rating_matrix_random_subsampling(dev_ratio=1./System.foldCount, test_ratio=1./System.foldCount, 
                train_dev_test=False, shuffle=True, fold_count=fold_count, unbag=unbag, bag_count=bag_count)

        n_users, n_items = R.shape[0], R.shape[1]
        print('(base_predictors) dim(R): {0}, dim(T): {1}, n_train: {2}, n_test: {3}'.format(R.shape, T.shape, len(L_test), len(L_test)) )

        df_perf = analyzeBasePerf(L_test, T, U, unbag=True, bag_count=bag_count)  # tracked_metrics = PerformanceMetrics.tracked_metrics
        perfx.append(df_perf)
    # else:  # subsampling 
    #     for index in range(System.n_runs): 
    #         R, T, L_train, L_test, U = uc.to_rating_matrix_random_subsampling(index, train_dev_test=False, verbose=True )

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

    return ret   

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

    n_fold = System.foldCount
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    
    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', System.n_factors)
    params['n_epochs'] = kargs.get('n_epochs', System.n_epochs)

    ret = {}

    perfMetrics = []

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    baseline = base_predictors(topk=kargs.get('topk_bps', -1), 
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

        Rh, Th = uc.predict_by_factors(P, Q, test_offset=len(L_train)) 
        if kargs.get('save', True): 
            uc.save_reconstructed_training_data(Rh, L_train, fold, method, verify=True, U=U)
            # uc.save_reconstructed_test_data(Th, L_test, fold, method, verify=True, U=U)

        for kind in kinds: 

            # metrics = compareEstimates0(T, L_test, Th=Th, R=None, L_train=None)

            # optinal params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
            method_id = MFEnsemble.get_method_id(method, kind, params=params)  # '{method}_{kind}'.format(method=method, kind=kind)
            
            # # also consider stacking on top of the reproduced probabilities
            # if kind in stackers: # is a kind of stacker => need special performance Handler
                
            #     # **kargs: classifier hyperparams
            #     perf, df_prediction = analyzePerfStacker(fold, Rh, Th, method=method_id)  # run stacker on top of the reproduced probabilities
            #     y_true, y_score = df_prediction['label'], df_prediction['prediction']
            #     assert all(L_test == y_true)
            #     nmfMetrics[kind].append(perf)
            #     nmfCV[kind].append((L_test, y_score))
            
            nmfMetrics[kind].append(analyzePerf(L_test, Th, method=method_id, aggregate_func=np.mean, T=T, fold=fold))  # analyzePerf -> { compare* } where compare* is a set of analysis functions (e.g.  compareEstimates_
            nmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )

        # bpMetrics.add(metrics['bp'])
        # mfMetrics.add(metrics['cf'])

    ## evaluation 
    for kind in kinds: 
        if kind in nmfCV: 
            method_specific = MFEnsemble.get_method_id(method, kind, params=params)
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
    if baseline: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

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

    n_fold = System.foldCount
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 

    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', System.n_factors)
    params['n_epochs'] = kargs.get('n_epochs', System.n_epochs)
    topk = 30

    # output
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    baseline = base_predictors(topk=kargs.get('topk_bps', -1), 
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
    if baseline: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
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
    
    n_fold = System.foldCount
    p_th = 0.5
    missing_value = 0 # marker for missing data
    topk = 30

    ret = {}  # output
    perfMetrics = [] # method-wise performance metrics including (cosine and pearson correlation-based methods)
    # options: target_metric is the metric used to select the topK models
    baseline = base_predictors(topk=kargs.get('topk_bps', -1), 
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
            # Note: get_method_id returns a shorter ID (cf: get_dset_id)
            method_specific = MFEnsemble.get_method_id(method='cosine', kind=kind, params=params)
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
            method_specific = MFEnsemble.get_method_id(method='corr', kind=kind, params=params)
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
    if baseline: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
    return perfAll

def als_cv_routine(fold, **kargs): 
    """
    
    **kargs
       n_factors 
       n_epochs 
       alpha
       mode

    """
    from utils_als import implicit_als_cg, implicit_als

    n_fold = System.foldCount
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    
    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', System.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', System.n_epochs)
    params['alpha'] = alpha_val = kargs.get('alpha', System.alpha)
    
    # parameters for confidence matrix
    params['conf_measure'] = kargs.get('conf_measure', 'brier')  # confidence matrix

    Cui, R, T, L_train, L_test, U = uc.toConfidenceMatrix(fold, p_threshold=p_th, fill=missing_value, verbose=True, is_augmented=True, mode=params['conf_measure'])
    n_users, n_items = Cui.shape[0], Cui.shape[1]

    # n_users, n_items = R.shape[0], R.shape[1]
    print('(als_cv_routine) Fold: %d, n_factors: %d | dim(Cui): %s, L_train: %d, n_test: %d' % (fold, params['n_factors'], str(Cui.shape), len(L_train), len(L_test)))

    conf_data = (Cui * params['alpha']).astype('double')
    n_nonzeros = sparse.csr_matrix.count_nonzero(conf_data)
    n_zeros = n_users * n_items - n_nonzeros
    print('... n_zeros: %d, n_nonzeros: %d, ratio: %f' % (n_zeros, n_nonzeros, n_zeros/(n_zeros+n_nonzeros+0.0)))  # [log] 104902189710

    ### change ALS algorithm here ### 
    P, Q = implicit_als(conf_data, R=np.hstack((R, T)), iterations=params['n_iter'], features=params['n_factors'])

    return (P, Q)

def wmf_ensemble_iter(data, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', fold=-1, null_marker=0, verbose=1, 
        project_path='?', save=False, piggyback=True, dev_ratio=0.2, max_dev=None):
    """
    
    Params
    ------
    fold: defined in a more generic sense than the 'fold' in wmf_ensemble_fold(). 
          specifically, 'fold' can refer to any of the following: 

           i) cv fold ii) the index of iterations in random subsampling iii) other iteration index

    indices: dataframe index and columns
    save: default set to False because this subroutine is typically used for model selection

    Memo
    ----
    1. use wmf_ensemble_fold() for CV iteration

    """
    def initmap(): # variable map is a nested dictionary in which each variable has it's own keys (kind) representing subcategories of the MF algorithm
        vmap = {} 
        for var in vars:
            vmap[var] = [] # {k: [] for k in kinds}   # var -> kind -> a list
        return vmap
    def get_class_stats(labels, pos_label=1, neg_label=0, ratio_imbalanced=0.1):
        # if fold == 0:  # fold is defined in a more generic sense: i) cv fold ii) the index of iterations in random subsampling iii) other iteration index
        
        # keys: n_pos, n_neg, r_pos, r_neg, 0/neg_label, 1/pos_label, is_balanced, r_minority, min_class
        stats = uc.classPrior(labels, labels=[neg_label, pos_label], ratio_ref=ratio_imbalanced)
        div('(result) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2, 7, 8): 
            assert params['supervised']
        if params['setting'] in (3, 4, 9, 10): 
            assert not params['supervised']
        if params['setting'] in (7, 8, 9, 10): 
            assert params['predict_probs'], "Setting 7 - 10 should attempt to re-estimate the entire T"
    def verify_confidence_matrix(C, Cbar):
        if params['setting'] in (11, 12): 
            assert Cbar is not None
            assert params['policy_opt'].startswith('trade')
    def make_prediction_vector(X, L=[], M=None, policy=''):
        if not policy: policy=params['policy'] 
        if M is not None: assert len(L) == 0
        pv = uc.to_mean_vector(X, L=L, 
                M=M,  # message from training set when L is not accessible

                ratio_users=params['ratio_users'],  # filtering in the item direction 
                ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0),  # used for unsupervised mode 

                policy=policy,  # determining filtering dimension
                policy_threshold=params['policy_threshold'], # determining proba threshold

                    supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    fill=null_marker, fold=fold)
        return pv

    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us

    method = params.get('method', 'wmf')
    tMetaUsers = params.get('include_meta_users', False)  # if True, add extra meta classsifiers/users in the last rows of R and T
    vmap = initmap() # each call to this routine produces a new result

    # Note
    # if we don't re-configure cf_spec, then the information from the initial configuration (from sysConfig) 
    # will be lost => each thread sees its own version of cf_spec and its class definitions 
    ########################################################################################
    n_fold = FoldCount
    cf_spec.config(project_path=ProjectPath, domain=Domain, fold_count=FoldCount, bag_count=BagCount)
    ########################################################################################
    algorithm_setting = params.get('setting', '?') # see System.descriptions
    verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    ### Confidence Matrix
    # fold: iteration number (here, 'fold' is simply a repurposed variable borrowed from CV)
    # data: (R, T, L, Lt, U)-format
    if isinstance(data, DataFrame): # in this case, 'data' is a dataframe comprising a train-dev split 
        R, T, L_train, L_test, U =  uc.shuffle_split_data(data, ratio=dev_ratio, max_size=max_dev)
    else:
        # otherwise, assuming that shuffle-split operation had been invoked to produce the data input
        assert isinstance(data, (tuple, list)), "Invalid input data: %s" % data 
        print('(wmf_ensemble_iter) verify: len(data): {n}'.format(n=len(data)))
        R, T, L_train, L_test, U = data 
    n_samples = R.shape[1]+T.shape[1]; assert len(L_train)+len(L_test) == n_samples

    ### create extra prediction vectors (PVs) from R (say, mean vector) and attach these new PVs to R (piggyback)
    n_users, n_items = R.shape
    n_users_test, n_items_test = T.shape
    n_meta_users = 0

    ############################################################
    # piggy back extra meta-estimators
    meta_users = ['latent_mean', 'masked_latent_mean',]  # todo
    if tMetaUsers: 
        div(message='(wmf_ensemble_iter) Adding meta users (n={n})...'.format(n=len(meta_users)))
        
        # e.g. mean, and masked mean
        nU, nUT = n_users, n_users_test

        print('... augmenting T (by meta users)')
        mean_pv = make_prediction_vector(T, L=[], policy='none')
        masked_mean_pv = make_prediction_vector(T, L=[], M=(R, L_train), policy=params['policy_test'])
        T = np.vstack((T, mean_pv, masked_mean_pv))
        n_users_test = T.shape[0]

        print('... augmenting R (by meta usrs)')
        mean_pv = make_prediction_vector(R, L_train, policy='none')
        masked_mean_pv = make_prediction_vector(R, L_train, policy=params['policy'])
        R = np.vstack((R, mean_pv, masked_mean_pv))
        n_users = R.shape[0]

        n_meta_users = n_users - nU
        assert n_meta_users == (n_users_test - nUT) == len(meta_users)
    ############################################################

    # compute confidence matrix for R
    Cr, Cr_bar = uc.evalConfidenceMatrix(R, L=L_train, U=U,  
            ratio_users=params['ratio_users'], 
            ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< determines the subroutine for computing Cui
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], alpha=params['alpha'], 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, fold=fold) # project_path=System.projectPath 
    # ... Cui_bar is only used in policy = 'tradeoff'
    verify_confidence_matrix(Cr, Cr_bar)
    
    div("(wmf_ensemble_iter) Completed conficence matrix for training data C(R) | Cycle {0} | n_factors: {1}, alpha: {2} | dim(Cui): {3} | conf: {4}, conf_measure: {5}, optimization: {6} | predict ALL probabilities? {7} | policy_threshold: {8}".format(fold, 
        params['n_factors'], params['alpha'], str(Cr.shape), params['policy'], params['conf_measure'], params['policy_opt'], params['predict_probs'], params['policy_threshold']), symbol='#', border=1)
    print('...... Data | n_train: {ntr}, n_test: {nt}, ratio: {r}'.format(ntr=len(L_train), nt=len(L_test), r=(len(L_train)+len(L_test))/(len(L_test)+0.0)) )
    print('...... Classifiers | n_users: {nu}, n_meta_users: {nmu} => n_original_users: {no}'.format(nu=n_users, nmu=n_meta_users, no=n_users-n_meta_users))
    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} > FOLD: {f}'.format(dim=R.shape, f=fold))
    piggyback_msg = "+ thread/fold: {index} | setting: {setting}".format(index=fold, setting=algorithm_setting).rjust(5)   # keep track of the thread from which the iteration 
    
    ########################################################################################
    Pr, Qr, *Rh_errs = ua.implicit_als(Cr, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            label_confidence=Cr_bar, ratings=R, labels=L_train,
                            policy=params['policy_opt'], message=piggyback_msg, ret_rmse=True)
    Rh_err, Rh_err_weighted = Rh_errs
    ########################################################################################
    ne = 2
    e_pri, e_post = np.mean(Rh_err[:ne]), np.mean(Rh_err[-ne:])
    e_del = (e_pri-e_post)/e_pri * 100
    print('... (ALS 1) Complete | rmse: {e1} -> {e2} (n_errs: {n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(e1=e_pri, e2=e_post, 
                   e_del=e_del, ew1=np.mean(Rh_err_weighted[:ne]), ew2=np.mean(Rh_err_weighted[-ne:]), n=len(Rh_err) ))

    # e.g. 50 users, 3979 items with nf=100 > dim(P): (50, 100), dim(Q): (3979, 100)
    assert Pr.shape[0] == R.shape[0] and Pr.shape[1] == params['n_factors']
    assert Qr.shape[0] == R.shape[1] and Qr.shape[1] == params['n_factors']

    ### create extra prediction vectors (PVs) from T (say, mean vector) and attach these new PVs to T (piggyback)

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    # compute confidence matrix for T
    Ct, Ct_bar = uc.evalConfidenceMatrix(X=T, L=[], U=U, M=(R, L_train),  # set M/message to R, L so that we can use proba thresholds from R to estimte labels in T
            ratio_users=params['ratio_users'], 

            # parameters to be used for unsupervised mode
            ratio_small_class=class_stats['r_minority'], 
            factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy_test'], # <<< determine the dimension of filtering (user, item)
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=False,  # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], alpha=params['alpha'], 

                        # masked=params['masked'], mask_all_test=params['mask_all_test'],
                        fill=null_marker, fold=fold, L_true=L_test) # project_path=System.projectPath 
    # assert Ct.shape == T.shape    # ... ok
    verify_confidence_matrix(Ct, Ct_bar)
    assert n_users_test == Ct.shape[0] and n_items_test == Ct.shape[1]
    div("... Completed conficence matrix for test data C(T) | Cycle {0} | n_factors: {1}, alpha: {2} | dim(Ct): {3} | conf: {4} optimization: {5} | predict ALL probabilities? {6}".format(fold, 
        params['n_factors'], params['alpha'], str(Ct.shape), params['policy'], params['policy_opt'], params['predict_probs']), symbol='#', border=1)
     
    # use Qr in T 
    div('... (ALS 2) Going into the ALS loop on TEST data (T): {dim}  > FOLD: {f} | policy_opt_T: {policy}'.format(dim=T.shape, 
        f=fold, policy=params['policy_opt_T'])) 
    ########################################################################################
    policy_opt_eff = 'rating' if params['policy_opt'].startswith('trade') else params['policy_opt']
    
    resume_als = params.get('resume_als', False)
    do_als, do_transfer = True, False
    if params['policy_opt_T'].startswith('seed'): # options: fold-in, seeding, transfer 
        resume_als = True
        do_als = True
    elif params['policy_opt_T'].startswith('trans'): # transfer learned factors from R to T (no ALS involved) 
        do_als = False
        do_transfer = True
    elif params['policy_opt_T'] == 'transfer+seed':
        do_als = do_transfer = resume_als = True  # transfer learned factors as initial 'guess' of test item factors, followed by ALS
    else: # fold-in as default 
        do_als = True
        resume_als = False
        print('... policy_opt in test split: {policy} => do_als: True but fix the classfiier factors.'.format(policy=params['policy_opt_T']))

    if do_transfer: 
        X = (R, T)
        F = (Pr, Qr)
        Pt, Qt = uc.transfer_factor_by_similarity(X, F, topk=1)  # ... tr(1)
        Th_err = Th_err_weighted = [0, ]  # dummy
        
    if do_als: 
        if do_transfer: 
            user_vectors, item_vectors = Pt, Qt  # factors transfered from R   ... tr(2)
            resume_als = True  # must be True because we'll only use them for initialization
        else: 
            user_vectors = Pr # learned classifier vectors
            item_vectors = None
            # resume_als is optional

        Pt, Qt, *Th_errs = ua.implicit_als(Ct, features=params['n_factors'], 
                                iterations=params['n_iter_foldin'], 
                                lambda_val=System.lambda_val,  # 0.8 by default
                                
                                label_confidence=Ct_bar, ratings=T, labels=[],

                                    user_vectors=user_vectors,   # <<< in foldin mode, fix the user factors learned from R 
                                    item_vectors=item_vectors,   # <<< only used in transfer+seed mode
                                
                                    policy=policy_opt_eff,  # <<< in tradeoff mode, reduce this to 'rating' mode
                                    message=piggyback_msg, 
                                    ret_rmse=True, 
                                        resume_als=resume_als)

        # >>> Also, predict the mean vector
        # Pt, Qt, *Th_errs = ua.implicit_als(Ct, features=params['n_factors'], 
        #                         iterations=params['n_iter_foldin'], 
        #                         lambda_val=System.lambda_val,  # 0.8 by default
                                
        #                         label_confidence=Ct_bar, ratings=T, labels=[],

        #                             user_vectors=user_vectors,   # <<< in foldin mode, fix the user factors learned from R 
        #                             item_vectors=item_vectors,   # <<< only used in transfer+seed mode
                                
        #                             policy=policy_opt_eff,  # <<< in tradeoff mode, reduce this to 'rating' mode
        #                             message=piggyback_msg, 
        #                             ret_rmse=True, 
        #                                 resume_als=resume_als)


        Th_err, Th_err_weighted = Th_errs
    
    ########################################################################################
    if do_als: 
        ne = 2
        et_pri, et_post = np.mean(Th_err[:ne]), np.mean(Th_err[-ne:])
        et_del = (et_pri-et_post)/et_pri * 100   # e.g. 0.05 -> 0.01 => (0.05-0.01)/0.05

        if not resume_als: 
            assert LA.norm(Pt-Pr) < 1e3, "Pr or user vectors should not change (at least not much)!"
        print('... (ALS 2) Complete | rmse(R) ends at: {eR} | rmse: {e1} -> {e2} (n_err={n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(eR=e_post, 
                    e1=np.mean(Th_err[:ne]), e2=np.mean(Th_err[-ne:]), 
                    e_del=et_del,
                    ew1=np.mean(Th_err_weighted[:ne]), ew2=np.mean(Th_err_weighted[-ne:]), n=len(Rh_err) ))  

    ##################################
    # P = P.todense()
    # Q = Q.todense()

    # >>> the last fold may not have the same size (n_items)
    # if P.shape[0] == n_users and Q.shape[0] == n_items: 
    #     vmap['Pe'] += P
    #     vmap['Qe'] += Q
    #     vmap['n_averaged'] +=1 

    ### Rating matrix reconstruction (a. preference scores, b. ratings)

    # set canonicalize=True because the reconstructed probabilities could be not in [0.0, 1.0]
    if params['policy_opt'].startswith('pref'):
        ## 1. (Rh, Th) as preference scores
        Tpf = uc.predict_by_preference(Pt, Qt, canonicalize=True) 
        assert Tpf.shape == T.shape
        # Rh, Th represent preference scores, not probabilities
        
        #@ 2. (Rh, Th) as ratings (using preference scores to select & mask entries)
        if params['replace']:   # if not in prediction mode, then we are in the mode of replacing bad rating values
            print("... replace bad 'ratings' by preference scores.\n...... computing new scores > policy_replace: {0}".format(params['policy_replace'])) # replace by what ? e.g. rating
            
            # reconstructed ratings: overwrite preference vectors learned earlier # 
            Pt, Qt = ua.implicit_als(Ct, ratings=T, labels=[],
                        iterations=params['n_iter'], features=params['n_factors'], 
                        policy=params['policy_replace']) # the policy tells as what we are approximating: e.g. 'rating'

            # >>> use Rh and Th only to mark 'bad scores' but use Rp, Tp as new scores 
            M = Tpf # np.hstack((Rh, Th)) if params['augmented'] else Rh
            n_zeros = np.sum(M==null_marker)
            assert n_zeros > 0
            # >>> instead of using Cui as a mask, use M (consisting of preference scores)
            #     => replace Ra: (R, T?) by reconstructed scores via latent factors (Pp, Qp)

            print("... replacement mode: replacing bad 'ratings' | (n_zeros: {0}, ratio: {1}) | dim(T): {3}".format(
                n_zeros, n_zeros/(M.shape[0]*M.shape[1]+0.0), T.shape))

            # >>> use M to select entries in Ra that are potentially "bad" and replace these entries by the values given by
            #     the latent factors: (Pp, Qp)
            Th = uc.replace(Pt, Qt, X=(M, T), canonicalize=True, 
                               fill=null_marker, predict_func=ua.predict_by_factors, name='T')
    else: 
        ### predict or replace? 
        #   if 'predict_probs': True, then we reconstruct the entire matrix matrix via the latent factors
        #   if 'replace': True, then we only replace 'bad entries' in the original rating matrix by the new approximation given by the latent factors
        if params['predict_probs']: 
            Th = ua.predict_by_factors(Pt, Qt, canonicalize=True)
        else: # reconstructing only (replace 'bad probabilities')
            # assert params['replace'] == True 
            # case 1: Cui ~ R => Th: None, reconstructed R only
            # case 2: Cui ~ np.hstack((R, T)) => reconstructed (R, T)
            print("... replacement mode: replacing bad 'ratings' | dim(T): {1}".format(params['augmented'], T.shape))
            
            # test  
            n_masked = uc.verify_mask(Ct)  
            # if algorithm_setting in [7, 8, 9, 10]:
                # assert n_masked_T == T.shape[0]*T.shape[1], "n_masked(test)={n} while in setting {case}".format(n=n_masked_T, case=algorithm_setting)

            # wherever Cui was zerioed out (low confidence), replace these entries with new estimates
            Th = uc.replace(Pt, Qt, X=(Ct, T), canonicalize=True, 
                fill=null_marker, predict_func=ua.predict_by_factors, name='T')
    # ... CF-transform T to get Th (using the classifier/user vectors learned from the training set (R))
    
    isRating = not params['policy_opt'].startswith('pref') or params['replace'] 
    div("... Completed rating matrix reconstruction > FOLD: {0} | preference scores? {1}, action='{2}'".format(
        fold, isRating, 'replace' if params['replace'] else 'predict_probs')) # predict => predict probabilities

    ### ALS evaluation (RMS)
    Th_err = ua.prediction_error(Ct, Th, Pt, Qt, fill=0)

    # prediction via preference {0, 1}
    # ua.predict_by_preference(P, Q, len(L_train), canonicalize=True)
    # print('... (test) analyzePerf(...): {output}'.format( output=analyzePerf(L_test, Th, method=MFEnsemble.get_method_id('testwmf', 'als', params=params), aggregate_func='mean', T=T, fold=fold)) )

    # for kind in kinds:
    # X: (R, T, L_train, L_test, U)
    if save:  
        # note that we shall save the data only after model selecction is completed
        div('(wmf_ensemble_iter) Output: saving (T) (size={N0}/n(total)={NT}) and reconstructed matrix (Th) (size={N}) | delta: {delta} | algorithmic setting: {s}'.format(s=algorithm_setting, 
            N0=T.shape[1], NT=n_samples, N=T.shape[1], delta=LA.norm(Th-T, 'fro')), symbol='%') 
        # MFEnsemble.save_meta_tset((Rh, Th, L_train, L_test, U), fold=fold, method=method, params=params, verbose=verbose)

        pv_t = pv_th = None
        if tMetaUsers: 
            # drop meta users 
            T, pv_t = T[:-n_meta_users], T[-n_meta_users:] 
            Th, pv_th = Th[:-n_meta_users], Th[-n_meta_users:]
            assert Th.shape[0] + pv_th.shape[0] == n_users_test
            assert pv_t.shape[1] == pv_th.shape[1] == len(L_test)

        print('... saving CF-transformed meta data')
        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)
        MFEnsemble.save_data((T, L_test, U), fold=fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data
        MFEnsemble.save_data((Th, L_test, U), fold=fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data
            
        # save predictions from the meta users (and treat these data as if they came from an aggregation method)
        if tMetaUsers: 
            # vmap['prior_mean'], vmap['posterior_mean'] = {}, {}
            predictions = {'prior': pv_t, 'posterior': pv_th}  # each prediction dataframe consists of prediction vectors from n meta_users
            for file_type, pv in predictions.items(): 
                pn = '{ftype}_{model}'.format(ftype=file_type, model='mean') # pn: predictor name
                vmap[pn] = {}
                for i, meta_user in enumerate(meta_users): # ['latent_mean', 'masked_latent_mean',]
                    y_pred, y_label = pv[i], L_test
                    vmap[pn][meta_user] = DataFrame({'prediction':y_pred,'label':y_label, 'method':meta_user, 'fold': fold}, index=range(len(y_pred)))
    
                print('... saved predictions for data type: {dtype} & meta users (n={n}) to dataframes'.format(dtype=file_type, n=len(meta_users)))
        # [todo] now generate training and test splits (dtype = 'validation' and 'prediction' respectively)
        
    ### kind: specific MF algorithm
    if not kind: kind = 'als'
    wmfMetrics, wmfCV = vmap['wmfMetrics'], vmap['wmfCV']

    # naming convention: method_id = '{prefix}_{id}_{learner_type}' # e.g. rf_bp_stacker
    aggregate_func = 'mean' 
    method_id = '{prefix}_{id}_{learner_type}'.format(prefix='wmf_%s' % kind, id=MFEnsemble.get_method_id(method, kind, params=params), learner_type=aggregate_func)
    
    ### model evaluation
    #   a. use mean proba in T vs labels as a summary score
    #   b. but we can compute a performance score for each base method (and then take the average)

    # header = MFEnsemble.header_model_evaluation # ['method', 'score', 'posterior_score', 'mean_score', ] ... used in evaluate.analyzePerf2()
     
    wmfMetrics.append( analyzePerf(L_test, Th, method=method_id, aggregate_func=aggregate_func, T=T, fold=fold, mode='mean') ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
    wmfCV.append( (L_test, uc.combiner(Th, aggregate_func=aggregate_func, T=T)) )   # use mean predictions
    vmap['wmfMetrics'], vmap['wmfCV'] = wmfMetrics, wmfCV

    ##########################################################
    # other variables that may be of interest 
    # >>> piggyback inputs
    if piggyback: 
        # args, posargs = us.arguments()
        vmap['hyperparams'] = hyperparams  # keep track of hyperparameters for convenience 
    # vmap['fold'] = fold
    ##########################################################

    print("(wmf_ensemble_iter) Ending fold: {index} at setting {case} > returning vmaps: {keys} ... (verify) ".format(index=fold, 
        case=algorithm_setting, keys=vmap.keys()))
    return vmap  # keys are the variables to return to caller: wmfMetrics, wmfCV, etc. (see initmap())

def model_select_core(data, params, param_grid, vars=['wmfCV', 'wmfMetrics', ], kind='als', dev_ratio=None, max_dev=None, n_trial=0, null_marker=0, save=False, verbose=True): 
    """

    Memo
    ----
        R = train_df.values.T  # R: users vs items
        Td = dev_df.values.T
        Tt = test_df.values.T
        U = train_df.columns.values
    """
    import utils_cf as uc
    import utils_sys as us
    from sklearn.model_selection import ParameterGrid
    from evaluate import Metrics
    import common
    import operator

    # Convert input data to rating matrix format
    ########################################################################################
    assert hasattr(data, '__iter__') and len(data) >= 1, "Invalid input: %s" % data 
    df_train_dev, *rest = data  # data is an n-tuple where 1st component is a dataframe comprising the train-dev split
    # label_td = rest[0] if len(rest) > 0 else []
    ########################################################################################

    if not param_grid: param_grid = System.param_grid  # e.g. {'n_factors': [5, 10, 20, 50, 100, 500], 'alpha': [1, 10, 100, 1000]}
    # if dev_ratio is None: dev_ratio = params['dev_ratio']
    tModelSelection = sum(1 for v in param_grid.values() if len(v) > 1) > 0 
    method = 'wmf'

    ########################################################################################
    System.display()
    cf_spec.config(project_path=ProjectPath, domain=Domain, fold_count=FoldCount, bag_count=BagCount)
    ########################################################################################
    print('(model_select_core) MS cycle #%d | param_grid: %s > trigger model selection? %s  ... (verify) #' % (n_trial, param_grid, tModelSelection))

    # [note] 'fold_count' is the fold count through which the base predictors were trained 
    # R, Td, T, L_train, L_dev, L_test, *rest = uc.to_rating_matrix_random_subsampling(dev_ratio=dev_ratio, fold_count=System.foldCount, policy='random_cv_fold', shuffle=True, return_index=True)
    # print('(verify) Fold (outer loop index): {fold}, n_trial (inner loop index): {nt} | dim(R): {dR}, dim(Td): {dTd}, dim(T): {dT}'.format(fold=fold, nt=n_trial, dR=R.shape, dTd=Td.shape, dT=T.shape))
    # U, *Ix = rest

    # >>> steps: 
    # use (R, Td), (L_train, L_dev) to select the best model 
    # R <- R + Td
    # L_train <- L_train + L_dev 
    # use (R, T), (L_train, L_test) to evaluate the final performance (and performance comparisons with other stacking models)

    ### model selection loop
    ###########################################
    scores = {}
    if not tModelSelection: 
        best_params = us.frozendict( {p: vals[0] for p, vals in param_grid.items()} )  # Nothing else to choose from given only 1 option; we are done
        scores[best_params] = 0.5  # dummy score
    else:  
        div(message='size(train_dev): {n}'.format(n=max_dev if max_dev is not None else df_train_dev.shape[0]))

        # define 'train-test split' for model selection
        ##############################################
        # N_dev = R.shape[1] + Td.shape[1]
        # if max_dev is not None and max_dev < N_dev: 
        #     # subsample_array() # (A, axis=1, ratio=0.5, max_size=None) 
        #     rtt = R.shape[1]/(Td.shape[1]+0.0)  # train-to-test ratio
        #     assert rtt > 1, "train size smaller than test size? size(R): {nR}, size(Td): {nTd} => rtt: {r}".format(nR=R.shape[1], nTd=Td.shape[1], r=rtt)
        #     max_dev_train = int( np.ceil( rtt/(rtt+1.) * max_dev ) )
        #     max_dev_test = max_dev - max_dev_train
        #     assert max_dev_test > 0
        #     print('... (verify) MS cycle #{n_trial} | Model selection with controlled sample size | total: {nd}, max_dev_train: {ndt}, max_dev_test: {ntt}) ... '.format(n_trial=n_trial, nd=max_dev, ndt=max_dev_train, ntt=max_dev_test))
        #     D_minus = (R[:,:max_dev_train], Td[:,:max_dev_test], L_train[:max_dev_train], L_dev[:max_dev_test], U)
        # else: 
        # D_minus = (R, Td, L_train, L_dev, U)
        # D_minus = (R, Td, L_train, L_dev, U) = uc.shuffle_split_data(D, ratio=dev_ratio, max_size=max_dev) # labels=labels
        ##############################################

        # save <- False to not save the data
        # options: use python thread: prefer="threads"
        models = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_ensemble_iter)(
                            data=uc.shuffle_split_data(df_train_dev, ratio=dev_ratio, max_size=max_dev), # ~> (R, Td, L_train, L_dev, U)
                                params=params, hyperparams=hyperp, vars=vars, kind=kind, null_marker=null_marker, fold=n_trial, save=False) 
                                    for i, hyperp in enumerate(ParameterGrid(param_grid)) )  
        print('... finished model selection cycle #{n_trial} | completed {n} models/combinations #\n... example: {ex}'.format(
            n_trial=n_trial, n=len(models), ex=models[0]))
        ##############################################

        for im, model in enumerate(models):  

            # >>> design: how to index into hyperparameter settings?
            # entry = MFEnsemble.get_method_id(method, params=model['hyperparams']) 
            entry = us.frozendict(model['hyperparams']) #  ... (1a) hyperparams is a dictionary
            perf = PerformanceMetrics.merge([ PerformanceMetrics.consolidate(model['wmfMetrics']), ])

            # [test] number of method should be equal to len(kinds): i.e. one WMF algorithm (kind) -> one method #
            #            dim(table): 6 (metrics) by 1 (method)
            #            n_method == n_kinds
            print('... (verify) model #{i} | dim(perf.table): {dim}, methods: {methods}, n_method:{n} =?= n_kinds: {nk}'.format(i=im, dim=perf.table.shape, methods=list(perf.table.columns), n=perf.n_methods(), nk=1) )
            print('...... Fmax: {fmax}\n...... AUC: {auc}\n'.format(fmax=np.mean(perf.table.loc['fmax']), auc=np.mean(perf.table.loc['auc'])) )
        
            if not entry in scores: scores[entry] = 0
            scores[entry] = perf.sort(metric='fmax')[0][1]  # perf.sort(metric='fmax') returns a sorted list of (methed, score)-tuples
        ### end foreach model

        print('>>> Cycle #{n_trial} | score values:\n{scores}\n'.format(n_trial=n_trial, scores=scores.values())) # should be all different!
        print( us.format_sort_dict(scores, reverse=True, padding=4, title="(result) model selection (metric: {metric})".format(metric='fmax')) )
    
        # sorted(...) returns the frozendicts with hyperparams in the order of their corresponding fmax scores 
        best_params = sorted(scores, key=scores.__getitem__, reverse=True)[0]
        print('(model_select_core) best params in trial #{nt} > n_factors: {nf}, alpha: {a}'.format(nt=n_trial+1, nf=dict(best_params)['n_factors'], a=dict(best_params)['alpha']))  # ... (1b)
        # best_score = scores[best_params]

    return best_params, scores # best params (in frozen dict) -> score (e.g. fmax)

def wmf_ensemble_model_select(params, param_grid={}, vars=['wmfCV', 'wmfMetrics', ], kind='als', n_trials=1, 
        dev_ratio=0.2, test_ratio=0.5, max_dev=None, policy_ms='freq', null_marker=0, fold=-1, unbag=False, save=True, verbose=True):
    """
    Wrapper of wmf_ensemble_iter

    Params
    ------
    save: if True, save the training data of the concluded model with the 'best params'

    dev_ratio:
    max_dev: max sample size for model selection (used to control speed)
    policy_ms: the policy by which the best parameter setting is determined
               'freq'
               'mean'
    vars: 
        note that 'vars' happens to be one of Python's built-in funciton but we do not need it here. 

    fold: fold can either represent a fold number in a CV or the index into a particular run when this subroutine is used with Parallel()


    Call
    ----
    wmf_ensemble_model_select(params, param_grid, vars=['wmfCV', 'wmfMetrics', ], kinds=['als', ], null_marker=null_marker, dev_ratio=dev_ratio)

    """ 
    import utils_cf as uc
    import utils_sys as us
    import pandas as pd
    from sklearn.model_selection import ParameterGrid
    from evaluate import Metrics
    import common   
    import operator

    if not param_grid: param_grid = System.param_grid  # e.g. {'n_factors': [5, 10, 20, 50, 100, 500], 'alpha': [1, 10, 100, 1000]}
    if dev_ratio is None: dev_ratio = params['dev_ratio']
    
    ########################################################################################
    # predicates
    tModelSelection = sum(1 for v in param_grid.values() if len(v) > 1) > 0 
    tSaveBestModel = save
    ########################################################################################

    vmap = {}
    method = 'wmf'

    ########################################################################################
    System.display()
    cf_spec.config(project_path=ProjectPath, domain=Domain, fold_count=FoldCount, bag_count=BagCount)
    ########################################################################################
    print('(verify) param_grid: %s > trigger model selection? %s' % (param_grid, tModelSelection))

    # train-dev-test split
    # train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.read_random_fold(ProjectPath, fold_count=FoldCount, dev_ratio=dev_ratio, shuffle=True)
    # note: 
    #   1. set test_ratio to 0.5 so that half of the transformed data can be saved. 
    train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.shuffle_split(ProjectPath, split_number=3, dev_ratio=dev_ratio, test_ratio=test_ratio, 
        fold_count=-1, max_size=None, random_state=None)
    if unbag:
        train_df = common.unbag(train_df, BagCount) # mean aggregates (average over all bags)
        dev_df = common.unbag(dev_df, BagCount)
        test_df = common.unbag(test_df, BagCount)
    
    # data statistics 
    n_train, n_dev, n_test = train_df.shape[0], dev_df.shape[0], test_df.shape[0]
    N = n_train + n_dev + n_test
    print('(verify) data split ratios | train: {rtr}, dev: {rd}, test: {rt}'.format(rtr=n_train/(N+0.0), rd=n_dev/(N+0.0), rt=n_test/(N+0.0)))

    # Convert input data to rating matrix format
    ########################################################################################
    R = train_df.values.T
    Td = dev_df.values.T
    T = test_df.values.T
    U = train_df.columns.values
    L_train, L_dev, L_test = train_labels, dev_labels, test_labels

    # combine train and dev split for model selection (so that each run has its own separate random splits between train and dev)
    df = pd.concat([train_df, dev_df])
    labels = np.hstack((train_labels, dev_labels))
    D_minus = (df, labels)
    Ix = (df.index, test_df.index)  # i.e. index of the combined training set (train+dev) and index of the test set
    # ... Ix: used only for the convenience of saving reconstructed data set (see wmf_ensemble_iter())
    ########################################################################################

    tMode = 'parallelize'

    ### model selection loop
    # D = kargs.get('data', ())
    bestScores = {} # Metrics(op=np.mean)
    totalCounts = {} # keep track of the number of time a particular hyperparams was chosen as the "best"
    if tMode.startswith('para'): 
        best_models = Parallel(n_jobs = -1, verbose = 1)(delayed(model_select_core)(
                                                            data=D_minus, params=params, param_grid=param_grid, vars=vars, dev_ratio=dev_ratio, max_dev=max_dev, kind=kind, null_marker=null_marker, n_trial=n_trial, save=False) 
                                                                for n_trial in range(n_trials) ) 
        assert len(best_models) == n_trials
        for i, best_model in enumerate(best_models):  # foreach 'best model in a trial'
            best_params, scores = best_model  # scores is a map from best_params to its performance score (e.g. fmax)

            if not best_params in bestScores: bestScores[best_params] = []  # best_params is a frozendict
            bestScores[best_params].append(scores[best_params])
        print("... (verify) all 'best_params' after {n} ms-cycles: {list}".format(n=n_trials, list=bestScores.keys()))
    else: 
        print('(wmf_ensemble_model_select) Cycle #{c}, to run {n} model-selection routine ...'.format(c=fold, n=n_trials))
        for n_trial in range(n_trials): 

            # [note] 'fold_count' is the fold count through which the base predictors were trained 
            R, Td, T, L_train, L_dev, L_test, *rest = uc.to_rating_matrix_random_subsampling(dev_ratio=dev_ratio, 
                fold_count=System.foldCount, policy='random_cv_fold', shuffle=True, return_index=True, unbag=System.unbag, bag_count=BagCount)
            print('(verify) Fold (outer loop index): {fold}, n_trial (inner loop index): {nt} | dim(R): {dR}, dim(Td): {dTd}, dim(T): {dT}'.format(fold=fold, nt=n_trial, dR=R.shape, dTd=Td.shape, dT=T.shape))
            U, *Ix = rest

            # >>> steps: 
            # use (R, Td), (L_train, L_dev) to select the best model 
            # R <- R + Td
            # L_train <- L_train + L_dev 
            # use (R, T), (L_train, L_test) to evaluate the final performance (and performance comparisons with other stacking models)

            ### model selection loop
            ###########################################
            scores = {}
            if not tModelSelection: 
                best_params = us.frozendict( {p: vals[0] for p, vals in param_grid.items()} )  # Nothing else to choose from given only 1 option; we are done
                scores[best_params] = 0.5  # dummy score
            else:  
                
                # define 'train-test split' for model selection
                ##############################################
                N_dev = R.shape[1] + Td.shape[1]
                if max_dev is not None and max_dev < N_dev: 
                    # subsample_array() # (A, axis=1, ratio=0.5, max_size=None) 
                    rtt = R.shape[1]/(Td.shape[1]+0.0)  # train-to-test ratio
                    assert rtt > 1, "train size smaller than test size? size(R): {nR}, size(Td): {nTd} => rtt: {r}".format(nR=R.shape[1], nTd=Td.shape[1], r=rtt)
                    max_dev_train = int( np.ceil( rtt/(rtt+1.) * max_dev ) )
                    max_dev_test = max_dev - max_dev_train
                    assert max_dev_test > 0
                    print('(verify) Model selection with controlled sample size | total: {nd}, max_dev_train: {ndt}, max_dev_test: {ntt}) ... '.format(nd=max_dev, ndt=max_dev_train, ntt=max_dev_test))
                    D_minus = (R[:,:max_dev_train], Td[:,:max_dev_test], L_train[:max_dev_train], L_dev[:max_dev_test], U)
                else: 
                    D_minus = (R, Td, L_train, L_dev, U)
                ##############################################

                # save <- False to not save the data
                # options: use python thread: prefer="threads"
                models = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_ensemble_iter)(
                                    data=D_minus, params=params, hyperparams=hyperp, vars=vars, kind=kind, null_marker=null_marker, fold=n_trial, save=False) 
                                        for i, hyperp in enumerate(ParameterGrid(param_grid)) )  
                print('(wmf_ensemble_model_select) finished model selection iter #{n_iter} | completed {n} models #\n... example: {ex}'.format(
                    n_iter=n_trial, n=len(models), ex=models[0]))
                for im, model in enumerate(models):  
                    # model = merge(model)

                    # >>> design: how to index into hyperparameter settings?
                    # entry = tuple(model['hyperparams'][p] for p in hyperparams) 
                    # entry = MFEnsemble.get_method_id(method, params=model['hyperparams']) 
                    entry = us.frozendict(model['hyperparams']) #  ... (1a) hyperparams is a dictionary
                    # perf = PerformanceMetrics.merge([PerformanceMetrics.consolidate(model['wmfMetrics'][kind]) for kind in kinds])
                    perf = PerformanceMetrics.merge([ PerformanceMetrics.consolidate(model['wmfMetrics']), ])

                    # [test] number of method should be equal to len(kinds): i.e. one WMF algorithm (kind) -> one method #
                    #            dim(table): 6 (metrics) by 1 (method)
                    #            n_method == n_kinds
                    print('... (verify) model #{i} | dim(perf.table): {dim}, methods: {methods}, n_method:{n} =?= n_kinds: {nk}'.format(i=im, dim=perf.table.shape, methods=list(perf.table.columns), n=perf.n_methods(), nk=1) )
                    print('...... Fmax: {fmax}\n...... AUC: {auc}\n'.format(fmax=np.mean(perf.table.loc['fmax']), auc=np.mean(perf.table.loc['auc'])) )
                
                    if not entry in scores: scores[entry] = 0
                    scores[entry] = perf.sort(metric='fmax')[0][1]  # perf.sort(metric='fmax') returns a sorted list of (methed, score)-tuples
                ### end foreach model

                # print('... (verify) scores:\n%s\n' % scores)
                print( us.format_sort_dict(scores, reverse=True, padding=4, title="(result) model selection (metric: {metric})".format(metric='fmax')) )
            
                # sorted(...) returns the frozendicts with hyperparams in the order of their corresponding fmax scores 
                best_params = sorted(scores, key=scores.__getitem__, reverse=True)[0]
                print('...... best params in trial #{nt} > n_factors: {nf}, alpha: {a}'.format(nt=n_trial+1, nf=dict(best_params)['n_factors'], a=dict(best_params)['alpha']))  # ... (1b)

            # bestScores.add((best_params, scores[best_params]))
            if not best_params in bestScores: bestScores[best_params] = []  # best_params is a frozendict
            bestScores[best_params].append(scores[best_params])
            
            if not tModelSelection: break
            #############################################
        ### end foreach model selection iteration
    
    ##########################################################################################
    # ... at this point, we know the best hyperparams in 'bestScores'
    M = Metrics(bestScores, op=np.mean) 
    print("(model_select_core) Cycle: {fold} | found {n} sets of 'best params' with {nv} scores ... (verify) #".format(fold=fold, n=M.size(), nv=M.size_bags())) # len(next(iter(bestScores.values())))
    M.display(by='freq')  # display multibags

    print('(model_select_core) now take the average of the scores for each parameter setting ...')
    M_mean = M.aggregate(by='mean')  # aggregate performance scores via 'op'; if op is a mean function, then this equates to taking the average
    # M.sort(by='mean') # ... return a sorted list of 2-tuples
    
    title = "(result) Model selection performance ordering (via policy {pms}) in Cycle (outer): {fold} | n_trials={nt}, metric={metric} ...".format(pms='mean', fold=fold, nt=n_trials, metric='fmax')
    print( us.format_sort_dict(M_mean, reverse=True, padding=5, title=title)) # symbol='#', border=1

    M_freq = M.aggregate(by='freq') # M.sort(by='freq')  # aggregate performance scores via 'op'; if op is a mean function, then this equates to taking the average
    title = "(result) Model selection performance ordering (via policy {pms}) in Cycle (outer): {fold} | n_trials={nt}, metric={metric} ...".format(pms='freq', fold=fold, nt=n_trials, metric='fmax')
    print( us.format_sort_dict(M_freq , reverse=True, padding=5, title=title)) # symbol='#', border=1

    # >>> select 'best of the best'
    ############################################
    # use frequency or average? frequency is more stable
    if policy_ms.startswith('freq'):
        models = M.sort(by='freq')[0] 
    else: 
        models = M.sort(by='mean')[0]
    # best_params = sorted(models, key=models.__getitem__, reverse=True)[0] 
    best_params, best_score = models[0], models[1]
    # best_score = models[best_params]
    best_params = dict(best_params) # remember to defrost
    div(message='(result) Cycle {index} | Best of the best params across n_trials={nt} > n_factors: {nf}, alpha: {a} | policy_ms={pms} ... (verify) #'.format(index=fold, nt=n_trials, 
        nf=best_params['n_factors'], a=best_params['alpha'], pms=policy_ms), symbol='#', border=2)  # ... (1c)
    ############################################
    # ... now, narrow down to only a single 'best' model


    # use D instead of D_minus to train the final model
    ####################################################################################
    # ... train the final model using the best parameters obtained from the dev set
    D = (np.hstack((R, Td)), T, np.hstack((L_train, L_dev)), L_test, U) # + Ix   # in (R, T, L, Lt, U)-format
    # Ix: training set index (train_df + dev_df i.e. D_plus) and test set index   
    params = {**params, **best_params}  # update params by best_params

    # note: nth 'fold' really means the nth cycle of running wmb_ensemble_model_select() here
    finalModel = wmf_ensemble_iter(data=D, params=params, vars=vars, indices=Ix, kind=kind, null_marker=null_marker, fold=fold, piggyback=False, save=tSaveBestModel) # set piggyback to False to avoid returning additional variables as side effects (e.g. hyperparams)
    ####################################################################################

    # print('... (verify) n_vars in finalModel: {nv}\n... {list}\n'.format(nv=len(finalModel), list=finalModel.keys())) # ... ok, two vars as indicated by the function arg: vars
    # perf = PerformanceMetrics.merge([PerformanceMetrics.consolidate(finalModel['wmfMetrics'][kind]) for kind in kinds])
    perf = PerformanceMetrics.merge([PerformanceMetrics.consolidate(finalModel['wmfMetrics']), ] )
    # print("... (verify) how many best/final models? n={n} =?= n_kinds: {nk}".format(n=perf.n_methods(), nk=1 ))

    assert sum(1 for v in vars if not v in finalModel) == 0, "missing metric variables? finalModel keys: %s" % finalModel.keys()

    # >>> remove side effects (todo)
    # for v in ['hyperparams', 'fold', ]: 
    #     if v in finalModel: 
    #         finalModel.pop(v) 
    vmap.update(finalModel)  # wmfCV, wmfMetrics, (+ hyperparams, fold)

    ########################################################################################
    # ... additional variables  # [todo]
    model_entry = 'models'
    assert not model_entry in finalModel, "model var name conflicts with existing vars: {vars}".format(vars=list(finalModel.keys()))
    vmap[model_entry] = []
    kinds = [kind, ]
    for kind in kinds: 
        best_model = MFEnsemble.get_method_id(method, kind=kind, params=params)  # base method + specific kind
        # vmap[frozenset(best_params)] = perf.sort(metric='fmax')[0][1]  # frozenset(hyperparams) -> score
        entry = {'name': best_model, 'best_params': best_params, 'score': perf.sort(metric='fmax')[0][1]}
        vmap[model_entry].append(entry) # (best_model, perf.sort(metric='fmax')[0][1])
        div('(wmf_ensemble_model_select) Cycle {fold} | final model (kind={algo}) | name: {name}, hyperparams: {params}, score: {s} ... (verify) #'.format(
            fold=fold, algo=kind, name=entry['name'], params=entry['best_params'], s=entry['score']))
    ########################################################################################

    div('Model seletion iteration (cycle: {n_iter}) complete ... (verify)'.format(n_iter=fold), symbol='=', border=2)
    return vmap  # keys: wmfCV, wmfMetrics, {model}

def wmf_ensemble_fold(fold, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', null_marker=0, verbose=1, 
        project_path='?', save=False, piggyback=True, dev_ratio=0.2, max_dev=None, unbag=False): 
    def initmap(): 
        vmap = {} 
        for var in vars:
            vmap[var] = [] # {k: [] for k in kinds}   # var -> kind -> a list
        return vmap
    def get_class_stats(labels, pos_label=1, neg_label=0, ratio_imbalanced=0.1):
        # if fold == 0:  # fold is defined in a more generic sense: i) cv fold ii) the index of iterations in random subsampling iii) other iteration index
        
        # keys: n_pos, n_neg, r_pos, r_neg, 0/neg_label, 1/pos_label, is_balanced, r_minority, min_class
        stats = uc.classPrior(labels, labels=[neg_label, pos_label], ratio_ref=ratio_imbalanced)
        div('(result) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2): 
            assert params['supervised']
        if params['setting'] in (3, 4): 
            assert not params['supervised']
        if params['setting'] in (7, 8): 
            assert params['predict_probs'], "Setting 7 and 8 should attempt to re-estimate the entire T"

    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us

    method = params.get('method', 'wmf')
    vmap = initmap() # each call to this routine produces a new result

    # Note
    # if we don't re-configure cf_spec, then the information from the initial configuration (from sysConfig) 
    # will be lost => each thread sees its own version of cf_spec and its class definitions 
    ########################################################################################
    n_fold = FoldCount
    cf_spec.config(project_path=ProjectPath, domain=Domain, fold_count=FoldCount, bag_count=BagCount)
    ########################################################################################
    algorithm_setting = params.get('setting', '?') # see System.descriptions
    verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    ### Confidence Matrix
    # fold: CV fold number
    # data: (R, T, L, Lt, U)-format
    R, T, L_train, L_test, U = uc.to_rating_matrix(fold, unbag=unbag, bag_count=BagCount)  # other params: unbag
    n_samples = R.shape[1]+T.shape[1]; assert len(L_train)+len(L_test) == n_samples

    ### train model and estimate test
    #   Cui_bar is only used in policy = 'tradeoff'
    Cr, Cr_bar = uc.evalConfidenceMatrix(R, L=L_train, U=U,  
            ratio_users=params['ratio_users'], 
            ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< determines the subroutine for computing Cui
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], alpha=params['alpha'], 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, fold=fold) # project_path=System.projectPath 
    assert Cr.shape == R.shape
    n_users, n_items = Cr.shape[0], Cr.shape[1]

    # assert Cui.shape == Cui_bar.shape
    
    div("... Completed conficence matrix for training data C(R) | Cycle {0} | n_factors: {1}, alpha: {2} | dim(Cui): {3} | conf: {4} optimization: {5} | predict ALL probabilities? {6}".format(fold, 
        params['n_factors'], params['alpha'], str(Cr.shape), params['policy'], params['policy_opt'], params['predict_probs']), symbol='#', border=1)
    print('...... n_train: {ntr}, n_test: {nt}, ratio: {r}'.format(ntr=len(L_train), nt=len(L_test), r=(len(L_train)+len(L_test))/(len(L_test)+0.0)) )
    
    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} > FOLD: {f}'.format(dim=R.shape, f=fold))
    piggyback_msg = "+ thread/fold: {index}".format(index=fold).rjust(5)   # keep track of the thread from which the iteration 
    if params['policy'].startswith('trade'): 
        # Cui, Cui_bar are not in sparse format
        assert Cr_bar is not None
    else: 
        assert Cr_bar is None
    
    ########################################################################################
    Pr, Qr, Rh_err = ua.implicit_als(Cr, features=params['n_factors'], iterations=params['n_iter'],
                            label_confidence=Cr_bar, ratings=R, labels=L_train,
                            policy=params['policy_opt'], message=piggyback_msg, ret_rmse=True)
    ########################################################################################
    ne = 1
    print('... (ALS 1) Complete | rmse: {e1} -> {e2} (n_errs: {n}) ... (verify) #'.format(e1=np.mean(Rh_err[:ne]), e2=np.mean(Rh_err[-ne:]), n=len(Rh_err) ))

    # e.g. 50 users, 3979 items with nf=100 > dim(P): (50, 100), dim(Q): (3979, 100)
    assert Pr.shape[0] == R.shape[0] and Pr.shape[1] == params['n_factors']
    assert Qr.shape[0] == R.shape[1] and Qr.shape[1] == params['n_factors']

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    # now predict the test set using the user factors learned from the training data
    Ct, Ct_bar = uc.evalConfidenceMatrix(T, L=[], U=U,  
            ratio_users=params['ratio_users'], 

            # parameters to be used for unsupervised mode
            ratio_small_class=class_stats['r_minority'], 
            factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< determine the dimension of filtering (user, item)
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=False,  # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], alpha=params['alpha'], 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],
                        fill=null_marker, fold=fold, L_true=L_test) # project_path=System.projectPath 
    assert Ct.shape == T.shape
    n_users_test, n_items_test = Ct.shape[0], Ct.shape[1]
     
    # use Qr in T 
    div('... (ALS 2) Going into the ALS loop on TEST data (T): {dim}  > FOLD: {f}'.format(dim=T.shape, f=fold)) 
    ########################################################################################
    Pt, Qt, Th_err = ua.implicit_als(Ct, features=params['n_factors'], iterations=params['n_iter_foldin'], 
                    label_confidence=Ct_bar, ratings=T, labels=[],

                        user_vectors=Pr,   # <<< fix the user factors learned from R 
                            policy=params['policy_opt'], message=piggyback_msg, ret_rmse=True) 
    ########################################################################################
    ne = 1 
    assert LA.norm(Pt-Pr) < 1e3, "Pr or user vectors should not change (at least not much)!"
    print('... (ALS 2) Complete | rmse: {e1} -> {e2} (n_err={n}) ... (verify) #'.format(e1=np.mean(Th_err[:ne]), e2=np.mean(Th_err[-ne:]), n=len(Rh_err) ))  

    div("... Completed conficence matrix for test data C(T) | Cycle {0} | n_factors: {1}, alpha: {2} | dim(Ct): {3} | conf: {4} optimization: {5} | predict ALL probabilities? {6}".format(fold, 
        params['n_factors'], params['alpha'], str(Ct.shape), params['policy'], params['policy_opt'], params['predict_probs']), symbol='#', border=1)
    
    ##################################
    # P = P.todense()
    # Q = Q.todense()

    # >>> the last fold may not have the same size (n_items)
    # if P.shape[0] == n_users and Q.shape[0] == n_items: 
    #     vmap['Pe'] += P
    #     vmap['Qe'] += Q
    #     vmap['n_averaged'] +=1 

    ### Rating matrix reconstruction (a. preference scores, b. ratings)

    # set canonicalize=True because the reconstructed probabilities could be not in [0.0, 1.0]
    if params['policy_opt'].startswith('pref'):
        ## 1. (Rh, Th) as preference scores
        Tpf = uc.predict_by_preference(Pt, Qt, canonicalize=True) 
        assert Tpf.shape == T.shape
        # Rh, Th represent preference scores, not probabilities
        
        #@ 2. (Rh, Th) as ratings (using preference scores to select & mask entries)
        if params['replace']:   # if not in prediction mode, then we are in the mode of replacing bad rating values
            print("... replace bad 'ratings' by preference scores.\n...... computing new scores > policy_replace: {0}".format(params['policy_replace'])) # replace by what ? e.g. rating
            
            # reconstructed ratings: overwrite preference vectors learned earlier # 
            Pt, Qt = ua.implicit_als(Ct, ratings=T, labels=[],
                        iterations=params['n_iter'], features=params['n_factors'], 
                        policy=params['policy_replace']) # the policy tells as what we are approximating: e.g. 'rating'

            # >>> use Rh and Th only to mark 'bad scores' but use Rp, Tp as new scores 
            M = Tpf # np.hstack((Rh, Th)) if params['augmented'] else Rh
            n_zeros = np.sum(M==null_marker)
            assert n_zeros > 0
            # >>> instead of using Cui as a mask, use M (consisting of preference scores)
            #     => replace Ra: (R, T?) by reconstructed scores via latent factors (Pp, Qp)

            print("... replacement mode: replacing bad 'ratings' | (n_zeros: {0}, ratio: {1}) | dim(T): {3}".format(
                n_zeros, n_zeros/(M.shape[0]*M.shape[1]+0.0), T.shape))

            # >>> use M to select entries in Ra that are potentially "bad" and replace these entries by the values given by
            #     the latent factors: (Pp, Qp)
            Th = uc.replace(Pt, Qt, X=(M, T), canonicalize=True, 
                               fill=null_marker, predict_func=ua.predict_by_factors, name='T')
    else: 
        ### predict or replace? 
        #   if 'predict_probs': True, then we reconstruct the entire matrix matrix via the latent factors
        #   if 'replace': True, then we only replace 'bad entries' in the original rating matrix by the new approximation given by the latent factors
        if params['predict_probs']: 
            Th = ua.predict_by_factors(Pt, Qt, canonicalize=True)
        else: # reconstructing only (replace 'bad probabilities')
            # assert params['replace'] == True 
            # case 1: Cui ~ R => Th: None, reconstructed R only
            # case 2: Cui ~ np.hstack((R, T)) => reconstructed (R, T)
            print("... replacement mode: replacing bad 'ratings' | dim(T): {1}".format(params['augmented'], T.shape))
            
            # test  
            n_masked = uc.verify_mask(Ct)  
            # if algorithm_setting in [7, 8, 9, 10]:
                # assert n_masked_T == T.shape[0]*T.shape[1], "n_masked(test)={n} while in setting {case}".format(n=n_masked_T, case=algorithm_setting)

            # wherever Cui was zerioed out (low confidence), replace these entries with new estimates
            Th = uc.replace(Pt, Qt, X=(Ct, T), canonicalize=True, 
                fill=null_marker, predict_func=ua.predict_by_factors, name='T')
    # ... CF-transform T to get Th (using the classifier/user vectors learned from the training set (R))
    
    isRating = not params['policy_opt'].startswith('pref') or params['replace'] 
    div("... Completed rating matrix reconstruction > FOLD: {0} | preference scores? {1}, action='{2}'".format(
        fold, isRating, 'replace' if params['replace'] else 'predict_probs')) # predict => predict probabilities

    ### ALS evaluation (RMS)
    Th_err = ua.prediction_error(Ct, Th, Pt, Qt, fill=0)

    # prediction via preference {0, 1}
    # ua.predict_by_preference(P, Q, len(L_train), canonicalize=True)
    # print('... (test) analyzePerf(...): {output}'.format( output=analyzePerf(L_test, Th, method=MFEnsemble.get_method_id('testwmf', 'als', params=params), aggregate_func='mean', T=T, fold=fold)) )

    # for kind in kinds:
    # X: (R, T, L_train, L_test, U)
    if save:  
        # note that we shall save the data only after model selecction is completed
        div('(wmf_ensemble_iter) Output: saving (T) (size={N0}/n(total)={NT}) and reconstructed matrix (Th) (size={N}) | algorithmic setting: {s}'.format(s=algorithm_setting, 
            N0=T.shape[1], NT=n_samples, N=T.shape[1]), symbol='%') 
        # MFEnsemble.save_meta_tset((Rh, Th, L_train, L_test, U), fold=fold, method=method, params=params, verbose=verbose)

        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)
        MFEnsemble.save_data((T, L_test, U), fold=fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data
        MFEnsemble.save_data((Th, L_test, U), fold=fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data

        # now generate training and test splits (dtype = 'validation' and 'prediction' respectively)
        
    ### kind: specific MF algorithm
    if not kind: kind = 'als'
    wmfMetrics, wmfCV = vmap['wmfMetrics'], vmap['wmfCV']

    # naming convention: method_id = '{prefix}_{id}_{learner_type}' # e.g. rf_bp_stacker
    aggregate_func = 'mean' 
    method_id = '{prefix}_{id}_{learner_type}'.format(prefix='wmf_%s' % kind, id=MFEnsemble.get_method_id(method, kind, params=params), learner_type=aggregate_func)
    
    ### model evaluation
    #   a. use mean proba in T vs labels as a summary score
    #   b. but we can compute a performance score for each base method (and then take the average)

    # header = MFEnsemble.header_model_evaluation # ['method', 'score', 'posterior_score', 'mean_score', ] ... used in evaluate.analyzePerf2()
     
    wmfMetrics.append( analyzePerf(L_test, Th, method=method_id, aggregate_func=aggregate_func, T=T, fold=fold, mode='mean') ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
    wmfCV.append( (L_test, uc.combiner(Th, aggregate_func=aggregate_func, T=T)) )   # use mean predictions
    vmap['wmfMetrics'], vmap['wmfCV'] = wmfMetrics, wmfCV

    ##########################################################
    # other variables that may be of interest 
    # >>> piggyback inputs
    if piggyback: 
        # args, posargs = us.arguments()
        vmap['hyperparams'] = hyperparams  # keep track of hyperparameters for convenience 
    # vmap['fold'] = fold
    ##########################################################

    print("(wmf_ensemble_iter) Ending fold: {index} at setting {case} > returning vmaps: {keys} ... (verify) ".format(index=fold, 
        case=algorithm_setting, keys=vmap.keys()))
    return vmap

def wmf_ensemble(**kargs):
    """

    Memo
    ----
    1. how to merge dictionaries? 

       https://treyhunner.com/2016/02/how-to-merge-dictionaries-in-python/

    """
    def merge(vmaps): # shared_variables=['wmfMetrics', 'wmfCV', ]

        # first, figure out what are all the variables, usually each vmap references the same set of variables
        n_threads = len(vmaps)
        Vars, Kinds = [], []
        for i in range(n_threads):
            if i == 0: 
                # get first key of vmaps[i]: ith return value of wmf_ensemble routine
                # metric_vars = [v for v in vmaps[i].keys() if isinstance(vmaps[i][v], dict)]
                Vars = list(vmaps[i].keys()) # need to cast to list type because dict_keys does not support indexing
                fv = Vars[0]  
                # next(iter(vmaps[i].keys()))  # vmaps[i]: vars: 'wmfCV' -> kinds: 'als'
                
                print("(merge) variables: {k}, example: {v}... (verify) ".format(k=Vars, v=fv))
                for v in Vars: 
                    e = vmaps[i][v]
                    # print('... var:{0} => {1}'.format(v, e))

                # kinds = vmaps[i][fv].keys()
            else: 
                # model (name) may not be the same from different threads/iterations
                assert sum(1 for v in Vars if not v in vmaps[i]) == 0, "Inconsistent variables: {vars} =?= {vars0}".format(vars=list(vmaps[i].keys()), vars0=Vars)
                # Vars_i = list(vmaps[i].keys())
                # assert all([sv in Vars_i for sv in shared_variables]) # 

                # assert vmaps[i][random.sample(Vars, 1)[0]].keys() == kinds, "All metric variables should reference the same internal structure."
        print('(merge) Found {n} variables:\n... {l}\n'.format(n=len(Vars), l=Vars))
        
        # init data structure 
        vmap = initmap(Vars)

        # now merge results
        for i in range(n_threads): 
            for var in Vars: 
                # for kind in kinds: 
                if isinstance(vmaps[i][var], list): 
                    vmap[var].extend(vmaps[i][var])
                else: 
                    vmap[var].append(vmaps[i][var])
        # verify 
        for var in Vars: 
            # for kind in kinds: 
            print("(verify) size of var '{0}': {1}".format(var, len(vmap[var])) )
            print("...     after Parallel() call, wmfCV has {0} sets/cvfold, wmfMetrics has {1} sets.  #".format(len(vmap[var]), len(vmap[var])))
        return vmap
    def initmap(vars=['wmfMetrics', 'wmfCV', ]): 
        vmap = {} 
        for var in vars:  # [note] vars() in python takes an object (with __dict__ attribute) and returns a dictionary
            vmap[var] = [] # {k: [] for k in kinds}   # var -> kind -> a list
        return vmap
    def verify(vmap):
        div('Verify performance metrics ... (verify)', symbol='%')
        
        # structure 
        for k, v in vmap.items(): 
            assert isinstance(v, list), "Problematic variable structure on var=%s\n... image(vmap):\n%s\n" % (k, vmap)

        n_perfs = len(vmap['wmfMetrics'])
        for perf in vmap['wmfMetrics']: 
            print('... n_methods: {n}, columns:\n...{cols}\n'.format(n=n_perfs, cols=perf.table.columns.values))
        return  
    def eval_model_performance(multibag, dtype=frozenset, cols_params=['n_factors', 'alpha'], 
            setting=-1, 
            col_score='score', col_freq='freq', 
            output_path='', sep='|', save=True):  # refactor to evaluate
        from tabulate import tabulate
        # adict: maps a frozenset (containing model parameters) to its scores 
       
        # load old records 
        if not output_path: 
            analysis_path = os.path.join(System.projectPath, 'analysis')
            if setting > 0: 
                output_path = os.path.join(analysis_path, 'model_performance-S{case}.csv'.format(case=setting))  
            else: 
                output_path = os.path.join(analysis_path, 'model_performance.csv')  # todo

        df0 = None
        if os.path.exists(output_path):
            df0 = pd.read_csv(output_path, sep=sep, header=0, index_col=False) 
            print('... loaded old model performance (dim={dim})'.format(dim=df0.shape))

        header = cols_params + [col_score, ] # 'freq'
        adict = {h: [] for h in header}
        for i, (k, bag) in enumerate(multibag.items()): 
            assert isinstance(k, dtype), "Data type mismatch | expect {0} but got {1}".format(dtype, type(k)) # then k is a frozen dictionary of hyperparemeters                 
            
            hparams = dict(k)  # parameter dictionary
            n_scores = len(bag)  # number of scores recorded
            for col in cols_params: 
                adict[col].extend( [hparams[col]] * n_scores)
            adict[col_score].extend(bag)
            # adict['freq'].extend([n_scores] * n_scores)

        df = DataFrame(adict, columns=header)
        if df0 is not None and not df0.empty: 
            df = pd.concat([df0, df], ignore_index=True)

        # calculate & sort frequencies
        df = analyze_model_performance(df=df, cols_params=cols_params, col_score=col_score, col_freq=col_freq, save=False)

        # output
        ##############################################
        if save: 
            df.to_csv(output_path, index=False, sep=sep)
            print('... saved performance scores from model selection to:\n{path}\n'.format(path=output_path))
        ##############################################

        n_display = 10
        print( tabulate(df.head(n_display), headers='keys', tablefmt='psql') )

        return df
    def analyze_model_performance(df=None, path='', cols_params=['n_factors', 'alpha'], col_score='score', col_freq='freq', seq='|', 
            setting=-1, 
            save=False):
        from tabulate import tabulate
        
        # ref: https://stackoverflow.com/questions/35268817/unique-combinations-of-values-in-selected-columns-in-pandas-data-frame-and-count

        # I/O 
        if not path: 
            analysis_path = os.path.join(System.projectPath, 'analysis')
            if setting > 0: 
                output_path = os.path.join(analysis_path, 'model_performance-S{case}.csv'.format(case=setting))  
            else: 
                output_path = os.path.join(analysis_path, 'model_performance.csv')  # todo

        # load records
        if df is None:  
            assert os.path.exists(path), "No model performance data have been generated yet!"
            df = pd.read_csv(path, sep=sep, header=0, index_col=False) 
            print('... loaded old model performance (dim={dim}) from:\n{path}\n'.format(dim=df.shape, path=path))

        # populate and count the parameter combinations 
        # df_count = df.groupby(cols_params).size().reset_index().rename(columns={0: col_freq})
        # ... this only has ['n_factors', 'alpha', 'freq'] but missing 'score'

        # df[col_freq] = 0
        # for i, row in df_count.iterrows():
        #     df.loc[ (df.n_factors == row['n_factors']) & (df.alpha == row['alpha']), [col_freq]] = row[col_freq]

        dfs = []
        for params, dfg in df.groupby(cols_params):  # [note] params is a tuple
            dfg[col_freq] = dfg.shape[0]
            dfs.append(dfg)
        df = pd.concat(dfs, ignore_index=True)

        # sort 
        df = df.sort_values(by=[col_freq, ], ascending=False)

        # output
        ##############################################
        if save: 
            df.to_csv(path, index=False, sep=sep)
            print('... saved performance << score + freq >> to:\n{path}\n'.format(path=path))
        ##############################################

        print( tabulate(df.head(10), headers='keys', tablefmt='psql') )
            
        return df 
            
    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    # from utils_sys import format_sort_dict

    n_fold = System.foldCount
    missing_value = null_marker = 0 # marker for missing data
    p_th = 0.5
    n_users = n_items = 0 
    
    ### Algorithm parameters 
    #   a. parameters 
    #   b. meta parameters
    ####################################################################################################################################
    params = {}
    
    method = params['method'] = 'wmf'
    algorithm_setting = params['setting'] = kargs.get('setting', 11) # System.options.setting

    tMetaUsers = params['include_meta_users'] = kargs.get('include_meta_users', False)  # if True, wmf_ensemble routine will add meta classifiers in R and T

    # note: {n_factors, alpha} can be affected by command args, use System's default instead of MFEnsemble's default
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)  
    params['alpha'] = alpha_val = kargs.get('alpha', MFEnsemble.alpha)

    # >>> not the same as 'n_runs'
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', System.n_epochs)
    params['n_iter_foldin'] = params['n_epochs_foldin'] = kargs.get('n_epochs_foldin', System.n_epochs_foldin)

    # dev set ration 
    params['dev_ratio'] = 1./System.foldCount
    
    # parameters for confidence matrix
    params['conf_measure'] = kargs.get('conf_measure', 'brier')  # confidence matrix

    params['ratio_small_class'] = kargs.get('ratio_small_class', -1)  # <=0: use minority class ratio as default
    params['ratio_users'] = kargs.get('ratio_users', 0.5)  # select top k entries of R (through Cui) to approximate, only relevant when policy in {'user', 'item'}
    
    params['supervised'] = kargs.get('supervised', False)
    
    params['augmented'] = kargs.get('augmented', True)  # approximate R or both R & T? 
    
    unbag = params['unbag'] = kargs.get('unbag', False)
    params['resume_als'] = kargs.get('resume_als', False) # in ALS fold-in, use the learned factor vector as an init. or fix it so that ALS reduces to LS?
    #################################################################
    # Do we re-estimate the entire rating table or only the entries marked unreliable? (i.e. replace <- True)
    params['predict_probs'] = kargs.get('predict_probs', False)  # reconstruct the entire R or T (as opposed to 'replace' mode below)
    params['replace'] = kargs.get('replace', not params['predict_probs'])  # replace bad ratings or other scores (e.g. probabilities)
    #################################################################

    params['masked'] = kargs.get('masked', True) # if True, mask FP and FN, this makes Cui 'sparse'; if False, turn masking off 
    params['mask_all_test'] = kargs.get('mask_all_test', False) # mask all entries in T or not? only relevant when 'augmented' is True

    # params['p_threshold'] = p_th  # 'ratio'-based confidence matrix depends on threshould
    # params['delta'] = 0

    # parameters for ALS methods 
    params['policy_filter'] = params['policy'] = kargs.get('policy', 'item')  # II: {'item', 'user'}
    params['policy_filter_test'] = params['policy_test'] = kargs.get('policy_test', params['policy'])
    params['policy_opt'] = kargs.get('policy_opt', 'rating') # options: I {'rating', 'preference', 'tradeoff'}, 
    params['policy_opt_T'] = kargs.get('policy_opt_T', 'foldin')  # how the factors in test set are derived {'foldin', 'seeding', 'transfer', 'transfer+seed'}
    params['policy_replace'] = kargs.get('policy_replace', 'rating') # used only when policy_opt <- preference AND replace <- True
    params['policy_threshold'] = kargs.get('policy_threshold', 'prior')  # how to determine prob threshold? {'fmax', 'prior'/'topk', }
    
    policy_iter = kargs.pop('policy_iter', 'cv') # policy for train-dev-test iterations (nested CV, CV, randon subsampling)
    policy_ms = kargs.get('policy_ms', 'freq')   # the policy for determining the best model; tricky upon large variance, empiricially 'freq' may work better
    
    params['dev_ratio'] = kargs.get('dev_ratio', 1./System.foldCount)
    param_grid = kargs.pop('param_grid', {'n_factors': [5, 10, 20, 50, 100, 500], 'alpha': [1, 10, 100, 1000]})
    n_runs = kargs.pop('n_runs', 10)  # number of runs of random subsampling, only relevant when 'policy_iter' is 'subsampling'
    n_runs_modelselect = kargs.pop('n_runs_modelselect', 10)
    
    max_dev = kargs.get('max_dev', None) # by default use all train-dev split to do model selection but sometimes, we may want to control the sample size to save time
    
    # predicates
    tPlotROC = kargs.get('plot_roc', True)
    tModelSelection = len(param_grid) > 0 and sum(1 for v in param_grid.values() if len(v) > 1) > 0
    tSaveBestModel = kargs.get('save_best_model', True)
    if not tModelSelection: 
        n_runs_modelselect = 1
    ####################################################################################################################################

    ret = {} # <<< output 
    # ret['best_params'] = {}
    ########################################

    perfMetrics = []

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models

    # BP performance metrics are prefereable factored out of this routine, compute BP performance in _suite() routine instead
    baseline = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', False) else {}
    # perfMetrics.append(baseline['metrics'])
    # ALS_Routine = kargs.get('als_routine', implicit_als)

    kinds = ['als', ]
    kind = kinds[0]
    Vars = ['wmfCV', 'wmfMetrics', ]

    ############################################
    # >>> main loop
    # 'vmaps' is a list of return values (vmap) from the main process: wmf_ensemble_* (e.g. wmf_ensemble_fold, wmf_ensemble_iter)
    #    a. vmap = init(vars=['wmfMetrics', 'wmfCV', ], kinds=['als', ])
    #    b. [design] pass variables and the 'protocol' function that gives instructions of how to build necessary data structures for these variables 
    #                e.g. vars <- ['wmfCV', ], protocol <- init()
    ############################################
    # mf_routine=wmf_ensemble_fold
    ####################################################################################################################################
    if policy_iter == 'cv': 
        # R, T, L_train, L_test, U = uc.to_rating_matrix(fold, unbag=System.unbag, bag_count=BagCount)  # other params: unbag
        
        vmaps = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_ensemble_fold)(fold, params, vars=Vars, kind=kind, null_marker=null_marker) for fold in range(n_fold))

    elif policy_iter == 'seq': 
        ### sequential
        # R, T, L_train, L_test, U = uc.to_rating_matrix(fold, unbag=System.unbag, bag_count=BagCount)  # other params: unbag
        vmaps = []
        for fold in range(n_fold): 
            vmaps.append( wmf_ensemble_fold(fold, params, vars=Vars, kind=kind, null_marker=null_marker, verbose=1) ) 
    else: # random subsampling
        ### consider model selection or not? 
        div("Running model selection (n_runs:{n} * n_runs_modelselect:{nm} = n_total_run:{nt}) with {policy} on {grid} ... (verify)".format(n=n_runs, 
            nm=n_runs_modelselect, nt=n_runs * n_runs_modelselect, policy=policy_iter, grid=param_grid), symbol='#')  # ... ok 

        # R, Td, T, L_train, L_dev, L_test, U = uc.to_rating_matrix_random_subsampling(dev_ratio=dev_ratio, fold_count=System.foldCount, policy='random_cv_fold')

        # [note] 1. vmaps is a collection of the outputs from multiple threads => vmaps is a list
        #        2. use threads: prefer="threads"
        # suppose that n_runs = 10, n_runs_modelselect = 5
        # wmf_ensemble_model_select() is to run 10 times, each of which runs model-selection routine 5 times
        vmaps = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_ensemble_model_select)(params, param_grid, n_trials=n_runs_modelselect,
            vars=Vars, kind=kind, null_marker=null_marker, 
                dev_ratio=0.2, test_ratio=0.5, 
                    policy_ms=policy_ms,   # policy for model selection (e.g. frequency, average, )
                    fold=nr, max_dev=max_dev, save=tSaveBestModel, unbag=unbag) for nr in range(n_runs))
        #     vmaps['wmfMetrics'] = [perf1, perf2, ... ]
        # ... each wmf_ensemble_model_select() call should only produce a single 'best' model => N calls, N models

        #######################################################
        # ... model selection complete
    ####################################################################################################################################

    # combine parallelized results 
    vmap = merge(vmaps)  # variables: wmfCV, wmfMetrics, hyperparams, fold
    verify(vmap)
    # Pe = vmap['Pe']/vmap['n_averaged']
    # Qe = vmap['Qe']/vmap['n_averaged']
    # if kargs.get('save_factors', False): save_factors(Pe, Qe) # use these to compute the similarity matrix

    #################################################################
    # ... initiate model evaluation
    assert len(vmap['models']) == n_runs, "number of runs (of the model selection procedure): {n}, but got {nm} models".format(n=n_runs, nm=len(vmap['models']))
    M = Metrics(op=np.mean)
    for entry in vmap['models']:  # should have 'n_runs' number of entries
        # name, score, *others = entry
        name, score = entry['name'], entry['score'] # entry['best_params']
        assert MFEnsemble.isAMethodName(name), "Dubious model name: %s" % name
        M.add( (name, score) )  

    ####################################################################################################################################
    # ... in the case of model selection, need to verify which hyperparameter setttings are considered the best among all
    if n_runs > 1 or n_runs_modelselect > 1: 
        # models = [pair[0] for pair in vmap['models'] if MFEnsemble.isAMethodName(pair[0])] # if MFEnsemble.isAMethodName(pair[0]) 

        if tModelSelection:
            # models = M.records     
            print("(wmf_ensemble) Found {n} sets of 'best params' with {ns} scores ... (verify) #".format(n=M.size(), ns=M.size_bags()))
            M.display(by='raw')  # display multibags

            # n_runs: number of iterations involving model selection process (which itself comprises n cycles, where n = n_runs_modelselect)
            assert M.size_bags() == n_runs, "n_runs: %d whereas number of models returned from model selection: %d" % (n_runs, M.size_bags())

            # [test]
            # sort in terms of frequencies
            freq_models = M.sort_by_freq()
            print( us.format_sort_dict(dict(freq_models), reverse=True, padding=5, title="(result) Sorted frequent 'best parameters' using metric={metric} ...".format(metric='fmax')) ) # symbol='#', border=1

            mean_models = M.aggregate()
            print( us.format_sort_dict(mean_models, reverse=True, padding=5, title="(result) Average performance of 'best parameters' using metric={metric} ...".format(metric='fmax')) ) # symbol='#', border=1
            
        #######################################################
        
        M = Metrics()
        for entry in vmap['models']:
            best_params, score = entry['best_params'], entry['score'] 
            M.add( (us.frozendict(best_params), score) )
        ret['best_params'] = dict(M.sort(by=policy_ms)[0][0])  # sort by aggregate function defined in Metrics constructor

        if tModelSelection:  # save the parameter and its score for later use ... [todo] refactor to 'evaluate' module
            eval_model_performance(M.records, dtype=frozenset, cols_params=['n_factors', 'alpha'], 
                col_score='score', col_freq='freq', save=True, setting=params.get('setting', -1))

        div("(verify) Best parameter setting after model selection: {m}".format(m=ret['best_params']), symbol='#', border=1)
        params = {**params, **ret['best_params']} 
    ####################################################################################################################################
    
    assert not (None in {params['n_factors'], params['alpha']}), "Null hyperparams! Provide param_grid to activate model selection or specify via --n-factors and --alpha ..." 

    # file ID: e.g. wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior
    div('(wmf_ensemble) (S) setting: {0} | (F, A) n_factors: {1}, alpha: {2} | (CF) conf_measure: {3}, policy: {4}, policy_opt: {5}'.format(
        algorithm_setting,
        params['n_factors'], params['alpha'], 
        params['conf_measure'], params['policy'], params['policy_opt']), symbol='#', border=2)
    if params['policy_opt'].startswith('pref') and params['replace']: 
        div('... Use preference scores as a masking device to replace bad rating scores or probabilities  #', symbol='%', border=2)
    # ... say we use n_factors=20 and alpha=100 as default 

    # save meta-user predictions (as if they were stacking results)
    if tMetaUsers: 
        file_types = ['prior', 'posterior', ]
        meta_users = ['latent_mean', 'masked_latent_mean', ]

        meta_method = 'mean'
        for file_type in file_types:

            pn = '{ftype}_{model}'.format(ftype=file_type, model=meta_method) # pn: predictor name
            assert pn in vmap, "Missing prediction vectors for predictor type: {name}".format(name=pn)
            
            predictions = vmap[pn] 
            for meta_user in meta_users: # ['latent_mean', 'masked_latent_mean', ] 
                dfs = [prediction[meta_user] for prediction in predictions] 
                df_prediction = pd.concat(dfs, ignore_index=True)
                dset_type = file_type if file_type.startswith(('prior', 'post')) else 'prediction'
                dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)
                output_path = '{path}/analysis/{stacker}.S-{dataset}-{suffix}.csv'.format(path=System.projectPath, 
                        stacker=meta_user, dataset=dset_id, suffix=dset_type)
            
                print('(verify) saving aggregation result | dtype: {dtype}, meta user: {name}, data/method ID: {id} | output path:\n{path}\n'.format(dtype=file_type, name=meta_user, 
                    id=dset_id, path=output_path))
                
                df_prediction.to_csv(output_path, index = False)

    ## evaluation 
    wmfCV, wmfMetrics = vmap['wmfCV'], vmap['wmfMetrics']
    # for kind in kinds: 
    #     if kind in wmfCV: 
    #         method_id = MFEnsemble.get_method_id(method, kind, params=params) 
    #         div('(wmf_ensemble) Output: Performance plot | method: %s' % method_specific, symbol='%')
    #         plot_roc(wmfCV[kind], file_name='roc-{method}-{project}'.format(method=method_id, project=System.domain))  # an import from evaluate
    if tPlotROC:
        method_specific = MFEnsemble.get_method_id(method, kind=kind, params=params) 
        div('(wmf_ensemble) Output: Performance plot | method: %s' % method_specific, symbol='%')  
        plot_roc(wmfCV, file_name='roc-{method}-{project}'.format(method=method_specific, project=System.domain))  # an import from evaluate  

    # wmfMetrics[kind] is a list of PM objects, consolidate() takes an average
    # perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(wmfMetrics[kind]) for kind in kinds])
    # [note] .consolidate() merges/averages Perf objects across CV folds (or subsampling iterations)
    #        .merge() merges multiple consolidated Perf objects across different algorithms            
    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(wmfMetrics), ])  

    # ret: output dictionary
    #      key: {methods}, {metrics}
    ret['metrics'] = perfAll
    div("(result) sorted performance metrics on %s (n_metrics=%d)" % (PerformanceMetrics.tracked, len(PerformanceMetrics.tracked)), symbol='%')
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    if baseline: 
        docs = {'method': method}
        div("(result) performance comparison with BPs ..." % metric, symbol='%')
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    #####################################################
    # output 

    # byproduct 

    #   1. model selection dataframe
    #   2. meta user predictions

    #####################################################
    return ret  # keys: metrics, model

def wmf_similarity_ensemble(data, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', fold=-1, null_marker=0, verbose=1, 
        project_path='?', save=False, piggyback=True, dev_ratio=0.2, max_dev=None):
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
    def save_array(S, kind, fold=0, method='wmf'): 
        MFEnsemble.save_array(S, file_name='{method}_{kind}_S.csv'.format(method=method, kind=kind))
    def initmap(): # variable map is a nested dictionary in which each variable has it's own keys (kind) representing subcategories of the MF algorithm
        vmap = {} 
        for var in vars:
            vmap[var] = [] # {k: [] for k in kinds}   # var -> kind -> a list
        return vmap
    def get_class_stats(labels, pos_label=1, neg_label=0, ratio_imbalanced=0.1):
        # if fold == 0:  # fold is defined in a more generic sense: i) cv fold ii) the index of iterations in random subsampling iii) other iteration index
        
        # keys: n_pos, n_neg, r_pos, r_neg, 0/neg_label, 1/pos_label, is_balanced, r_minority, min_class
        stats = uc.classPrior(labels, labels=[neg_label, pos_label], ratio_ref=ratio_imbalanced)
        div('(result) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2, 7, 8): 
            assert params['supervised']
        if params['setting'] in (3, 4, 9, 10): 
            assert not params['supervised']
        if params['setting'] in (7, 8, 9, 10): 
            assert params['predict_probs'], "Setting 7 - 10 should attempt to re-estimate the entire T"
    def verify_confidence_matrix(C, Cbar):
        if params['setting'] in (11, 12): 
            assert Cbar is not None
            assert params['policy_opt'].startswith('trade')
    def make_prediction_vector(X, L=[], M=None, policy=''):
        if not policy: policy=params['policy'] 
        if M is not None: assert len(L) == 0
        pv = uc.to_mean_vector(X, L=L, 
                M=M,  # message from training set when L is not accessible

                ratio_users=params['ratio_users'],  # filtering in the item direction 
                ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0),  # used for unsupervised mode 

                policy=policy,  # determining filtering dimension
                policy_threshold=params['policy_threshold'], # determining proba threshold

                    supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    fill=null_marker, fold=fold)
        return pv

    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    from itertools import product 

    method = params.get('method', 'wmf')
    tMetaUsers = params.get('include_meta_users', False)  # if True, add extra meta classsifiers/users in the last rows of R and T
    vmap = initmap() # each call to this routine produces a new result

    # Note
    # if we don't re-configure cf_spec, then the information from the initial configuration (from sysConfig) 
    # will be lost => each thread sees its own version of cf_spec and its class definitions 
    ########################################################################################
    n_fold = FoldCount
    cf_spec.config(project_path=ProjectPath, domain=Domain, fold_count=FoldCount, bag_count=BagCount)
    ########################################################################################
    algorithm_setting = params.get('setting', '?') # see System.descriptions
    verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    ### Confidence Matrix
    # fold: iteration number (here, 'fold' is simply a repurposed variable borrowed from CV)
    # data: (R, T, L, Lt, U)-format
    if isinstance(data, DataFrame): # in this case, 'data' is a dataframe comprising a train-dev split 
        R, T, L_train, L_test, U =  uc.shuffle_split_data(data, ratio=dev_ratio, max_size=max_dev)
    else:
        # otherwise, assuming that shuffle-split operation had been invoked to produce the data input
        assert isinstance(data, (tuple, list)), "Invalid input data: %s" % data 
        print('(wmf_similarity_ensemble) verify: len(data): {n}'.format(n=len(data)))
        R, T, L_train, L_test, U = data 
    n_samples = R.shape[1]+T.shape[1]; assert len(L_train)+len(L_test) == n_samples

    ### create extra prediction vectors (PVs) from R (say, mean vector) and attach these new PVs to R (piggyback)
    n_users, n_items = R.shape
    n_users_test, n_items_test = T.shape
    n_meta_users = 0

    ############################################################
    # piggy back extra meta-estimators
    meta_users = ['latent_mean', 'masked_latent_mean',]  # todo
    if tMetaUsers: 
        div(message='(wmf_similarity_ensemble) Adding meta users (n={n})...'.format(n=len(meta_users)))
        
        # e.g. mean, and masked mean
        nU, nUT = n_users, n_users_test

        print('... augmenting T (by meta users)')
        mean_pv = make_prediction_vector(T, L=[], policy='none')
        masked_mean_pv = make_prediction_vector(T, L=[], M=(R, L_train), policy=params['policy_test'])
        T = np.vstack((T, mean_pv, masked_mean_pv))
        n_users_test = T.shape[0]

        print('... augmenting R (by meta usrs)')
        mean_pv = make_prediction_vector(R, L_train, policy='none')
        masked_mean_pv = make_prediction_vector(R, L_train, policy=params['policy'])
        R = np.vstack((R, mean_pv, masked_mean_pv))
        n_users = R.shape[0]

        n_meta_users = n_users - nU
        assert n_meta_users == (n_users_test - nUT) == len(meta_users)
    ############################################################

    # bpMetrics, fullMetrics, topKMetrics = Metrics(), Metrics(), Metrics() # matrix factorization metrics
    perfMetrics = []
    fullMetrics, topKMetrics = [], []
    wmfCV, topKWMFCV = [], []

    clustering_methods = ['kmeans', 'spectral']  # product(*[kind[:2], clusterings])
    kinds = ['user', 'item', 'user_topk', 'item_topk', ]
    if kargs.get('run_clustering', False): 
        kinds += ['_'.join(pair) for pair in product(*[kinds[:2], clusterings])]
    # kinds = ['user', 'item', 'user_topk', 'item_topk', 'user_kmeans', 'item_kmeans', 'user_spectral', 'item_spectral', ]

    wmfMetrics = {k: [] for k in kinds}  # a list of PerformanceMetrics objects; [old] Metrics()
    wmfCV = {k: [] for k in kinds}

    Pe = Qe = 0. 
        
    # compute confidence matrix for R
    Cr, Cr_bar = uc.evalConfidenceMatrix(R, L=L_train, U=U,  
        ratio_users=params['ratio_users'], 
        ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

        policy=params['policy'], # <<< determines the subroutine for computing Cui
        policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
        policy_threshold=params['policy_threshold'],

            supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                conf_measure=params['conf_measure'], alpha=params['alpha'], 
                    # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                    fill=null_marker, fold=fold) # project_path=System.projectPath 

    div("(clustering) Completed C(R) | Cycle {0} | n_factors: {1}, alpha: {2} | dim(Cui): {3} | conf: {4}, conf_measure: {5}, optimization: {6} | predict ALL probabilities? {7} | policy_threshold: {8}".format(fold, 
        params['n_factors'], params['alpha'], str(Cr.shape), params['policy'], params['conf_measure'], params['policy_opt'], params['predict_probs'], params['policy_threshold']), symbol='#', border=1)
    print('...... Data | n_train: {ntr}, n_test: {nt}, ratio: {r}'.format(ntr=len(L_train), nt=len(L_test), r=(len(L_train)+len(L_test))/(len(L_test)+0.0)) )
    print('...... Classifiers | n_users: {nu}, n_meta_users: {nmu} => n_original_users: {no}'.format(nu=n_users, nmu=n_meta_users, no=n_users-n_meta_users))
    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} > FOLD: {f}'.format(dim=R.shape, f=fold))
    piggyback_msg = "+ thread/fold: {index} | setting: {setting}".format(index=fold, setting=algorithm_setting).rjust(5)   # keep track of the thread from which the iteration 
    
    ########################################################################################
    Pr, Qr, *Rh_errs = ua.implicit_als(Cr, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            label_confidence=Cr_bar, ratings=R, labels=L_train,
                            policy=params['policy_opt'], message=piggyback_msg, ret_rmse=True)
    Rh_err, Rh_err_weighted = Rh_errs
    ########################################################################################
    ne = 2
    e_pri, e_post = np.mean(Rh_err[:ne]), np.mean(Rh_err[-ne:])
    e_del = (e_pri-e_post)/e_pri * 100
    print('... (ALS 1) Complete | rmse: {e1} -> {e2} (n_errs: {n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(e1=e_pri, e2=e_post, 
                   e_del=e_del, ew1=np.mean(Rh_err_weighted[:ne]), ew2=np.mean(Rh_err_weighted[-ne:]), n=len(Rh_err) ))

    # e.g. 50 users, 3979 items with nf=100 > dim(P): (50, 100), dim(Q): (3979, 100)
    assert Pr.shape[0] == R.shape[0] and Pr.shape[1] == params['n_factors']
    assert Qr.shape[0] == R.shape[1] and Qr.shape[1] == params['n_factors']

    ### create extra prediction vectors (PVs) from T (say, mean vector) and attach these new PVs to T (piggyback)

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    # given latent factors
    for kind in kinds[:2]:  # foreach user or item
        factors = P if kind == 'user' else Q
        S = uc.evalSimilarityByLatentFeatures(factors, epsilon=1e-9)
        save_array(S, kind) # fold
        
        # axis = 0 if kind == 'user' else 1
        dimS = n_users if kind == 'user' else n_items
        assert S.shape[0] == S.shape[1] == dimS
        print('... (verify) kind={0} | dim(S): {1} (S[i,j] in [0, 1]?):\n{2}\n'.format(kind, S.shape, S[:4, :4]))

        # # [note] R, T are to be merged prior to calling predict_nobias or predict_topk
        # Rh, Th = uc.predict(R, T, S=S, kind=kind)
        # assert T.shape == Th.shape, "dim(T): {0} but dim(Th): {1}".format(T.shape, Th.shape)

        # method_specific = MFEnsemble.name_sim_method(method, kind=kind, params=params) # '{base}_{kind}_sim'.format(base=base, kind=kind)
        # wmfMetrics[kind].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean))
        # wmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )  # plot K-Fold CV
        # if kargs.get('save', False): uc.save_reconstructed_probs((Rh, Th), labels=(L_train, L_test), fold=fold, method=method, verify=True, U=U)

        # # use top K only 
        # topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
        # Rh_topK, Th_topK = uc.predict(R, T, S=S, kind=kind, topk=topk)
    
        # kind_topk = '%s_topk' % kind
        # method_specific = MFEnsemble.name_sim_method(method, kind=kind_topk, params=params) # '{base}_{kind}'.format(base=base, kind=kind_topk)
        # wmfMetrics[kind_topk].append(analyzePerf(L_test, Th_topK, method=method_specific, aggregate_func=np.mean) )
        # wmfCV[kind_topk].append( (L_test, uc.combiner(Th_topK, aggregate_func=np.mean)) )
        
        ## WMF + Clustering
        for clustering in clustering_methods:  # foreach clustering method
            kind_cluster = '%s_%s' % (kind, clustering)
            method_specific = MFEnsemble.name_sim_method(method, kind=kind_cluster, params=params)

            # factors could be user-based or item-based (depending on kind)
            cluster_labels = uc.runClustering(factors, n_clusters=params['n_factors'], method=clustering) 
            # Rh, Th = uc.predict_by_cluster(R, T, similarity=S, kind=kind, C=cluster_labels)


        ### end user-item loop
    
    ### end foreach CV fold 
    
    # perf_unfold = [PerformanceMetrics.consolidate(nmfMetrics[kind]) for kind in kinds]
    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(wmfMetrics[kind]) for kind in kinds])
    ret['metrics'] = perfAll # PerformanceMetrics.consolidate(fullMetrics)  # foreach metric, take average over CV folds

    # ret['nmf_sim_top%d' % topk] = PerformanceMetrics.consolidate(topKMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    if baseline: 
        docs = {'method': method}
        PerformanceMetrics.report(p_baseline=baseline['metrics'], p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return ret

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
    from utils_als import implicit_als_cg, implicit_als
    from evaluate import plot_roc, analyzePerf, Metrics, PerformanceMetrics
    import utils_cf as uc
    from itertools import product

    n_fold = System.foldCount
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 

    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', System.n_factors)
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', System.n_epochs)
    params['alpha'] = alpha_val = kargs.get('alpha', System.alpha)

    # parameters for confidence matrix 
    params['conf_measure'] = kargs.get('conf_measure', 'brier') # {'ratio', }

    topk = 30

    # output
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    baseline = base_predictors(topk=kargs.get('topk_bps', -1), 
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
                                            fill=missing_value, verbose=True, is_augmented=True, mode=params['conf_measure'])
        n_users, n_items, n_items_total = Cui.shape[0], Cui.shape[1], len(L_train)+len(L_test)
        print('(wmf_similarity_ensemble) dim(Cui/R):{0}, dim(T): {1}, n_train: {2}, n_test: {3}'.format(
            Cui.shape, T.shape, len(L_train), len(L_test)))

        conf_data = (Cui * params['alpha']).astype('double')
        
        P, Q = implicit_als(conf_data, R=np.hstack((R, T)), iterations=params['n_iter'], features=params['n_factors'])
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
        baseline = base_predictors(topk=topk_bps, metric='fmax')
        perfMetrics.append(baseline['metrics'])

    ## how does it compare to stacking methods?
    if kargs.pop('run_bp_stacker', True):
        topk_bp_stackers = kargs.get('topk_bp_stackers', -1)
        ret = run_baseline_stacker(topk=topk_bp_stackers, metric='fmax')  # dataset='bp'
        perfMetrics.append(ret['metrics'])

    ### matrix factorization methods based on Surprise 
    if kargs.get('run_nmf_ensemble', True): 
        div(message='Running NMF approach ...', symbol='#', border=1)
        pm1 = nmf_ensemble(**kargs)  # metrics -> perf, fmax -> sorted methods
        perfMetrics.append(pm1)

    if kargs.get('run_nmf_stacker', True): 
        div(message='Running Stackers on top of NMF-reproduced trainining data ...', symbol='#', border=1)
        ret = run_stacker(dataset='nmf')
        perfMetrics.append(ret['metrics'])

    if kargs.get('run_nmf_similarity', True): 
        div(message='Running NMF-induced similarity ensemble ...', symbol='#', border=1)
        pm2 = nmf_similarity_ensemble(**kargs)
        perfMetrics.append(pm2)

    perfAll = PerformanceMetrics.merge(perfMetrics)
    # print('(mf_ensemble_suite) how many methods in total? %d' % perfAll.n_methods())

    post_analysis(perfAll, context='nmf_ensemble_suite', highlight=['nmf', ])

    return perfAll

def wmf_ensemble_suite(**kargs):
    perfMetrics = []

    # base predictors 
    if kargs.get('run_bp', True):  
        ret = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
        perfMetrics.append(ret['metrics'])

    ## how does it compare to stacking methods?
    if kargs.pop('run_bp_stacker', True):
        ret = base_stackers(topk=kargs.get('topk_bp_stackers', -1), metric='fmax')
        perfMetrics.append(ret['metrics'])
    
    # try different variations
    # param_grid = kargs.get('param_grid', {'n_factors':[10, 20], 'alpha':[100, ]})
    
    # very hyperparameters 
    if kargs.get('run_wmf_ensemble', True): 
        ret = wmf_ensemble(**kargs)
        perfMetrics.append(ret['metrics'])

    # stacker on top of wmf
    if kargs.get('run_wmf_stacker', False): 
        ret = run_stacker(dataset='wmf')
        perfMetrics.append(ret['metrics'])

    ## WMF-derived neighborhood methods
    if kargs.get('run_wmf_similarity', False): 
        ret = wmf_similarity_ensemble(**kargs)
        perfMetrics.append(ret['metrics'])

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

def wmf_ensemble_model_select0(**kargs):
    """
    Model selection without dev set; higher risk of overfitting, used for test only. 

    Related
    -------
    1. test_wmb_probs(): this test suite attempts to find out if the reconstructed probabilities via WMF can help label predictions


    """
    import evaluate 
    from sklearn.model_selection import ParameterGrid

    div(message='(wmf_ensemble_model_select0) Model selection (project: %s) ...' % ProjectPath, symbol='#', border=1)

    perfMetrics = []
    
    # base predictors 
    if kargs.get('run_bp', True):  
        topk_bps = kargs.get('topk_bps', -1)
        baseline = base_predictors(topk=topk_bps, metric='fmax')
        perfMetrics.append(baseline['metrics'])

    ## how does it compare to stacking methods?
    ## how does it compare to stacking methods?
    if kargs.pop('run_bp_stacker', False):
        basestacker = run_stacker()
        perfMetrics.append(basestacker['metrics'])

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
    baseline = base_predictors(topk=topk_bps, metric='fmax')
    perfMetrics.append(baseline['metrics'])

    ## how does it compare to stacking methods?
    topk_bp_stackers = kargs.get('topk_bp_stackers', -1)
    basestacker = base_stackers(topk=topk_bp_stackers, metric='fmax')
    perfMetrics.append(basestacker['metrics'])

    ## Model-based ensemble learning, Basics MF-based methods (compare this with more advanced methods e.g. weighted MF)
    perf_mf = nmf_ensemble_suite()  # (basic) model-based
    perfMetrics.append(perf_mf)    
    
    ## Neighborhood ensemble, memory-based ensemble
    perf_neighborhood = t_neighborhood_ensemble()  # memory-based
    perfMetrics.append(perf_neighborhood)

    ### Weighted MF via ALS
    # perf_wmf = wmf_ensemble_suite(n_factors=params.get('n_factors', 10), alpha=params.get('alpha', 100))
    
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
        print('(post_analysis) No-op #')
        return
    # precondition: after all the performance metrics are collected for all objects 
    # operations: merge, save, sort, plot
    
    if isinstance(perfMetrics, list): 
        perfGrand = PerformanceMetrics.merge(perfMetrics)
    else: 
        perfGrand = perfMetrics
    
    if perfGrand.isEmpty(): 
        print('(post_analysis) Null PerformanceMetrics. No-op. #')
        return

    ##########################################################################################
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
        ######################################################################################
        file_name = '{method}_{metric}_comparison-N{size}-D{domain}'.format(method=category, 
            metric=metric, size=perfGrand.n_methods(), domain=Domain)
        evaluate.plot_performance(perfGrand, metric=metric, ascending=True, domain=Domain, file_name=file_name)

        # file_name = '{method}_{metric}_pair_comparison-N{size}-D{domain}'.format(method=category, 
        #     metric=metric, size=perfGrand.n_methods(), domain=Domain)
        # evaluate.plot_pairwise_performance(perfGrand, metric=metric, ascending=True, domain=Domain, file_name=file_name)

    # summary (mean of target methods)

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
    div(message='(test_stacker) Comparison of stackers (project: %s) ...' % ProjectPath, symbol='#', border=1) # stackers on MF-reproduced probabilities vs stackers on BP predictions
    exact_match = kargs.get('exact', True)   # match the file name (of the training data) exactly as they are (format: <dset_id>-<dtype>-<fold>.csv.gz, e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-validation-1.csv.gz)
    tRunBP = kargs.get('run_bp', False if exact_match else True)
    tRunBaseStacker = kargs.get('run_bp_stacker', False if exact_match else True)
    
    # perf = PerformanceMetrics()
    perfMetrics = []
    if exact_match: 
        for dataset in kargs.get('datasets', ['wmf', 'nmf', ]):  # foreach dataset: (names of the training data 
            # 1. dataset is named via the MF/CF method and its parameter setting
            # 2. if aug_data is set to True, then don't do stacking on top of preference-augmented dataset
            # 3. set 'evaluation' to False to disable post_analysis(); in doing so, we get the performance metrics as a return value 
            #    then later on, pass this performance metrics to the caller in order to compare with other methods
            print("... (verify) matching {name} exactly ...".format(name=dataset))
            perf = run_stacker_suite(context='{mf_method}_stackers'.format(mf_method=dataset), method=dataset, exact=True, 
                        run_bp=tRunBP, run_bp_stacker=tRunBaseStacker, 
                        topk_bps=3, topk_bp_stackers=3,  # only comparing with the top 3 baseline methods 
                        parallelize=True, reconstructed_testset=True, 
                        aug_data=kargs.get('aug_data', True), # only relevant when dataset has 'pref' as a keyword
                        evaluation=kargs.get('evaluation', False))  
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

def test_stacker_subsampling(context, dset_id, evaluation=True, **kargs):
    """

    Memo
    ----
    context: example 'test_wmf_probs_via_stackers-{0}'.format(dset_id),  # ID for PerformanceMetrics

    """
    # Test stackering in subsampling mode (based on the data generated by wmf_ensemble_iter()): 
    #    1. we will examine the stacking performance on 'prior' dataset (the training data prior to CF transformation)
    #    2. then examine the stacking performance on 'posterior' dataset
    div(message='(test_stacker) Comparison of stackers (project: %s) ...' % ProjectPath, symbol='#', border=1) # stackers on MF-reproduced probabilities vs stackers on BP predictions
    exact_match = True  # match the file name (of the training data) exactly as they are (format: <dset_id>-<dtype>-<fold>.csv.gz, e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-validation-1.csv.gz)

    # now do match-execute-evaluate 
    perfMetrics = []
    for dtype in ['prior', 'posterior', ]: 

        # 1. match
        datasets = common.match_exact(path=ProjectPath, method=dset_id, file_type=dtype, ext='csv.gz', verify=True, policy_iter='subsampling') # exception_=False
        print("(test_stacker) Found {n} sets of '{dtype}' data:\n{list}\n".format(n=len(datasets), dtype=dtype, list=datasets))

        # 2. execute (usuall there's only 1 matching data set)
        n_indices = 0
        for dataset, indices in datasets.items():
            if not indices: indices = range(System.n_runs)
            n_indices = len(indices)

            print('... (verify) method ID: {name}, indices: {idx}'.format(name=dataset, idx=indices)) # example method ID: wmf_F100_A100_XCFuser_S2
            ret = run_stacker(dataset=dataset, parallelize=kargs.get('parallelize', True), indices=indices, file_type=dtype)
            perfMetrics.append(ret['metrics'])

    # merge all performance metrics associated with different methods (columns) into one big table 
    ret = {}
    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)
    
    # 3. evaluate
    if evaluation: 
        for metric in PerformanceMetrics.tracked: 
            ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

        post_analysis(perfAll, context=context) # highlight=['stacker', ]

    return perfAll

def test_combiner_subsampling(context, dset_id, evaluation=True, **kargs):

    div(message='(test_stacker) Comparison of simple aggregations (project: %s) ...' % ProjectPath, symbol='#', border=1) # stackers on MF-reproduced probabilities vs stackers on BP predictions
    exact_match = True  # match the file name (of the training data) exactly as they are (format: <dset_id>-<dtype>-<fold>.csv.gz, e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-validation-1.csv.gz)

    perfMetrics = []
    for dtype in ['prior', 'posterior', ]: 
        # 1. match
        datasets = common.match_exact(path=ProjectPath, method=dset_id, file_type=dtype, ext='csv.gz', verify=True, policy_iter='subsampling') # exception_=False
        print("(test_combiner) Found {n} sets of '{dtype}' data:\n{list}\n".format(n=len(datasets), dtype=dtype, list=datasets))

        # 2. excute  
        for dataset, indices in datasets.items():
            if not indices: indices = range(System.n_runs)

            # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
            combiner = run_combiner(dataset=dataset, aggregation_func='mean', file_type=dtype, n_runs=System.n_runs)  # run_simple_combiner(dataset, aggregation_func, file_type='')  
            perfMetrics.append( combiner['metrics'] ) # kargs: method, aggregation_func, (test, ) 

    # merge all performance metrics associated with different methods (columns) into one big table 
    ret = {}
    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)
    
    # 3. evaluate
    if evaluation: 
        for metric in PerformanceMetrics.tracked: 
            ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

        post_analysis(perfAll, context=context) # highlight=['stacker', ]

    return perfAll

def test_meta_users(context, dset_id, evaluation=True, **kargs): 
    meta_users = ['latent_mean', 'masked_latent_mean']
    file_types = ['prior', 'posterior', ]
    sep = kargs.get('sep', ',')
    col_index = 'fold'
    col_prediction = 'prediction'
    col_label = 'label'

    indices = list(range(5))
    n_evaluated = 0
    ret = {}
    perfMetrics = []
    for file_type in file_types: 
        for meta_user in meta_users: 
            
            dset_type = file_type if file_type.startswith(('prior', 'post')) else 'prediction'
            fpath = '{path}/analysis/{stacker}.S-{dataset}-{suffix}.csv'.format(path=ProjectPath, stacker=meta_user, dataset=dset_id, suffix=dset_type)
            assert os.path.exists(fpath), "predictions not found | dtype: {dtype}, method_id: {id} | path: {path}".format(dtype=dset_type, id=method_id, path=fpath)
            df = pd.read_csv(fpath, sep=sep, header=0, index_col=False) # error_bad_lines=True 

            if n_evaluated == 0: 
                indices = df[col_index].unique()
                print('... found {n}-run worth of predictions'.format(n=len(indices)))

            perf_per_fold = []
            for i, dfi in df.groupby([col_index, ]): 
                print('### processing cycle #{i} '.format(i=i))
                predictions = dfi.prediction.values
                labels = dfi.label.values
                if i == 0: 
                    print('sample predictions: {val}'.format(val=predictions[:10]))

                full_method = '{prefix}_{dataset}_{dtype}'.format(prefix=meta_user, dataset=dset_id, dtype=file_type)
                perf_per_fold.append( analyzePerf(labels, predictions, method=full_method) )
            
            perf = PerformanceMetrics.consolidate(perf_per_fold, test_=kargs.get('test', False))  # foreach metric, take average over CV folds
            perfMetrics.append(perf)
            n_evaluated += 1

    ret['metrics'] = perfAll = PerformanceMetrics.merge(perfMetrics)

    # 3. evaluate
    if evaluation: 
        for metric in PerformanceMetrics.tracked: 
            ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

        post_analysis(perfAll, context=context) # highlight=['stacker', ]
    
    return perfAll

def test_wmf_probs_suite(**kargs): 
    def display(): 
        div('-- Experimental Setting --', symbol='#', border=2)
        for k, v in kargs.items():
            print('  [%s] = %s' % (k, v)) 
        return
    import utils_cf as uc 

    # >>> may need to use the setting in the training set name
    kargs['setting'] = setting = kargs.get('setting', System.options.setting if System.options is not None else 11) # predefined setting
    # print('... setting: %s' % setting)
    
    # see System.descriptions
    System.descriptions = descriptions = \
            {0: 'baseline', 

                1: 'item_centered', 2: 'user-centered', 
                    3: 'item-centered-unsupervised', 4: 'user-centered-unsupervised', 

                    5: 'item-centered-low-support', 6: 'preference-masked', 

                    7: 'item-centered-reconstruct', 8: 'user-centered-reconstruct', 
                    9: 'item-centered-mask-test-unsupervised', 10: 'user-centered-mask-test-unsupervised', 
                    11: 'item-centered-tradeoff', 12: 'user-centered-tradeoff', 
                    17: 'item-centered-tradeoff-reconstruct', 18: 'user-centered-tradeoff-reconstruct', 

                    21: 'item-centered-transfer', 22: 'user-centered-transfer', 

                    # proba threshold policy
                    31: 'item-centered-fmax', 32: 'user-centered-fmax', 

                    # ALS, optimization
                    41: 'item-centered-seed', 42: 'user-centered-seed', 
                    43: 'item-centered-long-iter', 44: 'user-centered-long-iter', 
                    45: 'item-centered-low-reg', 46: 'user-centered-low-reg',

                    # uniform confidence scores 
                    51: 'item-centered-uniform', 52: 'user-centered-uniform',

                    # meta users 
                    63: 'meta-users-filter-user-item',

                    # algorithmic control group 
                    100: 'uniform', 
                    }

    actions = {'replace': 'replace bad ratings (i.e. predict <- False)', 
               'reconstruct': 'reconstruct the entire rating table (i.e. predict <- True)', 
               'support': 'using only %{ratio} of users as a support for each item', 
            }

    # default parameter grid
    ctrl_params = ['n_factors', 'alpha', ]
    
    ##########################################
    # ... select algorithmic setting

    # global parameter 
    print('(test_wmf_probs_suite) testing setting #{0}\n'.format(setting))


    tReconstructMatrix = False  # if True, reconstruct the entire rating matrix; set to False to use 'replace'
    tMetaUsers = False

    # global setting 
    kargs['include_meta_users'] = tMetaUsers
    kargs['predict_probs'] = tReconstructMatrix    # if False, call uc.replace() => replace bad ratings 
    kargs['policy_iter'] = System.policy_iter

    # 1. approximate ratings using 'item'-centered confidence matrix (i.e. each datum is represented by selected users/classifiers in terms
    #    of their predictions) 
    if setting in (0, descriptions[1]):  # [design] 
        div('... Testing basic facilities (base predictors, basic stackers)')
        test_baselines(**kargs)
        sys.exit(0)    

    elif setting in (1, 'item-centered'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'   
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (2, 'user-centered'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    elif setting in (3, 'item-centered-unsupervised'): 
        # for each item/datum, we select k most reliable user/classifiers and their probability estimates; the 'k' is determined by 
        # ratio_small_class
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'  
        kargs['ratio_users'] = 0.5 
        kargs['ratio_small_class'] = -1 # 0: to use default, used to estimates lh, which is then used to select the k most 'reliable' classifiers/users
        kargs['supervised'] = False

        div('[{0}] Approximate {scores} using ((UnSupervised)) {repr}-centered confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (4, 'user-centered-unsupervised'):  
        # more conservative than item-centered unsupervised; for each user/classifier, we keep only the top k probabilities and the lowest k probabilities to represent 
        # reliable esimates of positive and negative classes respectively
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'   

        # used to estimate the highest k probabilities and lowest k probabilities for each user/classifier
        # [note] use uc.classPrior() to estimate minority class ratio
        kargs['ratio_small_class'] = 0  # set to 0 to use minority class ratio and a correction factor as an estimate
        kargs['supervised'] = False

        div('[{0}] Approximate {scores} using ((UnSupervised)) {repr}-centered confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (5, 'item-centered-low-support'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'  
        kargs['ratio_users'] = 0.1  
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using {repr}-centered confidence matrix -- {action} -- #'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['support'].format(ratio=kargs['ratio_users'])),
            symbol='=', border=2)

    elif setting in (6, 'preference-masked-user'): 
        # 2. Use preference scores to select entries of (R, T) while masking the rest (by setting them to zeros)
        #    policy: 'preference'
        kargs['policy_opt'] = 'preference' 
        
        kargs['policy_replace'] = 'rating'  # replace preference scores by ratings using preferences as a mask
        kargs['policy'] = 'user'      # >>> still need to specify the policy for computing the mask

        if kargs['policy'] == 'user': 
            kargs['ratio_users'] = 0.5

        div('[{0}] Preference scores used as meta data for (R, T)-entry selection >> {action}'.format(setting, 
            action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']), 
            symbol='=', border=2)
    elif setting in (7, 'item-centered-mask-test'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'  
        kargs['predict_probs'] = True  # if False, call uc.replace() => replace bad ratings 
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True # True is default

        kargs['mask_all_test'] = True  # mark all test set as not reliable, do not try to estimate which ones to re-estimate
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (8, 'user-centered-mask-test'): 
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'  
        kargs['predict_probs'] = True  # if False, call uc.replace() => replace bad ratings 
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = -1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        kargs['mask_all_test'] = True
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (9, 'item-centered-mask-test-unsupervised'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'  
        kargs['predict_probs'] = True  # if False, call uc.replace() => replace bad ratings 
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = False

        kargs['mask_all_test'] = True
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (10, 'user-centered-mask-test-unsupervised'): 
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'  
        kargs['predict_probs'] = True  # if False, call uc.replace() => replace bad ratings 
        kargs['supervised'] = False
        kargs['ratio_small_class'] = 0  # only relevant when in unsupervised mode i.e. suerpvised <- False

        kargs['mask_all_test'] = True
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    # tradeoff in the cost function 
    elif setting in (11, 'item-centered-tradeoff'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (12, 'user-centered-tradeoff'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (17, 'item-centered-tradeoff-reconstruct'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True # True is default

        kargs['mask_all_test'] = True  # mark all test set as not reliable, do not try to estimate which ones to re-estimate
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (18, 'user-centered-tradeoff-reconstruct'): 
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = -1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        kargs['mask_all_test'] = True
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (21, 'item-centered-transfer'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'transfer'
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (22, 'user-centered-transfer'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'  
        kargs['policy_opt_T'] = 'transfer'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    # examining proba thresholds
    elif setting in (31, 'item-centered-fmax'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'  
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['policy_threshold'] = 'fmax'
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (32, 'user-centered-fmax'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['policy_threshold'] = 'fmax'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    # examining optmization
    elif setting in (41, 'item-centered-seeding'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        # kargs['policy_threshold'] = 'prior'
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (42, 'user-centered-seeding'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        # kargs['policy_threshold'] = 'prior'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (43, 'item-centered-long-iter'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        # kargs['policy_threshold'] = 'prior'
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        System.n_epochs = 100
        System.n_epochs_foldin = 50
        System.lambda_val = 0.7  # 0.8 by default

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (44, 'user-centered-long-iter'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        # kargs['policy_threshold'] = 'prior'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        System.n_epochs = 100
        System.n_epochs_foldin = 50
        System.lambda_val = 0.7  # 0.8 by default

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (45, 'item-centered-low-reg'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        
        # kargs['policy_threshold'] = 'prior'
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        # System.n_epochs = 100
        # System.n_epochs_foldin = 50
        System.lambda_val = 0.5  # 0.8 by default

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (46, 'user-centered-low-reg'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        
        # kargs['policy_threshold'] = 'prior'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        # System.n_epochs = 100
        # System.n_epochs_foldin = 50
        System.lambda_val = 0.5  # 0.8 by default

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    # confidence scores and mask fucntions
    elif setting in (51, 'item-centered-uniform'):  # uniform confidence scores
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'
        kargs['conf_measure'] = 'uniform'
        
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (52, 'user-centered-uniform'): # uniform confidence scores
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'rating'
        kargs['conf_measure'] = 'uniform'
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (63, 'meta-users-filter-user-item'):  # uniform confidence scores
        kargs['policy'] = 'user'
        kargs['policy_test'] = 'item'
        kargs['include_meta_users'] = True

        kargs['policy_opt'] = 'rating'
         
        kargs['ratio_users'] = 0.5    # need this for unsupervised item-centered filtering in T
        kargs['supervised'] = True   # supervised in R

        div('[{0}] Approximate {scores} using (({repr}-centered in R but {reprtest}-centered in T)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], reprtest=kargs['policy_test'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    # algorithmic 'control group' (those that may not work but still interesting to try)
    elif setting in (100, 'uniform'):
        kargs['policy'] = 'uniform'
        kargs['policy_opt'] = 'rating'
        kargs['conf_measure'] = 'uniform'
         
        kargs['ratio_users'] = 0.5  # don't care
        kargs['supervised'] = True   # don't care

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    else: 
        raise NotImplementedError

    display()
    ########################################
    # Main test routine 

    test_wmf_probs(**kargs)

    #########################################
    # note: this message has to be consistent with Job.end_job (see cf_run.py, parse_job.py)
    div('(test_wmf_probs_suite) Completed experimental setting #{0}: {1}  ---\n'.format(setting, descriptions.get(setting, 'generic')), symbol='#', border=1)

    return 

def test_wmf_probs(**kargs): 
    """

    Memo
    ----
    1. meta parameters
       conf_measure
    """
    import evaluate 
    from evaluate import PerformanceMetrics
    from utils_sys import format_list, format_sort_dict
    from sklearn.model_selection import ParameterGrid
    div(message="(test_wmf_probs) Do the reconstructed probabilities have higher predictive strength? (domain: {0})".format(Domain), symbol='#', border=1)

    target_metric = kargs.get('metric', 'fmax')
    topk_bps = 3  # only show top K BPs on the performance chart
    topk_bp_stackers = -1  # set to a number < 0 to consider ALL stackers in the performance comparison

    # default parameters
    n_epochs = kargs.get('n_epochs', System.n_epochs)
    n_epochs_foldin = kargs.get('n_epochs_foldin', System.n_epochs_foldin)
    tTrainModel = kargs.get('train_model', True)
    tTrainBaseModel = kargs.get('train_base_model', System.run_baseline)
    tMetaUsers = kargs.get('include_meta_users', False)

    # default setting: executed through command line? if so, look up command line options; if not, 11 
    algorithm_setting = System.options.setting if System.options is not None else 11
    ### Section I: Model selection

    # parameter settings
    ## a. fix (nf, alpha), vary mode
    #     {'n_factors':[10, ], 'alpha':[100, ], 'conf_measure': ['brier', 'ratio']}
    ## b. fix mode, vary (nf, alpha)

    # >>> paramters by senarios
    # 1. Ensure each data point (item) has represenative BP predictions
    #    policy: 'item'
    # 2. Use preference scores to select entries of (R, T) while masking the rest (by setting them to zeros)
    #    policy: 'preference'

    # meta_params do not go into the model selection loop for the following reasons: 
    #     1) if included, there'll be too many parameters to tune 
    #     2) these parameters may not play a key role in the performance or other research questions we care about
    #     3) we wish to analyze these parameter settings separately and independently 
    #     4) Group I policy specifies both the method to construct the confidence matrix and the optimization objective 
    #        Group II, however, specifies only the method to construct the confidence matrix; the optimization objective in this case
    #        is specified through policy_opt
    meta_params = { 
                   'setting': kargs.get('setting', algorithm_setting), 
                   'conf_measure': kargs.get('conf_measure', 'brier'),   # options: 'brier', 'uniform', 'ratio', 'corr', 'auc' ...
                   
                   'policy': kargs.get('policy', 'user'),  # {'user', 'item', }
                   'policy_test': kargs.get('policy_test', 'user'),   # filtering policy for the test set; default to be same as that of the training set
                   'policy_opt': kargs.get('policy_opt', 'preference'),  # {'rating', 'preference', 'label', 'tradeoff'}
                   'policy_opt_T': kargs.get('policy_opt_T', 'foldin'),  # {'foldin', 'seed', 'transfer', 'transfer+seed'}
                   'policy_replace': kargs.get('policy_replace', 'rating'),  # only relevant when policy_opt <- 'preference'
                   'policy_iter': kargs.get('policy_iter', 'subsampling'),      # iteration policy; with model selection, set policy_iter to 'subsampling' 
                   'policy_threshold': kargs.get('policy_threshold', 'prior'),  # group II, how prob thresholds are determined {'fmax', 'prior'/'topk'}

                   'unbag': kargs.get('unbag', False),  # combine bagged classifiers when preparig for the rating matrix data?  
                   'resume_als': kargs.get('resume_als', True), # in ALS fold-in, use the learned factor vector as an init. or fix it so that ALS reduces to LS? 

                   'n_epochs': n_epochs,  # number of iterations within the MF algorithm (e.g. ALS)
                   'n_epochs_foldin': n_epochs_foldin, 
                   'n_runs': kargs.get('n_runs', System.n_runs),  # number of runs of random subsampling with model selection  
                   'n_runs_modelselect': kargs.get('n_runs_modelselect', System.n_runs_modelselect),  # number of model selection iterations used to determine the best hyperparams setting
                   'max_dev': kargs.get('max_dev', 3000),  # max sample size for model selection loop; set None to use all 

                   'ratio_users': kargs.get('ratio_users', 0.5),   # param for policy group II 
                   'ratio_small_class': kargs.get('ratio_small_class', -1),  # set to <= 0 to let the system estimate it (via miniority class proportion) 
                   'factor_small_class': kargs.get('factor_small_class', 1.0), # use ratio of the minority class (divided by this factor) as a ratio to estimate probability thresholds (for positive and negative)

                   'augmented': kargs.get('augmented', True),   # approximate only R (training split), or both R & T (i.e. augmetned: True)? 
                   'include_meta_users': tMetaUsers,  # introduce meta classifiers (e.g. mean predictor)

                   # if 'predict_probs' is True, then reconstruct the entire R and T using latent factors (P, Q) ...
                   # if False, then replace 'bad entries' while preserving good entries (i.e. reconstruct only the bad entries)
                   'predict_probs': kargs.get('predict_probs', False),    # if predict is set to False => call uc.replace() instead of predict_by_factors()

                   'supervised': kargs.get('supervised', True),  # param for policy group II
                   'mask_all_test': kargs.get('mask_all_test', False),  # if True, mask all entries in T, policy group II: item-centered, user-centered 
                   'masked': kargs.get('masked', True),    
                   'aug_data': kargs.get('aug_data', True)}  # options: {'rating', 'preference', 'label', 'tradeoff'}
    
    # default params
    # configure hyperparameters if specified by the command line
    ############################################################
    param_grid = kargs.get('param_grid', System.param_grid)
    assert len(param_grid) > 0
    n_factors = kargs.get('n_factors', System.n_factors)  
    alpha = kargs.get('alpha', System.alpha)

    # configuration dependency
    ############################################################
    if meta_params['resume_als']: System.n_epochs_foldin = System.n_epochs

    # params = {k: v[0] for k, v in param_grid.items()}  # default hyperparams when not in model selection mode
    
    # model selection? 
    hyperparams = ['n_factors', 'alpha', ]
    if sum(1 for v in param_grid.values() if len(v) > 1) > 0: 
    # if sum(len(param_grid[p]) for p in hyperparams) > len(param_grid): # if any of the parameter has >= 2 values, then we are automatically in model selection mode
        # then we are in model selection mode, which means we want policy_iter to be 'subsampling'
        meta_params['policy_iter'] = 'subsampling'  # cv is too expensive for model selection 
    ############################################################
    print('(test_wmf_probs) policy_iter? %s, policy_opt: %s, parameter grid: %s' % \
        (meta_params['policy_iter'], meta_params['policy_opt'], param_grid))
 
    ### Analysis
    analysis_files = [ 'test_wmf_probs_training', 
                       'test_wmf_probs_via_stackers-{0}', "test_wmf_probs_via_combiner-{0}", 
                       "test_wmf_probs_all_methods"]

    perfMetrics = []
    # set reverse to True if target_metric is of the-greater-the-better kind (e.g. auc)

    fmaxSummary = {m: float('nan') for m in ['bp', 'stacker', ]}
    perfBase = perfStacker = None
    if tTrainBaseModel:
        ### base predictors
        baseline = base_predictors(topk=topk_bps, metric='fmax')
        perfBase = baseline['metrics']
        nBP = perfBase.n_methods()
        assert nBP > 0 and nBP <= topk_bps, "Found %d base predictors while requesting %d" % (perfBase.n_methods(), topk_bps)
        perfMetrics.append(perfBase)

        #############################################
        summary = PerformanceMetrics.summarize(perfBase, metric='fmax', keywords=[])  # keys: methods, metrics, mean, median, max, min
        fmaxSummary['bp'] = summary['mean']
        div("(result) Average base predictor performance in fmax: {score}".format(score=summary['mean']), symbol='#', border=2)
        print('... BP methods:\n... \n%s\n' % format_list(zip(summary['methods'], summary['scores']), mode='v', sep=', ', padding=5))
        #############################################

        ### baseline stackers
        basestacker = base_stackers(topk=topk_bp_stackers, metric='fmax', parallelize=True)
        perfStacker = basestacker['metrics']
        perfMetrics.append(perfStacker)

        #############################################
        # mean performance of stackers
        summary = PerformanceMetrics.summarize(perfStacker, metric='fmax', keywords=[])  # keys: methods, metrics, mean, median, max, min
        fmaxSummary['stacker'] = summary['mean']
        div("(result) Average stacker performance in fmax: {score}".format(score=summary['mean']), symbol='#', border=2)
        print('... stacker methods:\n... \n%s\n' % format_list(zip(summary['methods'], summary['scores']), mode='v', sep=', ', padding=5))
        #############################################
    
    ### Model Training by verying hyperparameters 
    best_params = {'n_factors': n_factors, 'alpha': alpha}  # default values to be modified by the model selection routine
    if tTrainModel:
        n_target_methods = 0
        assert meta_params['policy_opt'] != 'preference'
        # for params in list(ParameterGrid(param_grid_outer)):  # a list of dictionaries containing target (hyper)parameters
        ret = wmf_ensemble(
                        # run_bp=False, run_bp_stacker=False,  
                        # run_wmf_ensemble=True, run_wmf_stacker=False, run_wmf_similarity=False, 
                        setting=meta_params['setting'],  # important file descriptor
                        
                        augmented=meta_params['augmented'], # if False, run uc.replace() instead of uc.predict_by_factors() by default
                        include_meta_users=meta_params['include_meta_users'],

                        predict_probs=meta_params['predict_probs'],     # prediction vs reconstruction (if False) 
                        supervised=meta_params['supervised'],   # using supervised methods to estimate the mask (of the confidence matrix)
                        # mask_all_test=meta_params['mask_all_test'], 
                        # masked=meta_params['masked'],   # if True, apply mask function to either reduce confidence weights or replace 'bad ratings'
                        
                        # conf_user=meta_params.get('conf_user', True), 

                        conf_measure=meta_params['conf_measure'], 
                        policy=meta_params['policy'],  # specifies the policy for confidence matrx (and for optimization policy when 'policy_opt' is not given)
                        policy_test=meta_params['policy_test'], 
                        policy_opt=meta_params['policy_opt'],  # optimization
                        policy_opt_T=meta_params['policy_opt_T'],  # optimization on the test set
                        policy_threshold=meta_params['policy_threshold'],
                        
                        policy_iter=meta_params['policy_iter'],  # {'cv', 'subsampling'}
                        unbag=meta_params['unbag'], 
                        resume_als=meta_params['resume_als'],

                        # >>> param_grid for model selection
                        param_grid=param_grid,   # only relevant when policy_iter <- 'subsampling' which takes into account model selection
                        n_epochs=meta_params['n_epochs'],  # number of ALS iterations (n_iter)
                        n_epochs_foldin=meta_params['n_epochs_foldin'], 
                        n_runs=meta_params['n_runs'],  # number of runs for the entire ensemble learning process
                        n_runs_modelselect=meta_params['n_runs_modelselect'], # number of model selection iterations

                        max_dev=meta_params['max_dev'], 
                        
                        # default parameter
                        n_factors=n_factors, alpha=alpha, 

                        ratio_small_class=meta_params['ratio_small_class'], 
                        ratio_users=meta_params['ratio_users'])

        perf = ret['metrics']
        ############################################################################################################
        # ... best parameters among those combinations from param_grid 
        #     (after n_runs of *_ensemble routines (e.g. wmf_ensemble) and n_runs_modelselect of model selection iterations)
        # ... 'best_params' only overwritten when model selection took place
        if 'best_params' in ret: best_params = ret['best_params'] # best_params after running model selection
        ############################################################################################################
        
        if not perf.isEmpty(): # unless we are in prediciton mode (meta_params['predict_probs']: True), perf object should be empty (or a null PerformanceMetrics)
            perfMetrics.append(perf)
            n_target_methods = perf.n_methods()

        if n_target_methods > 0: 
            perfAll = PerformanceMetrics.merge(perfMetrics)
            n_baseline = perfAll.n_methods()-n_target_methods
            print('(test_wmf_probs) how many WMF methods in total? {0} (typically 1) vs n_baseline: {1}'.format(n_target_methods, n_baseline))
            post_analysis(perfAll, context='cf_training_phase', highlight=['wmf', ], metrics=['fmax', 'auc', ])
        else: 
            # print("(test_wmf_probs) in data reconstruction mode: augmented? {0}".format(meta_params['augmented']))
            div(message="(test_wmf_probs) No WFM model trained!")

    ## Setting II. After obtaining the MF-produced training data
    # note: 
    #     1. in subsampling mode
    #           the transformed data will be smaller than the original dataset |D'| < |D| 
    #     2. in CV mode 
    #           we use a CV loop to get chuncks of transformed data |T'<i>|, and combine them together to get |D'| == |D|
    #           but this is computationally expensive, esp when considering model selection loops
    #
    if tTrainModel:            
        div('After CF model training | best paramters: {opt_params} ... (verify) #'.format(opt_params=best_params))
    else: 
        div("Given (default) paramters: {opt_params}, test stacking and mean aggregation on 'prior' and 'posterior' data ... (verify) #".format(opt_params=best_params))

    perfMetrics = [perfBase, perfStacker] if tTrainBaseModel else []
    ## now test the quality using stacking methodology 
    # for params in list(ParameterGrid(param_grid)): 
    print("(1) testing stackers on 'prior' and 'posterior' datasets ... ")
    for params in [best_params, ]:
        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data

        # [test] test if the validation and prediction sets exist
        print("(test_wmf_probs) Stackers on D prime | dset_id: %s, params: %s" % (dset_id, params)) # ex. params: {'n_factors': 250, 'alpha': 100}

        # perf = test_stacker(context='stackers-prior-vs-posterior-{0}'.format(dset_id),  # ID for PerformanceMetrics
        #                         run_bp=False, run_bp_stacker=False,  # focused on wmf stackers

        #                         datasets=[dset_id, ],  # a set of datasets named after training data IDs
        #                         exact=True)  # aug_data=meta_params['aug_data']

        perf = test_stacker_subsampling(context='stackers-prior-vs-posterior-{0}'.format(dset_id), dset_id=dset_id, evaluation=True)
        perfMetrics.append(perf)
        #############################################
        # ... perf object include both 'prior' and 'posterior'
        
        for dtype in ['prior', 'posterior', ]:
            ret = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[dtype, ])  # keys: methods, metrics, mean, median, max, min
            entry = 'stacker_{dtype}'.format(dtype=dtype)
            fmaxSummary[ entry ] = ret['mean']

            div("(result) Average WMF-stacker fmax on {dtype} data (n_methods: {n}): {score}".format(dtype=dtype, n=len(ret['methods']), score=ret['mean']), symbol='#', border=2)
            print('... WMF stacker methods:\n... \n%s\n' % format_list(zip(ret['methods'], ret['scores']), mode='v', sep=', ', padding=5))
        #############################################
        
    # test via simple aggregation method
    # for params in list(ParameterGrid(param_grid)): 
    print("(2) testing simple aggregation methods on 'prior' and 'posterior' datasets ... ")
    for params in [best_params, ]:
        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data    
        print('(test_wmf_probs) Basic aggregate on (Rh, Th), e.g. mean predictions (no stacking) | dset_id: %s, params: %s' % (dset_id, params))
        
        # note: can use this routine to aggregate preference scores as well
        # perf = test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(dset_id), 
        #                         datasets=[dset_id, ], method='mean') # params: aggregate_func,aug_data, test  
        
        perf = test_combiner_subsampling(context='combiners-prior-vs-posterior-{0}'.format(dset_id), dset_id=dset_id, evaluation=True)
        perfMetrics.append(perf)
        #############################################
        # ... perf object include both 'prior' and 'posterior'

        for dtype in ['prior', 'posterior', ]:
            ret = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[dtype, ])  # keys: methods, metrics, mean, median, max, min
            entry = 'combiner_{dtype}'.format(dtype=dtype)
            fmaxSummary[entry] = ret['mean']

            div("(result) Average WMF-combiner fmax on '{dtype}' data (n_methods: {n}): {score}".format(dtype=dtype, n=len(ret['methods']), score=ret['mean']), symbol='#', border=2)
            print('... WMF combiner methods:\n... \n%s\n' % format_list(zip(ret['methods'], ret['scores']), mode='v', sep=', ', padding=5))
        #############################################

    print("(3) testing meta user predictions ...")
    if tMetaUsers: 
        for params in [best_params, ]:
            dset_id = MFEnsemble.get_dset_id(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data  

            context = 'meta_users' 
            perf = test_meta_users(context, dset_id, evaluation=True, sep=',') 
            perfMetrics.append(perf)

            # e.g. example methods 
            #      latent_mean_wmf_F100_A100_XCFuser_S63', 'masked_latent_mean_wmf_F100_A100_XCFuser_S63'
            for dtype in ['prior', 'posterior', ]:
                ret = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[dtype, ])  # keys: methods, metrics, mean, median, max, min
                entry = 'meta_user_{dtype}'.format(dtype=dtype)
                fmaxSummary[entry] = ret['mean']

                div("(result) Average WMF-meta-users fmax on '{dtype}' data (n_methods: {n}): {score}".format(dtype=dtype, n=len(ret['methods']), score=ret['mean']), symbol='#', border=2)
                print('... WMF meta users:\n... \n%s\n' % format_list(zip(ret['methods'], ret['scores']), mode='v', sep=', ', padding=5))

    # Alternatively, put parameter grid iteration inside the test_<ensemble method> function
    # test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(dset_id), 
    #     param_grid=param_grid, meta=meta, method='mean', target_method='wmf') # method refers to the combining method, specifiy specific MF method via target method
    
    ####################################################
    # context: for the moment, include algorithm parameter as well
    # 
    nf = param_grid['n_factors'][0] if len(param_grid['n_factors']) == 1 else best_params['n_factors']
    alpha = param_grid['alpha'][0] if len(param_grid['alpha']) == 1 else best_params['alpha'] 
    context = 'test_wmf_probs_all_methods-n{n}a{a}'.format(n=nf, a=alpha)
    post_analysis(PerformanceMetrics.merge(perfMetrics), context=context, metrics=['fmax', 'auc', ])  # highlight=['wmf', ]

    ##########################################
    # div('(result) Mean performance in {metric}:\n... BP: {b}\n ... stacker: {s}\n... WMF stacker: {ws}\n... WMF combiner: {wc}'.format(metric='fmax', 
    #     b=fmaxSummary['bp'], s=fmaxSummary['stacker'], ws=fmaxSummary['wmf_stacker'], wc=fmaxSummary['wmf_combiner']), symbol='#', border=2)
    
    title = '(result) Mean performance in {metric} (domain: {domain}) | setting: ({s}, {descrp}) hyperparams: {params}'.format(metric='fmax', 
        domain=System.domain, s=algorithm_setting, descrp=System.descriptions.get(algorithm_setting, 'generic'), params=best_params)
    print( format_sort_dict(fmaxSummary, reverse=True, padding=5, title=title) ) # symbol='#', border=1

    return

def test_wmf_probs_stacker(best_params, meta_params, param_grid=[], **kargs):
    
    algorithm_setting = kargs.get('setting', System.options.setting if System.options is not None else 11) # default: 
    perfMetrics = []

    for params in [best_params, ]:
        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data

        # [test] test if the validation and prediction sets exist
        print("(test_wmf_probs) Stackers on D prime | dset_id: %s, params: %s" % (dset_id, params)) # ex. params: {'n_factors': 250, 'alpha': 100}

        # perf = test_stacker(context='stackers-prior-vs-posterior-{0}'.format(dset_id),  # ID for PerformanceMetrics
        #                         run_bp=False, run_bp_stacker=False,  # focused on wmf stackers

        #                         datasets=[dset_id, ],  # a set of datasets named after training data IDs
        #                         exact=True)  # aug_data=meta_params['aug_data']

        perf = test_stacker_subsampling(context='stackers-prior-vs-posterior-{0}'.format(dset_id), dset_id=dset_id, evaluation=True)
        perfMetrics.append(perf)

        #############################################
        ret = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[])  # keys: methods, metrics, mean, median, max, min
        fmaxSummary['wmf_stacker'] = ret['mean']
        div("(result) Average WMF-stacker performance in fmax: {score}".format(score=ret['mean']), symbol='#', border=2)
        print('... WMF stacker methods:\n... \n%s\n' % format_list(zip(ret['methods'], ret['scores']), mode='v', sep=', ', padding=5))
        #############################################
        
    # test via simple aggregation method
    # for params in list(ParameterGrid(param_grid)): 
    for params in [best_params, ]:
        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data    
        print('(test_wmf_probs) Basic aggregate on (Rh, Th), e.g. mean predictions (no stacking) | dset_id: %s, params: %s' % (dset_id, params))
        
        # note: can use this routine to aggregate preference scores as well
        # perf = test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(dset_id), 
        #                         datasets=[dset_id, ], method='mean') # params: aggregate_func,aug_data, test  
        
        perf = test_combiner_subsampling(context='combiners-prior-vs-posterior-{0}'.format(dset_id), dset_id=dset_id, evaluation=True)
        perfMetrics.append(perf)

        #############################################
        ret = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[])  # keys: methods, metrics, mean, median, max, min
        fmaxSummary['wmf_combiner'] = ret['mean']
        div("(result) Average WFM-combiner performance in fmax: {score}".format(score=ret['mean']), symbol='#', border=2)
        print('... WMF combiner methods:\n... \n%s\n' % format_list(zip(ret['methods'], ret['scores']), mode='v', sep=', ', padding=5))
        #############################################

    # Alternatively, put parameter grid iteration inside the test_<ensemble method> function
    # test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(dset_id), 
    #     param_grid=param_grid, meta=meta, method='mean', target_method='wmf') # method refers to the combining method, specifiy specific MF method via target method
    
    ####################################################
    # context: for the moment, include algorithm parameter as well
    # 
    nf = param_grid['n_factors'][0] if len(param_grid['n_factors']) == 1 else best_params['n_factors']
    alpha = param_grid['alpha'][0] if len(param_grid['alpha']) == 1 else best_params['alpha'] 
    context = 'test_wmf_probs_all_methods-n{n}a{a}'.format(n=nf, a=alpha)
    post_analysis(PerformanceMetrics.merge(perfMetrics), context=context, metrics=['fmax', 'auc', ])  # highlight=['wmf', ]

    ##########################################
    # div('(result) Mean performance in {metric}:\n... BP: {b}\n ... stacker: {s}\n... WMF stacker: {ws}\n... WMF combiner: {wc}'.format(metric='fmax', 
    #     b=fmaxSummary['bp'], s=fmaxSummary['stacker'], ws=fmaxSummary['wmf_stacker'], wc=fmaxSummary['wmf_combiner']), symbol='#', border=2)
    
    title = '(result) Mean performance in {metric} (domain: {domain}) | setting: ({s}, {descrp}) hyperparams: {params}'.format(metric='fmax', 
        domain=System.domain, s=algorithm_setting, descrp=System.descriptions.get(algorithm_setting, 'generic'), params=best_params)
    print( format_sort_dict(fmaxSummary, reverse=True, padding=5, title=title) ) # symbol='#', border=1

    return

def test_baselines(**kargs):
    import evaluate 
    from evaluate import PerformanceMetrics 

    target_metric = kargs.get('metric', 'fmax')
    topk_bps = 3  # only show top K BPs on the performance chart
    topk_bp_stackers = 3

    perfMetrics = []

    # set reverse to True if target_metric is of the-greater-the-better kind (e.g. auc)
    ### base predictors
    ret = base_predictors(topk=topk_bps, metric='fmax')
    perfBase = ret['metrics']
    nBP = perfBase.n_methods()
    assert nBP > 0 and nBP <= topk_bps, "Found %d base predictors while requesting %d" % (perfBase.n_methods(), topk_bps)
    perfMetrics.append(perfBase)

    ### baseline stackers 
    ret = base_stackers(topk=topk_bp_stackers, metric='fmax', parallelize=True)
    perfStacker = ret['metrics']
    perfMetrics.append(perfStacker)

    post_analysis(PerformanceMetrics.merge(perfMetrics), context='test_baselines', highlight=['stacker', ], metrics=['fmax', ]) 

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

    baseline = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append( baseline['metrics'])

    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        perfWMF = wmf_ensemble_suite(n_factors=nf, save=True, 
                    run_stacker=False, run_wmf_ensemble=False, run_wmf_stacker=False, run_wmf_similarity=True, 
                        run_clustering=False, context='wmf_similarity') # set save to True to save R' and T' (reproduced probabilities)
        perfMetrics.append(perfWMF)
        perfNMF = nmf_ensemble_suite(n_factors=nf, save=True, 
                    run_stacker=True, run_nmf_ensemble=False, run_nmf_stacker=False, run_nmf_similarity=True, 
                        run_clustering=False, context='nmf_similarity')
        perfMetrics.append(perfNMF)

    perfAll = post_analysis(perfMetrics, context='wmf_vs_nmf_similarity')

    return perfAll

def test_wmf_clustering(**kargs):
    perfMetrics = []

    baseline = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append(baseline['metrics'])

    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        perfWMF = wmf_ensemble_suite(n_factors=nf, save=True, 
                    run_stacker=False, run_wmf_ensemble=False, run_wmf_stacker=False, run_wmf_similarity=True, 
                        run_clustering=True, context='wmf_similarity') # set save to True to save R' and T' (reproduced probabilities)
        perfMetrics.append(perfWMF)

    perfAll = post_analysis(perfMetrics, context='wmf_clustering')

    return perfAll 

def test_confidence(*kargs):
    perfMetrics = []

    baseline = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append(baseline['metrics'])

    div(message='(test) Compare different confidence matrix (project: %s)' % ProjectPath, symbol='#', border=1)

    # perf_wmf = wmf_ensemble()
    for nf in [10,  ]: 
        for mode in ['brier', 'ratio', ]: 
            for alpha in [1, 50, 100, 1000, ]: 
                perfWMF = wmf_ensemble_suite(n_factors=nf, conf_measure=mode, alpha=alpha, save=True, 
                            run_stacker=False, run_wmf_ensemble=False, run_wmf_stacker=False, run_wmf_similarity=True, 
                                run_clustering=False, context='wmf_confidence_matrices-C{mode}-A{alpha}'.format(mode=mode, alpha=alpha)) # set save to True to save R' and T' (reproduced probabilities)
                perfMetrics.append(perfWMF)

    perfAll = post_analysis(perfMetrics, context='wmf_confidence_matrices')

    return 

def test_similarity(**kargs):

    div(message='(test) Initiate tests on CF-based ensemble learning (project: %s)' % ProjectPath, symbol='#', border=1)

    perfMetrics = []

    # t_recommender()
    baseline = base_predictors(topk=kargs.get('topk_bps', -1), metric='fmax')
    perfMetrics.append(baseline['metrics'])

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

### System configuration
###############################################################################################################
def parse_args():
    import time, os
    from optparse import OptionParser
    from utils_sys import parse_params_list

    ret = {}
    System.startTime = timestamp = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())) # time()
    parser = OptionParser()

    # parser.add_option('--wms-probs', action="store_true", dest="wms_probs", default=False) 
    parser.add_option('-n', '--nfact', '--n-factors', '--factor', dest='n_factors')
    parser.add_option('-a', '--alpha', dest='alpha')  # can be a comma separated list of values or just a single value
    parser.add_option('-s', '--setting', dest = 'setting', type='int', default=2)
    parser.add_option('-d', '--disable-baseline', action="store_true", dest="disable_baseline", default=False)
    parser.add_option('-i', '--iteration', dest='policy_iter', default='subsampling')  # {'cv', 'subsampling'}
    parser.add_option('-r', '--runs', dest='n_runs', type='int', default=10)
    parser.add_option('-m', '--runs-model-select', dest='n_runs_modelselect', type='int', default=1)

    # parser.add_option('--dryrun', action="store_true", dest="dryrun", default=False)
    
    System.options, System.args = options, args = parser.parse_args()
    # System.test_wmf_probs_suite = ret['test_wmf_probs_suite'] = options.wms_probs
    
    # System.n_factors = ret['n_factors'] = options.n_factors
    param_grid = {}
    if options.n_factors is not None:
        # can be i) an integer, ii) a string (of an integer), iii) comma-separated list of integers 
        param_grid['n_factors'] = ret['n_factors'] = parse_params_list(options.n_factors, dtype=int) # [int(str(e).strip()) for e in str(options.n_factors).split(',') if len(str(e).strip()) > 0]
        if len(ret['n_factors']) == 1: System.n_factors = ret['n_factors'][0]
    
    # System.alpha = ret['alpha'] = options.alpha
    if options.alpha is not None: 
         # can be i) an integer, ii) a string (of an integer), iii) comma-separated list of integers 
        param_grid['alpha'] = ret['alpha'] = parse_params_list(options.alpha, dtype=(int, float)) # [int(str(e).strip()) for e in str(options.alpha).split(',') if len(str(e).strip()) > 0]
        if len(ret['alpha']) == 1: System.alpha = ret['alpha'][0]
    
    # >>> param_grid for model selection if any of the (hyper-)parameters have more than one value
    #     if there's no parameters given via the options (e.g. n_factors), then fall back to the default parameter grid (defined in System.param_grid)
    if param_grid: System.param_grid = param_grid  # overwrite the default
    print('(verify) param_grid(opt): {0} System.param_grid: {1}'.format(param_grid, System.param_grid))

    ################################
    # ... default hyperparams 
    # System.n_factors = min(System.param_grid['n_factors'])  # smallest
    # System.alpha = max(System.param_grid['alpha'])

    ### iteration scheme 
    System.policy_iter = options.policy_iter
    System.n_runs = options.n_runs
    System.n_runs_modelselect = options.n_runs_modelselect

    ### other system configurations
    System.run_baseline = not options.disable_baseline

    # print('(parse_args) setting: %s, nf: %s, alpha: %s' % (System.options.setting, System.n_factors, System.alpha))
    return ret 

def resolve_path(path, analysis_dn='analysis'): 
    # resolve project path e.g. /Users/<user>/work/data/pf1
    # home_dir = os.path.expanduser('~')
    # working_dir_default = '/'.join([home_dir, 'work/data', ])
    parentdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(parentdir, 'data')  # e.g. /Users/<user>/work/data

    tCheck = False
    if not os.path.isdir(path):
        if path.find('/') < 0: # only given the domain name e.g. 'pf1' 
            domain = path  # then assume the 'path' is just a domain string
            path = os.path.join(datadir, domain) # e.g. /Users/<user>/work/data/pf1
            assert os.path.exists(path), "Unknown domain: {domain} | its project path do not exist at {path} ...".format(
                domain=domain, path=path)
        else: 
            path = os.path.abspath(path)
            tCheck = True
    else: 
        tCheck = True
    
    if tCheck and not os.path.exists(path): 
        msg = "Invalid project path (which must include domain): {data_path}".format(data_path=path)
        raise ValueError(msg)
    return path

def sysConfig(config_file='config.txt'):

    import utils_sys, common
    # import cf_spec
    # from cf_spec import System, MFEnsemble
    # from evaluate import PerformanceMetrics
    global ProjectPath, Domain, FoldCount, BagCount
    
    np.set_printoptions(precision=3, linewidth=90) # Number of digits of precision for floating point output (default 8). 

    ### parse options 
    ret = parse_args()
    
    # [todo] cleaner solution for command line arguments

    ### project path and domain
    try: 
        # [note] need a copy in the global scope when this main thread spawns subprocesses, they need to have access to certain variables (e.g. project path)
        System.projectPath = ProjectPath = resolve_path(System.args[0])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
        System.analysisPath = os.path.join(ProjectPath, 'analysis')
        System.domain = Domain = os.path.basename(ProjectPath)
    except Exception as e: 
        msg = 'Hint: Missing project path? (e.g. %s)' % System.prefix
        raise ValueError("%s\n... %s" % (e, msg))
    div('Project path: %s' % ProjectPath, symbol='%', border=1)

    ### reconfigure sys.argv for other modules that parse the command line via the sys module
    System.stacker_method = stacker_method = 'standard'
    sys.argv = ['python', ProjectPath , stacker_method]   # System.args[0]
    print('(sysConfig) options: %s, args: %s, sys.argv: %s' % (System.options, System.args, sys.argv))   # options: {'setting': 0}, args: ['/Users/chiup04/Documents/work/data/pf2']

    ### classifier settings
    Properties = common.load_properties(ProjectPath, config_file=config_file)  # parse config.txt (instead of weka.properties)
    
    ### random subsampling or CV 
    System.foldCount = FoldCount = int(Properties['foldCount'])
    System.nestedFoldCount = nestedFoldCount = int(Properties['nestedFoldCount'])
    # if System.policy_iter == 'cv': 
    #     System.foldCount = FoldCount = int(Properties['foldCount'])
    #     System.nestedFoldCount = nestedFoldCount = int(Properties['nestedFoldCount'])
    # else: 
    #     # [problem] the base predictors may have been trained using a particular fold count which may or may not be the same as 'n_runs'
    #     System.foldCount = FoldCount = System.n_runs
    #     System.nestedFoldCount = nestedFoldCount = System.n_runs_modelselect
    
    System.bagCount = BagCount = int(Properties['bagCount']) if 'bagCount' in Properties else int(Properties['bags']) 

    # >>> this does not seem to propogate to subprocesses (e.g. project path seen by wmb_ensemble_fold still contains '?')
    cf_spec.config(project_path=ProjectPath, domain=Domain, fold_count=FoldCount, bag_count=BagCount)  # to be shared by all relavant modules 
    print('(sysConfig) domain: {0}, project path: {1} | iteration: {2} | foldCount: {3}, bagCount: {4}'.format(System.domain, 
        System.projectPath, System.policy_iter, FoldCount, BagCount))
    if not System.policy_iter.startswith(('cv', 'cross')): 
        print('(sysConfig) n_runs: {nr}, n_runs_modelselect: {nm}'.format(nr=System.n_runs, nm=System.n_runs_modelselect))

    # System.projectPath = utils_sys.getProjectPath(domain=Domain, verify_=False)  # default
    # System.domain = Domain

    # all directories depend on this prefix including data_dir, log_dir, plot_dir
    # >>> these have to be configured per thread as well  
    #     following operations are now turned over to cf_spec.config()
    # PerformanceMetrics.set_path(prefix=System.projectPath) # see cf_spec.conifg()
    # MFEnsemble.set_path(prefix=System.projectPath) 
    System.display()

    return ret 
###############################################################################################################


### test and experimental utilities
def test(**kargs): 
    import utils_cf as uc

    ### external options 
    setting = System.options.setting
    if setting in (0, ): 
        div('(test) Testing basic facilities (base predictors, basic stackers) ...')
        test_baselines(**kargs)
        return
   
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

    ### test predictive strength of ME methods

    # >>> map research setting to appropriate parameter configurations
    # test_wmf_probs(**kargs)
    
    test_wmf_probs_suite(**kargs)

    # similairty metrics comparison (off-the-shelf vs MF-induced)
    # test_similarity()

    # test_similarity_nmf_vs_wmf()  # no clustering

    # test_wmf_clustering() # wmf + clustering, which clustering algorithms seem better?

    # test_confidence()
     
    ## stacker comparisons: stackers on MF-reproduced probabilities vs stackers on BP predictions
    # test_stacker()

    return perfAll

def runTest(routine=test_wmf_probs_suite):
    sysConfig()

    algo_settings = {'conf_measure': ['brier', ], 'policy': ['item', 'user', ], 'predict_probs': [True, False]}
    # param_grid = kargs.get('param_grid', algo_settings)  

    assert hasattr(routine, '__call__'), "Invalid test routine: {0}".format(routine)

    # param_grid = list(product(conf_measures, policy_opts, ratio_users))
    # Parallel(n_jobs=get_num_cores(), verbose=50)(delayed(routine)(**params) for params in list(ParameterGrid(param_grid)))
    routine(alpha=100, n_factors=100, setting=4, train_model=False, train_base_model=False)

    return 

def run(**kargs): 

    ### system configuration
    # config(**kargs)

    # 1. tune model 
    # 2. run suite() using the hyperparams setting obtained from step 1

    ## model selection 
    # wmf_ensemble_model_select0(**kargs)

    ### MF-based ensemble learning
    suite(**kargs)  # comparison of groups of algorithms

    ### stacking on MF-reproduced dataset 
    # test_stacker()

    ### plot 
    # plot(file_name='performance_metrics-model-select-pf2.csv')

    return 

def main(**kargs): 
    """

    Memo
    ----
    1. Commonly used commands: 

       python cf.py /Users/pleiades/Documents/work/data/diabetes_cf -d --runs 2 --n_factors "10, 98" --alpha "10, 90"

    """
    import time, timing, timeit

    sysConfig()
    t1 = time.time()

    # run()
    test(train_model=True, train_base_model=False)
    t2 = time.time()
    
    del_t = t2 - t1
    div("Total execution time (via time.time()): {h} hrs ~ {m} mins".format(h=del_t/3600., m=del_t/60.), symbol='#', border=1)
    print('\n> options: {opts}, args: {args}\n... #'.format(opts=System.options, args=System.args))
    return 

if __name__ == "__main__":     
    main()
    # runTest()

