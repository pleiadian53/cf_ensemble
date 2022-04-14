#!/usr/bin/env python
# encoding: utf-8

### configurations
import os, math, sys, gc
import collections, random, math
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
import scipy.sparse
import numpy as np
import pandas as pd

from pandas import DataFrame, Series
import timeit

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import joblib
from joblib import Parallel, delayed
# from sklearn.externals.joblib import Parallel, delayed # [deprecated]

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

    descriptions = {0: 'baseline',

                    1: 'item_centered', 2: 'user-centered', 
                    3: 'item-centered-unsupervised', 4: 'user-centered-unsupervised', 

                    5: 'rating-cascade-sequence', 6: 'rating-cascade-classifier', 
                    7: 'filter-by-polarity', 8: 'filter-by-polarity-cascade', 
                    9: 'filter-by-polarity-sequence-model', 

                    10: 'filter-by-polarity-classifier', 
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

                    # enable meta users piggybacked in the rating matrix (so that their latent factors can be learned) 
                    62: 'meta-users-filter-user', 
                    63: 'meta-users-filter-user-item',

                    # using stacker to optimze model parameters 
                    72: 'user-centered-stacker', 

                    # algorithmic control group 
                    100: 'uniform', 

                    }

    param_grid = {'n_factors': [5, 10, 50, 100, 250, 500], 'alpha': np.logspace(0, 3, num=4, base=10, dtype=int)}  # {'n_factors':[100, ], 'alpha':[100, ]}
    
    policy_iter = 'subsampling'  # options: {'cv', 'seq', 'subsampling', }
    n_epochs = 30
    n_epochs_foldin = 30  # used in the "reduced" ALS to LS at test time
    lambda_val = 0.8
    n_runs = 1
    n_runs_modelselect = 1
    run_baseline = True
    
    # post-CF aggregation
    simple_aggregation = ['mean', 'median', ]  # simple ensemble methods that do not require training split but only test split
    latent_aggregation = ['latent_mean', 'latent_mean_masked',]
    stacker_aggregation = ['log', 'rf' ]
    aggregation_methods = ['mean', 'median', 'log', 'rf'] # other options: 'rf', 'enet', 'knn', 'qda', ... 

    meta_users = ['latent_mean', 'latent_mean_masked',]
    file_types = ['prior', 'posterior', ]

    unbag = False

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
    

class MC(BaseEstimator): # from sklearn.base import BaseEstimator
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

def bestBP(ts, labels, scoring_func=None):
    # ts: train_df, test_df
    if scoring_func is None: scoring_func = common.score

    # a Series (index={classifiers}, value: performance score)
    return ts.apply(lambda x: scoring_func(labels, x)).sort_values(ascending = not common.greater_is_better) 

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

                    perf, pv = analyzePerf(test_labels, predictions, method='{combiner_type}_pref'.format(combiner_type=method))
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

            perf, pv = analyzePerf(test_labels, predictions, method=full_method)
            perf_per_fold.append( perf )
    
    div('(run_preference_combiner) Completed method %s #' % full_method)
    ret['metrics'] = PerformanceMetrics.consolidate(perf_per_fold, test_= False)  # foreach metric, take average over CV folds
    return ret

def make_prediction_dataframe(y_pred, y_label, method, index):
    return DataFrame({'prediction':y_pred,'label':y_label, 'method': method, 'fold': index}, index=range(len(y_pred))) 

def run_simple_combiner(dataset, method='mean', file_type='posterior', n_runs=-1, **kargs):
    from cf_spec import System 
    import pandas as pd
    import utils_sys as us
    from tabulate import tabulate

    ret = {}   # output
    input_path = kargs.get('project_path', ProjectPath)

    aggregate_method = method  # a string or a function
    hasPerformanceDataframe = kargs.get('has_performance_dataframe', True)
    skipIfNotAvail = kargs.get('skip_if_not_avail', True)
    # n_fold = System.foldCount

    # tset, labels = common.readAll(input_path, dataset=method_id, file_type='predictions', exception_=True) # note: set exception_ to True to preclude multiple matches
    perf_per_fold = []
    dfs = []  # save data

    full_method = '?'
    index = 0
    if file_type.startswith(('pri', 'post')): 
        
        if hasPerformanceDataframe:

            method_id = dataset
            assert isinstance(method_id, str) and len(method_id) > 0

            sep = kargs.get('sep', ',')
            dset_type = file_type if file_type.startswith(('prior', 'post')) else 'prediction'

            fpath = '{path}/analysis/{stacker}.S-{dataset}-{suffix}.csv'.format(path=input_path, stacker=aggregate_method, dataset=method_id, suffix=dset_type)
            if not os.path.exists(fpath): 
                if skipIfNotAvail: 
                    print('(run_simple_combiner) Could not find method performance dataframe for method={method}, dataID={id}, dtype={dtype}'.format(
                        method=aggregate_method, id=method_id, dtype=file_type))
                    return {}
                else:
                    msg = "prediction dataframe not found | dtype: {dtype}, method_id: {id} | path: {path}".format(dtype=dset_type, id=method_id, path=fpath)
                    raise FileNotFoundError(msg)

            print('(run_simple_combiner) reading performance df:\n{path}\n   ... (verify)'.format(path=fpath))
            df = pd.read_csv(fpath, sep=sep, header=0, index_col=False) # error_bad_lines=True    # header: 0, no header
            indices = df['fold'].unique()
            # ... columns: prediction,label,method,fold,params
            
            full_method = '{stacker}.S-{dataset}-{suffix}'.format(stacker=aggregate_method, dataset=method_id, suffix=dset_type)
            for index, dfi in df.groupby(['fold', ]): 
                predictions = dfi.prediction.values
                labels = dfi.label.values
                # print('... index: {i}, size: {n} (type: {t1}=?={t2})'.format(i=index, n=len(predictions), t1=type(predictions), t2=type(labels)))

                perf, pv = analyzePerf(labels, predictions, method=full_method)
                perf_per_fold.append( perf )

        else: # otherwise, we have to estimate performance score here
            # if n_runs <= 0: n_runs = System.n_runs


            # # readAllIter() will attempt to resolve the indices by itself
            # for train_df, train_labels, test_df, test_labels in common.readAllIter(path=input_path, dataset=method_id, file_type=file_type, n_runs=n_runs): 
                
            #     # simple combiner very often do not even look at the training split 
            #     Th = test_df.values.T
            #     labels = test_labels # test_df.index.get_level_values('label').values

            #     if aggregate_method == 'mean': 
            #         predictions = combiner(Th, aggregate_func=np.mean)
            #     elif aggregate_method == 'median': 
            #         predictions = combiner(Th, aggregate_func=np.median)
            #     else:  # user provide a customize aggregation function (e.g. weighted average)
            #         customized_func = aggregation_func
            #         assert hasattr(customized_func, '__call__')
            #         aggregate_method = customized_func.__name__
            #         predictions = combiner(Th, aggregate_func=customized_func)

            #     # the algorithm behind the dset_id + combiner method
            #     # note: stacker naming format: '{prefix}_{dataset}_stacker'.format(prefix=method, dataset=dataset)
            #     full_method = '{prefix}_{dataset}_combiner'.format(prefix=aggregate_method, dataset=dataset)
            #     if file_type: 
            #         full_method = '{prefix}_{dataset}_combiner_{dtype}'.format(prefix=aggregate_method, dataset=dataset, dtype=file_type)
                
            #     perf_per_fold.append( analyzePerf(labels, predictions, method=full_method) )

            #     dfe = DataFrame({'prediction':predictions,'label':labels, 'method': aggregate_method, 'fold': index}, index=range(len(predictions)))
            #     dfe['label'] = dfe['label'].astype(int)
            #     dfe['fold'] = dfe['fold'].astype(int)
            #     dfs.append(dfe)
           
            #     index += 1 
            raise NotImplementedError

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

            pref, pv = analyzePerf(labels, predictions, method=full_method)
            perf_per_fold.append( perf )

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
    # perf = PerformanceMetrics()
    # if MFEnsemble.is_preference_data(dataset): 
    #     # if kargs.get('aug_data', False): 
    #     # perf = run_pref_stacker(dataset)  # use preference scores to combine the result in the test set
    #     ret = run_preference_combiner(dataset, **kargs)  # simple combining rule using preference scores as heuristics
    # else: 
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

                    perf, pv = analyzePerf(test_labels, predictions, method=full_method)
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
    # ... dataset is either a string or a dictionary (mapping from index to datasets)
    #     when dataset is a dictionary, each index can reference multiple datasets

    # probably always want to compare with base predictors 
    topk_bps = kargs.get('topk_bps', -1)
    baseline = base_predictors(topk=topk_bps, metric='fmax') if kargs.get('run_bp', False) else {} 
    if baseline: perfMetrics.append(baseline['metrics'])

    # candidate default stackers 
    # a. ['lasso', 'enet', 'rf', 'gb',  ]
    # b. ['log', 'qda', 'enet', 'svm', 'naive', 'rf', 'ada', 'knn', ]  # rf: random forest, 'gb': gradient boosting tree
    policy_iter = kargs.get('policy_iter', System.policy_iter)
    mode_evaluation = kargs.get('mode', 'train-test-split')
    file_type = kargs.get('file_type', '')
    performance_id = kargs.get('performance_id', '')  # <<< use this to name performance dataframe (rather than using dataset, which can be a dictionary)

    if isinstance(dataset, dict): 
        indices = sorted(dataset.keys())
    else: 
        # otherwise, indices depend on evaluation mode: i) 'cv': cross validation ii) 'subsampling'
        indices = kargs.get('indices', range(System.foldCount) if policy_iter == 'cv' else range(System.n_runs))

    # determine method_id when 

    for method in ['log',  ]:  # 'qda', 'enet', 'svm', 'naive', 'rf', 'ada', 'knn',  ]:  # special stackers:  
        print('(run_stacker) Running stacker {model} under evaluation mode: {mode} ...'.format(model=method, mode=mode_evaluation))

        ################################################################################################
        predictions_df = stacking.run(stacker=method, dataset=dataset, parallelize=kargs.pop('parallelize', True), 
            indices=indices, policy_iter=policy_iter, file_type=file_type, mode=mode_evaluation, performance_id=performance_id)  # stacking.run() -> stacked_generalization()
        ################################################################################################

        ### apply a scoring function to each fold and then take the average
        # predictions_df.groupby('fold').apply(lambda x: common.score(x.label, x.prediction)).mean()
        
        ### method naming
        # {stacker}.S-{dataset}-{suffix}
        # method_id = '{prefix}_{id}_stacker'.format(prefix=method, id=dataset)  # e.g. rf_bp_stacker
        if isinstance(dataset, dict): assert performance_id != ''

        method_id = "{stacker}.S-{dataset}-{suffix}".format(stacker=method, dataset=performance_id, suffix=file_type)
        if file_type.startswith('post'):
            print("(run_stacker) Stacking '{dtype}' dataset: {0} using stacker: {1}".format(dataset, method_id, dtype=file_type)) 
            # method_id = '{prefix}_{id}_stacker_{dtype}'.format(prefix=method, id=dataset, dtype=file_type) 

        # foreach CV fold
        perf_per_fold = []
        for name, group in predictions_df.groupby('fold'): 
            # mdict = evalTestSet(group['label'], group['prediction'], aggregate_func=np.mean, fold=name)
            # stackerMetrics.add(mdict)

            # perf.add(scores=mdict, method=method_id)  # use a dataframe to keep tracck of performance scores, which then faciliates plotting
            perf, pv = analyzePerf(group['label'], group['prediction'], method=method_id)
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
        docs = {'method': 'stacking', 'dataset': performance_id}
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

        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
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
     
        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
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
        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
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
            perf, pv = analyzePerf(L_test, Th, method=method_id, aggregate_func=np.mean, T=T, fold=fold)
            nmfMetrics[kind].append( perf )  # analyzePerf -> { compare* } where compare* is a set of analysis functions (e.g.  compareEstimates_
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
     
        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
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
            S = uc.eval_similarity_by_latent_factors(factors, epsilon=1e-9)
            
            # axis = 0 if kind == 'user' else 1
            dimS = n_users if kind == 'user' else n_items_total
            assert (S.shape[0] == S.shape[1] == dimS), "kind={0}, dim(S)={1} but expecting: {2}".format(kind, S.shape, dimS)
            print('(test) kind={0} | dim(S): {1} (S[i,j] in [0, 1]?):\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # [note] R, T are to be merged prior to calling predict_debiased or predict_topk
            Rh, Th = uc.predict(R, T, S=S, kind=kind, canonicalize=True)
            assert T.shape == Th.shape, "dim(T): {0} but dim(Th): {1}".format(T.shape, Th.shape)

            method_specific = MFEnsemble.name_sim_method(method, kind=kind, params=params) # '{base}_{kind}_sim'.format(base=base, kind=kind)

            perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
            nmfMetrics[kind].append(perf)

            nmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )  # plot K-Fold CV
            if kargs.get('save', True): uc.save_reconstructed_training_data(Rh, L_train, fold, method=method_specific, verify=True, U=U)

            ## use top K only 
            topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
            Rh, Th = uc.predict(R, T, S=S, kind=kind, topk=topk, canonicalize=True)
        
            kind_topk = '%s_topk' % kind
            method_specific = MFEnsemble.name_sim_method(method, kind=kind_topk, params=params) # '{base}_{kind}'.format(base=base, kind=kind_topk)

            perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
            nmfMetrics[kind_topk].append( pv )

            nmfCV[kind_topk].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )

            ## NMF + Clustering
            if kargs.get('run_clustering', False):
                for clustering in clusterings: 
                    kind_cluster = '%s_%s' % (kind, clustering)
                    method_specific = MFEnsemble.name_sim_method(method, kind=kind_cluster, params=params)
                    user_labels, item_labels = uc.nmfCluster(P, Q, n_clusters=params['n_factors'], method=clustering, evaluate=True)
                    
                    cluster_labels = user_labels if kind == 'user' else item_labels
                    Rh, Th = uc.predict_by_cluster(R, T, similarity=S, kind=kind, C=cluster_labels)

                    perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
                    nmfMetrics[kind_cluster].append(perf )
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

    # S = uc.eval_similarity_by_latent_factors(factors, epsilon=1e-9)
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

            Rc= uc.center(Ra, kind=kind) # Rc: c, centered
            S = uc.eval_similarity(Rc, kind=kind) # cosine similairty

            axis = 0 if kind == 'user' else 1
            assert S.shape[0] == Rc.shape[axis]
            print('(test) kind={0} | dim(S):{1} Sim:\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # full vs topk 
            Rh, Th = uc.predict(R, T, S=S, kind=kind, topk=None) # uc.predict_biased(T, S, kind=kind) # full similarity 
            
            method_specific = '{base}_{kind}_sim'.format(base=base, kind=kind)

            perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
            simMetrics[kind].append(perf)
            simCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )
        
            topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
            Rh_tokK, Th_topK = uc.predict(R, T, S=S, kind=kind, topk=topk) # uc.predict_topk(T, S, kind=kind, k=topk)  # S(top k)

            kind_topk = '%s_topk' % kind
            method_specific = '{base}_{kind}'.format(base=base, kind=kind_topk)        

            perf, pv = analyzePerf(L_test, Th_topK, method=method_specific, aggregate_func=np.mean)   
            simMetrics[kind_topk].append(perf)
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
                S = uc.eval_correlation(Ra, kind=kind, epsilon=1e-9, to_distance=False)
                Rh, Th = uc.predict(R, T, S=S, kind=kind, topk=None) # uc.predictNewItemsByCorr(T, R, L_train)
            elif kind.startswith('l'):  # predictions vs true labels 
                Rh = None # undefined
                Th = uc.predictByCorrWithLabels(T, R, labels=L_train, topk=None)

            # [log] Input contains NaN, infinity or a value too large for dtype('float64')
            method_specific = '{base}_{kind}_sim'.format(base=base, kind=kind)

            perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
            corrMetrics[kind].append(perf)
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

            perf, pv = analyzePerf(L_test, Th_topK, method=method_specific, aggregate_func=np.mean)
            corrMetrics[kind_topk].append(perf)
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
    #     R, T, L_train, L_test, U = uc.to_rating_matrix2(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
    
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

def run_cluster_analysis(F, U=None, X=None, kind='user', n_clusters=-1, index=0, params={}, **kargs):
    """
    Run cluster analysis with factor vectors. 

    Params
    ------
    F: factor matrix
    U: names of users or items (optional)
    X: rating matrix (optional)


    Memo
    ----
    1. typically used as a subroutine for wmf_ensemble*()  (e.g. wmf_ensemble_iter())

    2. better heapmap

       <ref> https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec

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
    def save_array(S, file_id='', name='', index=-1): # closure: kind, params, Domain
        # MFEnsemble.save_array(S, file_name='{method}_{kind}_S.csv'.format(method=method, kind=kind))
        if not name: name = '{kind}-sim'.format(kind=kind)
        if not file_id: file_id = MFEnsemble.get_dset_id(method='wmf', params=params)
        file_type = kargs.get('file_type', 'posterior')
        MFEnsemble.save_array(S, U=U, name=name, dset_id=file_id, file_type=file_type, sep=',', domain=Domain, index=index) # other params: project_path
    def compress_bags(n_max=5, infer_bagcount=False, sep='.'):  # closure: U, BagCount/global
        n_bagged_cls = sum([1 for col in U if len(str(col).split(sep))==2] )  # raise exception when col is not a strong (e.g. numbers)
        tBagged = True if n_bagged_cls > 0 else False

        # infer bag count if None
        bag_count = BagCount
        if tBagged and infer_bagcount: 
            counts = Counter([col.split(sep)[0] for col in U])
            bag_count = counts[list(counts.keys())[0]]

            # assuming that we do not mixed unbagged with bagged
            for name, count in counts.items(): 
                if count != bag_count: 
                    msg = "Inconsistent bag counts: %s" % counts
                    if exception_: 
                        raise ValueError(msg)
                    else: 
                        print(msg)
            print('(compress_bags) inferred bag_count=%d' % bag_count)

        idx = []
        bag_start_indices = range(0, len(U), bag_count)
        # names = [c.split(sep)[0] for c in U[bag_start_indices]]
        for i in bag_start_indices:
            idx.extend( list(range(i, i+n_max)) )

        idx = np.array(idx)

        Up = U[idx]
        Fp = F[idx, :]
        return Up, Fp
    def subset_index(U, keep_index=[0, 5], bagcount=10, sep='.'): 
        irange = []
        assert len(keep_index) > 0
        keep_index = [str(k) for k in keep_index]
        for u in U:
            name, *res = str(u).split(sep) 
            if len(res) > 0: 
                bag_number = res[0]
                if bag_number in keep_index: 
                    irange.append(u)
        print("(subset_index) bagged classifiers to keep:\n... {}".format(irange))
        return irange

    from analyze_similarity import plot_heatmap
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    from itertools import product 
    
    # parameters 
    tSaveData = kargs.get('save', True)
    tSavePlot = kargs.get('save_plot', False)
    output_path = kargs.get('output_path', '') # output path should include the file name
    tAnnotateSimDeg = kargs.get('annot', False)
    n_bag_cutoff = kargs.get('n_bag_cutoff', 5)  # keep at most 5 bagged results

    # file ID
    file_type = kargs.get('file_type', '')
    
    # number of clusters 
    nD, nF = F.shape  # number of users/items vs number of latent factors
    if X is not None: 
        if kind.startswith( ('u', 'cl', )):   # user, classifier 
            assert X.shape[0] == F.shape[0], "Inconsistent user dimension: {} vs {}".format(X.shape[0], F.shape[0])
        elif kind.startswith( ('i', 'd',) ):  # item, data
            assert X.shape[1] == F.shape[0], "Inconsistent item dimension: {} vs {}".format(X.shape[1], F.shape[1])
    if U is not None: 
        assert len(U) == F.shape[0], "Dimension inconsistency: len(U): {n} but dim(F): {dim}".format(n=len(U), dim=F.shape)
    else: 
        U = ['x{i}'.format(i=i) for i in range(nD)]
    print('(run_cluster_analysis) Running cluster analysis on {role} represented by {n} factors | dim(F): {dimF} | dim(X): (dimX): {dimX}'.format(role=kind, n=nF, dimF=F.shape, dimX=X.shape if X is not None else '?'))
    
    S = uc.eval_similarity_by_latent_factors(F, epsilon=1e-9)  # e.g. F <- P when examining user latent vectors
    
    # axis = 0 if kind == 'user' else 1
    dimS = S.shape[0]
    assert S.shape[0] == S.shape[1]
    print('(run_cluster_analysis) Kind={} | dim(S): {} (S[i,j] in [0, 1]?):\n{}\n'.format(kind, S.shape, S[:4, :4]))

    # I/O
    if tSaveData: 
        # save similarity matrix
        save_array(S, index=index)

    ### visualize similarity
    if tSavePlot: 
        df = DataFrame(S, columns=U, index=U)

        # subset dataframe  [todo]
        Usub = subset_index(U, keep_index=[0, 5], bagcount=10, sep='.')
        df = df.loc[Usub, Usub]

        if not output_path: 
            # default
            ext = 'pdf'
            if params: 
                dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)
                fname = '{prefix}.P-{dataset}-{suffix}-{index}.{ext}'.format(prefix='similarity', dataset=dset_id, suffix=file_type, index=index, ext=ext)
            else: 
                fname = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix='similarity', suffix=file_type, index=index, ext=ext)
            output_path = os.path.join(System.analysisPath, fname)
        else: 
            # if called externally, provide a customized output path that includes the file name
            assert os.path.exists(output_path)
        plot_heatmap(data=df, output_path=output_path, dpi=300, annot=tAnnotateSimDeg, mask_upper=False)

    ### WMF + Clustering
    clustering_methods = ['spectral', ]  # 'kmeans', product(*[kind[:2], clusterings])
    for clustering in clustering_methods:  # foreach clustering method
        kind_cluster = '%s-%s' % (kind, clustering)
        # method_specific = MFEnsemble.name_sim_method(method, kind=kind_cluster, params=params)

        # factors could be user-based or item-based (depending on kind)
        cluster_labels = uc.runClustering(F, n_clusters=n_clusters, method=clustering) 
    
    return S, cluster_labels

def reconstruct_by_preference(C, X, prefs, factors=[], labels=[], 
        test_labels=[], 
        polarity_matrix=None,   
        p_threshold=[],  
        pref_threshold=-1,
        policy_opt='rating', policy_replace='rating',
        policy_calibration='agreement', 
        replace_subset=True, replace_all=False, 
        is_test_set=False, binarize=True,
           is_cascade=False,  
           n_train=-1,  
               params={}, null_marker=0, 
               negative_pref=0.0, positive_pref=1.0, 
               name='X', verify=False, index=0,
               n_factors=-1, n_iter=30, unweighted=False, message=''):
    """
    Use the latent factors P and Q to reconstruct X which using C as a mask. 
    If policy_opt is set to 'preference', then the mask is computed via the input latent factors (P, Q) (instead of using C). 

    Params
    ------
    M: should be provided when (P, Q) is used to compute 'preference' (good proba vs bad proba)
    params: parameters passed down from the caller (e.g. n_iter, n_factors)

    test_labels: used for evaluting the preference calibration for test set

    for testing only: 

    is_cascade: if True, the input X is a concatenation of R and T
    n_train: the number of training examples; used to split X back to R and T
    name: 
    verify 
    index: usually corresponds to the tuple: (outer fold, fold)

    n_factors: used to compute latent factors for approximating ratings

    """
    import utils_als as ua
    import utils_cf as uc
    from sklearn.metrics import roc_auc_score  # test

    ##############################################
    replace_all = not replace_subset
    ##############################################
    P = Q = None
    tHasPrecomputedFactors = False
    if factors is not None: 
        try: 
            P, Q = factors 
            tHasPrecomputedFactors = True
        except: 
            raise ValueError("Factors must be in a 2-tuple format but given: {}".format(factors))

    Pp, Qp = prefs 

    # set canonicalize=True because the reconstructed probabilities could be not in [0.0, 1.0]
    Mc_ref = Lh_ref = None   # CM_ref, CM
    tHasTestLabels = True if len(test_labels)>0 else False
    if tHasTestLabels > 0:  
        Mc_ref, Lh_ref = uc.probability_filter(X, test_labels, p_threshold, estimate_labels=False)  # note: pass X (not Xp)

    ## 1. Pp, Qp as preference scores
    Xp = uc.compute_preference(Pp, Qp, binarize=False, canonicalize=False, verify=True) # use Tpf as a mask
    # ... Xp is just 'raw' preference scores

    if binarize: 

        # if X <- R, pref_threshold probably has not be determined ... 
        # ... but if X <- T, then pref_threshold will be the one to be determined here
        if pref_threshold < 0:   
            # estimate preference threshold from training data
            if len(labels) == 0:
                assert len(p_threshold) > 0, "Need probabilty threshold to estimate labels"
                labels = lh = uc.estimateLabels(X, L=[], p_th=p_threshold, pos_label=1)

            ##############################################
            # use factors to compute preference scores and find the preference threshold to binarize the scores
            
            # ... Xp is not yet binarized here
            assert Xp.shape == X.shape

            # >>> binarize
            Pf, Lh_X = uc.probability_filter(X, labels, p_threshold, estimate_labels=False)  # note: pass X (not Xp), Pf: (approx) correctness matrix
            Xp, pref_threshold, _ = uc.calibrate_preference(Xp, Pf=Pf, step=0.01, 
                policy=policy_calibration,  # options: agreement, {f-pref, f-align}, hit-to-miss, ... 
                
                Lh=Lh_X)  
            ##############################################
            # ... Xp is normalized and binarized

            # [test]
            # if true (test) labels were given, use it; o.w. construct an approximate correctness matrix
            correctness = Mc_ref if len(test_labels) > 0 else Pf # Mc_ref: correctness matrix VS Pf: (approx) correctness matrix 
            Lh = Lh_ref if tHasTestLabels else Lh_X

            print('(reconstruct_by_preference) Quality of the seed (X={})| pref_threshold: {} | test data? {}, policy_calibration: {}  ... Cycle: {}'.format(
                name, pref_threshold, is_test_set, policy_calibration, index))
            ret = uc.ratio_of_alignment2(Xp, correctness, Lh, verbose=True)  

            # assert abs(r - c_ratio) < 1e-3
            
    
        else: # almost never recommended because there is usually a large gap in the preference threshold between training and test splits
        
            # pref_threshold is given (e.g. estimated from training split R)
            Xp = uc.binarize_pref(Xp, p_th=pref_threshold, cutoff=True)
            print('(reconstruct_by_preference) Quality of the seed (X={}) | binarized directly | pref_threshold (given): {} | test split? {} | dim(X): {}'.format(name, 
                pref_threshold, is_test_set, X.shape))

            if verify: 
                assert len(p_threshold) > 0 and len(labels) > 0, "Missing either proba threshold or labels:\n... p_threshold: {}, labels: {}\n".format(p_threshold, labels)
                Pf, Lh_X = uc.probability_filter(X, labels, p_threshold)  # note: pass X (not Xp)

                correctness = Mc_ref if tHasTestLabels else Pf  # Pf: (approx) correctness matrix VS Mc_ref: correctness matrix
                Lh = Lh_ref if len(test_labels) > 0 else Lh_X

                print('(reconstruct_by_preference) Quality of the seed given pref_threshold | th(X): {} | test data? {}, policy_calibration: {}  ... Cycle: {}'.format(is_test_set, 
                    pref_threshold, policy_calibration, index))
                ret = uc.ratio_of_alignment2(Xp, correctness, Lh, verbose=True)  
        # ... Xp is binarized

        pref_scores = np.unique(Xp)
        assert len(pref_scores) == 2, "(reconstruct_by_preference) Degenerated preference scores, containing only {}".format(pref_scores)
        # if len(pref_scores) < 2: print("(reconstruct_by_preference) Warning: degenerated preference scores, containing only {}".format(pref_scores))
    else: 
        # Xp is continous, probabilty-like matrix
        # Xp = canonicalize_pref(Xp, binarize=False, name=name, verify=1, min_score=negative_pref, max_score=positive_pref)
        
        # # ... Rh, Th are normalized (to [0, 1])
        # n_uniq_test = len(np.unique(Xp))
        # assert np.max(Xp) <= 1.0 and np.min(Xp) >= 0.0
        # assert n_uniq_test > 2 

        raise ValueError("Xp must be a binary matrix.")

    ################################################
    Mask = Xp  # binarized Xp is a mask
    n_masked = uc.verify_mask(Mask)  
    n_zeros = np.sum(Mask==0)
    n_ones = np.sum(Mask==1)
    if not is_test_set: assert n_zeros > 0

    N = Mask.shape[0]*Mask.shape[1]+0.0
    print("(reconstruct_by_preference) replacement mode: replacing bad 'ratings' | n(zeros)<X>: {} =?= n_masked: {}, ratio: {}) | n(ones)<X>: {}, ratio(preferred): {} | dim(X): {}".format(
                n_zeros, n_masked, n_zeros/N, n_ones, n_ones/N, X.shape))

    # before replacing with new estimates, we need to ask if each datum has non-zero "support"
    # ... tHasDegenerated = any(np.sum(Xp, axis=0) == 0)  # ncol_zeros > 0 
    wcol_sum_to_zero = np.sum(X, axis=0) == 0
    isDegenerated = any(wcol_sum_to_zero)  # ncol_zeros > 0 
    wcol_idx = np.where(wcol_sum_to_zero)[0]
    if isDegenerated: 
        # it's possible that none of the classfier's predictions for a given data point was consider "reliable"; hence, some columns are all zeros
        
        # revert to average? 
        # for j in range(Xp.shape[1]): 
        #     if all(Xp[:, j]==0): 
        #         Xp[:, j] = 0.5  # all equal weights  
        print('(reconstruct_by_preference) Found degenerated cases: {} columns are all zeros!'.format(len(wcol_idx)))
    # ... all columns in Xpf has at least one non-zero 

    ################################################
    Cn = C
    if not tHasPrecomputedFactors: 
        print('(reconstruct_by_preference) factors not given | computing P, Q based on preference scores (via Xp) to reestiamte rating table X ...')
        assert binarize, "Xp must have been binarized ..."
        # Cn = C.toarray() if scipy.sparse.issparse(C) else np.copy(C)
        # # ... C.toarray() makes a new copy in dense format
        # Cn[Xp==0]=0.0
        # if unweighted: Cn[Xp > 0] = 1.0
        # Cn = scipy.sparse.csr_matrix(Cn)  # Cn has to be in sparse format now

        P, Q, *Xh_errs = ua.implicit_als(Cn, features=n_factors, 

                            iterations=n_iter,
                            lambda_val=System.lambda_val,  # 0.8 by default

                            # label_confidence=Cx_bar, 
                            polarity=polarity_matrix,   # polarity is not used when approximting ratings
                            p_threshold=p_threshold, 
                            ratings=X, labels=labels,
                            policy='rating', message=message, ret_rmse=True)
        Xh_err, Xh_err_weighted = Xh_errs

        ne = 2
        e_pri, e_post = np.mean(Xh_err[:ne]), np.mean(Xh_err[-ne:])
        e_del = (e_pri-e_post)/e_pri * 100
        print('(reconstruct_by_preference) ALS 1 | Complete | rmse: {e1} -> {e2} (n_errs: {n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(e1=e_pri, e2=e_post, 
                       e_del=e_del, ew1=np.mean(Xh_err_weighted[:ne]), ew2=np.mean(Xh_err_weighted[-ne:]), n=len(Xh_err) ))
    ################################################################################################
    assert P is not None and Q is not None
    # ... now we definitely have (P, Q) to re-estimate proba values

    # Use X'<- dot(P, Q.T) as new proba estimtes to replace values in X according to Xp
    # essentially, the final estimate Xh is a weighted average between X' and X, for which weights are given by Xp
    Xs = uc.predict_by_factors(P, Q, canonicalize=True, name='Xs')
    # ... raw re-estimated

    # Xh = uc.replace(P, Q, X=(Xp, X), canonicalize=True, 
    #             fill=null_marker, predict_func=ua.predict_by_factors, name=name)
    Xh = uc.interpolate(X, Xs, Xp, 1.0-Xp) # interpolate between X (original) and Xs (reconstructed) based on Xp (preference scores)
    # ... Xh = Xp * X_old + (1.0-Xp) * X_new 
    #        = Xp * X  + (1-Xp) * dot(P, Q)
    #        = Xp * X  + (1-Xp) * Xs
    # ...... Xh is a new rating/probability matrix

    #############################################################################################################################
    # [test]
    CMr = None 
    Po = uc.from_color_to_polarity(polarity_matrix) # external polarity matrix
    # ... from_color_to_polarity() also works with {-1, 0, 1}-polarity matrix
    Pr = Pt = None
    if verify: 
        if is_cascade: 
            assert n_train > 0
            Cr, Ct = Cn[:, :n_train], Cn[:, n_train:]
            R, T = X[:, :n_train], X[:, n_train:]
            Rp, Tp = Xp[:, :n_train], Xp[:, n_train:]

            ################################################
            if Po is not None: 
                if scipy.sparse.issparse(Po): Po = Po.toarray()
                assert Po.shape == Cn.shape
                
                polarities = np.unique(Po)
                n_colors = len(polarities)
                assert n_colors <= 3

                n_pos = np.sum(Po == 1)
                n_neutral = np.sum(Po == 0)
                n_neg = np.sum(Po == -1)

                msg = ''
                msg += "(reconstruct_by_preference) n(pos): {}, n(neg): {}, n(neutral): {}\n".format(n_pos, n_neg, n_neutral)

                # Po[Po < 0] = 0  # so that it can be used as preference score
                Pr, Pt = Po[:, :n_train], Po[:, n_train:]

                n_pos = np.sum(Pt == 1)
                n_neutral = np.sum(Pt == 0)
                n_neg = np.sum(Pt == -1)                
                msg += "... T -> Pt | n(pos): {}, n(neg): {}, n(neutral): {}\n".format(n_pos, n_neg, n_neutral)
                print(msg)
            ################################################
     
            div("(reconstruct_by_preference) Quality of the seed after splitting X into R, T | test labels? {}".format(tHasTestLabels), symbol='#', border=1)

            msg = ''
            # use L_test to get correctness matrix
            L = test_labels if tHasTestLabels else labels  # if true labels were given (in regards to test set) then use it otherwise use the estimted labels  
            Lr, Lt = L[:n_train], L[n_train:]
            assert len(Lr) == R.shape[1] and len(Lt) == T.shape[1], "Inconsistent sample sizes size(Lt): {}, size(T): {}".format(len(Lt), T.shape[1])
            
            ### test R?
            CMr, Lhr = uc.probability_filter(R, Lr, p_threshold)  # Lh(T, p_threshold)
            ret = uc.ratio_of_alignment2(Rp, CMr, Lhr, verbose=True, message='R') 
            # what happens if we were to make predictions based on weighted average (by selecting only preferred entreis)
            # pvr = np.average(R, weights=Rp, axis=0)  # ... weights sum to zero
            pvr = uc.predict_by_preference(R, Rp, name='R', aggregate_func='mean', fallback_on_low_weight=False, verify=True)
            pvr_weighted = uc.predict_by_preference(R, Rp, W=Cr, name='Rw', aggregate_func='mean', fallback_on_low_weight=False, verify=True)
            fmax_ref = common.fmax_score(Lr, np.mean(R, axis=0), beta = 1.0, pos_label = 1)
            fmax_r = common.fmax_score(Lr, pvr, beta = 1.0, pos_label = 1); 
            fmax_r_weighted = common.fmax_score(Lr, pvr_weighted, beta = 1.0, pos_label = 1);
            msg += '(by_preference) Quality of the seed X -> (R) | th(R): {} | fmax(base): {}, fmax(R): {}, fmax(Rw): {} ... Cycle: {}\n'.format(pref_threshold, 
                fmax_ref, fmax_r, fmax_r_weighted, index)
            ###################################################
            
            ### test T
            CMt, Lht = uc.probability_filter(T, Lt, p_threshold)  # find ratio of alignment wrt true labels (not the estimted labels lh) because we want to find how preference matrix is aligned wrt to true labels 
            ret = uc.ratio_of_alignment2(Tp, CMt, Lht, verbose=True, message='T')  # [note] use message to avoid cluttered I/O in parallel computing
            pvt = uc.predict_by_preference(T, Tp, name='T', aggregate_func='mean', fallback_on_low_weight=False, verify=True)
            pvt_weighted = uc.predict_by_preference(T, Tp, W=Ct, name='Tw', aggregate_func='mean', fallback_on_low_weight=False, verify=True) 
            fmax_ref = common.fmax_score(Lt, np.mean(T, axis=0), beta = 1.0, pos_label = 1)
            fmax_t = common.fmax_score(Lt, pvt, beta = 1.0, pos_label = 1)
            fmax_t_weighted = common.fmax_score(Lt, pvt_weighted, beta = 1.0, pos_label = 1)
            msg += '(by_preference) Quality of the seed X -> (T) | th(T): {} | fmax(base): {}, fmax(T): {}, fmax(Tw): {} ... Cycle: {}\n'.format(pref_threshold, 
                fmax_ref, fmax_t, fmax_t_weighted, index)

            ###################################################
            if Po is not None: 
                # use the polarity matrix directly as the preference
                # Wt = Pt * Ct
                
                # double check 
                ret = uc.eval_polarity(Pt, CMt, Lht, verbose=True, name='Tpo', neg_po=-1, title='(by_preference) Quality of the polarity matrix (Po)')

                # how does it compare to majority votes? 
                lh = estimateLabels(T, p_th=p_threshold, pos_label=1) 
                Mct_max, Lht_max = probability_filter(T, lh, p_threshold)  # Pf is a (0, 1)-matrix
                ret = uc.eval_polarity(uc.preference_to_polarity(Mct_max), CMt, Lht, verbose=True, name='Tmax', neg_po=-1, title='(by_preference) -- {}: majority votes --'.format(name))

                # make predictions via the preference matrix
                Prt = uc.to_preference(Pt)
                pvt_po = uc.predict_by_preference(T, Prt, name='Tpo', aggregate_func='mean', fallback_on_low_weight=False, verify=True) 
                pvt_po_weighted = uc.predict_by_preference(T, Prt, W=Ct, name='Tpo', aggregate_func='mean', fallback_on_low_weight=False, verify=True)

                fmax_t = common.fmax_score(Lt, pvt_po, beta = 1.0, pos_label = 1)
                fmax_t_weighted = common.fmax_score(Lt, pvt_po_weighted, beta = 1.0, pos_label = 1);
                msg += '(by_preference) Quality of the polarity matrix (Po) | fmax(base): {}, fmax(T): {}, fmax(Tw): {} ... Cycle: {}\n'.format(fmax_ref, fmax_t, fmax_t_weighted, index)
            ###################################################

            ### overall X 
            CMx, Lhx = uc.probability_filter(X, L, p_threshold)  # find ratio of alignment wrt true labels (not the estimted labels lh) because we want to find how preference matrix is aligned wrt to true labels 
            ret = uc.ratio_of_alignment2(Xp, CMx, Lhx, verbose=True, message='X: (R|T)')  # [note] use message to avoid cluttered I/O in parallel computing
            # pvx = np.average(X, weights=Xp, axis=0) # ... weights may sum to zero
            pvx = uc.predict_by_preference(X, Xp, name='X', aggregate_func='mean', fallback_on_low_weight=False, verify=True)
            pvx_weighted = uc.predict_by_preference(X, Xp, W=Cn, name='Xw', aggregate_func='mean', fallback_on_low_weight=False, verify=True)
            fmax_ref = common.fmax_score(L, np.mean(X, axis=0), beta = 1.0, pos_label = 1)
            fmax_x = common.fmax_score(L, pvx, beta = 1.0, pos_label = 1)
            fmax_x_weighted = common.fmax_score(L, pvx_weighted, beta = 1.0, pos_label = 1);
            msg += '... Quality of the seed X: R+T | th(T)=th(R): {} | policy_calibration: {} | given test/true labels? {} | fmax: {} ... Cycle: {}\n'.format(pref_threshold, 
                policy_calibration, tHasTestLabels, fmax_x, index)

            msg += "-" * 80 + '\n'
            ### Test Xs
            # raw reestimates
            msg += "(by_preference) Quality of the reconstructed rating matrix (Xs); compare this with fmax computed earlier\n"
            Pvr, Pvt = Xs[:,:n_train], Xs[:,n_train:]   # reconstructed
            pvr, pvt = np.mean(Pvr, axis=0), np.mean(Pvt, axis=0)
            fmax_r = common.fmax_score(Lr, pvr, beta = 1.0, pos_label = 1)
            fmax_t = common.fmax_score(Lt, pvt, beta = 1.0, pos_label = 1)
            fmax_r_ref = common.fmax_score(Lr, np.mean(R, axis=0), beta = 1.0, pos_label = 1)
            fmax_t_ref = common.fmax_score(Lt, np.mean(T, axis=0), beta = 1.0, pos_label = 1)
            msg += "... mean(Xs) | fmax(baseline): (R: {}, T: {}), fmax(Rs): {}, fmax(Ts): {}".format(fmax_r_ref, fmax_t_ref, fmax_r, fmax_t)

            msg += "-" * 80 + '\n'
            ### Test Xh
            Pvr, Pvt = Xh[:,:n_train], Xh[:,n_train:]
            msg += "(reconstruct_by_preference) Quality of the reconstructed matrix via preference (Xh); compare this with fmax computed earlier\n"
            pvr, pvt = np.mean(Pvr, axis=0), np.mean(Pvt, axis=0)
            fmax_r = common.fmax_score(Lr, pvr, beta = 1.0, pos_label = 1)
            fmax_t = common.fmax_score(Lt, pvt, beta = 1.0, pos_label = 1)
            msg += "... mean(Xh) | fmax(Rh): {}, fmax(Th): {}".format(fmax_r, fmax_t)
            
            print(msg)
            ##########################################
        else: 
            L = test_labels if tHasTestLabels else labels  # if true labels were given (in regards to test set) then use it otherwise use the estimted labels 
            
            CM, Lh = uc.probability_filter(X, L, p_threshold)  # Lh(T, p_threshold)
            # CMr, CMt = CM[:, :n_train], CM[:, n_train:]
            print('(reconstruct_by_preference) Quality of the seed X -> ({X}) | th({X}): {th} | policy_calibration: {policy} ... Cycle: {c}'.format(X=name,
                th=pref_threshold, policy=policy_calibration, c=index))
            ret = uc.ratio_of_alignment2(Xp, CM, Lh, verbose=True, message='X') 
        #############################################################################################################################
    #### end test
        
    if not tHasPrecomputedFactors: 
        return Xh, Xp, pref_threshold, P, Q
    
    return Xh, Xp, pref_threshold

def reconstruct(C, X, P, Q, 
        Pc=None, L=[], p_threshold=[],
            use_confidence_weights=False,
            policy_opt='rating', policy_replace='rating', 
            is_cascade=True,
            replace_subset=True, replace_all=False, params={}, null_marker=0, name='X', **kargs):
    """
    Use the latent factors P and Q to reconstruct X which using C as a mask. 
    If policy_opt is set to 'preference', then the mask is computed via the input latent factors (P, Q) (instead of using C). 

    Parameters
    ----------
    Pc: Color matrix encoding TP, TN, FP, FN (and neutral) for X
        Note that `Pc` should be provided when (P, Q) is used to compute 'preference' (reliable proba vs unreliable proba)
    params: parameters passed down from the caller (e.g. n_iter, n_factors)

    """
    import utils_als as ua
    import utils_cf as uc
    import scipy

    verbose = kargs.get('verbose', 1)
    ##############################################
    replace_all = not replace_subset
    ##############################################
    is_test_set = kargs.get('is_test_set', False)
    n_train = kargs.get('n_train', -1)
    # tRecalibrate = kargs.get('recalibrate', False)

    # set canonicalize=True because the reconstructed probabilities could be not in [0.0, 1.0]
    Xh = None
    if policy_opt.startswith('pref'):
        ## 1. (Rh, Th) as preference scores

        # p_th: preference threshould
        binarize = kargs.get('binarize', False)
        pref_th = kargs.get('pref_threshold', -1) 
        
        Xh = Mask = uc.compute_preference(P, Q, binarize=binarize, p_th=pref_th, canonicalize=False, verify=True) # use Tpf as a mask
        assert Mask.shape == X.shape
        # ... Xh: raw preference matrix; apply binarize operation after this call (via preference calibration)

        # assert len(np.unique(Xh)) == 2, "Xh should a binary matrix but got values: {}".format(np.unique(Xh))
        # Rh, Th represent preference scores, not probabilities
        if verbose: div("(reconstruct) Reconstructed matrix holds preference scores (binarized? {}, threshold given? {}) | is T? {}".format(binarize, pref_th < 0, is_test_set), symbol='#')
        
        #@ 2. (Rh, Th) as ratings (using preference scores to select & mask entries)
        # if replace_subset:   # if not in prediction mode, then we are in the mode of replacing bad rating values
        #     print("... replace bad 'ratings' by preference scores.\n...... computing new scores > policy_replace: {0}".format(policy_replace)) # replace by what ? e.g. rating

        #     # reconstructed ratings: overwrite preference vectors learned earlier # 
        #     Pt, Qt = ua.implicit_als(C, ratings=X, labels=[],
        #                 iterations=params['n_iter'], features=params['n_factors'], 
        #                 policy=policy_replace) # the policy tells as what we are approximating: e.g. 'rating'

        #     # >>> use Rh and Th only to mark 'bad scores' but use Rp, Tp as new scores 
        #     n_zeros = np.sum(Mask==null_marker)
        #     assert n_zeros > 0
        #     # >>> instead of using Cui as a mask, use M (consisting of preference scores)
        #     #     => replace Ra: (R, T?) by reconstructed scores via latent factors (Pp, Qp)

        #     print("... replacement mode: replacing bad 'ratings' | (n_zeros: {0}, ratio: {1}) | dim(T): {3}".format(
        #         n_zeros, n_zeros/(Mask.shape[0]*Mask.shape[1]+0.0), X.shape))

        #     # >>> use M to select entries in Ra that are potentially "bad" and replace these entries by the values given by
        #     #     the latent factors: (Pp, Qp)
        #     Xh = uc.replace(Pt, Qt, X=(Mask, X), canonicalize=True, 
        #                        fill=null_marker, predict_func=ua.predict_by_factors, name=name)
        # else: 
        #     # augmented data representation is not supported now 
        #     msg = "Augmented data representation is not supported now.\n"
        #     msg += "Preference scores are only used to represent a MASK, which separates reliable and unreliable ratings >> We need to compute new rating estimates in order to replace unreliable rating entries in X.\n"
        #     raise ValueError(msg)

        # use the mask to make final prediction 
        # Xh = X * Mask

        # Xh = uc.predict_by_preference(X, Xh, canonicalize=True, verify=verify)

    else: 
        ### predict or replace? 
        #   if 'predict_probs': True, then we reconstruct the entire matrix matrix via the latent factors
        #   if 'replace_subset': True, then we only replace 'bad entries' in the original rating matrix by the new approximation given by the latent factors
        if replace_all: # params['predict_probs']: 
            Xh = ua.predict_by_factors(P, Q, canonicalize=True)
            if verbose: div("(reconstruct) reconstructing the entire proba table ...", symbol='#')
        else: # reconstructing only (replace 'bad probabilities')
            # assert params['replace_subset'] == True 
            # case 1: Cui ~ R => Th: None, reconstructed R only
            # case 2: Cui ~ np.hstack((R, T)) => reconstructed (R, T)
            if verbose: div("(reconstruct) weighted averaing between X/original and Xh/new, where Xh = dot(P, Q) | use confidence matrix (C) as weights? {}".format(use_confidence_weights), symbol='#')
            
            W = Pc # Pc is a color matrix
            if use_confidence_weights: # Use confidence scores as weights 
                if scipy.sparse.issparse(C): C = C.toarray()
                W = uc.softmax(C, axis=0)
            else: 
                if W is None: 
                    W, _ = uc.probability_filter(X, L, p_threshold)
                else: 
                    W = Pc.A if scipy.sparse.issparse(Pc) else Pc
                    # Note: Why converting to dense? Subtracting a sparse matrix from a nonzero scalar is not supported 
                    #       E.g. can't do 1.0-W if W is sparse

                    if verbose > 1: 
                        print('(reconstruct) Converting color matrix to a standard probability filter (aka preference matrix) ...')
                    W = uc.to_preference(W) 

            wmin, wmax = np.min(W), np.max(W)
            assert wmin >= 0 and wmax <= 1, "W is not a probability filter | values: [{}, {}]".format(wmin, wmax)

            # Note: 
            # W as a weight matrix: the higher the W[i,j], the more weight on X[i,j]
            # W as a preference matrix: W[i,j] = 1 => use X[i,j] (original value), if W[i,j] = 0, use Xh[i,j] (re-est value)

            # Xh = uc.replace(P, Q, X=(W, X), canonicalize=True, 
            #         fill=null_marker, predict_func=ua.predict_by_factors, name=name)

            Xs = ua.predict_by_factors(P, Q, canonicalize=True)
            Xh = uc.interpolate(X, Xs, W, 1.0-W) # replace X by Xs selectively according to W (and 1-W)

        # [test] 
        test_labels = kargs.get('test_labels', [])
        index = kargs.get('index', -1)  # outer CV index
        if verbose and (is_cascade and len(L) > 0):

            # if scipy.sparse.issparse(Pc): 
            #     Po = np.copy( Pc.toarray() )
            # else: 
            #     Po = np.copy( Pc )
            # Pc[Pc < 0] = 0  # these entries are to be replaced by new estimates: P'Q

            msg = ''
            assert n_train > 0
            if len(test_labels) == 0: test_labels = L

            if scipy.sparse.issparse(C): C = C.toarray()
            W = uc.softmax(C, axis=0)
            assert W.shape == C.shape, "(reconstruct) dim(W): {}, dim(C): {}".format(W.shape, C.shape)
            # print("(reconstruct) dim(W): {}, dim(C): {}".format(W.shape, C.shape))

            Cr, Ct = C[:,:n_train], C[:,n_train:]

            R, T = X[:,:n_train], X[:,n_train:]
            Lx = test_labels
            Lr, Lt = test_labels[:n_train], test_labels[n_train:]
            Rh, Th = Xh[:,:n_train], Xh[:,n_train:]
             
            fmax_r_ref = common.fmax_score(Lr, np.mean(R, axis=0), beta = 1.0, pos_label = 1)
            fmax_t_ref = common.fmax_score(Lt, np.mean(T, axis=0), beta = 1.0, pos_label = 1)

            pvr, pvt = np.mean(Rh, axis=0), np.mean(Th, axis=0)
            fmax_r = common.fmax_score(Lr, pvr, beta = 1.0, pos_label = 1)
            fmax_t = common.fmax_score(Lt, pvt, beta = 1.0, pos_label = 1)

            pvr_weighted = uc.combiner(Rh, weights=Cr, aggregate_func='mean')
            pvt_weighted = uc.combiner(Th, weights=Ct, aggregate_func='mean')
            fmax_r_weighted = common.fmax_score(Lr, pvr_weighted, beta = 1.0, pos_label = 1)
            fmax_t_weighted = common.fmax_score(Lt, pvt_weighted, beta = 1.0, pos_label = 1)

            msg += '(reconstruct) Quality of Rh | fmax(Rbase): {}, fmax(Rh): {}, fmax(Rhw): {}\n'.format(fmax_r_ref, fmax_r, fmax_r_weighted)
            msg += '...           Quality of Th | fmax(Tbase): {}, fmax(Th): {}, fmax(Thw): {}\n'.format(fmax_t_ref, fmax_t, fmax_t_weighted)

            fmax_x_ref = common.fmax_score(test_labels, np.mean(X, axis=0), beta = 1.0, pos_label = 1)
            pvx = np.mean(Xh, axis=0)
            fmax_x = common.fmax_score(Lx, pvx, beta = 1.0, pos_label = 1)
            pvx_weighted = uc.combiner(X, weights=C, aggregate_func='mean')
            fmax_x_weighted = common.fmax_score(Lx, pvx_weighted, beta = 1.0, pos_label = 1)

            msg += "...           Quality of Xh | fmax(Xbase): {}, fmax(Xh): {}, fmax(Xhw): {}\n".format(fmax_x_ref, fmax_x, fmax_x_weighted)

            # what if we replace the whole table? 
            Xh2 = ua.predict_by_factors(P, Q, canonicalize=True)
            Rh2, Th2 = Xh2[:,:n_train], Xh2[:,n_train:]
            assert Rh2.shape == R.shape, "dim(R): {} =?= dim(Rh2): {}".format(R.shape, Rh2.shape)

            pvx2 = np.mean(Xh2, axis=0)
            pvr2, pvt2 = np.mean(Rh2, axis=0), np.mean(Th2, axis=0)
            fmax_r = common.fmax_score(Lr, pvr2, beta = 1.0, pos_label = 1)
            fmax_t = common.fmax_score(Lt, pvt2, beta = 1.0, pos_label = 1)
            fmax_x = common.fmax_score(test_labels, pvx2, beta = 1.0, pos_label = 1)

            # W = common.softmax(C, axis=0)
            pvr_weighted2 = uc.combiner(Rh2, weights=Cr, aggregate_func='mean')
            pvt_weighted2 = uc.combiner(Th2, weights=Ct, aggregate_func='mean')
            pvx_weighted2 = uc.combiner(Xh2, weights=C, aggregate_func='mean')
            fmax_r_weighted = common.fmax_score(Lr, pvr_weighted2, beta = 1.0, pos_label = 1)
            fmax_t_weighted = common.fmax_score(Lt, pvt_weighted2, beta = 1.0, pos_label = 1)
            fmax_x_weighted = common.fmax_score(test_labels, pvx_weighted2, beta = 1.0, pos_label = 1)

            msg += '(reconstruct) Quality of Rh + weights | fmax(Rbase): {}, fmax(Rh2): {}, fmax(Rw2): {}\n'.format(fmax_r_ref, fmax_r, fmax_r_weighted)
            msg += '...           Quality of Th + weights | fmax(Tbase): {}, fmax(Th2): {}, fmax(Tw2): {}\n'.format(fmax_t_ref, fmax_t,  fmax_t_weighted)
            msg += '...           Quality of Xh + weights | fmax(Xbase): {}, fmax(Xh2): {}, fmax(Xw2): {}\n'.format(fmax_x_ref, fmax_x, fmax_x_weighted)
            msg += '--- Xh2: all entries re-estimated, Xw2: weighted by C ---\n'

            #############################################################
            Po = uc.from_color_to_polarity(Pc, verify=True)
            Pr, Pt = Po[:,:n_train], Po[:,n_train:]  # Mcr, Mct 

            Mct, Lht = uc.probability_filter(T, Lt, p_threshold)
            ret = uc.eval_polarity(Pt, Mct, Lht, verbose=True, name='Tpo', neg_po=-1, title='(reconstruct) Quality of the polarity matrix (Po)')

            # use the polarity matrix directly as the preference
            # Wt = Pt * Ct
            # vmin, vmax = np.min(Pt), np.max(Pt)
            # assert vmin == -1
            Prt = uc.to_preference(Pt)  # polarity matrix to preference matrix
            vmin, vmax = np.min(Prt), np.max(Prt)
            assert vmin == 0.0 and vmax == 1.0, "Prt is not a valid preference matrix | vmin: {}, vmax: {}".format(vmin, vmax)
            
            pvt_po = uc.predict_by_importance_weights(T, Prt, aggregate_func='mean') # np.average(T, weights=Mct, axis=0)
            pvt_po_weighted = uc.predict_by_importance_weights(T, Prt * Ct, aggregate_func='mean')

            fmax_t = common.fmax_score(Lt, pvt_po, beta = 1.0, pos_label = 1)
            fmax_t_weighted = common.fmax_score(Lt, pvt_po_weighted, beta = 1.0, pos_label = 1);
            msg += '(reconstruct) Quality of the polarity matrix (Po) | fmax(base): {}, fmax(TPo): {}, fmax(TPow): {} ... Cycle: {}\n'.format(fmax_t_ref, fmax_t, fmax_t_weighted, index)

            print(msg)
        else: 
            pass
    
    return Xh

def wmf_ensemble_iter(data, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', fold=-1, outer_fold=-1, null_marker=0, verbose=1, 
        project_path='?', piggyback=True, dev_ratio=0.2, max_dev=5000, aggregation_methods=[], 
        post_hoc_analysis=False, save_data=False, enable_cluster_analysis=False):
    """
    
    Params
    ------
    fold: defined in a more generic sense than the 'fold' in wmf_ensemble_fold(). 
          specifically, 'fold' can refer to any of the following: 

           i) a CV fold ii) the index of iterations in random subsampling iii) other iteration index
    outer_fold: the iteration/fold number of the outer loop when wmf_ensemble_iter() is invoked for model selection (e.g. by model_select_core())

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
        div('(wmf_ensemble_iter) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2): 
            assert params['supervised']
        if params['setting'] in (3, 4, 9, 10): 
            assert not params['supervised']
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
    def name_params_setting(method_params=['F', 'A']):   # [todo]
        # MFEnsemble.abridge(meta, identifiers=method_params, replace=replacement, verify=True)

        # use MFEnsemble.params_to_ids
        return 'F{nf}A{a}'.format(nf=params['n_factors'], a=params['alpha'])

    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    import stacking

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
    # verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    ### Confidence Matrix
    # fold: iteration number (here, 'fold' is simply a repurposed variable borrowed from CV)
    # data: (R, T, L, Lt, U)-format
    if isinstance(data, DataFrame): # in this case, 'data' is a dataframe comprising a train-dev split 
        R, T, L_train, L_test, U =  uc.shuffle_split(data, ratio=dev_ratio, max_size=max_dev)
    else:
        # otherwise, assuming that shuffle-split operation had been invoked to produce the data input
        assert isinstance(data, (tuple, list)), "Invalid input data: %s" % data 
        print('(wmf_ensemble_iter) Input data is an n-tuple, whrere n={n}'.format(n=len(data)))
        R, T, L_train, L_test, U = data 
    n_samples = R.shape[1]+T.shape[1]; assert len(L_train)+len(L_test) == n_samples

    ### create extra prediction vectors (PVs) from R (say, mean vector) and attach these new PVs to R (piggyback)
    n_users, n_items = R.shape
    n_classifiers = int(n_users/BagCount)
    n_users0, n_items0 = n_users, n_items  # keep a copy of the original number of users and items
    n_users_test, n_items_test = T.shape
    n_meta_users = 0

    # preference parameters
    isPreferenceScore = params['policy_opt'].startswith('pref') # and not params['replace_subset'] 
    tPreferenceCalibration = isPreferenceScore and params['binarize_pref'] # params.get('preference_calibration', True)
    pref_threshold = pmax = params.get('pref_threshold', -1)
    tWeightedPrediction = params.get('weighted_output', False)

    ############################################################
    # piggy back extra meta-estimators
    meta_users = ['latent_mean', 'latent_mean_masked',]  # todo
    if tMetaUsers: 
        div(message='(wmf_ensemble_iter) Adding meta users (n={n})...'.format(n=len(meta_users)))
        
        # e.g. mean, and masked mean
        nU, nUT = n_users, n_users_test

        print('... augmenting T (by meta users)')
        mean_pv = make_prediction_vector(T, L=[], policy='none')   # policy: 'none' => no masking
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

    # rank transformation 

    ############################################################
    # compute confidence matrix for R
    CR = uc.evalConfidenceMatrix(R, L=L_train, U=U,  
            ratio_users=params['ratio_users'], 
            ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< determines the subroutine for computing Cui
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], 
                        alpha=params['alpha'], beta=params.get('beta', 1.0),
                        estimated_labels=False, 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, fold=fold) # project_path=System.projectPath 
    Cr, Mcr, p_threshold, *CR_res = CR    # note: No Cr_bar
    ############################################################
    assert len(p_threshold) > 0

    # ... Cui_bar is only used in policy = 'tradeoff'
    # verify_confidence_matrix(Cr, Mcr)    
    
    div("(wmf_ensemble_iter) Completed C(R) | Cycle {cycle} |  dim(Cui): {dim}, filter_axis: {fdim} | conf_measure: {measure}, optimization: {opt} | predict ALL probabilities? {tval} | policy_threshold: {p_th}".format(
        cycle=(outer_fold, fold), 
            dim=str(Cr.shape), fdim=params['policy'], measure=params['conf_measure'], opt=params['policy_opt'], tval=params['predict_probs'], p_th=params['policy_threshold']), symbol='#', border=1)
    print('... Model | n_factors: {nf}, alpha: {a} | predict pref? {pref}, binarize? {bin}, supervised? {s}'.format(nf=params['n_factors'], a=params['alpha'], pref=isPreferenceScore, bin=params['binarize_pref'], s=params['supervised']))
    print('... Data | n_train: {ntr}, n_test: {nt}, ratio: {r}'.format(ntr=len(L_train), nt=len(L_test), r=(len(L_train)+len(L_test))/(len(L_test)+0.0)) )
    print('... Classifiers | n_users: {nu}, n_meta_users: {nmu} => n_original_users: {no}'.format(nu=n_users, nmu=n_meta_users, no=n_users-n_meta_users))
    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} | Cycle: ({fo}, {f})'.format(dim=R.shape, f=fold, fo=outer_fold))
    piggyback_msg = "+  Cycle: ({fo}, {f}) | setting: {setting}".format(fo=outer_fold, f=fold, setting=algorithm_setting).rjust(5)   # keep track of the thread from which the iteration 
    
    ########################################################################################
    Pr, Qr, *Rh_errs = ua.implicit_als(Cr, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            # label_confidence=Cr_bar, 
                            polarity=Mcr,   # derived from correctness matrix        
                            ratings=R, labels=L_train,

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

    # compute reconstructed training data (so that later on we can test its utility for stacking)
    # Pr, Qr => Rh, use Rh in place of R whenever Cr == fill
    Rh = reconstruct(Cr, R, Pr, Qr, policy_opt=params['policy_opt'], policy_replace=params['policy_replace'], 
            replace_subset=params['replace_subset'], params=params, null_marker=null_marker, binarize=False, name='R')
    # ... Rh is not yet binarized 
    if tPreferenceCalibration: 
        Pf, Lh_R = uc.probability_filter(R, L_train, p_threshold)  # CM entries: 1, if correct predictions (TP, TN); 0 o.w. 
        Rh, pref_threshold, score = uc.calibrate_preference(Rh, Pf=Pf, Lh=Lh_R, step=0.01, 
            message='training split (R)', policy=params['policy_calibration'])  # Lh is only used in 'precision'
        # ... Rh is binarized, score depends on the policy
        
        print('(wmf_ensemble_iter) Quality of the seed on R ... Cycle: {}'.format( (outer_fold, fold)) )
        ret = uc.ratio_of_alignment2(Rh, Pf, Lh_R, verbose=True)  
        
    # ... Rh is binarized 
    
    ### create extra prediction vectors (PVs) from T (say, mean vector) and attach these new PVs to T (piggyback)

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    # compute confidence matrix for T
    CT = uc.evalConfidenceMatrix(X=T, L=[], U=U, 
            message=(R, L_train, Cr),  # set M/message to R, L so that we can use proba thresholds from R to estimte labels in T; optionally, use Cr to identify useless classifiers (producing no correct positive predictions)
            ratio_users=params['ratio_users'], 

            # parameters to be used for unsupervised mode
            ratio_small_class=class_stats['r_minority'], 
            factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy_test'], # <<< determine the dimension of filtering (user, item)
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'],  # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], alpha=params['alpha'], 
                    estimated_labels=True,

                        # masked=params['masked'], mask_all_test=params['mask_all_test'],
                        fill=null_marker, fold=fold, L_test=L_test) # project_path=System.projectPath 
    Ct, Mct, _, *CT_rest = CT     # note: Ct_bar removed
    ############################################################

    # assert Ct.shape == T.shape    # ... ok
    # verify_confidence_matrix(Ct, Mct)
    assert n_users_test == Ct.shape[0] and n_items_test == Ct.shape[1]
    div("(wmf_ensemble_iter) Completed C(T) | Cycle ({fo}, {fi}) | dim(Ct): {dim}, filter_axis: {fdim} | optimization: {opt} | replace bad probabilities only? {tval}".format(
        fo=outer_fold, fi=fold, nf=params['n_factors'], a=params['alpha'], dim=Ct.shape, fdim=params['policy_test'], opt=params['policy_opt'], tval=params['replace_subset']), symbol='#', border=1)
    print('... Model | n_factors: {nf}, alpha: {a}'.format(nf=params['n_factors'], a=params['alpha']))
     
    # use Qr in T 
    div('... (ALS 2) Going into the ALS loop on TEST data (T): {dim}  | Cycle ({fo}, {fi}) | policy_opt_T: {policy}'.format(dim=T.shape, 
        fo=outer_fold, fi=fold, policy=params['policy_opt_T'])) 
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

    assert do_transfer or do_als, "do_transfer and do_als cannot be both inactive"
    if do_transfer: 
        X = (R, T)
        F = (Pr, Qr)
        Pt, Qt = uc.transfer_factor_by_similarity(X, F, topk=1)  # ... tr(1)
        Th_err = Th_err_weighted = [0, ]  # dummy
        # user_vectors, item_vectors = Pt, Qt
        
    if do_als: 
        if do_transfer: 
            user_vectors, item_vectors = Pt, Qt  # factors transfered from R   ... tr(2)
            resume_als = True  # must be True because we'll only use them for initialization
        else: 
            user_vectors = Pr # learned classifier vectors
            item_vectors = None
            # resume_als is optional

        Pt, Qt, *Th_errs = ua.implicit_als(Ct, features=params['n_factors'], 
                                # label_confidence=Ct_bar, 

                                iterations=params['n_iter_foldin'], 
                                lambda_val=System.lambda_val,  # 0.8 by default
                                
                                polarity=Mct, 
                                ratings=T, labels=[],

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

    # set canonicalize=True because the reconstructed probabilities could be not in [0.0, 1.0]
    Th = reconstruct(Ct, T, Pt, Qt, policy_opt=params['policy_opt'], policy_replace=params['policy_replace'], 
            replace_subset=params['replace_subset'], params=params, null_marker=null_marker, 
                binarize=False,  # tPreferenceCalibration,  # binarize Th via pmax estimated from R
                    pref_threshold=pref_threshold, 
                        is_test_set=True,  
                        name='T')
    # binarize Th ... (a)
    # if we use the same 'pref_threshold' derived from R, Th tends not to have a balanced 1s and 0s

    # assert len(np.unique(Th)) == 2, "Th was not binarized | tPreferenceCalibration? {}, binarize_pref? {} | unique values (n={}): {}".format(tPreferenceCalibration, 
    #     params['binarize_pref'], len(np.unique(Th)), np.unique(Th)[:10])

    # [test]
    pref_threshold_test = pref_threshold
    if tPreferenceCalibration: 
        # binarize Th ... (b)
        # p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=params['policy_threshold']) 
        div(message='(wmf_ensemble_iter) Use estimated label (by majority vote) to binarize T', symbol='#')
        Thb, pref_threshold_test, rc_test = uc.estimate_pref_threshold(Th, T, L=[], p_threshold=p_threshold, message='Tb using estimated labels')
        # ... now got the preference threshold

        # use L_test to get correctness matrix
        Pf, Lh_T = uc.probability_filter(T, L_test, p_threshold)  # Lh(T, p_threshold)

        # degree of alignment between preference matrix Th (via esimated labels Lh) and corrrectness CMt(computed from true labels)
        print("... (1) Quality of the seed (via estimated labels): cycle {} | th(R): {} ~? th(T): {}".format( (outer_fold, fold), pref_threshold, pref_threshold_test))
        ret = uc.ratio_of_alignment2(Thb, Pf, Lh_T, verbose=True) # set binarize to False if Thb is already binarized

        ########################################
        # ... how does it compare to using true labels?
        div('(wmf_ensemble_iter) 2. How does it fare with using true labels?') 
     
        Thb2, pth_test, rc_test = uc.estimate_pref_threshold(Th, T, L=L_test, p_threshold=p_threshold, message='Tb using true test labels')

        print('... (2) Quality of the seed (via true labels): cycle {} | th(R): {} ~? th(T): {}'.format( (outer_fold, fold), pref_threshold, pth_test))
        ret = uc.ratio_of_alignment2(Thb2, Pf, Lh_T, verbose=True) # set binarize to False if Thb is already binarized

        div('(wmf_ensemble_iter) 3. How about just using the threshold from (R)?')
        pth_test = pref_threshold
        Thb3 = uc.binarize_pref(Th, p_th=pref_threshold)

        print("... (3) Quality of the seed (via R): cycle {} | th(R): {} == th(T): {}".format( (outer_fold, fold), pref_threshold, pth_test))
        ret = uc.ratio_of_alignment2(Thb3, Pf, Lh_T, verbose=True) # set binarize to False if Thb is already binarized

        # use (1) as our preference matrix
        Th = Thb
        assert len(np.unique(Th)) in (1, 2), "Th was not binarized | tPreferenceCalibration? {}, binarize_pref? {} | unique values (n={}): {}".format(tPreferenceCalibration, 
            params['binarize_pref'], len(np.unique(Th)), np.unique(Th)[:10])
    # ... if Th represents preference matrix, then Th is binarized 
    
    div("(wmf_ensemble_iter) Completed rating matrix reconstruction | Cycle: ({fo}, {fi}) | preference scores? {tval}, action='{act}'".format(
        fo=outer_fold, fi=fold, tval=isPreferenceScore, act='Replace Subset' if params['replace_subset'] else 'Replace All')) # predict => predict probabilities
    if isPreferenceScore: 
        print('... binarize preference matrix? {}'.format(params['binarize_pref']))

    ### ALS evaluation (RMS)
    Th_err = ua.prediction_error(Ct, Th, Pt, Qt, fill=0)  # only meaningful when approximating ratings
    delta_R, delta_T = LA.norm(Rh-R, 'fro'), LA.norm(Th-T, 'fro')

    # classifier/user weights (usually just use confidence matrix)
    Cw = Cwt = None
    if tWeightedPrediction: 
        # assert tPreferenceCalibration, "Rh, Th must be calibrated binary matrices representing preferences"
        Cw, Cwt = Cr, Ct
        print('(wmf_ensemble_iter) using weighetd output via confidence matrix | dim(Cwt): {}'.format(Cwt.shape))
    else: 
        print('(wmf_ensemble_iter) using only preference matrix (non-weigthed) for making final predictions ...')

    # X: (R, T, L_train, L_test, U)
    n_samples_reconstructed = Rh.shape[1]+Th.shape[1]
    ##############################################################################################################
    # ... prediction 
    if not aggregation_methods: aggregation_methods = System.aggregation_methods # e.g. ['mean', 'median', 'log', ]  
    # pv_mean = uc.combiner(Th, aggregate_func='mean')           

    ##############################################################################################################
    # ... output
    file_types = ['prior', 'posterior', ]
    if post_hoc_analysis:   # only triggered after model selection cycle is complete and 'best params' has been determined
        
        # note that we shall save the data only after model selection loop is completed
        assert outer_fold >= 0 and fold == -1, "Intended action: only save the final model after model selection is complete (outer fold: {fo}, inner fold: {fi}".format(fo=outer_fold, fi=fold)
        the_params = name_params_setting(method_params=['F', 'A'])

        div('(wmf_ensemble_iter) Running posthoc analysis | algorithmic setting: {s}'.format(s=algorithm_setting), symbol='%') 
        # MFEnsemble.save_meta_tset((Rh, Th, L_train, L_test, U), fold=fold, method=method, params=params, verbose=verbose)

        pv_t = pv_th = None
        if tMetaUsers: 
            # drop meta users 
            R = R[:-n_meta_users]
            Rh = Rh[:-n_meta_users]
            T, pv_t = T[:-n_meta_users], T[-n_meta_users:] 
            Th, pv_th = Th[:-n_meta_users], Th[-n_meta_users:]
            assert Th.shape[0] + pv_th.shape[0] == n_users_test
            assert pv_t.shape[1] == pv_th.shape[1] == len(L_test)

            Pr = Pr[:-n_meta_users]
            Pt = Pt[:-n_meta_users]

        ### save predictions
        isBinaryPrefMatrix = False
        if isPreferenceScore: 
            n_uniq, n_uniq_test = len(np.unique(Rh)), len(np.unique(Th))
            if tPreferenceCalibration: # then Xh must be a binary matrix
                assert n_uniq == 2 and n_uniq_test == 2, "Th is not binary or is degenerated | n_uniq: {}, n_uniq(T): {} | values(T): {}".format(n_uniq, n_uniq_test, np.unique(Th))
            else: 
                # Th is a continuous rating matrix
                assert n_uniq > 2 and n_uniq_test > 2, "Unique values should be >> 1 | n_uniq(R): {}, u_uniq(T): {}".format(n_uniq, n_uniq_test)
            if n_uniq_test == 2: isBinaryPrefMatrix = True
            print('(wmf_ensemble_iter) Prior to making predictions | Th is a {}  ... (verify)'.format(
                'binary matrix' if isBinaryPrefMatrix else 'rating matrix'))

        # the_params = name_params_setting(method_params=['F', 'A'])
        dataset = {'prior': [R, T], 'posterior': [Rh, Th]}
        for file_type in file_types: 
            X_train, X_test = dataset[file_type]
            for i, aggr in enumerate(aggregation_methods):    # defined earlier e.g. ['mean', 'median']
                pn = '{ftype}_{model}'.format(ftype=file_type, model=aggr)  # pn: predictor name
                
                # TODO: if aggr in ['logistic', ...]  # need to further train on X_train and test on X_test
                if aggr in ['mean', 'median', ]:   # System.simple_aggregation
                    if isPreferenceScore: 
                        if file_type.startswith('pri'):
                            pv = uc.combiner(T, aggregate_func=aggr)
                        else: 
                            # note: Th should have been in the right format, NO need to set canonicalize to True
                            pv = uc.predict_by_preference(T, Th, W=Cwt, name='Th', aggregate_func=aggr, fallback_on_low_weight=False) # verify/False
                    else: 
                        pv = uc.combiner(X_test, weights=Cwt, aggregate_func=aggr)  # T vs Th
                else:
                    ### put stacker code here! 
                    stacker = stacking.choose_classifier(aggr)  # e.g. log, enet, knn

                    if isPreferenceScore: 
                        # use Rh to predict Th? 
                        # 1. consider entries of low or zero preference scores as "dropouts" 
                        # binarize the preference scores and use them as features?

                        if file_type.startswith('post'):
                            # regular mode: R and T have different preference thresholds
                            # Rb = uc.binarize_pref(Rh, p_th=pref_threshold, cutoff=True) # uc.canonicalize_pref(Rh, name='Rb', binarize=True, verify=2)
                            # Tb = uc.binarize_pref(Th, p_th=pref_threshold_test, cutoff=True) # uc.canonicalize_pref(Th, name='Tb', binarize=True, verify=2)
                            X_train = np.vstack((R, Rh))   # R * Rb doesn't work well
                            X_test = np.vstack((T, Th))    # T * Tb doesn't work well
                        
                        model = stacker.fit(X_train.T, L_train) # remember to take transpose
                        pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data    
                    else: 
                        model = stacker.fit(X_train.T, L_train) # remember to take transpose
                        pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data
                    
                y_pred, y_label = pv, L_test
                vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

        ### save predictions from the meta users (and treat these data as if they came from an aggregation method)
        if tMetaUsers: 
            # vmap['prior_mean'], vmap['posterior_mean'] = {}, {}
            predictions = {'prior': pv_t, 'posterior': pv_th}  # each prediction dataframe consists of prediction vectors from n meta_users
            for file_type in file_types: 
                for i, meta_user in enumerate(meta_users): 
                    pn = '{ftype}_{model}'.format(ftype=file_type, model=meta_user) 

                    if isPreferenceScore:
                        if file_type.startswith('pri'):
                            pv = pv_t
                        else:      
                            pv = pv_t * pv_th  # use pref scores to 'up-regulate' or 'down-regulate' the predictions
                    else: 
                        pv = predictions[file_type]

                    y_pred, y_label = pv[i], L_test
                    vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

            # for file_type, pv in predictions.items(): 
            #     pn = '{ftype}_{model}'.format(ftype=file_type, model='mean') # pn: predictor name
            #     vmap[pn] = {}
            #     for i, meta_user in enumerate(meta_users): # ['latent_mean', 'latent_mean_masked',]
            #         y_pred, y_label = pv[i], L_test
            #         vmap[pn][meta_user] = DataFrame({'prediction':y_pred,'label':y_label, 'method':meta_user, 'fold': fold}, index=range(len(y_pred)))
    
            print('... saved predictions for data type: {dtype} & meta users (n={n}) to dataframes'.format(dtype=file_type, n=len(meta_users)))
         
        #### optionally, run cluster analysis
        filter_axes = ['user', ]  # 'item'
        if enable_cluster_analysis: 
            # if meta users were included, consider R[:-n_meta_users] 
            for fdim in filter_axes: 
                
                # training split; if meta users were included, consider R[:-n_meta_users] 
                run_cluster_analysis(Pr, U=U, X=R, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=False, save_plot=True, file_type='train') # usually only at the training of the final model (i.e. after the best hyperparameters are determined; see wmf_ensemble_model_select())

                # test split
                run_cluster_analysis(Pt, U=U, X=T, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=False, save_plot=True, file_type='test') # usually only at the training of the final model (i.e. after the best hyperparameters are determined; see wmf_ensemble_model_select())

        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)

        # save training data?
        # test set
        # MFEnsemble.save_data((T, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data 
        # MFEnsemble.save_data((Th, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data  (~ prediciton)

        # training set
        # MFEnsemble.save_data((R, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-prior', verbose=verbose, subsampling=True)
        # MFEnsemble.save_data((Rh, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-posterior', verbose=verbose, subsampling=True)

        # return the dataset ID
        #    this is necessary because the 'best params' in each cycle may not be the same; for example, we may end up getting training data like this: 
        #       wmf_F100_A100_XCFuser_S2-train-prior-1.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-0.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-2.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-3.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-4.csv.gz
        #    => in cycle {0, 2, 3, 4}, F75_A100 was the best, but in cycle {1, }, F100_A100 was the best
        vmap['dset_id'] = dset_id 
        vmap['best_params_inner'] = the_params

    ##############################################################################################################
    if save_data: 
        div('(wmf_ensemble_iter) Output: saving transformed training and test sets (size: n(R): {nR}, n(T): {nT}), total size: {N}| delta(R): {dR}, delta(T): {dT} | algorithmic setting: {s}'.format(s=algorithm_setting, 
                    nR=Rh.shape[1], nT=Th.shape[1], N=n_samples_reconstructed, dR=delta_R, dT=delta_T), symbol='>')

        # test set
        MFEnsemble.save_data((T, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data 
        MFEnsemble.save_data((Th, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data  (~ prediciton)

        # training set
        MFEnsemble.save_data((R, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-prior', verbose=verbose, subsampling=True)
        MFEnsemble.save_data((Rh, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-posterior', verbose=verbose, subsampling=True)
    

    ### kind: specific MF algorithm
    if not kind: kind = 'als'
    wmfMetrics, wmfCV = [], [] # vmap['wmfMetrics'], vmap['wmfCV']

    # naming convention: method_id = '{prefix}_{id}_{learner_type}' # e.g. rf_bp_stacker
    aggregate_mode = params.get('policy_ms_model', 'mean')
    aggregate_func = params.get('policy_aggregate_func', 'mean')
    div("(wmf_ensemble_iter) Comparison of model parameters | aggregate_func: {func}, mode: {mode}".format(func=aggregate_func, mode=aggregate_mode)) 
    # {stacker}.S-{dataset}-{suffix}
    # method_id = '{prefix}_{id}_{learner_type}'.format(prefix='wmf_%s' % kind, id=MFEnsemble.get_method_id(method, kind, params=params), learner_type=aggregate_func)
    method_id = "{prefix}.W-{id}-{suffix}".format(prefix=aggregate_func, id=MFEnsemble.get_method_id(method, kind, params=params), suffix=kind)
    ### model evaluation
    #   a. use mean proba in T vs labels as a summary score
    #   b. but we can compute a performance score for each base method (and then take the average)

    # header = MFEnsemble.header_model_evaluation # ['method', 'score', 'posterior_score', 'mean_score', ] ... used in evaluate.analyzePerf2()

    ############################################################################################################################
    # PerformanceMetrics object in wmfMetrics is the basis for selecting the best hyperparameter
    # 
    performance_metrics, pv = analyzePerf(L_test, Th, 
                                    T=T,  # pass T=T to compare with T
                                    method=method_id, aggregate_func=aggregate_func,
                                        weights=Cwt, 
                                        outer_fold=outer_fold, fold=fold,  # keep track of the iteration (debugging only when comparing Th and T)
                                        train_data=(Rh, L_train),  # only used in stacking mode
                                        mode=aggregate_mode)  # pass T=T to compare with T
    wmfMetrics.append( performance_metrics ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
    wmfCV.append( (L_test, pv) )   
    vmap['wmfMetrics'], vmap['wmfCV'] = wmfMetrics, wmfCV

    ##########################################################
    # other variables that may be of interest 
    # >>> piggyback inputs
    if piggyback: 
        # args, posargs = us.arguments()
        vmap['hyperparams'] = hyperparams  # keep track of hyperparameters for convenience 
    # vmap['fold'] = fold
    ##########################################################

    print("(wmf_ensemble_iter) Ending cycle: ({fo}, {fi}) at setting {case} > returning vmaps: {keys} ... (verify) ".format(fo=outer_fold, fi=fold, 
        case=algorithm_setting, keys=vmap.keys()))

    # keys of vmap are the variables to return to caller: 
    #   i) saved in every cycle: wmfMetrics, wmfCV, hyperparams
    #  ii) saved only when training the final model after n cycles of model selection is completed: 
    #      dset_id, 
    #      best_params_nruns 
    return vmap  

def wmf_ensemble_preferred_ratings(data, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', fold=-1, outer_fold=-1, null_marker=0, verbose=1, 
        project_path='?', piggyback=True, dev_ratio=0.2, max_dev=5000, resample=False, aggregation_methods=[], 
        post_hoc_analysis=False, save_data=False, enable_cluster_analysis=False):
    """
    
    Params
    ------
    fold: defined in a more generic sense than the 'fold' in wmf_ensemble_fold(). 
          specifically, 'fold' can refer to any of the following: 

           i) a CV fold ii) the index of iterations in random subsampling iii) other iteration index
    outer_fold: the iteration/fold number of the outer loop when wmf_ensemble_iter() is invoked for model selection (e.g. by model_select_core())

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
        div('(wmf_ensemble_iter) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2, 7, 8): 
            assert params['supervised']
        if params['setting'] in (3, 4, 9, 10): 
            assert not params['supervised']
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
    def name_params_setting(method_params=['F', 'A']):   # [todo]
        # MFEnsemble.abridge(meta, identifiers=method_params, replace=replacement, verify=True)

        # use MFEnsemble.params_to_ids
        return 'F{nf}A{a}'.format(nf=params['n_factors'], a=params['alpha'])

    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    import stacking

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
    # verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    ### Confidence Matrix
    # fold: iteration number (here, 'fold' is simply a repurposed variable borrowed from CV)
    # data: (R, T, L, Lt, U)-format
    if isinstance(data, DataFrame): # in this case, 'data' is a dataframe comprising a train-dev split 
        R, T, L_train, L_test, U =  uc.shuffle_split(data, ratio=dev_ratio, max_size=max_dev)
    else:
        # otherwise, assuming that shuffle-split operation had been invoked to produce the data input
        assert isinstance(data, (tuple, list)), "Invalid input data: %s" % data 
        print('(wmf_ensemble_iter) Input data is an n-tuple, whrere n={n}'.format(n=len(data)))
        R, T, L_train, L_test, U = data 

    ### resampline (balancing classes)
    # todo

    n_samples = R.shape[1]+T.shape[1]; assert len(L_train)+len(L_test) == n_samples

    ### create extra prediction vectors (PVs) from R (say, mean vector) and attach these new PVs to R (piggyback)
    n_users, n_items = R.shape
    n_classifiers = int(n_users/BagCount)
    n_users0, n_items0 = n_users, n_items  # keep a copy of the original number of users and items
    n_users_test, n_items_test = T.shape
    n_meta_users = 0

    # preference parameters
    isPreferenceScore = True; assert params['policy_opt'].startswith('pref') # and not params['replace_subset'] 
    tPreferenceCalibration = params['binarize_pref'] # params.get('preference_calibration', True)
    pref_threshold = 0.5
    tWeightedPrediction = params.get('weighted_output', False)

    ############################################################
    # piggy back extra meta-estimators
    meta_users = ['latent_mean', 'latent_mean_masked',]  # todo
    if tMetaUsers: 
        div(message='(wmf_ensemble_preferred_ratings) Adding meta users (n={n})...'.format(n=len(meta_users)))
        
        # e.g. mean, and masked mean
        nU, nUT = n_users, n_users_test

        print('... augmenting T (by meta users)')
        mean_pv = make_prediction_vector(T, L=[], policy='none')   # policy: 'none' => no masking
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

    # rank transformation 

    ########################################################################################
    # compute confidence matrix for R
    CR = uc.evalConfidenceMatrix(R, L=L_train, U=U,  
            ratio_users=params['ratio_users'], 
            ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< determines the subroutine for computing Cui
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], 
                        alpha=params['alpha'], beta=params.get('beta', 1.0),
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, fold=fold) # project_path=System.projectPath 
    Cr, Mcr, p_threshold, *CR_res = CR  
    ########################################################################################

    # ... Cui_bar is only used in policy = 'tradeoff'
    # verify_confidence_matrix(Cr, Cr_bar)
    
    div("(wmf_ensemble_preferred_ratings) Completed C(R) | Cycle {cycle} |  dim(Cui): {dim}, filter_axis: {fdim} | conf_measure: {measure}, optimization: {opt} | predict ALL probabilities? {tval} | policy_threshold: {p_th}".format(
        cycle=(outer_fold, fold), 
            dim=str(Cr.shape), fdim=params['policy'], measure=params['conf_measure'], opt=params['policy_opt'], tval=params['predict_probs'], p_th=params['policy_threshold']), symbol='#', border=1)
    print('... Model | n_factors: {nf}, alpha: {a} | predict pref? {pref}, binarize? {bin}, supervised? {s}'.format(nf=params['n_factors'], a=params['alpha'], pref=isPreferenceScore, bin=params['binarize_pref'], s=params['supervised']))
    print('... Data | n_train: {ntr}, n_test: {nt}, ratio: {r}'.format(ntr=len(L_train), nt=len(L_test), r=(len(L_train)+len(L_test))/(len(L_test)+0.0)) )
    print('... Classifiers | n_users: {nu}, n_meta_users: {nmu} => n_original_users: {no}'.format(nu=n_users, nmu=n_meta_users, no=n_users-n_meta_users))
    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} | Cycle: ({fo}, {f})'.format(dim=R.shape, f=fold, fo=outer_fold))
    piggyback_msg = "+  Cycle: ({fo}, {f}) | setting: {setting}".format(fo=outer_fold, f=fold, setting=algorithm_setting).rjust(5)   # keep track of the thread from which the iteration 
    
    ########################################################################################
    print('... approximating ratings (R) | Cycle ({fo}, {fi})'.format(fo=outer_fold, fi=fold))
    Pr, Qr, *Rh_errs = ua.implicit_als(Cr, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            # label_confidence=Cr_bar, 
                            polarity=Mcr, 
                            ratings=R, labels=L_train,

                            policy='rating', message=piggyback_msg, ret_rmse=True)
    Rh_err, Rh_err_weighted = Rh_errs

    print('... estimating preferences (R) | Cycle ({fo}, {fi})'.format(fo=outer_fold, fi=fold))
    Ppr, Qpr, *Rpr_errs = ua.implicit_als(Cr, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            # label_confidence=Cr_bar, 
                            polarity=Mcr, 
                            ratings=R, labels=L_train,

                            policy='preference', message=piggyback_msg, ret_rmse=True)

    ########################################################################################
    ne = 2
    e_pri, e_post = np.mean(Rh_err[:ne]), np.mean(Rh_err[-ne:])
    e_del = (e_pri-e_post)/e_pri * 100
    print('... (ALS 1) Complete | rmse: {e1} -> {e2} (n_errs: {n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(e1=e_pri, e2=e_post, 
                   e_del=e_del, ew1=np.mean(Rh_err_weighted[:ne]), ew2=np.mean(Rh_err_weighted[-ne:]), n=len(Rh_err) ))

    # e.g. 50 users, 3979 items with nf=100 > dim(P): (50, 100), dim(Q): (3979, 100)
    assert Pr.shape[0] == R.shape[0] and Pr.shape[1] == params['n_factors']
    assert Qr.shape[0] == R.shape[1] and Qr.shape[1] == params['n_factors']

    # compute reconstructed training data (so that later on we can test its utility for stacking)
    # Pr, Qr => Rh, use Rh in place of R whenever Cr == fill
    # CM = uc.probability_filter(R, L_train, p_threshold)
    Rh, Rp, pref_threshold = \
        reconstruct_by_preference(Cr, R, factors=(Pr, Qr), prefs=(Ppr, Qpr), labels=L_train, 
                binarize=tPreferenceCalibration, 
                    p_threshold=p_threshold, 
                    pref_threshold=-1, # we don't know the preference threshold yet
                        policy_calibration=params['policy_calibration'],
                        replace_subset=True, replace_all=False, params=params, null_marker=0, name='R', verify=True, index=(outer_fold, fold))
    # ... Rh: probability matrix, Rp: preference matrix (binary)

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    # compute confidence matrix for T
    ########################################################################################
    CT = uc.evalConfidenceMatrix(X=T, L=[], U=U, 
            message=(R, L_train, Cr),  # set M/message to R, L so that we can use proba thresholds from R to estimte labels in T; optionally, use Cr to identify useless classifiers (producing no correct positive predictions)
            ratio_users=params['ratio_users'], 

            # parameters to be used for unsupervised mode
            ratio_small_class=class_stats['r_minority'], 
            factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy_test'], # <<< determine the dimension of filtering (user, item)
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'],  # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], 
                        alpha=params['alpha'], beta=params.get('beta', 1.0), 

                        # masked=params['masked'], mask_all_test=params['mask_all_test'],
                        fill=null_marker, fold=fold, L_test=L_test) # project_path=System.projectPath 
    Ct, Mct, _, *CT_rest = CT 
    ########################################################################################

    # assert Ct.shape == T.shape    # ... ok
    # verify_confidence_matrix(Ct, Ct_bar)
    assert n_users_test == Ct.shape[0] and n_items_test == Ct.shape[1]
    div("(wmf_ensemble_preferred_ratings) Completed C(T) | Cycle ({fo}, {fi}) | dim(Ct): {dim}, filter_axis: {fdim} | optimization: {opt} | replace bad probabilities only? {tval}".format(
        fo=outer_fold, fi=fold, nf=params['n_factors'], a=params['alpha'], dim=Ct.shape, fdim=params['policy_test'], opt=params['policy_opt'], tval=params['replace_subset']), symbol='#', border=1)
    print('... Model | n_factors: {nf}, alpha: {a}'.format(nf=params['n_factors'], a=params['alpha']))
     
    # use Qr in T 
    div('... (ALS 2) Going into the ALS loop on TEST data (T): {dim}  | Cycle ({fo}, {fi}) | policy_opt_T: {policy}'.format(dim=T.shape, 
        fo=outer_fold, fi=fold, policy=params['policy_opt_T'])) 
    ########################################################################################
    policy_opt_eff = 'rating' if params['policy_opt'].startswith('trade') else params['policy_opt']
    
    resume_als = params.get('resume_als', False)
    do_als, do_transfer = True, False

    Pt = Qt = None
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

    assert do_transfer or do_als, "do_transfer and do_als cannot be both inactive"
    if do_transfer: 
        X = (R, T)
        F = (Pr, Qr)
        Pt, Qt = uc.transfer_factor_by_similarity(X, F, topk=1)  # ... tr(1)
        Th_err = Th_err_weighted = [0, ]  # dummy
        # user_vectors, item_vectors = Pt, Qt
        
    if do_als: 
        if do_transfer: 
            user_vectors, item_vectors = Pt, Qt  # factors transfered from R   ... tr(2)
            resume_als = True  # must be True because we'll only use them for initialization
        else: 
            user_vectors = Pr # learned classifier vectors
            item_vectors = None
            # resume_als is optional

        print('... approximating ratings (T) ... | Cycle ({fo}, {fi})'.format(fo=outer_fold, fi=fold))
        Pt, Qt, *Th_errs = ua.implicit_als(Ct, features=params['n_factors'], 
                                iterations=params['n_iter_foldin'], 
                                lambda_val=System.lambda_val,  # 0.8 by default
                                
                                # label_confidence=Ct_bar, 
                                
                                polarity=Mct, 
                                ratings=T, labels=[],

                                    user_vectors=user_vectors,   # <<< in foldin mode, fix the user factors learned from R 
                                    item_vectors=item_vectors,   # <<< only used in transfer+seed mode
                                
                                    policy='rating',  # <<< in tradeoff mode, reduce this to 'rating' mode
                                    message=piggyback_msg, 
                                    ret_rmse=True, 
                                        resume_als=resume_als)
        Th_err, Th_err_weighted = Th_errs

        print('... estimating preferences (T) | Cycle ({fo}, {fi})'.format(fo=outer_fold, fi=fold))
        Ppt, Qpt, *Tpt_errs = ua.implicit_als(Ct, features=params['n_factors'], 
                                iterations=params['n_iter_foldin'], 
                                lambda_val=System.lambda_val,  # 0.8 by default
                                
                                # label_confidence=Ct_bar, 
                                polarity=Mtc, 
                                ratings=T, labels=[],

                                    user_vectors=user_vectors,   # <<< in foldin mode, fix the user factors learned from R 
                                    item_vectors=item_vectors,   # <<< only used in transfer+seed mode
                                
                                    policy='preference',  # <<< in tradeoff mode, reduce this to 'rating' mode
                                    message=piggyback_msg, 
                                    ret_rmse=True, 
                                        resume_als=resume_als)
    
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

    # set canonicalize=True because the reconstructed probabilities could be not in [0.0, 1.0]
    assert Pt is not None and Ppt is not None
    Th, Tp, pref_threshold_test = \
            reconstruct_by_preference(Ct, T, factors=(Pt, Qt), prefs=(Ppt, Qpt),
                labels=[],    # labels must be estimated
                test_labels=L_test, # note: only used for testing!
                p_threshold=p_threshold, # note: only used for testing!
                
                binarize=tPreferenceCalibration, 
                pref_threshold=-1, # pref_threshold, # computed from the training split (R), pass -1 to re-calibrate
                    policy_calibration=params['policy_calibration'],
                        is_test_set=True, 

                        replace_subset=True, replace_all=False, params=params, null_marker=0, name='T', verify=True, index=(outer_fold, fold))
    assert np.sum(Th) > 0, "Preference scores summed to zero! All degenerated?"
    # ... Th: probabilty matrix, Tp: preference matrix (binarized)

    # ... CF-transform T to get Th (using the classifier/user vectors learned from the training set (R))
    div("(wmf_ensemble_preferred_ratings) Completed rating matrix reconstruction | Cycle: ({fo}, {fi}) | preference scores? {tval}, action='{act}'".format(
        fo=outer_fold, fi=fold, tval=isPreferenceScore, act='Replace Subset')) # predict => predict probabilities
    if isPreferenceScore: 
        print('... binarize preference matrix? {}'.format(True)) # params['binarize_pref']

    ### ALS evaluation (RMS)
    Th_err = ua.prediction_error(Ct, Th, Pt, Qt, fill=0)  # only meaningful when approximating ratings
    delta_R, delta_T = LA.norm(Rh-R, 'fro'), LA.norm(Th-T, 'fro')

    # classifier/user weights (usually just use confidence matrix)
    Cw = Cwt = None
    if tWeightedPrediction: 
        Cwt = Ct
        print('(wmf_ensemble_preferred_ratings) using weighetd output via confidence matrix | dim(Cwt): {}'.format(Wt.shape))
    else: 
        print('(wmf_ensemble_preferred_ratings) using only preference matrix (non-weigthed) for making final predictions ...')

    # X: (R, T, L_train, L_test, U)
    n_samples_reconstructed = Rh.shape[1]+Th.shape[1]
    ##############################################################################################################
    # ... prediction 
    if not aggregation_methods: aggregation_methods = System.aggregation_methods # e.g. ['mean', 'median', 'log', ]  
    # pv_mean = uc.combiner(Th, aggregate_func='mean')           

    ##############################################################################################################
    # ... output
    file_types = ['prior', 'posterior', ]
    if post_hoc_analysis:   # only triggered after model selection cycle is complete and 'best params' has been determined
        
        # note that we shall save the data only after model selection loop is completed
        assert outer_fold >= 0 and fold == -1, "Intended action: only save the final model after model selection is complete (outer fold: {fo}, inner fold: {fi}".format(fo=outer_fold, fi=fold)
        the_params = name_params_setting(method_params=['F', 'A'])

        div('(wmf_ensemble_preferred_ratings) Running posthoc analysis | algorithmic setting: {s}'.format(s=algorithm_setting), symbol='%') 
        # MFEnsemble.save_meta_tset((Rh, Th, L_train, L_test, U), fold=fold, method=method, params=params, verbose=verbose)

        pv_t = pv_th = None
        if tMetaUsers: 
            # drop meta users 
            R = R[:-n_meta_users]
            Rh = Rh[:-n_meta_users]
            Rp = Rp[:-n_meta_users]
            T, pv_t = T[:-n_meta_users], T[-n_meta_users:] 
            Th, pv_th = Th[:-n_meta_users], Th[-n_meta_users:]
            Tp, pv_pref = Tp[:-n_meta_users], Tp[-n_meta_users:]
            assert Th.shape[0] + pv_th.shape[0] == n_users_test
            assert pv_t.shape[1] == pv_th.shape[1] == len(L_test)

            Pr = Pr[:-n_meta_users]
            Pt = Pt[:-n_meta_users]

        ### save predictions
        # Tp is a binary matrix
        n_uniq_pref = len(np.unique(Tp))
        assert n_uniq_pref == 2, "(wmf_ensemble_preferred_ratings) Tp is not binary or is degenerated | n_uniq(Tp): {} | values(Tp): {}".format(n_uniq_pref, np.unique(Tp))
        # Th is a continuous rating matrix
        n_uniq = len(np.unique(Th))
        assert n_uniq > 2, "(wmf_ensemble_preferred_ratings) Unique ratings should be >> 1 | u_uniq(T): {}".format(n_uniq)
        # ... although in this routine, Th is expected to be already a probability matrix (conditioned on the preference matrix).

        # the_params = name_params_setting(method_params=['F', 'A'])
        dataset = {'prior': [R, T], 'posterior': [Rh, Th]}
        for file_type in file_types: 
            X_train, X_test = dataset[file_type]
            # ... both are rating/probabilty matrices

            for i, aggr in enumerate(aggregation_methods):    # defined earlier e.g. ['mean', 'median']
                pn = '{ftype}_{model}'.format(ftype=file_type, model=aggr)  # pn: predictor name
                
                # TODO: if aggr in ['logistic', ...]  # need to further train on X_train and test on X_test
                if aggr in ['mean', 'median', ]:   # System.simple_aggregation
                    pv = uc.combiner(X_test, weights=Cwt, aggregate_func=aggr)  # T vs Th
                else:
                    ### put stacker code here! 
                    stacker = stacking.choose_classifier(aggr)  # e.g. log, enet, knn
                    model = stacker.fit(X_train.T, L_train) # remember to take transpose
                    pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data
                    
                y_pred, y_label = pv, L_test
                vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

        ### save predictions from the meta users (and treat these data as if they came from an aggregation method)
        if tMetaUsers: 
            # vmap['prior_mean'], vmap['posterior_mean'] = {}, {}
            predictions = {'prior': pv_t, 'posterior': pv_th}  # each prediction dataframe consists of prediction vectors from n meta_users
            for file_type in file_types: 
                for i, meta_user in enumerate(meta_users): 
                    pn = '{ftype}_{model}'.format(ftype=file_type, model=meta_user) 
                    pv = predictions[file_type]

                    y_pred, y_label = pv[i], L_test
                    vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

            # for file_type, pv in predictions.items(): 
            #     pn = '{ftype}_{model}'.format(ftype=file_type, model='mean') # pn: predictor name
            #     vmap[pn] = {}
            #     for i, meta_user in enumerate(meta_users): # ['latent_mean', 'latent_mean_masked',]
            #         y_pred, y_label = pv[i], L_test
            #         vmap[pn][meta_user] = DataFrame({'prediction':y_pred,'label':y_label, 'method':meta_user, 'fold': fold}, index=range(len(y_pred)))
    
            print('... saved predictions for data type: {dtype} & meta users (n={n}) to dataframes'.format(dtype=file_type, n=len(meta_users)))
         
        #### optionally, run cluster analysis
        filter_axes = ['user', ]  # 'item'
        if enable_cluster_analysis: 
            # if meta users were included, consider R[:-n_meta_users] 
            for fdim in filter_axes: 
                
                # training split; if meta users were included, consider R[:-n_meta_users] 
                run_cluster_analysis(Pr, U=U, X=R, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=False, save_plot=True, file_type='train') # usually only at the training of the final model (i.e. after the best hyperparameters are determined; see wmf_ensemble_model_select())

                # test split
                run_cluster_analysis(Pt, U=U, X=T, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=False, save_plot=True, file_type='test') # usually only at the training of the final model (i.e. after the best hyperparameters are determined; see wmf_ensemble_model_select())

        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)

        # save training data?
        # test set
        # MFEnsemble.save_data((T, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data 
        # MFEnsemble.save_data((Th, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data  (~ prediciton)

        # training set
        # MFEnsemble.save_data((R, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-prior', verbose=verbose, subsampling=True)
        # MFEnsemble.save_data((Rh, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-posterior', verbose=verbose, subsampling=True)

        # return the dataset ID
        #    this is necessary because the 'best params' in each cycle may not be the same; for example, we may end up getting training data like this: 
        #       wmf_F100_A100_XCFuser_S2-train-prior-1.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-0.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-2.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-3.csv.gz
        #       wmf_F75_A100_XCFuser_S2-train-prior-4.csv.gz
        #    => in cycle {0, 2, 3, 4}, F75_A100 was the best, but in cycle {1, }, F100_A100 was the best
        vmap['dset_id'] = dset_id 
        vmap['best_params_inner'] = the_params

    ##############################################################################################################
    if save_data: 
        div('(wmf_ensemble_preferred_ratings) Output: saving transformed training and test sets (size: n(R): {nR}, n(T): {nT}), total size: {N}| delta(R): {dR}, delta(T): {dT} | algorithmic setting: {s}'.format(s=algorithm_setting, 
                    nR=Rh.shape[1], nT=Th.shape[1], N=n_samples_reconstructed, dR=delta_R, dT=delta_T), symbol='>')

        # test set
        MFEnsemble.save_data((T, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data 
        MFEnsemble.save_data((Th, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data  (~ prediciton)

        # training set
        MFEnsemble.save_data((R, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-prior', verbose=verbose, subsampling=True)
        MFEnsemble.save_data((Rh, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-posterior', verbose=verbose, subsampling=True)
    

    ### kind: specific MF algorithm
    if not kind: kind = 'als'
    wmfMetrics, wmfCV = [], [] # vmap['wmfMetrics'], vmap['wmfCV']

    # naming convention: method_id = '{prefix}_{id}_{learner_type}' # e.g. rf_bp_stacker
    aggregate_mode = params.get('policy_ms_model', 'mean')
    aggregate_func = params.get('policy_aggregate_func', 'mean')
    div("(wmf_ensemble_preferred_ratings) Comparison of model parameters | aggregate_func: {func}, mode: {mode}".format(func=aggregate_func, mode=aggregate_mode)) 
    # {stacker}.S-{dataset}-{suffix}
    # method_id = '{prefix}_{id}_{learner_type}'.format(prefix='wmf_%s' % kind, id=MFEnsemble.get_method_id(method, kind, params=params), learner_type=aggregate_func)
    method_id = "{prefix}.W-{id}-{suffix}".format(prefix=aggregate_func, id=MFEnsemble.get_method_id(method, kind, params=params), suffix=kind)
    ### model evaluation
    #   a. use mean proba in T vs labels as a summary score
    #   b. but we can compute a performance score for each base method (and then take the average)

    # header = MFEnsemble.header_model_evaluation # ['method', 'score', 'posterior_score', 'mean_score', ] ... used in evaluate.analyzePerf2()

    ############################################################################################################################
    # PerformanceMetrics object in wmfMetrics is the basis for selecting the best hyperparameter
    # 
    performance_metrics, pv = analyzePerf(L_test, Th, 
                                    method=method_id, aggregate_func=aggregate_func, 
                                        weights=Cwt,    # predcition by weighted average? 
                                        outer_fold=outer_fold, fold=fold,  # keep track of the iteration (debugging only when comparing Th and T)
                                        train_data=(Rh, L_train),  # only used in stacking mode
                                        mode=aggregate_mode)  # pass T=T to compare with T
    wmfMetrics.append( performance_metrics ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
    wmfCV.append( (L_test, pv) )   
    vmap['wmfMetrics'], vmap['wmfCV'] = wmfMetrics, wmfCV

    ##########################################################
    # other variables that may be of interest 
    # >>> piggyback inputs
    if piggyback: 
        # args, posargs = us.arguments()
        vmap['hyperparams'] = hyperparams  # keep track of hyperparameters for convenience 
    # vmap['fold'] = fold
    ##########################################################

    print("(wmf_ensemble_preferred_ratings) Ending cycle: ({fo}, {fi}) at setting {case} > returning vmaps: {keys} ... (verify) ".format(fo=outer_fold, fi=fold, 
        case=algorithm_setting, keys=vmap.keys()))

    # keys of vmap are the variables to return to caller: 
    #   i) saved in every cycle: wmfMetrics, wmfCV, hyperparams
    #  ii) saved only when training the final model after n cycles of model selection is completed: 
    #      dset_id, 
    #      best_params_nruns 
    return vmap 

def wmf_ensemble_preferred_ratings2(data, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', fold=-1, outer_fold=-1, null_marker=0, verbose=1, 
        project_path='?', piggyback=True, dev_ratio=0.2, max_dev=5000, resample=False, aggregation_methods=[], 
        post_hoc_analysis=False, save_data=False, enable_cluster_analysis=False):
    """
    
    Params
    ------
    fold: defined in a more generic sense than the 'fold' in wmf_ensemble_fold(). 
          specifically, 'fold' can refer to any of the following: 

           i) a CV fold ii) the index of iterations in random subsampling iii) other iteration index
    outer_fold: the iteration/fold number of the outer loop when wmf_ensemble_iter() is invoked for model selection (e.g. by model_select_core())

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
        div('(wmf_ensemble_preferred_ratings2) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2, 7, 8): 
            assert params['supervised']
        if params['setting'] in (3, 4, 9, 10): 
            assert not params['supervised']
        if params['setting'] in ( 9, 10): 
            assert params['predict_probs'], "Setting 9 - 10 should attempt to re-estimate the entire T"
    def verify_confidence_matrix(C, X, L, p_threshold, U=[], Cbar=None, measure='rank', message='', test_cases=[], plot=False, index=0):  # closure: params
        # if params['setting'] in (11, 12): 
        #     assert Cbar is not None
        #     assert params['policy_opt'].startswith('trade')
        if plot: 
            # closure: alpha, beta
            analyzer.plot_confidence_matrix(C, X, L, p_threshold, U=U, n_max=100, path=System.analysisPath, 
                measure=measure, target_label=None, alpha=params['alpha'], beta=2, index=index)

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
    def name_params_setting(method_params=['F', 'A']):   # [todo]
        # MFEnsemble.abridge(meta, identifiers=method_params, replace=replacement, verify=True)

        # use MFEnsemble.params_to_ids
        return 'F{nf}A{a}'.format(nf=params['n_factors'], a=params['alpha'])

    from evaluate import Metrics, plot_roc, analyzePerf
    import analyzer  # testing
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    import stacking

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
    # verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    # fold: iteration number (here, 'fold' is simply a repurposed variable borrowed from CV)
    # data: (R, T, L, Lt, U)-format
    if isinstance(data, DataFrame): # in this case, 'data' is a dataframe comprising a train-dev split 
        R, T, L_train, L_test, U =  uc.shuffle_split(data, ratio=dev_ratio, max_size=max_dev)
    else:
        # otherwise, assuming that shuffle-split operation had been invoked to produce the data input
        assert isinstance(data, (tuple, list)), "Invalid input data: %s" % data 
        print('(wmf_ensemble_preferred_ratings2) Input data is an n-tuple, whrere n={n}'.format(n=len(data)))
        R, T, L_train, L_test, U = data 

    ########################################################################################
    # resampling
    msg = ""
    if resample:

        # ver = 3
        # resampling_method = 'NearMiss(v{})'.format(ver)
        # nm = NearMiss(version=ver)    # undersampling based on KNNs (version 3 is less affected by noise)
        resampling_method = 'NeighbourhoodCleaningRule'
  
        msg += '(wmf_ensemble_preferred_ratings2) resampling to training data applied, method: {}\n'.format(resampling_method)
        msg += '... original dataset shape: %s\n' % collections.Counter(L_train)
        # for X, L in [(R, L_train), (Td, L_dev), ]:

        # training set
        R, L_train = uc.apply_resample(R, L_train, method=resampling_method)

        # test set
        # no-op
        msg += '... resampled dataset shape: %s\n' % collections.Counter(L_train)
    ########################################################################################
    print(msg)
    
    n_train, n_test = len(L_train), len(L_test)
    n_samples = R.shape[1]+T.shape[1]; assert n_train+n_test == n_samples

    ### create extra prediction vectors (PVs) from R (say, mean vector) and attach these new PVs to R (piggyback)
    n_users, n_items = R.shape
    n_classifiers = int(n_users/BagCount)
    n_users0, n_items0 = n_users, n_items  # keep a copy of the original number of users and items
    n_users_test, n_items_test = T.shape
    n_meta_users = 0

    # preference parameters
    isPreferenceScore = True; assert params['policy_opt'].startswith('pref') # and not params['replace_subset'] 
    tPreferenceCalibration = params['binarize_pref'] # params.get('preference_calibration', True)
    pref_threshold = params.get('pref_threshold', -1)
    pref_threshold_test = params.get('pref_threshold_test', -1)

    tWeightedPrediction = params.get('weighted_output', False)
    tCalibrateTwoWay = params.get('two_way_calibration', False)
    tExplicit = params.get('explicit_mf', False) if not isPreferenceScore else False
    tApproximateRatingsViaPreference = params.get('approx_ratings_via_pref', False)
    ############################################################
    # piggy back extra meta-estimators
    meta_users = ['latent_mean', 'latent_mean_masked',]  # todo
    if tMetaUsers: 
        div(message='(wmf_ensemble_preferred_ratings2) Adding meta users (n={n})...'.format(n=len(meta_users)))
        
        # e.g. mean, and masked mean
        nU, nUT = n_users, n_users_test

        print('... augmenting T (by meta users)')
        mean_pv = make_prediction_vector(T, L=[], policy='none')   # policy: 'none' => no masking
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
    
    # estimate labels (lh) in the test split 
    pos_label = 1
    # Eu = identify_effective_users(Cr, L_train=L_train, fill=null_marker)
    Eu = []
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=params['policy_threshold']) 
    Lh = lh = uc.estimateLabels(T, L=[], Eu=Eu, p_th=p_threshold, pos_label=pos_label, ratio_small_class=params['ratio_small_class']) 

    X = np.hstack((R, T))
    L = np.hstack((L_train, Lh))

    ########################################################################################
    # A. confidence matrix for re-estimating proba values
    policy_polarity = params.get('policy_polarity', 'sequence')

    # CX = uc.evalConfidenceMatrix(X, L=L, U=U,  
    #             ratio_users=-1,  # params['ratio_users'], 
    #             # ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

    #             policy=params['policy'], # <<< filter axis for the training split in X
    #             policy_test=params['policy_test'],  # <<< filter axis for the test split in X

    #             # policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
    #             policy_threshold=params['policy_threshold'],

    #                 # supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
    #                     conf_measure=params['conf_measure'], 
    #                         alpha=params['alpha'], beta=params.get('beta', 1.0), 
    #                         # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
    #                         fill=null_marker, 
    #                             is_cascade=True, n_train=n_train,
    #                             suppress_negative_examples=params['suppress_negative_examples'], 

    #                             # polarity matrix parameters 
    #                             constrained=False,  # params.get('constrained', True),
    #                             stochastic=params.get('stochastic', True),
    #                             estimate_sample_type=params.get('estimate_sample_type', True),
    #                             labeling_model='simple', # params.get('labeling_model', 'simple'),
    #                             policy_polarity=policy_polarity, # options: classification, median
    #                             # ... if policy_polarity is 'classification', then labeling_model is irrelavent
                                 
    #                             ##### for testing only 
    #                             L_test=L_test,
                                
    #                             fold=fold, path=System.analysisPath) # project_path=System.projectPath 
    # C0r, Mcxr, p_threshold, *CX_res = CX    # Cx_bar is removed
    # # assert np.sum(Mcxr==0) == 0, "policy_polarity: {} but got neutral particles (n={})?".format(policy_polarity, np.sum(Mcxr==0)) # ok

    # B. confidence matrix for preference scores
    # compute confidence matrix for approximating preference 
    CX = uc.evalConfidenceMatrix(X, L=L, U=U,  
            ratio_users=params['ratio_users'], 
            ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< filter axis for the training split in X
            policy_test=params['policy_test'],  # <<< filter axis for the test split in X

            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], 
                        alpha=params['alpha'], beta=params.get('beta', 1.0), 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, 
                            is_cascade=True, n_train=n_train,
                            suppress_negative_examples=params['suppress_negative_examples'], 

                            # polarity matrix parameters 
                            constrained=params.get('constrained', True),
                            stochastic=params.get('stochastic', True),
                            estimate_sample_type=params.get('estimate_sample_type', True),
                            labeling_model=params.get('labeling_model', 'simple'),
                            policy_polarity=policy_polarity, # options: classification, median
                             
                            ##### for testing only 
                            L_test=L_test,

                            fold=fold, path=System.analysisPath) # project_path=System.projectPath 
    C0, Po, p_threshold, *CX_res = CX    # Cx_bar is removed
    # else: 
    #     print('(wmf_ensemble_preferred_ratings2) using the same confidence matrix for approximating ratings ...')
    #     C0, Mcx = C0r, Mcxr
    ########################################################################################
    # ... Cx: confidence scores, zeros for neutral particles
    
    # verify_confidence_matrix(Cx, X, L, p_threshold, U=U, plot=False)
    msg = ''
    div("(wmf_ensemble_preferred_ratings2) Completed C(X) | Cycle {cycle} |  dim(Cui): {dim}, filter_axis: {fdim} | conf_measure: {measure}, optimization: {opt} | predict ALL probabilities? {tval} | policy_threshold: {p_th}".format(
        cycle=(outer_fold, fold), 
            dim=str(C0.shape), fdim=params['policy'], measure=params['conf_measure'], opt=params['policy_opt'], tval=params['predict_probs'], p_th=params['policy_threshold']), symbol='#', border=1)
    msg += '... Model | n_factors: {nf}, alpha: {a} | predict pref? {pref}, binarize? {bin}, supervised? {s}\n'.format(nf=params['n_factors'], a=params['alpha'], pref=isPreferenceScore, bin=params['binarize_pref'], s=params['supervised'])
    msg += '... Data | n_train: {ntr}, n_test: {nt}, ratio: {r}\n'.format(ntr=len(L_train), nt=len(L_test), r=(len(L_train)+len(L_test))/(len(L_test)+0.0)) 
    msg += '... Classifiers | n_users: {nu}, n_meta_users: {nmu} => n_original_users: {no}\n'.format(nu=n_users, nmu=n_meta_users, no=n_users-n_meta_users)
    print(msg)

    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} | Cycle: ({fo}, {f})'.format(dim=R.shape, f=fold, fo=outer_fold))
    piggyback_msg = "+  Cycle: ({fo}, {f}) | setting: {setting}".format(fo=outer_fold, f=fold, setting=algorithm_setting).rjust(5)   # keep track of the thread from which the iteration 
    
    ########################################################################################
    # ... determining Cn (n: neutral)
    # Pc = Mcxr # or Mcx?    # choose which polarity estimator and which confidence score
    tMaskNeutral = True
    weight_neutral = 0.0
    
    Cn = uc.make_cn(C0, Po, is_unweighted=False, weight_neutral=weight_neutral)
    Cn = uc.balance_and_scale(Cn, X=X, L=L, Po=Po, p_threshold=p_threshold, U=U, 
                        alpha=params['alpha'], beta=params.get('beta', 1.0), 
                        # ... beta: used to increase the weighting for TPs and FNs further
                            conf_measure=params['conf_measure'], 
                                suppress_max_class=params['suppress_negative_examples'], 
                                discount_test=params.get('discount_test', False), 
                                # .... discount_test: if True, suppress the weighting of the test split by gamma (0.5 by default)
                                    n_train=n_train, is_cascade=True, is_test_split=False, fold=fold)

    Cx = uc.make_cp(C0, Po, is_unweighted=False)  # mask only neutrals (whose corresponding entries are dropped out of the cost function)
    Cx = uc.balance_and_scale(Cx, X=X, L=L, Po=Po, p_threshold=p_threshold, U=U, 
                        alpha=params['alpha'], beta=params.get('beta', 1.0), 
                            conf_measure=params['conf_measure'], 
                                suppress_max_class=params['suppress_negative_examples'], 
                                discount_test=params.get('discount_test', False), 
                                    n_train=n_train, is_cascade=True, is_test_split=False, fold=fold)
    # ... Cn: both neutrals and negatives are masked (neutrals can be retained by assigning non-zero weights)
    # ... Cx: only neutrals are masked
    assert type(Cn) == type(Po) == type(Cx), "(dtype(Cn): {}, dtype(Cx): {}) <> dtype(Po): {}".format(type(Cn), type(Po), type(Cx))

    n_neutral_cx, n_neutral_cn = np.sum(Cx == 0), np.sum(Cn == 0)
    n_positive_cx, n_positive_cn = np.sum(Cx > 0), np.sum(Cn > 0)
    # assert n_neutral_cn > n_neutral_cx, \
    #     "Masked entries of C when approximating 'ratings' must be more than those of C when approximating preference | n_masked(Cn): {}, n_masked(Cx): {}".format(n_neutral_cn, n_neutral_cx)
    # ... Cw: is a masked version of Cx, where both neutral and negative poloarity examples have zero weights
    print("... Cx vs Cn | n_neutral_cx: {}, n_neutral_cn: {} | n_positive_cx: {}, n_positive_cn: {} | policy_polarity: {}".format(n_neutral_cx, n_neutral_cn, n_positive_cx, n_positive_cn, policy_polarity))
    P = Q = None
    if not tApproximateRatingsViaPreference: 
        print('... (1) approximating ratings (X) via Cn (n_masked: {}) | Cycle ({fo}, {fi})'.format(n_neutral_cn, fo=outer_fold, fi=fold))
        P, Q, *Xh_errs = ua.implicit_als(Cn, features=params['n_factors'], 

                                iterations=params['n_iter'],
                                lambda_val=System.lambda_val,  # 0.8 by default

                                # label_confidence=Cx_bar, 
                                polarity=Po,     # colored polarity
                                ratings=X, labels=L,
                                policy='rating', message=piggyback_msg, ret_rmse=True)
        Xh_err, Xh_err_weighted = Xh_errs

        ne = 2
        e_pri, e_post = np.mean(Xh_err[:ne]), np.mean(Xh_err[-ne:])
        e_del = (e_pri-e_post)/e_pri * 100
        print('... (ALS 1) Complete | rmse: {e1} -> {e2} (n_errs: {n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(e1=e_pri, e2=e_post, 
                       e_del=e_del, ew1=np.mean(Xh_err_weighted[:ne]), ew2=np.mean(Xh_err_weighted[-ne:]), n=len(Xh_err) ))
    else: 
        print("... (1) defer ratings (X) approximation after preference scores are obtained ...")
    ########################################################################################

    print('... (2) estimating preferences (X) via Cx (n_masked: {}) | Cycle ({fo}, {fi})'.format(n_neutral_cx, fo=outer_fold, fi=fold))
    Pp, Qp, *Xp_errs = ua.implicit_als(Cx, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            # label_confidence=Cx_bar, 
                            polarity=Po,   
                            ratings=X, labels=L,
                            policy='preference', message=piggyback_msg, ret_rmse=True)

    ########################################################################################
    # e.g. 50 users, 3979 items with nf=100 > dim(P): (50, 100), dim(Q): (3979, 100)
    if P is not None and Q is not None: 
        assert P.shape[0] == X.shape[0] and P.shape[1] == params['n_factors']
        assert Q.shape[0] == X.shape[1] and Q.shape[1] == params['n_factors']
        assert Pp.shape == P.shape and Qp.shape == Q.shape

    ### ALS evaluation (RMS)
    print('(wmf_ensemble_preferred_ratings2) Prior to reconstruct_by_preferernce() | pref_threshold given? {}'.format(pref_threshold))
    # -- A. calibrate together
    factors = (P, Q) if not tApproximateRatingsViaPreference else None
    Xh, Xp, pref_threshold, *res = \
        reconstruct_by_preference(Cn, X, factors=factors, prefs=(Pp, Qp), labels=L, 
                
                # [test]
                polarity_matrix=Po,  # external polarity matrix 
                test_labels=np.hstack((L_train, L_test)),
                
                binarize=tPreferenceCalibration, 
                    p_threshold=p_threshold, 
                    pref_threshold=pref_threshold, # we don't know the preference threshold yet
                    policy_calibration=params['policy_calibration'],
                        is_cascade=True, n_train=n_train, 
                           replace_subset=True, replace_all=False, params=params, null_marker=0, 
                           name='X', verify=True, index=(outer_fold, fold), 

                               # only relevant when factors=(P, Q) has not been computed or not given
                               n_factors=params['n_factors'], n_iter=params['n_iter'],
                                   unweighted=tExplicit, message=piggyback_msg)
    if len(res) > 0: P, Q = res[0], res[1]
    # ... Now we have Xh, Xp, (Pp, Qp), and (P, Q)
    
    delta_X = LA.norm(Xh-X, 'fro')
    Rh, Th = Xh[:,:n_train], Xh[:,n_train:]
    Rp, Tp = Xp[:,:n_train], Xp[:,n_train:]
    pref_threshold_test = pref_threshold
    # ... Xh is a (re-estimated) proba matrix; Xp: preference matrix (binarized)
    
    # -- B. cailbrate separately
    if tCalibrateTwoWay: 
        Cr, Ct = Cx[:,:n_train], Cx[:,n_train:]
        Qr, Qt = Q[:n_train, :], Q[n_train:, :]  # row(Q) ~ items/data
        Qpr, Qpt = Qp[:n_train,:], Qp[n_train:, :]
        Lh_R, Lh_T = L_train, L[n_train:]

        Rh, Rp, pref_threshold = \
            reconstruct_by_preference(Cr, R, factors=(P, Qr), prefs=(Pp, Qpr), labels=Lh_R, 
                    # test_labels=np.hstack((L_train, L_test)),
                    binarize=tPreferenceCalibration, 
                        p_threshold=p_threshold, 
                        pref_threshold=pref_threshold, # we don't know the preference threshold yet
                        policy_calibration=params['policy_calibration'],
                            is_cascade=False, 
                               replace_subset=True, replace_all=False, params=params, null_marker=0, name='R', verify=True, index=(outer_fold, fold))
        # ... Th is a proba matrix; Xp: preference matrix (binarized)
        Th, Tp, pref_threshold_test = \
            reconstruct_by_preference(Ct, T, factors=(P, Qt), prefs=(Pp, Qpt), labels=Lh_T, 
                    test_labels=L_test,
                    binarize=tPreferenceCalibration, 
                        p_threshold=p_threshold_test, 
                        pref_threshold=-1, # we don't know the preference threshold yet
                        policy_calibration=params['policy_calibration'],
                            is_cascade=False, 
                               replace_subset=True, replace_all=False, params=params, null_marker=0, name='T', verify=True, index=(outer_fold, fold))
        print('(wmf_ensemble_preferred_ratings2) two-way calibration | th(R): {} <?> th(T): {}'.format(pref_threshold, pref_threshold_test))

    print('(wmf_ensemble_preferred_ratings2) preference thresholds | th(R): {} ~? th(T): {}'.format(pref_threshold, pref_threshold_test))
    assert Rh.shape == R.shape, "dim(R): {}, dim(Rh): {}".format(R.shape, Rh.shape)
    assert Th.shape == T.shape, "dim(T): {}, dim(Th): {}".format(T.shape, Th.shape)

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    # ... CF-transform T to get Th (using the classifier/user vectors learned from the training set (R))
    div("(wmf_ensemble_preferred_ratings2) Completed rating matrix reconstruction | Cycle: ({fo}, {fi}) | preference scores? {tval}, action='{act}'".format(
        fo=outer_fold, fi=fold, tval=isPreferenceScore, act='Replace Subset')) # predict => predict probabilities

    # classifier/user weights (usually just use confidence matrix)
    Cw = Cwt = None
    if tWeightedPrediction:
        Cw = Xp * C0 
        Cw, Cwt = Cw[:,:n_train], Cw[:,n_train:]
        print('(wmf_ensemble_preferred_ratings2) using weighetd output via confidence matrix | dim(Cwt): {}'.format(Cwt.shape))
    else: 
        Cw, Cwt = Xp[:,:n_train], Xp[:,n_train:]  # this will result in discarding new estimates wherever preference scores == 0
        print('(wmf_ensemble_preferred_ratings2) using only preference matrix (non-weigthed) for making final predictions ...')

    # X: (R, T, L_train, L_test, U)
    n_samples_reconstructed = Rh.shape[1]+Th.shape[1]
    ##############################################################################################################
    # ... prediction 
    if not aggregation_methods: aggregation_methods = System.aggregation_methods # e.g. ['mean', 'median', 'log', ]  
    # pv_mean = uc.combiner(Th, aggregate_func='mean')           

    ##############################################################################################################
    # ... input: 
    #     Rh: new ratings by replacing unreliable entries with new estimates (reliabilty is estimated by preference Rp)
    #     Rp: (estimated) preference scores in R
    #     Th: new ratings for T 
    #     Tp: (estimated) prefeerence scores in T
    file_types = ['prior', 'posterior', ]
    if post_hoc_analysis:   # only triggered after model selection cycle is complete and 'best params' has been determined
        
        # note that we shall save the data only after model selection loop is completed
        assert outer_fold >= 0 and fold == -1, "Intended action: only save the final model after model selection is complete (outer fold: {fo}, inner fold: {fi}".format(fo=outer_fold, fi=fold)
        the_params = name_params_setting(method_params=['F', 'A'])

        div('(wmf_ensemble_preferred_ratings2) Running posthoc analysis | algorithmic setting: {s}'.format(s=algorithm_setting), symbol='%') 
        # MFEnsemble.save_meta_tset((Rh, Th, L_train, L_test, U), fold=fold, method=method, params=params, verbose=verbose)

        pv_t = pv_th = pv_pref = None
        if tMetaUsers: 
            # drop meta users 
            R = R[:-n_meta_users]
            Rh = Rh[:-n_meta_users]
            Rp = Rp[:-n_meta_users]
            T, pv_t = T[:-n_meta_users], T[-n_meta_users:] 
            Th, pv_th = Th[:-n_meta_users], Th[-n_meta_users:]
            Tp, pv_pref = Tp[:-n_meta_users], Tp[-n_meta_users:]
            assert Th.shape[0] + pv_th.shape[0] == n_users_test
            assert pv_t.shape[1] == pv_th.shape[1] == len(L_test)

            P = P[:-n_meta_users]

            if Cwt is not None: 
                Cwt = Cwt[:-n_meta_users]
                assert Th.shape == Cwt.shape, "dim(Th): {}, dim(Cwt): {}".format(Th.shape, Cwt.shape)

        ### save predictions
        # Tp is a binary matrix
        n_uniq_pref = len(np.unique(Tp))
        assert n_uniq_pref == 2, "(wmf_ensemble_preferred_ratings2) Tp is not binary or is degenerated | n_uniq(Tp): {} | values(Tp): {}".format(n_uniq_pref, np.unique(Tp))
        # Th is a continuous rating matrix
        n_uniq = len(np.unique(Th))
        assert n_uniq > 2, "(wmf_ensemble_preferred_ratings2) Unique ratings should be >> 1 | u_uniq(T): {}".format(n_uniq)
        # ... although in this routine, Th is expected to be already a probability matrix (conditioned on the preference matrix).

        # the_params = name_params_setting(method_params=['F', 'A'])
        dataset = {'prior': [R, T], 'posterior': [Rh, Th]}
        for file_type in file_types: 
            X_train, X_test = dataset[file_type]
            for i, aggr in enumerate(aggregation_methods):    # defined earlier e.g. ['mean', 'median']
                pn = '{ftype}_{model}'.format(ftype=file_type, model=aggr)  # pn: predictor name
                
                # TODO: if aggr in ['logistic', ...]  # need to further train on X_train and test on X_test
                if aggr in ['mean', 'median', ]:   # System.simple_aggregation

                    # use Xh
                    if file_type.startswith('pri'): 
                        pv = uc.combiner(T, aggregate_func=aggr)  # T 
                    else: 
                        # 1. use Xh
                        pv = uc.combiner(Th, weights=Cwt, aggregate_func=aggr)  # Th
                        
                        # 2. use Xp
                        # pv = uc.predict_by_preference(T, Tp, W=Cwt, name='weighted preference', aggregate_func=aggr, fallback_on_low_weight=False, verify=True) 

                else:
                    ### put stacker code here! 

                    stacker = stacking.choose_classifier(aggr)  # e.g. log, enet, knn
                    model = stacker.fit(X_train.T, L_train) # remember to take transpose
                    pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data

                    # stacker training on Rh and testing on Th does not make sense
                    # if file_type.startswith('pri'): 
                    #     stacker = stacking.choose_classifier(aggr)  # e.g. log, enet, knn
                    #     model = stacker.fit(X_train.T, L_train) # remember to take transpose
                    #     pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data
                    # else: 
                    #     pv = []

                if len(pv) == len(L_test): # if prediction vector is not null
                    y_pred, y_label = pv, L_test
                    vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

        ### save predictions from the meta users (and treat these data as if they came from an aggregation method)
        if tMetaUsers: 
            # vmap['prior_mean'], vmap['posterior_mean'] = {}, {}
            predictions = {'prior': pv_t, 'posterior': pv_th}  # each prediction dataframe consists of prediction vectors from n meta_users
            for file_type in file_types: 
                for i, meta_user in enumerate(meta_users): 
                    pn = '{ftype}_{model}'.format(ftype=file_type, model=meta_user) 
                    pv = predictions[file_type]

                    y_pred, y_label = pv[i], L_test
                    vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

            # for file_type, pv in predictions.items(): 
            #     pn = '{ftype}_{model}'.format(ftype=file_type, model='mean') # pn: predictor name
            #     vmap[pn] = {}
            #     for i, meta_user in enumerate(meta_users): # ['latent_mean', 'latent_mean_masked',]
            #         y_pred, y_label = pv[i], L_test
            #         vmap[pn][meta_user] = DataFrame({'prediction':y_pred,'label':y_label, 'method':meta_user, 'fold': fold}, index=range(len(y_pred)))
    
            print('... saved predictions for data type: {dtype} & meta users (n={n}) to dataframes'.format(dtype=file_type, n=len(meta_users)))
         
        #### optionally, run cluster analysis
        filter_axes = ['user', ]  # 'item'
        if enable_cluster_analysis: 
            # if meta users were included, consider R[:-n_meta_users] 
            for fdim in filter_axes: 
                
                # training split; if meta users were included, consider R[:-n_meta_users] 
                run_cluster_analysis(P, U=U, X=X, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=False, save_plot=True, file_type='train') # usually only at the training of the final model (i.e. after the best hyperparameters are determined; see wmf_ensemble_model_select())

        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)

        vmap['dset_id'] = dset_id 
        vmap['best_params_inner'] = the_params

    ##############################################################################################################
    if save_data: 
        div('(wmf_ensemble_preferred_ratings2) Output: saving transformed training and test sets (size: n(R): {nR}, n(T): {nT}), total size: {N}| delta(R): {dR}, delta(T): {dT} | algorithmic setting: {s}'.format(s=algorithm_setting, 
                    nR=Rh.shape[1], nT=Th.shape[1], N=n_samples_reconstructed, dR=delta_R, dT=delta_T), symbol='>')

        # test set
        MFEnsemble.save_data((T, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data 
        MFEnsemble.save_data((Th, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data  (~ prediciton)

        # training set
        MFEnsemble.save_data((R, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-prior', verbose=verbose, subsampling=True)
        MFEnsemble.save_data((Rh, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-posterior', verbose=verbose, subsampling=True)
    

    ### kind: specific MF algorithm
    if not kind: kind = 'als'
    wmfMetrics, wmfCV = [], [] # vmap['wmfMetrics'], vmap['wmfCV']

    # naming convention: method_id = '{prefix}_{id}_{learner_type}' # e.g. rf_bp_stacker
    aggregate_mode = params.get('policy_ms_model', 'mean')
    aggregate_func = params.get('policy_aggregate_func', 'mean')
    div("(wmf_ensemble_preferred_ratings2) Comparison of model parameters | aggregate_func: {func}, mode: {mode}".format(func=aggregate_func, mode=aggregate_mode)) 
    # {stacker}.S-{dataset}-{suffix}
    # method_id = '{prefix}_{id}_{learner_type}'.format(prefix='wmf_%s' % kind, id=MFEnsemble.get_method_id(method, kind, params=params), learner_type=aggregate_func)
    method_id = "{prefix}.W-{id}-{suffix}".format(prefix=aggregate_func, id=MFEnsemble.get_method_id(method, kind, params=params), suffix=kind)
    ### model evaluation
    #   a. use mean proba in T vs labels as a summary score
    #   b. but we can compute a performance score for each base method (and then take the average)

    # header = MFEnsemble.header_model_evaluation # ['method', 'score', 'posterior_score', 'mean_score', ] ... used in evaluate.analyzePerf2()

    ############################################################################################################################
    # PerformanceMetrics object in wmfMetrics is the basis for selecting the best hyperparameter
    # 
    pv = uc.predict_by_preference(T, Tp, W=Cwt, name='weighted preference', aggregate_func='mean', fallback_on_low_weight=False, verify=True)
    prediction = pv # or Th
    performance_metrics, pv = analyzePerf(L_test, prediction,   # or (L_test, Th)
                                    method=method_id, aggregate_func=aggregate_func,
                                        weights=Cwt,  
                                        outer_fold=outer_fold, fold=fold,  # keep track of the iteration (debugging only when comparing Th and T)
                                        train_data=(Rh, L_train),  # only used in stacking mode
                                        mode=aggregate_mode)  # pass T=T to compare with T
    wmfMetrics.append( performance_metrics ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
    wmfCV.append( (L_test, pv) )   
    vmap['wmfMetrics'], vmap['wmfCV'] = wmfMetrics, wmfCV

    ##########################################################
    # other variables that may be of interest 
    # >>> piggyback inputs
    if piggyback: 
        # args, posargs = us.arguments()
        vmap['hyperparams'] = hyperparams  # keep track of hyperparameters for convenience 
    # vmap['fold'] = fold
    ##########################################################

    print("(wmf_ensemble_preferred_ratings2) Ending cycle: ({fo}, {fi}) at setting {case} > returning vmaps: {keys} ... (verify) ".format(fo=outer_fold, fi=fold, 
        case=algorithm_setting, keys=vmap.keys()))

    # keys of vmap are the variables to return to caller: 
    #   i) saved in every cycle: wmfMetrics, wmfCV, hyperparams
    #  ii) saved only when training the final model after n cycles of model selection is completed: 
    #      dset_id, 
    #      best_params_nruns 
    return vmap

def model_select_core(data, params, param_grid, vars=['wmfCV', 'wmfMetrics', ], kind='als', dev_ratio=None, max_dev=5000, n_trial=0, outer_fold=-1, null_marker=0, 
        post_hoc_analysis=False, enable_cluster_analysis=False, resample=False, verbose=True): 
    """
    This subroutine loops through all possible combinations of (hyper-)parameters. 

    Memo
    ----
        R = train_df.values.T  # R: users vs items
        Td = dev_df.values.T
        Tt = test_df.values.T
        U = train_df.columns.values
    """
    def select_training_mode():  # closure: params 
        training_mode = params.get('training_mode', 'regular')
        if training_mode.startswith(('reg', 'train')):
            return wmf_ensemble_iter  # wmf_ensemble_preferred_ratings
        elif training_mode == 'cascade': # 'cascade'
            return wmf_ensemble_iter2  # train R and T (minus the label) at the same time in order to get T's latent factors
        elif training_mode == 'pref':  # 'preferred_rating'
            return wmf_ensemble_preferred_ratings

        div('(model_select_core) training mode: {}'.format(training_mode))
        return wmf_ensemble_preferred_ratings2  # mode: 'pref_cascade': train both preference and rating models

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
    inner_fold = n_trial
    print('(model_select_core) MS cycle (%d, %d) | param_grid: %s > trigger model selection? %s  ... (verify) #' % (outer_fold, inner_fold, param_grid, tModelSelection))

    # [note] 'fold_count' is the fold count through which the base predictors were trained 
    # R, Td, T, L_train, L_dev, L_test, *rest = uc.to_rating_matrix_random_subsampling(dev_ratio=dev_ratio, fold_count=System.foldCount, policy='random_cv_fold', shuffle=True, return_index=True)
    # print('(verify) Fold (outer loop index): {fold}, n_trial (inner loop index): {nt} | dim(R): {dR}, dim(Td): {dTd}, dim(T): {dT}'.format(fold=fold, nt=n_trial, dR=R.shape, dTd=Td.shape, dT=T.shape))
    # U, *Ix = rest

    # >>> steps: 
    # use (R, Td), (L_train, L_dev) to select the best model 
    # R <- R + Td
    # L_train <- L_train + L_dev 
    # use (R, T), (L_train, L_test) to evaluate the final performance (and performance comparisons with other stacking models)

    wmf_core_routine = select_training_mode()  # closure: params 

    ### model selection loop
    ###########################################
    scores = {}
    n_train_dev = max_dev if max_dev is not None else df_train_dev.shape[0]
    if not tModelSelection: 
        print("(model_select_core) Skipping model selection loop | n(train+dev): {} ... ".format(n_train_dev))
        best_params = us.frozendict( {p: vals[0] for p, vals in param_grid.items()} )  # Nothing else to choose from given only 1 option; we are done
        scores[best_params] = 0.5  # dummy score
    else:  
        div(message='(model_select_core) n(train_dev): {n} | to sample only a subset? {tval} | n_total(train+dev): {N}'.format(n=n_train_dev, 
            tval=max_dev is not None, N=df_train_dev.shape[0]))

        # save <- False to not save the data
        # options: use python thread: prefer="threads"
        models = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_core_routine)(
                            data=uc.shuffle_split(df_train_dev, ratio=dev_ratio, max_size=max_dev, resample=resample, index=inner_fold), # uc.shuffle_split return a 5-tuple: (R, Td, L_train, L_dev, U)
                                params=params, hyperparams=hyperp, vars=vars, kind=kind, null_marker=null_marker, fold=inner_fold, outer_fold=outer_fold, 
                                    post_hoc_analysis=post_hoc_analysis, enable_cluster_analysis=enable_cluster_analysis) # both analyis set to False by default
                                        for i, hyperp in enumerate(ParameterGrid(param_grid)) )  
        print('... finished model selection cycle #{cycle} | completed {n} models/combinations #\n... example: {ex}'.format(
            cycle=inner_fold, n=len(models), ex=models[0]))

        # [note] strictly speaking, the first call to uc.shuffle_split() is redundant because the input train-dev split (df, labels) from the caller wmf_ensemble_model_select() 
        # is already a randomized version of the train-dev split
        ##############################################

        for im, model in enumerate(models):  

            # >>> design: how to index into hyperparameter settings?
            # entry = MFEnsemble.get_method_id(method, params=model['hyperparams']) 
            entry = us.frozendict(model['hyperparams']) #  ... (1a) hyperparams is a dictionary
            perf = PerformanceMetrics.merge([ PerformanceMetrics.consolidate(model['wmfMetrics']), ])  # consolidate predictions from multiple cycles (or CV fold when applicable)

            # [test] number of method should be equal to len(kinds): i.e. one WMF algorithm (kind) -> one method #
            #            dim(table): 6 (metrics) by 1 (method)
            #            n_method == n_kinds
            print('... (verify) model #{i} | dim(perf.table): {dim}, methods: {methods}, n_method:{n} =?= n_kinds: {nk}'.format(i=im, dim=perf.table.shape, methods=list(perf.table.columns), n=perf.n_methods(), nk=1) )
            print('...... Fmax: {fmax}\n...... AUC: {auc}\n'.format(fmax=np.mean(perf.table.loc['fmax']), auc=np.mean(perf.table.loc['auc'])) )
        
            if not entry in scores: scores[entry] = 0

            # scores: hyperparams -> score
            scores[entry] = perf.sort(metric='fmax')[0][1]  # perf.sort(metric='fmax') returns a sorted list of (methed, score)-tuples
        ### end foreach model
        # ... scores: hyperparameter -> performance score

        # print('>>> Cycle #{n_trial} | score values:\n{scores}\n'.format(n_trial=n_trial, scores=scores.values())) # should be all different!
        print( us.format_sort_dict(scores, reverse=True, padding=4, title="(model_select_core) model selection (metric: {metric}) | cycle: ({fo}, {fi})".format(metric='fmax', fo=outer_fold, fi=inner_fold)) )
    
        # sorted(...) returns the frozendicts with hyperparams in the order of their corresponding fmax scores 
        best_params = sorted(scores, key=scores.__getitem__, reverse=True)[0]
        print('(model_select_core) best params in MS cycle #{nt} > n_factors: {nf}, alpha: {a}'.format(nt=inner_fold+1, nf=dict(best_params)['n_factors'], a=dict(best_params)['alpha']))  # ... (1b)
        # best_score = scores[best_params]

    return best_params, scores # best params (in frozen dict) -> score (e.g. fmax)

def wmf_ensemble_model_select(params, param_grid={}, vars=['wmfCV', 'wmfMetrics', ], kind='als', n_trials=1, 
        dev_ratio=0.2, test_ratio=0.3, max_dev=5000, resample=False, null_marker=0, fold=-1, unbag=False, 
        save_data=False, post_hoc_analysis=True, enable_cluster_analysis=True, verbose=True, input_path=None):
    """
    Each call to wmf_ensemble_model_select() is an independent run that computes a "best model" (from among a set of candidate parameter combinations), followed by 
    evaluating the resulting model in a test set. In other words, each call to this subroutine references a different test set (generated from an external CV routine). 

    The fold number (e.g. 5 in a 5-fold CV) determines the number of calls to this subroutine. 


    Params
    ------
    n_trials: number of runs for the model selection loop
    save: if True, save the training data of the concluded model with the 'best params'

    dev_ratio:
    max_dev: max sample size for model selection (used to control speed)
    policy_ms: the policy by which the best parameter setting is determined
               'freq'
               'mean'
    vars: 
        note that 'vars' happens to be one of Python's built-in funciton but we do not need it here. 

    fold: fold can either represent a fold number in a CV or the index into a particular run (of a subsampling iteration)


    Call
    ----
    wmf_ensemble_model_select(params, param_grid, vars=['wmfCV', 'wmfMetrics', ], kinds=['als', ], null_marker=null_marker, dev_ratio=dev_ratio)

    """ 
    def select_training_mode():  # closure: params 
        training_mode = params.get('training_mode', 'regular')
        if training_mode.startswith(('reg', 'train')):
            return wmf_ensemble_iter  # wmf_ensemble_preferred_ratings
        elif training_mode == 'cascade': # 'cascade'
            return wmf_ensemble_iter2  # train R and T (minus the label) at the same time in order to get T's latent factors
        elif training_mode == 'pref':  # 'preferred_rating'
            return wmf_ensemble_preferred_ratings

        div('(wmf_ensemble_model_select) training mode: {}'.format(training_mode))
        return wmf_ensemble_preferred_ratings2  # mode: 'pref_cascade': train both preference and rating models

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
    # model selection setting
    tModelSelection = sum(1 for v in param_grid.values() if len(v) > 1) > 0 
    tPosthoc = tSaveBestModel = post_hoc_analysis  # if True, will activate upon training the final model (post model selection)
    tClusterAnalysis = enable_cluster_analysis # if True, will activate upon training the final model
    tSaveData = save_data
    policy_ms = params.get('policy_ms', 'freq')
    policy_ms_model = params.get('policy_ms_model', 'mean')
    policy_aggregate_func = params.get('policy_aggregate_func', 'mean')
    ########################################################################################

    vmap = {}
    method = 'wmf'

    if input_path is None: 
        input_path = ProjectPath
    else: 
        assert os.path.exists(input_path), "Invalid path to dataset: {path}".format(path=input_path)
    ########################################################################################
    System.display()
    cf_spec.config(project_path=input_path, domain=Domain, fold_count=FoldCount, bag_count=BagCount)
    ########################################################################################
    
    wmf_core_routine = select_training_mode()
    print('(wmf_ensemble_model_select) param_grid: %s > trigger model selection? %s ... (verify)' % (param_grid, tModelSelection))
    # train-dev-test split
    # train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.read_random_fold(input_path, fold_count=FoldCount, dev_ratio=dev_ratio, shuffle=True)
    # note: 
    #   1. set test_ratio to 0.5 so that half of the transformed data can be saved. 
    
    # train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.shuffle_split(input_path, split_number=3, dev_ratio=dev_ratio, test_ratio=test_ratio, 
    #     fold_count=-1, max_size=None, random_state=None)
    outer_fold = fold # this 'fold' number really corresponds to the outer fold
    assert fold < FoldCount, "Cannot have a fold/index number larger than the fold count which generated the level-0 dataset. (fold count: {n})".format(n=FoldCount)
    train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.shuffle_split_cv(input_path, 
        shuffle=False, split_number=3, dev_ratio=dev_ratio, max_size=None, fold=outer_fold, verbose=True)
    # ... test split (test_df, test_labels) depends on the fold number (index) while train-dev split results from applying subsampling on the train split

    tHasDev = True if dev_df.shape[0] > 0 else False
    if unbag:
        train_df = common.unbag(train_df, BagCount) # mean aggregates (average over all bags)
        if tHasDev: dev_df = common.unbag(dev_df, BagCount)
        test_df = common.unbag(test_df, BagCount)
    
    # data statistics 
    n_train, n_dev, n_test = train_df.shape[0], dev_df.shape[0], test_df.shape[0]
    N = n_train + n_dev + n_test
    print('(wmf_ensemble_model_select) data split ratios | train: {rtr}, dev: {rd}, test: {rt} | routine used: {rn}  ... (verify)'.format(rtr=n_train/(N+0.0), rd=n_dev/(N+0.0), rt=n_test/(N+0.0), rn='common.shuffle_split_cv'))
    # ... for 5-fold CV, should be approx. 60%, 20%, 20%

    # Convert input data to rating matrix format
    ########################################################################################
    R = train_df.values.T
    Td = dev_df.values.T
    T = test_df.values.T
    U = train_df.columns.values
    L_train, L_dev, L_test = train_labels, dev_labels, test_labels

    # combine train and dev split for model selection (so that each run has its own separate random splits between train and dev)
    if tHasDev: 
        df = pd.concat([train_df, dev_df])
        labels = np.hstack((train_labels, dev_labels))
    else: 
        df = train_df 
        labels = train_labels

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
        ### model selection loop (analogous to inner CV iterations)
        best_models = Parallel(n_jobs = -1, verbose = 1)(delayed(model_select_core)(
                                                            data=D_minus, params=params, param_grid=param_grid, vars=vars, dev_ratio=dev_ratio, max_dev=max_dev, resample=resample, kind=kind, null_marker=null_marker, 
                                                                n_trial=n_trial, outer_fold=outer_fold, post_hoc_analysis=False, enable_cluster_analysis=False) 
                                                                    for n_trial in range(n_trials) ) 
        assert len(best_models) == n_trials
        for i, best_model in enumerate(best_models):  # foreach 'best model in a trial'
            best_params, scores = best_model  # scores is a map from best_params to its performance score (e.g. fmax)

            if not best_params in bestScores: bestScores[best_params] = []  # best_params is a frozendict
            bestScores[best_params].append(scores[best_params])
        print("(wmf_ensemble_model_select) all 'best_params' after {n} MS-cycles: {list} ... (verify)".format(n=n_trials, list=bestScores.keys()))
    else: 
        print('(wmf_ensemble_model_select) Cycle #{c}, to run {n} model-selection routine ...'.format(c=outer_fold, n=n_trials))
        for n_trial in range(n_trials): 

            # [note] 'fold_count' is the fold count through which the base predictors were trained 
            R, Td, T, L_train, L_dev, L_test, *rest = uc.to_rating_matrix_random_subsampling(dev_ratio=dev_ratio, 
                fold_count=System.foldCount, policy='random_cv_fold', shuffle=True, return_index=True, unbag=System.unbag, bag_count=BagCount)
            print('(verify) Fold (outer loop index): {fold}, n_trial (inner loop index): {nt} | dim(R): {dR}, dim(Td): {dTd}, dim(T): {dT}'.format(fold=outer_fold, nt=n_trial, dR=R.shape, dTd=Td.shape, dT=T.shape))
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
                models = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_core_routine)(
                                    data=D_minus, params=params, hyperparams=hyperp, vars=vars, kind=kind, null_marker=null_marker, fold=n_trial, 
                                        post_hoc_analysis=False, enable_cluster_analysis=False) 
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
    # ... bestScore: best hyperparams -> a list of scores

    M = Metrics(bestScores, op=np.mean) 
    print("(wmf_ensemble_model_select) Cycle: {fold} | found {n} sets of 'best params' with {nv} scores ... (verify) #".format(fold=outer_fold, n=M.size(), nv=M.size_bags())) # len(next(iter(bestScores.values())))
    M.display(by='freq')  # display multibags

    print('(wmf_ensemble_model_select) Now take the average of the scores for each parameter setting ...')
    M_mean = M.aggregate(by='mean')  # aggregate performance scores via 'op'; if op is a mean function, then this equates to taking the average
    # M.sort(by='mean') # ... return a sorted list of 2-tuples
    
    title = "(wmf_ensemble_model_select) Model selection performance ordering (via policy {pms}) in Cycle (outer): {fold} | n_trials={nt}, metric={metric} ... (result)".format(pms='mean', fold=outer_fold, nt=n_trials, metric='fmax')
    print( us.format_sort_dict(M_mean, reverse=True, padding=5, title=title)) # symbol='#', border=1

    M_freq = M.aggregate(by='freq') # M.sort(by='freq')  # aggregate performance scores via 'op'; if op is a mean function, then this equates to taking the average
    title = "(wmf_ensemble_model_select) Model selection performance ordering (via policy {pms}) in Cycle (outer): {fold} | n_trials={nt}, metric={metric} ... (result)".format(pms='freq', fold=outer_fold, nt=n_trials, metric='fmax')
    print( us.format_sort_dict(M_freq , reverse=True, padding=5, title=title)) # symbol='#', border=1

    # >>> select 'best of the best'
    ############################################
    # use frequency or average? frequency is more stable
    best_models, models_sorted = (), []
    if policy_ms.startswith('freq'):
        # models: a list of 2-tuples
        models_sorted = M.sort(by='freq')
        best_models = models_sorted[0]  #  0 indexes into the best model by frequency
    else:
        models_sorted = M.sort(by='mean')  
        best_models = models_sorted[0]
    
    best_params, best_score = best_models[0], best_models[1]  # best_params: a frozen dictionary
    best_params = dict(best_params) # remember to defrost
    div(message='(wmf_ensemble_model_select) Cycle {index} | Best of the best params across n_trials={nt} > n_factors: {nf}, alpha: {a} | policy_ms={pms}, score: {score}... (verify) #'.format(index=outer_fold, nt=n_trials, 
        nf=best_params['n_factors'], a=best_params['alpha'], pms=policy_ms, score=best_score), symbol='#', border=2)  # ... (1c)
    ############################################
    # ... now, narrow down to only a single 'best' model
    vmap['rank'] = models_sorted   # keep track of the ranking of models within each run

    # use D instead of D_minus to train the final model
    ####################################################################################
    # ... train the final model using the best parameters obtained from the dev set
    D = (np.hstack((R, Td)), T, np.hstack((L_train, L_dev)), L_test, U) # + Ix   
    # ... D is typically a 5-tuple: (R, T, L, Lt, U)-format
    # ... Ix: (df.index, test_df.index) i.e. the original training set index and test set index   
    params = {**params, **best_params}  # update params by best_params

    # note: nth 'fold' really means the nth cycle of running wmb_ensemble_model_select() here
    finalModel = wmf_core_routine(data=D, params=params, vars=vars, indices=Ix, kind=kind, null_marker=null_marker, 
                    outer_fold=outer_fold, resample=resample, 
                    piggyback=False, # set piggyback to False to avoid returning additional variables as side effects (e.g. hyperparams)
                        save_data=tSaveData, 
                        post_hoc_analysis=tPosthoc, enable_cluster_analysis=tClusterAnalysis) 
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
            fold=outer_fold, algo=kind, name=entry['name'], params=entry['best_params'], s=entry['score']))
    ########################################################################################

    div('Model seletion iteration (cycle: {n_iter}) complete ... (verify)'.format(n_iter=outer_fold), symbol='=', border=2)
    return vmap  # keys: wmfCV, wmfMetrics, {model}

def wmf_ensemble_iter2(data, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', fold=-1, outer_fold=-1, null_marker=0, verbose=1, 
        project_path='?', piggyback=True, dev_ratio=0.2, max_dev=None, resample=False, aggregation_methods=[], 
        post_hoc_analysis=False, save_data=False, enable_cluster_analysis=False):
    """
    
    Params
    ------
    fold: defined in a more generic sense than the 'fold' in wmf_ensemble_fold(). 
          specifically, 'fold' can refer to any of the following: 

           i) a CV fold ii) the index of iterations in random subsampling iii) other iteration index
    outer_fold: the iteration/fold number of the outer loop when wmf_ensemble_iter() is invoked for model selection (e.g. by model_select_core())

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
        div('(wmf_ensemble_iter) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2, 7, 8): 
            assert params['supervised']
        if params['setting'] in (3, 4, 9, 10): 
            assert not params['supervised']
        if params['setting'] in (9, 10): 
            assert params['predict_probs'], "Setting 9 - 10 should attempt to re-estimate the entire T"
    def verify_confidence_matrix(C, X, L, p_threshold, U=[], Cbar=None, measure='rank', message='', test_cases=[], plot=False, index=0):  # closure: params
        # if params['setting'] in (11, 12): 
        #     assert Cbar is not None
        #     assert params['policy_opt'].startswith('trade')
        if plot: 
            # closure: alpha, beta
            analyzer.plot_confidence_matrix(C, X, L, p_threshold, U=U, n_max=100, path=System.analysisPath, 
                measure=measure, target_label=None, alpha=params['alpha'], beta=2, index=index)
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
    def name_params_setting(method_params=['F', 'A']):   # [todo]
        # MFEnsemble.abridge(meta, identifiers=method_params, replace=replacement, verify=True)

        # use MFEnsemble.params_to_ids
        return 'F{nf}A{a}'.format(nf=params['n_factors'], a=params['alpha'])
    def identify_effective_users(Cr, L_train, fill=0): 
        n_users = Cr.shape[0]
        effective_states = np.zeros(n_users)
        for i in range(n_users):
            # non_masked_positive = (Cr[i] > fill) & (L_train == 1)
            n_non_masked = np.sum(Cr[i] > fill)
            if n_non_masked > 0: 
                effective_states[i] = 1
        return effective_states

    from evaluate import Metrics, plot_roc, analyzePerf
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    from imblearn.under_sampling import NearMiss
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    import stacking

    method = params.get('method', 'wmf2')
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
    # verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    ### Confidence Matrix
    # fold: iteration number (here, 'fold' is simply a repurposed variable borrowed from CV)
    # data: (R, T, L, Lt, U)-format
    if isinstance(data, DataFrame): # in this case, 'data' is a dataframe comprising a train-dev split 
        R, T, L_train, L_test, U =  uc.shuffle_split(data, ratio=dev_ratio, max_size=max_dev, index=outer_fold)
    else:
        # otherwise, assuming that shuffle-split operation had been invoked to produce the data input
        assert isinstance(data, (tuple, list)), "Invalid input data: %s" % data 
        print('(wmf_ensemble_iter2) Input data is an n-tuple, whrere n={n}'.format(n=len(data)))
        R, T, L_train, L_test, U = data 

    ########################################################################################
    # resampling
    msg = ''
    if resample:
        # ver = 3
        # resampling_method = 'NearMiss(v{})'.format(ver)
        # nm = NearMiss(version=ver)    # undersampling based on KNNs (version 3 is less affected by noise)

        resampling_method = 'NeighbourhoodCleaningRule'
  
        msg += '(wmf_ensemble_iter2) resampling to training data applied, method: {}\n'.format(resampling_method)
        msg += '... original dataset shape: %s\n' % collections.Counter(L_train)
        # for X, L in [(R, L_train), (Td, L_dev), ]:

        # training set
        R, L_train = uc.apply_resample(R, L_train, method=resampling_method)
        msg += '... resampled dataset shape: %s\n' % collections.Counter(L_train)

        # test set
        # no-op
    ########################################################################################
    print(msg)
    
    n_train, n_test = len(L_train), len(L_test)
    n_samples = R.shape[1]+T.shape[1]; assert n_train+n_test == n_samples
     
    ### create extra prediction vectors (PVs) from R (say, mean vector) and attach these new PVs to R (piggyback)
    n_users, n_items = R.shape
    n_classifiers = int(n_users/BagCount)
    n_users0, n_items0 = n_users, n_items  # keep a copy of the original number of users and items
    n_users_test, n_items_test = T.shape
    n_meta_users = 0

    # Preference parameters
    isPreferenceScore = params['policy_opt'].startswith('pref') # and not params['replace_subset'] 
    tPreferenceCalibration = isPreferenceScore and params['binarize_pref'] # params.get('preference_calibration', True)
    tWeightedPrediction = params.get('weighted_output', False)
    pref_threshold = pmax = params.get('pref_threshold', -1)
    pref_threshold_test = params.get('pref_threshold_test', -1)
    policy_calibration = params.get('policy_calibration', 'agreement')
    tExplicit = params.get('explicit_mf', False) if not isPreferenceScore else False

    ############################################################
    # Extra meta-estimators
    meta_users = ['latent_mean', 'latent_mean_masked',]  # todo
    if tMetaUsers: # add additional base predictors 
        div(message='(wmf_ensemble_iter2) Adding meta users (n={n})...'.format(n=len(meta_users)))
        
        # e.g. mean, and masked mean
        nU, nUT = n_users, n_users_test

        print('... augmenting T (by meta users)')
        mean_pv = make_prediction_vector(T, L=[], policy='none')   # policy: 'none' => no masking
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

    ### Estimate some statistics from training split (R)

    # estimate the labels for the test split 
    pos_label = 1
    # Eu = identify_effective_users(Cr, L_train=L_train, fill=null_marker)
    Eu = []
    print('(wmf_ensemble_iter2) Estimating probability threshold via policy={}'.format(params['policy_threshold']))
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=params['policy_threshold']) 
    Lh = lh = uc.estimateLabels(T, L=[], Eu=Eu, p_th=p_threshold, pos_label=pos_label, ratio_small_class=params['ratio_small_class']) 

    # combine (R, T) and (L_train, Lh)
    X = np.hstack((R, T))
    lh_X = L = np.hstack((L_train, Lh)) 

    McR, _ = uc.probability_filter(R, L_train, p_threshold)  # Pf: correct, Mh: estimated
    ratio_users = np.sum(McR, axis=0)/McR.shape[0]
    ratio_items = np.sum(McR, axis=1)/McR.shape[1]   # overall accuracy of each user/classifier
    n_ones, n_zeros = np.sum(McR==1), np.sum(McR==0)
    print('(wmf_ensemble_iter2) ratio<R>(item support) | min: {}, max: {}, mean: {} | vs pre-specified value: {}'.format(np.min(ratio_users), np.max(ratio_users), np.mean(ratio_users), params['ratio_users']))
    print('...                  ratio<R>(user support) | min: {}, max: {}, mean: {} | vs pre-specified value: n/a'.format(np.min(ratio_items), np.max(ratio_items), np.mean(ratio_items)))
    print('...                  n(tp,tn) vs n(fp,fn)   | n(zeros): {}, n(ones): {}, ratio: {}'.format(n_zeros, n_ones, n_zeros/(n_ones+1e-3)))
    # ... mean is the same

    ############################################################
    # compute confidence matrix for X: [R|T] # i.e. cascade mode where X contains both R and T
    CX = uc.evalConfidenceMatrix(X, L=L, U=U,  
            # test_labels=np.hstack([L_train, L_test]),  # for testing predictive performance (e.g. polarity matrix)
            # ... use L_test to pass true labels for test set

            ratio_users=params['ratio_users'],  # this can be estimated from training split (R)
            
            # Parameters for unsupervised mode (L is unknown), which is not favorable 
            # ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy='item', # params['policy'], # <<< filtering axis for training split in X
            policy_test=params['policy_test'], # filtering axis for the test split in X

            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 

            p_threshold=p_threshold, 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], 
                        alpha=params['alpha'], beta=params.get('beta', 1.0), 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, 
                            is_cascade=True, n_train=n_train,  # n_train serves as a cutoff that separates X into R and T when necessary
                            suppress_negative_examples=params['suppress_negative_examples'], 

                            # polarity matrix parameters 
                            constrained=params.get('constrained', True),
                            stochastic=params.get('stochastic', True),
                            estimate_sample_type=params.get('estimate_sample_type', True),
                            labeling_model=params.get('labeling_model', 'simple'),
                            policy_polarity=params.get('policy_polarity', 'sequence'),

                            # test polarity matrix 
                            L_test=L_test,

                            # for testing only
                            fold=fold, path=System.analysisPath) # project_path=System.projectPath 
    C0, Po, p_threshold, *CX_res = CX     # Cx_bar is removed, Cx//C0
    ############################################################
    assert C0.shape == Po.shape

    # ... C0: confidence scores, Po: polarity/correctness matrix
    uc.test_polarity(T, labels=L_test, Pref=uc.to_preference(Po[:, n_train:]), p_th=p_threshold, lh=[], name='T', title='(wmf_ensemble_iter2) -- T --')

    # ... Cui_bar is only used in policy = 'tradeoff'
    # verify_confidence_matrix(C0, X, L, p_threshold, U=U, plot=False)
    
    div("(wmf_ensemble_iter2) Completed C(R,T) | Cycle {cycle} |  dim(Cui): {dim}, filter_axis: {fdim} | conf_measure: {measure}, optimization: {opt} | predict ALL probabilities? {tval} | policy_threshold: {p_th}".format(
        cycle=(outer_fold, fold),   # fold = -1? 
            dim=str(C0.shape), fdim=params['policy'], measure=params['conf_measure'], opt=params['policy_opt'], tval=params['predict_probs'], p_th=params['policy_threshold']), symbol='#', border=1)
    print('... Model | n_factors: {nf}, alpha: {a} | predict pref? {pref}, binarize? {bin}, supervised? {s}'.format(nf=params['n_factors'], a=params['alpha'], pref=isPreferenceScore, bin=params['binarize_pref'], s=params['supervised']))
    print('... Data | n_train: {ntr}, n_test: {nt}, ratio: {r}'.format(ntr=n_train, nt=n_test, r=n_samples/(n_test+0.0)) )
    print('... Classifiers | n_users: {nu}, n_meta_users: {nmu} => n_original_users: {no}'.format(nu=n_users, nmu=n_meta_users, no=n_users-n_meta_users))
    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} | Cycle: ({fo}, {f})'.format(dim=X.shape, f=fold, fo=outer_fold))
    piggyback_msg = "+  Cycle: ({fo}, {f}) | setting: {setting}".format(fo=outer_fold, f=fold, setting=algorithm_setting).rjust(5)   # keep track of the thread from which the iteration 
    
    positive_pref, negative_pref = 1.0, 0.0
    ########################################################################################
    # ... determining Cn (n: neutral)
    # ... 1. masking negative (wrong predictions)? Probably not. Want latent factors to produce new estimates further away 
    #        from the original estimate => keep their negative weights
    # ... 2. masking neutral (undetermined)? Yes, keep entries with high uncertainty out of the opt objective

    Cn = uc.make_cn(C0, Po, is_unweighted=False, weight_neutral=0.0, sparsify=True)
    Cn = uc.balance_and_scale(Cn, X=X, L=L, Po=Po, p_threshold=p_threshold, U=U, 
                        alpha=params['alpha'], beta=params.get('beta', 1.0), 
                            conf_measure=params['conf_measure'], 
                                suppress_max_class=params['suppress_negative_examples'], 
                                discount_test=params.get('discount_test', False), 
                                    n_train=n_train, is_cascade=True, is_test_split=False, fold=fold)
    assert type(Cn) == type(Po), "dtype(Cn): {} <> dtype(Po): {}".format(type(Cn), type(Po))
    # assert isPreferenceScore or (n_neutral_cn > n_neutral_cx), \
    #     "Masked entries of C when approximating 'ratings' must be more than those of C when approximating preference | n_masked(Cn): {}, n_masked(Cn): {}".format(n_neutral_cn, n_neutral_cx)
    # ... Cw: is a masked version of C0, where both neutral and negative poloarity examples have zero weights

    P, Q, *Xh_errs = ua.implicit_als(Cn, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            # label_confidence=Cx_bar, 
                            polarity=Po,   # color matrix
                            p_threshold=p_threshold,
                            positive_pref=positive_pref, 
                            negative_pref=negative_pref, 

                            ratings=X, labels=L,
                            policy=params['policy_opt'], message=piggyback_msg, ret_rmse=True)
    Xh_err, Xh_err_weighted = Xh_errs
    ########################################################################################
    ne = 2
    e_pri, e_post = np.mean(Xh_err[:ne]), np.mean(Xh_err[-ne:])
    e_del = (e_pri-e_post)/e_pri * 100
    print('... (ALS 1) Complete | rmse: {e1} -> {e2} (n_errs: {n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(e1=e_pri, e2=e_post, 
                   e_del=e_del, ew1=np.mean(Xh_err_weighted[:ne]), ew2=np.mean(Xh_err_weighted[-ne:]), n=len(Xh_err) ))

    # e.g. 50 users, 3979 items with nf=100 > dim(P): (50, 100), dim(Q): (3979, 100)
    assert (P.shape[0] == R.shape[0]) and (P.shape[1] == params['n_factors'])
    assert (Q.shape[0] == R.shape[1]+T.shape[1]) and (Q.shape[1] == params['n_factors'])

    # compute reconstructed training data (so that later on we can test its utility for stacking)
    # Pr, Qr => Rh, use Rh in place of R whenever Cr == fill
    Xh = reconstruct(Cn, X, P, Q, 
                Pc=Po, 
                L=L, 
                test_labels=np.hstack([L_train, L_test]), # test performance only
                    p_threshold=p_threshold,   
                    use_confidence_weights=False,
                    policy_opt=params['policy_opt'], policy_replace=params['policy_replace'], 
                       n_train=n_train, is_cascade=True,
                           replace_subset=params['replace_subset'], params=params, null_marker=null_marker, binarize=False, name='R+T', index=outer_fold)

    ### ALS evaluation (RMS)
    delta_X = LA.norm(Xh-X, 'fro')
    Rh, Th = Xh[:,:n_train], Xh[:,n_train:]  
    # ... # Xh is a continuous estimate of binary preference matrix

    Qr, Qt = Q[:n_train, :], Q[n_train:, :]  # row(Q) ~ items/data
    assert Rh.shape == R.shape, "dim(R): {}, dim(Rh): {}".format(R.shape, Rh.shape)
    assert Th.shape == T.shape, "dim(T): {}, dim(Th): {}".format(T.shape, Th.shape)

    # pref_threshold_test = pref_threshold = 0.5
    if tPreferenceCalibration: 
        # operations: normalize -> calibrate -> binarize
        #   1. normalize preference scores in Xh to [0, 1] range
        #   2. search for 'optimal' pref threshold 
        #   3. binarize X based on the threshold determined by step 2

        # pref_threshold is given (e.g. estimated from training split R)
        print('(wmf_ensemble_iter2) pref_threshold given? {}'.format(True if pref_threshold > 0.0 else False))

        lhx = L  # L is a concatenation of training set labels and estimated test set labels 
        MhX, Lhx = uc.probability_filter(X, lhx, p_threshold)  # Lh_X(X, p_threshould)

        if pref_threshold > 0.0: 
            Xh = uc.binarize_pref(Xh, p_th=pref_threshold, cutoff=True)
        else: 
            # -- A. calibrate together
            
            # ... MhX is an estimated corrected matrix (because the test-split labels are estimated)
            # ... Xh: continous preference matrix
            Xh, pref_threshold, score = uc.calibrate_preference(Xh, Pf=MhX, Lh=Lhx, step=0.01, 
                    policy=policy_calibration, message='train-test split (X: R+T)') 
            # ... Xh: binary preference matrix

        pref_threshold_ref = pref_threshold_test = pref_threshold
        
        McR, MhT = MhX[:,:n_train], MhX[:,n_train:]
        Rh, Th = Xh[:,:n_train], Xh[:,n_train:]
        # ... Xh is a binary preference matrix

        # -- B. calibrate separately 
        lr, lht = L[:n_train], L[n_train:]  # lht is estimated via heuristics (e.g. majority vote)
        
        McR, Lr = uc.probability_filter(R, lr, p_threshold)  # Pf: correct, Mh: estimated
        # Rh, pref_threshold, score = uc.calibrate_preference(Rh, Pf=McR, Lh=Lr, step=0.01, 
        #     policy=policy_calibration, message='train split (R)')

        MhT, Lht = uc.probability_filter(T, lht, p_threshold)  # Mh: estimated Pf
        # Th, pref_threshold_test, score = uc.calibrate_preference(Th, Pf=MhT, Lh=Lht, step=0.01, 
        #     policy=policy_calibration, message='test split (T)')
        # ... now (Rh, Th) are binary matrices

        # >>> the evaluation should be wrt true labels 
        # ... use L_test to get correctness matrix
        # Pf, _ = uc.probability_filter(X, np.hstack((L_train, L_test)), p_threshold)  # Lh(T, p_threshold)
        McT, Lt = uc.probability_filter(T, L_test, p_threshold)  # Lh(T, p_threshold)

        # how aligned is MhT with McT? 
        n_aligned = np.sum(MhT == McT)
        n_mis_aligned = np.sum(MhT != McT)
        label_est_method = 'majority_vote'
        print("(wmf_ensemble_iter2) Quality of alignment (MhT|{} vs McT) | n_aligned: {}, n_mis_aligned: {}, ratio: {}".format(label_est_method, 
            n_aligned, n_mis_aligned, n_aligned/(n_mis_aligned+n_aligned+0.0)))

        # print('(wmf_ensemble_iter2) Quality of seed on (Xh) | th(Xh): {} => policy_calibration: {}    ... cycle: {}'.format( pref_threshold, policy_calibration, (outer_fold, fold) ))
        # p_tp_preferred, p_fp_preferred, p_tn_preferred, p_fn_preferred, p_agreed, p_correct_agreed = \
        #     uc.ratio_of_alignment2(Xh, Pf, Lh_X, verbose=True)  

        print('(wmf_ensemble_iter2) Quality of seed on (Rh) | th(Rh): {} ~? th(Xh): {} | policy_calibration: {}    ... cycle: {}'.format( pref_threshold, pref_threshold_ref, 
            policy_calibration, (outer_fold, fold) ))
        ret = uc.ratio_of_alignment2(Rh, McR, Lr, verbose=True)  

        print('(wmf_ensemble_iter2) Quality of seed on (Th) | th(Th): {} ~? th(Xh): {} | policy_calibration: {}    ... cycle: {}'.format( pref_threshold_test, pref_threshold_ref, 
            policy_calibration, (outer_fold, fold) ))
        ret = uc.ratio_of_alignment2(Th, McT, Lt, verbose=True)  
        # ... always use Mc_T (not Mh_T) and Lt, i.e. gold standard, to evaluate

        # ... Rh, Th are binary matrices
        assert len(np.unique(Th)) == 2

    elif isPreferenceScore: 
        # still need to normalize 
        Rh = canonicalize_pref(Rh, binarize=False, name='Rh', verify=1, min_score=negative_pref, max_score=positive_pref)
        Th = canonicalize_pref(Th, binarize=False, name='Th', verify=1, min_score=negative_pref, max_score=positive_pref)
        
        # ... Rh, Th are normalized (to [0, 1])
        n_uniq_test = len(np.unique(Th))
        assert np.max(Th) <= 1.0 and np.min(Th) >= 0.0
        assert n_uniq_test > 2

    ### create extra prediction vectors (PVs) from T (say, mean vector) and attach these new PVs to T (piggyback)

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    ################################## 
    # P = P.todense()
    # Q = Q.todense()
    
    div("(wmf_ensemble_iter2) Completed rating matrix reconstruction | Cycle: ({fo}, {fi}) | preference scores? {tval}, action='{act}'".format(
        fo=outer_fold, fi=fold, tval=isPreferenceScore, act='Replace Subset' if params['replace_subset'] else 'Replace All')) # predict => predict probabilities
    if isPreferenceScore: 
        print('... binarize preference matrix? {}'.format(params['binarize_pref']))

    # classifier/user weights (usually just use confidence matrix)
    Cw = Cwt = None
    if tWeightedPrediction: 
        Cw, Cwt = Cn[:,:n_train], Cn[:,n_train:]
        print('(wmf_ensemble_iter2) using weighetd preference scores (via confidence matrix) | dim(Cwt): {}'.format(Cwt.shape))
    else: 
        print('(wmf_ensemble_iter2) using non-weighted preference scores ...')

    # if tPreferenceCalibration: 
    #     # [test]
    #     ##############################################
    #     # use L_test to get correctness matrix
    #     Pf, Lh_T = uc.probability_filter(T, L_test, p_threshold)  # Lh(T, p_threshold)
    #     # ... only used for evaluation
        
    #     # how does preference matrix Th match with true correctness matrhx Pf and true labels Lh_T? Lh_T is needed because we want to only focus on a target label (e.g. 1 or positive)

    #     print('(wmf_ensemble_iter2) Quality of seed on (Th) | th(R+T): {} | policy_calibration: {}    ... cycle: {}'.format(pref_threshold, policy_calibration, (outer_fold, fold) ))
    #     ret = uc.ratio_of_alignment2(Th, Pf, Lh_T, binarize=False, verbose=True)  # target, overall, correct only 
    #     # ... Th had been binarized via calibrate_preference(Xh)

    # X: (R, T, L_train, L_test, U)
    n_samples_reconstructed = Rh.shape[1]+Th.shape[1]
    ##############################################################################################################
    # ... prediction 
    if not aggregation_methods: aggregation_methods = System.aggregation_methods # e.g. ['mean', 'median', 'log', ]  
    # pv_mean = uc.combiner(Th, aggregate_func='mean')           

    ##############################################################################################################
    # ... output
    file_types = ['prior', 'posterior', ]
    if post_hoc_analysis:   # only triggered after model selection cycle is complete and 'best params' has been determined
        
        # note that we shall save the data only after model selection loop is completed
        assert outer_fold >= 0 and fold == -1, "Intended action: only save the final model after model selection is complete (outer fold: {fo}, inner fold: {fi}".format(fo=outer_fold, fi=fold)
        the_params = name_params_setting(method_params=['F', 'A'])

        div('(wmf_ensemble_iter2) Running posthoc analysis | algorithmic setting: {s}'.format(s=algorithm_setting), symbol='%') 
        # MFEnsemble.save_meta_tset((Rh, Th, L_train, L_test, U), fold=fold, method=method, params=params, verbose=verbose)

        pv_t = pv_th = None
        if tMetaUsers: 
            # drop meta users 
            R = R[:-n_meta_users]
            Rh = Rh[:-n_meta_users]
            T, pv_t = T[:-n_meta_users], T[-n_meta_users:] 
            Th, pv_th = Th[:-n_meta_users], Th[-n_meta_users:]
            assert Th.shape[0] + pv_th.shape[0] == n_users_test
            assert pv_t.shape[1] == pv_th.shape[1] == len(L_test)
            assert Th.shape == T.shape, "dim(T): {}, dim(Th): {}".format(T.shape, Th.shape)

            P = P[:-n_meta_users]
            # Qr, Qt <- Q

            if Cwt is not None: 
                Cwt = Cwt[:-n_meta_users]
                assert Th.shape == Cwt.shape, "dim(Th): {}, dim(Cwt): {}".format(Th.shape, Cwt.shape)

        ### save predictions
        isBinaryPrefMatrix = False
        if isPreferenceScore: 
            n_uniq, n_uniq_test = len(np.unique(Rh)), len(np.unique(Th))
            if tPreferenceCalibration: # then Xh must be a binary matrix
                assert n_uniq == 2 and n_uniq_test == 2, "Th is not binary or is degenerated | n_uniq: {}, n_uniq(T): {} | values(T): {}".format(n_uniq, n_uniq_test, np.unique(Th))
            # else: 
            #     print("(wmf_ensemble_iter2) No preference calibration was applied to Th => remains to be a continuous matrix with entries in [0, 1]")
            #     # Th is a continuous rating matrix
            #     assert n_uniq > 2 and n_uniq_test > 2, "Unique values should be >> 1 | n_uniq(R): {}, u_uniq(T): {}".format(n_uniq, n_uniq_test)
            if n_uniq_test == 2: isBinaryPrefMatrix = True
            print('(wmf_ensemble_iter2) Prior to making predictions | Th is a {}  ... (verify)'.format(
                'binary matrix' if isBinaryPrefMatrix else 'rating matrix'))

        # the_params = name_params_setting(method_params=['F', 'A'])
        dataset = {'prior': [R, T], 'posterior': [Rh, Th]}
        comparisons = {}  # scores
        for file_type in file_types: 
            comparisons[file_type] = {}
            X_train, X_test = dataset[file_type]

            for i, aggr in enumerate(aggregation_methods):    # defined earlier e.g. ['mean', 'median']
                pn = '{ftype}_{model}'.format(ftype=file_type, model=aggr)  # pn: predictor name
                
                # TODO: if aggr in ['logistic', ...]  # need to further train on X_train and test on X_test
                if aggr in ['mean', 'median', ]:   # System.simple_aggregation
                    if isPreferenceScore: 
                        if file_type.startswith('pri'):  # prior is not associated with a preference matrix
                            pv = uc.combiner(T, aggregate_func=aggr)
                        else: 
                            # add confidence weights? Cwt: None if not
                            # note: Th should have been in the right format, NO need to set canonicalize to True
                            # ... with Cwt, Th' <- Th * Cwt
                            pv = uc.predict_by_preference(T, Th, W=Cwt, name='Th', aggregate_func=aggr, 
                                        fallback_on_low_weight=False, verify=True) # verify/False: used to verify e.g. existence of NaNs
                    else: 
                        print("(wmf_ensemble_iter2) making final prediction {} weights (Cwt)".format('with' if Cwt is not None else 'without'))

                        # 1. reconstructing by replacement
                        pv = uc.combiner(X_test, weights=Cwt, aggregate_func=aggr)  # T vs Th
                else:
                    ### put stacker code here! 
                    stacker = stacking.choose_classifier(aggr)  # e.g. log, enet, knn

                    if isPreferenceScore: 
                        # 1. [note] consider entries of low or zero preference scores as "dropouts" 
                        # 2. preference becomes new features in stacker models

                        if file_type.startswith('post'):
                            # cascade mode: R and T share the same pref_threshold
                            # Rb = uc.binarize_pref(Rh, p_th=pref_threshold, cutoff=True) # uc.canonicalize_pref(Rh, name='Rb', binarize=True, verify=2)
                            # Tb = uc.binarize_pref(Th, p_th=pref_threshold, cutoff=True) # uc.canonicalize_pref(Th, name='Tb', binarize=True, verify=2)
                            X_train = np.vstack((R, Rh))   
                            X_test = np.vstack((T, Th))    
                        
                        model = stacker.fit(X_train.T, L_train) # remember to take transpose
                        pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data    
                    else: 
                        model = stacker.fit(X_train.T, L_train) # remember to take transpose
                        pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data
                    
                y_pred, y_label = pv, L_test
                vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

                # [test]
                ############################################################################################################
                comparisons[file_type][aggr] = common.fmax_score(y_label, y_pred, beta = 1.0, pos_label = 1)
                # 'ideal' baseline: if preference matrix is perfect, what happens? 
                if file_type.startswith('post') and isPreferenceScore and tPreferenceCalibration: 
                    if aggr == 'mean': 
                        # ideal cases
                        Pf, Lh_T = uc.probability_filter(T, L_test, p_threshold)  # Lh(T, p_threshold)
                        pv_perfect = uc.predict_by_preference(T, Pf, W=None, name='Pf', aggregate_func=aggr, fallback_on_low_weight=False) 
                        comparisons[file_type]['perfect_{}'.format(aggr)] = common.fmax_score(y_label, pv_perfect, beta = 1.0, pos_label = 1)

                        # Cwt may not have been set
                        W = Cn[:,n_train:]
                        # if scipy.sparse.issparse(W): W = W.toarray()
                        pv_perfect_weighted = uc.predict_by_preference(T, Pf, W=W, name='Pf', aggregate_func=aggr, 
                            fallback_on_low_weight=False) 
                        comparisons[file_type]['perfect_weighed_{}'.format(aggr)] = common.fmax_score(y_label, pv_perfect_weighted, beta = 1.0, pos_label = 1)
            
                        # hypothetical cases
                        # 1. weighted preference scores
                        pv_weighted = uc.predict_by_preference(T, Th, W=W, name='Th', aggregate_func=aggr, 
                                        fallback_on_low_weight=False, verify=True)
                        comparisons[file_type]['weighted_{}'.format(aggr)] = common.fmax_score(y_label, pv_weighted, beta = 1.0, pos_label = 1)

                        # 2. all is preferred
                        pv_all_yes = uc.predict_by_preference(T, np.ones(Th.shape), W=None, name='Ones', aggregate_func=aggr, 
                                        fallback_on_low_weight=False, verify=True) 
                        comparisons[file_type]['prefer_all_{}'.format(aggr)] = common.fmax_score(y_label, pv_all_yes, beta = 1.0, pos_label = 1)
                ############################################################################################################

            ### ... end foreach aggregation method
        
        # [test]
        # div('(wmf_ensemble_iter2) Ideal performance | Cycle: {} | file type: {}, ID: {}'.format( (outer_fold, fold), file_type, MFEnsemble.get_dset_id(method=method, params=params)), symbol='#')
        msg = "(wmf_ensemble_iter2) Comparing with 'ideal' cases | Cycle: {} | file type: {}, ID: {}\n".format( (outer_fold, fold), file_type, MFEnsemble.get_dset_id(method=method, params=params))
        for file_type in file_types: 
            msg += '--- ({}) ---\n'.format(file_type)
            for i, (aggr, score) in enumerate(comparisons[file_type].items()): 
                msg += '... [{}] method: {} => score: {}\n'.format(i, aggr, comparisons[file_type][aggr] )
        div(msg, symbol='#')
                
        ### save predictions from the meta users (and treat these data as if they came from an aggregation method)
        if tMetaUsers: 
            # vmap['prior_mean'], vmap['posterior_mean'] = {}, {}
            predictions = {'prior': pv_t, 'posterior': pv_th}  # each prediction dataframe consists of prediction vectors from n meta_users
            for file_type in file_types: 
                for i, meta_user in enumerate(meta_users): 
                    pn = '{ftype}_{model}'.format(ftype=file_type, model=meta_user) 

                    if isPreferenceScore:
                        if file_type.startswith('pri'):
                            pv = pv_t
                        else:      
                            pv = pv_t * pv_th  # use pref scores to 'up-regulate' or 'down-regulate' the predictions
                    else: 
                        pv = predictions[file_type]

                    y_pred, y_label = pv[i], L_test
                    vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

            # for file_type, pv in predictions.items(): 
            #     pn = '{ftype}_{model}'.format(ftype=file_type, model='mean') # pn: predictor name
            #     vmap[pn] = {}
            #     for i, meta_user in enumerate(meta_users): # ['latent_mean', 'latent_mean_masked',]
            #         y_pred, y_label = pv[i], L_test
            #         vmap[pn][meta_user] = DataFrame({'prediction':y_pred,'label':y_label, 'method':meta_user, 'fold': fold}, index=range(len(y_pred)))
    
            print('... saved predictions for data type: {dtype} & meta users (n={n}) to dataframes'.format(dtype=file_type, n=len(meta_users)))
         
        #### optionally, run cluster analysis
        filter_axes = ['user', ]  # 'item'
        if enable_cluster_analysis: 
            # if meta users were included, consider R[:-n_meta_users] 
            for fdim in filter_axes: 
                
                # training split; if meta users were included, consider R[:-n_meta_users] 
                run_cluster_analysis(P, U=U, X=X, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=True, save_plot=True, file_type='train') # usually only at the training of the final model (i.e. after the best hyperparameters are determined; see wmf_ensemble_model_select())

        dset_id = MFEnsemble.get_dset_id(method=method, params=params)

        # return the dataset ID
        #    this is necessary because the 'best params' in each cycle may not be the same; for example, we may end up getting training data like this: 
        #       wmf_F100_A100_XCFuser_S2-train-prior-1.csv.gz
        #           ... 
        #       wmf_F75_A100_XCFuser_S2-train-prior-4.csv.gz
        #    => in cycle {0, 2, 3, 4}, F75_A100 was the best, but in cycle {1, }, F100_A100 was the best
        vmap['dset_id'] = dset_id 
        vmap['best_params_inner'] = the_params

    ##############################################################################################################
    if save_data: 
        div('(wmf_ensemble_iter2) Output: saving transformed training and test sets (size: n(R): {nR}, n(T): {nT}), total size: {N}| delta(R): {dR}, delta(T): {dT} | algorithmic setting: {s}'.format(s=algorithm_setting, 
                    nR=Rh.shape[1], nT=Th.shape[1], N=n_samples_reconstructed, dR=delta_R, dT=delta_T), symbol='>')

        # test set
        MFEnsemble.save_data((T, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data 
        MFEnsemble.save_data((Th, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data  (~ prediciton)

        # training set
        MFEnsemble.save_data((R, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-prior', verbose=verbose, subsampling=True)
        MFEnsemble.save_data((Rh, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-posterior', verbose=verbose, subsampling=True)
    

    ### kind: specific MF algorithm
    if not kind: kind = 'als'
    wmfMetrics, wmfCV = [], [] # vmap['wmfMetrics'], vmap['wmfCV']

    # naming convention: method_id = '{prefix}_{id}_{learner_type}' # e.g. rf_bp_stacker
    aggregate_mode = params.get('policy_ms_model', 'mean')
    aggregate_func = params.get('policy_aggregate_func', 'mean')
    div("(wmf_ensemble_iter2) Comparison of model parameters | aggregate_func: {func}, mode: {mode}".format(func=aggregate_func, mode=aggregate_mode)) 
    # {stacker}.S-{dataset}-{suffix}
    # method_id = '{prefix}_{id}_{learner_type}'.format(prefix='wmf_%s' % kind, id=MFEnsemble.get_method_id(method, kind, params=params), learner_type=aggregate_func)
    method_id = "{prefix}.W-{id}-{suffix}".format(prefix=aggregate_func, id=MFEnsemble.get_method_id(method, kind, params=params), suffix=kind)
    ### model evaluation
    #   a. use mean proba in T vs labels as a summary score
    #   b. but we can compute a performance score for each base method (and then take the average)

    # header = MFEnsemble.header_model_evaluation # ['method', 'score', 'posterior_score', 'mean_score', ] ... used in evaluate.analyzePerf2()

    ############################################################################################################################
    # PerformanceMetrics object in wmfMetrics is the basis for selecting the best hyperparameter
    # 
    assert not tWeightedPrediction or (Cwt is not None)
    performance_metrics, pv = analyzePerf(L_test, Th, 
                                    method=method_id, aggregate_func=aggregate_func, 
                                        weights=Cwt,
                                        outer_fold=outer_fold, fold=fold,  # keep track of the iteration (debugging only when comparing Th and T)
                                        train_data=(Rh, L_train),  # only used in stacking mode
                                        mode=aggregate_mode)  # pass T=T to compare with T
    wmfMetrics.append( performance_metrics ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
    wmfCV.append( (L_test, pv) )   
    vmap['wmfMetrics'], vmap['wmfCV'] = wmfMetrics, wmfCV

    ##########################################################
    # other variables that may be of interest 
    # >>> piggyback inputs
    if piggyback: 
        # args, posargs = us.arguments()
        vmap['hyperparams'] = hyperparams  # keep track of hyperparameters for convenience 
    # vmap['fold'] = fold
    ##########################################################

    print("(wmf_ensemble_iter2) Ending cycle: ({fo}, {fi}) at setting {case} > returning vmaps: {keys} ... (verify) ".format(fo=outer_fold, fi=fold, 
        case=algorithm_setting, keys=vmap.keys()))

    # keys of vmap are the variables to return to caller: 
    #   i) saved in every cycle: wmfMetrics, wmfCV, hyperparams
    #  ii) saved only when training the final model after n cycles of model selection is completed: 
    #      dset_id, 
    #      best_params_nruns 
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
                
                print("(merge) variables: {k}, example: {v}... (verify) ".format(k=Vars, v=fv))  # ['wmfCV', 'wmfMetrics', 'models', 'rank']
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
        # [test] 
        # for var in Vars: 
        #     # for kind in kinds: 
        #     print("(verify) size of var '{0}': {1}".format(var, len(vmap[var])) )
        #     print("...     after Parallel() call, wmfCV has {0} sets/cvfold, wmfMetrics has {1} sets.  #".format(len(vmap[var]), len(vmap[var])))
        return vmap
    def initmap(vars=['wmfMetrics', 'wmfCV', ]): 
        vmap = {} 
        for var in vars:  # [note] vars() in python takes an object (with __dict__ attribute) and returns a dictionary
            vmap[var] = [] # {k: [] for k in kinds}   # var -> kind -> a list
        return vmap
    def verify(vmap):
        div('(wmf_ensemble) verify vmap (vars: {alist})'.format(alist=list(vmap.keys()) ), symbol='%')
        
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
      
        # columns: alpha|freq|n_factors|score
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
            
    import operator
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
    params['training_mode'] = kargs.get('training_mode', 'regular') # used in model_select_core()

    tMetaUsers = params['include_meta_users'] = kargs.get('include_meta_users', False)  # if True, wmf_ensemble routine will add meta classifiers in R and T

    # note: {n_factors, alpha} can be affected by command args, use System's default instead of MFEnsemble's default
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)  
    params['alpha'] = alpha_val = kargs.get('alpha', MFEnsemble.alpha)
    params['beta'] = beta = kargs.get('beta', 1.0)
    params['pref_threshold'] = kargs.get('pref_threshold', -1)

    # >>> not the same as 'n_runs'
    params['n_iter'] = params['n_epochs'] = kargs.get('n_epochs', System.n_epochs)
    params['n_iter_foldin'] = params['n_epochs_foldin'] = kargs.get('n_epochs_foldin', System.n_epochs_foldin)
    
    # parameters for confidence matrix
    params['conf_measure'] = kargs.get('conf_measure', 'brier')  # confidence matrix

    params['ratio_small_class'] = kargs.get('ratio_small_class', -1)  # <=0: use minority class ratio as default
    params['ratio_users'] = kargs.get('ratio_users', 0.5)  # select top k entries of R (through Cui) to approximate, only relevant when policy in {'user', 'item'}
    params['suppress_negative_examples'] = kargs.get('suppress_negative_examples', False)

    params['supervised'] = kargs.get('supervised', True)
    params['augmented'] = kargs.get('augmented', True)  # approximate R or both R & T? 
    
    unbag = params['unbag'] = kargs.get('unbag', False)
    params['resume_als'] = kargs.get('resume_als', False) # in ALS fold-in, use the learned factor vector as an init. or fix it so that ALS reduces to LS?
    #################################################################
    # Do we re-estimate the entire rating table or only the entries marked unreliable? (i.e. replace <- True)
    
    params['replace_subset'] = kargs.get('replace_subset', True)  # replace bad ratings or other scores (e.g. probabilities)
    params['replace_all'] = params['predict_probs'] = not params['replace_subset']
    #################################################################

    params['masked'] = kargs.get('masked', True) # if True, mask FP and FN, this makes Cui 'sparse'; if False, turn masking off 
    params['mask_all_test'] = kargs.get('mask_all_test', False) # mask all entries in T or not? only relevant when 'augmented' is True

    # params['p_threshold'] = p_th  # 'ratio'-based confidence matrix depends on threshould
    # params['delta'] = 0

    # parameters for ALS methods 
    params['policy_filter'] = params['policy'] = kargs.get('policy', 'item')  # II: {'item', 'user', 'polarity'}
    params['policy_filter_test'] = params['policy_test'] = kargs.get('policy_test', params['policy'])
    
    #################################################################
    # polarity matrix parameters
    params['labeling_model'] = kargs.get('labeling_model', 'simple')  # used to determine polarity matrix; only relevant when policy_filter (or policy_filter_test) is 'polarity'
    params['constrained'] = kargs.get('constrained', True)
    params['stochastic'] = kargs.get('stochastic', True)
    params['policy_polarity'] = kargs.get('policy_polarity', 'sequence')
    params['estimate_sample_type'] = kargs.get('estimate_sample_type', True)
    #################################################################
    
    tResample = params['balance_class_resampling'] = kargs.get('balance_class_resampling', False)

    params['policy_opt'] = kargs.get('policy_opt', 'rating') # options: I {'rating', 'preference', 'tradeoff'}, 
    params['policy_opt_T'] = kargs.get('policy_opt_T', 'foldin')  # how the factors in test set are derived {'foldin', 'seeding', 'transfer', 'transfer+seed'}
    params['explicit_mf'] = kargs.get('explicit_mf', False)
    params['approx_ratings_via_pref'] = kargs.get('approx_ratings_via_pref', False)

    params['policy_calibration'] = kargs.get('policy_calibration', 'agreement')
    params['two_way_calibration'] = kargs.get('two_way_calibration', False)

    params['binarize_pref'] = kargs.get('binarize_pref', True)
    params['preference_calibration'] = kargs.get('preference_calibration', True)

    params['policy_replace'] = kargs.get('policy_replace', 'rating') # used only when policy_opt <- preference AND replace <- True
    params['policy_threshold'] = kargs.get('policy_threshold', 'prior')  # how to determine prob threshold? {'fmax', 'prior'/'topk', }
    
    policy_iter = kargs.pop('policy_iter', 'cv') # policy for train-dev-test iterations (nested CV, CV, randon subsampling)
    policy_ms = params['policy_ms'] = kargs.get('policy_ms', 'freq')   # the policy for determining the best model; tricky upon large variance, empiricially 'freq' may work better
    policy_ms_model = params['policy_ms_model'] = kargs.get('policy_ms_model', 'mean')
    params['policy_aggregate_func'] = kargs.get('policy_aggregate_func', 'mean')
    # aggregation_methods = System.aggregation_methods

    tGlobalMS = kargs.get('ms_global', False)
    
    params['dev_ratio'] = kargs.get('dev_ratio', 1./System.foldCount)
    param_grid = kargs.pop('param_grid', {'n_factors': [5, 10, 20, 50, 100, 500], 'alpha': [1, 10, 100, 1000]})
    
    n_runs = System.foldCount # kargs.pop('n_runs', 10)  # number of runs of random subsampling, only relevant when 'policy_iter' is 'subsampling'
    # ... n_runs is no longer an option but fully depends on the base-level fold count with shuffle_split_cv()   ... 08.02.19

    n_runs_global = n_runs # kargs.pop('n_runs_global', System.foldCount)
    n_runs_modelselect = kargs.pop('n_runs_modelselect', 10)
    # ... can still run an arbitrary number of iterations in the inner loop with shuffle_split_cv() ... 08.02.19
    
    max_dev = kargs.get('max_dev', 5000) # by default use all train-dev split to do model selection but sometimes, we may want to control the sample size to save time
    
    # predicates
    tPlotROC = kargs.get('plot_roc', True)
    tModelSelection = len(param_grid) > 0 and sum(1 for v in param_grid.values() if len(v) > 1) > 0
    tSaveBestModel = kargs.get('post_hoc_analysis', True)  # run posthoc analysis
    tClusterAnalysis = kargs.get('enable_cluster_analysis', True) # only enable cluster analysis after 'best params' have been determined
    tSaveData = kargs.get('save_data', False) # save transformed data
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
    
    ### consider model selection or not? 
    div("(wmf_ensemble) Running model selection (n_runs:{n} * n_runs_modelselect:{nm} = n_total_run:{nt}) with {policy} on {grid} ... (verify)".format(n=n_runs, 
            nm=n_runs_modelselect, nt=n_runs * n_runs_modelselect, policy=policy_iter, grid=param_grid), symbol='#')  # ... ok 
    
    # [note] 1. vmaps is a collection of the outputs from multiple threads => vmaps is a list
    #        2. use threads: prefer="threads"
    # suppose that n_runs = 10, n_runs_modelselect = 5
    # wmf_ensemble_model_select() is to run 10 times, each of which runs model-selection routine 5 times
    vmaps = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_ensemble_model_select)(params, param_grid, n_trials=n_runs_modelselect,
                        vars=Vars, kind=kind, null_marker=null_marker, 
                        dev_ratio=0.2, test_ratio=0.3, 
                        # policy_ms=policy_ms,   # this is included in 'params'; policy for model selection (e.g. frequency, average, )
                            fold=nr, max_dev=max_dev, resample=tResample, 
                                post_hoc_analysis=tSaveBestModel, 
                                enable_cluster_analysis=tClusterAnalysis, 
                                    unbag=unbag) 
                                        for nr in range(n_runs))
    #     vmaps['wmfMetrics'] = [perf1, perf2, ... ]
    # ... each wmf_ensemble_model_select() call should only produce a single 'best' model => N calls, N models

    #######################################################
    # ... model selection complete
    ####################################################################################################################################

    # combine parallelized results 
    vmap = merge(vmaps)  # variables: wmfCV, wmfMetrics, hyperparams, fold, dset_id
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
        
        best_params_nruns = best_params = {} 
        if tModelSelection:  # True only if param_grid has effectively multiple parameters to choose from
            # models = M.records     

            # <result> hyperparameters vs scores (e.g. fmax)
            #################
            div("(wmf_ensemble) Found {n} sets of 'best params' with {ns} scores ... (best parameters) #".format(n=M.size(), ns=M.size_bags()))
            M.display(by='raw')  # display multibags
            print('-' * 100)
            #################

            # n_runs: number of iterations involving model selection process (which itself comprises n cycles, where n = n_runs_modelselect)
            assert M.size_bags() == n_runs, "n_runs: %d whereas number of models returned from model selection: %d" % (n_runs, M.size_bags())

            # [test]
            # sort in terms of frequencies
            freq_models = M.sort_by_freq()
            print( us.format_sort_dict(dict(freq_models), reverse=True, padding=5, title="(wmf_ensemble) Sorted 'best parameters' by frequency | metric={metric} ... (result)".format(metric='fmax')) ) # symbol='#', border=1

            mean_models = M.aggregate()
            print( us.format_sort_dict(mean_models, reverse=True, padding=5, title="(wmf_ensemble) Average performance of 'best parameters' | metric={metric} ... (result)".format(metric='fmax')) ) # symbol='#', border=1

            # 'global' frequency across multiple runs
            if policy_ms.startswith('freq'):  # find the overall popularity of the hyperparams across multiple runs
                modelCount = {}
                for cycle, model in enumerate(vmap['rank']):  # best_params, best_score = best_models[0], best_models[1]  # best_params: a frozen dictionary
                    hyparams, count = model
                    print("... cycle: {index} | hyperparams: {p} => count: {c}".format(index=cycle, p=hyparams, c=count)) 
                    if not hyparams in modelCount: modelCount[hyparams] = 0
                    modelCount[hyparams] += count
                # global_freq_models = sorted(modelCount, key=modelCount.__getitem__, reverse=True)[0]
                global_freq_models = sorted(modelCount.items(), key=operator.itemgetter(1), reverse=True)
                best_params_nruns, best_score_nruns = global_freq_models[0][0], global_freq_models[0][1]   # [0] to get the best; best[0]: hyparams, best[1]: score

                print( us.format_sort_dict(dict(global_freq_models), reverse=True, padding=5, title="(result) 'best parameters' by GLOBAL frequency | metric={metric} ...".format(metric='fmax')) )
                div('(wmf_ensemble) Globally best params (n_runs={nruns}): {params} | frequency: {freq}'.format(nruns=n_runs, params=best_params_nruns, freq=best_score_nruns), symbol='#', border=2)
                # print('... consistent with max of the max? | best (max(max)): {p0} =?= best_nruns: {p1}'.format(p0=ret['best_params'], p1=dict(best_params_nruns)))

                if tGlobalMS:  # global (best params from n_runs * n_runs_modelselect)
                    param_grid_post = {name: [value, ] for name, value in dict(best_params_nruns).items()}
                    div(message='Re-compute the model using globally best params: {params}'.format(params=param_grid_post), symbol='#', border=2)
                    vmaps = Parallel(n_jobs = -1, verbose = 1)(delayed(wmf_ensemble_model_select)(params, param_grid_post, n_trials=1,
                                vars=Vars, kind=kind, null_marker=null_marker, 
                                dev_ratio=0.2, test_ratio=0.3, 
                                    policy_ms=policy_ms,   # policy for model selection (e.g. frequency, average, )
                                    fold=nr, max_dev=max_dev, save=tSaveBestModel, unbag=unbag) for nr in range(n_runs_global))      
        #######################################################
        
        M = Metrics()
        for entry in vmap['models']:
            bestp, score = entry['best_params'], entry['score'] 
            M.add( (us.frozendict(bestp), score) )
        ret['best_params'] = best_params = dict(M.sort(by=policy_ms)[0][0])  # 'best params' across n_runs (outer loop)
        # ... best parameters after n_runs, each of which runs model selection routine multiple times (n_runs_modelselect)

        ##############################################################################################################
        # ... these return values are only available through training the final model (see finalModel = wmf_ensemble_iter(...))
        if 'dset_id' in vmap: 
            ret['dset_id'] = vmap['dset_id'] # each cycle may conclude a its own 'best params' different from the other cycles
            assert len(ret['dset_id']) == n_runs, "Had {n} runs, each of which should have its own 'best params' that defines the 'dset_id' (but got n={nd})".format(n=n_runs, nd=len(ret['dset_id']))
        # if 'dset_id_performance' in vmap: ret['dset_id_performance'] = vmap['dset_id_performance'] # use this retrieve stacker performance scores
        if 'best_params_inner' in vmap: 
            ret['best_params_inner'] = vmap['best_params_inner']
            assert len(ret['best_params_inner']) == n_runs
        ##############################################################################################################

        if tModelSelection:  # save the parameter and its score for later use ... [todo] refactor to 'evaluate' module
            eval_model_performance(M.records, dtype=frozenset, cols_params=['n_factors', 'alpha'], 
                col_score='score', col_freq='freq', save=True, setting=params.get('setting', -1))

            print('(verify) Best params consistency | best (max(max)): {p0} =?= best_nruns: {p1}'.format(p0=best_params, p1=best_params_nruns))

        div("(verify) Best parameter setting after model selection: {m}".format(m=ret['best_params']), symbol='#', border=1)
        params = {**params, **best_params} 
    ####################################################################################################################################
    
    assert not (None in {params['n_factors'], params['alpha']}), "Null hyperparams! Provide param_grid to activate model selection or specify via --n-factors and --alpha ..." 

    # file ID: e.g. wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior
    div('(wmf_ensemble) (S) setting: {0} | (F, A) n_factors: {1}, alpha: {2} | (CF) conf_measure: {3}, policy: {4}, policy_opt: {5}'.format(
        algorithm_setting,
        params['n_factors'], params['alpha'], 
        params['conf_measure'], params['policy'], params['policy_opt']), symbol='#', border=2)
    if params['policy_opt'].startswith('pref'): 
        div('... Use preference scores as a masking device to replace bad rating scores or probabilities  #', symbol='%', border=2)
    # ... say we use n_factors=20 and alpha=100 as default 

    file_types = ['prior', 'posterior', ]
    aggregation_methods = System.aggregation_methods # ['mean', 'median', 'log', ]
            
    # save predictions (as if they were stacking results)
    if tMetaUsers: 
        meta_users = ['latent_mean', 'latent_mean_masked', ]
        aggregation_methods += meta_users
        
    ####################################################################################

    # >>> use the 'best params' across multiple runs as the dataID in the naming of the performance dataframe (see wmf_ensemble())
    ret['dset_id_performance'] = dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)
    print('... best params: {p} => dset_id_performance: {id}'.format(p=best_params, id=dset_id))

    n_stackers = 0
    for file_type in file_types:
        for aggr in aggregation_methods: # ['latent_mean', 'latent_mean_masked', ] 
            pn = '{ftype}_{model}'.format(ftype=file_type, model=aggr) # pn: predictor name

            if file_type.startswith('prior'): 
                assert pn in vmap, "Missing prediction vectors for predictor type: {name}".format(name=pn)
            # ... stacker on posterior data is not sensible
        
            if pn in vmap: 
                # predictions = vmap[pn] # predictions is a dataframe: {'prediction' 'label', 'method', 'fold', 'params'}
                df_prediction = pd.concat(vmap[pn], ignore_index=True)
                
                # [test]
                if n_stackers == 0 and file_type.startswith('post'): 
                    print('(wmf_ensemble) best parameter history:\n{alist}\n ... (verify)'.format(alist=df_prediction['params'].unique()))

                # dset_type = file_type # file_type if file_type.startswith(('prior', 'post')) else 'train'
                output_path = '{path}/analysis/{stacker}.S-{dataset}-{suffix}.csv'.format(path=System.projectPath, 
                        stacker=aggr, dataset=dset_id, suffix=file_type)
            
                print('... (verify) Saving prediction result | dtype: {dtype}, meta user: {name}, data/method ID: {id} | output path:\n{path}\n'.format(dtype=file_type, name=aggr, 
                    id=dset_id, path=output_path))
                
                df_prediction.to_csv(output_path, index = False)
                n_stackers += 1 
    ####################################################################################
    print('(wmf_ensemble) Saved {n} sets of (basic) stacker predictions'.format(n=n_stackers))

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
    # [note] .consolidate() merges/averages Perf objects across CV folds (or subsampling iterations)
    #        .merge() merges multiple consolidated Perf objects across different algorithms            
    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(wmfMetrics), ])  

    # ret: output dictionary
    #      key: {methods}, {metrics}
    ret['metrics'] = perfAll
    div("(result) Sorted performance metrics on %s (n_metrics=%d)" % (PerformanceMetrics.tracked, len(PerformanceMetrics.tracked)), symbol='%')
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
    return ret  # keys: metrics, model, dset_id, dset_id_performance, best_params_inner 

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
    from utils_als import implicit_als_cg, implicit_als
    from evaluate import plot_roc, analyzePerf, Metrics, PerformanceMetrics
    import utils_cf as uc
    from itertools import product

    ### output
    ret = {}

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
    print('(wmf_similarity_ensemble) Confidence matrix based on mode={0}, alpha={1}'.format(params['conf_measure'], params['alpha']))

    topk = 30

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    # options: target_metric is the metric used to select the topK models
    baseline = base_predictors(topk=kargs.get('topk_bps', -1), 
                                    metric=kargs.get('metric', 'fmax')) if kargs.get('run_bp', False) else {}

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

        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
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
        # save_factors(P, Q, fold)
        Pe += P
        Qe += Q

        # given latent factors
        for kind in kinds[:2]:  # foreach user or item
            factors = P if kind == 'user' else Q
            S = uc.eval_similarity_by_latent_factors(factors, epsilon=1e-9)
            save_array(S, kind, fold)
            
            # axis = 0 if kind == 'user' else 1
            dimS = n_users if kind == 'user' else n_items_total
            assert S.shape[0] == S.shape[1] == dimS
            print('(test) kind={0} | dim(S): {1} (S[i,j] in [0, 1]?):\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # [note] R, T are to be merged prior to calling predict_debiased or predict_topk
            Rh, Th = uc.predict(R, T, S=S, kind=kind)
            assert T.shape == Th.shape, "dim(T): {0} but dim(Th): {1}".format(T.shape, Th.shape)

            method_specific = MFEnsemble.name_sim_method(method, kind=kind, params=params) # '{base}_{kind}_sim'.format(base=base, kind=kind)

            perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
            wmfMetrics[kind].append(perf)

            wmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )  # plot K-Fold CV
            if kargs.get('save', False): uc.save_reconstructed_probs((Rh, Th), labels=(L_train, L_test), fold=fold, method=method, verify=True, U=U)

            # use top K only 
            topk = int(math.floor(n_users/2)) if kind == 'user' else int(math.floor(n_items/10))
            Rh_topK, Th_topK = uc.predict(R, T, S=S, kind=kind, topk=topk)
        
            kind_topk = '%s_topk' % kind
            method_specific = MFEnsemble.name_sim_method(method, kind=kind_topk, params=params) # '{base}_{kind}'.format(base=base, kind=kind_topk)

            perf, pv = analyzePerf(L_test, Th_topK, method=method_specific, aggregate_func=np.mean) 
            wmfMetrics[kind_topk].append(perf )

            wmfCV[kind_topk].append( (L_test, uc.combiner(Th_topK, aggregate_func=np.mean)) )
            
            ## WMF + Clustering
            if kargs.get('run_clustering', False):
                for clustering in clusterings:  # foreach clustering method
                    kind_cluster = '%s_%s' % (kind, clustering)
                    method_specific = MFEnsemble.name_sim_method(method, kind=kind_cluster, params=params)

                    # factors could be user-based or item-based (depending on kind)
                    cluster_labels = uc.runClustering(factors, n_clusters=params['n_factors'], method=clustering) 
                    Rh, Th = uc.predict_by_cluster(R, T, similarity=S, kind=kind, C=cluster_labels)

                    perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
                    wmfMetrics[kind_cluster].append(perf)
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
     
        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
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
            S = uc.eval_similarity_by_latent_factors(factors, epsilon=1e-9)
            
            # axis = 0 if kind == 'user' else 1
            dimS = n_users if kind == 'user' else n_items_total
            assert S.shape[0] == S.shape[1] == dimS
            print('(test) kind={0} | dim(S): {1} (S[i,j] in [0, 1]?):\n{2}\n'.format(kind, S.shape, S[:4, :4]))

            # [note] R, T are to be merged prior to calling predict_debiased or predict_topk
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

                    perf, pv = analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean)
                    wmfMetrics[kind_cluster].append(perf )
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

def test_stacker_subsampling(context, dset_ids, evaluation=True, **kargs):
    """

    Memo
    ----
    context: example 'test_wmf_probs_via_stackers-{0}'.format(dset_id),  # ID for PerformanceMetrics

    """
    input_path = kargs.get('project_path', ProjectPath)
    # Test stackering in subsampling mode (based on the data generated by wmf_ensemble_iter()): 
    #    1. we will examine the stacking performance on 'prior' dataset (the training data prior to CF transformation)
    #    2. then examine the stacking performance on 'posterior' dataset
    domain = os.path.basename(input_path)
    div(message='(test_stacker_subsampling) Comparison of stackers (domain: {domain}, data IDs: {id})'.format(domain=domain, 
         id=dset_ids), symbol='#', border=1) # stackers on MF-reproduced probabilities vs stackers on BP predictions
    exact_match = True  # match the file name (of the training data) exactly as they are (format: <dset_id>-<dtype>-<fold>.csv.gz, e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-validation-1.csv.gz)
    mode = kargs.get('mode', 'train-test-split')  # mode: pair-wise, train-test-split
    performance_id = kargs.get('performance_id', '')

    # now do match-execute-evaluate 
    perfMetrics = []

    if mode.startswith('pair'): 
        for dtype in ['prior', 'posterior', ]: 
            # in this case, there is usually only one dataset ID in dset_ids

            # 1. match
            datasets = common.match_exact(path=input_path, method=dset_ids, file_type=dtype, ext='csv.gz', verify=True, policy_iter='subsampling') # exception_=False
            print("(test_stacker_subsampling) Found {n} sets of '{dtype}' data:\n{list}\n".format(n=len(datasets), dtype=dtype, list=datasets))

            # [todo]
            # 2. convert to id -> datasets
            
            # 3. execute (usuall there's only 1 matching data set)
            n_indices = 0
            for dataset, indices in datasets.items():
                if not indices: indices = range(System.n_runs)
                n_indices = len(indices)

                print('... (verify) method ID: {name}, indices: {idx}'.format(name=dataset, idx=indices)) # example method ID: wmf_F100_A100_XCFuser_S2
                # ... e.g. method ID: wmf_F75_A100_XCFuser_S2
                ret = run_stacker(dataset=dataset, parallelize=kargs.get('parallelize', True), indices=indices, file_type=dtype, mode=mode)
                perfMetrics.append(ret['metrics'])
    else: # train-test-split 
        for dtype in ['prior', 'posterior', ]: 

            # 1. matching
            dataIDs = common.match_exact(path=input_path, method=dset_ids, file_type=dtype, ext='csv.gz', verify=True, policy_iter='subsampling', mode=mode) # exception_=False
            print("(test_stacker_subsampling) Found {n} sets of '{dtype}' data:\n{list}\n".format(n=len(dataIDs), dtype=dtype, list=dataIDs)) 
            assert len(dataIDs) > 0, "Could not find any matching datasets for methods={ids}, dtype:{dtype}".format(ids=dset_ids, dtype=dtype)

            # e.g. 
            #      {'wmf_F75_A100_XCFuser_S2': [0, 2, 3, 4], 'wmf_F100_A100_XCFuser_S2': [1]}
            # 1a. set default performance ID if not given, where 'performance ID' is the dataID for the performance dataframe under analysis directory
            #     a separate dataID is necessary for performance dataframe beause we can no longer depend on a fixed dataID due to a mixture of models with potentially different 'best params'
            if not performance_id: 
                # use the param setting selected most frequently 
                performance_id = sorted(dataIDs, key=lambda k: len(dataIDs[k]), reverse=True)[0]

            # 2. convert to id -> datasets
            id_to_data = {}
            for dataID, indices in dataIDs.items():
                for index in indices: 
                    id_to_data[index] = dataID
                    # print('... index: {id1} => data: {id2}'.format(id1=index, id2=dataID))

            # 3. execute
            ret = run_stacker(dataset=id_to_data, parallelize=kargs.get('parallelize', True), indices=[], file_type=dtype, mode=mode, performance_id=performance_id)
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

    input_path = kargs.get('project_path', ProjectPath)
    div(message='(test_combiner_subsampling) Comparison of simple aggregations (project: {path}) | dset_id: {id}'.format(path=input_path, 
        id=dset_id), symbol='#', border=1) # stackers on MF-reproduced probabilities vs stackers on BP predictions
    exact_match = True  # match the file name (of the training data) exactly as they are (format: <dset_id>-<dtype>-<fold>.csv.gz, e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-validation-1.csv.gz)
    mode_evaluation = kargs.get('mode', 'train-test-split')
    
    perfMetrics = []

    aggregation_methods = System.simple_aggregation + System.latent_aggregation + System.stacker_aggregation
    # ... ['mean', 'median', ] + ['latent_mean', 'latent_mean_masked', ] + ['log', 'rf'] 
    for dtype in ['prior', 'posterior', ]: 
        # 1. match
        # datasets = common.match_exact(path=input_path, method=dset_id, file_type=dtype, ext='csv.gz', verify=True, policy_iter='subsampling') # exception_=False
        # print("(test_combiner) Found {n} sets of '{dtype}' data:\n{list}\n".format(n=len(datasets), dtype=dtype, list=datasets))

        # [todo]
        # 2. convert to id -> datasets

        # 3. excute  
        # for dataset, indices in datasets.items():
        #     if not indices: indices = range(System.n_runs)

        # e.g. wmf_F10_A100_Xbrier_preference-validation-3.csv.gz | prefix: wmf_F10_A100_Xbrier_preference
        for aggr in aggregation_methods: 
            combiner = run_combiner(dataset=dset_id, method=aggr, file_type=dtype, 
                has_performance_dataframe=True, skip_if_not_avail=True)  # run_simple_combiner(dataset, aggregation_func, file_type='')  

            # combiner['metrics'] consolidates scores from across different runs (or CV fold)
            if combiner and 'metrics' in combiner: 
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
    meta_users = ['latent_mean', 'latent_mean_masked']
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

                perf, pv = analyzePerf(labels, predictions, method=full_method)
                perf_per_fold.append( perf )
            
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

                    5: 'rating-cascade-sequence', 6: 'rating-cascade-classifier', 
                    7: 'filter-by-polarity', 
                    8: 'filter-by-polarity-cascade', 
                    9: 'filter-by-polarity-sequence-model', 

                    10: 'filter-by-polarity-classifier', 
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

                    # using stacker to optimze model parameters 
                    72: 'user-centered-stacker', 

                    # algorithmic control group 
                    100: 'uniform', 
                    }

    actions = {'replace_subset': 'replace bad ratings (i.e. predict <- False)', 
               'reconstruct': 'reconstruct the entire rating table (i.e. predict <- True)', 
               'support': 'using only %{ratio} of users as a support for each item', 
            }

    # default parameter grid
    ctrl_params = ['n_factors', 'alpha', ]
    
    ##########################################
    # ... select algorithmic setting

    # global parameter 
    print('(test_wmf_probs_suite) testing setting #{0}\n'.format(setting))

    tReconstructMatrix = False  # if True, reconstruct the entire rating matrix (replace_all); set to False to use 'replace_subset'
    tMetaUsers = False

    # global setting 
    kargs['include_meta_users'] = tMetaUsers
    kargs['replace_all'] = kargs['predict_probs'] = tReconstructMatrix    # if False, call uc.replace() => replace bad ratings 
    kargs['replace_subset'] = not kargs['replace_all']
    kargs['policy_iter'] = System.policy_iter
    kargs['conf_measure'] = 'fmax' # 'brier', 'rank', 'fmax'

    tWeightedPrediction = kargs['weighted_output'] = True
    kargs['policy_calibration'] = 'f-pref' # 'precision' # 'error' # 'hit-to-miss'  
    # ... 'agreement', 'positive', 'tp', 'f-pref', 'f-align', 'hit-to-miss'/'ratio', 'conditional-ratio'
    training_mode = kargs['training_mode'] = 'cascade' 
    # ... options: 'regular', 'cascade' | 'pref'/'preferred_rating', 'pref_cascade' 
    #     {'pref', 'pref_cascade'} involves re-estimating ratings for entries that are deemed to be not reliable (or not preferred)

    # test time filtering axis 
    policy_filter_test = 'item'

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
        # kargs['training_mode'] = 'regular' 

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (2, 'user-centered'):
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test   # filtering along the axis of item
        
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True

        # kargs['training_mode'] = 'regular' 
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
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
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
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
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (5, 'rating-cascade-sequence'): 
        kargs['training_mode'] = 'cascade'
        kargs['policy'] = 'item' # 'user'

        # if training_mode.startswith('pref'): 
        #     policy_filter_test == kargs['policy']  # without probability reestimates, it is not a good idea to have 'conservative' preference matrix in the test test
        kargs['policy_test'] = 'polarity' # 'item', 'user', 'polarity'  # filtering along the axis of item
        # kargs['labeling_model'] = 'simple'  # use logistic regression to estimate the labeling of the test split   <<< key distinction from setting=8
        # kargs['constrained'] = False
        # kargs['stochastic'] = True
        kargs['policy_polarity'] = 'sequence' # options: 'classification', 'sequence', 'median'

        kargs['suppress_negative_examples'] = False
        kargs['beta'] = 1.0

        kargs['policy_opt'] = 'rating' # 'preference'
        kargs['explicit_mf'] = False

        kargs['binarize_pref'] = True
        kargs['weighted_output'] = tWeightedPrediction
        kargs['replace_subset'] = False   # if False, reconstruct the entire T to make predictions

        kargs['policy_threshold'] = 'fmax' 
        # ... options: 'fmax' 'prior'

        # kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        kargs['ratio_users'] = 0.5  # two-sided; used in estimating Pf for T; more conservative in the test split

        div('[{0}] Preference scores used as meta data for (R, T)-entry selection >> {action}'.format(setting, 
            # action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']), 
            action='weighted averaging'), symbol='=', border=2)
    elif setting in (6, 'rating-cascade-classifier'): 
        kargs['training_mode'] = 'cascade'
        kargs['policy'] = 'item' # 'user'

        # if training_mode.startswith('pref'): 
        #     policy_filter_test == kargs['policy']  # without probability reestimates, it is not a good idea to have 'conservative' preference matrix in the test test
        kargs['policy_test'] = 'polarity' # 'item', 'user', 'polarity'  # filtering along the axis of item
        # kargs['labeling_model'] = 'simple'  # use logistic regression to estimate the labeling of the test split   <<< key distinction from setting=8
        # kargs['constrained'] = False
        # kargs['stochastic'] = True
        kargs['policy_polarity'] = 'classification' # options: 'classification', 'sequence', 'median'
        
        # resampling 
        kargs['balance_class_resampling'] = False  # balance class distributions

        kargs['suppress_negative_examples'] = False
        kargs['beta'] = 1.0

        kargs['policy_opt'] = 'rating' # 'preference'
        kargs['explicit_mf'] = False

        kargs['binarize_pref'] = True
        kargs['weighted_output'] = False
        kargs['replace_subset'] = False   # if False, reconstruct the entire T to make predictions

        kargs['policy_threshold'] = 'fmax' 
        # ... options: 'fmax' 'prior'

        # kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        kargs['ratio_users'] = 0.5  # two-sided; used in estimating Pf for T; more conservative in the test split

        div('[{}] Preference scores used as meta data for (R, T)-entry selection | balance_class_resampling: {}, weighted_output: {} '.format(setting, 
                kargs['balance_class_resampling'], kargs['weighted_output']), symbol='=', border=2)
    elif setting in (7, 'filter-by-polarity'): 
        kargs['training_mode'] = 'cascade'
        kargs['policy'] = 'item' # 'user'

        # if training_mode.startswith('pref'): 
        #     policy_filter_test == kargs['policy']  # without probability reestimates, it is not a good idea to have 'conservative' preference matrix in the test test
        kargs['policy_test'] = 'polarity' # 'user', policy_filter_test   # filtering along the axis of item
        kargs['suppress_negative_examples'] = False
        kargs['beta'] = 1.0   # weight multiple for TP, FP, FN

        kargs['policy_opt'] = 'preference'
        kargs['binarize_pref'] = True
        kargs['pref_threshold'] = 0.5    # set to -1 to calibrate

        kargs['weighted_output'] = tWeightedPrediction

        kargs['policy_threshold'] = 'fmax' 
        # ... options: 'fmax' 'prior'

        # kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        kargs['ratio_users'] = -1  # two-sided; used in estimating Pf for T; more conservative in the test split

        div('[{0}] Preference scores used as meta data for (R, T)-entry selection >> {action}'.format(setting, 
            action='weighted averaging'), symbol='=', border=2)
    elif setting in (8, 'filter-by-polarity-cascade'): 
        kargs['training_mode'] = 'pref_cascade'
        kargs['policy'] = 'item' # 'user'

        # if training_mode.startswith('pref'): 
        #     policy_filter_test == kargs['policy']  # without probability reestimates, it is not a good idea to have 'conservative' preference matrix in the test test
        kargs['policy_test'] = 'polarity'   # 'user', policy_filter_test   # filtering along the axis of item
        kargs['labeling_model'] = 'simple'  # use logistic regression to estimate the labeling of the test split   <<< key distinction from setting=8
        kargs['constrained'] = True
        kargs['stochastic'] = True

        kargs['suppress_negative_examples'] = False
        kargs['beta'] = 1.0   # weight multiple for TP, FP, FN

        kargs['policy_opt'] = 'preference'
        kargs['explicit_mf'] = True
        kargs['approx_ratings_via_pref'] = False

        kargs['binarize_pref'] = True
        kargs['pref_threshold'] = 0.5   # set to -1 to calibrate

        kargs['weighted_output'] = tWeightedPrediction
        kargs['policy_threshold'] = 'fmax' 
        # ... options: 'fmax' 'prior'

        # kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        kargs['ratio_users'] = 0.5  # two-sided; used in estimating Pf for T; more conservative in the test split

        div('[{0}] Preference scores used as meta data for (R, T)-entry selection >> {action}'.format(setting, 
            action='weighted averaging'), symbol='=', border=2)

    elif setting in (9, 'filter-by-polarity-sequence-model'): 
        kargs['training_mode'] = 'pref_cascade'
        kargs['policy'] = 'item' # 'user'

        # if training_mode.startswith('pref'): 
        #     policy_filter_test == kargs['policy']  # without probability reestimates, it is not a good idea to have 'conservative' preference matrix in the test test
        kargs['policy_test'] = 'polarity' # 'user', policy_filter_test   # filtering along the axis of item
        kargs['labeling_model'] = 'logistic'   # use logistic regression to estimate the labeling of the test split   <<< key distinction from setting=8
        kargs['constrained'] = True
        kargs['stochastic'] = False
        kargs['policy_polarity'] = 'sequence' # 'median', 'classification'

        kargs['estimate_sample_type'] = True

        kargs['suppress_negative_examples'] = False
        kargs['beta'] = 1.0   # weight multiple for TP, FP, FN

        kargs['policy_opt'] = 'preference'
        kargs['explicit_mf'] = False
        kargs['approx_ratings_via_pref'] = False 
        # ... if False, use masked confidence matrix (masked using polarity matrix) to construct cost function
        # ... if True, use the preference matrix (derived from latent factors via ALS) as the polarity matrix for proba value re-estimation

        kargs['binarize_pref'] = True
        kargs['pref_threshold'] = 0.5  # set to -1 to calibrate

        kargs['weighted_output'] = False # tWeightedPrediction
        kargs['policy_threshold'] = 'fmax' 
        # ... options: 'fmax' 'prior'

        # kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        kargs['ratio_users'] = 0.5  # two-sided; used in estimating Pf for T; more conservative in the test split
    elif setting in (10, 'filter-by-polarity-classifier'): 
        kargs['training_mode'] = 'pref_cascade'
        kargs['policy'] = 'item' # 'user'

        # if training_mode.startswith('pref'): 
        #     policy_filter_test == kargs['policy']  # without probability reestimates, it is not a good idea to have 'conservative' preference matrix in the test test
        kargs['policy_test'] = 'polarity' # 'user', policy_filter_test   # filtering along the axis of item
        kargs['labeling_model'] = 'logistic'   # use logistic regression to estimate the labeling of the test split   <<< key distinction from setting=8
        kargs['constrained'] = True
        kargs['stochastic'] = False
        kargs['policy_polarity'] = 'classification' # 'median', 'classification'

        # resampling 
        kargs['balance_class_resampling'] = False  # balance class distributions
        
        kargs['estimate_sample_type'] = True
        kargs['suppress_negative_examples'] = False
        kargs['beta'] = 1.0   # weight multiple for TP, FP, FN

        kargs['policy_opt'] = 'preference'
        kargs['explicit_mf'] = False
        kargs['approx_ratings_via_pref'] = False
        # ... if False, use masked confidence matrix (masked using polarity matrix) to construct cost function
        # ... if True, use the preference matrix (derived from latent factors via ALS) as the polarity matrix for proba value re-estimation

        kargs['binarize_pref'] = True
        kargs['pref_threshold'] = 0.5  # set to -1 to calibrate

        kargs['weighted_output'] = True # tWeightedPrediction
        kargs['policy_threshold'] = 'fmax' 
        # ... options: 'fmax' 'prior'

        # kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        kargs['ratio_users'] = 0.5  # two-sided; used in estimating Pf for T; more conservative in the test split

        div('[{}] Preference scores used as meta data for (R, T)-entry selection | balance_class_resampling: {}, weighted_output: {} '.format(setting, 
                kargs['balance_class_resampling'], kargs['weighted_output']), symbol='=', border=2)

    # tradeoff in the cost function 
    elif setting in (11, 'item-centered-tradeoff'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (12, 'user-centered-tradeoff'):
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (17, 'item-centered-tradeoff-reconstruct'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True # True is default

        kargs['mask_all_test'] = True  # mark all test set as not reliable, do not try to estimate which ones to re-estimate
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (18, 'user-centered-tradeoff-reconstruct'): 
        kargs['policy'] = 'user'
        kargs['policy_opt'] = 'tradeoff'  
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = -1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        kargs['mask_all_test'] = True
        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix while MASKING ENTIRE TEST SET >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (21, 'item-centered-transfer'): 
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'transfer'
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (22, 'user-centered-transfer'):
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test   # filtering along the axis of item
        kargs['policy_opt'] = 'rating'  
        kargs['policy_opt_T'] = 'transfer'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
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
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (32, 'user-centered-fmax'):
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test   # filtering along the axis of item
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['policy_threshold'] = 'fmax'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
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
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (42, 'user-centered-seeding'):
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        # kargs['policy_threshold'] = 'prior'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
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
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (44, 'user-centered-long-iter'):
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test

        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        # kargs['policy_threshold'] = 'prior'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        System.n_epochs = 100
        System.n_epochs_foldin = 50
        System.lambda_val = 0.7  # 0.8 by default

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
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
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (46, 'user-centered-low-reg'):
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test

        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'seed' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        
        # kargs['policy_threshold'] = 'prior'
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        # System.n_epochs = 100
        # System.n_epochs_foldin = 50
        System.lambda_val = 0.5  # 0.8 by default

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    # confidence scores and mask fucntions
    elif setting in (51, 'item-centered-uniform'):  # uniform confidence scores
        kargs['policy'] = 'item'
        kargs['policy_opt'] = 'rating'
        kargs['conf_measure'] = 'uniform'
        kargs['ratio_users'] = 0.5 
        kargs['supervised'] = True

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (52, 'user-centered-uniform'): # uniform confidence scores
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test

        kargs['policy_opt'] = 'rating'
        kargs['conf_measure'] = 'uniform'
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        
        kargs['supervised'] = True
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (62, 'meta-users-filter-user'):  # uniform confidence scores
        kargs['policy'] = 'user'
        kargs['policy_test'] = 'user'
        kargs['include_meta_users'] = True

        kargs['policy_opt'] = 'rating'
         
        kargs['ratio_users'] = 0.5    # need this for unsupervised item-centered filtering in T
        kargs['supervised'] = True   # supervised in R

        div('[{0}] Approximate {scores} using (({repr}-centered in R but {reprtest}-centered in T)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], reprtest=kargs['policy_test'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (63, 'meta-users-filter-user-item'):  # uniform confidence scores
        kargs['policy'] = 'user'
        kargs['policy_test'] = 'item'
        kargs['include_meta_users'] = True

        kargs['policy_opt'] = 'rating'
         
        kargs['ratio_users'] = 0.5    # need this for unsupervised item-centered filtering in T
        kargs['supervised'] = True   # supervised in R

        div('[{0}] Approximate {scores} using (({repr}-centered in R but {reprtest}-centered in T)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], reprtest=kargs['policy_test'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)
    elif setting in (72, 'user-centered-stacker'):
        kargs['policy'] = 'user'
        kargs['policy_test'] = policy_filter_test
        kargs['policy_opt'] = 'rating'
        kargs['policy_opt_T'] = 'foldin' # {'foldin', 'seed', 'transfer', 'transfer+seed'} 
        kargs['supervised'] = True
        ms_mode = kargs['policy_ms_model'] = 'stacking'
        aggr_func = kargs['policy_aggregate_func'] = 'log'   # <<< choose a stacker here
        # kargs['ratio_small_class'] = 0.1  # only relevant when in unsupervised mode i.e. suerpvised <- False

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix | model selection (mode: {mode}, method: {method}) >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], mode=ms_mode, method=aggr_func, 
            action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    # algorithmic 'control group' (those that may not work but still interesting to try)
    elif setting in (100, 'uniform'):
        kargs['policy'] = 'uniform'
        kargs['policy_opt'] = 'rating'
        kargs['conf_measure'] = 'uniform'
         
        kargs['ratio_users'] = 0.5  # don't care
        kargs['supervised'] = True   # don't care

        div('[{0}] Approximate {scores} using (({repr}-centered)) confidence matrix >> {action}'.format(setting, 
            scores=kargs['policy_opt'], repr=kargs['policy'], action=actions['replace_subset'] if not kargs['predict_probs'] else actions['reconstruct']),
            symbol='=', border=2)

    else: 
        raise NotImplementedError

    display()

    # logically dependent options 
    if training_mode.startswith('pref'):
        assert kargs['policy_opt'].startswith('pref'), "Conflicting training and optimization modes | training mode: {}, policy_opt: {}".format(training_mode, kargs['policy_opt'])

    ########################################
    # Main test routine 

    test_wmf_probs(**kargs)

    #########################################
    # note: this message has to be consistent with Job.end_job (see cf_run.py, parse_job.py)
    div('(test_wmf_probs_suite) Completed experimental setting #{}: {} | training mode: {} ---#\n'.format(setting, 
        descriptions.get(setting, 'generic'), training_mode), symbol='#', border=1)

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
    
    tTestStacker = kargs.get('test_stacker', False)  
    tTestCombiner = kargs.get('test_combiner', False)
    # mode_evaluation = kargs.get('mode_evaluation', 'train-test-split')

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
    training_mode = kargs.get('training_mode', 'regular')
    meta_params = { 
                   'setting': kargs.get('setting', algorithm_setting), 
                   'training_mode': training_mode, 
                   
                   'conf_measure': kargs.get('conf_measure', 'brier'),   # options: 'brier', 'uniform', 'ratio', 'corr', 'auc' ...
                   'suppress_negative_examples': kargs.get('suppress_negative_examples', False), 

                   'policy': kargs.get('policy', 'user'),  # {'user', 'item', }
                   'policy_test': kargs.get('policy_test', 'user'),   # filtering policy for the test set; default to be same as that of the training set
                   
                   'policy_opt': kargs.get('policy_opt', 'preference'),  # {'rating', 'preference', 'label', 'tradeoff'}
                   'explicit_mf': kargs.get('explicit_mf', False), 
                   'approx_ratings_via_pref': kargs.get('approx_ratings_via_pref', False), 

                   'policy_opt_T': kargs.get('policy_opt_T', 'foldin'),  # {'foldin', 'seed', 'transfer', 'transfer+seed'}
                   'policy_replace': kargs.get('policy_replace', 'rating'),  # only relevant when policy_opt <- 'preference'
                   'policy_iter': kargs.get('policy_iter', 'subsampling'),      # iteration policy; with model selection, set policy_iter to 'subsampling' 
                   'policy_threshold': kargs.get('policy_threshold', 'prior'),  # group II, how prob thresholds are determined {'fmax', 'prior'/'topk'}
                   'policy_calibration': kargs.get('policy_calibration', 'agreement'),    # preference calibration 
                   'two_way_calibration': kargs.get('two_way_calibration', False),

                    # parameters for polarity matrix
                   'constrained': kargs.get('constrained', True),
                   'stochastic': kargs.get('stochastic', True), 
                   'estimate_sample_type': kargs.get('estimate_sample_type', True),
                   'labeling_model': kargs.get('labeling_model', 'simple'),   # options: 'simple', 'stacking'; used to determine polarity matrix (only relevant when policy_filter or policy_filter_test is 'polarity')
                   'policy_polarity': kargs.get('policy_polarity', 'sequence'),
                   'balance_class_resampling': kargs.get('balance_class_resampling', False), 

                   'binarize_pref': kargs.get('binarize_pref', False),
                   'preference_calibration': kargs.get('preference_calibration', True),  
                   'pref_threshold': kargs.get('pref_threshold', -1),
                   'pref_threshold_test': kargs.get('pref_threshold_test', -1), 

                   'weighted_output': kargs.get('weighted_output', False), 
                   'beta': kargs.get('beta', 1.0),   # the multiple at which positive sample weights are magnified
                   
                   'policy_ms': kargs.get('policy_ms', 'freq'),
                   'policy_ms_model': kargs.get('policy_ms_model', 'mean'),  # 'mean', 'stacking', 'user'
                   'policy_aggregate_func': kargs.get('policy_aggregate_func', 'mean'), # 'mean', 'median', 'log', ... 

                   'ms_global': kargs.get('ms_global', False),   # if True, invoke (again after model selection finalized) wmf_ensemble_model_select with the globally best hyperparams

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
                   
                   'replace_subset': kargs.get('replace_subset', True), 

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
    # if sum(1 for v in param_grid.values() if len(v) > 1) > 0: 
    # # if sum(len(param_grid[p]) for p in hyperparams) > len(param_grid): # if any of the parameter has >= 2 values, then we are automatically in model selection mode
    #     # then we are in model selection mode, which means we want policy_iter to be 'subsampling'
    #     meta_params['policy_iter'] = 'subsampling'  # cv is too expensive for model selection 
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
    ret = {}

    if tTrainModel:
        n_target_methods = 0

        # for params in list(ParameterGrid(param_grid_outer)):  # a list of dictionaries containing target (hyper)parameters
        ret = wmf_ensemble(
                        # run_bp=False, run_bp_stacker=False,  
                        # run_wmf_ensemble=True, run_wmf_stacker=False, run_wmf_similarity=False, 
                        setting=meta_params['setting'],  # important file descriptor
                        training_mode=meta_params['training_mode'], 
                        
                        augmented=meta_params['augmented'], # if False, run uc.replace() instead of uc.predict_by_factors() by default
                        include_meta_users=meta_params['include_meta_users'],

                        # predict_probs=meta_params['predict_probs'],     # prediction vs reconstruction (if False) 
                        replace_subset=meta_params['replace_subset'],
                        supervised=meta_params['supervised'],   # using supervised methods to estimate the mask (of the confidence matrix)
                        # mask_all_test=meta_params['mask_all_test'], 
                        # masked=meta_params['masked'],   # if True, apply mask function to either reduce confidence weights or replace 'bad ratings'
                        
                        # conf_user=meta_params.get('conf_user', True), 
                        conf_measure=meta_params['conf_measure'], 
                        suppress_negative_examples=meta_params['suppress_negative_examples'],
                        policy=meta_params['policy'],  # specifies the policy for confidence matrx (and for optimization policy when 'policy_opt' is not given)
                        policy_test=meta_params['policy_test'], 

                        policy_opt=meta_params['policy_opt'],  # optimization
                        policy_opt_T=meta_params['policy_opt_T'],  # optimization on the test set
                        explicit_mf = meta_params['explicit_mf'],
                        approx_ratings_via_pref = meta_params['approx_ratings_via_pref'], 

                        policy_threshold=meta_params['policy_threshold'],
                        
                        policy_ms=meta_params['policy_ms'],  # e.g. 'freq', 'mean'
                        policy_ms_model=meta_params['policy_ms_model'],  # 'mean', 'stacking', 'user'
                        policy_aggregate_func=meta_params['policy_aggregate_func'],
                        policy_calibration=meta_params['policy_calibration'], 
                        two_way_calibration=meta_params['two_way_calibration'],

                        constrained=meta_params.get('constrained', True),
                        stochastic=meta_params.get('stochastic', True), 
                        estimate_sample_type=meta_params.get('estimate_sample_type', True),
                        labeling_model=meta_params.get('labeling_model', 'simple'),  # used to determine polarity matrix
                        policy_polarity=meta_params.get('policy_polarity', 'sequence'),
                        balance_class_resampling=meta_params.get('balance_class_resampling', False),  # apply resampling so that the final class distribution is balanced (e.g. undersample majority class)

                        binarize_pref=meta_params['binarize_pref'],
                        preference_calibration=meta_params['preference_calibration'],
                        pref_threshold=meta_params['pref_threshold'],
                        pref_threshold_test=meta_params['pref_threshold_test'],

                        weighted_output=meta_params['weighted_output'],
                        beta=meta_params['beta'],

                        ms_global=meta_params['ms_global'],  # use globally best params if True
                        
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

    assert 'dset_id_performance' in ret
    dset_id_performance = ret['dset_id_performance']
    print('(test_wmf_probs) Using performance dataID: {id}'.format(id=dset_id_performance))
    # dset_id_performance = MFEnsemble.get_dset_id(method='wmf', params=best_params, meta_params=meta_params) 

    mode_evaluation = kargs.get('mode_evaluation', 'train-test-split')
    if tTestStacker: 
        print("(1) testing stackers on 'prior' and 'posterior' datasets ... ")

        dset_ids = []
        if 'dset_id' in ret:
            assert len(ret['dset_id']) == meta_params['n_runs']
            dset_ids = ret['dset_id']
        else: 
            # if we only focus on a single best params ... 
            for params in [best_params, ]:
                dset_ids.append( MFEnsemble.get_dset_id(method='wmf', params=params, meta_params=meta_params) ) # meta: extra info needed to differentiate training data

        # [test] test if the validation and prediction sets exist
        print("(test_wmf_probs) Evaluating stackers under mode: {mode} | dset_ids: {id}".format(mode=mode_evaluation, id=dset_ids)) # ex. params: {'n_factors': 250, 'alpha': 100}

        # perf = test_stacker(context='stackers-prior-vs-posterior-{0}'.format(dset_id),  # ID for PerformanceMetrics
        #                         run_bp=False, run_bp_stacker=False,  # focused on wmf stackers

        #                         datasets=[dset_id, ],  # a set of datasets named after training data IDs
        #                         exact=True)  # aug_data=meta_params['aug_data']
        dset_id_performance = ret.get('dset_id_performance', '?')
        perf = test_stacker_subsampling(context='stackers-prior-vs-posterior-{params}'.format(params=dset_id_performance), 
                dset_ids=dset_ids, evaluation=True, mode=mode_evaluation, performance_id=dset_id_performance)  
        # ... where 'dset_id_performance' refers to the substring like 'F100A100' in the performance dataframe under analysis e.g. mean.S-wmf_F100_A100_XCFuser_S2-prior.csv
        perfMetrics.append(perf)
        #############################################
        # ... perf object include both 'prior' and 'posterior'
        
        for dtype in ['prior', 'posterior', ]:
            adict = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[dtype, ])  # keys: methods, metrics, mean, median, max, min
            entry = 'stacker_{dtype}'.format(dtype=dtype)
            fmaxSummary[ entry ] = adict['mean']

            div("(result) Average WMF-stacker fmax on {dtype} data (n_methods: {n}): {score}".format(dtype=dtype, n=len(adict['methods']), score=adict['mean']), symbol='#', border=2)
            print('... WMF stacker methods:\n... \n%s\n' % format_list(zip(adict['methods'], adict['scores']), mode='v', sep=', ', padding=5))
        #############################################
        
    # test via simple aggregation method
    # for params in list(ParameterGrid(param_grid)): 

    if tTestCombiner: 
        print("(2) testing simple aggregation methods on 'prior' and 'posterior' datasets ... ")

        print('(test_wmf_probs) Basic aggregate on (Rh, Th), e.g. mean predictions (no stacking) | dset_id(performance): %s, params: %s' % (dset_id_performance, best_params))
        
        # note: can use this routine to aggregate preference scores as well
        # perf = test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(dset_id), 
        #                         datasets=[dset_id, ], method='mean') # params: aggregate_func,aug_data, test  
        perf = test_combiner_subsampling(context='combiners-prior-vs-posterior-{0}'.format(dset_id_performance), dset_id=dset_id_performance, evaluation=True)
        perfMetrics.append(perf)
        #############################################
        # ... perf object include both 'prior' and 'posterior'

        for dtype in ['prior', 'posterior', ]:
            adict = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[dtype, ])  # keys: methods, metrics, mean, median, max, min
            entry = 'combiner_{dtype}'.format(dtype=dtype)
            fmaxSummary[entry] = adict['mean']

            div("(result) Average WMF-combiner fmax on '{dtype}' data (n_methods: {n}): {score}".format(dtype=dtype, n=len(adict['methods']), score=adict['mean']), symbol='#', border=2)
            print('... WMF combiner methods:\n... \n%s\n' % format_list(zip(adict['methods'], adict['scores']), mode='v', sep=', ', padding=5))
        #############################################

    # Below has been subsumed by case (2) above
    # print("(3) testing meta user predictions ...")
    # if tMetaUsers: 
    #     for params in [best_params, ]:
    #         dset_id = MFEnsemble.get_dset_id(method='wmf', params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data  

    #         context = 'meta_users' 
    #         perf = test_meta_users(context, dset_id, evaluation=True, sep=',') 
    #         perfMetrics.append(perf)

    #         # e.g. example methods 
    #         #      latent_mean_wmf_F100_A100_XCFuser_S63', 'masked_latent_mean_wmf_F100_A100_XCFuser_S63'
    #         for dtype in ['prior', 'posterior', ]:
    #             ret = PerformanceMetrics.summarize(perf, metric='fmax', keywords=[dtype, ])  # keys: methods, metrics, mean, median, max, min
    #             entry = 'meta_user_{dtype}'.format(dtype=dtype)
    #             fmaxSummary[entry] = ret['mean']

    #             div("(result) Average WMF-meta-users fmax on '{dtype}' data (n_methods: {n}): {score}".format(dtype=dtype, n=len(ret['methods']), score=ret['mean']), symbol='#', border=2)
    #             print('... WMF meta users:\n... \n%s\n' % format_list(zip(ret['methods'], ret['scores']), mode='v', sep=', ', padding=5))

    # Alternatively, put parameter grid iteration inside the test_<ensemble method> function
    # test_combiner(context='test_wmf_probs_via_combiner-{0}'.format(dset_id), 
    #     param_grid=param_grid, meta=meta, method='mean', target_method='wmf') # method refers to the combining method, specifiy specific MF method via target method
    
    ####################################################
    # context: for the moment, include algorithm parameter as well
    # 
    nf = param_grid['n_factors'][0] if len(param_grid['n_factors']) == 1 else best_params['n_factors']
    alpha = param_grid['alpha'][0] if len(param_grid['alpha']) == 1 else best_params['alpha'] 
    context = 'test_wmf_probs_all_methods-{params}'.format(params=dset_id_performance)   # n{n}a{a}'.format(n=nf, a=alpha)
    post_analysis(PerformanceMetrics.merge(perfMetrics), context=context, metrics=['fmax', 'auc', ])  # highlight=['wmf', ]

    ##########################################
    # div('(result) Mean performance in {metric}:\n... BP: {b}\n ... stacker: {s}\n... WMF stacker: {ws}\n... WMF combiner: {wc}'.format(metric='fmax', 
    #     b=fmaxSummary['bp'], s=fmaxSummary['stacker'], ws=fmaxSummary['wmf_stacker'], wc=fmaxSummary['wmf_combiner']), symbol='#', border=2)
    
    title = '(result) Mean performance in {metric} (domain: {domain}) | setting: ({s}, {descrp}) hyperparams: {params}'.format(metric='fmax', 
        domain=System.domain, s=algorithm_setting, descrp=System.descriptions.get(algorithm_setting, 'generic'), params=best_params)
    print( format_sort_dict(fmaxSummary, reverse=True, padding=5, title=title) ) # symbol='#', border=1


    div('(test_wmf_probs) Model training complete | training mode: {} ... #'.format(training_mode))
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
    print('(verify) param_grid(opt): {opt} System.param_grid: {sys} | default n_factors: {nf}'.format(opt=param_grid, sys=System.param_grid, nf=System.n_factors))

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
    # R, T, L_train, L_test, U = utils_cf.to_rating_matrix2(fold, p_threshold=0.5, missing_value=0, verbose=True)
    # print('(test) dim(R): %s, dim(T): %s' % (str(R.shape), str(T.shape)))

    # label matrix doesn't quite make sense because all "users" i.e. classifiers will share the same ground truths 
    # L, Lt = to_label_matrix(fold, p_threshold=0.5, save_=False, missing_value=-1, merge_=False)
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

def test2(**kargs):

    # read train test splits created by wmf_ensemble() 
    # common.match_exact(path=input_path, method=dset_id, file_type=dtype, ext='csv.gz', verify=True, policy_iter='subsampling')

    return 

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
    test(train_model=True, train_base_model=False, test_stacker=False, test_combiner=True)
    # ... test_stacker set to False: stackers are now run within wmf_ensmeble()
    t2 = time.time()
    
    del_t = t2 - t1
    div("Total execution time (via time.time()): {h} hrs ~ {m} mins".format(h=del_t/3600., m=del_t/60.), symbol='#', border=1)
    print('\n> options: {opts}, args: {args}\n... #'.format(opts=System.options, args=System.args))
    return 

if __name__ == "__main__":     
    main()
    # runTest()

