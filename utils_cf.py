# encoding: utf-8

# Surprise
# from __future__ import (absolute_import, division, print_function,
#                         unicode_literals)

import os, sys, re, math, random, time
import collections, operator
from sys import argv
import scipy
import scipy.io
# import scipy.stats as stats
import numpy as np
import pickle
import timeit
from pandas import DataFrame, Series

# Scikit-learn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier

# import snips as snp  # my snippets
# snp.prettyplot(matplotlib)  # my aesthetic preferences for plotting

# Plotting 
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('ggplot')  # values: {'seaborn', 'ggplot', }
# from utils_plot import saveFig, plot_path

# CF modules
from analyzer import is_sparse
import cf_spec
from cf_spec import System
from nnls import NNLS
import common, utilities

import utils_knn as uknn
import polarity_model as pmodel

import utils_sys
from utils_sys import div

from evaluate import visualizeCoeffs
import cluster
######################################################################################################
#
#
#
#
#
#   Memo
#   ----
#   1. Ethan Rosenthal
#      
#      + neighborhood-based CF
#          https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/
#      + user-user vs item-item
#          https://kerpanic.wordpress.com/2018/03/26/a-gentle-guide-to-recommender-systems-with-surprise/
# 
#   2. Towards Data Science 
#      + Prince Grover 
#           https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
#
#   3. Surprise 
#      + http://surpriselib.com/
#   

### configurations
ProjectPath = cf_spec.ProjectPath   # utils_sys.getProjectPath(domain=Domain, verify_=False)  # e.g. /Users/pleiades/Documents/work/data/diabetes_cf
Domain = cf_spec.Domain  # os.path.basename(ProjectPath) 
# assert os.path.exists(ProjectPath), "Invalid project path: %s" % ProjectPath

class MFEnsemble(cf_spec.MFEnsemble):

    header_model_evaluation = ['method', 'score', 'posterior_score', 'mean_score', ]

    @staticmethod
    def save_meta_tset(X, fold, params, indices=[], method='wmf', subsampling=False, verbose=1): # module: tset
        # MFEnsemble.meta_keys: ['conf_measure', 'policy', 'supervised', 'masked']
        assert len(X) >= 2 and len(X) <= 5, "X: (R, T, L_train, L_test, U)"
        if len(X) < 5:
            # assert len(indices) == 0, "In random subsampling mode, one cannot use CV fold to recover the labeling info." 
            Rh, Th = X
            (_, _, L_train, L_test, U) = to_rating_matrix(fold, unbag=False)
        else: 
            Rh, Th, L_train, L_test, *U = X
            if len(U) > 1: 
                U, *indices = U
            else: 
                U = U[0] 

        meta_params = {k: params[k] for k in MFEnsemble.meta_keys}  # meta_params are a subset of params that do not go into model selection loop
        tset_id = MFEnsemble.get_tset_id(method=method, params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data
        print('(save_meta_tset) verify > tset_id: {id}'.format(id=tset_id))

        isAugmented = params.get('augmented', True)
        if not isAugmented: 
            assert Th is None
        else: 
            assert Th is not None
            print('(test) dim(Rh): {0}, dim(Th): {1}, n_train: {2}, n_test: {3}'.format(Rh.shape, Th.shape, L_train.size, L_test.size))
        
        if verbose > 1: print('(save_meta_tset) Saving reconstructed data > Rh? {0}, Th? {1}  #'.format(Rh is not None, Th is not None))

        # >>> (Rh, Th) represents new rating scores or not? 
        isRating = not params['policy_opt'].startswith('pref') or params['replace']
        
        if verbose > 0: print('... (Rh, Th) represent new rating scores? {0}'.format(isRating))
        save_reconstructed_probs((Rh, Th), labels=(L_train, L_test), indices=indices, fold=fold, method=tset_id, subsampling=subsampling, verify=True, U=U, is_rating=isRating)
        return

    @staticmethod
    def save_array(A, U=None, name='user-sim', dset_id='generic', file_type='posterior', sep=',', format_='csv', domain='', project_path='', index=-1, verbose=True):
        """

        Use 
        ---
        
        """
        import utils_sys as us

        if domain: 
            output_path = us.resolve_analysis_path( domain )   # can be a domain/project name or a project_path (path to the data)
        else: 
            output_path = us.resolve_analysis_path(project_path if project_path else ProjectPath)

        # save 2d array to .npz format? 
        if format_ == 'csv': 
            df = DataFrame(A)  # format: rows <- latent factors, cols <- attributes 

            if U is not None: 
                # then this is assumed to be a similarity matrix or distance matrix
                assert len(U) == A.shape[0] == A.shape[1], "Input A is not a square matrix: dim(A): {dim} but len(U): {n}".format(dim=A.shape, n=len(U))
                df.columns = df.index = U
            
            ext = 'csv'
            if index is None or index < 0: # no index
                file_name = '{prefix}.M-{dataset}-{suffix}.{ext}'.format(prefix=name, dataset=dset_id, suffix=file_type, ext=ext)
            else: 
                file_name = '{prefix}.M-{dataset}-{suffix}-{index}.{ext}'.format(prefix=name, dataset=dset_id, suffix=file_type, index=index, ext=ext)

            if verbose: print('(save_array) Saving input array of dim: {0} ...'.format(A.shape))
            df.to_csv(os.path.join(output_path, file_name), sep=sep, index=False, header=True)  # MFEnsemble.data_dir()
        else: 
            raise NotImplementedError

    @staticmethod
    def load_array(name='user-sim', dset_id='generic', file_type='posterior', sep=',', format_='csv', domain='', project_path='', index=-1):
        import pandas as pd
      
        if domain: 
            output_path = us.resolve_analysis_path( domain )   # can be a domain/project name or a project_path (path to the data)
        else: 
            output_path = us.resolve_analysis_path(project_path if project_path else ProjectPath)

        if index is None or index < 0: # no index
            file_name = '{prefix}.M-{dataset}-{suffix}.{ext}'.format(path=output_path, prefix=name, dataset=dset_id, suffix=file_type, ext=ext)
        else: 
            file_name = '{prefix}.M-{dataset}-{suffix}-{index}.{ext}'.format(path=output_path, prefix=name, dataset=dset_id, suffix=file_type, index=index, ext=ext)

        fpath = os.path.join(output_path, file_name)

        A = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
        if kargs.get('verbose', True): print('(load_array) Loading 2D array of dim: {0} ...'.format(A.shape))

        return A.values

    @staticmethod
    def save_model(df, name, file_type, **kargs):
        """
        Save stacker performance dataframe, similarity matrix, etc. in the analysis directory.

        Params
        ------


        """
        import utils_sys as us

        CF_method = 'wmf'
        file_params = kargs.get('params', {})
        dset_id = CF_method if not kargs else MFEnsemble.get_dset_id(method=CF_method, params=file_params)  # for the moment, assign default 'wmf' to the 'method' argument
        print('(save_model) dataset ID: {id}'.format(id=dset_id))

        output_path = us.resolve_analysis_path( kargs.get('path', ProjectPath) )   # can be a domain/project name or a project_path (path to the data)
        
        # path and file naming template
        # '{path}/analysis/{stacker}.S-{dataset}-{suffix}.csv'.format(path=path, 
        #                 stacker=aggr, dataset=dset_id, suffix=file_type)
        ext = kargs.get('ext', 'csv')
        file_name = '{prefix}.M-{dataset}-{suffix}.{ext}'.format(path=output_path, prefix=name, dataset=dset_id, suffix=file_type, ext=ext)        
            
        print('(save_model) Saving model file | dtype: {dtype}, model name: {name}, data/method ID: {id} | output path:\n{path}\n'.format(dtype=file_type, name=name, 
                    id=dset_id, path=output_path))
                
        df.to_csv(os.path.join(output_path, file_name), index = False)
        return 

    @staticmethod
    def save_data(X, fold, params={}, indices=[], dset_id='', dtype='posterior', subsampling=False, verbose=1, base_method='wmf'): # module: tset
        # MFEnsemble.meta_keys: ['conf_measure', 'policy', 'supervised', 'masked']
        assert len(X) >= 2 and len(X) <= 3, "X: (Th, L_test, U, Ix)"
        T, L, *rest = X
        assert T.shape[1] == len(L)
        
        U = []
        if len(rest) > 0: 
            U, *indices = rest

        if len(dset_id) == 0:  
            assert len(params) > 0, "Both dataset ID (dset_id) and parameters are not given."
            
            # meta_params = {k: params.get(k, MFEnsemble.default[k]) for k in MFEnsemble.meta_keys}  # meta_params are a subset of params that do not go into model selection loop
            # => meta_params are specified in and copied from params
            dset_id = MFEnsemble.get_dset_id(method=base_method, params=params, meta_params={}) # meta: extra info needed to differentiate training data
        
        # test
        if verbose: print('(save_meta_tset) dim(T): {dim} | dset_id: {id}'.format(dim=T.shape, id=dset_id))

        # >>> (Rh, Th) represents new rating scores or not? 
        isRating = True # not params['policy_opt'].startswith('pref') or params['replace']
        # if verbose > 0: print('... Data represent new rating scores? {0}'.format(isRating))

        save_data(T, L=L, indices=indices, fold=fold, method=dset_id, subsampling=subsampling, verify=True, U=U, is_rating=isRating, dtype=dtype)
        return
    
    @staticmethod
    def load_data(fold, params={}, dset_id='', dtype='posterior', verbose=1, base_method='wmf', project_path=''):
        import pandas as pd
        if not project_path: project_path = System.projectPath

        if len(dset_id) == 0:  
            assert len(params) > 0, "Both dataset ID (dset_id) and parameters are not given."
            
            # meta_params = {k: params.get(k, MFEnsemble.default[k]) for k in MFEnsemble.meta_keys}  # meta_params are a subset of params that do not go into model selection loop
            # => meta_params are specified in and copied from params
            dset_id = MFEnsemble.get_dset_id(method=base_method, params=params, meta_params={}) # meta: extra info needed to differentiate training data
        df_path = '{prefix}/{method}-{dtype}-{index}.csv.gz'.format(prefix=project_path, method=dset_id, dtype=dtype, index=fold)
        assert os.path.exists(df_path), "Invalid data path: {0}".format(df_path)
        return pd.read_csv(df_path, index_col = [0, 1], compression = 'gzip')

    @staticmethod
    def make_prediction_dataframe(y_pred, y_label, method, index):
        return DataFrame({'prediction':y_pred,'label':y_label, 'method': method, 'fold': index}, index=range(len(y_pred))) 

### end class MFEnsemble

############################################################
# ... Predicates 

def is_user_dim(cf_dim, setting=-1): 
    # given a filtering dimension key word, extracted from dset_id, is it in the classifier dimension? 
    is_cls = cf_dim.startswith(('u', 'cl'))  # user/classifier
    if setting > 0: 
        return is_cls and setting % 2 == 0   # e.g. cases in 2, 4, 8, 10
    return is_cls
def is_item_dim(cf_dim, setting=-1): 
    is_sample = cf_dim.startswith(('i', 'sa', 'da'))   # item/sample/data
    if setting > 0: 
        return is_sample and setting % 2 == 1   # e.g. cases in 1, 3, 5, 7
    return is_sample

############################################################

# utils_cf
def combiner(Th, weights=None, aggregate_func=np.mean, axis=0, **kargs): 
    """

    Use 
    ---
    1. Th contains reconstructed probabilities 

    2. Th contains preference scores 

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
    if isinstance(aggregate_func, str): 
        if aggregate_func.startswith('pref'): 
            assert 'T' in kargs and kargs['T'] is not None, "Missing T"
            T = kargs['T']
            return combiner_pref(Th, T)  # return predictive scores
        elif aggregate_func in ('mean', 'av'): # mean or average 
            if weights is not None: 
                print('(combiner) aggregate_func: mean | using predict_by_importance_weights() | n(zeros):{}'.format(np.sum(weights==0)))
                return predict_by_importance_weights(Th, weights, aggregate_func='mean', fallback_on_low_weight=True, min_weight=0.1)
            return np.mean(Th, axis=axis)  # e.g. mean prediction of users/classifiers
        elif aggregate_func == 'median':
            if weights is not None: 
                return predict_by_importance_weights(Th, weights, aggregate_func='median', fallback_on_low_weight=True, min_weight=0.1)
            return np.median(Th, axis=axis) 
        else: 
            raise NotImplementedError
    else: 
        assert hasattr(aggregate_func, '__call__')
        predictions = aggregate_func(Th, axis=axis)  # e.g. mean prediction of users/classifiers

    return predictions

def combiner_pref(Th, T): 
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
            print('(combiner_pref) None of the BP prediction are reliabel? sum(pref)=0 at data #%d' % i)
            y_score = np.mean(prob)
            n_zero_pref += 1 

        predictions.append(y_score)
    if n_zero_pref > 0: print('(combiner_pref) Found %d instances with preference score = 0!' % n_zero_pref)
    return np.array(predictions)

def pairwise_similarity(ratings, kind='user', epsilon=1e-9):
    return uknn.pairwise_similarity(ratings, kind=kind)

def corr(A,B, axis=1):
    return uknn.corr(A, B, axis=axis)
def eval_correlation(R, kind='user', epsilon=1e-9, to_distance=False): 
    return uknn.eval_correlation(R, kind=kind, to_distance=to_distance)

# [todo]
def eval_similarity(R, kind='user', centering=False, epsilon=1e-9): # 
    return uknn.eval_similarity(R, kind=kind, centering=centering)

def transfer_factor_by_similarity(X, F, topk=1): 
    R, T = X  # transfer learned factor (P and Q) from R to T
    P, Q = F  # latent factors

    k = topk
    n_users, n_items = P.shape[0], Q.shape[0]  # user vectors and item vectors for R
    n_items_test = T.shape[1]
    n_factors = P.shape[1]
    assert n_factors == Q.shape[1]

    # similarity between T(j) and R(i)
    S = eval_cross_similarity(T, R, kind='item', epsilon=1e-9, unbiased=True)
    # ... S: n_items_R by n_items_T 

    Pt, Qt = P, np.zeros((n_items_test, n_factors))  # assuming Pt = P, i.e. user/classifier vectors stay the same in T
    for j in range(S.shape[1]): # foreach item (index) in T
        top_k_R = tuple([np.argsort(S[:,j])[:-k-1:-1]])  # find top k items in R most similar to jth item in T (column vector)
        # print('... (verify) col({j}) | min: {m}, max: {M} | topk: {t}'.format(j=j, m=np.min(S[:,j]), M=np.max(S[:,j]), t=S[:, j][top_k_R]))
        
        w = S[:, j][top_k_R]  # weights by degrees of similarity 
        # print('> weights of the top k items in R: {w}'.format(w=w))

        # Qr = Q[top_k_R, :]  # select row vectors
        Qt[j] = np.average(Q[top_k_R], axis=0, weights=w) # weigthed average of the topk most similar item's vectors
        assert len(Qt[j]) == n_factors
    return (Pt, Qt)

def transfer_confidence_by_similarity(X, C, topk=1, **kargs):
    """

    Memo
    ----
    1. need a confidence measure e.g. brier score
    2. confidence2D()

    """
    import scipy.sparse as sparse

    R, T = X 
    Cr = C
    assert Cr.shape == R.shape
    null_marker = kargs.get('fill', 0)

    S = eval_cross_similarity(T, R, kind='item', epsilon=1e-9, unbiased=True) # compute item-wise similarity between T(j) and R(i) ... 
    #   ... where i, j are column indices 
    
    k = topk
    Ct = np.zeros(T.shape)
    for j in range(S.shape[1]): # foreach T(j)
        top_k_R = tuple([np.argsort(S[:,j])[:-k-1:-1]])  # find top k items in R most similar to jth item in T (column vector)
        w = S[:, j][top_k_R]  # weights of most similar R(i)

        # e.g. np.average(C[:, [1, 2]], axis=1, weights=[0.3, 0.7])
        Ct[:, j] = np.average(Cr[:, top_k_R], axis=1, weights=w)  # T(j)'s c-score is w-average of most similar R(i)'s  

    n_zeros = n_nonzeros = -1
    if kargs.get('sparse', True): 
        Ct= sparse.csr_matrix(Ct)

        # [test]
        n_nonzeros = sparse.csr_matrix.count_nonzero(Ct)
        n_zeros = Ct.shape[0] * Ct.shape[1] - n_nonzeros
        # assert n_zeros > 0

        print('... (verify) Ct being converted to sparse matrix.')
    else: 
        n_zeros = np.sum(Ct==null_marker)
        n_nonzeros = Ct.shape[0] * Ct.shape[1] - n_zeros 
    
    print('... Ct_h |  dim(Ct_h): {0}, n_zeros: {1} vs nonzeros: {2} (masked ratio={3})'.format(Ct.shape, 
            n_zeros, n_nonzeros, n_zeros/(Ct.shape[0] * Ct.shape[1] + 0.0)))   

    return Ct

def eval_similarity_by_latent_factors(A, epsilon=1e-9):
    """
    Compute pairwise similarity between "rows" of A (i.e. assuming A is in row vector format). 

    Use
    ---
    1. Pass either user vectors (P) or item vectors (Q)

    Memo
    ----
    1. If dot product preceeds normalization, the resulting similarity matrix is not symmetric
       and cannot be used for hierarchical clustering

    """
    return uknn.eval_similarity_by_latent_factors(A, epsilon=epsilon)

def to_affinity(A, sim_func=None, sig=0.5, verify=False):
    if sim_func is None: sim_func = eval_similarity_by_latent_factors  # similarity falls in [0, 1]

    S = sim_func(A)
    if verify: 
        ep = 1e-9
        low, high = np.min(S), np.max(S)
        assert abs(low-0.0) <= ep 
        assert abs(high-1.0) <= ep

    # to distance
    S = 1. - S

    # now to similarity measure that falls within [0, 1]
    S = np.exp(- S ** 2 / (2. * sig ** 2))

    return S  # if A is symmetric, then S is symmetric

def eval_cross_similarity(T, R, kind='item', unbiased=True, epsilon=1e-9): 
    return uknn.eval_cross_similarity(T, R, kind=kind, unbiased=unbiased)

def rank(X, **kargs):
    from scipy.stats import rankdata

    method = kargs.get('method', 'average')

    Rm = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Rm[i] = rankdata(X[i, :], method)
    return Rm
    
def rank2(X, L, **kargs): 
    """

    Memo
    ----
    1. argsort in descending order 

       np.argsort(Lp)[::-1]

    2. rankdata
       the larger the value, the higher the rank value 

       alternatively, use argsort twice. 

    """
    from scipy.stats import rankdata

    W = np.ones(X.shape)
    Wv = np.ones(X.shape)
    min_class = kargs.get('min_class', 1)

    if len(L) > 0: 
        pos_idx = np.where(L==1)[0]   #
        neg_idx = np.where(L==0)[0]
        n_neg = len(neg_idx)
        greater_is_better = True if min_class == 1 else False

        m = M = u = 0 
        t = 2
        for i in range(X.shape[0]):  # foreach user index 
            W[i] = rankdata(X[i])
            # if i == t:
            #     m, M = np.min(W[i]), np.max(W[i]) 
            #     u = len(np.unique(X[i]))

            # W[i] = W[i] - n_neg
            W[i][neg_idx] = 1
            # offset = np.min(W[i][pos_idx])

            if i == t: 
                print('(rank) W[{}]: \n{}\n'.format(i, W[i][:100]))

        # [test]
        # print('... max(W[t]): {}, min(W[t]): {} | n(neg): {} | n(uniq): {}'.format(np.max(W[t]), np.min(W[t]), n_neg, u))
        # sys.exit(0)

        for j in range(X.shape[1]): # foreach item index 
            if L[j] == 1: 
                Wv[:, j] = rankdata(X[:, j])
            else: 
                pass
    else: 
        raise ValueError("(estimated) labels must be given!")
        # for i in range(X.shape[0]):  # foreach user index 
        #     W[i] = rankdata(X[i])
        #     if i == 2: 
        #         print('(rank) W[{}]: \n{}\n'.format(i, W[i][:100]))

        # for j in range(X.shape[1]): # foreach item index 
        #     if L[j] == 1: 
        #         Wv[:, j] = rankdata(X[:, j])
        #     else: 
        #         pass

    # [test]
    ref = 10
    n = np.sum(W > ref)
    print('(rank) number of entries with rank (> {}): {} | avg/class: {}'.format(ref, n, n/(X.shape[0]+0.0) ))
    # print('... shape(W * Wv): {}'.format( (W * Wv).shape ))

    return W * Wv

def confidence2D(X, L, mode='brier', topk=(0, 0), scoring=None, greater_is_better=True, **kargs):
    """

    **kargs
    -------
        policy_threshold: 
            'prior' (when L is given), 'ratio' (L is empty), 'fmax', ...
        p_threshold: 
            pre-computed probability thresholds (one for each user/classifier)
        ratio_small_class: 
            the ratio of minority class, or a conservative estimate of it; required when L is not available
        pos_label


    """
    def verify(Wu, Wi, n=10): 
        print('(confidence2D) confidence weights (mode=%s)\n... Wu:\n%s\n' % (mode, Wu[:n]))
        print('... Wi:\n%s\n' % Wi[:n])
        Wt = np.outer(Wu[:n], Wi[:n])
        print('... W:\n%s\n' % Wt)

    # configure top k 
    topk_users, topk_items = topk

    conf_user, conf_item = kargs.get('conf_user', True), kargs.get('conf_item', False)  
    # ... this has no effect on rank() method

    n_users, n_items = X.shape[0], X.shape[1]
    assert n_items == len(L)

    # policy_threshold = kargs.get('policy_threshold', '')  # set to '' to automatically determine the policy based on the value of L
    p_threshold = kargs.get('p_threshold', [])
    verbose = kargs.get('verbose', 1)
    W = Wu = Wi = None

    ############################################################
    if len(p_threshold) > 0:
        # if p_threshold is given, we can use it to predict the labeling for each data point, from which ... 
        # ... we then associate a confidence measure for each prediction made by the base predictors
        Wi = confidence_pointwise_ensemble_prediction(X, L, p_threshold=p_threshold, mode='item')
        
        # [test]
        if verbose: 
            print("(confidence2D) item/data-wise confidence score distributions ... ")
            test_items = np.random.choice(range(X.shape[1]), 5) 
            for j in test_items: 
                print('... data [{}] | {}'.format(j, Wi[:5]))
    ############################################################
 
    if mode.startswith('b' ): # brier
        Wu = confidence_brier(X, L, mode='user', topk=topk_users)

        # now turn 1-D weight vector into a 2D column vector
        # np.repeat(W[:, np.newaxis], len(labels), axis=1)  # W[:, np.newaxis] => 2D col vec of W => repeats n_items of time
        
        # now compute item-label correlation
        if Wi is None: Wi = confidence_brier(X, L, mode='item', topk=topk_items)

    elif mode == 'rank': # rank 
        
        W = rank(X, L, **kargs)
        Wu = Wi = None

    elif mode == 'fmax':
        Wu = confidence_performance(X, L, mode='user', metric=mode)

        if Wi is None: Wi = np.ones(n_items)   # or create a data pair [todo]

    elif mode.startswith('u'): # uniform 
        Wu = np.ones(n_users)
        Wi = np.ones(n_items) 
        
        # W = np.outer(Wu, Wi)  # n_users by n_items 
    elif mode.startswith('r'): # ratio  
        p_threshold = kargs.get('p_threshold', [])
        if len(p_threshold) == 0: 
            p_threshold = estimateProbThresholds(X, L=L, pos_label=kargs.get('pos_label', 1), policy=policy_threshold, ratio_small_class=kargs.get('ratio_small_class', 0.01))
        else: 
            assert len(p_threshold) == X.shape[0]

        if len(L) == 0: 
            L = estimateLabels(X, pos_label=kargs.get('pos_label', 1), ratio_small_class=kargs.get('ratio_small_class', 0.01))

        Wu = confidence_ratio(X, L, mode='user', topk=topk_users, p_threshold=p_threshold) # T=None, p_threshold, delta
        Wi = confidence_ratio(X, L, mode='item', topk=topk_items, p_threshold=p_threshold)
    elif mode.startswith('corr'): 
        # todo: confidence_corr() does not seem to work for 'items'
        # raise NotImplementedError("Mode: %s is not available yet." % mode)
        Wu = confidence_corr(X, L, mode='user', topk=topk_users)

        if Wi is None: Wi = np.ones(n_items)  # item mode is undefined, assume equal
    else:   # mode: others => use a customized function to measure confidence level
        assert scoring is not None and hasattr(scoring, '__call__')

        ## use scoring function and the generic function confidence 
        Wu = confidence(X, L, mode='user', topk=topk_users, scoring=scoring, greater_is_better=greater_is_better)
        Wi = confidence(X, L, mode='item', topk=topk_items, scoring=scoring, greater_is_better=greater_is_better)
    
    if verbose: 
        if mode != 'rank': verify(Wu, Wi, n=10)

    ### return format 
    if mode.startswith('rank'): 
        return W  # return W directly since rank scores do not compute Wu and Wi separately

    if Wi is None: 
        if verbose: print(f'(confidence2D) Wi undefined under mode: {mode} => Effectively only use Wu to compute Cui')
        Wi = np.ones(n_items)

    W = np.outer(Wu, Wi)  # n_users by n_items
    assert W.shape == X.shape, "dim(W): %s while dim(R): %s" % (str(W.shape), str(X.shape))

    return W

## several different confidence matrices go here <<< 
def confidence_distance(R, labels, T=None, mode='label', topk=0, scoring=None, greater_is_better=True): 
    # confidence is evaluated "point-wise"
    pass

def confidence_ratio(R, labels, T=None, mode='user', topk=0, p_threshold=[]): 
    """
    Confidence score based on ratio of correct predictions. 

    Memo
    ----
    1. brier score does not result in a diversity of confidence score
    """
    # from sklearn.preprocessing import normalize

    p_threshold = kargs.get('p_threshold', None)
    if p_threshold is not None: 
        if isinstance(p_threshold, float): 
            p_threshold = np.repeat(p_threshold, R.shape[0])
        else: 
            assert hasattr(p_threshold, '__iter__')

    # p_th = kargs.get('p_threshold', 0.5)
    # delta = kargs.get('delta', 0.0)  # ensure that the probability is rigorous, high for positive, low for negative

    ## instead of using brier score or user-oriented scoring functions, use a metric that reflects prediction quality
    # for i in range(R.shape[1]): 
    #     W[i] = scoring_func( R[:, i], np.repeat(labels[i], R.shape[0]))   

    n_users, n_items = R.shape[0], R.shape[1]
    # pos_th = min(1.0, p_th+delta)
    # neg_th = max(0.0, p_th-delta)
    
    if mode.startswith('u'): 
        W = np.zeros(R.shape[0])  # weights over users/classifiers

        for i in range(R.shape[0]): 
            nTP = np.sum( (R[i, :] >= p_threshold[i]) & (labels == 1) )
            nTN = np.sum( (R[i, :] < p_threshold[i]) & (labels == 0) )
            n_correct = nTP + nTN
            W[i] = n_correct/(n_items+0.)

    elif mode.startswith('i'): 
        W = np.zeros(R.shape[1])
        for i in range(R.shape[1]):
            nTP = np.sum( (R[:, i] >= p_threshold) & (labels[i] == 1) )
            nTN = np.sum( (R[:, i] < p_threshold) & (labels[i] == 0) )
            n_correct = nTP + nTN
            W[i] = n_correct/(n_users+0.)

    # suppress the weights except for the topk 
    if topk: 
        for i in range(W.shape[0]):
            if not i in np.argsort(W)[::-1][:topk]:  # assume greater is better
               W[i] = 0

    return W

def confidence_corr(R, labels, mode='label', topk=0, ep=1e-9): 
    """
    Confidence score defined by the correlation between probability predictions and true labels. 

    Memo
    ----
    1. This metric is only applicable to user/classifier vs true labels 

       the problem with np.repeat(labels[i], R.shape[0])+ep is that it is a constant vector 
       => cannot define correlation in the item-label case 

    2. Confidence is a function of (user, item), when used in the ensemble learning setting ... 

    """ 
    # W = np.zeros((1, R.shape[0]))  # [note] if we use 2D, 1-by-k array, later W.dot(T) will also be a 2-D
  
    if mode.startswith(('l', 'u')) : # user-label correlation; i.e. compare classifier predictions to labels
        W = np.zeros(R.shape[0])  # weights over users/classifiers
        for i in range(R.shape[0]): 
            # W[0, i] = np.corrcoef(R[i, :], labels)[0, 1] # corrcoef(R[i, :], labels)
            print('(test) dim(R):{0}, n(labels):{1}'.format(R.shape, len(labels)))
            W[i] = np.corrcoef(R[i, :], labels)[0, 1] # [0, 1] corr between 1st and 2nd variable
    elif mode.startswith('i'):  # item-label corr
        # W = np.zeros(R.shape[1])
        # for i in range(R.shape[1]):

        #     # cannot generally compute correlation with a constant vector, need to pertube it
        #     W[i] = np.corrcoef( R[:, i], np.repeat(labels[i], R.shape[0])+ep )[0, 1] 
        raise ValueError("Cannot compute correlation with a constant vector (the label of the same item)")
    else: 
        raise NotImplementedError("Mode: %s is not available yet." % mode)

    # Suppress the weights except for the topk 
    if topk: 
        for i in range(W.shape[0]):
            if not i in np.argsort(W)[::-1][:topk]:
               W[i] = 0.

    # T 
    # confidence can also depend on items/data
    # a. is this example similar to those that were predicted with relatively higher confidence? 
    #    
    return W

def confidence_pointwise_ensemble_prediction(R, labels, p_threshold, mode='item', suppress_majority=False, multiple=-1): 
    """
    Compute item-wise confidence score

    Parameters 
    ----------
    R: Rating matrix 
    p_threshold: A vector of probability thresholds; each base predictor has a threshold, above which it predicts positive 
                 and below which, it predicts negative
    multiple: The factor by which the confidence score for majority class is discounted


    """

    assert len(p_threshold) == R.shape[0]

    if suppress_majority: 
        if multiple < 0 or multiple is None: 
            ret = classPrior(labels, labels=[0, 1], ratio_ref=0.1, verbose=False)

            multiple = ret['r_max_to_min'] # ratio between sample size of majority (max class) and that of the minority class (min class)
            # ... the more skewed the class distribution, the larger the max-to-min ratio and hence the larger the multiple
            # ... the larger this multiple, the less the confidence score given to a given data point (j)

    n_users = R.shape[0]
    Wi = np.zeros(R.shape[1]) # `i` in Wi denotes item, which corresponds to data point
    for j in range(R.shape[1]): # foreach data point
        
        if labels[j] == 1: # true positive (TP)
            n_tp = np.sum( R[:, j] >= p_threshold )

            # Confidence level is defined as the fraction of base predictors making TP-prediction on this data point
            Wi[j] = n_tp/(n_users+0.0) 

        else: # true negative (TN)
            n_tn = np.sum( R[:, j] < p_threshold )

            # Confidence level is defined as the fraction of base predictors making TP-prediction on this data point
            Wi[j] = n_tn/(n_users+0.0)
            
            if suppress_majority: 
                if multiple > 1: 
                    Wi[j] = Wi[j] * (1./multiple) # 
                else: 
                    assert multiple == 0, "multiple cannot be in (0, 1)"
                    Wi[j] = 0
                
    return Wi

def confidence_performance(R, labels, mode='user', metric='fmax', topk=0, marker=0): 
    # from sklearn.metrics import brier_score_loss, roc_auc_score

    labels = np.array(labels)
    W = np.zeros(R.shape[0])  # weights over users/classifiers

    if metric == 'fmax': 
        for i in range(R.shape[0]):
            score, threshold = common.fmax_score_threshold(labels, R[i], beta = 1.0, pos_label = 1)
            W[i] = score
    elif metric in ('auc', 'roc'):
        for i in range(R.shape[0]):
            score = roc_auc_score(labels, R[i], beta = 1.0, pos_label = 1)
            W[i] = score
    else: 
        raise NotImplementedError

    # item-wise performance is undefined

    return W

def confidence_brier(R, labels, mode='label', topk=0, p_threshold=[], masked_only=False, marker=0): 
    # value: between [0, 1]; a cost function so the smaller the better
    # from sklearn.metrics import brier_score_loss
    
    # confidence(R, labels, mode=mode, topk=topk, scoring=brier_score_loss, greater_is_better=False)
    # if p_threshold is not None: 
    #     if isinstance(p_threshold, float): 
    #         p_threshold = np.repeat(p_threshold, R.shape[0])
    #     else: 
    #         assert hasattr(p_threshold, '__iter__')
    labels = np.array(labels)

    if mode.startswith(('l', 'u')) : # user-label correlation; i.e. compare classifier predictions to labels
        W = np.zeros(R.shape[0])  # weights over users/classifiers

        if masked_only: 
            # only consider entries where correct predictions are made 
            for i in range(R.shape[0]):
                valid = np.where(R[i, :] != marker)

                # format: y_true, y_prob
                W[i] = brier_score_loss(labels[valid], R[i, :][valid])  # convert to confidence, the higher the better 
        else: 
            for i in range(R.shape[0]): 
                W[i] = brier_score_loss(labels, R[i, :])  # convert to confidence, the higher the better
    elif mode.startswith('i'):  # item-label corr
        W = np.zeros(R.shape[1])

        if masked_only: 
            # only consider entries where correct predictions are made 
            for j in range(R.shape[1]):
                valid = np.where(R[:, j] != marker)
                r_valid = R[:, j][valid]
                W[j] = brier_score_loss(np.repeat(labels[j], len(r_valid)), r_valid)  # convert to confidence, the higher the better 
        else: 
            for j in range(R.shape[1]): # foreach item/datum
                W[j] = brier_score_loss(np.repeat(labels[j], R.shape[0]),  R[:, j]) 
                # ... each datum has the same confidence score wrt the entire ensemble
    else: 
        raise NotImplementedError("Mode: %s is not available yet." % mode)

    # The smaller the brier score, the higher the confidence
    W = 1. - W

    # Suppress the weights except for the topk 
    if topk: 
        for i in range(W.shape[0]):
            if not i in np.argsort(W)[::-1][:topk]:  # assume greater is better
               W[i] = 0.

    return W

# general
def confidence(R, labels, T=None, mode='label', topk=0, scoring=None, greater_is_better=True): 
    # from sklearn.metrics import brier_score_loss  # MSE, the lower the better
    # configure top k 

    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

    # scoring can be a name or a function [todo]
    scoring_func = scoring
    if scoring is None: scoring_func = brier_score_loss
    assert hasattr(scoring_func, '__call__')

    # auto correct 
    if scoring_func.__name__.startswith(('brier', )): greater_is_better = False

    if mode.startswith(('l', 'u')) : # user-label correlation; i.e. compare classifier predictions to labels
        W = np.zeros(R.shape[0])  # weights over users/classifiers
        for i in range(R.shape[0]): 

            W[i] = scoring_func(R[i, :], labels)  # convert to confidence, the higher the better
    elif mode.startswith('i'):  # item-label corr
        W = np.zeros(R.shape[1])
        for i in range(R.shape[1]):

            # 
            W[i] = scoring_func( R[:, i], np.repeat(labels[i], R.shape[0])) 
    else: 
        raise NotImplementedError("Mode: %s is not available yet." % mode)

    max_w = np.max(W)
    min_w = np.min(W)

    tNormalize = False
    if max_w > 1.0 or min_w < 0.0: # an arbitrary score
        tNormalize = True
    if kargs.get('normalize', False): tNormalize = True

    # ensure that each weight is in [0, 1]
    if tNormalize: 
        # W.reshape(1, -1) => 1-by-n row vector in a 2D array ...
        # ... take [0], because we want W to be a 1D array here
        W = normalize(W.reshape(1, -1))[0]   
    
    ## convert a 'arbitrary' scores into a confidence level, the higher the better
    # cost function: smaller is better 
    if not greater_is_better: 
        if tNormalize: 
            W = 1.0 - W
        else: 
            W = -np.log(W)

    # [test]
    # print('(test): W0:\n%s\n' % W)

    # suppress the weights except for the topk 
    if topk: 
        for i in range(W.shape[0]):
            if not i in np.argsort(W)[::-1][:topk]:  # assume greater is better
               W[i] = 0.

    # T 
    # confidence can also depend on items/data
    # a. is this example similar to those that were predicted with relatively higher confidence? 
    #    
    return W

def get_weighted_average(W, T, topk=None): 
    """
    Get a weighted-averate prediction on the rating matrix, where 
    each user/classifier is associated with a prediction vector as a row vector in T
    """

    if topk: 
        topk_corr = np.argsort(W)[::-1][:topk]
        # only look at the rows that correspond to top k users/classifiers
        pred = W[topk_corr].dot(T[topk_corr, :]) / np.array([np.abs(W[topk_corr]).sum()])
    else: 
        pred = W.dot(T) / np.array([np.abs(W).sum()])

    return pred

# common 
def save_reconstructed_probs(Th, L, fold, method, **kargs):
    isRating = True # for the moment, we do not save preference scores but only (proba) ratings even if policy = 'preference'   ... 05.31

    # dataframe index 
    indices = kargs.pop('indices', [])
    Tx = None
    if len(indices) > 0: 
        Tx = indices  
        print('... (verify) unpacking indices:\n... Tx: {tx}'.format(tx=Tx.names))

    if isRating: # if (Rh, Th) do not represent rating scores, then method must have related keyword 
        if Tx is not None: kargs['index'] = Tx
        save_reconstructed_data(Th, L, fold, method, **kargs) 
    else: 
        update_msg = "For the moment, we do not save preference scores but only (proba) ratings even if policy = 'preference'"
        raise NotImplementedError(update_msg)

    return

def save_preference_training_data(R, L_train, fold, method, **kargs):
    import pandas as pd
 
    tRandomSubsampling = kargs.get('subsampling', False)
    if tRandomSubsampling: 
        # in subsampling mode, we cannot use read_fold() to recover the labeling
        index = kargs.get('index', None)
        if index is None: 
            index = pd.MultiIndex.from_tuples([(i, L_train[i]) for i in range(len(L_train))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new pref-TRAIN set (method: %s)' % method)
        assert hasattr(R, '__iter__') and len(R) == 2, "Need both rating and preference matrices"
        Rt, R_pref = R

        assert 'U' in kargs
        augmented_cols = list(U) + ['t_%s' % c for c in U]
    else: 
        R_pref = R
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
        index = train_df.index
        # level-1 training data: validation-<fold>.csv.gz

        Rt = train_df.values  # R: users vs items but train_df has users as columns
        augmented_cols = list(train_df.columns) + ['t_%s' % c for c in train_df.columns]
    # T = test_df.values.T 
    # U = train_df.columns.values

    # R, R_pref = R_pair
    # Ra = np.vstack((R, R_pref))  # users/classifier followed by indicators
    augmented_cols = list(train_df.columns) + ['t_%s' % c for c in train_df.columns]
    df = DataFrame(np.hstack( (Rt, R_pref.T) ), index=index, columns=augmented_cols)  # datasink convention> rows: data points, columns: classifiers

    # save 
    assert method.find('pref') > 0, "Questionable method: %s" % method

    # >>> cannot use 'fold' as an index (because then the resulting set may not have continous indices)
    df_path = '%s/%s-validation-%s.csv.gz' % (ProjectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving preference-augmented TRAIN set (dim: {0}) to:\n{path}\n'.format(df.shape, path=df_path))

    return

def save_preference_test_data(T, L_test, fold, method, **kargs): 
    import pandas as pd 
    if kargs.get('subsampling', False): 
        # in subsampling mode, we cannot use read_fold() to recover the labeling
        index = kargs.get('index', None)
        if index is None: 
            index = pd.MultiIndex.from_tuples([(i, L_test[i]) for i in range(len(L_test))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new pref-TEST set (method: %s)' % method)
        assert hasattr(T, '__iter__') and len(T) == 2, "Need both rating and preference matrices"
        Tt, T_pref = T

        assert 'U' in kargs
        augmented_cols = list(U) + ['t_%s' % c for c in U]

    else: 
        T_pref = T
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
        index = test_df.index
        
        Tt = test_df.values 
        augmented_cols = list(test_df.columns) + ['t_%s' % c for c in test_df.columns]

    df = DataFrame(np.hstack( (Tt, T_pref.T) ), index=index, columns=augmented_cols)  # datasink convention> rows: data points, columns: classifiers
    print('(save) new test set dim: {0} vs original: {1}'.format(df.shape, test_df.shape))

    df_path = '%s/%s-predictions-%s.csv.gz' % (ProjectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving preference-augmented TEST set (dim: {dim}) to:\n{path}\n'.format(dim=df.shape, path=df_path))

    return

# subsumed by save_data()
def save_reconstructed_training_data(Rh, L_train, fold, method, **kargs):
    """

    Memo
    ----
    1. MultiIndex: 
       
       L_train
       index = [(i, L_train[i]) for i in range(len(L_train))] 
       pd.MultiIndex.from_tuples(index)

    """
    import pandas as pd
    tRandomSubsampling = kargs.get('subsampling', False)
    if tRandomSubsampling: 
        index = kargs.get('index', None)
        if index is None: 
            # in subsampling mode, we cannot use read_fold() to recover the labeling
            index = pd.MultiIndex.from_tuples([(i, L_train[i]) for i in range(len(L_train))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new TRAIN set (method: %s)' % method)
        else: 
            # if 'names' is not part of the index attribute => FrozenList([None])
            assert index.names[0] is not None

        if not 'U' in kargs: 
            train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, 0) # [todo] single out this part
            cols = train_df.columns
        else: 
            cols = kargs['U']
    else:
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, 0) # [todo] single out this part
        index = train_df.index
        cols = train_df.columns 

        # verify 
        if kargs.get('verify', True):
            # train_df = train_df.reset_index() # convert multilevel index to flat index
            labels = train_df.index.get_level_values('label').values # ground truth labels
            assert all(L_train == labels)

            if 'U' in kargs: 
                assert all(kargs['U'] == train_df.columns.values)
    
    # level-1 training data: validation-<fold>.csv.gz
    df = DataFrame(Rh.T, index=index, columns=cols)  # datasink convention> rows: data points, columns: classifiers

    # save 
    df_path = '%s/%s-validation-%s.csv.gz' % (ProjectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving new training set (dim: {dim}) to:\n{path}\n'.format(dim=df.shape, path=df_path))

    return

# subsumed by save_data()
def save_reconstructed_test_data(Th, L_test, fold, method, **kargs):
    import pandas as pd
    tRandomSubsampling = kargs.get('subsampling', False)
    if tRandomSubsampling: 
        index = kargs.get('index', None)
        if index is None: 
            # in subsampling mode, we cannot use read_fold() to recover the labeling
            index = pd.MultiIndex.from_tuples([(i, L_test[i]) for i in range(len(L_test))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new TEST set (method: %s)' % method)
        else: 
            # if 'names' is not part of the index attribute => FrozenList([None])
            assert index.names[0] is not None

        if not 'U' in kargs: 
            train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, 0) # [todo] single out this part
            cols = test_df.columns
        else: 
            cols = kargs['U']
    else:
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
        index = test_df.index
        cols = test_df.columns

        # verify 
        if kargs.get('verify', True):
            # test_df = test_df.reset_index() # convert multilevel index to flat index
            labels = test_df.index.get_level_values('label').values
            assert all(L_test == labels)

            if 'U' in kargs: 
                assert all(kargs['U'] == test_df.columns.values)

    # level-1 training data: validation-<fold>.csv.gz
    df = DataFrame(Th.T, index=index, columns=cols)  # datasink convention> rows: data points, columns: classifiers
    
    # save   # note that regular level-1 test data has the name: predictions-<fold>.csv.gz
    df_path = '%s/%s-predictions-%s.csv.gz' % (System.projectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving new TEST set (dim: {0}) to:\n{path}\n'.format(df.shape, path=df_path))

    return

# common
def save_data(D, L, fold, method, **kargs):
    import pandas as pd

    project_path = kargs.get('project_path', System.projectPath)

    ##############################################################
    # ... determine index, label and users (classifier names)
    tRandomSubsampling = kargs.get('subsampling', False)
    if tRandomSubsampling: 
        index = kargs.get('indices', [])
        if len(index) == 0: 
            # in subsampling mode, we cannot use read_fold() to recover the labeling
            index = pd.MultiIndex.from_tuples([(i, L[i]) for i in range(len(L))])  # construct MultiIndex from tuples
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(save_data) Created indices for the new data (method: %s)' % method)   # ... ok
        else: 
            # if 'names' is not part of the index attribute => FrozenList([None])
            assert index.names[0] is not None

        if not 'U' in kargs: 
            train_df, train_labels, test_df, test_labels = common.read_fold(project_path, 0) # [todo] single out this part
            cols = test_df.columns
        else: 
            cols = kargs['U']
    else:
        train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold) # [todo] single out this part
        index = test_df.index
        cols = test_df.columns

        # verify 
        if kargs.get('verify', True):
            # test_df = test_df.reset_index() # convert multilevel index to flat index
            labels = test_df.index.get_level_values('label').values
            assert all(L == labels)

            if 'U' in kargs: 
                assert all(kargs['U'] == test_df.columns.values)
    ##############################################################

    dtype = kargs.get('dtype', 'posterior')  # prior (original) vs posterior (after the transformation)
    # level-1 training data: validation-<fold>.csv.gz
    df = DataFrame(D.T, index=index, columns=cols)  # datasink convention> rows: data points, columns: classifiers
    
    # save   # note that regular level-1 test data has the name: predictions-<fold>.csv.gz
    df_path = '{prefix}/{method}-{dtype}-{index}.csv.gz'.format(prefix=project_path, method=method, dtype=dtype, index=fold)
    df.to_csv(df_path, compression='gzip')
    print('(output) saving the data (dtype: {dtype} dim: {dim}) to:\n{path}\n'.format(dtype=dtype, dim=df.shape, path=df_path))

    return 

def shuffle_split(df, labels=[], ratio=0.2, max_size=None, **kargs): 
    """

    Use
    ---
    1. In model_select_core(), model selection is performed to choose the best parameter combination from among a set of candidates; we wish for each iteration
       in model selection to reference a different version of train-dev split sampled from a pre-specified train-dev split (i.e. the data minus the test set). 
    """
    import data_pipeline_datasink as dsp
    
    # return value: (R, Td, L_train, L_dev, U)
    return dsp.shuffle_split(df, labels=labels, ratio=ratio, max_size=max_size, **kargs)

def apply_resample(X, L, method=''): 
    import data_pipeline_datasink as dsp
    # return value: (Xr.T, Lr)
    return dsp.apply_resample(X, L, method=method)

def to_rating_matrix_random_subsampling(**kargs):
    """
    An extension to to_rating_matrix() by supporting random subsampling. 
    The premise of using this routine to construct rating matrices is to incorporate model selection, and therefore, 
    we shall assume by default that the return value should include dev set, which means that the return value 
    consider rating matrices structured as (R, Td, Tt) instead of (R, T), where 

    Td: the rating matrix for the dev set (for hyperparameter tunning)
    Tt: the rating matrix for the test set (for model evaluation)

    Use
    ---
        to_rating_matrix_random_subsampling(dev_ratio=1/5., fold_count=5, policy='random_cv_fold')
    
    """ 
    import data_pipeline_datasink as dsp
    # return (R, T, train_labels, test_labels, U)
    return dsp.to_rating_matrix_random_subsampling(**kargs)

# subsumed by to_rating_matrix()
def toPredictiveScores(fold, **kargs):
    """
    Same as to_rating_matrix() but perhaps this template code is easier to work with source codes
    in recommender system in general. 

    Memo
    ----
    1. analogous to toRatings()
    """
    import data_pipeline_datasink as dsp
    return dsp.toPredictiveScores(fold, **kargs)  # a dictionary of 5 entries: ['train', 'test', 'train_labels', 'test_labels', 'users', ]
def to_rating_matrix0(fold, **kargs):
    return toPredictiveScores(fold, **kargs)

# [data_pipeline] future: factor this function to data_pipeline module
def to_rating_matrix(fold, **kargs):
    """


    Memo
    ----
    1. train-dev-test split 
       <ref> https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213

       train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))]

    """
    import data_pipeline_datasink as dsp

    # return value: (R, T, train_labels, test_labels, U)
    return dsp.to_rating_matrix(fold, **kargs)

def to_rating_matrix2(fold, **kargs): 
    """

    kargs
        p_threshold: 0.5 by default
        missing_value: 0
        verbose: True
        mask_: True
        unbag: False
        bag_count: -1 by default 

    Memo
    ----
    1. diabetes
       nTP: 4689, nTN: 9011, nFP: 1671, nFN: 3049 (F: 4720)

    """
    import data_pipeline_datasink as dsp 
    # return (R, T, L_train, L_test, U)  # U: users/classifiers
    return dsp.to_rating_matrix2(fold, **kargs)

def estimateProbThresholds(R, L=[], pos_label=1, neg_label=0, policy='prior', ratio_small_class=0.01):
    """

    Memo
    ----

    1. suppose a = [0, 1, 2, ... 9] 
       a[:-5:-1] => the last 4 (-1, -2, -3, -4) elements ~> 9, 8, 7, 6 

       the last k elements 
       a[:-k-1: -1], which then include the last kth element

       a[:-3]: a[0] up to but not include the last 3rd element 
              => [0, 1, 2, 3, 4, 5, 6]

       a[:-3: -1]: counting from the back, up to but not include the last 3rd element 
               => [9, 8]  (excluding 7)

    1a. select columsn and rows by conditions 

            >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
            >>> a
            array([[ 1,  2,  3,  4],
                   [ 5,  6,  7,  8],
                   [ 9, 10, 11, 12]])

            >>> a[a[:,0] > 3] # select rows where first column is greater than 3
            array([[ 5,  6,  7,  8],
                   [ 9, 10, 11, 12]])

            >>> a[a[:,0] > 3][:,np.array([True, True, False, True])] # select columns
            array([[ 5,  6,  8],
                   [ 9, 10, 12]])

            # fancier equivalent of the previous
            >>> a[np.ix_(a[:,0] > 3, np.array([True, True, False, True]))]
            array([[ 5,  6,  8],
                   [ 9, 10, 12]])

    """
    if not policy: 
        if len(L) > 0:   
            policy = 'prior'  
        else: 
            policy = 'ratio'  # unsupervised, given only an estimted ratio of the minority class (typically positive class)  

    thresholds = []
    if policy.startswith( ('prior', 'top' )):  # i.e. top k probabilities as positives where k = n_pos examples in the training split
        assert len(L) > 0, "Labels must be given in 'prior' mode (in order to estimate the lower bound of top k probabilities for positive class)"
        k = np.sum(L == pos_label)  # top k probabilities => positive 
        for i in range(R.shape[0]):  # foreach user/classifier
            # use topk prob scores to determine the threshold
            p_th = R[i, np.argsort(R[i])[:-k-1:-1]][-1]  # min of top k probs
            thresholds.append(p_th)
    elif policy == 'fmax':
        print("(estimateProbThresholds) policy: fmax")
        from evaluate import findOptimalCutoff   #  p_th <- f(y_true, y_score, metric='fmax', pos_label=1)
        thresholds = findOptimalCutoff(L, R, metric=policy, beta=1.0, pos_label=1) 
    
    elif policy == 'ratio':  # user provided an estimated ratio of the minority class, perhaps more conservative than 'prior' 
        assert ratio_small_class > 0, "'ratio_small_class' was not set ... "
        k = math.ceil(ratio_small_class * R.shape[1])  # find k candidates in the item direction
        for i in range(R.shape[0]):  # foreach row/user/classifier
            
            cols_high = np.argsort(R[i])[:-k-1: -1]  # indices of highest probabilities (these are likely to be better probability estimates of the positive)
            p_th = R[i, cols_high][-1] # min of the largest probs
            thresholds.append(p_th)

            ### another thresholds for the lowest probs (negative classe)
            # cols_low = np.argsort(R[i])[:k]  # column indices of the lowest probs
            # p_th_lower = R[i, cols_low][-1] # max of the lowest probs
            # thresholds_lower.append(p_th_lower)
    assert len(thresholds) == R.shape[0], "Expecting {n} p-thresholds (= n_users/n_classifiers) but got {np}".format(len(thresholds), R.shape[0])
    return np.array(thresholds)

def estimateProbaByLabelMatrix(R, L, p_th=[], policy_threshold='prior', pos_label=1):
    def findsubsets(S,m):
        return set(itertools.combinations(S, m))

    import collections, itertools

    if len(p_th) > 0: 
        assert len(p_th) == R.shape[0]
        Lh = estimateLabelMatrix(R, L=L, p_th=p_th, pos_label=pos_label)
    else: 
        supervised_policies = ['prior', 'fmax', ]
        labels = np.unique(L)  

        # assert len(L) > 0
        assert policy_threshold in supervised_policies, "Expecting supervised policy but got {policy}".format(policy=policy_threshold)
        Lh = estimateLabelMatrixByThresholdingPolicy(R, L=L, policy_threshold=policy_threshold, pos_label=pos_label)

    # counter for estimated labels, lv (column by column)
    labels = np.unique(L)
    lv_count = {} # collections.Counter()
    print('(estimateProbaByLabelMatrix) n(+): {npos}, n(-): {nneg} | L'.format(npos=sum(L==1), nneg=sum(L==0)))

    for j in range(R.shape[1]): 
        y_true = L[j]
        lv = tuple(Lh[:, j])

        # init
        if not lv in lv_count: lv_count[lv] = {label:0 for label in labels}
        
        # lv_count.update([ tuple(Lh[:, j]), ])
        lv_count[lv][y_true] += 1

        # subsets of length (n-1)   ... todo

    # from count to probability
    for i, (lv, entry) in enumerate(lv_count.items()):  
        n = sum(entry[label] for label in labels)
        for label in labels: 
            lv_count[lv][label] = lv_count[lv][label]/(n+0.0)

    # compute the n-1 joint model? ... # todo

    return lv_count

def estimateLabelMatrixByThresholdingPolicy(R, L=[], policy_threshold='prior', ratio_small_class=0.01, pos_label=1): 
    supervised_policies = ['prior', 'fmax', ]
    if policy_threshold in supervised_policies:
        assert len(L) > 0, "Using supervised thresholding policy ({poilcy}) but labels were not given!".format(policy=policy_threshold)
    p_th = estimateProbThresholds(R, L=L, policy=policy_threshold, pos_label=pos_label, neg_label=0, ratio_small_class=ratio_small_class)
    return estimateLabelMatrix(R, L=L, p_th=p_th, pos_label=pos_label)

def estimateLabelMatrix(R, L=[], p_th=[], pos_label=1, neg_label=0, ratio_small_class=0.01):
    """
    Convert R, a probability matrix, into a label matrix. 

    Notes
    -----
    1. This is NOT the same as estimateLabels(), which outputs a vector of labels for the input matrix R;
       a label matrix can be transformed into a label vector via, for instance, majority vote. 
    """ 
    ### verify
    k = -1
    policy = 0  # dummy method p_th = 0.5
    if (hasattr(p_th, '__iter__') and len(p_th) > 0):
        policy = 'thresholding' # classifier-specific probability thresholding (e.g. p_th ~ fmax)
        assert R.shape[0] == len(p_th)
    elif isinstance(p_th, float): # heuristic, unsupervised but not practical
        policy = 'thresholding'
        p_th = [p_th, ] * R.shape[0]
    elif len(L) > 0:  # top k probability thresholding (highest probabilities => positive)
        # use L to estimate proba thresholds
        policy = 'prior'  # use L to estimate class ratio, from which to estimate p_th
    else: 
        policy = 'ratio'
        # use case: 
        #    R can be the rating matrix from training split or test split 
        #    L must be from the training split
        
        # use L to estimate k, which is then used to estimate prob threshold

        # assert R.shape[1] == len(L)
        # k = np.sum(L == pos_label)  # top k probabilities => positive
    # print('(estimateLabelMatrix) policy: {}'.format(policy))

    ### computation starts here
    Lh = np.zeros_like(R).astype(int)
    if policy.startswith('thr'): 
        # print("(estimateLabelMatrix) policy: 'thresholding' > thresholds pre-computed:\n{0}\n".format(p_th))
        Lh = np.where(R >= np.array(p_th)[:, None], pos_label, neg_label)

        # for i in range(R.shape[0]):  # foreach user/classifeir
        #     cols_pos = R[i] >= p_th[i]
        #     Lh[i, cols_pos] = pos_label

    elif policy.startswith(('pri', 'top')):
        assert len(L) == R.shape[1] 
        print("(estimateLabelMatrix) policy: 'prior' > use the top k probs as a gauge for positives...")

        p_th = estimateProbThresholds(R, L=L, pos_label=pos_label, policy='prior')
        Lh = np.where(R >= np.array(p_th)[:, None], pos_label, neg_label)

        # k = np.sum(L == pos_label)  # top k probabilities => positive
        
        # for i in range(R.shape[0]):  # foreach user/classifier
        #     # use topk prob scores to determine the threshold
        #     p_th = R[i, np.argsort(R[i])[:-k-1:-1]][-1]  # min of top k probs
        #     cols_pos = R[i] >= p_th
        #     Lh[i, cols_pos] = pos_label

        # may need to return the resulting thresholds
    elif policy == 'ratio': 
        # ratio_small_class estimated externally (e.g. from training set R)
        assert ratio_small_class > 0, "Minority class ratio must be > 0 in order to get an estimate of top k probabilities."
        p_th = estimateProbThresholds(R, L=[], pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class)
        Lh = np.where(R >= np.array(p_th)[:, None], pos_label, neg_label)

    else: 
        median_predictions = np.median(R, axis=0)
        mean_predictions = np.mean(R, axis=0)
        for j in range(R.shape[1]): 
            if median_predictions[j] > mean_predictions[j]:  # median > mean, skew to the left (mostly high probabilities)
                L_est[i] = pos_label

    return Lh

def estimateLabels(T, L=[], p_th=[], Eu=[], pos_label=1, neg_label=0, M=None, labels=[], policy='', ratio_small_class=0, joint_model=None):
    """
    Estimate the labels based on the input rating matrix. 

    Parameters
    ----------
    M: message from the training split (R); used to estimate P(y=1|Lh)
    joint_model: a joint model for answering queries P(y=1|Lh) computed externally

    Memo
    ----
    1. collections.Counter(Lh[:, i]) returns a list of 2-tuples: (label, count)

    2. Pass T and p_th estimated externally

    3. logic chain 
       class prior (or external estimate of class ratio) -> p-thresholds -> Lh -> lh
    """
    import operator
    # import collections
    # probs = np.mean(T, axis=0)  
    ### verify 
    if not policy:  
        # precedence: proba thresholds -> labels -> ratio_small_class -> other unsupervised methods (e.g. mean vs median)
        if hasattr(p_th, '__iter__') and len(p_th) > 0: # supervised (using fmax(L) to pre-compute prob threshold)
            policy = 'thresholding'
            assert len(p_th) == T.shape[0], "Each user/classifier has its own threshold. n(p_th): %d but n_users: %d" % (len(p_th), T.shape[0])
        elif isinstance(p_th, float): # heuristic, unsupervised but not practical
            policy = 'thresholding'
            p_th = [p_th, ] * T.shape[0]
        elif len(L) > 0:  # weakly supervised (using L to determine prob threshold)
            # p_th = estimateProbThresholds(R, L=L, pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class)
            p_th = estimateProbThresholds(T, L=L, pos_label=pos_label, policy='prior')
            policy = 'thresholding'
        else: # Neither 'p_th' nor 'L' was given ... 
            # policy = 'mean-median' # does not work well 
            policy = 'ratio'

    tHasMessageFromTrainingSplit = False
    if M is not None or joint_model is not None: 
        # X_train, L_train, *rest = M
        tHasMessageFromTrainingSplit = True
        policy = 'joint'
    
    ### label estimate starts here
    L_est = np.zeros(T.shape[1]).astype(int)

    # [policy] Current favorable policy for estimating labels: Majority vote given probabilty thresholds
    ####################################################
    if policy.startswith('thr'):  # given the probability thresholds (a vector in which each component represents a classifier-specific prob threshold, lg than which is positive, otherwise negative)
        # p_th is a vector/numpy array
        # print("(estimateLabels) Using 'majority vote' given proba thresholds ...")

        #-----------------------------
        # Todo: 
        if len(Eu) > 0: # excluse the votes from those classifiers that did not produce any correct predictions
            pass
        #------------------------------

        for j in range(T.shape[1]): 
            # majority vote of the user/classifiers for data j is True => positive
            if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: # if it's that majory vote says positive
                L_est[j] = pos_label
    ####################################################

    # elif policy.startswith('f'): # p_th is a fixed probability score (e.g. 0.5)
    #     L_est[np.where(np.median(T, axis=0) >= p_th)] = pos_label
    else:  # p: None or [] => use mean prediction as an estimate
        # print("(estimateLabels) policy: 'prior' > use the top k probs as a gauge for positives...")
        if policy.startswith( ('pri', 'top') ): ### weakly supervised
            assert len(L) > 0, "Need labels (L) to use this mode; if not, use 'ratio' instead by providing an estimate of the ratio of the minority class via 'ratio_small_class' ... "
            # need to estimate the label per classifier => label matrix
            p_th = estimateProbThresholds(T, L=L, pos_label=pos_label, policy='prior')

            for j in range(T.shape[1]): 
                # majority vote of the user/classifiers for data j is True => positive
                if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: # if it's that majory vote says positive
                     L_est[j] = pos_label

        elif policy.startswith('joint'):
            # the problem of this strategy is that P(y=1 | lv ) is probably 0 even if the queried prediction vector (pv) matches lv
            # => pv matching lv is too weak of a condition to conclude a case for a minority class
            # => need to 're-calibrate' the probability

            print("(estimateLabels) Estimate labels via 'joint model' (P(y=1|Lh)) ...")
            assert len(p_th) > 0, "We must assume that proba thresholds have been estimated prior to this call"
            assert M is not None, "Need access to the training data info to proceed (e.g. class prior, joint model)"
            
            r_min = r_max = 0.0
            #####################################
            # jointModel = joint_model
            # if jointModel is None: 
            X_train, L_train, *rest = M
            # if p_th is given, then use it; otherwise, use what 'policy_threshold' specifies to estimate p_th, from which to estimate Lh
            jointModel = estimateProbaByLabelMatrix(X_train, L_train, p_th=p_th, pos_label=1) # policy_threshold='prior' 
            #####################################
            ret = classPrior(L_train, labels=[0, 1], ratio_ref=0.1, verbose=False)
            r_min, r_max = ret['r_min'], ret['r_max']
            min_class, max_class = ret['min_class'], ret['max_class']
            #####################################
            Lh = estimateLabelMatrix(T, L=[], p_th=p_th)
            
            # now make queries into jointModel
            ulabels = [0, 1]
            # for i, (lv, u) in enumerate(jointModel.items()): 
            #     ulabels.update(u.keys())
            #     if i > 10: break 

            # L_est = np.zeros(T.shape[1]).astype(int)
            n_min = int(T.shape[1] * r_min) 
            print('... Expecting to observe {n} minority class examples in T (n={N}) | min_class: {cls}'.format(n=n_min, N=T.shape[1], cls=min_class))

            n_unmodeled = n_unmodeled_pos = n_unmodeled_neg = 0
            for j in range(T.shape[1]):
                lv = tuple(Lh[:, j]) 
                if lv in jointModel: 
                    # print('... test: {value}'.format(value=jointModel[lv]))
                    Py = {label: jointModel[lv][label] for label in ulabels}
                    
                    L_est[j] = max(Py.items(), key=operator.itemgetter(1))[0] # the label with higher proba is the estimated label
                    # ... chances are that the argmax leads to the majority class 
                
                    # [test]
                    if j < 10: 
                        print('... Py: {dict} | argmax: {max}'.format(dict=Py, max=L_est[j]))
                else: 
                    # reverse back to majority vote
                    n_unmodeled += 1 
                    if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: 
                        L_est[j] = pos_label
                        n_unmodeled_pos += 1
                    else: 
                        n_unmodeled_neg += 1
            print('... (debug) n_unmodeled: {n0}, n_unmodeled_pos: {n1}, n_unmodeled_neg: {n2}'.format(n0=n_unmodeled, n1=n_unmodeled_pos, n2=n_unmodeled_neg))
            # [observation] 
            #   ... unmodeled cases are relatively less frequent

            ##################################### 
            # ... calibration
            n_min_prior_calib = sum(L_est == min_class) 
            print('... Found only {n} estimated minority class instances prior to calibration!'.format(n=n_min_prior_calib))
            if n_min_prior_calib < n_min: 
                min_index_proba = collections.Counter()
                for j in range(T.shape[1]):
                    lv = tuple(Lh[:, j]) 
                    if lv in jointModel: 
                        Py = {label: jointModel[lv][label] for label in ulabels}
                        
                        if Py[min_class] > 0.0: # [todo] use min-max heap to reduce MEM 
                            min_index_proba[j] = Py[min_class] # append( (j, Py[min_class]) )
                    
                    else: 
                        # indices associated with the majority vote is not subject to re-calibration
                        pass
                n_min_eff = n_min - n_unmodeled
                print('... (debug) Found {n} indices subject to be re-calibrated'.format(n=n_min_eff))
                candidates = min_index_proba.most_common(n_min_eff)
                print('... top {n} indices and their proba:\n{alist}\n...'.format(n=n_min_eff, alist=candidates[:200]))
                
                for j, proba in candidates: 
                    L_est[j] = min_class
            n_min_post_calib = sum(L_est == min_class)
            print('... Found {n} estimated minority class instances AFTER calibration | expected: {ne}'.format(n=n_min_post_calib, ne=n_min))
  
        elif policy == 'ratio':
            # ratio_small_class estimated externally (e.g. from training set R)
            assert ratio_small_class > 0, "Minority class ratio must be > 0 in order to get an estimate of the lower bound of the top k probabilities."
            p_th = estimateProbThresholds(T, L=[], pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class)

            for j in range(T.shape[1]): 
                # majority vote of the user/classifiers for data j is True => positive
                if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: # if it's that majory vote says positive
                     L_est[j] = pos_label

        elif policy.startswith(('uns', 'mean-m')): ### unsupervised 
            median_predictions = np.median(T, axis=0)
            mean_predictions = np.mean(T, axis=0)
            for i in range(T.shape[1]): 
                if median_predictions[i] > mean_predictions[i]:  # median > mean, skew to the left (mostly high probabilities)
                    L_est[i] = pos_label
        else: 
            raise NotImplementedError('Unrecognized policy: %s' % policy)
                
    return L_est 

def classSummaryStats(R, L, labels=[0, 1], ratio_ref=0.1, policy='prior'):
    ret = classPrior(L, labels=labels, ratio_ref=ratio_ref)
    ret['thresholds'] = estimateProbThresholds(R, L=L, pos_label=labels[1], policy=policy) # policy: {'prior', 'fmax'}
    
    n_users, n_items = R.shape[0], R.shape[1]
    assert len(ret['thresholds']) == R.shape[0], "Each user/classifier has a prob threshold."

    return ret

def classPrior(L, labels=[0, 1], ratio_ref=0.1, verify=True, verbose=True):  # assuming binary class
    # import collections 
    if not labels: labels = np.unique(L)
    lstats = collections.Counter(L)
    ret = {} # {l:0.0 for l in labels}
    if len(labels) == 2: # binary class 
        neg_label, pos_label = labels

        ret['n_pos'] = nPos = lstats[pos_label] # np.sum(L==pos_label)
        ret['n_neg'] = nNeg = lstats[neg_label] # np.sum(L==neg_label)
        ret['n_min_class'] = nPos if nPos <= nNeg else nNeg
        ret['n_max_class'] = nNeg if nPos <= nNeg else nPos
        ret['n'] = nPos + nNeg
        if verify: assert len(L) == ret['n'], "n(labels) do not summed to total!"

        ret['r_pos'] = ret[pos_label] = rPos = nPos/(len(L)+0.0)
        # nNeg = len(L) - nPos 
        ret['r_neg'] = ret[neg_label] = rNeg = nNeg/(len(L)+0.0) # rNeg = 1.0 - rPos
        ret['r_min'] = ret['r_minority'] =  min(rPos, rNeg)
        ret['r_max'] = ret['r_majority'] = 1. - ret['r_min']
        ret['min_class'] = ret['minority_class'] = pos_label if rPos < rNeg else neg_label
        ret['max_class'] = ret['majority_class'] = neg_label if rPos < rNeg else pos_label

        ret['r_max_to_min'] = ret['multiple'] = ret['n_max_class']/(ret['n_min_class']+0.0)
        
        if min(rPos, rNeg) < ratio_ref:  # assuming pos labels are the minority
            if verbose: print('(class_prior) Imbalanced class distribution: n(+):{0}, n(-):{1}, r(+):{2}, r(-):{3}'.format(nPos, nNeg, rPos, rNeg))
            ret['is_balanced'] = False

        # print('... class distribution > n_pos: {0} (total: {1}, ratio: {2})'.format(nPos, len(L), rPos))
        # print('...                      n_neg: {0} (total: {1}, ratio: {2})'.format(nNeg, len(L), rNeg))
    else: # single class or multiclass problems
        raise NotImplementedError
    return ret  # keys: n_pos, n_neg, r_pos, r_neg, 0/neg_label, 1/pos_label, is_balanced, r_minority, min_class

def getMask(A, fill=0):
    # rows, cols = np.where(A == fill) # returns 2-tuple of row indices and col indices 
    return np.where(A == fill) 

def maskEntries(R, L=[], p_th=[], C=None, ratio_small_class=0.0, policy='', balance_class=False, 
        min_label=1, max_label=0, pos_label=1, eval_label_stats=True): 
    """

    Params
    ------
    C: (pre-computed) confidence matrix 

    Memo
    ----
    1. 'policy' is a string specifying which line of method to follow
        use '-'/dash or '+' to specify a composite policy
        e.g. 

        'thresholding-aggregate'
    
    2. Cases: 
        L is given and p_th is given 
           p_th is external (e.g. fmax)
           p_th is inferred from class prior

           if p_th is passed in, we shall assume it's been computed externally (e.g. fmax)

    """
    def mask_by_label_estimates(R, L, Lh, mask=None): 
        if mask is None: mask = np.ones(R.shape, dtype=int) # 2D array values=True: mask everything unless set to False when certain conditions are met (i.e. True: mask or set to null, False: retain)
        
        for i in range(R.shape[0]):  # foreach row/user/classifier 
            cols_active = Lh[i] == L
            mask[i, cols_active] = 1  # False: retained, True: to be zero out (e.g. X[mask] = fill, where 'fill' is usu 0)
        return mask
    def mask_by_balancing_estimates(R, L, Lh, multiple=10, mask=None): # closure: min_label, max_label  
        # select majority class examples with sample size 'comparable' to that of the minority class 

        if mask is None: mask = np.ones(R.shape, dtype=int)
        n_min_captured = 0
        for i in range(R.shape[0]):  # foreach row/user/classifier 
            idx_min_class = np.where( (Lh[i] == L) & (Lh[i] == min_label) )[0]
            idx_max_class = np.where( (Lh[i] == L) & (Lh[i] == max_label) )[0]

            # now select only a subset (n_min * multiple) of the matched majority-class examples
            ######################################
            n_min, n_max = len(idx_min_class), len(idx_max_class)  # let n: size(minority), then we will pick n * multiple examples from the majority class
            n_min_captured += n_min
            n_max_eff = int(n_min * multiple) # let n: size(minority), then we will pick, say, 10 * n examples from the majority class
            if n_max > n_max_eff:  
                # downsampling
                idx_max_class = np.random.choice(idx_max_class, n_max_eff, replace=False)
            else: 
                # do nothing 
                pass
            ############################################################################

            idx_active = np.hstack( (idx_min_class, idx_max_class) ) # pad the top choices
            mask[i, idx_active] = 1  # False: retained, True: to be zero out (e.g. X[mask] = fill, where 'fill' is usu 0)

        # [debug]
        idx_active0 = np.where(Lh == L)[0] # note that accuracy can be low (e.g. accuracy: 41.32% for pf2)
        print("(mask_by_balancing_estimates) captured {n} minority class examples (avg: {ne}/classifier <? n_min_class: {n_min}) | n(Lh=L): {nr} (accuracy: {a}, total: {nL}) | policy: {policy} ... ".format(n=n_min_captured,
            ne=n_min_captured/(R.shape[0]+0.0), n_min=n_min_class, nr=len(idx_active0), a=len(idx_active0)/(nL+0.0), nL=nL, policy=policy))
        return mask

    def mask_by_confidence_weights(R, L, C, multiple=2, mask=None): 
        pass

    def mask_by_balanced_true_labels(R, L, multiple=2, mask=None): # closure: min_label, max_label  
        # test only, not recommended 

        if mask is None: mask = np.ones(R.shape, dtype=int)
        n_min_captured = 0
        for i in range(R.shape[0]):  # foreach row/user/classifier 
            idx_min_class = np.where( L == min_label )[0]
            idx_max_class = np.where( L == max_label )[0]

            # now select only a subset (n_min * multiple) of the matched majority-class examples
            ######################################
            n_min, n_max = len(idx_min_class), len(idx_max_class)  # let n: size(minority), then we will pick n * multiple examples from the majority class
            n_min_captured += n_min
            n_min_eff = int(n_min * multiple) # let n: size(minority), then we will pick 2n examples from the majority class
            if n_max > n_min_eff:  
                idx_max_class = np.random.choice(idx_max_class, n_min_eff, replace=False)
            else: 
                idx_max_class = np.random.choice(idx_max_class, n_min_eff, replace=True)
            ############################################################################

            idx_active = np.hstack( (idx_min_class, idx_max_class) ) # pad the top choices
            mask[i, idx_active] = 1  # False: retained, True: to be zero out (e.g. X[mask] = fill, where 'fill' is usu 0)

        # [debug]
        # idx_active0 = np.where(Lh == L)[0]
        print("... (verify) captured {n} minority class examples (avg: {ne}/classifier <? n_min_class: {n_min}) | total: {nL}) | policy: {policy} ... ".format(n=n_min_captured,
            ne=n_min_captured/(R.shape[0]+0.0), n_min=n_min_class, nL=nL, policy=policy))
        return mask
    def mask_by_topk(R, k, mask=None): # unsupervised; L is not available
        if mask is None: mask = np.ones(R.shape, dtype=int)
        for i in range(R.shape[0]):
            cols_high = np.argsort(R[i])[:-k-1: -1]  # indices of highest probabilities (these are likely to be better probability estimates of the positive)
            cols_low = np.argsort(R[i])[:k] # lowest probabilities (these are likely to be better estimates of the negative)

            cols_active = list(set(np.hstack((cols_high, cols_low))))
            mask[i, cols_active] = 1 # False: don't mask (want to retain these entries)
        return mask 
    def mask_by_label_estimates_unsupervised(R, p_th, mask=None, multiple=2): # closure: balance_class
        if mask is None: mask = np.ones(R.shape, dtype=int)
        
        assert p_th is not None and len(p_th) == R.shape[0], "(mask_by_label_estimates_unsupervised) Invalid proba thresholds: %s" % p_th
        L = lh = estimateLabels(R, p_th=p_th, pos_label=pos_label)  # label is determined by majority vote
        Lh = estimateLabelMatrix(R, p_th=p_th, pos_label=pos_label)  
        if balance_class: 
            return mask_by_balancing_estimates(R, L, Lh, multiple=multiple, mask=mask)
        else: 
            return mask_by_label_estimates(R, L, Lh, mask=mask)

    ### verify the policy
    n_min_class = n_max_class = 0
    nL = 0
    if not policy: 
        if len(L) > 0: 
            L = np.array(L)
            nL = L.shape[0]

            # 1. L + p_th
            if hasattr(p_th, '__iter__') and len(p_th) > 0:  # 'thresholding+aggregate', 'thresholding'
                # supervised (using fmax(L) to pre-compute prob threshold)
                # >>> note that it's possible that extra instruction is given in the policy string (e.g. 'thresholding+aggregate'); 
                #     if not use the default 'thresholding'
                policy = 'thresholding'  
                assert len(p_th) == R.shape[0], "Each user/classifier has its own threshold. n(p_th): %d but n_users: %d" % (len(p_th), R.shape[0])
            elif isinstance(p_th, float): # heuristic, unsupervised but not practical
                raise ValueError("Policy with a fixed prob threshold is not recommended (p_th={0}".format(p_th))

            elif len(p_th) == 0: 
                p_th = estimateProbThresholds(R, L=L, pos_label=pos_label, policy='prior')
                policy = 'thresholding'
                # ... no need to use 'prior' mode
            
            if eval_label_stats: 
                # counts = collections.Counter(L)
                ret = classPrior(L, labels=[0, 1], ratio_ref=0.1, verbose=False)
                min_label, max_label = ret['min_class'], ret['max_class']
                n_min_class, n_max_class = ret['n_min_class'], ret['n_max_class']
        else: 
            # parameter: ratio_small_class: i) estimated externally ii) class prior 
            policy = 'ratio'  # default mode: consider upper top k and lower k; conservative

            # do we have a proba threshold estimates from the training data? if so, use it to estimate 'true labels'
            if hasattr(p_th, '__iter__') and len(p_th) > 0:
                policy = 'message_threshold'  # pretend that 'lh' is true

    n_items_per_user = []  # debug
    Lh = None
    if policy.startswith('thr'): # probability thresholding; use precomputed prob thresholds plus the true label as reference
        ### probability thresholds are provided as an argument; each classifier has its own threshold
        # p_th is a sequence of list or numpy array
        # e.g. thresholds associated with the fmax 
        # e.g. thresholds according to the 'top k' rule: 
        #          estimateProbThresholds(R, L, pos_label=1, neg_label=0)

        ### policy, consider a majority vote to reduce Lh (a matrix, each classifier having its own view of labeling) OR 
        # reduce Lh to a single vector? lh 
        if policy.find('aggregate') > 0: # thresholding+aggregate => label is determined by majority vote
            mask = np.ones(R.shape, dtype=int)
            lh = estimateLabels(R, p_th=p_th, pos_label=pos_label)  # label is determined by majority vote

            # mask by majority vote: use majority vote combined labels as the estimate 
            for i in range(R.shape[0]):  # foreach row 
                cols_active = lh == L
                mask[i, cols_active] = 1  # turn off the mask, so that these entries are retained
        else: # per-classifier view of the labeling
            # ... probably using Lh is better so that each classifier has its own view 
            Lh = estimateLabelMatrix(R, p_th=p_th, pos_label=pos_label)  # not passing in L
            
            # balance the class? i.e. downsample majority class
            if balance_class: 
                # multiple: e.g. let n: size(minority), then we sample n * multiple examples from the majority class
                mask = mask_by_balancing_estimates(R, L, Lh, multiple=2) # note: may end up having only a "small" subset of minority class examples (with estimated labels aligned with true labels)
                # mask = mask_by_balanced_true_labels(R, L, multiple=2) # note: some of these true labels may not have good proba estimtes

                # note if size(mask) == size(R), i.e. none was masked, then we shall roll back to mask_by_label_estimates()
            else: 
                mask = mask_by_label_estimates(R, L, Lh) # min_label=min_label, max_label=max_label

    ##############################
    # ... subsumed by 'thresholding' mode
    elif policy.startswith( ('pri', 'top') ): # topk
        # >>> this is really just a special case of "thresholding" because one could estimate the threshold using "prior" policy and pass it as an argument to this routine
        Lh = estimateLabelMatrix(R, L=L, pos_label=pos_label)  # use ratio of minornity class to estimate the proba threshold, from which labels are estimated

        if balance_class: 
            # multiple: e.g. let n: size(minority), then we sample n * multiple examples from the majority class
            mask = mask_by_balancing_estimates(R, L, Lh, multiple=2, min_label=min_label, max_label=max_label)  
            # mask = mask_by_balanced_true_labels(R, L, multiple=2) 

            # [debug]
            idx_active0 = np.where(Lh == L)[0]
            print("... (verify) captured {n} minority class examples (avg: {ne}/classifier <? n_min_class: {n_min}) | n(Lh=L): {nr} (accuracy: {a}, total: {nL}) | policy: {policy} ... ".format(n=n_min_class,
                    ne=n_min_class/(R.shape[0]+0.0), n_min=n_min_class, nr=len(idx_active0), a=len(idx_active0)/(nL+0.0), nL=nL, policy=policy))
        else: 
            mask = mask_by_label_estimates(R, L, Lh)
    ##############################
    
    elif policy.startswith( ('r', 'uns') ):  # neither L nor p_th was given => unsupervised
        
        assert ratio_small_class > 0, "'ratio_small_class' was not set. "
        k = math.ceil(R.shape[1] * ratio_small_class)  # user provided small class ratio
        print('(maskEntries) Unsupervised Mode: Given estimated proportion of the minority class (ratio: {0} => k: {1})'.format(ratio_small_class, k))

        mask = mask_by_topk(R, k)
    elif policy.startswith( 'message' ):  # message passing: L is not given but pre-computed proba threshold (p_th) is available 
        # should be rarely used because L is usually esitmated prior to this call 
        assert len(p_th) == R.shape[0] and len(L) == 0, "Message passing mode: L is not given but pre-computed proba threshold (p_th) is available "
        # estiamte labels based on proba thresholds followed by majority vote

        print('(maskEntries) Unsuerpvised Mode but given precomputed proba threshold estiamtes ... ')
        mask = mask_by_label_estimates_unsupervised(R, p_th)  
    else: 
        raise NotImplementedError("Unrecognized policy: %s" % policy)

        # for i in range(R.shape[1]):   
        #     # if positive, select highest probs; if negative, select lowest
        #     #    R[:, i][np.argsort(R[:, i])[:-k-1: -1]] => top tops in descending order
        #     rows_active = np.argsort(R[:, i])[:-k-1: -1] if L[i] == pos_label else np.argsort(R[:, i])[:k]
        #     mask[rows_active, i] = False  # False => don't mask => selected

    # [test] prior to introducing neutral particles
    if Lh is not None and len(L) > 0: 
        print('(maskEntries) verifying the mask ...')
        for i in range(mask.shape[0]):  # foreach row/user/classifier 
            cols_active = Lh[i] == L
            assert all(mask[i, ~cols_active]) == 0  # C[mask] = marker <- 0
            assert all(mask[i, cols_active]) == 1

    return mask
############################################
# ... alias 
mask_along_user_axis = maskEntries
############################################

def maskTestEntries(T, p_th=[], stats={}, policy='', masked=True, **kargs):
    # p_th: 0.5, class prior, array, null value ([]/None)
    # policy: item, item-match
 
    mask = np.ones(T.shape, dtype=bool)  # default: select all
    if not masked: 
        print('(maskTestEntries) Selecting all entries in test split | condition: T not given > select all entries #')
        return mask # choose all 

    pos_label, neg_label = kargs.get('pos_label', 1), kargs.get('neg_label', 0)
    if not policy: 
        if hasattr(p_th, '__iter__'):
            # supervised (using fmax(L) to pre-compute prob threshold)
            policy = 'thresholding'  # it's possible that extra instruction is given in the policy string (e.g. 'thresholding+aggregate'); if not use the default
            assert len(p_th) == T.shape[0], "Each user/classifier has its own threshold. n(p_th): %d but n_users: %d" % (len(p_th), T.shape[0])
        else: 
            policy = 'neighborhoood'

    ### training set summary statistics 
    # or use optional parameters to pass in R and L, followed by invoking classPrior() or classSummaryStats()
    if not stats: 
        R, L = kargs['R'], kargs['L']
        stats = classSummaryStats(R, L, labels=[0, 1], policy='prior')
    rPos, rNeg = stats[neg_label], stats[pos_label]
    rMinority = min(rPos, rNeg)

    ### filter entries in T

    # topk_dim: {'user', 'item'}
    #     user direction: each user/classifier has to reference (at most) k probabiliy estimates
    #     item direction: each item/datum has to reference (at most) k probability estimates from k users/classifiers
    topk_dim = kargs.get('topk_dim', 'item')  # filtering dimension
    print('(maskTestEntries) Masking the test split (T) | dim(T): {0}, policy: {1}, topk_dim: {2}'.format(T.shape, policy, topk_dim))
    if policy.startswith('thr'):  
        # >>> e.g. first use training data to estimte prob thresholds ...
        #          ... thresholds = estimateProbThresholds(R, L, pos_label=1, neg_label=0)

        #      or use the threshold associated with fmax

        ### suppose we have the prob threshold estimated from the training split ...

        # 1. use majority vote to estimate the labels 
        if len(p_th) == 0: p_th = stats['thresholds']  # threshold estimated via training data (R)
        lh = estimateLabels(T, p_th=p_th, pos_label=pos_label)

        # need to also estimate the ratio of the minority class from the training data
            
        # assuming that the top k probabilities (where k = n_pos in L) correspond to the true positives
        if topk_dim.startswith('u'): 
            k = math.ceil(rMinority * T.shape[1])
            print("... (topk_dim = 'user') keep topk probs per user | r_minority: {0} => k: {1}".format(rMinority, k))
            for i in range(T.shape[0]): # foreach user i    
                # if positive, select highest probs; if negative, select lowest
                #    R[:, i][np.argsort(R[:, i])[:-k-1: -1]] => top tops in descending order
                cols_high = np.argsort(T[i])[:-k-1: -1]  # indices of highest probabilities (these are likely to be better probability estimates of the positive)
                cols_low = np.argsort(T[i])[:k] # lowest probabilities (these are likely to be better estimates of the negative)
                cols_active = list(set(np.hstack((cols_high, cols_low)))) 
                mask[i, cols_active] = False  # False => don't mask => selected
        else: 
            # has to be "somewhat" supervised using estimated labels
            rUsers = kargs.get('ratio_users', 0.5)
            k = math.ceil(rUsers * T.shape[0])
            print("... (topk_dim = 'item') keep topk probs per item | r_users: {0} => k: {1}".format(rUsers, k))
            for j in range(T.shape[1]):

                # >>> cannot compute 'rows_aligned' as in the case of training split 
                # rows_aligned = np.where(Lh[:, j] == L[j])[0]  # Lh[:,j]=L[j] => boolean mask, np.where(...) gives indices where condition holds
                
                # indices of k highest probs vs indices of k lowest probs 
                rows_candidates = np.argsort(T[:, j])[:-k-1: -1] if lh[j] == pos_label else np.argsort(T[:, j])[:k]
                
                # >>> since we cannot compare row_aligned (TP, TN) with rows_candidates, we have to take them all in
                mask[rows_candidates, j] = False  # False => don't mask => selected

            n_masked = np.sum(mask)
            n_selected = np.sum(~mask)
            assert n_masked > 0
            assert n_selected > T.shape[1]
            print("... masked {0} entries in 'T' | selected {1} entries (out of {2}, ratio: {3} => ~{4} per item) in T #".format(n_masked, 
                n_selected, mask.shape[0]*mask.shape[1], n_selected/(mask.shape[0]*mask.shape[1]+0.0), n_selected/(T.shape[1]+0.0)))
            
    elif policy.startswith('neigh'): # neighborhoood
        # compare with the closest prob est. in the training set and see if it was selected
        # R, M, L = kargs.get('R', )
        # analyzeMask(R, M, L, pos_label=1, neg_label=0) 
        raise NotImplementedError
    else: 
        raise NotImplementedError
        
    return mask

def evalConfidenceMatrix(X, L=[], **kargs):
    """
    A (old) wrapper of different confidence matrix functions. 


    Compute confidence matrix i.e. ompute all confidence scores given the confidence measure (conf_measure, e.g. 'brier').
    Confidence scores measure the reliabilty of base classifiers' predictions (i.e. estimates of the conditional probability P(y=1|x))

    **kargs
    -------
    policy: used to define the mask function; specfies the filtering dimension: 'user', 'item' 
            also see 'ratio_small_class', 'ratio_users', conf_measure

    fill: a marker for unreliable rating values, or for missing values; default 0

    pos_label

    p_threshold: a vector of proba thresholds of the same size as X.shape[0], representing the number of users/classifiers; 
                 used to to estimate confidence scores in mode = 'ratio'
    policy_threshold 
    ratio_small_class

    conf_measure: 'brier', 'ratio', 'corr', 
    scoring: scoring function for computing confidence scores

    ratio_users: used as a parameter for the mask function in item-centered mode (i.e. filtering along the axis of the items/data) 

    U
    L_true: ground truth labels (for testing only)

    sparse: 
    alpha


    Similar to toConfidenceMatrix() but returns both Cui and its complement

    Memo
    ----
    1. mask function is determined by: 
        a. filtering dimension: along the user/classifier axis (policy='user'), or along the item/data axis (policy='item')
        b. policy_threshold: how probability thresholds are determined in order to estimate the labeling, which then allows for 
                            the determination of masked entries representing unreliable conditional probability estimates (or in general any ratings in the entries of X)
        c. depending on the true labels (L) being given or not
           we'll need
             ratio_small_class, when L isn't given in order to estimate proba thresholds
        d. ratio_users: for each item/data point, the ratio of user ratings (classifier probabilities) to consider as sufficiently reliable rating esitmates
                        only used in item-centered mode (policy='item') i.e. filtering along the axis of data

    Output
    ------
    Cui, Cui_bar if policy == 'tradeoff'

    Cui for all the other policies

    """
    verbose = kargs.get('verbose', 1)
    policy = kargs.get('policy', 'item')   
    policy_opt = kargs.get('policy_opt', 'rating')

    p_th = kargs.get('p_threshold', [])  # options: a float (e.g. 0.5), 1-D array, 'fmax', 'unsupervised', []/None 
    pos_label = kargs.get('pos_label', 1)
    null_marker = kargs.get('fill', 0)
    fold = kargs.get('fold', -1)  # only used for messaging and debugging
   
    U = kargs.get('U', [])
    ###########################################################
    # confidence measure
    conf_measure = kargs.get('conf_measure', 'brier')
    
    # conf_user, conf_item = kargs.get('conf_user', True), kargs.get('conf_item', True)
    # if conf_measure.startswith('uni'):  # uniform, i.e. all equally good or equally bad
    #     conf_user = conf_item = True  # use both, which will be all ONEs 
    #     # NOTE that if both are False, then the default C = 0; this will cause trouble for class balacning weights because of the zeros
 
    # share parameters between two classes of confidence policies
    # shared_params = {k:kargs[k] for k in kargs.keys() if k in ['conf_measure', 'alpha', 
    #     'masked', 'supervised', 'augmented', 'mask_all_test' ]}  # 'augmented', 'mask_all_test'

    shared_params = {'conf_measure': conf_measure, 
       'alpha': kargs.get('alpha', 10),
       'beta': kargs.get('beta', 1.0),

       'supervised': kargs.get('supervised', True),  # if True, try to use the label information to evaluate the mask (including the use of estimated labels)  

       # not used now
       'masked': kargs.get('masked', True),  # if True, mask the entries of FP, and FT (obsolete)
       'mask_all_test': kargs.get('mask_all_test', False),  # if True, consider the entire test split (T) as having low confidence scores 
       # 'path': kargs.get('path', os.getcwd()),  # path for saving plots (e.g. confidence weights) 

       # 'test_labels': kargs.get('test_labels', []), 
       
    }

    params = {'ratio_users': kargs.get('ratio_users', 0.5), 
                'ratio_small_class': kargs.get('ratio_small_class', 0), 'factor_small_class': kargs.get('factor_small_class', 1.0), 
                'policy_threshold': kargs.get('policy_threshold', 'prior'), # -> estimateProbThresholds(), confidence2D()
              
                # 'conf_measure': kargs.get('conf_measure'), 
                # 'conf_user': True, 
                # 'conf_item': True, 

                # balance class and class weights 
                'balance_class': kargs.get('balance_class', False), 
                'balance_and_scale': kargs.get('balance_and_scale', False), # False if conf_measure == 'rank' else True
                'suppress_negative_examples': kargs.get('suppress_negative_examples', False),
                # 'mask_all_test': kargs.get('mask_all_test', True)

                'is_cascade': kargs.get('is_cascade', True), # if True, X is a concatenation of R and T

                ##########
                # options: {'user', 'item', 'polarity'} but 'user' is not favorable
                'policy_test': kargs.get('policy_test', 'polarity'),  # only relevant when 'is_cascade' is True
                ##########

                'n_train': kargs.get('n_train', -1),   # only relevant when 'is_cascade' is True and filtering axis for train and test splits are different i.e. policy != policy_test 

                'estimated_labels': kargs.get('estimated_labels', False),   # True for test split

                # parameters for polarity matrix
                'labeling_model': kargs.get('labeling_model', 'simple'),   # used to determine polarity matrix
                'constrained': kargs.get('constrained', True),
                'stochastic': kargs.get('stochastic', True),  
                'estimate_sample_type': kargs.get('estimate_sample_type', False),
                'policy_polarity': kargs.get('policy_polarity', 'sequence'),  # options: classification, median

              }
    params.update(shared_params)

    if verbose: 
        div("(evalConfidenceMatrix) policy_filtering: {0}, policy_opt: {1} | conf_measure: {2} | policy_threshold: {3}, ratio_users: {4}, ratio_small_class: {5}, supervised? {6}, mask_all_test? {7} | alpha: {8}".format(policy, 
                        policy_opt, 
                        params['conf_measure'], params['policy_threshold'], 
                        params['ratio_users'], params['ratio_small_class'], 
                        params['supervised'], params['mask_all_test'], 
                        params['alpha']), symbol='=', border=2)
        if params['is_cascade']: 
            print('... Filtering policy in training split: {} =?= test split: {}'.format(policy, params['policy_test']))
        if params['policy_test'].startswith('po'):
            div("(evalConfidenceMatrix) labeling_model: {} | constrained? {}, stochastic? {}, est sample type? {}".format(
                params['labeling_model'], params['constrained'], params['stochastic'], params['estimate_sample_type']), symbol='#') 

        # algorithmic specifics
        print("... Balance class | balance sample size distribution? {t_sample}, balance class conf scores? {t_conf}".format(
            t_sample=params['balance_class'], t_conf=params['balance_and_scale']))
        print("... Posthoc weight adjustments? | beta: {}, suppress_negative_examples: {}".format(params['beta'], params['suppress_negative_examples']))
    
    #############################################
    # ... message passing from training split (R), only applicable when X references a test split (T)
    M = kargs.get('message', None)
    if M is not None: 
        assert isinstance(M, tuple) and len(M) >= 2
        # R, L = M
        if verbose: print('... Passing messages from R to T | X <- T, X_train <- R | Use sample statistics from R to estimate labels in T ... (verify) #')

    # if isinstance(X, tuple): 
    #     assert len(X) == 2
    #     print('... X <- (R, T) | Use sample statistics from R to estimate labels in T ... (verify) #')
    #############################################

    # policy group I. {'rating', 'preference', 'label', 'tradeoff'} 
    #              II. {'user', 'item', }
    Cui = Cui_bar = Pc = None 
    ##############################################################################
    ret = toConfidenceMatrix(X, L,           
                        # confidence score parameters
                        conf_measure=params['conf_measure'], 
                        # conf_user=params['conf_user'], conf_item=params['conf_item'], 
                 
                        # mask function parameters
                        p_threshold=p_th,
                        # policy=policy,   # filtering direction (e.g. user axis, item axis)
                        
                        is_cascade=params['is_cascade'],
                        policy_test=params['policy_test'],  # only relevant when 'is_cascade' is True
                        n_train=params['n_train'],

                        policy_threshold=params['policy_threshold'], 

                        ############# deprecated ############## 
                        ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 
                        ratio_users=params['ratio_users'], 
                        supervised=params['supervised'], # if len(L) > 0 else False? L can be estimated
                        ####################################### 

                        # weight
                        # alpha=params['alpha'],
                        # beta=params['beta'],

                        # balance class sample distribution and weights
                        # balance_class=params['balance_class'],
                        # balance_and_scale=params['balance_and_scale'],
                        # suppress_negative_examples=params['suppress_negative_examples'],
                        # policy_polarity=params['policy_polarity'],   # options: classification, median
                        estimated_labels=params['estimated_labels'],
                        
                        # polarity matrix parameters
                        # labeling_model=params.get('labeling_model', 'simple'),  # used to determine polarity matrix
                        # constrained=params.get('constrained', True),
                        # stochastic=params.get('stochastic', True), 
                        # estimate_sample_type=params.get('estimate_sample_type', True),

                        # message passing 
                        message=M,   # 2-tuple (R, L_train) or 3-tuple (R, L_train, Cr)
                    
                        # outdated
                        # mask_all_test=params['mask_all_test'], 

                        # debug/testing
                        U=kargs.get('U', []),
                        L_true=kargs.get('L_test', []),  # test the accuracy of the unsupervised estimte of labeling 
                        fold=fold, 
                        path=kargs.get('path', os.getcwd()), 
                        verbose=verbose 
                        )  
    Cui, Pc, p_th, *rest = ret

    return (Cui, Pc, p_th)

def evalConfidenceMatrices(X, L, alpha=10.0, p_threshold=[], conf_measure='brier', policy_threshold='fmax', **kargs): 
    """
    Compute confidence matrices in the following format: 

    C0: Confidence matrix with raw confidence scores (as determined by the given confidence measure `conf_measure`)
    Cw: Re-weighted confidence matrix 
    Cn: Masked confidence matrix

    """
    # Optional parameters
    ##################################################
    fold_number = kargs.get('fold_number', 0) # for debugging only
    n_train = kargs.get('n_train', -1) # used to separate X into R and T, from which to estimate `p_threshold`
    verbose = kargs.get('verbose', 0)
    U = kargs.get("U", []) # the set of users/classifiers; for debug/test only
    is_cascade = kargs.get('is_cascade', True) # True of X contains both R and T; false otherwise
    ##################################################

    CX = evalConfidenceMatrix(X, L=L, U=U, 
                                 p_threshold=p_threshold, # Optional: Not needed if L is given (suggested use: estimate L outside of this call)
                                 policy_threshold=policy_threshold,
                                 conf_measure=conf_measure, 
                                 fill=0, is_cascade=is_cascade, n_train=n_train, 
                                 fold=fold_number, 
                                 verbose=verbose) 
    C0, Pc, p_threshold, *CX_res = CX

    # Cw: A re-weighted (dense) confidence matrix in which confidence scores are adjusted to take into account 
    #     the disparity in sample sizes (e.g. the size of TPs is usually much smaller than that of TNs in class-imbalanced data)
    Cw = balance_and_scale(C0, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, 
                        alpha=alpha, conf_measure=conf_measure, n_train=n_train, verbose=verbose)

    # Cn: A masked confidence matrix where the confidence scores associated with FPs and FNs are set to 0
    Cn = mask_neutral_and_negative(C0, Pc, is_unweighted=False, weight_negative=0.0, sparsify=True)
    Cn = balance_and_scale(Cn, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, 
                        alpha=alpha, conf_measure=conf_measure, n_train=n_train, verbose=verbose)
    
    # Test: Wherever Pc is negative, the corresponding entries in Cn must be 0 (By constrast, C is a full/dense confidence matrix)
    assert np.all(Cn[Pc < 0]==0)
    assert np.all(Cn[Pc > 0]>0)

    # Color matrix should have 4 distinct values
    uniq_colors = np.unique(Pc.A if is_sparse(Pc) else Pc)
    assert len(uniq_colors) >= 4, f"n_colors: {uniq_colors}"

    return (Pc, C0, Cw, Cn)

def analyzeMask(R, M, L, pos_label=1, neg_label=0):
    """
    Get summary statistics of the training split R and its mask. 

    Use
    ---
    1. Estimate labels in the test split (T) under the 'neighborhood' policy

    """
    # R: rating/prob matrix, n_users by n_items 
    # M: mask matrix 
    # L: labels
    assert R.shape[1] == len(L)
    assert R.shape == M.shape
 
    ret = classPrior(L, labels=[neg_label, pos_label], ratio_ref=0.1)
    rPos, rNeg = ret[pos_label], ret['neg_label']
    rMinority = min(rPos, rNeg)
    k = rMinority * R.shape[1] 

    ret = {i: {} for i in range(R.shape[0])}   # user i -> label -> representative probs
    for i in range(R.shape[0]): # foreach user/classifier

        for label in [neg_label, pos_label, ]: 

            cols_selected = np.where(M[i, :] == False)  # find columns/items selected (i.e. entries with good probs) 
            cols_matched = np.where(L == label)[0]
            cols_repr = list(set(cols_selected).intersection(cols_matched))  
            assert len(cols_repr) > 0, "User/classier #%d does not represent any examples of label: %s!" % (i, label)
            
            # if negative, reverse <- False => ascending order => topk lowest
            # if positive, reverse <- True => descending order => topk highest
            probs = sorted(R[i, cols_repr], reverse=bool(label))[:k] # topk highest/representative probs
            ret[i][label] = probs

    return ret # a dictionary: user i -> label -> representative probs

def maskEntriesItemAxis(R, L=[], p_th=[], T=None, ratio_users=0.5, policy='', pos_label=1, ratio_small_class=0, suppress_max_class=False): # cf In mask_along_user_axis, params: ratio_small_class  
    ### verify the policy
    if not policy: 
        if len(L) > 0: 
            # Given L, p_th given externally? or p_th inferred from stats of L (e.g. class prior)

            # 1. L + p_th
            if hasattr(p_th, '__iter__') and len(p_th) > 0:  # 'thresholding+aggregate', 'thresholding'
                # supervised (using fmax(L) to pre-compute prob threshold)
                # >>> note that it's possible that extra instruction is given in the policy string (e.g. 'thresholding+aggregate'); 
                #     if not use the default 'thresholding'
                policy = 'thresholding'  
                assert len(p_th) == R.shape[0], "Each user/classifier has its own threshold. n(p_th): %d but n_users: %d" % (len(p_th), R.shape[0])
            elif isinstance(p_th, float): # heuristic, unsupervised but not practical
                raise ValueError("Policy with a fixed prob threshold is not recommended (p_th={0}".format(p_th))

            elif len(p_th) == 0: 
                p_th = estimateProbThresholds(R, L=L, pos_label=pos_label, policy='prior')
                policy = 'thresholding'
                # ... no need to use 'prior' mode
        else: 
            # parameter: ratio_small_class: i) estimated externally ii) class prior 
            policy = 'ratio'  # more conservative, consider upper top k and lower k
    print('(maskEntriesItemAxis) policy_threshold: {policy} ... (verify)'.format(policy=policy))
    mask = np.ones(R.shape, dtype=bool) # all True 2D array (True: retain, False: zero out or set to a fill value)
    if policy.startswith( ('thre', 'pri', )):   # label must have been given
        # assert len(L) > 0
        
        # estimated per-classifier thresholds (along the user axis) given earlier
        
        mask = mask_along_item_axis_given_labels(R, L, p_th=p_th, ratio_users=ratio_users, pos_label=pos_label, include_negative=suppress_max_class)
    elif policy == 'polarity':
        # require separating R and T

        # R, L, p_th, T, ratio_users=0.5, pos_label=1, verbose=True, include_negative=True
        mask = mask_along_item_axis_by_polarity(R, L, p_th=p_th, T=T, pos_label=pos_label, verbose=True)

    elif policy.startswith('ratio'):
        # no L was given 
        # p_th = estimateProbThresholds(R, L=[], pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class) 
        # mask = mask_along_item_axis_no_labels(R, ratio_users=0.5, pos_label=1)
        msg = "(maskEntriesItemAxis) Use estimateLabels() to obtain a label vector (lh) followed by mask_along_item_axis_given_labels()"
        raise ValueError(msg)
    else: 
        raise NotImplementedError('Unrecognized policy: {policy}'.format(policy=policy))

    return mask
####################################################
# ... alias 
mask_along_item_axis = maskEntriesItemAxis
####################################################

def mask_along_item_axis_no_labels(R, p_th=[], ratio_users=0.5, pos_label=1): 
    """


    Params
    ------
    p_th: probability thresholds 
          <todo>

    Memo
    ----
    1. domain: diabetes

       (mask_along_item_axis_no_labels) Found 154 under-represented items and 0 over-represented items (wrt k_min=3), ratio (under-represented): 1.000000
       ... masked 4312 entries in 'X' (mtype=?) | selected 308 entries (out of 4620, ratio: 0.06666666666666667 => ~ 2.0/item)  ... (verify) #

       ... Cui |  dim(Cui): (30, 154), n_zeros: 4312 vs nonzeros: 308 (masked ratio=0.9333333333333333)
       => most entries are masked in T

       ... n_candidates_per_item on average: 2.0 | min: 2, max: 2
       
    """
    mask = np.ones(R.shape, dtype=int) # all True 2D array (True: retain, False: zero out or set to a fill value)

    n_users = R.shape[0]

    k = select_k(n_users, ratio_users, r=1.0) # min(topk, n_users/2)
    k_max = min(k, math.ceil(n_users/2.))
    # k_min = max(1, math.ceil(n_users/10)) # each datum should at least have this many BP predictions
    k_min = 1 # max(1, math.ceil(n_users/10)) # each datum should at least have this many BP predictions
    print('(mask_along_item_axis_no_labels) k: {k}, n_users: {nu} | range: {min} ~ {max}'.format(k=k, nu=n_users, min=k_min, max=k_max))

    n_under_repr = n_over_repr = 0
    n_candidates_per_item = []  # debug

    for j in range(R.shape[1]):  # foreach column/item

        # now this cannot be computed
        # rows_aligned = np.where(Lh[:, j] == L[j])[0]  # Lh[:,j]=L[j] => boolean mask, np.where(...) gives indices where condition holds
        
        # indices of k highest probs vs 
        #    indices of k lowest probs
        kp = max(1, int(k/2)) # min(1, int(k_min)) 
        upper_half = np.argsort(R[:, j])[:-kp-1: -1]   # indices of the top half
        lower_half = np.argsort(R[:, j])[:kp]   # indices of the lower half 
        rows_candidates = np.hstack( (upper_half, lower_half) )
        # ... it's ok for duplicate row indices (i.e. when top half and bottom half have overlaps)

        # if len(rows_aligned) > k:  # too many (and all consistent)

        #     # rows_candidate | [r for r in rows_candidates if r in rows_aligned][:k]
        #     # nothing much to do because we do not know if these choices are aligned or not as in the case of ...
        #     # ... mask_along_item_axis_given_labels(R, L, Lh, ratio_users=0.5, pos_label=1)

        #     n_over_repr += 1
        # else: # too few that are consistent
        #     # keep all in rows_good + pad extra BP predictions that are not correct but perhaps close enough
            
        #     # nothing much to do
        #     n_under_repr += 1

            # [condition] it's possible that the number of the label-aligned candidates are too few to even reach k_min

        # >>> see condition 1 
        # assert len(rows_candidates) >= k_min, "rows_aligned: %d, rows_candidates: %d, k_min: %d" % (len(rows_aligned), len(rows_candidates), k_min)
        if not (len(rows_candidates) >= k_min): 
            msg = "Warning: Not enough row candidates | rows_candidates: %d, k_min: %d" % (len(rows_candidates), k_min)
            # div(msg, symbol='%', border=2)
            n_under_repr += 1
        else:
            n_over_repr += 1

        # [test]
        n_candidates_per_item.append(len(rows_candidates))
        assert len(rows_candidates) <= k

        mask[rows_candidates, j] = 0  # False/0 => don't mask

    print('(mask_along_item_axis_no_labels) Found %d under-represented items and %d over-represented items (wrt k_min=%d), ratio (under-represented): %f' % \
        (n_under_repr, n_over_repr, k_min, n_under_repr/(R.shape[1]+0.0)) )
    print('... n(users/classifeirs) per item on average: {avg} | min: {min}, max: {max}'.format(avg=np.mean(n_candidates_per_item), min=min(n_candidates_per_item), max=max(n_candidates_per_item) ))

    return mask

def select_k(N, k, r=1.0):
    if k < 1.0 and k > 0.0: 
        return math.ceil(N * k)
    if r > 1.0 or r < 0.0: r = 1.0

    # k is an integer
    assert isinstance(k, int)
    return min(k, int(N*r))

def mask_along_item_axis_by_polarity(R, L, p_th, T, Lt=None, C=None, U=None, labeling_model='simple', ratio_users=0.5, 
        constrained=True, stochastic=True, estimate_sample_type=True, policy_polarity='sequence',
            pos_label=1, verbose=True, include_negative=True, index=0): 
    """

    Params
    ------
    Lt: labels of the test data; only used for testing
    """

    # k_ref = math.ceil(ratio_users * n_users) # select_k(n_users, ratio_users, r=1.0) # min(topk, n_users/2)
    assert R.shape[0] > 2
    n_users = R.shape[0]
    halfu = int(R.shape[0]/2)

    ratio_users_upper, ratio_users_lower = ratio_users/2.0, ratio_users/2.0
    k_upper = min(halfu, max(1, math.floor(ratio_users_upper * n_users)))  #  don't go over 50%
    # ... inner max ensure that k_upper >= 1, outer min ensures that k_upper <= n_users/2
    k_lower = min(halfu, max(1, math.floor(ratio_users_lower * n_users)))

    k_max = k_upper+k_lower
    k_min = 1   # min(k_upper)+min(k_lower)

    n_under_repr = n_over_repr = 0
    n_candidates_per_item = []  # debug
    n_supports_per_item = []
    stats_support = {stype: 0 for stype in ['+', '-', 'o']}  # number of supports by types
    pos_class_padding = {stype: 0 for stype in ['+', '-', 'o']}
    neg_class_padding = {stype: 0 for stype in ['+', '-', 'o']}

    # estimate polarity based on training data statistics
    msg = '' 
    msg += '(mask_along_item_axis_by_polarity) labeling model: {} | constrained? {}, stochastic? {} | k_lower:{}, k_upper:{} | n_users: {}, k_max: {}\n'.format(labeling_model, constrained, stochastic, k_lower, k_upper, n_users, k_max)
    mask = estimate_polarity(R, L, p_th, T, 
                Lt=Lt,   # test labels; only used for testing & evaluation
                C=C, 
                U=U, 
                policy=policy_polarity,  # policy for determiing entry type based on the training data statistics and probability values
                    labeling_model=labeling_model,   # chooses how we model/estimate labeling, which then determines the polarity matrix
                    stochastic=stochastic, 
                    constrained=constrained, # if True, use k_upper, k_lower to put constraints on the sample selection
                        estimate_sample_type=estimate_sample_type,
                        k_upper=k_upper, k_lower=k_lower, k_max=k_max, k_min=k_min, index=index)
    # hard ... 
    # soft ... by fitting a beta distribution and then predict probability of being 1, -1, 0
    nu = len(np.unique(mask))
    if nu <= 3: tHardEstimate = True

    # given proba thresholds, estimate label matrix
    # Lh = estimateLabelMatrix(T, p_th=p_th, pos_label=pos_label)  # foreach class, use the topk probs in the horizontal direction as a proxy for positive labels
    
    # msg = '(mask_along_item_axis_by_polarity) Hard estimate\n'
    # no-op 
    for j in range(T.shape[1]):
        support_pos = np.where(mask[:, j] > 0)[0]
        support_neg = np.where(mask[:, j] < 0)[0]
        uncertain = np.where(mask[:, j] == 0)[0]
        n_supports = len(support_pos) + len(support_neg)
        n_supports_per_item.append(n_supports)

        stats_support['+'] += len(support_pos)
        stats_support['-'] += len(support_neg)
        stats_support['o'] += len(uncertain)

        # if n_candidates > k_max:
        #     support_pos = np.argsort(-mask[:, j])[:k_upper]   # indices of the top half
        #     support_neg = np.argsort(mask[:, j])[:k_lower] # indices of the lower half

    if verbose: 
        # msg += '(mask_along_item_axis_by_polarity) Found {} under-represented items and {} over-represented items (wrt k: [{}, {}]), ratio (under-represented): {}\n'.format(
        #             n_under_repr, n_over_repr, k_lower, k_upper, n_under_repr/(R.shape[1]+0.0) )
        # msg += '... n(raw candidates) | min={min}, max={max}, median={m}\n'.format( min=min(n_candidates_per_item), max=max(n_candidates_per_item), m=np.median(n_candidates_per_item))
        msg += "... n(support)(+,-,o)        | min={min}, max={max}, median={m}, examples:\n... {ex}\n".format( min=min(n_supports_per_item), max=max(n_supports_per_item), m=np.median(n_supports_per_item), ex=n_supports_per_item[:20])
        msg += '... n(pos): {np}, n(neg): {nn}, n(neutral): {nc}\n'.format(np=stats_support['+'], nn=stats_support['-'], nc=stats_support['o'])
        # msg += '...... ratio(Lh ~ L)     | min: {}, max: {}, median: {}, examples:\n... {}\n'.format(min(ratios_label_aligned), max(ratios_label_aligned), np.median(ratios_label_aligned), ratios_label_aligned[:20] )
        print(msg)

    return mask

def mask_along_item_axis_given_labels(R, L, p_th, ratio_users=0.5, pos_label=1, verbose=True, include_negative=True): 
    """

    Memo
    ---- 
    1. domain: diabetes

       ... n_candidates_per_item on average: 13.980456026058633 | min: 3, max: 15

    2. For test split, L is an estimated label vector (e.g. via applying majority vote)

    """
    import numpy as np
    # introducing colored particles
    # mask = np.ones(R.shape, dtype=bool) # all True 2D array (True: retain, False: zero out or set to a fill value)
    mask = np.zeros(R.shape, dtype=int)

    n_users = R.shape[0]
    halfu = int(R.shape[0]/2)

    # k_ref = math.ceil(ratio_users * n_users) # select_k(n_users, ratio_users, r=1.0) # min(topk, n_users/2)
    assert R.shape[0] > 2
    ratio_users_upper, ratio_users_lower = ratio_users/2.0, ratio_users/2.0
    k_upper = min(halfu, max(1, math.floor(ratio_users_upper * n_users)))  #  don't go over 50%
    # ... inner max ensure that k_upper >= 1, outer min ensures that k_upper <= n_users/2
    k_lower = min(halfu, max(1, math.floor(ratio_users_lower * n_users)))

    k_max = k_upper+k_lower
    k_min = 2 # max(1, math.ceil(n_users/10)) # each datum should at least have this many BP predictions
    print('(mask_along_item_axis_given_labels) k_lower:{}, k_upper:{} | n_users: {}, k_max: {}'.format(k_lower, k_upper, n_users, k_max))

    n_under_repr = n_over_repr = 0
    n_candidates_per_item = []  # debug
    n_supports_per_item = []
    stats_support = {stype: 0 for stype in ['+', '-', 'o']}  # number of supports by types
    pos_class_padding = {stype: 0 for stype in ['+', '-', 'o']}
    neg_class_padding = {stype: 0 for stype in ['+', '-', 'o']}
    # n_support = k_max

    # ratio_user_per_item = confidence_pointwise_ensemble_prediction(R, L, p_th, mode='item', suppress_majority=False)
    # print('(mask_along_item_axis_given_labels) ratio_user_per_item:\n... {}\n'.format(np.random.choice(ratio_user_per_item, 20)))
    
    # given proba thresholds, estimate label matrix
    Lh = estimateLabelMatrix(R, p_th=p_th, pos_label=pos_label)  # foreach class, use the topk probs in the horizontal direction as a proxy for positive labels

    ratios_label_aligned = []
    tPolarityNegative = True # include_negative
    tPadding = True
    # tBalanced = True
    tUnconstrained = False
    for j in range(R.shape[1]):  # foreach column/item

        # positive examples
        rows_aligned = np.where(Lh[:, j] == L[j])[0]  # Lh[:,j]=L[j] => boolean mask, np.where(...) gives indices where condition holds

        # negative examples
        rows_misaligned = np.where(Lh[:, j] != L[j])[0]

        k = k_hit = len(rows_aligned)  # select_k(n_users, ru, r=1.0)      # [note] instead of using ratio_users/0.5, use ru, which differs by items
        k_miss = len(rows_misaligned)

        ru = k/(n_users+0.0)
        ratios_label_aligned.append(ru)
        
        # if positive: choose indices of k highest probs
        # if negative: choose indices of k lowest probs  
        Np = Nn = 0  # 
        support_pos, support_neg = [], [] 
        if L[j] == pos_label: 
            rows_sorted = np.argsort(-R[:, j])  # np.argsort(R[:, j])[:-k-1: -1] # choose indices of k highest probs
            # ... row indices in descending order
            if tUnconstrained: k_upper = len(rows_sorted)

            ### choose positive examples; aligned first 
            support_pos = [r for r in rows_sorted if r in rows_aligned[:k_upper]] # aligned indices in decending order
            Np = len(support_pos)

            n_padding = 0 
            if tPadding: 
                # if Np < k_upper: n_padding = k_upper-Np  
                if Np < k_min: n_padding = k_min-Np  # n_to_pads
                
                # try to choose FN entries with higher probabilities 
                rows_extra = [r for r in rows_sorted if not (r in support_pos)][:n_padding]
                # ... rows_extra could be in rows_aligned or rows_misaligned

                support_pos.extend(rows_extra)

                # [test]
                pos_class_padding['+'] += len(rows_extra)
            # ... support(+) for positive examples

            ### choose negative examples
            #   a. balanced positive and negative 
            #   b. restricted negative   => high recall, but precision at 0.2-0.3, sample_size(positive) small
            #   c. no negative 

            # if mis-aligned, then they are false negatives (i.e. FNs) => sort in ascending order instead
            rows_sorted = np.argsort(R[:, j])  # np.argsort(R[:, j])[:k_miss]
            # ... row indices in ascending order
            if tUnconstrained: k_lower = len(rows_sorted)
            
            support_neg = []
            if tPolarityNegative:  # include negative examples
                support_neg = [r for r in rows_sorted if r in rows_misaligned[:k_lower]]
                Nn = len(support_neg)

                n_padding = 0
                if tPadding:
                    # if Nn < k_lower: n_padding = k_lower-Nn
                    # ... but we may not want to pad so many (-)

                    # try to choose entries with lower probabilities
                    # ... pad extra (-) only when Nn is too small?
                    if Nn < k_min: n_padding = k_min - Nn
                    
                    rows_extra = [r for r in rows_sorted if not (r in support_neg)][:n_padding]
                    # ... rows_extra could be in aligned or misaligined but their proba values should be low

                    support_neg.extend(rows_extra)
                    pos_class_padding['-'] += len(rows_extra)
                
            # ... support(-) for positive examples 

        else: # L[j] == neg_label
            rows_sorted = np.argsort(R[:, j]) # np.argsort(R[:, j])[:k] # choose indices of k lowest probs 
            # ... low -> high
            if tUnconstrained: k_upper = len(rows_sorted)

            ### positive examples for negative labels
            support_pos = [r for r in rows_sorted if r in rows_aligned[:k_upper]] 
            Np = len(support_pos) 

            n_padding = 0
            if tPadding: 
                # if Np < k_upper: n_padding = k_upper-Np
                if Np < k_min: n_padding = k_min-Np  
                
                # try to choose entries with lower probabilities 
                rows_extra = [r for r in rows_sorted if not (r in support_pos)][:n_padding]
                support_pos.extend(rows_extra)

                # [test]
                neg_class_padding['+'] += len(rows_extra)
            # ... support(+) for negative labels

            ### negative examples for negative labels (then they should have been positive, FPs)
            rows_sorted = np.argsort(-R[:, j]) # np.argsort(R[:, j])[:-k-1: -1] 
            # ... choose indices of k highest probs (since they are of a further departure from true label as being negative)
            # ... high to low
            if tUnconstrained: k_lower = len(rows_sorted)

            support_neg = [] 
            if tPolarityNegative: 
                support_neg = [r for r in rows_sorted if r in rows_misaligned[:k_lower]]
                Nn = len(support_neg)

                n_padding = 0
                if tPadding: 
                    # if Nn < k_lower: n_padding = k_lower-Nn
                    # ... but we may not want to pad so many (-)

                    # try to choose among TNs but with higher probabilities 
                    if Nn < k_min: n_padding = k_min - Nn

                    rows_extra = [r for r in rows_sorted if not (r in support_neg)][:n_padding]
                    support_neg.extend(rows_extra)

                    # # [test]
                    neg_class_padding['-'] += len(rows_extra)
            # ... support(-) for negative labels
    
        # rows_sorted = np.argsort(R[:, j])[:-k-1: -1] if L[j] == pos_label else np.argsort(R[:, j])[:k] 
        # ... all candidates
        stats_support['+'] += len(support_pos)
        stats_support['-'] += len(support_neg)
        support = np.hstack([support_pos, support_neg])  

        ns = len(support)
        if ns < k_upper+k_lower: 
            n_under_repr += 1
        elif ns > k_upper+k_lower: 
            n_over_repr += 1
        # ... all candidates that are label-algined

        # clip
        assert ns <= k_max, "Abnormally high support (ns: {})".format(ns)
        # if ns > k_max:  # too many (and all consistent)
        #     support = support[:k_max]  # indices aligned with (estimated) labels
        #     n_over_repr += 1

        # pad 
        assert ns >= k_min, "Abnormally low support (ns: {})".format(ns)
        # if ns < k_min: # too few that are consistent
        #     # keep all in rows_good + pad extra BP predictions that are not correct but perhaps close enough

        #     # need at least k_min BP predictions
        #     residual = k_min - ns
        #     if residual > 0: 
        #         rows_extra = [r for r in rows_sorted if r not in rows_aligned][:residual]
        #         support = np.hstack( (rows_extra, support) ) # pad the top choices
        #     n_under_repr += 1

        #     # [condition] it's possible that the number of the label-aligned candidates are too few to even reach k_min

        # >>> see condition 1 
        # assert len(rows_candidates) >= k_min, "rows_aligned: %d, rows_candidates: %d, k_min: %d" % (len(rows_aligned), len(rows_candidates), k_min)
        if not (len(support) >= k_min): 
            msg = "Warning: Not enough row candidates | label-aligned: %d, support: %d, k_min: %d" % (len(rows_aligned), len(support), k_min)
            div(msg, symbol='%', border=2)

        # n_support
        n_candidates_per_item.append(len(rows_aligned))  # raw support before clipping or padding
        n_supports_per_item.append(ns)  # final number of support user/classifier for each item, from which we'll use to predict the labels

        # masking 
        mask[support_pos, j] = 1   # positive
        mask[support_neg, j] = -1   # negative
        uncertain = list(set(range(R.shape[0]))-set(support))
        stats_support['o'] += len(uncertain)
        # neutral particles remain 0; they are used to adjust confidence weights
        # mask[uncertain, j] = 0   # neutral 
    ###
    # end foreach item R[:, j]
    
    if verbose: 
        msg = ''
        msg += '(mask_along_item_axis_given_labels) Found {} under-represented items and {} over-represented items (wrt k: [{}, {}]), ratio (under-represented): {}\n'.format(
                    n_under_repr, n_over_repr, k_lower, k_upper, n_under_repr/(R.shape[1]+0.0) )
        msg += '... n(raw candidates) | min={min}, max={max}, median={m}\n'.format( min=min(n_candidates_per_item), max=max(n_candidates_per_item), m=np.median(n_candidates_per_item))
        msg += "... n(support)(+,-,o)        | min={min}, max={max}, median={m}, examples:\n... {ex}\n".format( min=min(n_supports_per_item), max=max(n_supports_per_item), m=np.median(n_supports_per_item), ex=n_supports_per_item[:20])
        msg += '... n(pos): {np}, n(neg): {nn}, n(neutral): {nc}\n'.format(np=stats_support['+'], nn=stats_support['-'], nc=stats_support['o'])
        msg += '...... ratio(Lh ~ L)     | min: {}, max: {}, median: {}, examples:\n... {}\n'.format(min(ratios_label_aligned), max(ratios_label_aligned), np.median(ratios_label_aligned), ratios_label_aligned[:20] )
        msg += '........ pos_class_padding(+): {}, pos_class_padding(-): {}\n'.format(pos_class_padding['+'], pos_class_padding['-'])
        msg += '........ neg_class_padding(+): {}, neg_class_padding(-): {}\n'.format(neg_class_padding['+'], neg_class_padding['-'])
        print(msg)

    return mask  # positive: 1, negative: -1, neutral: 0

# mask function for computing Cui
def filter_along_item_axis(C, X, L=[], policy='', ratio_users=0.5, ratio_small_class=0, factor_small_class=1.0, 
        supervised=True, mask_all_test=True, **kargs):
    """
    Compute the mask for C using item-centered policy. In item-centered policy, each item/datum must be represented by at least k users/classifiers, 
    meaning that we must have at least k probability estimtes for each data point. Ideally these k probabilities should be as reliable as possible. 
    
    The non-selected entries are then masked, which will still go into the optimization objective but with only a base confidence score. 

    How do we determine this k? The answer is to find the top k most reliable probability/rating estimates, which we call "good entries."

    Select good entries in the rating matrix. 

    Params
    ------

    <not used>
    ratio_small_class: if 0, infer from class prior and the factor_small_class 
                       if factor_small_class > 1, then we estimate 'reliable probabilities' by a ratio smaller than the proportion of 
                       the minority class; i.e. in order to be considered as 'reliable probabilties' (for positive sample)
                       (P(y=1|x), we need to look at the the top fraction (=ratio_small_class/factor_small_class) of 
                       the conditional probability estimates. 

    """ 
    pos_label, neg_label = kargs.get('pos_label', 1), kargs.get('neg_label', 0)
    marker = kargs.get('marker', 0)
    fold = kargs.get('fold', -1)  # either a CV fold number or an interation index in random subsampling
    tEstimatedLabels = kargs.get('estimated_labels', False)  # if X comes from a test data, then L must have been estimated. 

    # topk: topk classifiers
    if isinstance(X, tuple): 
        update_msg = "X now can be either R, T but not both (and in general any matrix) with the same dimensionality as C"
        raise ValueError(update)
    else: 
        assert C.shape == X.shape
    
    # L can only contain training set labels
    n_users = X.shape[0]
    
    ### parameters
    assert ratio_users <= 1.0 and ratio_users > 0.0, '(filter_along_item_axis) Invalid ratio_users: %f' % ratio_users
    k = select_k(n_users, ratio_users, r=1.0) # min(topk, n_users/2)
    k_max = max(2, math.ceil(ratio_users * n_users)) # min(k, math.ceil(n_users/2))
    k_min = 1 # max(1, math.ceil(n_users/10)) # each datum should at least have this many BP predictions

    # mean_predictions = np.mean(R, axis=0)

    ## Training split R
    tSupervised = False
    print('(filter_along_item_axis) supervised? {}, labels given? {}'.format(supervised, len(L) > 0))
    if supervised and len(L) > 0:  # this means that we can also set supervised to False to ignore L
        tSupervised = True
        assert len(L) == X.shape[1], "dim(X): {0} != size(L): {1}".format(X.shape, len(L))
        if not policy: policy = 'prior'
    else: 
        tSupervised = False
        if not policy: policy = 'ratio'
    assert tSupervised, "Consider only supervised approach now because L can be Lh"

    ## determine k and appropriate minority class ratio
    statsR = ret = classPrior(L, labels=[neg_label, pos_label], ratio_ref=0.1) # proportion of the minority class smaller than a threshold (e.g. 0.1)? 
    rPos, rNeg = ret[pos_label], ret[neg_label]
    rMinority = min(rPos, rNeg)

    # [test]
    if not tEstimatedLabels: 
        assert rPos < rNeg, "Not likely in this bio experiment (rPos: %f < rNeg: %f ?)" % (rPos, rNeg)

    # use this to fix 'inappropriate' ratio_small_class
    if (ratio_small_class <= 0) or (ratio_small_class > rMinority): 
        print('(filter_along_item_axis) Adjusting minority class ratio %f -> %f (initial est. is too BIG)' % (ratio_small_class, rMinority))
        ratio_small_class = rMinority/factor_small_class

    mode = 'supervised' if tSupervised else 'unsupervised'
    print('(filter_along_item_axis) Each item is to be represented by %d (/%d) BP predictions k_min: %d, k_max: %d) | mode: %s, ratio_users: %f, (prior: %f)' % \
        (k, n_users, k_min, k_max, mode, ratio_users, rMinority))
    assert mode == 'supervised', "Only use supervised mode now because L can be estimated even if not available"

    thresholds = kargs.get('p_threshold', [])
    mask = np.ones(X.shape, dtype=int) # all True 2D array (True: retain, False: zero out or set to a fill value)
    
    if not tSupervised: 
        print("(filter_along_item_axis) Fold={i} | Filtering along item axis ((UnSupervised)) | policy (for p-threshold): {p}, ratio_small_class: {r} #".format(i=fold, 
                p=policy, r=ratio_small_class))

        # policy: ratio attempts to find the 'k' largest probabilities for each user/classifier, the lowest of which becomes the 'threshold' for the positive class
        #         how about the threshold for the negative class? 
        #         a. any values lower than the positive threshold is considered negative 
        #         b. we could use a more conservative measure by saying that the k lowest prob values are likely to be negative. Any values that fall between the k largest and k lowest are 
        #            considered to be ambiguous
        # params: ratio_small_class
        # thresholds = estimateProbThresholds(R, L=[], pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class) 
        # Lh = estimateLabelMatrix(R, p_th=thresholds, pos_label=pos_label)
    
        mask = maskEntriesItemAxis(X, L=[], p_th=thresholds, ratio_users=ratio_users, policy='ratio', pos_label=1)

        # >>> maskTestEntries() uses the same logic: i) estimate labels (via thresholds derived from the training set ...
        #     ... followed by majoriy vote to turn Lh, a matrix, to lh, a vector)
        # lh = estimateLabels(R, p_th=thresholds, pos_label=pos_label)   # unlike Lh, 'lh' is a vector after applying a combining strategy (which is done because we do not have access to true labels in this case)
        
        # A. majority vote
        # assuming that the top k probabilities (where k = n_pos in L) correspond to the true positives
        # for j in range(R.shape[1]):   
        #     # if positive, select highest probs; if negative, select lowest
        #     #    R[:, i][np.argsort(R[:, i])[:-k-1: -1]] => top tops in descending order
        #     rows_active = np.argsort(R[:, j])[:-k-1: -1] if lh[j] == pos_label else np.argsort(R[:, j])[:k]
        #     mask[rows_active, j] = False  # False => don't mask => selected

        # B. per-class view
        # Lh = estimateLabelMatrix(R, p_th=thresholds, pos_label=pos_label)  # foreach class, use the topk probs in the horizontal direction as a proxy for positive labels
        
    else: 
        print('(filter_along_item_axis) Filtering along item axis ((Supervised)) #')

        # if not policy: policy = 'prior'
        # # params: labels (L)
        # thresholds = estimateProbThresholds(R, L=L, pos_label=pos_label, policy=policy) # policy: {'prior'/'topk', 'fmax'}

        # steps: 1a. proba thresholds given? if so, use the pre-computed thresholds
        #        1b.                         if not, estimate the thresholds using class prior
        #        In case of 1b, labels (L) must be given: 
        #            Given label -> class prior -> prob thresholds -> Lh -> compare Lh with L -> select those consistent

        mask = maskEntriesItemAxis(X, L=L, p_th=thresholds, ratio_users=ratio_users, pos_label=1, policy=policy)
    
    ###
    # ... mask: positive/1, negative/-1, neutral/0

    mtype = kargs.get('mtype', '?')  # input matrix type: train, test, dev, generic, ... 
    n_negative_pref = np.sum(mask==-1)   # 1/True: masked, 0/False: retained
    n_pref = np.sum(mask==1)  # n_selected
    n_uncertain = np.sum(mask==0)
    # assert n_negative_pref > 0, "Must have negative preference examples (polarity: -)"
    assert n_pref > X.shape[1], "On average, every item/datum should have at least one positive support but n(pref): {}".format(n_pref)
    print("(filter_along_item_axis) Cycle: {} | n_masked: {}, n_pref: {}, n_uncertain: {} | ratio_pref: {}, n_selected_per_item: {}".format(
        fold, n_negative_pref, n_negative_pref, n_pref, n_uncertain, n_pref/(mask.shape[0]*mask.shape[1]+0.0), n_pref/(X.shape[1]+0.0)))

    # [old]
    # C[mask] = marker  # zero out (by value of 'marker') wherever mask[i,j] is True (or 1)
    return mask

# mask function for computing Cui
# user_centered_fitler, filter_along_user_axis
def filter_along_user_axis(C, X, L=[], policy='prior', ratio_small_class=0, factor_small_class=1.0, 
        pos_label=1, neg_label=0, marker=0, supervised=True, mask_all_test=True, **kargs): 
    """
    Compute the mask for C using user-centered policy. In user-centered policy, each user/classifier is responsible for representing 
    N data points. These N data points correspond to reliable probability/rating estimates for a given user/classifier. The strategy 
    for determine this N is to find the probability threshold for positive class and the negative class. 

    In unsupervised mode, we need to have an estimate of the proportion of the minority classs (ratio_small_class), which should be 
    equal or smaller than the true proportion of the minority class. Once this ratio is given, we can then compute the top k 
    probabilities, the bottom k probabilities and consider them as reliable estimates for positive and negative classes respectively. 
    Note that in this implementation, the unsupervised mode is really "weakly supervised" in the sense that we still need have a prior 
    knowledge about the proportional of the minority class and use it as a reference to correct our guess. We always want 'ratio_small_class'
    to be smaller or at most equal to this true minority class proportion. 

    Note that user-centered unsupervised mode is NOT the same as item-centered unsupervised mode. 

    In supervised mode, we use the true labels (L) to estimate the proportion of the minority class just like the unsupervised mode. This
    minority class proportion allows us to estimate the proba threshold for the positive class (because in our data set, the minority class
    is always the positive, which is harder to capture). Each user/classifier has a corresponding probability threshold that separates the
    positive from the negative. These classifier-specific thresholds allows us to estimate the labels that would be generated for each item/datum and
    each classifier has its own label estimates. This gives rise to the label matrix (Lh). By comparing Lh with L, we now have an estimate 
    for good entries vs bad entries. 

    Params
    ------
    ratio_small_class: if 0, infer from class prior and the factor_small_class 
                       if factor_small_class > 1, then we estimate 'reliable probabilities' by a ratio smaller than the proportion of 
                       the minority class; i.e. in order to be considered as 'reliable probabilties' (for positive sample)
                       (P(y=1|x), we need to look at the the top fraction (=ratio_small_class/factor_small_class) of 
                       the conditional probability estimates. 

    """
    # select entries of confidence matrix via topk probabiliteis, higheset for positive, lowest for negative
    pos_label, neg_label = kargs.get('pos_label', 1), kargs.get('neg_label', 0)
    marker = kargs.get('marker', 0)
    fold = kargs.get('fold', -1)  # either a CV fold number or an interation index in random subsampling
    balance_class = kargs.get('balance_class', False)
    tEstimatedLabels = kargs.get('estimated_labels', False)  # if X comes from a test set, then L must have been estimated. 
    
    if isinstance(X, tuple): 
        # R, T = X
        # assert C.shape[1] == R.shape[1]+T.shape[1], "n(C)=n_items(C):{0}, n(R):{1}, n(T):{2}".fomrat(C.shape[1], R.shape[1], T.shape[1])
        update_msg = "X now can be either R, T but not both (and in general any matrix) with the same dimensionality as C"
        raise ValueError(update)
    else: 
        # R = X
        assert C.shape == X.shape

    # L is only used to estimate class prior (proportion)
    # nPos = np.sum(L==pos_label)
    # nNeg = len(L) - nPos
    # k_high, k_low = min(nPos, k_high), min(nNeg, k_low)
    
    ### Training split R
    tSupervised = False
    if supervised and len(L) > 0: 
        tSupervised = supervised
        assert len(L) <= X.shape[1] 
        if not policy: policy = 'prior'
    else: 
        # must use a unsuerpvised method
        tSupervised = False
        if not policy: policy = 'ratio'  

    # In unsupervised mode, we still use the labels (L) to estiamte the minority class proportion ... 
    # ... but labels are not used to compute the mask
    mode = 'supervised' if tSupervised else 'unsupervised'
    ######################################################
    # policy for masking: i) prior in supervised mode ii) ratio in unsupervised mode
    
    ## determine k and appropriate minority class ratio
    statsR = ret = classPrior(L, labels=[neg_label, pos_label], ratio_ref=0.1) # proportion of the minority class smaller than a threshold (e.g. 0.1)? 
    rPos, rNeg = ret[pos_label], ret[neg_label]

    minClass, maxClass = statsR['min_class'], statsR['max_class']
    rMinority = min(rPos, rNeg)

    # [test]
    if tEstimatedLabels: 
        assert rPos < rNeg, "Not likely in bio experiment (rPos: %f < rNeg: %f ?)" % (rPos, rNeg)

    # use this to fix 'inappropriate' ratio_small_class
    if (ratio_small_class <=0) or (ratio_small_class > rMinority): 
        r0 = ratio_small_class
        ratio_small_class = rMinority/factor_small_class
        print('(filter_along_user_axis) Adjusting minority class ratio %f -> %f (minority class: %f) ... (verify) #' % \
            (r0, ratio_small_class, rMinority))

    ### estimate probability thresholds 
    #   policy: prior/topk, fmax, ratio
    #           where 
    #           prior: each classifier consider top k probabilities in R as positives 
    #           fmax
    #           ratio: a heuristic or an estimate for the proportion of the minority class (can be more conservative then prior)

    thresholds = kargs.get('p_threshold', [])  # probability thresholds, only used in unsupervised mode when attempting to pass the thresholds estimated from training split 
    Mc = np.ones(X.shape, dtype=int) # all True 2D array (True: retain, False: set to a fill value)
    if not tSupervised: # not supervised when L is not given
        # >>> not really "unsupervised" in the sense that we still use the labels L to estimate ratio_small_class

        # >>> [design] Should we have a separate threshold for positive and for negative? 
        # topk = ratio_small_class  # topk ratio is a nickname
        # if k_high is None: k_high = topk
        # if k_low is None: k_low = topk
        # k_high = select_k(R.shape[1], topk, r=1.0) 
        # k_low = select_k(R.shape[1], topk, r=1.0)  
        # print('(user_centered_select) k_high: {0}, k_low: {1}'.format(k_high, k_low))
        # print('... unsupervised: select {0} highest and {1} lowest probabilities to approximate, assuming they are more likely to be positives and negatives respectively'.format(k_high, k_low))
        
        # >>> probably not useful to estimtae the threshold in this case because we do not have access to L 
        #     so we cannot use the threshold to estimate labels and then compare the result with the true labels in L
        #     i.e. thresholds -> Lh, which allows as to compare L and Lh, which isn't the case here
        print("(filter_along_user_axis) Fold={i} | User-centered mask ((UnSupervised)) | p-th given? {tval} | ratio_small_class: {r} #".format(i=fold, 
                tval=True if len(thresholds) > 0 else False, r=ratio_small_class))
        # thresholds = estimateProbThresholds(R, L=[], pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class) # thresholds for positive classes
        # PS: Technically, there's another set of thresholds for the lowest probs

        # no labels, no thresholds => conservative estimate using ratio_small_class
        # steps: given (conservative) ratio -> (+, k highest proba) || (-, k lowest proba)
        #                                      ~ two sided filter
        Mc = maskEntries(X, L=[], p_th=thresholds, ratio_small_class=ratio_small_class)  # top-low symmetric (i.e. top k highest: '+' top k lowest: '-')
        # for i in range(R.shape[0]):
        #     k = k_high
        #     cols_high = np.argsort(R[i])[:-k-1: -1]  # indices of highest probabilities (these are likely to be better probability estimates of the positive)
        #     p_th = R[i, cols_high][-1]
        #     thresholds.append(p_th)

        #     k = k_low
        #     cols_low = np.argsort(R[i])[:k] # lowest probabilities (these are likely to be better estimates of the negative)

        #     cols_active = list(set(np.hstack((cols_high, cols_low))))
        #     mask[i, cols_active] = False # False: don't mask (want to retain these entries)
    else: 
        assert len(L) == X.shape[1]
        #   policy: prior/topk
        #           fmax

        print('(filter_along_user_axis) (fold={cycle}) classifier-centered masking ((Supervised))  #'.format(cycle=fold)) # [note] 'fold' is not local but in the closure
        # thresholds = estimateProbThresholds(R, L=L, pos_label=pos_label, policy=policy) # policy: {'prior'/'topk', 'fmax'}

        # assuming that the top k probabilities (where k = n_pos in L) correspond to the true positives
        # => mask the entries with inconsistent labeling according to the label estimate determined by the probabilty thresholds
        # steps: Given label vector -> class prior -> prob thresholds -> Lh -> compare Lh with L -> select those consistent
        #        if p_th is given, then it takes precedence
        Mc = maskEntries(X, L=L, p_th=thresholds, balance_class=balance_class, min_label=minClass) # use L to estimate p_th, which determines Lh; then compare Lh against L, keeping only those that are consistent
    ### 

    mtype = kargs.get('mtype', '?')  # input matrix type: train, test, dev, generic, ... 
    n_masked = np.sum(mask<=0)   # 1/True: masked, 0/False: retained
    n_selected = np.sum(mask==1)
    n_uncertain = 0 # np.sum(mask==0)
    assert n_masked > 0
    assert n_selected > X.shape[1]
    print("(filter_along_item_axis) Cycle: {} | n_masked: {}, n_selected: {}, n_uncertain: {} | ratio_selected: {}, n_selected_per_item: {}".format(
        fold, n_masked, n_masked, n_selected, n_uncertain, n_selected/(mask.shape[0]*mask.shape[1]+0.0), n_selected/(X.shape[1]+0.0)))     

    return mask

def toConfMatByTopProbs(X, L, **kargs):
    """

    Memo
    ----
    1. each user/classifier use 'top k' probabilities as representatives for positive sample: filter_along_user_axis()
            some of the data points may never be represeneted (or under-represented)
            i.e. no BP predictive info => cannot properly reconstruct their probabilities

    2. Example use cases: 

        toConfMatByTopProbs(fold, topk=0.1, kind='user', supervised=False)
        toConfMatByTopProbs(fold, topk=0.1, kind='user', supervised=True)
        toConfMatByTopProbs(fold, topk=0.1, kind='item', supervised=False) 

    """
    return toConfidenceMatrix(X, L, **kargs)

def mask_over(X, L, C=None, U=None, ratio_users=0.5, 
                ratio_small_class=0, factor_small_class=1.0, 

                p_threshold=[],  # used only for "messgae passing"
                messages={},     # a placeholder for more complex message passing

                kind='user', 
                supervised=True, mask_all_test=True, 

                n_train=-1, # split point by which X is separated into R and T
                test_labels=[], # true labels for the test set used for testing polarity matrix

                balance_class=False,
                estimated_labels=False, 
                    labeling_model='simple',  # options: 'simple', 'stacking' 
                    constrained=True, 
                    stochastic=True, 
                    estimate_sample_type=True,
                    policy_polarity='sequence',


                        pos_label=1, neg_label=0, marker=0, fold=-1): # mask C according to the values in R
    
    def run_analysis():  # closure 
        div("(mask_over.run_analysis) Examining 'masked' confidence matrix ...")
        assert np.sum(C == marker) > 0

        # wherever C is retained must correspond to correct labels if labels are given (training split)
        n_users, n_items = C.shape
        if len(L) > 0: 
            C_activated = C != marker
           
            user_support, item_support = {}, {}
            for i in range(C.shape[0]): # foreach user/classifier 
                if not i in user_support: user_support[i] = []
                user_support[i].append( C[i] )

    ### X: (R, T) or R (perhaps consider (R, ))
    #   if X <- R, then T will not be masked, meaning that all entries will be preserved
    #      X <- (R, T), then entries in T will be filtered according to the same rule as applied to R (i.e. rating matrix associated with the training split)

    N = len(L)
    n_users = C.shape[0] 
    labels = L # [] if not supervised else L

    ret = classPrior(L, labels=[neg_label, pos_label], ratio_ref=0.1, verbose=False)
    ratio_pos, ratio_neg = ret[pos_label], ret[neg_label]
    ratio_minority = min(ratio_pos, ratio_neg)

    # used for the 'unsupervised mode'
    if ratio_small_class <= 0 or ratio_small_class > ratio_minority: 
        r0 = ratio_small_class
        # if it was not specified, then the default should a ratio equal or more conservative than the minority class ratio
        ratio_small_class = min(ratio_pos, ratio_neg)/factor_small_class
        print('(mask_over) Adjusting minority class ratio %f -> %f (minority class: %f)' % (r0, ratio_small_class, ratio_minority))
    
    div('(mask_over) Using pre-computed proba thresholds? {} | (est) labels given? {} | supervised? {} ... (verify)'.format(
        len(p_threshold) > 0, len(labels) > 0, supervised)) 

    Mc = None
    if kind.startswith('i'):  # select entries item by item (this ensures that all data points are represented)
        # mode: supervised (passing labels), or unsupervised (not passing labels) but has an estimate of ratio_small_class
        
        # ratio_repr: 0.5 => each item/datum should be represnted by at most 50% of the BP predictive scores whenever possible ... 
        # ... sometimes less if enough BPs can be identified that produce consistent predictions with the true labels
        Mc = filter_along_item_axis(C, X, L=labels, 
                p_threshold=p_threshold,
                ratio_users=ratio_users, ratio_small_class=ratio_small_class, 
                    supervised=supervised, 
                        mask_all_test=mask_all_test, 
                         
                        # balance_class=balance_class,
                        estimated_labels=estimated_labels,

                        pos_label=pos_label, neg_label=neg_label, marker=marker, fold=fold)  # fold is only used for messaging and debugging
        # ... entries in Mc: {-1, 0, 1}
     
    elif kind.startswith('u'):  

        # if L is given, then ratio_small_class is ignored
        Mc = filter_along_user_axis(C, X, L=labels, 
                p_threshold=p_threshold,
                ratio_small_class=ratio_small_class, 
                    supervised=supervised, 
                    mask_all_test=mask_all_test, 

                    balance_class=balance_class, 
                    estimated_labels=estimated_labels,

                    pos_label=pos_label, neg_label=neg_label, marker=marker, fold=fold)
        # >>> In unsupervised mode, we still use the labels (L) to estiamte the minority class proportion ... 
        # ... but labels are not used to compute the mask

    elif kind.startswith('po'): # polarity 
        assert C is not None and C.shape == X.shape, "Invalid C (dim: {}) for the input X (dim: {})".format(C.shape if C is not None else 'n/a', X.shape)
        R, T = X[:,:n_train], X[:,n_train:]
        # Cr, Ct = C[:,:n_train], C[:,n_train:]
        Lr, Lt = L[:n_train], L[n_train:]
        # ... note that 'Lt' is just an estimted labeling but NOT true labels
        # ... to test polarity matrix using the true labels, pass 'test_labels' instead
        if len(test_labels) > 0: assert len(test_labels) == len(Lt), "expecting test labels to of the same size as Lt ({}) but got {}".format(len(Lt), len(test_labels))
        
        print('(mask_over) kind=polarity | constrained? {}, stochastic? {}, estimate_sample_type? {} | policy_polarity: {}'.format(constrained, 
            stochastic, estimate_sample_type, policy_polarity))
        Mc = mask_along_item_axis_by_polarity(R, Lr, p_th=p_threshold, T=T, 
                Lt=test_labels,   # test labels; only used for testing
                C=C,    # may be useful in polarity modeling
                U=U,
                labeling_model=labeling_model,
                constrained=constrained, stochastic=stochastic, 
                    estimate_sample_type=estimate_sample_type,
                    policy_polarity=policy_polarity,
                    ratio_users=ratio_users,
                    pos_label=pos_label, 
                        verbose=True, index=fold) 
        assert Mc.shape == T.shape
        # C = Ct 
    else:
        raise NotImplementedError('Unrecognized mask policy: %s' % kind)

    # masked C is the result of applying mask to original C
    # i.e. C[mask] = marker, where mask is a binary matrix; entries of 1s are masked, meaning that these 'ratings' are overwritten by marker
    
    ### interpreting polarity matrix Mc 
    # if C is None: 
    #     return Mc 

    # modulate confidence weights
    # C[Mc == 0] = 0.0 # neutral particles do not entire into the optimization objective

    # re-interpret polarity matrix so that 0: not preferred, 1: preferred
    # Mc[Mc == 0] = 0  # neutral particles are considered not preferred but it does't matter eventually
    # Mc[Mc == -1] = 0

    # [test] 
    # run_analysis()   
    # ... defer this to the main subroutine toConfidenceMatrix.balance_and_scale()

    return Mc  

# used in computing confidence matrix in 'tradeoff' mode (trade of between approximating probabilities and labels)
def mask_over_dual(C, X, L, ratio_users=0.5, 
                ratio_small_class=0, factor_small_class=1.0, 

                p_threshold = [],  # only used in "message passing"
                messages={},     # a placeholder for more complex message passing

                kind='user', 
                supervised=True, mask_all_test=True, balance_class=False, 
                pos_label=1, neg_label=0, marker=0, fold=-1): # mask C according to the values in R
    ### X: (R, T) or R (perhaps consider (R, ))
    #   if X <- R, then T will not be masked, meaning that all entries will be preserved
    #      X <- (R, T), then entries in T will be filtered according to the same rule as applied to R (i.e. rating matrix associated with the training split)

    N = len(L)
    n_users = C.shape[0] 
    labels = L # [] if not supervised else L

    ret = classPrior(L, labels=[neg_label, pos_label], ratio_ref=0.1)
    ratio_pos, ratio_neg = ret[pos_label], ret[neg_label]
    ratio_minority = min(ratio_pos, ratio_neg)

    C_prime = C.copy()  # save a copy of the 'full' confidence matrix (prior to masking)

    # used for the 'unsupervised mode'
    if ratio_small_class <= 0 or ratio_small_class > ratio_minority: 
        r0 = ratio_small_class
        # if it was not specified, then the default should a ratio equal or more conservative than the minority class ratio
        ratio_small_class = min(ratio_pos, ratio_neg)/factor_small_class
        print('(mask_over) Adjusting minority class ratio %f -> %f (minority class: %f)' % (r0, ratio_small_class, ratio_minority))
    
    if kind.startswith('i'):  # select entries item by item (this ensures that all data points are represented)
        # mode: supervised (passing labels), or unsupervised (not passing labels) but has an estimate of ratio_small_class
        
        # ratio_repr: 0.5 => each item/datum should be represnted by at most 50% of the BP predictive scores whenever possible ... 
        # ... sometimes less if enough BPs can be identified that produce consistent predictions with the true labels
        C = filter_along_item_axis(C, X, L=labels, 
                ratio_users=ratio_users, ratio_small_class=ratio_small_class, 
                    supervised=supervised, mask_all_test=mask_all_test, 
                    pos_label=pos_label, neg_label=neg_label, marker=marker, fold=fold)

    elif kind.startswith('u'):  

        # if L is given, then ratio_small_class is ignored
        C = filter_along_user_axis(C, X, L=labels, 
                p_threshold=p_threshold,     # used only in "message passing"
                ratio_small_class=ratio_small_class, 
                    supervised=supervised, mask_all_test=mask_all_test, balance_class=balance_class,
                    pos_label=pos_label, neg_label=neg_label, marker=marker, fold=fold)
        # >>> In unsupervised mode, we still use the labels (L) to estiamte the minority class proportion ... 
        # ... but labels are not used to compute the mask

    else: 
        raise NotImplementedError('Unrecognized mask policy: %s' % kind)

    C_prime = C_prime - C  # C_prime is the complement of C

    # if sparse_: 
    #     return sparse.csr_matrix(C), sparse.csr_matrix(C_prime)     
    return C, C_prime

def shift(C, offset=-1.0):
    if is_sparse(C): 
        C = C.todense() + offset
        return sparse.csr_matrix(C)
    return C + offset

def balance_and_scale(C, X, L, p_threshold, Po=None, U=[], alpha=1.0, beta=1.0, gamma=0.5,
        suppress_max_class=False, 
        is_cascade=True, discount_test=True,
        is_test_split=False, n_train=-1, **kargs):
    """
    Balance class weights and scale the confidence matrix (recall alpha * C in the optimization objective).

    Scaling is more straightford because the idea came from the CF paper with implicit feedback. 

    Why do we bother to rescale confidence scores in C according to the distribution of classes? 

    Recall that the confidence score in C effectively serve as weights to individual terms in the optimization objective. 
    Latent factors will strive to approximate (conditional) probabilites in X with relatively higher weights since 
    higher costs are incurred if these probabilties are not approximated well (recall 

    sum (u, i) C[u,i] * (r[u,i] - x'y) + ... )

    In biological datasets, we often deal with skewed class distributions, meaning that there are typically very few
    positive examples while majority of the sample are negative. To minimize the cost function, the latent factors 
    will give too much effort in approximating the probabilties associated with TNs (because they are the majority)
    but not emphysis on approximating TPs. 
    
    Parameters
    ----------
    alpha: the multiplying factor for confidence matrix 
    beta: the magnifiying factor for TP
    gamma: the discounting factor for suppressing confidence weights in the test split

    Po: polarity matrix, where {TP, TN} = 1 and {FP, FN} = 0 OR
        color matrix,    where TP=2, TN=1, FP=-2, FN=-1


    """
    import scipy.sparse as sparse
   
    C_is_sparse = False
    if sparse.issparse(C): 
        C = C.A # C.toarray()
        C_is_sparse = True  # convert back to sparse matrix when reweighting is done
    
    assert C.shape == X.shape
    assert len(L) == X.shape[1]
    if len(U) > 0: assert len(U) == C.shape[0]
    if n_train > 0 and X.shape[1] > n_train: is_cascade = True

    # dependency 
    #   classPrior
    #   polarity_matrix, correctness matrix
    conf_measure = kargs.get('conf_measure', 'brier')
    min_class, max_class = kargs.get('min_class', 1), kargs.get('max_class', 0)
    fold = index = kargs.get('index', 0)
    is_estimated_labels = kargs.get('is_test_split', False)
    test_cases = kargs.get('test_cases', list(range(5)))  # or np.random.choice(range(C.shape[0]), 5)
    sparsify = kargs.get('sparsify', True)
    verbose = kargs.get('verbose', 1)

    ret = classPrior(L, labels=[0, 1], ratio_ref=0.1, verbose=False)
    
    if ret['n_min_class'] == 0: 
        # label dtpye 
        lt = np.random.choice(L, 1)[0]
        print('(balance_and_scale) Warning: No minority class found in this batch => No-op! | n_max: {n_max}, n_min: {n_min} | dtype(L): {dtype} (expected int), value: {val}'.format(
            n_max=ret['n_max_class'], n_min=ret['n_min_class'], dtype=type(lt), val=lt ))
        # multiple = ret['n_max_class']/(ret['n_min_class']+1)   # n-/n+ 
        return 
        
    multiple = ret['n_max_class']/(ret['n_min_class'])   # n-/n+
    min_class, max_class = ret['min_class'], ret['max_class']

    n_users, n_items = C.shape
    # idx_max_class = np.where(L==max_class)[0]  # column-wise positional indices of majority class ... [0] because np.where returns a tuple
    # idx_min_class = np.where(L==min_class)[0]
    multiple_eff = []
    weights_min, weights_max = [], []
    if verbose: 
        print('(balance_and_scale) Balancing class weights by considering size disparity ...')
        verify_confidence_matrix(C, X=X, L=L, p_threshold=p_threshold, U=U, measure=conf_measure,
                message='(before) balanced + magnified (alpha={}, beta={}) | dtype: {}'.format(
                                     alpha, beta, 'test set' if is_estimated_labels else 'training set'), 
                    test_cases=[], plot=True if fold == 0 else False, 
                    test_weight_constraints=True)
        print('-' * 80)

    if is_test_split: 
        # some adjustments based on test split 
        
        # if unmask_on_test: 
        #     print('(balance_and_scale) Unmasking test data by assigning a minimum weight')
        #     unmask(C, L, U=U, test_cases=test_cases)
        pass

    ### A. re-weighting at global scale 

    Mc, Lh = polarity_matrix(X, L, p_threshold) # X, pth -> Lh | L -> Mc
    if Po is not None: # then we need (L, p_threshold to determine Po)
       Mc = Po.A if sparse.issparse(Po) else Po

    # Mc = Mc.astype(bool)
    w_min = np.min(C[C>0.0])
    # assert all(C[~Mc] == w_min), "FP and FN entries have non-mininum weights! w_min: {}".format(w_min)

    # [test] before and after reweighting 
    # weights_min_prior = C[Lh == min_class] # no need to apply Mc as in C[Mc & Lh == min_class]
    # weights_max_prior = C[Lh == max_class] # C[Mc & Lh == max_class]
    # print('... Before re-weighting | Positive(+) | 5 numbers: {}'.format(common.five_number(weights_min_prior)))
    # print('... Before re-weighting | Negative(-) | 5 numbers: {}'.format(common.five_number(weights_max_prior)))

    # the decision rules below work for both regular polarity matrix {-1, 0, 1} and color matrix {-2, -1, 0, 1, 2}
    cells_tp = (Mc > 0) & (Lh == min_class) # assuming min_class or minority class is positive
    cells_tn = (Mc > 0) & (Lh == max_class) # assuming max_class or majority class is negative
    cells_fp = (Mc < 0) & (Lh == min_class)
    cells_fn = (Mc < 0) & (Lh == max_class)

    msg = ''
    Wtp = C[ cells_tp ]
    Wtn = C[ cells_tn ]
    Wfp = C[ cells_fp ]
    Wfn = C[ cells_fn ]
    msg += '[info] Before re-weighting  TP(+) | 5 numbers: {}\n'.format(common.five_number(Wtp))
    msg += '                            TN(-) | 5 numbers: {}\n'.format(common.five_number(Wtn))
    msg += '                            FP(-) | 5 numbers: {}\n'.format(common.five_number(Wfp))
    msg += '                            FN(+) | 5 numbers: {}\n'.format(common.five_number(Wfn))
    if verbose: print(msg)
    
    msg = ''
    Ntp = np.sum( cells_tp )  # all the weights ~ positive class 
    Ntn = np.sum( cells_tn )
    Nfp = np.sum( cells_fp )
    Nfn = np.sum( cells_fn )
    N = Ntp+Ntn+Nfp+Nfn
    wtp, wtn, wfp, wfn = N/(Ntp+0.0), N/(Ntn+0.0), N/(Nfp+0.0), N/(Nfn+0.0)
    
    # Normalize weights so that they sum to 1
    w = sum([wtp, wtn, wfp, wfn])+0.0 # np.min([wtp, wtn, wfp, wfn])
    wtp, wtn, wfp, wfn = wtp/w, wtn/w, wfp/w, wfn/w

    msg += "... Reweight C inversely proprotional to samples sizes >\n"
    msg += f"... N (total): {N}"
    msg += f"... wtp: {wtp}, wtn: {wtn}, wfp: {wfp}, wfn: {wfn}"
    if verbose: print(msg)
     
    # re-adjust confidence weights inversely proprotional to individual sample sizes
    C[ cells_tp ] = C[ cells_tp ] * wtp
    C[ cells_tn ] = C[ cells_tn ] * wtn 
    C[ cells_fp ] = C[ cells_fp ] * wfp
    C[ cells_fn ] = C[ cells_fn ] * wfn

    # if Wtn > Wtp: 
    #     multiple_eff = Wtn/(Wtp+0.0)
    #     div('(balance_and_scale) W(TP): {} <? W(TN): {} | multiple<tn/tp>: {} | w_min: {}'.format(Wtp, Wtn, multiple_eff, w_min))
        
    #     # C[Mc & Lh == min_class] = C[Mc & Lh == min_class] * multiple_eff
    #     C[Lh == min_class] = C[Lh == min_class] * multiple_eff
    # else: 
    #     raise ValueError("W(tp) > W(tn)? This would not occur in highly-skewed data | W(tp):{}, W(tn): {}".format(Wtp, Wtn))
    #     # multiple_eff = Wtp/(Wtn+0.0)
    #     # div('(balance_and_scale) W(TP): {} > W(TN): {} ???| multiple<tn/tp>: {} | w_min: {}'.format(Wtp, Wtn, multiple_eff, w_min))
    #     # C[Mc & Lh == max_class] = C[Mc & Lh == max_class] * multiple_eff

    # up-regulating positive sample weights (and down-regulating negative sample weights)

    if beta > 1.0: 
        if verbose: print('(balance_and_scale) Amplifying weights associated with TPs at beta level={}'.format(beta))
        # C[Lh == min_class] = C[Lh == min_class] * beta

        # TP
        C[ cells_tp ] = C[ cells_tp ] * beta
        # ... want pref(TP) -> 1
        # FP 
        # C[ cells_fp ] = C[ cells_fp ] * beta
        # ... (?) want pref(FP) -> 0, they really should be negative examples
        # FN
        # C[ cells_fn ] = C[ cells_fn ] * beta
        # ... want pref(FN) -> 0, they really are positive examples, => want them to be 1?

    if suppress_max_class: 
        # max class (i.e. majority class) is the negative examples => suppress weights of TNs
        C[ cells_tn ] = C[ cells_tn ] * 0.01

    if is_cascade and discount_test: 
        # gamma = 0.5
        if verbose: print('(balance_and_scale) Discount test sample weights by {}'.format(gamma))
        Cr, Ct = C[:,:n_train], C[:,n_train:]
        Ct = Ct * gamma # discount the entire test split
        C = np.hstack([Cr, Ct])

    # weights_min = C[Lh == min_class] # C[Mc & Lh == min_class]
    # weights_max = C[Lh == max_class] # C[Mc & Lh == max_class]
    Wtp = C[ cells_tp ]
    Wtn = C[ cells_tn ]
    Wfp = C[ cells_fp ]
    Wfn = C[ cells_fn ]

    msg = ''
    msg += '[info] After re-weighting  TP(+) | 5 numbers: {}\n'.format(common.five_number(Wtp))
    msg += '                           TN(-) | 5 numbers: {}\n'.format(common.five_number(Wtn))
    msg += '                           FP(-) | 5 numbers: {}\n'.format(common.five_number(Wfp))
    msg += '                           FN(+) | 5 numbers: {}\n'.format(common.five_number(Wfn))      
    if verbose: print(msg)   

    # Scale confidence matrix collectively
    ###################################################

    C = alpha * C

    ###################################################
    if verbose: 
        verify_confidence_matrix(C, X=X, L=L, p_threshold=p_threshold, U=U, measure=conf_measure,
                message='(after) balanced + magnified (alpha={}, beta={}) | dtype: {}'.format(
                                     alpha, beta, 'test set' if is_estimated_labels else 'training set'), 
                    test_cases=test_cases, plot=True if fold == 0 else False, 
                    test_weight_constraints=True)
    
    ### B. re-weighting per classifier/user
    # for i in range(n_users): 
    #     user_name = U[i] if len(U) > 0 else i

    #     # indices of majority classe & predictions were correct
    #     idx_max_class_correct = np.where( (L==max_class) & (C[i] > 0) )[0]
    #     idx_min_class_correct = np.where( (L==min_class) & (C[i] > 0) )[0]

    #     weights_max_class = np.sum(C[i, idx_max_class_correct])
    #     weights_min_class = np.sum(C[i, idx_min_class_correct])

    #     if weights_min_class > 0 and weights_max_class > 0: 

    #         # each classifier/user has a different weight distribution
    #         multiple_eff_i = weights_max_class/weights_min_class
    #         print('... user: {} | w(pos): {}, w(neg): {}, multiple: {} | N={} (> max(w))?'.format(user_name, weights_min_class, weights_max_class, multiple_eff_i, n_items))

    #         # >>> there are cases where the mask function masks all the majority class exmaples, leading to zero weights (werid but true)
    #         if multiple_eff_i <= 1: 
    #             msg = "Warning: weights(class={max_class}) < weights(class={min_class}) ?? {w_min} > {w_max}".format(max_class=ret['max_class'], min_class=ret['min_class'],
    #                 w_min=weights_min_class, w_max=weights_max_class)
    #             print(msg)
    #             # assert multiple_eff_i > 1, "weights(class={max_class}) < weights(class={min_class}) ?? {w_min} > {w_max}".format(max_class=ret['max_class'], min_class=ret['min_class'],
    #             #     w_min=weights_min_class, w_max=weights_max_class)
    #             multiple_eff_i = multiple
        
    #         # [test]
    #         multiple_eff.append(multiple_eff_i)
    #         weights_min.append(weights_min_class)
    #         weights_max.append(weights_max_class)

    #         # update 
    #         prior_weights = C[i, idx_min_class_correct]
    #         ##########################
    #         C[i, idx_min_class_correct] = C[i, idx_min_class_correct] * multiple_eff_i  # magnify confidence scores of minority class
    #         ##########################
    #         post_weights = C[i, idx_min_class_correct]

    #         # [test]
    #         if i in test_cases: 
    #             tidx = np.random.choice(range(len(prior_weights)), 5)
    #             print('... sample weights (prior):\n... {}\n'.format(prior_weights[tidx]))
    #             print('... sample weights (post): \n... {}\n'.format(post_weights[tidx]))
    #     else: 
    #         if weights_min_class == 0:
    #             # ... not positive classes were selected => this classifier is probably not useful 
    #             # reduce all to min weights 
                

    #             C[i, idx_max_class_correct] = min_weight

    #             print('... suppress negative example weights for user {} to min weight {}'.format(user_name, min_weight))
    
    # div("(balance_and_scale) Effective min-to-max weight ratio distribution: {ratios} | all > 1?".format(ratios=multiple_eff))
    # print('(balance_and_scale) weights of minority class multiplied by {m} on average >? standard multiple (non-weighted): {m0}'.format(m=np.mean(multiple_eff), m0=multiple))
    # print('... weights<min_class>: {}'.format(np.random.choice(weights_min, 10)))
    # print('... weights<max_class>: {}'.format(np.random.choice(weights_max, 10)))
    
    if verbose: print('[info] class stats: {o}'.format(o=ret))

    # ... C is dense at this point
    if C_is_sparse or sparsify: # then Cn must also be sparse to be consistent
        C = sparse.csr_matrix(C)

    return C   
# [alias]
balance_class_weights = balance_and_scale


def verify_confidence_matrix(C, X, L, p_threshold, Po=None, U=[], measure='rank', message='', test_cases=[], plot=False, test_weight_constraints=True, seed=53): 
    from analyzer import plot_confidence_matrix
    from utils_sys import highlight

    # Set random seed for reporducibility
    np.random.seed(seed)

    ranges = []
    n_users = C.shape[0]
    n_test_cases = 3 # Examine only this many base classifiers
    assert n_users == X.shape[0]
    if len(test_cases) == 0:
        test_cases = np.random.choice(range(C.shape[0]), 5) if n_users > n_test_cases else list(range(n_test_cases))

    highlight("\n(verify_confidence_matrix) Are the confidence scores taking on values as expected?\n")
    if message: highlight(message, symbol='-', border=1)

    # Mc, Lh = polarity_matrix(X, L, p_threshold) # X, pth -> Lh | L -> Mc
    Pc, Lh = color_matrix(X, L, p_threshold) 
    Mc, Lh2 = polarity_matrix(X, L, p_threshold)
    # ... assuming that both C and Po are dense array
    assert np.sum(Lh != Lh2) == 0

    # Mc = Mc.astype(bool)  # if not booleanized, then C[Mc] turns into a 3D array, which then could overwhelm the memory
    w_min_abs = 0.0

    msg = '' 
    if test_weight_constraints: 

        # Note that these statistics depend on the filtering mechanism and whether L contains estimated labels
        n_correct = np.sum(Mc > 0)
        n_incorrect = np.sum(Mc < 0)
        msg += '[verify] n(TP+TN): {}, n(FP+FN): {}, ratio: {}\n'.format(n_correct, n_incorrect, n_correct/(n_correct+n_incorrect+0.0) )

    # Weight distribution
    adict = {0: 'Negative (-)', 1: 'Positive (+)', }

    Wpos = C[(Pc == 2) & (Pc == -2)] # P = TP + FP
    # ... Wpos is a 1D array referencing the confidence scores (weights) in C for which the condition holds

    Wneg = C[(Pc == 1) & (Pc == -1)] # N = TN + FN 
    Wtp = C[Pc == 2]
    Wtn = C[Pc == 1]

    # Also verify the weight using polarity matrix
    Wtp2 = C[(L == 1) & (Mc == 1)]
    Wtn2 = C[(L == 0) & (Mc == 1)]
    assert np.isclose(np.sum(Wtp), np.sum(Wtp2)), f"np.sum(Wtp) via color matrix: {np.sum(Wtp)} <> {np.sum(Wtp2)} via polarity matrix!"
    assert np.isclose(np.sum(Wtn), np.sum(Wtn2)), f"np.sum(Wtn) via color matrix: {np.sum(Wtn)} <> {np.sum(Wtn2)} via polarity matrix!"
    
    # w_min_pos, w_max_pos, w_mean_pos, w_med_pos = np.min(Wpos), np.max(Wpos), np.mean(Wpos), np.median(Wpos)
    # w_min_neg, w_max_neg, w_mean_neg, w_med_neg = np.min(Wneg), np.max(Wneg), np.mean(Wneg), np.median(Wneg)

    # Here, we only care about the weight distribution for TP and TN 
    wstats = {1: (np.min(Wtp), np.max(Wtp), np.mean(Wtp), np.median(Wtp)), 
              0: (np.min(Wtn), np.max(Wtn), np.mean(Wtn), np.median(Wtn)) }

    msg += '\n-- Class-wise weight distributions --\n'
    for cl in [1, 0]: 
        msg += '... Class {}: min = {}, max = {}, mean = {}, median = {}\n'.format(adict[cl], *wstats[cl])
    
    n_tp = np.sum(Pc == 2)
    n_tn = np.sum(Pc == 1)
    n_fp = np.sum(Pc == -2)
    n_fn = np.sum(Pc == -1)
    n_uncertain = np.sum(Pc == 0)
    N = n_tp + n_tn + n_fp + n_fn + n_uncertain
    assert N == Pc.shape[0] * Pc.shape[1]

    Wtp_total = np.sum(Wtp)
    Wtn_total = np.sum(Wtn)

    msg += '\n--- Confidence score (weight) sum total per class ---\n'
    msg += f"... N(TP): {n_tp}, N(TN): {n_tn}, N(TP)/N(TN): {n_tp/(n_tn+0.0)}\n"
    msg += f"... W(TP): {Wtp_total}, W(TN): {Wtn_total}, W(TP)/W(TN): {Wtp_total/(Wtn_total+0.0)}\n"
    msg += f"... Balanced? W(TP)/W(TN)={Wtp_total/(Wtn_total+0.0)} ~ 1.0\n"

    print(msg); print()
    ################################################
    highlight("(verify_confidence_matrix) Have we masked the neutral (uncertain) and negative entries (FPs, FNs)?")
    msg = ''
    n_zero_weights = np.sum(C==0)
    if n_zero_weights > 0: 
        msg += f"[verify] Neutral and negatives (FP, FN) are masked: n(zeros): {n_zero_weights} >=? n(fp+fn): {n_fp+n_fn}\n"

    print(msg); print()
    ################################################

    msg = '\n-- Base predictor weight distributions --\n' 
    for i in range(C.shape[0]): # foreach user/classifier

        if i in test_cases: # Examine only these classifiers
            user_name = U[i] if len(U) > 0 else i

            # for cl in [1, 0, ]: # foreach class
            idx_pos = np.where(L == 1)[0]
            idx_neg = np.where(L == 0)[0]
            idx_tp = np.where( (L == 1) & (Mc[i] > 0) )[0] 
            idx_tn = np.where( (L == 0) & (Mc[i] > 0) )[0]

            scores_tp = C[i][idx_tp]
            scores_tn = C[i][idx_tn]

            if len(idx_tp) > 0:
                m, M = np.min(scores_tp), np.max(scores_tp)
                med = np.median(scores_tp)
                wx = np.random.choice(scores_tp, 10)

                # Among all TPs, what do their confidence scores look like? 
                msg += f"[verify] [TP] BP=[{user_name}]: n(P): {len(idx_pos)}, n(TP): {len(idx_tp)}, recall: {len(idx_tp)/len(idx_pos)}\n"
                msg += f"\n{wx}\n... Min: {m}, max: {M}, median: {med}\n... condition: ({message})\n\n"

                
            if len(idx_tn) > 0: 
                m, M = np.min(scores_tn), np.max(scores_tn)
                med = np.median(scores_tn)
                wx = np.random.choice(scores_tn, 10)

                # Among all TPs, what do their confidence scores look like? 
                msg += f"[verify] [TN] BP=[{user_name}]: n(N): {len(idx_neg)}, n(TN): {len(idx_tn)}, TNR: {len(idx_tn)/len(idx_neg)}\n"
                msg += f"\n{wx}\n... Min: {m}, max: {M}, median: {med}\n... condition: ({message})\n\n"
            print(msg)
    if measure == 'rank': 
        pass

    # !!! don't do it here
    # if plot: 
    #     # closure: alpha, beta
    #     plot_confidence_matrix(C, X, L, p_threshold, U=U, n_max=100, path=kargs.get('path', os.getcwd()), 
    #         measure=measure, target_label=None, alpha=alpha, beta=beta, index=0)

    return

def toConfidenceMatrix2(X, L, **kargs): 
    """
    Similar to toConfidenceMatrix() but returns both confidence matrix (Cui) and its complement (Cui_bar)


    Memo
    ----
    1. balancing skewed classes
       ref: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
    """
    raise NotImplementedError

def toConfidenceMatrix(X, L, **kargs):
    """
    Compute confidence matrix. First estimate an appropriate probability filter (M) for X, where 
    reliable probabilities are marked by 1 and unreliable probabilities are marked by 0

    Then, we ompute the confidence scores for each prediction in X given the confidence measure (conf_measure, e.g. 'brier') to form
    an initial confidence matrix C0

    The final confidencee matrix is an element-wise product between M and C0, which allows us to encourage the latent factor 
    to give higher weight on reliable probabilties while giving 0 or negative weights on unreliable probablities.

    Probabilty filter or mask is determined by at least 4 factors (see Memo). 

    **kargs
    -------
    policy: used to define the mask function; specfies the filtering dimension: 'user', 'item' 
            also see 'ratio_small_class', 'ratio_users', conf_measure

    fill: a marker for unreliable rating values, or for missing values; default 0

    pos_label

    p_threshold: a vector of proba thresholds of the same size as X.shape[0], representing the number of users/classifiers; 
                 used to to estimate confidence scores in mode = 'ratio'
    policy_threshold 
    ratio_small_class

    conf_measure: 'brier', 'ratio', 'corr', 
    scoring: scoring function for computing confidence scores

    ratio_users: used as a parameter for the mask function in item-centered mode (i.e. filtering along the axis of the items/data) 

    U
    L_true: ground truth labels (for testing only)

    sparse: 
    alpha

    Similar to toConfidenceMatrix() but returns both Cui and its complement

    Memo
    ----
    1. mask function is determined by: 
        a. filtering dimension: along the user/classifier axis (policy='user'), or along the item/data axis (policy='item')

           - [update] perhaps we should always consider 'item' dimension

        b. policy_threshold: how probability thresholds are determined in order to estimate the labeling, which then allows for 
                            the determination of masked entries representing unreliable conditional probability estimates (or in general any ratings in the entries of X)
        
        [update] c and d (unsupervised) are not favorable

        c. depending on the true labels (L) being given or not
           we'll need
             ratio_small_class, when L isn't given in order to estimate proba thresholds
        d. ratio_users: for each item/data point, the ratio of user ratings (classifier probabilities) to consider as sufficiently reliable rating esitmates
                        only used in item-centered mode (policy='item') i.e. filtering along the axis of data

    2. balancing skewed classes
       ref: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/


    """
    def to_label(R, labels, p_th=0.5): 
        P = np.zero((R.shape[0], R.shape[1]))
        P[np.where(R >= p_th)] = 1
        return P
    def mask_fp_fn(R, labels, p_th=0.5, pos_label=1, neg_label=0, labelize=True):  # rule: keep only TP, TN
        # convert to sparse repr while masking FP and FN
        
        # label predictions
        P = to_label(R, labels, p_th=p_th)
         
        cond_tp = (R >= p_th) & (labels == pos_label)
        cond_tn = (R < p_th) & (labels == neg_label)
        rows, cols = np.where(cond_tp | cond_tn)
        
        good_pred = P[rows, cols] if labelize else R[rows, cols]
        # good_probs = R[rows, cols]  # R[np.where(cond_tp | cond_tn)]
        
        nU, nI = R.shape[0], R.shape[1]
        S = sparse.csr_matrix((good_pred, (rows, cols)), shape=(nU, nI))
        return S 
    def mask(R, labels, p_th=0.5, pos_label=1, neg_label=0, rule='false_prediction'):

        # mask rule
        cond_tp = (R >= p_th) & (labels == pos_label)
        cond_tn = (R < p_th) & (labels == neg_label)

        # entries
        rows, cols = np.where(cond_tp | cond_tn)
        return (rows, cols)

    def show_thresholds(p_th, users=[]): # closure: p_threshold
        policy_threshold = kargs.get('policy_threshold', 'prior')
        sort_by = {0: 'classifer', 1: 'probability'}
        if len(users) == 0: 
            print('... probability thresholds:\n.....{0}\n'.format(p_threshold))
        else: 
            sid = 0 # by classifier or by probablity values
            div("(toConfidenceMatrix) List of proba thresholds given policy: {p} | sorted according to -- {v} -- ".format(p=policy_threshold, v=sort_by[sid]))
            pth_sorted = sorted(zip(users, p_th), key=lambda x: x[sid])   
            for i, (u, pth) in enumerate(pth_sorted): 
                print('... [%d] %s: p_th = %f' % (i+1, u, pth))
    def scan_and_update(C, L, min_class=None, multiple=None):
        if None in (min_class, multiple): # not known a priori 
            ret = classPrior(L, labels=[0, 1], ratio_ref=0.1, verbose=False)
            multiple = ret['n_max_class']/ret['n_min_class']
            min_class = ret['min_class']
        idx = np.where(L==min_class)[0]  # column-wise positional indices of miniroty class
        C[:, idx] = C[:, idx] * multiple  # only effect on subset of 'idx' that are unmasked (not zeros)
        print('(scan_and_update) weights of minority class multiplied by {m} | len(idx): {n} | class stats: {o}'.format(m=multiple, 
            n=len(idx), o=ret))
        # return C    # not necessary
    def unmask(C, L, U=[], test_cases=[], min_class=1, max_class=0):
        # for the test split, since labels are only estimated, maybe we do not want to be overly confident in masking the "wrong" predictions 
        n_users, n_items = C.shape

        global_min_weight = np.min(C[C>0])
        print('(unmask) global min confidence weight: {}'.format(global_min_weight))
        for i in range(n_users): 
            user_name = U[i] if len(U) > 0 else i
   
            idx_max_class_correct = np.where( (L==max_class) & (C[i] > 0) )[0]
            if len(idx_max_class_correct) == 0: 
                min_weight = global_min_weight 
            else:
                # row min 
                min_weight = np.min(C[i][idx_max_class_correct])

            idx_masked = np.where( C[i] == 0 )[0]
            C[i, idx_masked] = min_weight

            if i in test_cases: 
                print('(unmask) classifier=[{}] | min weight: {}'.format(user_name, min_weight))

        assert np.sum(C[C==0]) == 0
        return

    def identify_effective_users(Cr, L_train, fill=0): 
        n_users = Cr.shape[0]
        effective_states = np.zeros(n_users)
        for i in range(n_users):
            # non_masked_positive = (Cr[i] > fill) & (L_train == 1)
            n_non_masked = np.sum(Cr[i] > fill)
            if n_non_masked > 0: 
                effective_states[i] = 1
        return effective_states
    def regulate_weights(C, beta=10, U=[], suppress_max_class=False, 
            min_class=1, max_class=0, bag_count=10, test_cases=[]):
        if beta <= 1: 
            # no-op 
            return  # C would've been modified in place

        if len(test_cases) == 0: test_cases = np.random.choice(range(C.shape[0]), 5)

        adict = {0: 'Negative', 1: 'Positive', }
        sample_weights = {}
        sample_weights_max_class = {}
        tns, tps = {}, {}

        w_min = np.min(C[C > 0])   # note: np.min(C > 0) -> False

        for i in range(C.shape[0]): 
            uname = U[i] if len(U) > 0 else i
            # idx = np.where(L == min_class)[0]
            idx_min_class_correct = np.where((L == min_class) & (C[i] > 0))[0]
            idx_max_class_correct = np.where((L == max_class) & (C[i] > 0))[0]

            n_tmin, n_tmax = len(idx_min_class_correct), len(idx_max_class_correct)

            if n_tmin > 0: 
                C[i, idx_min_class_correct] = C[i, idx_min_class_correct] * beta

                # [test]
                if i in test_cases: 
                    sample_weights[uname] = np.random.choice(C[i, idx_min_class_correct], 10)
                    tps[uname] = n_tmin
                    sample_weights_max_class[uname] = np.random.choice(C[i, idx_max_class_correct], 10)
                    tns[uname] = n_tmax

            if suppress_max_class: 
                
                C[i, idx_max_class_correct] = w_min    # this may conflict with unmask in balance_and_scale()
                sample_weights_max_class[uname] = w_min

        # [test]
        # idx = []
        # n_max = 5
        # bag_start_indices = range(0, len(U), bag_count)
        # # names = [c.split(sep)[0] for c in U[bag_start_indices]]
        # for i in bag_start_indices:
        #     idx.extend( list(range(i, i+n_max)) )
        # idx = np.array(idx)
        div(message='(regulate_weights) minority class weight distribution after magnified by beta={}...'.format(beta))
        for u, wx in sample_weights.items(): 
            print("... [{}] {} (size: TP)".format(u, tps[u]))
            print("...      {} ... (sample weights: TP)".format(wx))
            print("...      {} (size: TN)".format(tns[u]))
            print("...      {} ... (sample weights: TN)".format(sample_weights_max_class[u]))

        return  # C is modified in place

    # from sklearn.metrics import brier_score_loss
    import scipy.sparse as sparse
    from evaluate import findOptimalCutoff   #  p_th <- f(y_true, y_score, metric='fmax', pos_label=1)
    import collections

    # Key parameters
    conf_measure = kargs.get('conf_measure', 'brier')
    policy_threshold = kargs.get('policy_threshold', 'fmax')
    p_threshold = kargs.get('p_threshold', [])  # depends on policy_threshold: 'fmax', float, 'unsupervised' 
    # Note: Conditions on `p_threshold` 
    #   1) p_threshold given externally 
    #   2) p_threshold is to be inferred from training data message (X_train, L_train); used for computing confidnece scores for the test split 
    #   3) p_threshold is to be esimated via 'prior'; L must be given
    #   
    #   4) p_threshold could be estimated in a unsupervised way even without L, but not recommended
    
    # Misc parameters
    null_marker = kargs.get('fill', 0) # [todo] marker for missing data
    pos_label = kargs.get('pos_label', 1)

    # Optional parameters
    fold = kargs.get('fold', -1)  # only used for debugging
    n_train = kargs.get('n_train', -1) # Only used to separate L into L_train and "lh" (est labels), from which to estimate p_threshold
    U = users = kargs.get('U', []) # names of users/classifiers
    is_cascade = kargs.get('is_cascade', False) # [note] cascade mode seems more favorable (i.e. X=[R|T])
    verbose = kargs.get('verbose', 1)
    alpha = kargs.get('alpha', 10) # see `balance_and_scale()`

    # Obsolete parameters (but possibly useful) 
    # topk_users, topk_items = 0, 0  # default, use all by setting to 0
    # tSupervised = True # kargs.get('supervised', True) # Note: Always make use of the training set's labels if possible
    # policy_filtering = kargs.get('policy', 'item') # Probability filtering policy for the training set

    # Probability filtering policy for the test set (only relevant in casade mode i.e. X = [R | T])
    # policy_filtering_test = kargs.get('policy_test', 'polarity') if is_cascade else None # None as 'undefined'

    n_users, n_items = X.shape[0], X.shape[1]

    # [design] Is it better to use cascade mode (i.e. X = [R | T]), so that no need to worry about message passing for T? 
    #####################################################################################
    # "Message passing": Pass training set information useful for determining T's probaility filter (assuming that X contains only the test set data i.e. T)
    tHasMessageFromTrainingSplit = False
    R = T = L_train = L_test_est = None
    Eu = [] # effective classifiers
    M = kargs.get('message', None)
    if M is not None: 
        R, L_train, *rest = M
        tHasMessageFromTrainingSplit = True

        if verbose: 
            print('(toConfidenceMatrix) R is provided separately as a message => Assuming X = T, n(train): {}, n(test): {}'.format(
                    R.shape[1], X.shape[1]))

        # use confidence matrix from training split (Cr) to find classifiers with support (correct predictions)
        if len(rest) > 0: # we have additional information from the training set
            Cr = rest[0]

            # Eu? A classifier is "effective" if it's accumulated confidence score across all data points is greater than a baseline value (usu. 0)
            Eu = identify_effective_users(Cr, L_train=L_train, fill=null_marker)
            # ... Eu or effective user is a binary vector, where 1: effective 0: otherwise

            if verbose: print('(toConfidenceMatrix) How many classifiers were effective in the training set? {n} | total: {N}'.format(n=np.sum(Eu), N=Cr.shape[0]))
    
    hasSplit = False
    if is_cascade: # then X must be [R|T]
        if isinstance(X, (tuple, list)): 
            assert isinstance(L, (tuple, list))
            R, T = X
            L_train, L_test_est = L
            hasSplit = True
        else: 
            if len(p_threshold) == 0: 
                assert n_train > 0, f"(toConfidenceMatrix) No way of knowing how to split X into R and T without knowing size of R (given n_train {n_train})"
                assert len(L) == X.shape[1]

                R, T = X[:,:n_train], X[:,n_train:]
                L_train, L_test_est = L[:n_train], L[n_train: ] # note that `L_test_est` is only a guess since we do not know test set's true labels
                hasSplit = True

    # [design] Always pre-compute probability thresholds so that this block can be skipped
    #####################################################################################
    if len(p_threshold) == 0: # Need to estimate probabilty thresholdds
        if hasSplit: 
            assert L_train is not None and R.shape[1] == len(L_train)

            print('(toConfidenceMatrix) message passing: use training set statistics to estimate proba thresholds | policy={p}'.format(p=kargs.get('policy_threshold', 'prior')))
            policy_threshold = kargs.get('policy_threshold', 'prior')
            p_threshold = estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=policy_threshold) # policy: any policy that utilizes L
        else: 
            assert len(L) > 0, "Both p_threshold and L are not given, and neither was training data message given!"
            policy_threshold = 'prior' # kargs.get('policy_threshold', 'prior') if len(L) > 0 else 'ratio' 
            p_threshold = estimateProbThresholds(X, L=L, pos_label=pos_label, policy=policy_threshold, ratio_small_class=kargs.get('ratio_small_class', 0.01)) # findOptimalCutoff(L_train, R, metric='fmax', beta=1.0, pos_label=1)
    if verbose: show_thresholds(p_threshold, users=kargs.get('U', []))
    
    # Estimtae labels if not given
    # policy: 
    #     a. majority vote 
    #     b. best lh match (need statistics from R); mask transfer
    tEstimatedLabels = False
    # [design] Always pre-compute estimated labels before this call to make the logic clean
    ##################################################################################### 
    if len(L) == 0: 
        assert not is_cascade, "In cascade mode X=[R | T], we know at least the training data's labels, which must be provided through L"

        # then X must be just a test set (i.e. X = T)
        if verbose: 
            print("(toConfidenceMatrix) Estimating labels (Lh) using: 1. input X and 2. proba thresholds | policy={p} ...".format(p=policy_threshold))

        # [note] joint model doesn't work well
        # if tHasMessageFromTrainingSplit: 
        #     # jointModel = estimateProbaByLabelMatrix(R, L_train, p_th=p_threshold, pos_label=1) # policy_threshold='prior' 
        #     L = lh = estimateLabels(X, L=[], p_th=p_threshold, pos_label=pos_label, M=(R, L_train) ) # joint_model=jointModel
        
        # Why passing Eu? Because we want majority vote to account for only effective classifiers
        L = lh = estimateLabels(X, L=[], Eu=Eu, p_th=p_threshold, pos_label=pos_label, ratio_small_class=kargs.get('ratio_small_class', 0.01)) # policy: 'ratio' by default when no labels are provided
        tEstimatedLabels = True
    else: 
        # use the proba threshold to compute the mask and keep track of their statistics to be used in estimating labels in T
        pass
    
    # [design] L_test
    ##################################################################################### 
    # test accuracy (only useful when L is estimated)
    L_test = L_ext = kargs.get('L_test', [])
    if tEstimatedLabels and len(L_test) > 0: 
        accuracy = np.sum(lh == L_test) / (len(L_test)+0.0)
        div('(toConfidenceMatrix) Accuracy of estimated labels: {} | n(L_ext): {}'.format(accuracy, len(L_test)), symbol='#', border=2)
    if verbose: 
        print(f'(toConfidenceMatrix) Computing conficence scores using conf_measure: {conf_measure}')
        print('...                   p_threshold? {tval} | message passing? {tm} | policy: {p}'.format(
                 tval=len(p_threshold) > 0, p=policy_threshold, tm=tHasMessageFromTrainingSplit))

    # Compute considence scores that reflect quality of the predictions
    # - confidence scores are later to be used in the optimization for deriving latent factors
    ################################################################# 
    C0 = confidence2D(X, L, mode=conf_measure, 
                scoring=kargs.get('scoring', brier_score_loss), 
                outer_product=False, 

                    # following params are used only when mode = 'ratio'
                    p_threshold=p_threshold,  
                    policy_threshold=kargs.get('policy_threshold', ''), 
                    ratio_small_class=kargs.get('ratio_small_class', 0.01), verbose=verbose)  # don't return outer(wu, Wi) 
    
    # Cui/C0: Raw confidence scores 
    Cui = np.zeros(C0.shape)+C0

    # Old parameters for polarity modeling (not considered at the moment)
    #################################################################
    # Mc = None
    # tConservative = True
    # isPolarityMatrix = True
    #################################################################

    Pc, Lh = color_matrix(X, L, p_threshold) # TP=2, TN=1, FP=-2, FN=-1, used for polarity model
    # Pc, Lh = polarity_matrix(X, L, p_threshold) # {TP, TN}: 1, {FP, FN}: -1

    # [test]
    # verify_confidence_matrix(Cui, X=X, L=L, p_threshold=p_threshold, Po=Pc, U=U, measure=conf_measure, message='(before) raw weights', test_cases=[])  # test_cases <- [] to use default

    if verbose:
        # [test]
        n_incorrect = np.sum(Pc < 0)
        n_uncertain = np.sum(Pc == 0)
        n_correct = np.sum(Pc > 0)
        n_colors = len(np.unique(Pc))
        assert n_colors >= 4, "Expecting `colored` particles but got {}".format(colors)
        print('... Colored polarity matrix (Pc) | n_negative: {}, n_neutral: {} n_positive: {} | data: {}'.format(n_incorrect, 
                n_uncertain, n_correct, 'test set' if tEstimatedLabels else 'training set'))


    # Convert the matrices to sparse format if sparse output is desired 
    # - Only useful when proportionally large number of 0s is expected 
    ########################################################################
    n_zeros = n_nonzeros = 0
    if kargs.get('sparse', True): 
        Cui = sparse.csr_matrix(Cui)
        Pc = sparse.csr_matrix(Pc)

        # [test]
        n_nonzeros = sparse.csr_matrix.count_nonzero(Pc)
        n_zeros = Pc.shape[0] * Pc.shape[1] - n_nonzeros
        # if not tEstimatedLabels: 
        #     # for test split, consider unmask
        #     assert n_zeros > 0

        if verbose: print('[verify] Cui, Mc converted to sparse matrix.')
    else: 
        n_zeros = np.sum(Pc == 0)
        n_nonzeros = np.sum(Pc != 0)

    if verbose: 
        print('(toConfidenceMatrix) Cui: hape(Cui)={}, n_zeros (uncertain)={} (>? 0) vs n_nonzeros={} (masked ratio={})'.format( 
                 Cui.shape, n_zeros, n_nonzeros, n_zeros/(Cui.shape[0] * Cui.shape[1] + 0.0)))
        
    # Confidence score scaling is now factored into `balance_and_scale`
    ########################################################################
    # Cui = alpha * Cui
    ########################################################################

    # Cui/C0: Confidence matrix with raw confidence scores 
    # Pc: Colored matrix (colored polarity matrix, where colors represent types of Cui[i][j] (e.g. TP, TN, FP, FN, neutral/uncertain)
    # p_threshold: probability thresholds (associated with the base classifiers)
    return Cui, Pc, p_threshold   # where Pc represents colored polarity

def to_mean_vector(X, L, **kargs):
    """
    Operationally similar to toConfidenceMatrix() but the masking is applied to the input X instead of the 
    confidence matrix C. 

    """
    def show_thresholds(p_th, users=[]): # closure: p_threshold
        policy_threshold = kargs.get('policy_threshold', 'prior')
        if len(users) == 0: 
            print('... probability thresholds:\n.....{0}\n'.format(p_threshold))
        else: 
            div('... List of proba thresholds given policy: {p}'.format(p=policy_threshold))
            pth_sorted = sorted(zip(users, p_th), key=lambda x: x[1])
            for i, (u, pth) in enumerate(pth_sorted): 
                print('... [%d] %s: p_th = %f' % (i+1, u, pth))

    # from sklearn.metrics import brier_score_loss
    import scipy.sparse as sparse
    from evaluate import findOptimalCutoff   #  p_th <- f(y_true, y_score, metric='fmax', pos_label=1)
    import collections

    policy_filtering = kargs.get('policy', 'item')   # 'user', 'item', None (no filtering)
    null_marker = kargs.get('fill', 0) # [todo] marker for missing data
    topk_users, topk_items = 0, 0  # default, use all by setting to 0
    pos_label = kargs.get('pos_label', 1)
    n_users, n_items = X.shape[0], X.shape[1]
    fold = kargs.get('fold', -1)  # only used for debugging

    if policy_filtering.lower() in ('none', 'noop'): 
        return np.mean(X, axis=0)

    #####################################################################################
    # ... estimate probability thresholds used to estimate labels
    tSupervised = kargs.get('supervised', True if len(L) > 0 else False)

    ### balance class weights in the confidence score 
    tBalanceClassWeights = kargs.get('balance_and_scale', True)

    # message passing 
    tHasMessageFromTrainingSplit = False
    R = L_train = None # X_train
    M = kargs.get('message', None)
    if M is not None: 
        R, L_train, *rest = M
        tHasMessageFromTrainingSplit = True

    # [test]
    # two cases: if L <- [], then policy <- 'ratio' (need to provide ratio_small_class)
    #            if L is given, then policy <- 'prior' by default (or as determined by policy_threshold)
    p_threshold = []  # depends on policy_threshold: 'fmax', float, 'unsupervised' 
    if tHasMessageFromTrainingSplit: 
        # policy_threshold: prior, fmax, ...
        assert R is not None and L_train is not None

        print('(toMetaMatrix) message passing: use training set statistics to estimate proba thresholds | policy={p}'.format(p=kargs.get('policy_threshold', 'prior')))
        policy_threshold = kargs.get('policy_threshold', 'prior')
        p_threshold = estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=policy_threshold) # policy: any policy that utilizes L
    else: 
        policy_threshold = kargs.get('policy_threshold', 'prior') if len(L) > 0 else 'ratio' 
        p_threshold = estimateProbThresholds(X, L=L, pos_label=pos_label, policy=policy_threshold, ratio_small_class=kargs.get('ratio_small_class', 0.01)) # findOptimalCutoff(L_train, R, metric='fmax', beta=1.0, pos_label=1)
    show_thresholds(p_threshold, users=kargs.get('U', []))
    ##################################################################################### 
    # ... estimtae labels if not given
    if len(L) == 0: 
        # useful for test data
        print("(toMetaMatrix) Estimating labels (Lh) using (1) input X and (2) proba thresholds given earlier | policy={p} ...".format(p=policy_threshold))
        L = lh = estimateLabels(X, L=[], p_th=p_threshold, pos_label=pos_label, ratio_small_class=kargs.get('ratio_small_class', 0.01)) # policy: 'ratio' by default when no labels are provided
        
        # test accuracy
        L_test = L_ext = kargs.get('L_test', [])
        if len(L_test) > 0: 
            accuracy = np.sum(lh == L_test) / (len(L_test)+0.0)
            print('... (verify) accuracy of est. labels: {0}'.format(accuracy))
            # ... log: 
            #     domain: diabetes | accuracy of est. labels: 0.7402597402597403
    ##################################################################################### 
    # ... estimated labels ready L <- Lh

    # condition: masking FP, FN here? or compute confidence matrix first?
    if fold == 0: print('(toMetaMatrix) compute conficence scores | conf_measure: {0}, outer_product? {1}'.format(kargs.get('conf_measure', 'brier'), False))
    
    # Cui = sparse.csr_matrix.multiply(S, W) # hadamard product
    ### treat FP, FN as missing values
    print('... (verify) Given precomputed p_threshold? {tval} | message passing? {tm} | policy: {p}'.format(tval=True if len(p_threshold) > 0 else False, p=policy_threshold, tm=tHasMessageFromTrainingSplit))

    Xh = X
    if policy_filtering in ('none', ):
        # no-op 
        pass
    else:  
        Xh = np.copy(X)
        Xh = mask_over(Xh, X=X, L=L, ratio_users=kargs.get('ratio_users', 0.5), 

                    # message passing
                    p_threshold = p_threshold, # if tHasMessageFromTrainingSplit else [], 

                    ratio_small_class=kargs.get('ratio_small_class', 0), factor_small_class=kargs.get('factor_small_class', 1.0),  # used for unsupervised mode
                        kind=policy_filtering,   # <<< (collaborative) filtering policy
                        marker=null_marker, supervised=tSupervised, fold=kargs.get('fold', -1))
    
    # # side effect on W
    # if kargs.get('masked', True): 
    #     # mask FP, FN (and missing values if appilcable)

    #     # supervised vs unsupervised will lead to different augmenented label set La
    #     # [note] offset is used to indicate where test split starts
    #     Cui = mask_over(Cui, Ra, La, p_th=p_threshold, offset=offset) # other params: mask_all_test

    n_zeros = n_nonzeros = -1
    if kargs.get('sparse', False): 
        Xh = sparse.csr_matrix(Xh)

        # [test]
        n_nonzeros = sparse.csr_matrix.count_nonzero(Xh)
        n_zeros = Xh.shape[0] * Xh.shape[1] - n_nonzeros
        # assert n_zeros > 0

        print('... (verify) Xh converted to sparse matrix.')
    else: 
        n_zeros = np.sum(Xh == null_marker)
        n_nonzeros = Xh.shape[0] * Xh.shape[1] - n_zeros
        
    print('(to_mean_vector) New rating matrix under policy: {policy} |  dim(X): {0}, n_zeros: {1} vs nonzeros: {2} (masked ratio={3})'.format(Xh.shape, 
            n_zeros, n_nonzeros, n_zeros/(Xh.shape[0] * Xh.shape[1] + 0.0), policy=policy_filtering ))

    # take the mean column-wise 
    mX = np.zeros(Xh.shape[1])
    for j in range(Xh.shape[1]): 
        non_masked = np.where(Xh[:, j] != null_marker)[0]
        if len(non_masked) > 0: 
            mX[j] = np.mean(Xh[:, j][non_masked])
        else:
            mX[j] = np.mean(Xh[:, j])

    return mX
    
def predict_slow_simple(ratings, similarity, kind='user'):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):  # [note] xrange does not exist in Python3
            for j in range(ratings.shape[1]):

                # similarity[i, :]: similarity wrt user i
                # ratings[:,j]: ratings of item j from all users
                pred[i, j] = similarity[i, :].dot(ratings[:, j])\
                             /np.sum(np.abs(similarity[i, :]))
        return pred
    elif kind == 'item':
        for i in range(ratings.shape[0]):  # foreach user
            for j in range(ratings.shape[1]):  # foreach item

                # similarity[j, :]: similarlity wrt to item j (similarity between item j and all items)
                # find the ratings of all items that the same user has rated, ... 
                # ... and compute the weighted average of these ratings based on item similarity
                pred[i, j] = similarity[j, :].dot(ratings[i, :].T)\
                             /np.sum(np.abs(similarity[j, :]))

        return pred

def predict_fast_simple(ratings, similarity, kind='user'):
    """
    Vectorized form of predict_slow_simple(). 
    """
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

# [todo]
def predict_biased(ratings, similarity, kind='user'): 
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def predict_by_correlation_with_labels(R, T, L_train, topk=None, canonicalize=True): 
    """
    Predict new items/data through the correlation between probability predictions and true labels. 

    Only applicable in the case of user/classifier vs true labels

    Parameters 
    ----------
    L_train: training set labels

    Memo
    ----
    1. W is a 'global' property of the classifier based on the correlation between (probability) predictions 
       and true labels (this is different from the case in recommender system scenario)

    2. negatively correlated examples?

    """
    # Confidence weights based on correlation between predictions and true labels
    W = confidence_corr(R, L_train, mode='label') 

    # print('(predict_by_correlation) weight distribution:\n%s\n' % W)  # [note] similar
 
    # Only want those with positive correlations? 
    if topk: 
        topk_corr = np.argsort(W)[::-1][:topk]

        # only look at the rows that correspond to top k users/classifiers
        Th = W[topk_corr].dot(T[topk_corr, :]) / np.array([np.abs(W[topk_corr]).sum()])
    else: 
        Th = W.dot(T) / np.array([np.abs(W).sum()])

    if canonicalize: Th = canonicalize_prob(Th, name='Th')
    return Th
def predict_by_correlation(R, T, L_train, topk=None): 
    return predict_by_correlation_with_labels(R, T, L_train, topk=topk)

def predict_new_items(R, T, S=None, kind='item', topk=None, canonicalize=True, epsilon=1e-9): 
    """
    Same as predict_by_similarity() but return only the predictions of T (test split)
    """
    _, Th = predict_by_similarity(R, T, S=S, kind=kind, topk=topk, canonicalize=canonicalize)
    return Th

def predict_by_similarity(R, T, *, S=None, kind='user', topk=None, canonicalize=True, epsilon=1e-9):
    # return value: (Rh, Th)
    return uknn.predict_by_similarity(R, T, S=S, kind=kind, topk=topk, canonicalize=canonicalize)
    
# [alias]
def predict(R, T, **kargs): 
    return predict_by_similarity(R, T, **kargs)

def predict_debiased(ratings, similarity, kind='user'):
    """
    Predict ratings by substracting the means. 

    """
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        pred += user_bias[:, np.newaxis]
    elif kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]
        
    return pred

def predict_topk(ratings, similarity, kind='user', k=40):
    """

    Memo
    ----
    1. suppose a = [0, 1, 2, ... 9] 
       a[:-5:-1] => the last 4 (-1, -2, -3, -4) elements ~> 9, 8, 7, 6 

       the last k elements 
       a[:-k-1: -1], which then include the last kth element

       a[:-3]: a[0] up to but not include the last 3rd element 
              => [0, 1, 2, 3, 4, 5, 6]

       a[:-3: -1]: counting from the back, up to but not include the last 3rd element 
               => [9, 8]  (excluding 7)

    2. np.argsort() return the indices that sort the element in ascending order

    """

    pred = np.zeros(ratings.shape)
    if kind == 'user':
        assert ratings.shape[0] == similarity.shape[0], "n_users inferred from ratings: %d but dim(similarity): %d" % \
            (ratings.shape[0], similarity.shape[0])
        for i in range(ratings.shape[0]):
            top_k_users = tuple([np.argsort(similarity[:,i])[:-k-1:-1]])
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) # take the dot product (weighted sum) from the k users
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    elif kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = tuple([np.argsort(similarity[:,j])[:-k-1:-1]])
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    return pred

def runClustering(A, n_clusters=-1, method='kmeans', **kargs): 
    # import cluster

    if method == 'kmeans': 
        return kmeansCluster(A, n_clusters, **kargs)
    elif method.startswith('spe'): 
        return spectralCluster(A, n_clusters, **kargs)

    # else try to find the function in a particular module 

    tMethodSupported = False
    naming_protocal = '{base}Cluster'.format(base=method)
    for clustering_method in [method, naming_protocal, ]:
        try: 
            clustering_func = getattr(cluster, clustering_method)
            if hasattr(clustering_func, '__call__'): 
                tMethodSupported = True
        except: 
             pass
    if tMethodSupported: 
        return clustering_func(A, n_clusters=n_clusters, **kargs)

    # last resort is to find a supporting function in the current globals namespace
    return globals()[naming_protocal](A, n_clusters=n_clusters, **kargs)

def spectralCluster(A, n_clusters=-1, **kargs):  # A: latent factors P, Q
    # import cluster
    # from scipy.spatial import distance   # todo: use angular distance

    if n_clusters == -1: 
        n_clusters = A.shape[1]  # dimension of the latent factor matrix (e.g. user matrix, item matrix)

    if kargs.get('verbose', True): print('(clustering) method: spectral, n_clusters: {0} | dim(A): {1}'.format(n_clusters, A.shape))
    S = to_affinity(A, sig=kargs.get('bandwith', 0.5)) 
    return cluster.spectralCluster(X=S, n_clusters=n_clusters)  # return cluster label IDs (a numpy.ndarray)

def kmeansCluster(X, n_clusters=-1, **kargs): 
    # import cluster

    # from scipy.spatial import distance   # todo: use angular distance
    if n_clusters == -1: 
        n_clusters = X.shape[1]  # dimension of the latent factor matrix (e.g. user matrix, item matrix)

    if kargs.get('verbose', True): print('(clustering) method: kmeans, n_clusters: {0} | dim(A): {1}'.format(n_clusters, X.shape))

    return cluster.kmeansCluster(X, n_clusters=n_clusters)

def nmfCluster(P, Q, n_clusters=-1, **kargs):
    """

    Reference
    ---------
    1. document clustering based on non-negative matrix factorization by W Xu

    """
    from numpy import linalg as LA

    ### latent factors P, Q obtained from NMF 
    if n_clusters == -1: 
        n_clusters = P.shape[1]  # dimension of the latent factor matrix (e.g. user matrix, item matrix)
        assert n_clusters == Q.shape[1]
    
    # 'latent' or 'argmax' simply means we define clusters within the latent feature space itself without constructing a separate similarity matrix ...
    # ... and without running a separate clustering algorithm on the latent feature space
    method = kargs.get('method', 'argmax')  

    if method in ('argmax', 'latent',) and n_clusters == P.shape[1]: 
        
        # normalize P so that the Eulidean length of its column vector is 1 ... 
        # ... but also want P Q' to remain the same
        for k in range(P.shape[1]): # for each latent dimension
            u =  LA.norm(P[:, k], 2)
            P[:, k] = P[:, k] / u   # P[:, k] is an 1D array, a slice of column of P
            Q[:, k] = Q[:, k] * u
        
        user_labels = np.argmax(P, axis=1)
        item_labels = np.argmax(Q, axis=1)
    else: 
        ## use case: 
        #     methods that require building an affinity matrix e.g. spectral clustering 
        #     methods that compute clusters on the latent feature space e.g. kmeans 

        nCU = nCI = n_clusters
        if 'n_clusters_user' in kargs: nCU = kargs['n_clusters_user']
        if 'n_clusters_item' in kargs: nCI = kargs['n_clusters_item'] 

        user_labels = runClustering(P, nCU, **kargs) # kmeansCluster(P, n_clusters, **kargs)
        item_labels = runClustering(Q, nCI, **kargs) # kmeansCluster(Q, n_clusters, **kargs)

    ## cluster evaluation 
    if kargs.get('evaluate', False):
        # a. evaluate user  
        if 'U' in kargs: 
            outputdir = os.path.join(ProjectPath, 'cluster_analysis')
            cluster.clusterDistribution(user_labels, labels=U, outputdir=outputdir)

    return (user_labels, item_labels)

def evaluateClustering(cluster_labels, labels, **kargs):
    import cluster

    ## purity scores
    #  keys: ['unique_labels', 'purity_score', 'cluster_labels', 'ratios', 'fractions', 'ratios_max_votes', 'ranked_ratios', ]
    ret = cluster.eval_cluster(clusters, labels, cluster_to_labels=None, **kargs) 

    return 

def predict_by_cluster(R, T, similarity, kind='user', C=[], canonicalize=True):
    def cluster_label_to_group_indices(cluster_ids): 
        n_clusters = len(np.unique(C))
        groups = {c:[] for c in np.unique(C)}
        
        for i, cid in enumerate(C):             
            groups[cid].append(i)   # maps cluster ID to its corresponding positional indices in C
        
        return groups

    test_offset = R.shape[1] 
    if len(C)==0: 
        print('(predict_by_cluster) No cluster labels provided, which reduces to regular predict() routine that uses the whole similarity matrix ...')
        return predict_by_similarity(R, T, S=similarity, kind=kind)

    ### main subroutine
    ratings = np.hstack((R, T))

    # C: cluster membership
    #    if kind == 'user', then |C| <- n_users 
    #    if kind == 'item', then |C| <- n_items
    groups = cluster_label_to_group_indices(C)  # C is a list of labels (e.g. those returned from model.labels_, where model is a sklearn.cluster algorithm)

    pred = np.zeros(ratings.shape)
    if kind == 'user':
        assert ratings.shape[0] == similarity.shape[0], "n_users inferred from ratings: %d but dim(similarity): %d" % \
            (ratings.shape[0], similarity.shape[0])

        # Each user/classifier must be associated with its preferred topk user/classifier choices, from which predictions are to be made.
        n_users = ratings.shape[0]
        assert len(C) == n_users, "|C|={0} while ratings[0]={1}".format(len(C), n_users)

        for i in range(n_users):
            # top_k_users = tuple([np.argsort(similarity[:,i])[:-k-1:-1]])
            top_users = groups[C[i]]  # C[i] -> cluster ID, gropus[cid] -> indices of data

            for j in range(ratings.shape[1]):  # foreach item
                pred[i, j] = similarity[i, :][top_users].dot(ratings[:, j][top_users]) # take the dot product (weighted sum) from the k users
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_users]))

    elif kind == 'item':
        assert ratings.shape[1] == similarity.shape[0]

        n_items = ratings.shape[1]
        assert (len(C) == n_items), "|C|={0} while ratings[1]={1}".format(len(C), n_items)

        for j in range(n_items):
            
            # top_k_items = tuple([np.argsort(similarity[:,j])[:-k-1:-1]])
            top_items = groups[C[j]]
            
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_items].dot(ratings[i, :][top_items].T)  # look at all the items j that user i has ever rated, find weighted average of r(i, j)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_items]))        

    Rh = pred[:, :test_offset]
    if canonicalize: Rh = canonicalize_prob(Rh, name='Rh')  # why? 
    Th = pred[:, test_offset: ]
    if canonicalize: Th = canonicalize_prob(Th, name='Th')

    assert Rh.shape[1] == R.shape[1]
    assert Th.shape== T.shape, "dim(T):{0} but dim(Th):{1}".format(T.shape, Th.shape)

    return (Rh, Th) 

# def trainTestSplitRatings(df, test_size=0.25): 
#     import pandas as pd

#     ### Split the dataframe into a train and test set
#     from sklearn.model_selection import train_test_split
#     train_data, test_data = train_test_split(df, test_size=test_size) # use customized

#     train_data = pd.DataFrame(train_data)
#     test_data = pd.DataFrame(test_data)

#     # [redundant]
#     n_u = len(df["user_id"].unique())
#     n_m = len(df["item_id"].unique())

#     # Create training and test matrix
#     R = np.zeros((n_u, n_m))
#     for line in train_data.itertuples():
#         R[line[1]-1, line[2]-1] = line[3]  
    
#     T = np.zeros((n_u, n_m))
#     for line in test_data.itertuples():
#         T[line[1]-1, line[2]-1] = line[3]

#     return (R, T)

def train_test_split0(ratings):
    print('... call train_test_split')
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        # print('... ratings[user, :].nonzero()[0]: %s' % ratings[user, :].nonzero()[0])
        # print('... dim(test_ratings): %s' % str(test_ratings.shape))
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

def evalTestTset(P, Q, T, **kargs): 
    from utils_plot import saveFig, plot_path
    
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

def get_mse(pred, actual):
    # from sklearn.metrics import mean_squared_error

    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def model_select(train, test, user_similarity, item_similarity, k_array=[], **kargs): 
    """

    Memo
    ----
    1. user similarity matrix has to include all users 
       item similarity matrix has to include all items


    """
    from utils_plot import saveFig, plot_path
    import seaborn as sns
    
    # select the k in k-NN based CF 
    if not k_array: 
        k_array = [5, 15, 30, 50, 100, 200] 

    user_train_mse = []
    user_test_mse = []
    item_test_mse = []
    item_train_mse = []
    for k in k_array:
        user_pred = predict_topk(train, user_similarity, kind='user', k=k)
        item_pred = predict_topk(train, item_similarity, kind='item', k=k)
    
        user_train_mse += [get_mse(user_pred, train)]
        user_test_mse += [get_mse(user_pred, test)]
    
        item_train_mse += [get_mse(item_pred, train)]
        item_test_mse += [get_mse(item_pred, test)]  

    ### plot k vs MSK
    plt.clf()

    sns.set()
    pal = sns.color_palette("Set2", 2)

    plt.figure(figsize=(8, 8))
    plt.plot(k_array, user_train_mse, c=pal[0], label='User-based train', alpha=0.5, linewidth=5)
    plt.plot(k_array, user_test_mse, c=pal[0], label='User-based test', linewidth=5)
    plt.plot(k_array, item_train_mse, c=pal[1], label='Item-based train', alpha=0.5, linewidth=5)
    plt.plot(k_array, item_test_mse, c=pal[1], label='Item-based test', linewidth=5)
    plt.legend(loc='best', fontsize=20)
    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('k', fontsize=30);
    plt.ylabel('MSE', fontsize=30);

    plotName = kargs.get('plot_name', 'select-k')
    saveFig(plt, plot_path(name=plotName), message='select the best k for rating prediction')

    return

def demo_misc(**kargs): 

    ### np.random.choice(a, size=None, replace=True, p=None)
    #    If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)

    # Generate a non-uniform random sample from np.arange(5) of size 3 without replacement
    np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])

    ### np.argsort() ~ indices that sort the array
    # Numpy array created 
    a = np.array([9, 3, 1, 7, 4, 3, 6]) 
    # unsorted array print 
    print('Original array:\n', a)

    # Sort array indices 
    b = np.argsort(a) 
    print('Sorted indices of original array->', b)

    # To get sorted array using sorted indices 
    # c is temp array created of same len as of b 
    c = np.zeros(len(b), dtype = int) 
    for i in range(0, len(b)): 
        c[i]= a[b[i]] 
    print('Sorted array->', c)

    ### measure closeness of two distributions (or predictions vs true labels)
    div(message="Measure closeness of two distributions", symbol='*', border=2)
    # a. 
    labels = [0, 1, 0, 0, 1, 1, 1, 0]
    probs = [0.2, 0.9, 0.4, 0.3, 0.7, 0.7, 0.8, 0.1]
    print('> labels:\n%s\n  probs:\n%s\n  > corr:\n%s\n' % (labels, probs, np.corrcoef(probs, labels)))

    ### correlation 
    a = np.array([1,2,3,4,6,7,8,9])
    b = np.array([2,4,6,8,10,12,13,15])
    c = np.array([-1,-2,-2,-3,-4,-6,-7,-8])
    W = np.corrcoef([a,b,c])
    # outputs: 
    # 
    # array([[ 1.        ,  0.99535001, -0.9805214 ],
    #    [ 0.99535001,  1.        , -0.97172394],
    #    [-0.9805214 , -0.97172394,  1.        ]])

    df = DataFrame(np.random.random((5, 5)), columns=['gene_' + chr(i + ord('a')) for i in range(5)]) 
    print(df)

    #      gene_a    gene_b    gene_c    gene_d    gene_e
    # 0  0.471257  0.854139  0.781204  0.678567  0.697993
    # 1  0.292909  0.046159  0.250902  0.064004  0.307537
    # 2  0.422265  0.646988  0.084983  0.822375  0.713397
    # 3  0.113963  0.016122  0.227566  0.206324  0.792048
    # 4  0.357331  0.980479  0.157124  0.560889  0.973161

    correlations = {}
    columns = df.columns.tolist()

    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])

    result = DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']

    print(result.sort_index())

    #                      PCC   p-value
    # gene_a__gene_b  0.461357  0.434142
    # gene_a__gene_c  0.177936  0.774646
    # gene_a__gene_d -0.854884  0.064896
    # gene_a__gene_e -0.155440  0.802887
    # gene_b__gene_c -0.575056  0.310455
    # gene_b__gene_d -0.097054  0.876621
    # gene_b__gene_e  0.061175  0.922159
    # gene_c__gene_d -0.633302  0.251381
    # gene_c__gene_e -0.771120  0.126836
    # gene_d__gene_e  0.531805  0.356315

    ## regular 1D array to 2D column vector 
    a2 = np.array([[1, 3, 5], [2, 4, 6]])
    print ( a2.mean(1) ) # this is 1D array 
    print ( a2.mean(1)[:, None]) # a2.mean(1)[:, np.newaxis], this is a 2D array with one column vector

    return

def center(A, kind='user'): 
    """

    Memo
    ----
    1. use outer product or numpy broadcasting 

       https://bic-berkeley.github.io/psych-214-fall-2016/subtract_means.html
    """
    # nu, ni = A.shape[0], A.shape[1]
    # broadcast_centered = np.zeros((nu, ni))

    if kind.startswith( ('u', 'r') ):  # user, row
        row_bias = row_means = np.mean(A, axis=1) 
        
        # row_means_col_vec = row_means.reshape((ratings.shape[0], 1))  # Better: np.newaxis
        # broadcast_centered = ratings - row_means_col_vec
        broadcast_centered = (A - row_bias[:, np.newaxis]).copy() # turns row_means into a column vector

        # [test] should be all 0s
        # assert sum(broadcast_centered.mean(axis=1)) < 1e-9
    else: 
        col_bias = A.mean(axis=0)
        broadcast_centered = (A - col_bias[np.newaxis, :]).copy()

        # assert sum(broadcast_centered.mean(axis=0)) < 1e-9

    return broadcast_centered

def maskFN(R, labels, p_threshold=[], marker=0, pos_label=1):
    Rp = R.copy()
    if not p_threshold: p_threshold = 0.5 

    if hasattr(p_threshold, '__iter__'):
        assert len(p_threshold) == Rp.shape[0]
        
        L = np.array(labels)
        nFN = 0
        for i in range(Rp.shape[0]):
            nFN += np.sum(Rp[i][(Rp[i] < p_threshold[i]) & (L == pos_label)]) 
            Rp[i][(Rp[i] < p_threshold[i]) & (L == pos_label)] = marker

    else: 
        L = np.array(labels)[np.newaxis, :]
        
        # [test]
        # print('(maskFN) R:\n%s\n' % R[:4, :4])
        nFN = np.sum( (R<p_threshold) & (L == 1) )

        Rp[(R<p_threshold) & (L == pos_label)] = marker

    print('(maskFN) nFN=%d' % nFN)
    return Rp
def maskFP(R, labels, p_threshold=[], marker=0, neg_label=0): 
    Rp = R.copy()
    
    if not p_threshold: p_threshold = 0.5 
    if hasattr(p_threshold, '__iter__'):
        assert len(p_threshold) == Rp.shape[0]
        
        L = np.array(labels)
        nFP = 0
        for i in range(Rp.shape[0]):
            nFP += np.sum(Rp[i][(Rp[i] >= p_threshold[i]) & (L == neg_label)])
            Rp[i][(Rp[i] >= p_threshold[i]) & (L == neg_label)] = marker

    else: # single global threshold
        L = np.array(labels)[np.newaxis, :]
        nFP = np.sum( (R >= p_threshold) & (L == neg_label) )
        Rp[(R >= p_threshold) & (L == 0)] = marker

    print('(maskFP) nFP=%d' % nFP)
    return Rp

def applyCoCluster(fold, **kargs):
    from surprise import KNNBasic, NMF, CoClustering
    from surprise import Dataset, accuracy
    from surprise.model_selection import train_test_split, GridSearchCV

    nc_users = kargs.get('n_cltr_u', 5) # n_cltr_u
    nc_items = kargs.get('n_cltr_i', 5)
    n_epochs = kargs.get('n_epochs', 100)
    method = kargs.get('method', 'cocluster')
    tRandomSplit = False
    tMergeTrainTest = True

    data = toUserItem(fold, to_surprise_format=True, merge_=tMergeTrainTest) # merge_: set to True to merge train and test split (need to know all users and all items)
    if tMergeTrainTest: assert 'X' in data, "No combined data (train + test) generated."
    
    ## Retrieve the trainset.
    X_train = data['X'].build_full_trainset()  # this produces an instance of Trainset 
    # inspect(X_train, message='Trainset instance containing all users and items')

    ## Build an algorithm, and train it.
    if method.lower() == 'cocluster': 
        algo = CoClustering(n_cltr_u=nc_users, n_cltr_i=nc_items, n_epochs=n_epochs)
        algo.fit(X_train)
    else: 
        raise NotImplementedError("Unrecognized method: %s" % method)
        
    P, Q = algo.pu, algo.qi
    print('[applyMF] dim(P): %s, dim(Q): %s' % (str(algo.pu.shape), str(algo.qi.shape)))  

    return P, Q 


def applyMF(fold, **kargs): 
    """
    An interface that uses Suprise package's matrix facrorization method to derive 
    latent feature representation and make predictions. 

    Memo
    ----
    1. After prediciton phase, we still need a combining rule for ensemble learning

    Reference
    ---------
    1. Matrix Factorization 
       https://surprise.readthedocs.io/en/stable/matrix_factorization.html

    2. Dataset source code [todo] add toR(): returning 2D array of rating marix (instead of a list of raw_ratings)
       
       https://surprise.readthedocs.io/en/v1.0.0/_modules/surprise/dataset.html

    """
    def inspect(ts, message=''):
        # todo: how to view the raw user ids? 
        # ts is an instance of Trainset

        if message: div(message=message, symbol='-', border=2)

        # ts: an instance of Trainset
        users = [u for u in ts.all_users()]  # Inner id of users
        print('... users/classifiers:\n%s\n' % users)

        # [log] ['NaiveBayes.0', 'NaiveBayes.1', 'NaiveBayes.2', ... ]
        print('... raw user ids:\n%s\n' % [ts.to_raw_uid(u) for u in ts.all_users()]) 

        items = [i for i in ts.all_items()][:100]
        print('... items/data:\n%s\n' % items)  # index starts from 0

        # index starts from 1 
        print('... raw item ids:\n%s\n' % [ts.to_raw_iid(i) for i in items])

        # n_users: 30, n_times: 768 > n_ratings (|R| (+ |T|)): 23040
        print('... n_users: %d, n_items: %d > n_ratings (|R| (+ |T|)): %d' % (ts.n_users, ts.n_items, ts.n_ratings))
        print('... rating_scale: %s' % str(ts.rating_scale))
        
        return
    def make_unbiased_estimate(algo, data, mode='random', ratio=0.9): 
        # make unbiased estimate of the RMSE for the resulting factorization on the test split

        X_train = data['X']
        raw_ratings = X_train.raw_ratings  # all ratings including both train and test split
        # [note] R is a list of (user, item, rating, timestamp), where timestamp in our scenario is most likly just None 
 
        if mode.startswith('rand'): 
            # shuffle ratings if you want
            random.shuffle(raw_ratings)

            # 90% of the data for training and 10% of the data for testing
            threshold = int(ratio * len(raw_ratings))
            ratings_minus = raw_ratings[:threshold]
            ratings_test = raw_ratings[threshold:]

            # use: create a 
            X_train.raw_ratings = ratings_minus  # data is now the set ratings_minus 

            # X_train now holds a subset of the original 
            trainset = X_train.build_full_trainset()
            algo.fit(trainset)

            # Compute unbiased accuracy on B
            testset = X_train.construct_testset(ratings_test)  # testset is now the set B
            predictions = algo.test(testset)
            
            # print('+ Split: Random shuffling > Unbiased accuracy on test split,', end=' ') # syntax
            print('(make_unbiased_estimate() Random shuffling > Unbiased accuracy on test split ...')
            accuracy.rmse(predictions)
        else: # todo: how to specify test set? 
            raise NotImplementedError

        return

    from surprise import KNNBasic, NMF
    from surprise import Dataset, accuracy
    from surprise.model_selection import train_test_split, GridSearchCV

    # from surprise.model_selection import cross_validate

    ### Load data 
    #   > Also see demo_surprise.load_from_predefined_folds()
    n_factors = kargs.get('n_factors', 10)
    n_epochs = kargs.get('n_epochs', 300)
    method = kargs.get('method', 'nmf')
    tRandomSplit = False
    tMergeTrainTest = True

    data = toUserItem(fold, to_surprise_format=True, merge_=tMergeTrainTest) # merge_: set to True to merge train and test split (need to know all users and all items)
    if tMergeTrainTest: assert 'X' in data, "No combined data (train + test) generated."
    
    ## Retrieve the trainset.
    X_train = data['X'].build_full_trainset()  # this produces an instance of Trainset 
    # inspect(X_train, message='Trainset instance containing all users and items')

    ## Build an algorithm, and train it.
    if method.lower() == 'nmf': 
        algo = NMF(n_factors=n_factors, n_epochs=n_epochs, biased=True)  
        algo.fit(X_train)
    else: 
        raise NotImplementedError("Unrecognized method: %s" % method)
        
    P, Q = algo.pu, algo.qi
    print('[applyMF] dim(P): %s, dim(Q): %s' % (str(algo.pu.shape), str(algo.qi.shape)))  # ... ok
    # T = np.dot(P, Q.T)

    if kargs.get('evaluate_'): 
        eval_mode = kargs.get('evaluate_mode', 'random')
        ratio_train = kargsg.et('train_ratio', 0.9)
        make_unbiased_estimate(algo, data, mode=eval_mode, ratio=ratio_train)
    
    # now we still need a combined rule

    # [note]
    # return value conforms to the datasink format (see stacking module)
    # return DataFrame({'fold': fold, 'id': test_df.index.get_level_values('id'), 'label': test_labels, 'prediction': test_predictions, 'diversity': common.diversity_score(test_df.values)})
    return P, Q

def replace(P, Q, X, canonicalize=True, fill=0, predict_func=None, verify_=False, name='X'):  
    
    assert isinstance(X, (tuple, list)) and len(X) == 2, "(replace) X must be a 2-tuple: (W, R): {0}".format(X)
    W, R = X   # W: previously Cui 

    assert W.shape == R.shape, "Weight matrix (W) must have the same dimensionality as the rating matrix (R or T)"
    if predict_func is None: 
        predict_func = predict_by_factors
    else: 
        assert hasattr(predict_func, '__call__')

    nF = P.shape[1]; assert nF == Q.shape[1]
    n_users, n_items = P.shape[0], Q.shape[0]

    Rh = predict_func(P, Q, canonicalize=True, name=name)
    assert Rh.shape == R.shape, "dim({name}):{0} <> dim({name}h):{1}".format(R.shape, Rh.shape, name=name)

    # n_masked = np.sum(W==fill)
    n_values = len(np.unique(W))
    isBinaryWeight = True if n_values <= 2 else False

    if not isinstance(W, np.ndarray):
        W = np.array(W.todense())  # W has to be in dense form

    # if isBinaryWeight: 
        # this can be replaced by weighted average
        # Rh = np.where(W==fill, Rh, R)  # use Rh in place of R where the condition holds (i.e. wherever W == fill, typically referencing 'bad probabilities')
    W = W.astype(float)
    vmin, vmax = np.min(W), np.max(W)
    assert vmin >= 0.0 and vmax <= 1.0, "W's components were not standardized to [0, 1]!"

    # take weighted average 
    # W: preference matrix | confidence matrix | polarity matrix
    Rh = W * R + (1.0-W) * Rh   # if W[i,j] is higher, R[i,j] is more reliable  ... if W[i,j] lower, then 1-W must be higher => Rh[i,j]/reconstructed more important/reliable
     
    return Rh

def interpolate(X1, X2, W1, W2=None):
    """

    Use 
    ---
    X1: old rating matrix 
    X2: new rating matrix
    W1: probability filter (aka preference matrix)

        if W1[i,j]=1, then use X1[i, j]
        if W1[i,j]=0 => W2[i,j]=1 => use X2[i,j]; effectively replacing X1[i,j] by X2[i, j]

    Memo
    ----
    1. .todense() vs .toarray()

        toarray returns an ndarray; todense returns a matrix. If you want a matrix, use todense; otherwise, use toarray

    """
    import scipy.sparse as sparse
    assert X1.shape == X2.shape == W1.shape

    # if not isinstance(W1, np.ndarray): W1 = W1.toarray( )  # W1 has to be in dense form
    if sparse.issparse(W1):  
        W1 = W1.toarray()

    W1 = W1.astype(float)
    vmin, vmax = np.min(W1), np.max(W1)
    assert vmin >= 0.0 and vmax <= 1.0, "W1's components were not standardized to [0, 1]!"

    if W2 is None: 
        W2 = 1.0 - W1
    else: 
        # if not isinstance(W2, np.ndarray): W2 = W2.toarray() # np.array(W2.todense())  # W2 has to be in dense form
        if sparse.issparse(W2): W2 = W2.toarray()
        W2 = W2.astype(float)

    return W1 * X1 + W2 * X2   

def verify_mask(Cui, fill=0, predict_func=None): # replace(P, Q, Cui, canonicalize=True) 
    """

    Memo
    ----
    1. diabetes: 

       n_masked(R): 4751/18420, n_masked(T): 4620/4620 | n_total: 9371 (out of 23040, ratio: 0.40672743055555555)

    """ 
    nT = Cui.shape[0] * Cui.shape[1]  # nTR: n (replaced) Total on R
    if not isinstance(Cui, np.ndarray): 
        # Cui_R is in sparse format
        n_masked = np.sum(np.array(Cui.todense())==fill)
    else: 
        n_masked = np.sum(Cui==fill)  # number of entries replaced
     
    print('(verify) n_masked {n}/{N} | ratio: {r}'.format(
        n=n_masked, N=nT, r=n_masked/(Cui.shape[0]*Cui.shape[1]+0.0)) )

    return n_masked

def split_latent_matrix(Q, test_offset): 
    """

    Parameters 
    ----------
    Q: user/item vector in row-vector format (i.e. each user is represented as a row vector in Q)
    """

    train_set_only = False
    if Q.shape[0] == test_offset: 
        # only the training split (R) were used to compute P and Q    
        train_set_only = True

    if train_set_only: 
        Qr = Q
        Qt = None 
        print('... dim(P): {0}, dim(Qr): {1}, dim(Qt): {2}'.format(P.shape, Qr.shape, 'N/A'))
    else: 
        Qr = Q[:test_offset, :]
        Qt = Q[test_offset:, :]   # 30 * 10, (768-x) * 10
        print('... dim(P): {0}, dim(Qr): {1}, dim(Qt): {2}'.format(P.shape, Qr.shape, Qt.shape))

    return Qr, Qt

# prediction interface 
def predict_by_factors0(P, Q, test_offset, canonicalize=True, epsilon=1e-9): 
    """

    Memo
    ----
    1. weird bug if string formatting index doesn't start from 0 
       ~> tuple index out of range
    """
    Qr, Qt = split_latent_matrix(Q, test_offset)

    ### 

    Rh = np.dot(P, Qr.T)
    # Rh = np.array(Rh.todense())
    if canonicalize: Rh = canonicalize_prob(Rh, name='Rh')

    Th = None
    if Qt is not None: 
        Th = np.dot(P, Qt.T)   # [todo] predict interface
        # Th = np.array(Th.todense())
        if canonicalize: Th = canonicalize_prob(Th, name='Th')

    return (Rh, Th)

def predict_by_factors(P, Q, canonicalize=True, name='X'):
    Xh = np.dot(P, Q.T)
    # Rh = np.array(Rh.todense())
    print("(predict_by_factors) type(P): {}, type(Xh): {}".format(type(P), type(Xh)))
    if is_sparse(Xh): # not isinstance(Xh, np.ndarray): 
        Xh = Xh.A
    if canonicalize: Xh = canonicalize_prob(Xh, name=name)
  
    return Xh    

def ratio_of_alignment2(Xpf, Mc, Lh, verify=True, verbose=True, message=''):
    return pmodel.ratio_of_alignment2(Xpf, Mc, Lh, verify=verify, verbose=verbose, message=message)

def ratio_of_alignment(Xpf, Mc, verify=True, target_label=None):
    
    return pmodel.ratio_of_alignment(Xpf, Mc, verify=verify, target_label=target_label) 
    # rc: overall considerig both 0 and 1, rc_correct: consider only correct predictions (i.e. TP & TN)

def eval_alignment_by_precision(Xpf, Mc, Lh, by='alignment'):
    """
    Params
    ------
    policy: 
        'agreement': the fraction of entries in Xpf that is consistent with correctness matrix (Mc)
           objective: the preferred entries should those that correspond to correct predictions 

        'precision': TP/(TP+FP) computed from preferred entries (Xpf == 1)
    """
    return pmodel.eval_alignment_by_precision(Xpf, Mc, Lh, by=by)

def eval_alignment_by_recall(Xpf, Mc, Lh, by='alignment'): 
    return pmodel.eval_alignment_by_recall(Xpf, Mc, Lh, by=by)

# def eval_alignment_by_fscore(): 
def eval_alignment_by_fbeta(Xpf, Mc, Lh, by='preference', beta=1.0): 
    """

    Params
    ------
    beta: control the weight of recall

    The beta parameter determines the weight of recall in the combined score. 
    beta < 1 lends more weight to precision, while beta > 1 favors recall 
    (beta -> 0 considers only precision, beta -> inf only recall)
    """
    return pmodel.eval_alignment_by_fbeta(Xpf, Mc, Lh, by=by, beta=beta) # f_beta/(f_beta_bar+1.0)

def eval_alignment_minimize_fpfn(Xpf, Mc, Lh):
    return pmodel.eval_alignment_minimize_fpfn(Xpf, Mc, Lh)

def eval_alignment_hit_to_miss_ratio2(Xpf, Mc, Lh, error_avoidance=False): 
    """


    Memo
    ----
    1. examples
        ==> n_hit(TP): 1101, n_missed(FP): 6885 → Large?
        ==> n_hit(FP): 3880, n_missed(TP): 511 → Small?
    """
    return pmodel.eval_alignment_hit_to_miss_ratio2(Xpf, Mc, Lh, error_avoidance=error_avoidance)

def eval_alignment_hit_to_miss_ratio(Xpf, Mc, Lh, conditioned=True):
    return pmodel.eval_alignment_hit_to_miss_ratio(Xpf, Mc, Lh, conditioned=conditioned)  # ~ np.log(n_tp_hit * n_fp_missed)

# def eval_alignment_true_positive(Xpf, Mc, Lh): 
#     cells_tp = (Mc == 1) & (Lh == 1)  # but this is not necessarily good, because "true positive" is defined wrt estiamted labels (Lh)
#     return np.sum( (Xpf == Mc) & cells_tp)
def eval_alignment_positive(Xpf, Mc, Lh, verbose=False): 
    # Xpf: is a binary matrix
    return pmodel.eval_alignment_positive(Xpf, Mc, Lh, verbose=verbose)  # P(positive|aligned)

def eval_alignment(Xpf, Mc, Lh=None, conditioned=True): 
    # Xpf: is a binary matrix
    return pmodel.eval_alignment(Xpf, Mc, Lh=Lh, conditioned=conditioned)

def estimate_pref_threshold(Th, T, L=[], p_threshold=[], ratio_small_class=0.1, 
        pos_label=1, step=0.01, min_score=0.0, max_score=1.0, message=''): 
    if len(L) == 0: 
        assert len(p_threshold) > 0, "(estimate_pref_threshold) need proba threshold to estimate labels ..."
        L = lh = estimateLabels(T, L=[], p_th=p_threshold, pos_label=pos_label, ratio_small_class=ratio_small_class)
    correctness, _ = probability_filter(T, L, p_threshold)  # correctness matrix (Mc) entries: 1, if correct predictions (TP, TN); 0 o.w. 

    # calibrate Th (preference matrix) wrt the estimated correctness matrix (estimated via lh)
    Thb, pref_threshold, rc = calibrate_preference(Th, Mc=correctness, step=step, 
        min_score=min_score, max_score=max_score,
            message=message)
    return Thb, pref_threshold, rc

def calibrate_preference(Xpf, step=0.01, **kargs):
    """
    Used for binarizing a preference matrix by selecting the most appropriate preference threshould 
    whereby the preference matrix best captures the correct labeling. 
    """
    from functools import partial

    # fundamental quantities 
    X, L, p_threshold = kargs.get('X', None), kargs.get('L', []), kargs.get('p_threshold', [])
    tConservativeArgmax = kargs.get('conservative_argmax', False)

    Mc = kargs.get('Mc', None)
    Lh = kargs.get('Lh', None)
    if Mc is None: # if correctness matrix is not given ... 
        if X is None or len(L) == 0 or len(p_threshold) == 0: 
            msg = "(calibrate_preference) Need (X, L, p_th) to compute prediction correctness."
            raise ValueError(msg)
        Mc, Lh = probability_filter(X, L, p_th)
    if Lh is None:
        assert X is not None and len(p_threshold) > 0 
        Lh = estimateLabelMatrix(X, p_th=p_threshold)
    # ... correctness matrix Mc is given; optionally, Lh

    # [test]
    polarities = np.unique(Mc)
    assert min(polarities) == 0, "(calibrate_preference) polarities: {}".format(polarities)
    if Mc is not None: assert len(polarities) == 2
    if Lh is not None: assert len(np.unique(Lh)) == 2

    X = Xpf.copy() # make a copy to avoid overwriting the input Xpf

    #####################################################################
    min_score, max_score = kargs.get('min_score', 0.0), kargs.get('max_score', 1.0)

    # clipping
    X[X > max_score] = max_score
    X[X < min_score] = min_score

    # rescale to a value in [0, 1]
    if max_score != 1.0 and min_score != 0.0: 
        X = (X- min_score)/(max_score-min_score)
    #####################################################################
    # ... values in X are 'standardized' to values in [0, 1]

    nc = []  
    thresholds = np.arange(min_score, max_score, step)
    th_min = min_score + (max_score-min_score)/20.0

    policy = kargs.get('policy', 'agreement')
    ######################################
    policy_func = None 
    if policy.startswith('agree'):  # agreement  
        policy_func = partial(eval_alignment, conditioned=True)
    elif policy.startswith('pos'):  # positive
        policy_func = eval_alignment_positive
    elif policy in ('tp', 'true_positive'): 
        policy_func = eval_alignment_true_positive
    
    elif policy == 'min-false':  # minimize false predictions (including FP and FN)
        policy_func = eval_alignment_minimize_fpfn

    elif policy == 'precision': # default wrt preferred
        policy_func = partial(eval_alignment_by_precision, by='preference')
    elif policy.startswith('precision-align'): 
        policy_func = partial(eval_alignment_by_precision, by='alignment')
    elif policy == 'recall': # default wrt preferred
        policy_func = partial(eval_alignment_by_recall, by='preference')
    elif policy.startswith('recall-align'): 
        policy_func = partial(eval_alignment_by_recall, by='alignment')

    elif policy.startswith('f-pref'):  
        # ... leads to very low pref threshold
        # options: by={'alignment', 'preference'}
        policy_func = partial(eval_alignment_by_fbeta, by='preference')
    elif policy.startswith('f-align'): 
        # ... ok but still too high of a FP rate
        policy_func = partial(eval_alignment_by_fbeta, by='alignment')
    # elif policy == 'conditional-ratio': 
    #     policy_func = partial(eval_alignment_hit_to_miss_ratio, conditioned=True)   # old implementation: eval_alignment_hit_to_miss_ratio0
    elif policy.startswith( ('ratio', 'hit') ):  # hit-to-miss ratio
        # ... leads to very low pref_th and therefore won't miss any TP (and TN), but this is not useful
        # policy_func = partial(eval_alignment_hit_to_miss_ratio, conditioned=False)   # old implementation: eval_alignment_hit_to_miss_ratio0

        policy_func = partial(eval_alignment_hit_to_miss_ratio2, error_avoidance=False)
    elif policy.startswith('error'):  # error avoidance
        policy_func = partial(eval_alignment_hit_to_miss_ratio2, error_avoidance=True) 

    else: 
        raise NotImplementedError  
    ######################################

    for th in thresholds: 
        assert len(np.unique(X)) > 2, "X must not have been binarized here"
        Xb = binarize_pref(X, p_th=th, cutoff=False, min_score=min_score, max_score=max_score)  # p_th is a preference score threshold
        
        # optimization objective ... (1) maximizing agreement
        nc.append( policy_func(Xb, Mc, Lh) )  # prefered vs correct, how much are they aligned? 

    # [test] 
    #  1. precision: plateau after a certain threshould (all 1s beyond a certain point)
    
    ### choose the best index
    i0 = i =  np.nanargmax(nc) # np.argmax(nc) # the index (of threshold) that leads to the highest alignmen

    # posthoc adjustment
    max_value = max(nc)
    if tConservativeArgmax: 
        offset = max_value/20.0
        if nc[i] > max_value-offset: 
            
            # backtrack
            while i > 0: 
                i = i-1 
                if nc[i] <= max_value-offset: 
                    break   # this is the index we want 
        # ... adjusted index (i)

    pmax = thresholds[i]
    score = nc[i]  # best precision is the score

    # assert pmax >= th_min, "Very small preference threshold {} (<{}). Probably not a good idea".format(pmax, th_min)
    if pmax < th_min: 
        print("Very small preference threshold {} (<{}). Probably not a good idea".format(pmax, th_min))
        pmax = th_min
    print('(calibrate_preference) policy: {} | policy_func: {} | i: {} -> {} | sample scores:\n... {}\n'.format(policy, 
        policy_func,  # .__name__ if hasattr(policy_func, '__call__') else policy_func,    # note: partial function does not have __name__ attribute
        i0, i, nc[:100]))
    # ... found 'best' preference threshold pmax

    print('(calibrate_preference) pref threshold: {}, score: {} | message: {}'.format(pmax, score, kargs.get('message', 'using policy={}'.format(policy)) ))
    
    # ... ensure that values in X falls in [0, 1] and pmax was derived based on values in [0, 1]
    Xb = binarize_pref(X, p_th=pmax, cutoff=False, inplace=True, min_score=0.0, max_score=1.0)

    return Xb, pmax, score

def estimate_support_ratio(X, L, p_th, percentile=50, gamma=1.0): 
    Mc, Lh = probability_filter(X, L, p_th)  # Mc is a (0, 1)-matrix 
    # ratios = np.sum(Lh == L[None, :], axis=0)/Lh.shape[1]
    ratios = np.sum(Mc, axis=0)/Mc.shape[0]

    # ratio = np.percentile(ratios, percentile)
    ratio = np.mean(ratios) * gamma
    print("(estimate_support_ratio) support/user ratios | min: {}, max: {}, mean: {} | decision: {}, gamma: {}".format(np.min(ratios), np.max(ratios), np.mean(ratios), ratio, gamma))

    # m, q25, q50, q75, M = common.five_number(ratios)
    return ratio # np.mean(ratios)

def standardize(X, scaler=None, target_vars=['ks.pvalue', ], verbose=True): 
    """
    X: numpy array, or 
       a list of (list of dictionaries) for sequence model (e.g. CRF)
       

    """
    from sklearn.preprocessing import MinMaxScaler

    if isinstance(X[0][0], dict):  # X[0] is a list (of dictionaries)
        scaler = {}  
        N = len(X)

        if len(target_vars) == 0: 
            fset = set()
            for j in range(N): 
                for d in X[j]: 
                    fset.update( d.keys() )
        else: 
            fset = target_vars
        # ... if fset is empty, then no-op
        if len(fset) == 0: return X, {}

        # ... fit
        min_init, max_init = 1e4, -1e4 
        scaler = {f: {'min': min_init , 'max': max_init} for f in fset}  # value dictionary
        for f in fset: 
            
            for j in range(N): 
                dseq = X[j]
                fv = [di[f] for di in dseq if f in di]  # this variable across all feature dict 
                if len(fv) > 0: 
                    min_j, max_j = min(fv), max(fv)

                    if min_j < scaler[f]['min']: scaler[f]['min'] = min_j
                    if max_j > scaler[f]['max']: scaler[f]['max'] = max_j
                else: 
                    # if this feature is in none of the feature dictionaries, then skip
                    pass

        drop_list = []
        for f, entry in scaler.items(): 
            if entry['min'] == min_init or entry['max'] == max_init: 
                drop_list.append(f)  # drop candidates that were not observed in the training data
        for f in drop_list: 
            scaler.pop(f)
            if verbose: print("(standardize) dropping var {}".format(f))
        if verbose: 
            msg = "(standardize) feature scopes:\n"
            for f in fset:
                if f in scaler: 
                    msg += "... var: {} => {}\n".format(f, scaler[f]) 
            print(msg)
        # each feature now has its effective min and max (estimated from those actually observed in the training data)

        # ... transform
        # X2 = []
        for f in scaler.keys(): 
            for j in range(N): 
                # dseq2 = []
                for i, di in enumerate(X[j]): 
                    # di2 = {}
                    if f in di: 
                        # if scaler[f]['max'] > 1.0 or scaler[f]['min'] < 0.0
                        denom = (scaler[f]['max']-scaler[f]['min']+0.0) 
                        if denom > 0: 
                            di[f] = (di[f]-scaler[f]['min'])/denom
                        else: 
                            # no-op
                            # print("... warning: max-min is zero for var: {}".format(f))
                            pass
                # end foreach feature dictionary
            # end foreach feature dictionary sequence
    else: 
        if scaler is None: scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
    return X, scaler

def transform(X, scaler=None): 
    if scaler is None or not scaler: 
        # no-op
        return X

    if isinstance(X[0][0], dict):  # X[0] is a list (of dictionaries)
        assert isinstance(scaler, dict)

        # fset = list(scaler.keys())
        N = len(X)

        # ... transform
        # X2 = []
        for f in scaler.keys(): 
            for j in range(N): 
                # dseq2 = []
                for i, di in enumerate(X[j]): 
                    # di2 = {}
                    if f in di: 
                        # if scaler[f]['max'] > 1.0 or scaler[f]['min'] < 0.0
                        denom = (scaler[f]['max']-scaler[f]['min']+0.0) 
                        if denom > 0: 
                            di[f] = (di[f]-scaler[f]['min'])/denom
                        else: 
                            # no-op
                            # print("... warning: max-min is zero for var: {}".format(f))
                            pass
                # end foreach feature dictionary
            # end foreach feature dictionary sequence
    else: 
        X = scaler.transform(X)
    
    return X

# Polarity Modeling (shortcuts)
####################################################################

def get_feature_sequence(R, j, p_th, **kargs): 
     
    # Optional
    #    Rm=None, C=None, U=None, Lh=None, p_model={}, 
    #    name='', index=0, verbose=False, wsize=20
    
    # Output: a list of feature dictionaries, one per entry/classifier/user while holding column(j) fixed
    return pmodel.get_feature_sequence(R, j, p_th, **kargs)

def get_seq_stats(R, i, j, p_th, **kargs):  
    """

    Params
    ------
    tags: external dictionary keep track of per-position (i) statistics

    Reference
    ---------
    1. crfsuite: 

        https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#api-reference

    """
    # Optional parameters 
    #    Rm=None, C=None, U=None, p_model={}, r_min=0.1, name='', 
    #    index=0, verbose=False, wsize=20, to_dict=False, neg_label=0, pos_label=1, 
    #    tags={}, tagging_only=False, include_chain=True
    return pmodel.get_seq_stats(R, i, j, p_th, **kargs)

def get_vars_hstats(R, i, j, p_th, **kargs):  
    """

    Parameters
    ----------


    Memo
    ----

    """
    # Optional parameters
    #    Rm=None, C=None, Lh=None, U=None, Uc=None, encoder=None, p_model={}, r_min=0.1, name='', 
    #    index=0, verbose=False, wsize=20, to_dict=False, neg_label=0, pos_label=1
    return pmodel.get_vars_hstats(R, i, j, p_th, **kargs)

def get_vars_vstats(R, i, j, p_th, **kargs):
    # Optional parameters 
    #    Rm=None, C=None, Lh=None, p_model={}, 
    #    r_min=0.1, name='', index=0, verbose=False, wsize=10, to_dict=False, neg_label=0, pos_label=1
    return pmodel.get_vars_vstats(R, i, j, p_th, **kargs)

def polarity_correction(Po, **kargs):
    # Optional
    #    p_th=[], p_model={}, n_symbols_col=2, 
    #    pos_label=1, neg_label=0
    return pmodel.polarity_correction(Po, **kargs)

def polarity_sample_bootstrap(R, C, **kargs): 
    """

    Memo
    ----
    1. flatten a 2D (n-by-1 or 1-by-n) array to 1D 

       np.ndarray.flatten(x)


    """
    # Optional 
    #    p_th=[], Lr=[], p_model={}, 
    #    n_samples=100, Ro=None, target_label=1, pos_label=1, neg_label=0
    return pmodel.polarity_sample_bootstrap(R, C, **kargs) # return (Ra, Ca)

def polarity_feature_extraction(R, Lr, p_th, T, **kargs): 
    """
 
    Params
    ------
    max_size_kde: 
    max_size_user: max sample size used to keep track of the rating/probability values for each user/classifier.
    
    """
    # Optional parameters
    #     Lt=None, C=None, U=None, 
    #     max_size_kde=1000, max_size_user=10000,
    #     pos_label=1, neg_label=0, bag_count=10, fold_count=5, index=0
    return pmodel.polarity_feature_extraction(R, Lr, p_th, T, **kargs)

def define_outliers(R, p_th): 
    """

    Memo
    ----
    1. outiler detection in sklearn

       https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_housing.html#sphx-glr-auto-examples-applications-plot-outlier-detection-housing-py
    
    """
    return pmodel.define_outliers(R, p_th)

def polarity_modeling(R, Lr, p_th, T, **kargs): 
    # Optional parameters 
    #    Lt=None, C=None, U=None, policy='sequence', p_classifier='rf', constrained=True, 
    #    pos_label=1, neg_label=0, bag_count=10, fold_count=5, index=0
    return pmodel.polarity_modeling(R, Lr, p_th, T, **kargs)

def make_cn(C, Po, **kargs):
    """
    Given polarity matrix (Po), mask the neutral and negative entries in the confidence matrix so that 
    they do not enter the optimization objective (i.e. latent factors will not be made to approximate 
    these entries well because these entries do not matter). 

    Note
    ----
    1. this routine is effectively the same as make_cp() now ... 02.07.22
    """
    # Optional parameters
    #    is_unweighted=False, weight_neutral=0.0, weight_negative=-1.0, 
    #    sparsify=True, verbose=1
    return pmodel.make_cn(C, Po, **kargs)
# [alias]
mask_neutral_and_negative = make_cn

def make_cp(C, Po, **kargs):
    """
    Similar to make_cn() but mask only the nuetral (i.e. entries with so much uncertainty 
    that we do not know if they are TP, TN or FP, FN)

    Notes
    -----
    Naming: Cp? 
            The `p` emphasizes on "postive" i.e. only entries with positive polarities (e.g. TP, TN) 
            have non-zero weights 

    """
    # Optional parameters
    #    is_unweighted=False, 
    #    sparsify=True
    return pmodel.make_cp(C, Po, **kargs)
# [alias]
mask_neutral_only = make_cp

def polarity_to_preference(**kargs): 
    return to_preference(**kargs)
def to_preference_matrix(**kargs):
    return to_preference(**kargs)
def to_preference(Po, neutral=0.0):
    return pmodel.to_preference(Po, neutral=neutral) # {0, 1}

def to_polarity(M, verify=False): 
    return pmodel.to_polarity(M, verify=verify)
def preference_to_polarity(M):
    return to_polarity(M)
def to_polarity_matrix(M):
    # preference matrix {0, 1} to polarity matrix 
    return to_polarity(M)

def from_color_to_preference(M, **kargs): 
    # Optional parameters 
    #    codes={}, verify=True
    return pmodel.from_color_to_preference(M, **kargs)

def from_color_to_reduced_color(M, **kargs):
    """
    Convert from color (polarity) matrix to reduced color polarity matrix {-1, 0, 1, 2}
    """
    # Optional 
    #    codes={}, verify=True
    return pmodel.from_color_to_reduced_color(M, **kargs)

def from_color_to_polarity(M, **kargs): 
    """
    Convert from color (polarity) matrix to regular polarity matrix, whose non-zero entries 
    can only be either 1 or -1
    
    Converting color matrix to polarity matrix is useful for preference score approximation

    Note that both color matrix and polarity matrix can have 0s, whose corresponding probabilities 
    will not enter the optimization objective. 
    """
    # Optional 
    #   codes={}, verify=False
    return pmodel.from_color_to_polarity(M, **kargs)

def to_colored_preference_matrix(**kargs): 
    # use: for approximating ratings
    return to_colored_preference(**kargs)
def to_colored_preference(M, codes):
    return pmodel.to_colored_preference(M, codes)

def test_polarity(T, labels, **kargs):
    # Optional parameters 
    #    Pref=None, p_th=[], lh=[], name='T', pos_label=1, neg_label=0, title=''
    return pmodel.test_polarity(T, labels, **kargs)

def eval_polarity(Po, Mc, Lh, **kargs):     
    # Optional parameters
    #     pos_po=1, neg_po=-1, verbose=False, name='X', title=''
    return pmodel.eval_polarity(Po, Mc, Lh, **kargs)

def estimate_polarity(R, Lr, p_th, T, **kargs):
    # Optional parameters
    #     Lt=None, C=None, U=None, policy='sequence', 
    #     labeling_model='simple', p_classifier='rf', 
    #     constrained=True, stochastic=True, estimate_sample_type=True,
    #     k_upper=-1, k_lower=-1, k_max=-1, k_min=2, verbose=True, pos_label=1, neg_label=0, index=0
    return pmodel.estimate_polarity(R, Lr, p_th, T, **kargs)

def estimate_polarity_simple(R, Lr, p_th, T, **kargs):
    # Optional Parameters 
    #     policy='median', labeling_model='simple',
    #     p_classifier='rf',
    #     constrained=1, stochastic=True, 
    #     k_upper=-1, k_lower=-1, k_max=-1, k_min=1, verbose=True, pos_label=1, neg_label=0
    return pmodel.estimate_polarity_simple(R, Lr, p_th, T, **kargs)

def estimate_polarity_stacker(R, Lr, p_th, T, **kargs): 
    # Optional parameters 
    #    policy='median', labeling_model='logistic', 
    #    constrained=True, stochastic=True, p_classifier='rf',
    #    k_upper=-1, k_lower=-1, k_max=-1, k_min=2, verbose=True, pos_label=1, neg_label=0
    return pmodel.estimate_polarity_stacker(R, Lr, p_th, T, **kargs)

def estimate_polarity_stacker2(R, Lr, p_th, T, **kargs):
    # Optional params: 
    #    policy='median', labeling_model='logistic', 
    #    constrained=True, stochastic=True, 
    #    k_upper=-1, k_lower=-1, k_max=-1, k_min=1, verbose=True, pos_label=1, neg_label=0 
    return pmodel.estimate_polarity_stacker2(R, Lr, p_th, T, **kargs)

def probability_filter(X, L, p_th, *, target_label=None): 
    """
    Compute a binary matrix in which 1 represents a correct prediction (i.e. TP or TN), 
    and 0 represents a false prediction (i.e. FP or FN). Predicted labels (Lh) are determined by the given probability threshold (p_th). 
    
    Lh(X, p_th)

    Lh are compared with the "ground truth" L to determine the correctness. 


    Returns 
    -------
    A 2-tuple, where 

        first matrix is a probability filter comprising 0s and 1s (1 for TPs and TNs and 0 for FPs and FNs)
        second matrix is an estimated label matrix given proba thresholds
    """
    Lh = estimateLabelMatrix(X, p_th=p_th) # this is a label matrix NOT an estimate for true labels (lh); no 'majority vote' involved
    
    if target_label is not None: 
        Pf = ((Lh == L[None, :]) & (Lh == target_label)).astype(int)
    else: 
        Pf = (Lh == L[None, :]).astype(int)

    return (Pf, Lh)
def preference_matrix(X, L, p_th, **kargs):
    return probability_filter(X, L, p_th, **kargs) 
def correctness_matrix(X, L, p_th, **kargs):
    return probability_filter(X, L, p_th, **kargs)
def polarity_matrix(X, L, p_th, reduced_negative=-1, pos_label=1, neg_label=0): 
    Pf, Lh = probability_filter(X, L, p_th)
    return to_polarity(Pf), Lh

def is_color_matrix(Pc):
    return pmodel.is_color_matrix(Pc)
def color_matrix(X, L, p_th, **kargs): 
    """

    Parameters
    ----------
    X: Probability/rating matrix (this could be R, T, [R|T] or any arbitrary matrix)
    L: True labels

    """
    # Optional parameteres 
    #    reduced_negative=False, codes={},
    #    pos_label=1, neg_label=0
    return pmodel.color_matrix(X, L, p_th, **kargs) # return (Pc, Lh)

def compute_preference(P, Q, canonicalize=True, binarize=False, p_th=-1, name='Xh', verify=True, max_score=1.0, min_score=0.0): 
    # p_th: preference threshold used to binarize preference scores
    Xpf = np.dot(P, Q.T)  
    if canonicalize: 
        if binarize and p_th < 0: 
            print('(compute_preference) Warning: Attempting to binarize preference scores without a threshold (will use the grand mean as the threshold) ...')
        
        Xpf = canonicalize_pref(Xpf, binarize=binarize, p_th=p_th, name=name, verify=verify)
        # ... preference scores in [0, 1]

    return Xpf

def predict_by_importance_weights(X, W, aggregate_func='mean', fallback_on_low_weight=True, min_weight=0.1, axis=0):
    def fallback(pv, func): # closure: min_weight
        # consider the columns where all preference scores are zero, what to do? use the row mean
        for j in range(W.shape[1]):  
            if all(W[:, j] <= min_weight): 
                pv[j] = func(X[:, j])
        return 

    if scipy.sparse.issparse(W): W = W.toarray()
    W = W.astype(float)

    wcol_sum_to_zero = np.sum(W, axis=axis) == 0
    isDegenerated = any(wcol_sum_to_zero)  # each column should sum to a non-zero weight; o.w. it is degenerated
    wcol_idx = np.where(wcol_sum_to_zero)[0]
    if isDegenerated: 
        # it's possible that none of the classfier's predictions for a given data point was consider "reliable"; hence, some columns are all zeros
        print('(predict_by_importance_weights) Found degenerated cases: {} columns are all zeros!'.format(len(wcol_idx)))

        # revert to average? 
        for j in wcol_idx: 
            if np.all(W[:, j]==0): 
                W[:, j] = 0.5  # all equal weights  
    # W = softmax(W, axis=0)
        
    # ... all columns in Xpf has at least one non-zero 
    if aggregate_func == 'mean':

        pv = np.average(X, weights=W, axis=axis)

    elif aggregate_func == 'median': 
        Xh = X * W
        pv = np.zeros(X.shape[1]) 
        for j in range(X.shape[1]): 
            pv[j] = np.median(Xh[:, j][Xh[:, j]>0])  # take median of the non-zero values in a column
    else: 
        raise NotImplementedError

    if fallback_on_low_weight:
        func = None
        if aggregate_func == 'mean': 
            func = np.mean 
        elif aggregate_func == 'median': 
            func = np.median

        # consider the columns where all preference scores are zero, what to do? use the row mean
        if func is not None: 
            fallback(pv, func)
    return pv  

# factor: utilities
def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)

def predict_by_preference(X, Xpf, L=[], W=None, name='Xh', aggregate_func='mean', 
        fallback_on_low_weight=False, min_weight=0.01, verify=True): 
    """
    Params
    ------
    pref_threshold/-1, min_score/0, max_score/1: 
       only relevant when 'canonicalize' is True but usually, Xpf will have been canonicalized (e.g. binarized)
       before this call.

    """
    def fallback(pv, func): # closure: min_weight
        # consider the columns where all preference scores are zero, what to do? use the row mean
        for j in range(W.shape[1]):  
            if all(W[:, j] <= min_weight): 
                pv[j] = func(X[:, j])
        return 

    # similar to predict_by_importance_weights()

    # we shall assume that Xpf has been normalized
    if verify: 
        # value rescaling, clipping but leave binarize to calibrate_preference()
        vmin, vmax = np.min(Xpf), np.max(Xpf)
        assert vmin >= 0.0 and vmax <= 1.0
        # Xpf = canonicalize_pref(Xpf, name=name, L=L, binarize=False, p_th=pref_threshold, verify=verify)
        # ... (min_score, max_score) = (-1, 1) in {-1, 1}-polarity representation

    ########################################
    Xpf = Xpf.astype(float)  
    ########################################
    if W is not None: 
        # be aware if W is sparse, softmax() won't work because sparse matrix may not support exp
        if scipy.sparse.issparse(W): W = W.toarray()

        vmin = np.min(W)
        assert vmin >= 0.0, "(predict_by_preference) W's entries must be non-negative."
        if aggregate_func in ['mean', 'median' ]: 
            # W is used to amplify or suppress preference scores
            # ... usually W ~ confidence matrix
            print('(predict_by_preference) Coupling W to preference matrix')
            
            # normalize W first to ensure that W[:,j] falls in [0, 1]
            Xpf = Xpf * softmax(W, axis=0)

    if np.isnan(Xpf).any(): 
        print("(predict_by_preference) Pref matrix should not have NaNs but got n={}".format( np.sum(np.isnan(Xpf))) )
        Xpf = np.nan_to_num(Xpf)   # replace NaNs by 0s (not preferred)

    # [test]
    # if verify: 
    #     assert len(np.unique(Xpf)) == 2, "Xpf should a binary matrix but got values: {}".format(np.unique(Xpf))
    # ncol_zeros = np.sum( np.sum(Xpf, axis=0) == 0 )
    tHasDegenerated = any(np.sum(Xpf, axis=0) == 0)  # ncol_zeros > 0 
    degenerated_j = []
    if tHasDegenerated: 
        # it's possible that none of the classfier's predictions for a given data point was consider "reliable"; hence, some columns are all zeros
        # revert to average? 
        ncol_zeros = 0
        for j in range(Xpf.shape[1]): 
            if np.all(Xpf[:, j]==0.0): 
                Xpf[:, j] = 0.5  # assign some equal weights [note] remember to cast Xpf to float; o.w. this line has no effect!!! 
                ncol_zeros += 1 
                degenerated_j.append(j)
        print('(predict_by_preference) Found degenerated cases: {} columns are all zeros!'.format(ncol_zeros))
    # ... all columns in Xpf has at least one non-zero 

    if verify: 
        assert not np.any(np.sum(Xpf, axis=0) == 0), "Xpf should not have degenerated cases at this point but got {} cases".format(  np.sum(np.sum(Xpf, axis=0) == 0) )
        # assert not np.isnan(Xpf).any(), "Pref matrix should not have NaNs but got n={}".format( np.sum(np.isnan(Xpf)) )

        assert not np.isnan(X).any(), "Rating matrix should not have NaNs but got n={}".format( np.sum(np.isnan(X)) )
        min_pref, max_pref, median_pref = np.min(Xpf), np.max(Xpf), np.median(Xpf)
        min_rating, max_rating, median_rating = np.min(X), np.max(X), np.median(X)
        print('(predict_by_preference) min(pref): {}, max(pref): {}, median(pref): {}'.format(min_pref, max_pref, median_pref))
        print("... min(rating): {}, max(rating): {}, median(rating): {}".format(min_rating, max_rating, median_rating))
        assert min_pref >= 0.0, "Pref scores are non-negative!"
        assert min_rating >= 0.0, "Ratings are non-negative!"

    if aggregate_func == 'mean':
        pv = np.average(X, weights=Xpf, axis=0)

    elif aggregate_func == 'median': 
        # common.weighted_median(data, weights)

        Xh = X * Xpf
        pv = np.zeros(X.shape[1])
        # if binarize: 
        #     for j in range(X.shape[1]): 
        #         pv[j] = np.median(Xh[:, j][Xh[:, j]>0])
        # else: 
        #     pv = np.median(Xh, axis=0)
        n_all_negative = 0
        for j in range(X.shape[1]):
            pvj = Xh[:, j][Xh[:, j]>0]
            if len(pvj) > 0:   
                pv[j] = np.median(pvj)  # np.median() on an empty set results in 'nan'
            else: 
                n_all_negative += 1
                # print('(predict_by_preference) median mode | Found strange columns!\n{}\n'.format(Xh[:, j]))
                pv[j] = 0.0

        # [test]
        if n_all_negative: 
            print('(predict_by_preference) median mode | Found {} all-zero columns!\n'.format(n_all_negative))
    else: 
        raise NotImplementedError("unsupported aggregate_func: {}".format(aggregate_func))

    if fallback_on_low_weight:
        func = None
        if aggregate_func == 'mean': 
            func = np.mean 
        elif aggregate_func == 'median': 
            func = np.median

        # consider the columns where all preference scores are zero, what to do? use the row mean
        if func is not None: 
            fallback(pv, func)

    if verify: 
        n_nan = np.sum(np.isnan(pv))
        if n_nan > 0: 
            msg = "Predictions should not have NaNs but got n={} | aggregate_func: {}\n".format(n_nan, aggregate_func)
            print(msg)
            # pv2 = np.zeros(X.shape[1])
         
            for j in range(X.shape[1]):
                pvj = np.average(X[:, j], weights=Xpf[:, j])
                if np.isnan(pvj): 
                    err = "... offending vectors:\nX[:, j]={}\nXpf[:, j]={}\n".format(X[:, j], Xpf[:, j])
                    raise ValueError(msg + err)
        # pv should have 'probability-like' values as vector components
        min_pvj, max_pvj, median_pvj = np.min(pv), np.max(pv), np.median(pv)
        assert min_pvj >= 0.0 and max_pvj <= 1.0, "min(pv):{}, max(pv):{}, median(pv):{}".format(min_pvj, max_pvj, median_pvj)
        print('(predict_by_preference) Prediction vector (by {}+weights?{}) | min(pv):{}, max(pv):{}, median(pv):{}'.format(aggregate_func,
            True if W is not None else False, min_pvj, max_pvj, median_pvj))

        # assert not np.isnan(pv).any(), "Predictions should not have NaNs but got n={} | aggregate_func: {}".format(np.sum(np.isnan(pv)), aggregate_func)

    return pv 

def binarize_pref(X, p_th=-1, cutoff=False, max_score=1.0, min_score=0.0, rescale=True, inplace=False):
    # calibrate_preference -> binarize_pref 
    # canonicalize_pref -> binarize_pref
    A = X if inplace else X.copy() 
    
    if cutoff: 
        A[A > max_score] = max_score
        A[A < min_score] = min_score

    vmin, vmax = np.min(A), np.max(A)
    assert vmin >= min_score and vmax <= max_score   # [test]
    
    # preference calibration
    if p_th < 0: 
        p_th = np.mean(A)

    assert p_th >= vmin and p_th <= vmax  # [test]
    A[A >= p_th] = max_score
    A[A < p_th] = min_score

    return A

def canonicalize_pref(A, binarize=False, p_th=-1, name='', verify=1, min_score=0.0, max_score=1.0): 
    """
    L: estimated labels or true labels
    """
    if verify: 
        n_overflow = np.sum(A > max_score)
        n_underflow = np.sum(A < min_score)
        if n_overflow > 0 or n_underflow > 0: 
            if name: print('(canonicalize_pref) Matrix(%s) has illegal preference scores:' % name)
            print('... %d entries with p > 1!' % n_overflow)
            print('... %d entries with p < 0!' % n_underflow)
            n = n_overflow + n_underflow
            print('...... {0} illegal preference scores'.format(n))

            if verify > 1: 
                assert n == 0, "Input preference matrix has > 0 illegal entries (n={})".format(n)
    
    #####################################################################
    min_score, max_score = kargs.get('min_score', 0.0), kargs.get('max_score', 1.0)

    # clipping
    X[X > max_score] = max_score
    X[X < min_score] = min_score

    # rescale to a value in [0, 1]
    if max_score != 1.0 and min_score != 0.0: 
        X = (X- min_score)/(max_score-min_score)
    #####################################################################
    # ... values in X are 'normalized'

    if verify: 
        n_values = len(np.unique(A))
        if n_values <= 2: 
            print('(canonicalize_pref) Input matrix A had been binarized!')
            binarize = False
        else: 
            binarize = True
    if binarize: 
        if p_th < 0: 
            if verify: print('(canonicalize_pref) Missing preference threshold! Will use default np.mean(A) ...')
            p_th = np.mean(A)
        A[A >= p_th] = max_score
        A[A < p_th] = min_score

    return A

def canonicalize_prob(A, verbose=True, name='', epsilon=1e-9):
    
    if verbose: 
        n_overflow = np.sum(A > 1.0)
        n_underflow = np.sum(A < 0.0)
        if n_overflow > 0 or n_underflow > 0: 
            print('(canonicalize_prob) Matrix({}) has illegal probabilities:'.format(name if len(name)>0 else '?'))
            print('... {} entries with p > 1.0!'.format(n_overflow))
            print('... {} entries with p < 0.0!'.format(n_underflow))
            print('...... {} illegal probabilities'.format(n_overflow + n_underflow))
    
    A[A > 1.0] = 1.0 - epsilon 
    A[A < 0.0] = 0.0 + epsilon

    return A

# [data_pipeline_datasink]
def toUserItem(fold, **kargs): 
    import data_pipeline_datasink as dsp
    return dsp.toUserItem(fold, **kargs)


def demo_load_from_dataframe(**kargs): 
    """
    Prepare the input for Surprise package
    """
    import pandas as pd
    from surprise import NormalPredictor
    from surprise import Reader, Dataset
    from surprise.model_selection import cross_validate


    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                    'userID': [9, 32, 2, 45, 'user_foo'],
                    'rating': [0.3, 0.8, 0.7, 0.9, 0.1],

                    ## following attributes won't work
                    # 'prediction': [0.3, 0.8, 0.7, 0.9, 0.1], 
                    # 'label':      [0, 1, 1, 1, 0]}
                    }
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    # data = Dataset.load_from_df(df[['userID', 'itemID', 'prediction', 'label']],
    #                                 rating_scale=(0, 1))  # no longer works! => 'rating_scale' is not recognized! 

    # A reader is still needed but only the rating_scale param is required.
    # The Reader class is used to parse a file containing ratings.
    reader = Reader(rating_scale=(0, 1))

    # dataframe: It must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings, in this order.
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
   
    # We can now use this dataset as we please, e.g. calling cross_validate
    cross_validate(NormalPredictor(), data, cv=2)

    return

def demo_memory_based_recommender(**kargs): 
    def toRatings(df):
        n_users = df.user_id.unique().shape[0]
        n_items = df.item_id.unique().shape[0]

        ratings = np.zeros((n_users, n_items))
        for row in df.itertuples():
            ratings[row[1]-1, row[2]-1] = row[3] 

        return ratings

    import pandas as pd
    from sklearn.model_selection import train_test_split
    import getpass # portable way of getting username and password

    ### load data 
    # a. load from file
    user = getpass.getuser() # 'pleiades' 
    prefix = '/Users/%s/Documents/work/data/recommender' % user  # /Users/pleiades/Documents/work/data/recommender
    dataset = 'u.data'
    data_path = os.path.join(prefix, 'ml-100k/%s' % dataset)
    assert os.path.exists(data_path), "Invalid path: %s" % data_path

    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(data_path, sep='\t', names=names)
    print(df.head())

    # b. automatic generation based on datasink's output (run step1_generate or step1a_generate)
    # fold = random.randint(0, 4) # e.g. (0, 4), any int in [0, 4]
    # toUserItem(fold, split='train', save_=False)

    # regression-based algorithm 
    n_u = len(df["user_id"].unique())
    n_m = len(df["item_id"].unique())
    sparsity = len(df)/((n_u*n_m)+0.)
    print("sparsity of ratings is %.2f%%" %(sparsity*100))

    # evalTestTset(P, Q, T, **kargs)
    R = toRatings(df) 

    # centering R? 
    # R = center(R, kind='user')

    train, test = train_test_split(R)
    print('... dim(R): %s, dim(train): %s, dim(test): %s' % (str(R.shape), str(train.shape), str(test.shape)))
    print('... nU: %d, nI: %d' % (n_u, n_m))

    ### compute similarity matrix
    pairwise_similarity(train, kind='user')
    user_similarity = pairwise_similarity(train, kind='user')
    item_similarity = pairwise_similarity(train, kind='item')
    # print("... dim(sim(user): %s, dim(sim(item)): %s" % (str(user_similarity.shape), str(item_similarity.shape)) )
    print("... nU_train: %d, dim(train): %s, dim(user_similarity_train): %s" % (train.shape[0], str(train.shape), str(user_similarity.shape)))
    print("... nItem_train: %d, dim(train): %s, dim(user_similarity_train): %s" % (train.shape[1], str(train.shape), str(item_similarity.shape)))
    print (item_similarity[:4, :4])

    ### prediction
    item_prediction = predict_fast_simple(train, item_similarity, kind='item')
    user_prediction = predict_fast_simple(train, user_similarity, kind='user')

    print ('User-based CF MSE: ' + str(get_mse(user_prediction, test)))
    print ('Item-based CF MSE: ' + str(get_mse(item_prediction, test)))

    # user_similiarty derived from train 
    pred = predict_topk(train, user_similarity, kind='user', k=40) 
    print( 'Top-k User-based CF MSE: ' + str(get_mse(pred, test)) )

    pred = predict_topk(train, item_similarity, kind='item', k=40)
    print( 'Top-k Item-based CF MSE: ' + str(get_mse(pred, test)) )
    
    kx = [5, 15, 30, 50, 100, 200] 
    model_select(train, test, user_similarity, item_similarity, k_array=kx)

    return

def demo_to_rating_matrix(**kargs):
    import common
    # global ProjectPath, Domain
    
    ProjectPath = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    Domain = os.path.basename(ProjectPath) 

    kargs['p_threshold'] = 0.5
    for fold in [0, ]: # range(n_fold): 
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold)
        R, T, L_train, L_test, U = to_rating_matrix2(fold, **kargs)
        assert train_df.shape == R.T.shape 
        print('dim(train_df): {0} vs dim(R.T): {1}'.format(train_df.shape, R.T.shape))

    return 

def demo_cluster(**kargs):
    import numpy as np
    # import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    # plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # Incorrect number of clusters
    y_preds = runClustering(X, n_clusters=3, method='kmeans', random_state=random_state)
    # y_preds = kmeansCluster(X, n_clusters=3, random_state=random_state) 
    print('(test) dim(X):{0}'.format(X.shape))
    print('... labels:\n%s\n' % y_preds[:100])

    return

def demo_confidence_matrix(**kargs): 
    import data_pipeline as dp 
    import utils_cf as uc
    from utils_sys import highlight
    from analyzer import is_sparse
    import matplotlib.pylab as plt

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
                                        verbose=1)

    return

def demo_factor_similarity(**kargs):
    """

    Memo
    ----
    1.  transfer_factor_by_similarity()


    """
    from analyze_performance import Analysis
    import collections
    import getpass
    # debugging 
    np.set_printoptions(precision=3)

    domain = 'pf1' # 'diabetes_cf'
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    user = getpass.getuser() # 'pleiades'
    fill_marker = 0

    # load R, T 
    dev_ratio = 0.2
    max_dev = None
    df, labels = common.get_data(project_path, dataset='bp', fold_count=5)
    R, T, L_train, L_test, U =  shuffle_split_data(df, ratio=0.2, max_size=max_dev, unbag=True) # bag_count=10
    n_users_R, n_items_R = R.shape[0], R.shape[1]
    n_users_T, n_items_T = T.shape[0], T.shape[1]
    print("> dim(df): {dim}, n(train): {ntr}, n(test): {nt}".format(dim=df.shape, ntr=R.shape[1], nt=T.shape[1])) 

    # similarity 
    S = eval_cross_similarity(T, R, kind='item', epsilon=1e-9, unbiased=True)
    print('> dim(T): {dt}, dim(R): {dr} => dim(S): {ds}'.format(dt=T.shape, dr=R.shape, ds=S.shape))
    assert S.shape[0] == n_items_R
    assert S.shape[1] == n_items_T

    # S: train items vs test items
    top_k_items = k = 1 
    n_factors = 50
    n_users = R.shape[0]
    n_items = R.shape[1]
    n_items_test = T.shape[1]

    #################################################
    Pr, Qr = np.random.random((n_users, n_factors)), np.random.random((n_items, n_factors))  
    Pr[np.where(Pr >= 0.5)] = 1; Pr[np.where(Pr < 0.5)] = 0
    Qr[np.where(Qr >= 0.5)] = 1; Qr[np.where(Qr < 0.5)] = 0
    #################################################

    Pt, Qt = np.zeros((n_users, n_factors)), np.zeros((n_items_test, n_factors))
    for j in range(S.shape[1]): # foreach item (index) in T
        top_k_R = tuple([np.argsort(S[:,j])[:-k-1:-1]])  # find top k items in R most similar to jth item in T (column vector)
        if j % 10 == 0: print('> col({j}) | min: {m}, max: {M} | topk: {t}'.format(j=j, m=np.min(S[:,j]), M=np.max(S[:,j]), t=S[:, j][top_k_R]))
        
        w = S[:, j][top_k_R]  # weights 
        # print('> weights of the top k items in R: {w}'.format(w=w))

        # Qr = Q[top_k_R, :]  # select row vectors
        Qt[j] = np.average(Qr[top_k_R], axis=0, weights=w) # weigthed average of the topk most similar item's vectors
        assert len(Qt[j]) == n_factors
        if j % 10 == 0: 
            print('> Qt[{j}]: {value}'.format(j=j, value=Qt[j]))
            print('#' * 100)
            print('> weights: {w}'.format(w=w))
            print('> Qr[top_k_R, :]:\n{qr}\n'.format(qr=Qr[top_k_R]))

        # for i in range(ratings.shape[0]):
        #     pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
        #     pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))  
    
    return

def test(**kargs):
    
    ### demo: misc
    # demo_misc()

    ### data processing
    # demo_load_from_dataframe()

    # Convert training data in datasink format into rating matrices
    # demo_to_rating_matrix() 

    ### Surprise library  
    # demo_recommender()

    ### Memory-based Recommender (no parametric ML models are used; also include clustering-based, non-parametric methods)
    # demo_memory_based_recommender()

    ### Clustering
    demo_cluster()

    ### Confidence matrix 
    # demo_confidence_matrix()

    ### similarity and factor transfer 
    # demo_factor_similarity()
   
    return

if __name__ == "__main__": 
    test()



