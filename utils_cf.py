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

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier

# import snips as snp  # my snippets
# snp.prettyplot(matplotlib)  # my aesthetic preferences for plotting

from nnls import NNLS
import common, utilities

import utils_sys
from utils_sys import div

from evaluate import visualizeCoeffs

### Plotting 
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('ggplot')  # values: {'seaborn', 'ggplot', }
# from utils_plot import saveFig, plot_path

import cf_spec
from cf_spec import System

# try: 
#     from cluster import cluster  # works for python 2.7 but not python 3, why?
# except: 
# import cluster.cluster as cluster
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

class Polarity(object): 
    sample_types = ['tp', 'tn'] + ['fp', 'fn']
    codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
            'unk': 0, 't': 3, 'f': -3, 
            'pos': 1, 'neg': -1, '+': 1, '-': -1}

############################################################
# ... Predicates 

def isClassifierDim(cf_dim, setting=-1): 
    # given a filtering dimension key word, extracted from dset_id, is it in the classifier dimension? 
    is_cls = cf_dim.startswith(('u', 'cl'))  # user/classifier
    if setting > 0: 
        return is_cls and setting % 2 == 0   # e.g. cases in 2, 4, 8, 10
    return is_cls
def isDataDim(cf_dim, setting=-1): 
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

def slow_similarity(ratings, kind='user'):
    if kind == 'user':
        axmax = 0
        axmin = 1
    elif kind == 'item':
        axmax = 1
        axmin = 0
    sim = np.zeros((ratings.shape[axmax], ratings.shape[axmax]))
    for u in range(ratings.shape[axmax]):
        for uprime in range(ratings.shape[axmax]):
            rui_sqrd = 0.
            ruprimei_sqrd = 0.
            for i in range(ratings.shape[axmin]):
                sim[u, uprime] = ratings[u, i] * ratings[uprime, i]
                rui_sqrd += ratings[u, i] ** 2
                ruprimei_sqrd += ratings[uprime, i] ** 2
            sim[u, uprime] /= rui_sqrd * ruprimei_sqrd
    return sim

def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

# demo only
def pearson_correlation(object1, object2):
    values = range(len(object1))
    
    # Summation over all attributes for both objects
    sum_object1 = sum([float(object1[i]) for i in values]) 
    sum_object2 = sum([float(object2[i]) for i in values])

    # Sum the squares
    square_sum1 = sum([pow(object1[i],2) for i in values])
    square_sum2 = sum([pow(object2[i],2) for i in values])

    # Add up the products
    product = sum([object1[i]*object2[i] for i in values])

    #Calculate Pearson Correlation score
    numerator = product - (sum_object1*sum_object2/len(object1))
    denominator = ((square_sum1 - pow(sum_object1,2)/len(object1)) * (square_sum2 - 
        pow(sum_object2,2)/len(object1))) ** 0.5
        
    # Can"t have division by 0
    if denominator == 0:
        return 0

    result = numerator/denominator
    return result

def corr0(X,y):
    """
    X: n by k, y: 1 by k 
    find correlation between row vectors X[i] and y

    """
    # map(lambda x : numpy.correlate(x,y), X)

    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def corr(A,B, axis=1):
    # axis=1 => Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(axis)[:, np.newaxis]  # substract column mean
    B_mB = B - B.mean(axis)[:, np.newaxis]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(axis);
    ssB = (B_mB**2).sum(axis);

    # Finally get corr coeff
    # ssA[:, None]: 2D array, col vector;  ssB[None]: 2D array, row vector
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def evalCorrelation(R, kind='user', epsilon=1e-9, to_distance=False): 
    # from scipy.stats.stats import pearsonr
    if kind == 'user': 
        cor = np.corrcoef(R)
    else: 
        cor = np.corrcoef(R.T)

    # convert to similarity measure that falls in [0, 1]? 
    sim = cor
    if to_distance: 
        # [todo]
        sim = np.sqrt(1.0-cor)

    # W[i] = np.corrcoef(R[i, :], labels)[0, 1] # [0, 1] corr between 1st and 2nd variable
    return sim

# [todo]
def evalSimilarity(ratings, kind='user', epsilon=1e-9): 
    from sklearn.preprocessing import normalize

    if kind == 'user': 
        # each user rates items => user: row vectors
        Ru = normalize(ratings, axis=1, norm='l2')

        # ratings = (ratings - user_bias[:, np.newaxis]).copy()   # np.newaxis turns user_bias into a column vector
        sim = np.dot(Ru, Ru.T) # ratings.dot(ratings.T) + epsilon

    elif kind == 'item':
        # each item is rated by users => item: column vectors 
        # sim = ratings.T.dot(ratings) + epsilon
        Ri = normalize(ratings, axis=0, norm='l2')
        sim = np.dot(Ri.T, Ri)

    # norms = np.array([np.sqrt(np.diagonal(sim))])
    # return (sim / norms / norms.T)

    # make the matrix symmetric 
    # if make_symmetric: 
    #     sim = .5 * (sim + sim.T)  # make sure P is symmetric

    return sim

def transfer_factor_by_similarity(X, F, topk=1): 
    R, T = X  # transfer learned factor (P and Q) from R to T
    P, Q = F

    k = topk
    n_users, n_items = P.shape[0], Q.shape[0]  # user vectors and item vectors for R
    n_items_test = T.shape[1]
    n_factors = P.shape[1]
    assert n_factors == Q.shape[1]

    # similarity between T(j) and R(i)
    S = evaCrossSimilarity(T, R, kind='item', epsilon=1e-9, unbiased=True)
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
        # if j % 10 == 0: 
        #     print('> Qt[{j}]: {value}'.format(j=j, value=Qt[j]))
        #     print('#' * 100)
        #     print('> weights: {w}'.format(w=w))
        #     print('> Q[top_k_R, :]:\n{qr}\n'.format(qr=Q[top_k_R]))
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

    S = evaCrossSimilarity(T, R, kind='item', epsilon=1e-9, unbiased=True) # compute item-wise similarity between T(j) and R(i) ... 
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

def evalSimilarityByWeightedLatentFactors(R, C, F, is_user=True, lambda_val=0.8):
    pass
#     # import scipy.sparse as sparse

#     if is_user:
#         C_i = C[i] # .toarray()
#         R_i = R[i] # .toarray()
#     else: # is_item 
#         C_i = C[:, i].T # .toarray()
#         R_i = R[:, i].T # .toarray()

#     n_factors = F.shape[1]
#     lambda_eye = lambda_val * sparse.eye(n_factors) # np.matrix(lambda_val * np.eye(n_factors)) # * sparse.eye(n_factors)

#     FTF = F.T.dot(F)

#     CuI = sparse.diags(C_i, [0])  # per-user or per-item diagonal matrix

#     # Y'CuY => Y'Y + Y'(Cu-I)Y
#     FTCuIF = F.T.dot(CuI).dot(F)

#     W = (FTF+FTCuIF+lambda_eye).I  # FTF+FTCuIF+lambda_eye is a numpy matrix, where lambda_eye is a sparse matrix
#     # ... .I => inv(FTF+FTCuIF+lambda_eye)

#     return

def evalSimilarityByLatentFeatures(A, epsilon=1e-9):
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
    from sklearn.preprocessing import normalize

    A = normalize(A, axis=1, norm='l2')

    # Below is NOT recommmended see Memo [1]
    # sim = np.dot(A, A.T) # A.dot(A.T) # + epsilon
    # norms = np.array([np.sqrt(np.diagonal(sim))])
    # return (sim / norms / norms.T) 

    return np.dot(A, A.T)
### alias 
evalSimilarity2 = evalSimilarityByLatentFeatures

def toAffinity(A, sim_func=None, sig=0.5, verify=False):
    if sim_func is None: sim_func = evalSimilarityByLatentFeatures  # similarity falls in [0, 1]

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

def evaCrossSimilarity(T, R, kind='item', epsilon=1e-9, unbiased=True): 
    """
    Evaluate new item similarity (test split) wrt to existing items (train split). 

    T: test set scores/ratings (users vs items)
    R: training set scores/ratings  (users vs items)

    Memo
    ----
    1. T: u/3 * i/50, R: u/3 * i/100 => (50, 3), (3, 100) -> (50, 100)
   
       each ith row of T repr similarities betweeen instance i in T and all the intsances in R

    """
    if kind == 'item': 
        if unbiased: 
            Tc = demean(T, kind='item')
            Rc = demean(R, kind='item')

            # T(i): as column vectors 
            #    R: u/3 * i/100, T: u/3 * i/50 => (100, 3), (3, 50) -> (100, 50)
            # T(i) as row vectors: 
            #    T: u/3 * i/50, R: u/3 * i/100 => (50, 3), (3, 100) -> (50, 100)
            D = Rc.T.dot(Tc) + epsilon    
        else: 
            D = R.T.dot(T) + epsilon # leads to the similarity in terms of item_train by item_test
        
        # D: items_train vs items_test (similarity between train set and test set)

        assert D.shape[0] == R.shape[1], "n_row(D): {nrow} should be equal to size of test data (n_items): {nt}".format(nrow=D.shape[0], nt=T.shape[1])
        assert D.shape[1] == T.shape[1] 
    elif kind == 'user':  # T and R are separated by users, both have all items
        # if unbiased: 
        #     Tc = demean(T, kind='user')  # 3 * 50
        #     Rc = demean(R, kind='user')  # 3 * 100 => 3 * 100, 50  = > 5 users_train vs 2 users_test
        #     D = Rc.dot(Tc.T) + epsilon 
        # else: 
        #     D = T.dot(R.T) + epsilon # leads to the similarity in terms of item_train by item_test 
        
        # # D: users_train vs users_test
        # assert D.shape[0] == R.shape[0]
        # assert D.shape[1] == T.shape[0]
        msg = "Invalid dimension: {kind} unless each user in R and T represent exactly the same number of items (rare to be useful).".format(kind=kind)
        raise ValueError(msg)

    return D 

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

    # suppress the weights except for the topk 
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

def getWeightedAverage(W, T, topk=None): 
    if topk: 
        topk_corr = np.argsort(W)[::-1][:topk]
        # only look at the rows that correspond to top k users/classifiers
        pred = W[topk_corr].dot(T[topk_corr, :]) / np.array([np.abs(W[topk_corr]).sum()])
    else: 
        pred = W.dot(T) / np.array([np.abs(W).sum()])

    return pred

def toRatings0(df, test_size=0.25): 
    """
    Converts dataframe to rating matrix (users vs items)
    
    Memo
    ----
    1. see toPredictiveScores() for the ensemble learning setting. 
    """
    import pandas as pd

    # Split the dataframe into a train and test set
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(df, test_size=test_size)

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # [redundant]
    n_u = len(df["user_id"].unique())
    n_m = len(df["item_id"].unique())

    # Create training and test matrix
    R = np.zeros((n_u, n_m))
    for line in train_data.itertuples():
        R[line[1]-1, line[2]-1] = line[3]  
    
    T = np.zeros((n_u, n_m))
    for line in test_data.itertuples():
        T[line[1]-1, line[2]-1] = line[3]

    return (R, T)

def toRatings(df):
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3] 

    return ratings

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

# [data_pipeline] future: factor this function to data_pipeline module
def to_rating_matrix(fold, **kargs):
    """


    Memo
    ----
    1. train-dev-test split 
       <ref> https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213

       train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))]

    """
    # from cf_spec import System  # [note] the value configured in cf does not propogate here
    project_path = kargs.get('project_path', System.projectPath)
    print('(to_rating_matrix) Verify | domain: {0}, project path: {1} =?= {2}'.format(System.domain, project_path, ProjectPath))
    # project_path = kargs.get('project_path', '?')
    
    # tDev = kargs.get('include_devset', False)  # if True, return a train-dev-test split (instead of just train-test split)
    train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold) # [todo] single out this part
    if kargs.get('unbag', False):
        bag_count = kargs['bag_count'] if 'bag_count' in kargs else None
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        test_df = common.unbag(test_df, bag_count)

    R = train_df.values.T  # R: users vs items
    T = test_df.values.T 
    U = train_df.columns.values

    return (R, T, train_labels, test_labels, U)

def to_rating_matrix_dev(**kargs):
    # consider dev set
    rDev = kargs.get('dev_ratio', 1./System.foldCount)
    train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.read_fold2(ProjectPath, fold, dev_ratio=rDev)        
    assert dev_df.shape[0] > 0

    if kargs.get('unbag', False):
        # assert 'bag_count' in kargs
        bag_count = kargs['bag_count'] if 'bag_count' in kargs else None
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        dev_df = common.unbag(dev_df, bag_count)
        test_df = common.unbag(test_df, bag_count)

    R = train_df.values.T  # R: users vs items
    Td = dev_df.values.T
    Tt = test_df.values.T
    U = train_df.columns.values

    return (R, Td, Tt, train_labels, dev_labels, test_labels, U)

def shuffle_split(df, labels=[], ratio=0.2, max_size=None, **kargs): 
    """

    Use
    ---
    1. In model_select_core(), model selection is performed to choose the best parameter combination from among a set of candidates; we wish for each iteration
       in model selection to reference a different version of train-dev split sampled from a pre-specified train-dev split (i.e. the data minus the test set). 
    """
    from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule 

    index = kargs.get('index', -1)  # used for determining random state (and testing)
    
    # note: common.split() returns the output of a train_test_split call
    train_df, dev_df, train_labels, dev_labels = common.split(df, labels=labels, ratio=ratio, shuffle=True, max_size=max_size, index=index)  # shuffle + split

    # [test]
    print('(uc.shuffle_split) Cycle #{n} | counts(train_labels): {ctr} | counts(dev_labels): {cdev}'.format(n=kargs.get('index', '?'), ctr=collections.Counter(train_labels), cdev=collections.Counter(dev_labels)))

    if kargs.get('unbag', False):
        bag_count = kargs['bag_count'] if 'bag_count' in kargs else None
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        dev_df = common.unbag(dev_df, bag_count)

    R = train_df.values.T
    Td = dev_df.values.T
    U = train_df.columns.values
    L_train, L_dev = train_labels, dev_labels

    # test
    assert R.shape[1] == len(L_train)
    assert Td.shape[1] == len(L_dev)
    assert R.shape[0] == Td.shape[0]

    # apply resampling to the training data
    if kargs.get('resample', False):
        ver = 3
        # resampling_method = 'NearMiss(v{})'.format(ver)
        # sampler = NearMiss(version=ver)    # undersampling based on KNNs (version 3 is less affected by noise)

        resampling_method = "NeighbourhoodCleaningRule"
        R, L_train = apply_resample(R, L_train, method=resampling_method)

        # dev set 
        # Xd, Ld = Td.T, L_dev
        # Xd, Ld = sampler.fit_resample(Xd, Ld)
        # Td, L_dev = Xd.T, Ld
    
    return (R, Td, L_train, L_dev, U)

def apply_resample(X, L, method=''): 
    from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule 

    sampler = None
    if not method: 
        ver = 3
        method = 'NearMiss(v{})'.format(ver)
        sampler = NearMiss(version=ver)    # undersampling based on KNNs (version 3 is less affected by noise)
    
    if method.lower().startswith('neighb'):
        sampler = NeighbourhoodCleaningRule()  # sampling_strategy: 'auto' (resample all classes but the minority class)
        print('(apply_resample() resampling method: {}'.format(method))
    else: 
        raise NotImplementedError
 
    Xr, Lr = sampler.fit_resample(X.T, L)
    return Xr.T, Lr

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
    import pandas as pd
    from cf_spec import System

    project_path = kargs.get('project_path', System.projectPath)
    print('(to_rating_matrix_random_subsampling) domain: {0}, project path: {1} =?= {2} ... (verify)'.format(System.domain, project_path, ProjectPath))

    fold = kargs.pop('fold', -1)
    rDev = kargs.get('dev_ratio', 1./System.foldCount)
    rTest = kargs.get('test_ratio', 1./System.foldCount)

    # kargs['include_devset'] = True
    if fold > 0: return to_rating_matrix_dev(**kargs) # (R, Td, Tt, train_labels, dev_labels, test_labels, U)

    # ... otherwise, proceed with the random subsampling 
    fold_count = kargs.get('fold_count', System.foldCount)
    
    #########################################
    # ... consider subsampling of the entire dataset (todo)
    policy = 'random_cv_fold' # assuming that data within each fold is already a random subset of the entire dataset
    # if policy.startswith('random_cv'): 
    # else: 
    #     # for fold in range(fold_count):
    #     #     train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
    #     raise NotImplementedError
    #########################################
        
    # the dev set is slightly smaller than test test because dev set is derived from an inner cv partition
    shuffle = kargs.get('shuffle', True)
    train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.read_random_fold(project_path, fold_count=fold_count, dev_ratio=rDev, test_ratio=rTest, shuffle=shuffle)

    if kargs.get('unbag', False):
        # assert 'bag_count' in kargs
        bag_count = kargs['bag_count'] if 'bag_count' in kargs else None
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        dev_df = common.unbag(dev_df, bag_count)
        test_df = common.unbag(test_df, bag_count)

    # >>> by default, we will consider a train-dev-test split unless otherwise specified 
    tTrainDevTest = kargs.get('train_dev_test', True)
    if tTrainDevTest: 

        # print('... (verify) dev_df | type: %s, value: %s' % (type(dev_df), dev_df.head(10)))
        assert dev_df.shape[0] > 0 
        assert train_df.shape[1] == dev_df.shape[1] == test_df.shape[1], "dim(train_df): {0}, dim(dev_df): {1}, dim(test_df): {2}".format(train_df.shape, dev_df.shape, test_df.shape)

        print('... size(train): {Ntr}, size(dev): {Nd}, size(test): {Nt}'.format(Ntr=train_df.shape[0], Nd=dev_df.shape[0], Nt=test_df.shape[0]))
        R = train_df.values.T  # R: users vs items
        Td = dev_df.values.T
        Tt = test_df.values.T
        U = train_df.columns.values

        # >>> save index data, this is important when saving the CF-transformed training data
        if kargs.get('return_index', False): 
            train_combined = pd.concat([train_df, dev_df])
            Rx = train_combined.index # a MultiIndex with names=['id', 'label']
            Tx = test_df.index
            return (R, Td, Tt, train_labels, dev_labels, test_labels, U, Rx, Tx)

        return (R, Td, Tt, train_labels, dev_labels, test_labels, U)
    else: 
        train_df = pd.concat([train_df, dev_df])
        train_labels = np.hstack((train_labels, dev_labels))

        print('... size(train): {Ntr}, size(dev): {Nd}, size(test): {Nt}'.format(Ntr=train_df.shape[0], Nd=0, Nt=test_df.shape[0]))
        R = train_df.values.T  # R: users vs items
        T = test_df.values.T
        U = train_df.columns.values

        if kargs.get('return_index', False): 
            Rx = train_df.index # a MultiIndex with names=['id', 'label']
            Tx = test_df.index
            return (R, T, train_labels, test_labels, U, Rx, Tx)

        return (R, T, train_labels, test_labels, U)

# subsumed by to_rating_matrix()
def toPredictiveScores(fold, **kargs):
    """
    Same as to_rating_matrix() but perhaps this template code is easier to work with source codes
    in recommender system in general. 

    Memo
    ----
    1. analogous to toRatings()
    """
    train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
    if kargs.get('unbag', False):
        bag_count = kargs['bag_count'] if 'bag_count' in kargs else None
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        test_df = common.unbag(test_df, bag_count)

    # get all data IDs 
    users = train_df.columns.values

    # [note]
    #   train: predictive scores (analogous to 'ratings') in the training split 
    #   test:  predictive scores in the test split
    cols = ['train', 'test', 'train_labels', 'test_labels', ]  
    data = {col: None for col in cols}

    data['users'] = data['classifiers'] = users
    data['train_labels'] = train_labels; data['test_labels'] = test_labels
    
    for split in ['train', 'test', ]: 
        ts = train_df if split.startswith('tr') else test_df

        ts = ts.reset_index() # convert multilevel index to flat index
        idx = ts['id'].values  # item/data IDS
        assert len(idx) == len(set(idx)), "Data IDs are not unique!"
        labels = ts['label'].values # ground truth labels

        # split = 'train'
        nU = nUsers = len(users) # number of users/classifiers
        nI = nItems = len(idx)  # number of items/data points

        R = []
        # rating matrix for the training split
        for i, user in enumerate(users): 
            predictions = ts[user].values
            # print('(toPredictiveScores) clf: %s, predictions: %s' % (user, predictions[:10]))
            if i == 0: assert len(idx) == len(predictions)
            R.append(predictions)
        data[split] = np.array(R)

    return data  # a dictionary of 5 entries: ['train', 'test', 'train_labels', 'test_labels', 'users', ]

# # utils_cf
# def maskFN(R, labels, p_threshold=0.5, marker=0):
#     Rp = R.copy()
#     L = np.array(labels)[np.newaxis, :]
    
#     # [test]
#     # print('(maskFN) R:\n%s\n' % R[:4, :4])
#     nFN = np.sum( (R<p_threshold) & (L == 1) ); print('(maskFN) nFN=%d' % nFN)

#     Rp[(R<p_threshold) & (L == 1)] = marker

#     return Rp
# # utils_cf
# def maskFP(R, labels, p_threshold=0.5, marker=0): 
#     Rp = R.copy()
#     L = np.array(labels)[np.newaxis, :]
    
#     nFP = np.sum( (R >= p_threshold) & (L == 0) ); print('(maskFP) nFP=%d' % nFP)

#     Rp[(R >= p_threshold) & (L == 0)] = marker

#     return Rp

def toRatingMatrix(fold, **kargs): 
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
    def verify(A):
        n_total = A.shape[0] * A.shape[1]
        n_missing = n_total - np.count_nonzero(A)
        r_missing = n_missing/(n_total + 0.0)

        # print('... A[:10]:\n%s\n' % R[:10])
        print('toRatingMatrix> n_missing: %d, n_total: %d => ratio: %f' % (n_missing, n_total, r_missing)) 
        return

    missing_value = kargs.get('missing_value', 0)

    # a single floating number or a list
    p_threshold = kargs.get('p_threshold', 0.5)   # suggested: the probability threshold that leads to best fmax in the training data

    # thresholds = kargs.get('thresholds', [])
    
    # data = toPredictiveScores(fold, project_path=System.projectPath, unbag=kargs.get('unbag', False), bag_count=kargs.get('bag_count', -1))
    # R = data['train']  # "rating matrix" for the train split  # print('... R0:\n%s\n' % R[:10, :10])
    # T = data['test']
    # L_train, L_test = data['train_labels'], data['test_labels']
    # U = data['users'] if 'users' in data else np.array(range(R.shape[0]))
    R, T, L_train, L_test, U = to_rating_matrix(fold, **kargs)  # other params: project_path=System.projectPath, unbag=kargs.get('unbag', False), bag_count=kargs.get('bag_count', -1)
    assert len(U) == R.shape[0]

    # mask the entries of false predicted values
    if kargs.get('masked', True):

        # user/classifier dependent probability threshoulds
        # ... p_threshold is a list
        if hasattr(p_threshold, '__iter__'): assert len(p_threshold) == R.shape[0] 
        
        print('(toRatingMatrix) Fold=%d, masking FP and/or FN ...' % fold)
        R = maskFN(R, L_train, p_threshold=p_threshold, marker=missing_value)
        R = maskFP(R, L_train,  p_threshold=p_threshold, marker=missing_value)

    #[test]
    if kargs.get('verbose', True): 
        nMasked = np.sum( R == missing_value ); print('(toRatingMatrix) fold=%d > nMasked (nFN+nFP): %d' % (fold, nMasked))
        # print('... R:\n%s\n' % R[:10, :10])

    # [test] toRatingMatrix0() somehow outputs different ordering of probabilities ... ( )  but nMasked is correct!
    # R2, T2 = toRatingMatrix0(fold, p_threshold=p_threshold, merge_=False, missing_value=missing_value) # training matrix
    # nMasked = np.sum( R2 == missing_value ); print('(toRatingMatrix0) nMasked: %d' % nMasked)
    # print('... R2:\n%s\n' % R2[:10, :10])

    # assert np.array_equal(R, R2), "dim(R): %s, dim(R2): %s" % (str(R.shape), str(R2.shape))
    # assert np.array_equal(T, T2)

    return (R, T, L_train, L_test, U)  # U: users/classifiers

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
    A wrapper of different confidence matrix functions. 

    Compute confidence matrix. First compute all confidence scores given the confidence measure (conf_measure, e.g. 'brier'), followed by 
    masking the unreliable entries (zeroing out). Mask function is determined by at least 4 factors (see Memo). 


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

                'is_cascade': kargs.get('is_cascade', False), # if True, X is a concatenation of R and T

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
    Cui = Cui_bar = Po = None  # Mc
    ##############################################################################
    ret = toConfidenceMatrix(X, L,           
                        # confidence score parameters
                        conf_measure=params['conf_measure'], 
                        # conf_user=params['conf_user'], conf_item=params['conf_item'], 
                 
                        # mask function parameters
                        p_threshold=p_th,
                        policy=policy,   # filtering direction (e.g. user axis, item axis)
                        
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
                        alpha=params['alpha'],
                        beta=params['beta'],

                        # balance class sample distribution and weights
                        balance_class=params['balance_class'],
                        balance_and_scale=params['balance_and_scale'],
                        suppress_negative_examples=params['suppress_negative_examples'],
                        policy_polarity=params['policy_polarity'],   # options: classification, median
                        estimated_labels=params['estimated_labels'],
                        
                        # polarity matrix parameters
                        labeling_model=params.get('labeling_model', 'simple'),  # used to determine polarity matrix
                        constrained=params.get('constrained', True),
                        stochastic=params.get('stochastic', True), 
                        estimate_sample_type=params.get('estimate_sample_type', True),

                        # message passing 
                        message=M,   # 2-tuple (R, L_train) or 3-tuple (R, L_train, Cr)
                    
                        # outdated
                        mask_all_test=params['mask_all_test'], 

                        # debug/testing
                        U=kargs.get('U', []),
                        L_true=kargs.get('L_test', []),  # test the accuracy of the unsupervised estimte of labeling 
                        fold=fold, 
                        path=kargs.get('path', os.getcwd()), 
                        verbose=verbose 
                        )  
    Cui, Po, p_th, *rest = ret

    return (Cui, Po, p_th)

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
    if sparse.issparse(C): 
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

    tSparsify = False
    if sparse.issparse(C): 
        C = C.toarray()
        tSparsify = True  # convert back to sparse matrix when reweighting is done
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
    if len(U) > 0: assert len(U) == C.shape[0]

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
                    test_cases=[], plot=True if fold == 1 else False, 
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
    if Po is not None: 
        Mc = Po.toarray() if sparse.issparse(Po) else Po
    # ... Mc <- Po, as Po has a higher precedence
    # ... Mc is dense

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
        if verbose: print('(balance_and_scale) Suppress majority sample weights to {}'.format(w_min))
        
        # TNs
        C[ cells_tn ] = C[ cells_tn ] * 0.01

    if is_cascade and discount_test: 
        # gamma = 0.5
        if verbose: print('(balance_and_scale) Discount test sample weights by {}'.format(gamma))
        Cr, Ct = C[:,:n_train], C[:,n_train:]
        Ct = Ct * gamma
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
                    test_cases=test_cases, plot=True if fold == 1 else False, 
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
    if sparse.issparse(Po) or sparsify: # then Cn must also be sparse to be consistent
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

    is_cascade = kargs.get('is_cascade', False) # [note] cascade mode seems more favorable (i.e. X=[R|T])

    # Probability filtering policy for the training set
    # policy_filtering = kargs.get('policy', 'item') # [obsolete]

    # Probability filtering policy for the test set (only relevant in casade mode i.e. X = [R | T])
    # policy_filtering_test = kargs.get('policy_test', 'polarity') if is_cascade else None # None as 'undefined'

    n_train = kargs.get('n_train', -1)
    # ... if not in cascade mode (where X = [R | T]), then policy_filtering_test is undefined and not meaningful

    null_marker = kargs.get('fill', 0) # [todo] marker for missing data
    # topk_users, topk_items = 0, 0  # default, use all by setting to 0
    pos_label = kargs.get('pos_label', 1)
    n_users, n_items = X.shape[0], X.shape[1]
    fold = kargs.get('fold', -1)  # only used for debugging
    
    conf_measure = kargs.get('conf_measure', 'brier')
    U = users = kargs.get('U', []) # names of users/classifiers
    tSupervised = True # kargs.get('supervised', True) # Note: Always make use of the training set's labels if possible
    verbose = kargs.get('verbose', 1)

    # [design] Balancing class weights is factored to cf.wmf-related routines (e.g. wmf_ensemble_iter, wmf_ensemble_iter2())
    #####################################################################################
    # Balance class weights in the confidence score 

    # tBalanceClassWhileMasking = kargs.get('balance_class', False) 
    # tBalanceClassWeights = kargs.get('balance_and_scale', False) # False if conf_measure == 'rank' else True
    # tSuppressMaxClass = kargs.get('suppress_negative_examples', False)
    # print('(toConfidenceMatrix) balance while masking? {}, balance class weights? {}'.format(tBalanceClassWhileMasking, tBalanceClassWeights))

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
    if is_cascade: # then X must be [R|T]
        if isinstance(X, (tuple, list)): 
            assert isinstance(L, (tuple, list))
            R, T = X
            L_train, L_test_est = L
        else: 
            assert n_train > 0, f"(toConfidenceMatrix) No way of knowing how to split X into R and T without knowing size of R (given n_train {n_train})"
            assert len(L) == X.shape[1]

            R, T = X[:,:n_train], X[:,n_train:]
            L_train, L_test_est = L[:n_train], L[n_train: ] # note that `L_test_est` is only a guess since we do not know test set's true labels
    
    #####################################################################################

    # conditions with p_threshold 
    #   1) p_threshold given externally 
    #   2) p_threshold is to be inferred from training data message (X_train, L_train); used for computing confidnece scores for the test split 
    #   3) p_threshold is to be esimated via 'prior'; L must be given
    #   
    #   4) p_threshold could be estimated in a unsupervised way even without L, but not recommended
    policy_threshold = kargs.get('policy_threshold', 'fmax')
    p_threshold = kargs.get('p_threshold', [])  # depends on policy_threshold: 'fmax', float, 'unsupervised' 

    # nickname 
    # X_train, X_test = R, T

    # [design] Always pre-compute probability thresholds so that this block can be skipped
    #####################################################################################
    if len(p_threshold) == 0: 
        if tHasMessageFromTrainingSplit: # estimate probabilty thresholdds
            # policy_threshold: prior, fmax, ...
            assert R is not None and L_train is not None

            print('(toConfidenceMatrix) message passing: use training set statistics to estimate proba thresholds | policy={p}'.format(p=kargs.get('policy_threshold', 'prior')))
            policy_threshold = kargs.get('policy_threshold', 'prior')
            p_threshold = estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=policy_threshold) # policy: any policy that utilizes L
        else: 
            assert len(L) > 0, "Both p_threshold and L are not given, and neither was training data message given!"
            policy_threshold = 'prior' # kargs.get('policy_threshold', 'prior') if len(L) > 0 else 'ratio' 
            p_threshold = estimateProbThresholds(X, L=L, pos_label=pos_label, policy=policy_threshold, ratio_small_class=kargs.get('ratio_small_class', 0.01)) # findOptimalCutoff(L_train, R, metric='fmax', beta=1.0, pos_label=1)
    ##################################################################################### 
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
    ##################################################################################### 
    # ... estimated labels ready L <- Lh
    # ... tEstimatedLabels is True when 'estimated_labels' passed externally is True or when L wasn't given
    
    # [design] L_test
    ##################################################################################### 
    # test accuracy (only useful when L is estimated)
    L_test = L_ext = kargs.get('L_test', [])
    if tEstimatedLabels and len(L_test) > 0: 
        accuracy = np.sum(lh == L_test) / (len(L_test)+0.0)
        div('(toConfidenceMatrix) Accuracy of estimated labels: {} | n(L_ext): {}'.format(accuracy, len(L_test)), symbol='#', border=2)
    ##################################################################################### 

    if verbose: 
        print(f'(toConfidenceMatrix) Computing conficence scores using conf_measure: {conf_measure}')
        print('...                   p_threshold? {tval} | message passing? {tm} | policy: {p}'.format(
                 tval=len(p_threshold) > 0, p=policy_threshold, tm=tHasMessageFromTrainingSplit))

    # Scoring would be ignored in mode: {'ratio', }
    C0 = confidence2D(X, L, mode=conf_measure, 
                scoring=kargs.get('scoring', brier_score_loss), 
                outer_product=False, 

                    # following params are used only when mode = 'ratio'
                    p_threshold=p_threshold,  
                    policy_threshold=kargs.get('policy_threshold', ''), 
                    ratio_small_class=kargs.get('ratio_small_class', 0.01), verbose=verbose)  # don't return outer(wu, Wi) 
    ################################################################# 
    # ... C0: raw confidence scores 
    Cui = np.zeros(C0.shape)+C0


    #################################################################
    Mc = None
    tConservative = True
    isPolarityMatrix = True

    Pc, Lh = color_matrix(X, L, p_threshold) # TP=2, TN=1, FP=-2, FN=-1, used for polarity model
    # Pc, Lh = polarity_matrix(X, L, p_threshold) # {TP, TN}: 1, {FP, FN}: -1

    #################################################################
    # Condition: either polarity matrix (Po) or color matrix (Pc) is determined

    # verify_confidence_matrix(Cui, X=X, L=L, p_threshold=p_threshold, Po=Pc, U=U, measure=conf_measure, message='(before) raw weights', test_cases=[])  # test_cases <- [] to use default

    # scaling C <- alpha * C 
    ####################################################
    alpha = kargs.get('alpha', 1.0)
    beta = kargs.get('beta', 1.0)  # increase the confidence weight further by this factor
    tDiscountTest = kargs.get('discount_test', False) # True to discount T's weights
    
    # [design] Balancing class weights is factored to cf.wmf-related routines (e.g. wmf_ensemble_iter, wmf_ensemble_iter2())
    ####################################################
    # ... Mc: (1: {TP, TN}, 0: {FP, FN})
    # if tBalanceClassWeights and not tBalanceClassWhileMasking: 
    #     # if test split: 
    #     #    a. unmask? 
    #     balance_and_scale(Cui, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, alpha=alpha, beta=beta, 
    #         suppress_max_class=tSuppressMaxClass, 
    #             discount_test=tDiscountTest, n_train=n_train, 
    #                 is_cascade=is_cascade, 
    #                     test_cases=test_cases, is_test_split=tEstimatedLabels)
    #     # ... alpha, is_test_split => eventually regulate_weights(), umask() will be merged in

    #     # up-regulate minority class weights and down-regulate majority class weights
    #     # regulate_weights(Cui, beta=beta, U=U, test_cases=test_cases, suppress_max_class=tSuppressMaxClass) 
    # ####################################################
    # verify_confidence_matrix(Cui, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, measure=conf_measure,
    #     message='(after) balanced + magnified (alpha={}, beta={}) | dtype: {}'.format(alpha, beta, 'test set' if tEstimatedLabels else 'training set'), 
    #         test_cases=test_cases, plot=True if fold == 1 else False, 
    #         test_weight_constraints=True)

    n_incorrect = np.sum(Pc < 0)
    n_uncertain = np.sum(Pc == 0)
    n_correct = np.sum(Pc > 0)
    n_colors = len(np.unique(Pc))
    assert n_colors >= 2, "Expecting colored particles but got {}".format(colors)

    # if not is_cascade: 
    #     print('... Colored polarity matrix (Pc) | n_negative: {}, n_neutral: {} n_positive: {} | data: {}'.format(n_incorrect, 
    #         n_uncertain, n_correct, 'test set' if tEstimatedLabels else 'training set'))

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
        print('(toConfidenceMatrix) Cui: alpha={}, shape(Cui)={}, n_zeros (uncertain)={} (>? 0) vs n_nonzeros={} (masked ratio={})'.format(alpha, 
                 Cui.shape, n_zeros, n_nonzeros, n_zeros/(Cui.shape[0] * Cui.shape[1] + 0.0)))
        
    # Cui = alpha * R 

    # C0: Raw confidence scores for all entries in X prior to filtering (unreliable entries)
    # Cui: masked confidence scores, where polarity neutral is 0
    # Po: polarity matrix ()
    # Pc: colored matrix (colored polarity matrix)
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

def predictByCorrWithLabels(T, R, labels, topk=None, canonicalize=True): 
    """
    Predict new items/data through the correlation between probability predictions and true labels. 

    Only applicable in the case of user/classifier vs true labels

    labels: training set labels

    Memo
    ----
    1. W is a 'global' property of the classifier based on the correlation between (probability) predictions 
       and true labels (this is different from the case in recommender system scenario)

    2. negatively correlated examples?

    """
    # confidence weights based on correlation between predictions and true labels
    W = confidence_corr(R, labels, mode='label') 

    # print('(predictNewItemsByCorr) weight distribution:\n%s\n' % W)  # [note] similar
 
    # only want those with positive correlations? 
    if topk: 
        topk_corr = np.argsort(W)[::-1][:topk]

        # only look at the rows that correspond to top k users/classifiers
        Th = W[topk_corr].dot(T[topk_corr, :]) / np.array([np.abs(W[topk_corr]).sum()])
    else: 
        Th = W.dot(T) / np.array([np.abs(W).sum()])

    if canonicalize: Th = canonicalize_prob(Th, name='Th')
    return Th
def predictNewItemsByCorr(T, R, labels, topk=None): 
    return predictByCorrWithLabels(T, R, labels, topk=topk)

def predictNewItems(R, T, topk=None, epsilon=1e-9): 
    """
    Similar to predict() but return only the predictions of T (test split)

    Memo
    ----
    1. Two choices to go about this 

    2. S, use mean center R or not? 
    """
    kind = 'item'
    test_offset = R.shape[1]
    Ra = np.hstack((R, T))

    Rc = demean(Ra, kind=kind) # Rc: mean centered R 
    S = evalSimilarity(Rc, kind=kind)  # mean-adjusted cosine similarity
    assert S.shape[0] == Rc.shape[1], "Similarity metrics (dim=%d) should take on the dimension of items: %d" % (S.shape[0], T.shape[1])
    
    if topk: 
        Rt = predict_topk(Ra, S, kind=kind, k=topk)  # users * (items_train + items_test)
    else: 
        Rt = predict_nobias(Ra, S, kind=kind) 

    Th = Rt[:, test_offset: ]
    assert Th.shape[0] == T.shape[0] and Th.shape[1] == T.shape[1]

    return Th

def predict(R, T, S=None, kind='user', topk=None, canonicalize=True, epsilon=1e-9):
    """

    args
        T: rating matrix for the test set 
        S: similarity matrix

    """ 
    # kind = kargs.get('kind', 'user')
    test_offset = R.shape[1]
    Ra = np.hstack((R, T))

    if S is None: # then use cosine similarity by default 
        Rc = demean(Ra, kind=kind) # Rc: mean centered R 
        S = evalSimilarity(Rc, kind=kind)  # mean-adjusted cosine similarity
    
    if kind.startswith('i'):     
        assert S.shape[0] == Ra.shape[1], "Similarity metrics (dim=%d) should take on the dimension of items: %d" % (S.shape[0], T.shape[1])
    else: 
        assert S.shape[0] == Ra.shape[0]
    
    if topk: 
        Rt = predict_topk(Ra, S, kind=kind, k=topk)  # users * (items_train + items_test)
    else: 
        Rt = predict_nobias(Ra, S, kind=kind) 

    Rh = Rt[:, :test_offset]
    if canonicalize: Rh = canonicalize_prob(Rh, name='Rh')  

    Th = Rt[:, test_offset: ]
    if canonicalize: Th = canonicalize_prob(Th, name='Th')  

    assert Rh.shape[1] == R.shape[1]
    assert Th.shape== T.shape, "dim(T):{0} but dim(Th):{1}".format(T.shape, Th.shape)

    return (Rh, Th)

def predict_nobias(ratings, similarity, kind='user'):
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
    import cluster.cluster as cluster

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
    import cluster.cluster as cluster
    # from scipy.spatial import distance   # todo: use angular distance
    if n_clusters == -1: 
        n_clusters = A.shape[1]  # dimension of the latent factor matrix (e.g. user matrix, item matrix)

    if kargs.get('verbose', True): print('(clustering) method: spectral, n_clusters: {0} | dim(A): {1}'.format(n_clusters, A.shape))
    S = toAffinity(A, sig=kargs.get('bandwith', 0.5)) # evalSimilarityByLatentFeatures(D) 
    return cluster.spectralCluster(X=S, n_clusters=n_clusters)  # return cluster label IDs (a numpy.ndarray)

def kmeansCluster(X, n_clusters=-1, **kargs): 
    import cluster.cluster as cluster
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
    import cluster.cluster as cluster

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
        return predict(R, T, S=similarity, kind=kind)

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

def demo(**kargs): 

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

def demean(A, kind='user'): 
    """

    Memo
    ----
    1. use outer product or numpy broadcasting 

       https://bic-berkeley.github.io/psych-214-fall-2016/subtract_means.html
    """
    # nu, ni = A.shape[0], A.shape[1]
    # broadcast_demeaned = np.zeros((nu, ni))

    if kind.startswith( ('u', 'r') ):  # user, row
        row_bias = row_means = np.mean(A, axis=1) 
        
        # row_means_col_vec = row_means.reshape((ratings.shape[0], 1))  # Better: np.newaxis
        # broadcast_demeaned = ratings - row_means_col_vec
        broadcast_demeaned = (A - row_bias[:, np.newaxis]).copy() # turns row_means into a column vector

        # [test] should be all 0s
        # assert sum(broadcast_demeaned.mean(axis=1)) < 1e-9
    else: 
        col_bias = A.mean(axis=0)
        broadcast_demeaned = (A - col_bias[np.newaxis, :]).copy()

        # assert sum(broadcast_demeaned.mean(axis=0)) < 1e-9

    return broadcast_demeaned

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

    return P, Q# 


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

def replace0(P, Q, X, test_offset=None, canonicalize=True, fill=0, predict_func=None, verify_=False):  
    Cui = R = None
    if len(X) == 2: 
        Cui, R = X 
    # elif len(X) == 3: 
    #     Cui, R, T = X
    else: 
        msg = "(replace) X must be a 2-tuple: (Cui, R): {0}".format(X)
        raise ValueError(msg)

    assert Cui.shape == R.shape, "Confidence or mask matrix must have the same dimensionality as the rating matrix (R or T)"
    if predict_func is None: 
        predict_func = predict_by_factors
    else: 
        assert hasattr(predict_func, '__call__')

    nF = P.shape[1]; assert nF == Q.shape[1]
    n_users, n_items = P.shape[0], Q.shape[0]

    Rh = predict_func(P, Q, canonicalize=True)
    is_augmented = False if Th is None else True

    if not is_augmented: # only R is reconstructed 
        assert n_items == test_offset
        n_masked = np.sum(Cui==fill)
        assert Rh.shape == R.shape, "dim(R):{0} <> dim(Rh):{1}".format(R.shape, Rh.shape)

        # Cui has to be in dense form
        if not isinstance(Cui, np.ndarray): 
            # Cui is in sparse format
            Rh = np.where(np.array(Cui.todense())==fill, Rh, R)  # use Rh in place of R where the condition holds (i.e. wherever Cui == fill, typically referencing 'bad probabilities')
        else: 
            Rh = np.where(Cui==fill, Rh, R) 
        Th = None
    else:  # Cui ~ R+T
        assert Th is not None
        assert Cui.shape[1] == Rh.shape[1]+Th.shape[1]

        nrR = nrT = 0
        R0, T0 = R[:, :test_offset], R[:, test_offset:]  # first portion (by selected columns) goes to R, the rest goes to T
        Cui_R, Cui_T = Cui[:, :test_offset], Cui[:, test_offset:]

        assert Rh.shape == R0.shape
        if not isinstance(Cui_R, np.ndarray): 
            # Cui_R is in sparse format
            Rh = np.where(np.array(Cui_R.todense())==fill, Rh, R0) # where condition holds (i.e. confidence scores are zeros), use Rh's values in replace of R0's value
            nrR = np.sum(np.array(Cui_R.todense())==fill)
        else: 
            Rh = np.where(Cui_R==fill, Rh, R0) 
            nrR = np.sum(Cui_R==fill)  # number of entries replaced

        assert Th.shape == T0.shape
        if not isinstance(Cui_T, np.ndarray):
            # Cui_T is in sparse format
            Th = np.where(np.array(Cui_T.todense())==fill, Th, T0)  # fill T0's entries where confidence scores are zeros
            nrT = np.sum(np.array(Cui_T.todense())==fill)
        else:  
            Th = np.where(Cui_T==fill, Th, T0)
            nrT = np.sum(Cui_T==fill)
        
        n_masked = nrR + nrT

    print('(verify) Found %d null markers (out of %d, ratio: %f)' % (n_masked, Cui.shape[0]*Cui.shape[1], n_masked/(Cui.shape[0]*Cui.shape[1]+0.0) ))

    return (Rh, Th)

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
    x2: new rating matrix
    W1: probability filter (aka preference matrix)

        if pref[i,j] == 1, then use X1[i, j]
        if pref[i,j] == 0, then use X2[i, j]; effectively replacing X1[i,j] by X2[i, j]

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

def split_item_vectors(Q, test_offset): 
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
    Qr, Qt = split_item_vectors(Q, test_offset)

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

def predict_by_preference0(P, Q, test_offset, canonicalize=True):

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
    ### 
    R_pref = T_pref = None

    R_pref = np.dot(P, Qr.T)  
    if canonicalize: R_pref = canonicalize_pref(R_pref, name='R_pref')

    if Qt is not None: 
        T_pref = np.dot(P, Qt.T)   # [todo] predict interface
        if canonicalize: T_pref = canonicalize_pref(T_pref, name='T_pref')

    return (R_pref, T_pref)

def predict_by_factors(P, Q, canonicalize=True, name='X'):
    Xh = np.dot(P, Q.T)
    # Rh = np.array(Rh.todense())
    print("(predict_by_factors) type(P): {}, type(Xh): {}".format(type(P), type(Xh)))
    if not isinstance(Xh, np.ndarray): 
        Xh = np.array(Xh.todense())
    if canonicalize: Xh = canonicalize_prob(Xh, name=name)
  
    return Xh    

def error_analysis(Xpf, Mc, Lh, verbose=True, message=''):
    scores = np.unique(Xpf)
    if len(scores) > 2: binarize = True

    # pref_threshold = p_th
    # if binarize: 
    #     if p_th < 0: print('(ratio_of_alignment) Warning: Attempting to binarize preference scores without a threshold (will use the grand mean as the threshold) ...') 
    #     # ... alternatively use calibrate_preference() to find the optimal threshold
    #     Xpf = binarize_pref(Xpf, p_th=pref_threshold, cutoff=True)
    assert len(np.unique(Xpf)) <= 2, "Xpf has not been binarized (n(values): {})".format(len(np.unique(Xpf)))
    # ... Xpf is a binary matrix

    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos
    cells_tn = (Mc == 1) & predict_neg
    # cells_fp = (Mc == 0) & predict_pos
    # cells_fn = (Mc == 0) & predict_neg

    # aligned = (Xpf == Mc)
    # n_aligned_tp = np.sum(aligned & cells_tp)
    # n_aligned_fp = np.sum(aligned & cells_fp)
    # n_aligned_fn = np.sum(aligned & cells_fn)

    # precision_aligned = n_aligned_tp/(n_aligned_tp+n_aligned_fp+0.0)
    # recall_aligned = n_aligned_tp/(n_aligned_tp+n_aligned_fn+0.0)

    missed = (Xpf == 0)
    n_missed_tp = np.sum(missed & cells_tp)
    n_missed_tn = np.sum(missed & cells_tn)

    return (n_missing_tp, n_missed_tn)

def ratio_of_alignment2(Xpf, Mc, Lh, verify=True, verbose=True, message=''):
    # scores = np.unique(Xpf)
    # if len(scores) > 2: binarize = True

    # pref_threshold = p_th
    # if binarize: 
    #     if p_th < 0: print('(ratio_of_alignment) Warning: Attempting to binarize preference scores without a threshold (will use the grand mean as the threshold) ...') 
    #     # ... alternatively use calibrate_preference() to find the optimal threshold
    #     Xpf = binarize_pref(Xpf, p_th=pref_threshold, cutoff=True)
    # ... Xpf is a binary matrix
    assert len(np.unique(Xpf)) <= 2, "Xpf has not been binarized (n(values): {})".format(len(np.unique(Xpf)))

    ret = {}
    p_agreed, p_correct_agreed  = ratio_of_alignment(Xpf, Mc)   # rc: overall considerig both 0 and 1, rc_correct: consider only correct predictions (i.e. TP & TN)
    
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    # nct = np.sum( (Xpf == Mc) & predict_pos )  # preference matrix aligned with correctnes matrix AND entries matching target labels
    # rtp = nct/(N+0.0) 

    # A. active preference
    cells_positive = cells_preferred = (Xpf == 1)
    cells_negative = (Xpf == 0)

    # cells_not_preferred = (Xpf == 0)
    n_pref = np.sum(cells_preferred)
    # n_not_pref = np.sum(cells_not_preferred)

    cells_tp = (Mc == 1) & predict_pos
    cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    cells_fn = (Mc == 0) & predict_neg

    # n_agreed_tp = np.sum( (Xpf == Mc) & cells_tp )  # agreed & tp
    # n_agreed_fp = np.sum( (Xpf == Mc) & cells_fp )  # agreed & tn
    
    # cells_preferred_tp = cells_preferred & cells_tp
    # cells_preferred_tn = cells_preferred & cells_tn

    n_tp_hit = n_tp_pref = np.sum( cells_preferred & cells_tp ) # correctly aligned TPs
    n_tn_hit = n_tn_pref = np.sum( cells_preferred & cells_tn )
    p_tp_preferred = n_tp_pref/(n_pref+1e-3)  # P(tp|preferred): the fraction of TPs among those preferred entries
    p_tn_preferred = n_tn_pref/(n_pref+1e-3)  # P(tn|preferred): the fraction of TNs among those preferred entries

    n_fp_hit = n_fp_pref = np.sum( cells_preferred & cells_fp )
    n_fn_hit = n_fn_pref = np.sum( cells_preferred & cells_fn )
    p_fp_preferred = n_fp_pref/(n_pref+1e-3)
    p_fn_preferred = n_fn_pref/(n_pref+1e-3)

    ret['precision_pref'] = precision_pref = n_tp_pref/(n_tp_pref+n_fp_pref+1e-3)
    ret['recall_pref'] = recall_pref = n_tp_pref/(n_tp_pref+n_fn_pref+1e-3)
    ret['npv_pref'] = npv_pref = n_tn_pref/(n_tn_pref+n_fn_pref+1e-3)
    ret['specificity_pref'] = specificity_pref = n_tn_pref/(n_tn_pref+n_fp_pref+1e-3)

    # B. aligned
    aligned = (Xpf == Mc)
    n_aligned = np.sum(aligned)
    n_aligned_tp = np.sum(aligned & cells_tp)
    n_aligned_tn = np.sum(aligned & cells_tn)
    n_aligned_fp = np.sum(aligned & cells_fp)
    n_aligned_fn = np.sum(aligned & cells_fn)

    precision_aligned = n_aligned_tp/(n_aligned_tp+n_aligned_fp+1e-3)
    recall_aligned = n_aligned_tp/(n_aligned_tp+n_aligned_fn+1e-3)

    # C. missed 
    missed = not_pref = (Xpf == 0)
    n_tp_missed = np.sum(missed & cells_tp)   # want small
    n_tn_missed = np.sum(missed & cells_tn)   # want small but don't care as much
    n_fp_missed = np.sum(missed & cells_fp)   # want large
    n_fn_missed = np.sum(missed & cells_fn)

    # if polarity = 1 corresponds to tp or tn, then it's a TP for polarity
    n_tp_polarity = np.sum( cells_positive & (cells_tp | cells_tn) )
    # if polarity = 1 ~              fp or fn, then it's a FP for polarity 
    n_fp_polarity = np.sum( cells_positive & (cells_fp | cells_fn) )
    # if polarity = 0 ~              fp or fn, then it's a TN for polarity 
    n_tn_polarity = np.sum( cells_negative & (cells_fp | cells_fn) )
    # if poarlity = 0 ~              tp or tn, then it's a FN for polarity
    n_fn_polarity = np.sum( cells_negative & (cells_tp | cells_tn) )

    ret['precision_polarity'] = n_tp_polarity/(n_tp_polarity+n_fp_polarity+1e-3)
    ret['recall_polarity'] = n_tp_polarity/(n_tp_polarity+n_fn_polarity+1e-3)
    ret['npv_polarity'] = n_tn_polarity/(n_tn_polarity+n_fn_polarity+1e-3)
    ret['specificity_polarity'] = n_tn_polarity/(n_tn_polarity+n_fp_polarity+1e-3)
    
    if verbose: 
        # if message: print(message)
        msg = '' if not message else '(ratio_of_alignment2) -- {} --\n'.format(message)
        msg += '(ratio_of_alignment2) P(TP|pref): {}, P(FP|pref): {}\n'.format(p_tp_preferred, p_fp_preferred)
        msg += "...  precison(pref): {}, recall(pref): {} | npv(pref): {}, specificity(pref): {}\n".format(
            precision_pref, recall_pref, npv_pref, specificity_pref)
        msg += "...  p-precision: {}, p-recall: {}, p-npv: {}, p-specificity: {}\n".format(
            ret['precision_polarity'], ret['recall_polarity'], ret['npv_polarity'], ret['specificity_polarity'])
        # precision(aligned): {}, recall(aligned): {}
        # precision_aligned, recall_aligned

        msg += '... n_missed(TP): {}, n_missed(TN):  {}\n'.format(n_tp_missed, n_tn_missed)
        msg += '... n_missed(FP): {}, n_missed(FN):  {}\n'.format(n_fp_missed, n_fn_missed)
        msg += '... n_hit(TP):    {}, n_hit(TN):     {}\n'.format(n_tp_hit, n_tn_hit)
        msg += '... n_hit(FP):    {}, n_hit(FN):     {}\n'.format(n_fp_hit, n_fn_hit)
        msg += '==> n_hit(TP):    {}, n_missed(FP):  {} -> Large?\n'.format(n_tp_hit, n_fp_missed)
        msg += '==> n_hit(FP):    {}, n_missed(TP):  {} -> Small?\n'.format(n_fp_hit, n_tp_missed)
        # msg += '... n_aligned(TP) {}, n_aligned(TN): {}\n'.format(n_aligned_tp, n_aligned_tn)
        # msg += '... n_aligned(FP) {}, n_aligned(FN): {}\n'.format(n_aligned_fp, n_aligned_fn)
        msg += '... n_pref:       {}, n_aligned:     {}\n'.format(n_pref, n_aligned)   # n_aligned >? n_pref
        msg += '-' * 80; msg += '\n'
        # msg += '... P(agreed): {}, P(correct|agreed): {}\n'.format(p_agreed, p_correct_agreed)
        # msg += '... P(TN|pref):{}, P(FN|pref): {}\n'.format(p_tn_preferred, p_fn_preferred)

        # fraction of entries corresponding to preferred ...
        n_zeros, n_ones = np.sum(Xpf==0), np.sum(Xpf==1)
        n_incorrect, n_correct = np.sum(Mc==0), np.sum(Mc==1)
        N = Xpf.shape[0] * Xpf.shape[1] + 0.0
        msg += '-' * 80; msg += '\n'
        msg += '... number of 0s and 1s | n(zeros):     {} (r={}) vs n(ones):    {} (r={})\n'.format(n_zeros, n_zeros/N, n_ones, n_ones/N)
        msg += '... number of (+)s, (-)s| n(incorrect): {} (r={}) vs n(correct): {} (r={})\n'.format(n_incorrect, n_incorrect/(n_incorrect+n_correct+0.0), n_correct, n_correct/(n_correct+n_incorrect+0.0) )
        print(msg)

    return ret

def ratio_of_alignment(Xpf, Mc, verify=True, target_label=None):
    # scores = np.unique(Xpf)
    # if len(scores) > 2: binarize = True

    # if binarize: 
    #     if p_th < 0: print('(ratio_of_alignment) Warning: Attempting to binarize preference scores without a threshold (will use the grand mean as the threshold) ...') 
    #     Xpf = binarize_pref(Xpf, p_th=p_th, cutoff=True)
    assert len(np.unique(Xpf)) <= 2, "Xpf has not been binarized (n(values): {})".format(len(np.unique(Xpf)))
    # ... Xpf is a binary matrix

    N = Xpf.shape[0] * Xpf.shape[1] # np.sum(Mc) # total correct predictions

    n_aligned = np.sum(Xpf == Mc)
    p_aligned = n_aligned/(N+0.0)  # the fraction of the entries for which preference matrix is aligned with the correct entries
    # ... hope: preferred is also correct

    # only consider correct entries 
    # Nc = np.sum(Mc == 1)
    n_correct_aligned = np.sum( (Mc == 1) & (Xpf == Mc) ) 
    p_correct_aligned = n_correct_aligned/(n_aligned+0.0)
    # ... assuming that Mc is computed from true labels

    # to focus on only positive examples or negative examples, we need the label matrix Lh
    # ... see ratio_of_alignment2()

    return p_aligned, p_correct_aligned  # rc: overall considerig both 0 and 1, rc_correct: consider only correct predictions (i.e. TP & TN)

def eval_alignment_by_precision(Xpf, Mc, Lh, by='alignment'):
    """
    Params
    ------
    policy: 
        'agreement': the fraction of entries in Xpf that is consistent with correctness matrix (Mc)
           objective: the preferred entries should those that correspond to correct predictions 

        'precision': TP/(TP+FP) computed from preferred entries (Xpf == 1)
    """
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos
    # cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    # cells_fn = (Mc == 0) & predict_neg

    precision = 0.0
    if by == 'alignment':
        aligned = (Xpf == Mc)
        # n_aligned = np.sum(aligned)  # computed in ratio_of_alignment2()

        n_aligned_tp = np.sum(aligned & cells_tp)
        n_aligned_fp = np.sum(aligned & cells_fp)
        # n_aligned_fn = np.sum(aligned & cells_fn)

        precision= n_aligned_tp/(n_aligned_tp+n_aligned_fp+0.0)
        # recall_aligned = n_aligned_tp/(n_aligned_tp+n_aligned_fn+0.0)
    elif by == 'preference': 
        preferred = (Xpf == 1)
        # n_pref = np.sum(preferred)

        n_pref_tp = np.sum( preferred & cells_tp ) # correctly aligned TPs
        # n_pref_tn = np.sum( preferred & cells_tn )
        
        n_pref_fp = np.sum( preferred & cells_fp )
        # n_pref_fn = np.sum( preferred & cells_fn ) 
        precision = n_pref_fp/(n_pref_tp+n_pref_fp+0.0)
    else: 
        raise NotImplementedError

    return precision

def eval_alignment_by_recall(Xpf, Mc, Lh, by='alignment'): 
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos
    # cells_tn = (Mc == 1) & predict_neg
    # cells_fp = (Mc == 0) & predict_pos
    cells_fn = (Mc == 0) & predict_neg

    recall = 0.0 
    if by == 'alignment':
        aligned = (Xpf == Mc)
        # n_aligned = np.sum(aligned)  # computed in ratio_of_alignment2()

        n_aligned_tp = np.sum(aligned & cells_tp)
        # n_aligned_fp = np.sum(aligned & cells_fp)
        n_aligned_fn = np.sum(aligned & cells_fn)

        # precision= n_aligned_tp/(n_aligned_tp+n_aligned_fp+0.0)
        recall = n_aligned_tp/(n_aligned_tp+n_aligned_fn+0.0)
    elif by == 'preference': 
        preferred = (Xpf == 1)
        # n_pref = np.sum(cells_preferred)
        n_tp_pref = np.sum( preferred & cells_tp ) # correctly aligned TPs
        # n_tn_pref = np.sum( cells_preferred & cells_tn )

        # n_fp_pref = np.sum( cells_preferred & cells_fp )
        n_fn_pref = np.sum( preferred & cells_fn )

        recall = n_tp_pref/(n_tp_pref+n_fn_pref+0.0)
    else: 
        raise NotImplementedError

    return recall

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
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    # Need to consider the fact that Mc is derived from estimated labels (e.g. majority vote)
    cells_tp = (Mc == 1) & predict_pos
    cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos   # want small
    cells_fn = (Mc == 0) & predict_neg   # want small

    precision = recall = 0.5
    f_beta = f_beta_bar = 0.5
    if by == 'alignment': 
        raise ValueError

    elif by == 'preference': 

        cells_positive = preferred = (Xpf == 1)
        cells_negative = preferred = (Xpf == 0)

        ret = {}

        # n_tp_hit = n_tp_pref = np.sum( cells_positive & cells_tp ) # correctly aligned TPs
        # n_tn_hit = n_tn_pref = np.sum( cells_positive & cells_tn )
        # n_fp_hit = n_fp_pref = np.sum( cells_positive & cells_fp )
        # n_fn_hit = n_fn_pref = np.sum( cells_positive & cells_fn )

        # ret['precision_pref'] = precision_pref = n_tp_pref/(n_tp_pref+n_fp_pref+1e-3)
        # ret['recall_pref'] = recall_pref = n_tp_pref/(n_tp_pref+n_fn_pref+1e-3)
        # ret['npv_pref'] = npv_pref = n_tn_pref/(n_tn_pref+n_fn_pref+1e-3)
        # ret['specificity_pref'] = specificity_pref = n_tn_pref/(n_tn_pref+n_fp_pref+1e-3)

        # combined measure 
        # n_hit = n_tp_hit + n_tn_hit
        # n_miss = n_fp_hit + n_fn_hit
        # accuracy = n_hit/(n_hit+n_miss)

        # if polarity = 1 corresponds to tp or tn, then it's a TP for polarity
        n_tp_polarity = np.sum( cells_positive & (cells_tp | cells_tn) )
        # if polarity = 1 ~              fp or fn, then it's a FP for polarity 
        n_fp_polarity = np.sum( cells_positive & (cells_fp | cells_fn) )
        # if polarity = 0 ~              fp or fn, then it's a TN for polarity 
        n_tn_polarity = np.sum( cells_negative & (cells_fp | cells_fn) )
        # if poarlity = 0 ~              tp or tn, then it's a FN for polarity
        n_fn_polarity = np.sum( cells_negative & (cells_tp | cells_tn) )

        ret['precision_polarity'] = n_tp_polarity/(n_tp_polarity+n_fp_polarity+1e-3)
        ret['recall_polarity'] = n_tp_polarity/(n_tp_polarity+n_fn_polarity+1e-3)
        ret['npv_polarity'] = n_tn_polarity/(n_tn_polarity+n_fn_polarity+1e-3)
        ret['specificity_polarity'] = n_tn_polarity/(n_tn_polarity+n_fp_polarity+1e-3)

        ########################################################
        precision = n_tp_polarity/(n_tp_polarity+n_fp_polarity+1e-3)
        recall = n_tp_polarity/(n_tp_polarity+n_fn_polarity+1e-3)
        ########################################################

        # n_tp_not_pref = np.sum( not_preferred & cells_tp ) # correctly aligned TPs
        # n_fp_not_pref = np.sum( not_preferred & cells_fp )
        # n_fn_not_pref = np.sum( not_preferred & cells_fn )

        # precision_bar = n_tp_not_pref/(n_tp_not_pref+n_fp_not_pref+1e-3)
        # recall_bar = n_tp_not_pref/(n_tp_not_pref+n_fn_not_pref+1e-3)

    else: 
        raise NotImplementedError
    
    # polarity (+): aligned or preferred
    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # polarity (-): 
    # f_beta_bar = (1 + beta**2) * (precision_bar * recall_bar) / ((beta**2 * precision_bar) + recall_bar)
    # f1: beta -> 1 
    # if beta -> 0, f_bete -> precision
    # if beta -> inf, f_beta -> recall
   
    return f_beta # f_beta/(f_beta_bar+1.0)

def eval_alignment_minimize_fpfn(Xpf, Mc, Lh):
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    ########################################
    cells_tp = (Mc == 1) & predict_pos
    # cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos   # want small
    cells_fn = (Mc == 0) & predict_neg   # want small
    ########################################
    # ... estimated quantities due to Mc being estimated

    preferred = (Xpf == 1)
    # n_pref = np.sum(preferred)

    n_tp_pref = np.sum( preferred & cells_tp ) # correctly aligned TPs
    # n_tn_pref = np.sum( cells_preferred & cells_tn )

    n_fp_pref = np.sum( preferred & cells_fp )
    n_fn_pref = np.sum( preferred & cells_fn )

    nf = n_fp_pref + n_fn_pref
    # precision = n_tp_pref/(n_tp_pref+n_fp_pref+0.0)
    # recall = n_tp_pref/(n_tp_pref+n_fn_pref+0.0)

    # return -(n_fp_pref+n_fn_pref)   # '-' because we want n_f* to be as small as possible
    score = 0 if nf == 0 else np.log(1./nf) 
    return score

def eval_alignment_hit_to_miss_ratio2(Xpf, Mc, Lh, error_avoidance=False): 
    """


    Memo
    ----
    1. examples
        ==> n_hit(TP): 1101, n_missed(FP): 6885 → Large?
        ==> n_hit(FP): 3880, n_missed(TP): 511 → Small?
    """

    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    # predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos   # estimated
    # cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    # cells_fn = (Mc == 0) & predict_neg

    pref = (Xpf == 1)
    n_tp_hit = np.sum( pref & cells_tp ) # correctly aligned TPs, want high
    n_fp_hit = np.sum( pref & cells_fp ) # want small
    # n_tn_hit = n_tn_pref = np.sum( cells_preferred & cells_tn )

    pref_bar = (Xpf == 0)
    n_tp_missed = np.sum(pref_bar & cells_tp)  # want small 
    n_fp_missed = np.sum(pref_bar & cells_fp)  # want high

    if error_avoidance: 
        # only consider FP statistics (in a larger-is-better fashion)
        return np.log((n_fp_missed+1.0)/(n_fp_hit+1.0))
   
    # want stats(FP) to be small
    # fp_stats = np.log( (n_fp_hit+1.0)/(n_fp_missed+1.0) )  # lower n_hit(fp) but higher n_missed(fp)

    # want stats(TP) to be large
    # tp_stats = np.log( (n_tp_hit+1.0)/(n_tp_missed+1.0) )  # higher n_hit(tp) but lower n_missed(tp)
    return  np.log( (n_tp_hit * n_fp_missed + 1.0) / (n_tp_missed * n_fp_hit + 1.0 ) )

def eval_alignment_hit_to_miss_ratio(Xpf, Mc, Lh, conditioned=True):
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    # predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos   # estimated
    # cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    # cells_fn = (Mc == 0) & predict_neg

    pref = (Xpf == 1)
    n_tp_hit = np.sum( pref & cells_tp ) # correctly aligned TPs, want high
    n_fp_hit = np.sum( pref & cells_fp ) # want small
    # n_tn_hit = n_tn_pref = np.sum( cells_preferred & cells_tn )

    pref_bar = (Xpf == 0)
    n_tp_missed = np.sum(pref_bar & cells_tp)  # want small 
    n_fp_missed = np.sum(pref_bar & cells_fp)  # want high

    if conditioned: 
        n_pref = np.sum(Xpf == 1)
        n_pref_bar = np.sum(Xpf == 0)
        if n_pref >= n_pref_bar: 
            # focus on reducing false preference => increasing true non-preference
            return np.log( (n_fp_missed+1.0)/(n_tp_missed+1.0))
        else: # n_pref < n_pref_bar    n(1s) < n(0s)
            # focus on reducing false non-preference => increasing true preference
            return np.log( (n_tp_hit+1.0)/(n_fp_hit+1.0))
        # ... thought: instead of adding 1, could also add a prior count estimated from the training split (R)
        #              suppose that we also use the same method to 'estimate' training set's label 
        #              (e.g. via majority vote), then the related counts can be used as priors

    return np.log( (n_tp_hit+n_fp_missed+1.0)/(n_fp_hit+n_tp_missed+1.0) )  # ~ np.log(n_tp_hit * n_fp_missed)

def eval_alignment_hit_to_miss_ratio0(Xpf, Mc, Lh):
    """

    Memo
    ----
    1. Tends to lead to very low pref threshold, not recommended
    """
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    # predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos
    # cells_tn = (Mc == 1) & predict_neg
    # cells_fp = (Mc == 0) & predict_pos
    # cells_fn = (Mc == 0) & predict_neg

    cells_preferred = (Xpf == 1)
    n_tp_hit = np.sum( cells_preferred & cells_tp ) # correctly aligned TPs
    # n_tn_hit = n_tn_pref = np.sum( cells_preferred & cells_tn )

    not_pref = (Xpf == 0)
    n_tp_missed = np.sum(not_pref & cells_tp) 
    return n_tp_hit/(n_tp_missed+1.0)

# def eval_alignment_true_positive(Xpf, Mc, Lh): 
#     cells_tp = (Mc == 1) & (Lh == 1)  # but this is not necessarily good, because "true positive" is defined wrt estiamted labels (Lh)
#     return np.sum( (Xpf == Mc) & cells_tp)
def eval_alignment_positive(Xpf, Mc, Lh, verbose=False): 
    # Xpf: is a binary matrix

    # objective: whenver predicting positive, want to be aligned with correctness  

    n_aligned = np.sum(Xpf == Mc)    
    n_aligned_positive = np.sum( n_aligned & (Lh == 1) )
    # cells_tp = (Mc == 1) & (Lh == 1)  # but this is not necessarily good, because "true positive" is defined wrt estiamted labels (Lh)

    return n_aligned_positive/n_aligned  # P(positive|aligned)

def eval_alignment(Xpf, Mc, Lh=None, conditioned=True): 
    # Xpf: is a binary matrix

    aligned = (Xpf == Mc)
    n_aligned = np.sum(aligned)

    if conditioned:    
        n_pref_aligned = np.sum( aligned & (Xpf == 1) ) 
        return n_pref_aligned/(n_aligned+0.0)  # P(preferred | aligned)
    
    N = Xpf.shape[0] * Xpf.shape[1] + 0.0
    return n_aligned/N

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

def estimate_polarity0(R, Lr, p_th, T, policy='median', 
        constrained=True, stochastic=False,
        k_upper=-1, k_lower=-1, k_max=-1, k_min=-1, verbose=True):
    def is_within(v, scope): 
        if not scope: return False
        return (v >= scope['min']) and (v <= scope['max'])
    def is_within_or_above(v, scope): 
        if not scope: return False
        return ((v >= scope['min']) and (v <= scope['max'])) or (v > scope['max'])
    def is_within_or_below(v, scope): 
        if not scope: return False
        return ((v >= scope['min']) and (v <= scope['max'])) or (v < scope['min'])
    def is_above(v, scope, k='mean'):
        if not scope: return False
        return v >= scope[k]
    def is_below(v, scope, k='mean'):
        if not scope: return False
        return v <= scope[k]  

    Mc, Lh = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
    
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos   # estimated
    cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    cells_fn = (Mc == 0) & predict_neg

    # estimate proba range for each classifier based on sample type (TP, TN, FP, FN)
    sample_types = Polarity.sample_types
    codes = Polarity.codes 

    scopes = {st: {} for st in sample_types}   # scope['tp'][0]: to be true positive, 0th classifier must have this proba range
    for i in range(R.shape[0]):  # foreach classifier
        scopes['tp'][i] = {}
        
        # TPs
        v = R[i, :][cells_tp[i, :]]
        if len(v) > 0: 
            scopes['tp'][i] = {'min': np.min(v), 'max': np.max(v), 'mean': np.mean(v), 'median': np.median(v)}   # min, max, mean, median

        # TNs 
        v2 = R[i, :][cells_tn[i, :]]
        if len(v2) > 0: 
            scopes['tn'][i] = {'min': np.min(v2), 'max': np.max(v2), 'mean': np.mean(v2), 'median': np.median(v2)}   
        
        # ... positive polarity candidates 
        assert scopes['tp'][i]['median'] != scopes['tn'][i]['median'] 

        # FPs ~ TPs
        v3 = R[i, :][cells_fp[i, :]]
        if len(v3) > 0: 
            scopes['fp'][i] = {'min': np.min(v3), 'max': np.max(v3), 'mean': np.mean(v3), 'median': np.median(v3)}

        # FNs ~ TNs
        v4 = R[i, :][cells_fn[i, :]]
        if len(v4) > 0: 
            scopes['fn'][i] = {'min': np.min(v4), 'max': np.max(v4), 'mean': np.mean(v4), 'median': np.median(v4)}   
        # ... negative polarity candidates
    
    # [test]
    msg = '(estimate_polarity) Policy: {}\n'.format(policy)
    for st in sample_types: 
        msg += '\n--- Sample Type: {} ---\n'.format(st.upper())
        for i in range(R.shape[0]):  # foreach classifier
            if i % 2 == 0: 
                if len(scopes[st][i]) > 0: 
                    msg += '... type: {} | min: {}, max: {}, mean: {}\n'.format(st.upper(), scopes[st][i]['min'], scopes[st][i]['max'], scopes[st][i]['mean'])
    print(msg)

    tConstrained = True if constrained and ((k_upper > 0) or (k_lower > 0)) else False
    # now scan through T while looking up table scope to determine if entries in T should be considered as positive or negative or neutral/unknown
    M = np.zeros(T.shape)
    if policy in ('mean', 'median', ): 

        for j in range(T.shape[1]):  # foreach item/datum
            for i in range(T.shape[0]):  # foreach user/classifier
                # positive? 
                is_tp = is_above(T[i, j], scopes['tp'][i], k=policy)    
                is_tn = is_below(T[i, j], scopes['tn'][i], k=policy)

                if is_tp != is_tn: # only one of them is true
                    M[i, j] = codes['tp'] if is_tp else codes['tn'] 

                elif not is_tp and not is_tn: 
                    M[i, j] = -1   # negative, either FP or FN depending on the label
                else: 
                    # both are true, then it's a neutral
                    # no-op 
                    # M[i, j] = 0
                    pass

    elif policy.startswith('interv'):  # interval 
        for j in range(T.shape[1]): 
            for i in range(T.shape[0]):

                # positive? 
                is_tp = is_within_or_above(T[i, j], scopes['tp'][i]) 
                is_tn = is_within_or_below(T[i, j], scopes['tn'][i]) 

                if is_tp != is_tn: # only one of them is true
                    M[i, j] = codes['tp'] if is_tp else codes['tn']

                elif not is_tp and not is_tn: 
                    M[i, j] = -1
                    

                else: 
                    # both are true, then it's a neutral
                    # no-op 
                    # M[i, j] = 0
                    pass
    else: 
        raise NotImplementedError
    # ... M encodes an estimate of sample types (e.g. TP, TN, negative, unknown)

    ### resolve sample types 
    # ... M eventually should only consists of 3 types of polarity: 0/neutral, 1/poistive, -1/negative
    #     positive => high confidence of being in {TP, TN}, negative: incorrect {FP, FN}, neutral: unknown 
    tStochastic = stochastic
    print('(estimate_polarity) Constrained? {}, Stochastic? {}'.format(constrained, stochastic))

    M2 = np.zeros(T.shape)
    if not tConstrained: 
        
        n_conflict_evidence = n_no_positive = 0
        n_tp_dominant = n_tn_dominant = 0

        for j in range(T.shape[1]):  # foreach item/datum
            tp_i = np.where(M[:, j] == codes['tp'])[0] 
            tn_i = np.where(M[:, j] == codes['tn'])[0]
            neg_i = np.where(M[:, j] == -1)[0]
            ntp_i, ntn_i = len(tp_i), len(tn_i)
            
            # k_upper_tp = k_upper_tn = 0
            if (ntp_i > 0) and (ntn_i > 0): 
                n_conflict_evidence += 1

            if ntp_i == 0 and ntn_i == 0: n_no_positive += 1  

            # T[:, j][M[:, j] == codes['tp']]
            ########################################################
            if tStochastic: 
                m_tp_dominant = ntp_i / (ntp_i + ntn_i+0.0)
                p = np.random.uniform(0, 1, 1)[0]
                is_tp_dominant = True if p <= m_tp_dominant else False
            else:
                # majority vote; deterministic 
                is_tp_dominant = True if ntp_i > ntn_i else False
            ########################################################

            if is_tp_dominant: 
                support_pos = tp_i 
                support_neg = neg_i

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1

                n_tp_dominant += 1
            else:  
                support_pos = tn_i
                support_neg = neg_i

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1
        
    else: # constrained
        # M2 = np.zeros(T.shape)

        n_conflict_evidence = n_no_positive = 0
        n_tp_dominant = n_tn_dominant = 0
        for j in range(T.shape[1]):  # foreach item/datum

            tp_i = np.where(M[:, j] == codes['tp'])[0] 
            tn_i = np.where(M[:, j] == codes['tn'])[0]
            neg_i = np.where(M[:, j] == -1)[0]
            ntp_i, ntn_i = len(tp_i), len(tn_i)
            
            # k_upper_tp = k_upper_tn = 0
            if (ntp_i > 0) and (ntn_i > 0): 
                n_conflict_evidence += 1
                # k_upper_tp = k_upper_tn = k_upper/2.0
                # k_lower_tp = k_lower_tn = k_lower/2.0
            # assert not ((ntp_i > 0) and (ntn_i) > 0), "Conflicting evidence in row {}, where n(tp): {}, n(tn): {}".format(i, ntp_i, ntn_i)
            # ... it's possible that some classfieris consider j as TPs while others consider it as TNs (simply because the value falls within the range)
            # ... which ones are more likely to be true?
            if ntp_i == 0 and ntn_i == 0: n_no_positive += 1  

            # is_tp_dominant? 
            ########################################################
            if tStochastic: 
                m_tp_dominant = ntp_i / (ntp_i + ntn_i+0.0)
                p = np.random.uniform(0, 1, 1)[0]
                is_tp_dominant = True if p <= m_tp_dominant else False
            else:
                # majority vote; deterministic 
                is_tp_dominant = True if ntp_i > ntn_i else False
            ########################################################

            # T[:, j][M[:, j] == codes['tp']]
            if is_tp_dominant: 
                # assert len(tp_i) > 0  
                rows = np.argsort(-T[:, j])  # np.argsort(R[:, j])[:-k-1: -1] # choose indices of k highest probs
                # ... high to low
                
                # 1. positive examples: the larger the better
                support_pos = [i for i in rows if i in tp_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tp_i][:k_upper]
                # ... if k_upper_tp > 0, then we have conflicting evidence

                # 2. tn examples are demoted to neutral

                # 3. negative examples remain negative (not TP nor TN)
                support_neg = [i for i in rows[::-1] if i in neg_i][:k_lower]

                # [test]
                # if j % 100 == 0: assert np.min(T[support_pos,j]) >= np.max(T[support_neg,j]), "min(pos): {} <? max(neg): {}".format(np.min(T[support_pos,j]), np.max(T[support_neg,j]))

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1

                n_tp_dominant += 1
            else:  
                rows = np.argsort(T[:, j])
                # ... low to high

                if len(tn_i) > 0: 
                    
                    # ... low to high

                    # 1. positive examples: the smaller the better
                    support_pos = [i for i in rows if i in tn_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tn_i][:k_upper]

                    # 2. tp examples are demoted to neutral

                    # 3. negative examples stay negative: the higher the better 
                    support_neg = [i for i in rows[::-1] if i in neg_i][:k_lower]
                    
                    n_tn_dominant += 1
                else: 
                    # tp_i == tn_i == 0
                    # then there's no way to tell just pick the smallest
                    support_pos = [i for i in rows if i in tn_i][:1]
                    support_neg = [i for i in rows[::-1] if i in neg_i][:k_lower]

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1
        ### ... end foreach item

    if verbose: 
        msg = ''
        r = n_conflict_evidence/(T.shape[1]+0.0)
        msg += "(estimate_polarity) tContrained: True | Found n_conflict_evidence: {}, n_no_positive: {} | N={}, ratio_conflict_evidence: {}\n".format(n_conflict_evidence, n_no_positive, T.shape[1], r)
        msg += "... n_tp_dominant: {}, n_tn_dominant: {}".format(n_tp_dominant, n_tn_dominant)
        msg += "... n(pos): {}, n(neg): {}, n(neutral): {}".format(np.sum(M2>0), np.sum(M2<0), np.sum(M2==0))
        print(msg)

    return M2

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

def standardize0(X, scaler=None):
    """
    X: numpy array, or 
       a list of (list of dictionaries) for sequence model (e.g. CRF)
       

    """
    from sklearn.preprocessing import MinMaxScaler

    if scaler is None: scaler = MinMaxScaler()

    X2 = []
    if isinstance(X[0][0], dict):  # X[0] is a list (of dictionaries)
        fset = X[0][0].keys()

        nf = len(fset)   # number of features (per dictionary)
        nfe = len(X[0])  # each feature dict ~ column has this many components (~ n_users)
        Xl = []
        for j, xj in enumerate(X):  # foreach list (of dict);  each list-of-dict corresponds to a feature repr for a column/item
            assert isinstance(xj, list)
            xv = []
            for d in xj:  # foreach dict
                assert isinstance(d, dict) 
                xvd = [d[f] for f in fset] 
                Xl.append(xvd)  # expand feature dict into a vector in the order of 'fset'

        Nt = len(Xl)
        assert Nt == len(X) * len(X[0])

        Xl = scaler.fit_transform(np.array(Xl)) 

        # convert back to list-of-dictionary format

        X2 = []
        for j in range(0, Nt, nfe):  # 0 ~ number of expanded instances with a step size of nfe
            xs = []
            for i in range(nfe):
                xs.append( dict(zip(fset, Xl[j+i])) )
            # ... xv: a sequence/list of dicts
            X2.append(xs)
        assert len(X2) == len(X)
        assert len(X2[0]) == len(X[0]) 
    else: 
        X2 = scaler.fit_transform(X)

    return X2, scaler
def transform0(X, scaler=None):
    # X
    #   i) a list of (list of dictioanries) i.e. x[i][j] is a feature dictionary
    #   i) a list of dictionary (correspoding to a column in R) i.e. x[i] is a feature dictionary

    # assuming that scaler has been fitted with data
    # x ~ feature sequence associated with a column
    if scaler is None: 
        # no-op
        return X

    X2 = []
    if isinstance(X[0][0], dict):  # X[0] is a list (of dictionaries)
        fset = X[0][0].keys()

        nf = len(fset)   # number of features (per dictionary)
        nfe = len(X[0])  # each feature dict ~ column has this many components (~ n_users)
        Xl = []
        for j, xj in enumerate(X):  # foreach list (of dict);  each list-of-dict corresponds to a feature repr for a column/item
            xv = []
            for d in xj:  # foreach dict 
                xvd = [d[f] for f in fset] 
                Xl.append(xvd)  # expand feature dict into a vector in the order of 'fset'

        Nt = len(Xl)
        assert Nt == len(X) * len(X[0])

        Xl = scaler.transform(np.array(Xl)) 

        # convert back to list-of-dictionary format

        X2 = []
        for j in range(0, Nt, nfe):  # 0 ~ number of expanded instances with a step size of nfe
            xs = []
            for i in range(nfe):
                xs.append( dict(zip(fset, Xl[j+i])) )
            # ... xv: a sequence/list of dicts
            X2.append(xs)
        assert len(X2) == len(X)
        assert len(X2[0]) == len(X[0]) 
    else: 
        X2 = scaler.transform(X)
    
    return X2

def get_feature_sequence(R, j, p_th, Rm=None, C=None, U=None, Lh=None, p_model={}, name='', index=0, verbose=False, wsize=20): 
    def get_vars(tags):
        fset = set()
        for i, entry in tags.items(): 
            fset.update( entry.keys() )
        return fset

    Nu, Ni = R.shape
    fdx = []
    fset = []
    tags = {i: {} for i in range(Nu)}

    # tagging first
    for i in range(Nu):
        get_seq_stats(R, i, j, p_th=p_th, Rm=Rm, C=C, U=U, p_model=p_model, name=name, 
            wsize=wsize, verbose=verbose, index=index, to_dict=True, tags=tags, tagging_only=True) 
    assert all([len(tag) > 0 for i, tag in tags.items()]), "All entries in tags should be non-empty: {}".format(tags)
    if verbose: print("(get_feature_sequence) Tagging complete. Feature set:\n{}\n".format(get_vars(tags)))

    for i in range(Nu):  # foreach user/classifier index while holding item index fixed 
        # one feature dictionary per entry
        fd = get_seq_stats(R, i, j, p_th=p_th, Rm=Rm, C=C, U=U, p_model=p_model, name=name, 
            wsize=wsize, verbose=verbose, index=index, to_dict=True, tags=tags) 

        # [test]
        if i >= Nu-2: 
            fset = list(fd.keys())
            if j in [0, 200, 400, 600, ] == 0: 
                print("(get_feature_sequence) name: {} | i={}, fd:\n... {}\n".format(name, i, fd))
        # fdv = get_vars_vstats(R, i, j, p_th=p_th, Rm=Rm, C=C, Lh=Lh, name=name, wsize=wsize, verbose=verbose, index=index, to_dict=True)
        # fd.update(fdv) # merge two dictionaries

        fdx.append(fd)

    # output: a list of feature dictionaries, one per entry/classifier/user while holding column(j) fixed
    return fdx  

def get_seq_stats(R, i, j, p_th, Rm=None, C=None, U=None, p_model={}, r_min=0.1, name='', 
        index=0, verbose=False, wsize=20, to_dict=False, neg_label=0, pos_label=1, 
        tags={}, tagging_only=False, include_chain=True):  
    """

    Params
    ------
    tags: external dictionary keep track of per-position (i) statistics

    Reference
    ---------
    1. crfsuite: 

        https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#api-reference

    """
    def base_name(cls_name, prefix='', sep='.'):
        bn = 0
        cn, *x = cls_name.split(sep)
        if prefix: cn = "{}-{}".format(prefix, cn) 
        if len(x) > 0: bn = x[0]
        return cn, bn
    def resolve_var(tags, i, prefix='user', exception=True):
        name = ''  
        for u, v in tags[i].items(): 
            if u.startswith(prefix):
                name = u
        if exception: assert len(name) > 0, "Did not find a user/classifier at index={} | tags[i]: {}".format(i, tags[i])

        # add time step to distingush from current user?
        return name
    def index_var(vn, i, prefix=''): 
        if prefix: 
            return '{}-{}({})'.format(prefix, vn, i)
        return '{}({})'.format(vn, i)
    def decorate_var(vn, descr='activated'):   # add description
        return "{}({})".format(vn, descr)

    # get BP prediction vector statistics as variables
    import scipy.stats as stats
    from scipy.stats import kurtosis, skew, ks_2samp
    import numpy as np

    # sample_types = ['tp', 'tn'] + ['fp', 'fn']
    # codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
    #         'unk': 0, 't': 3, 'f': -3}
    sample_types = Polarity.sample_types
    codes = Polarity.codes
    # neg_label, pos_label = 0, 1

    to_dict = False if tagging_only else True

    msg = ""

    fv = []  # features 
    fvn = []  # feature names
    ##################################
    # query point 
    i_cur =  i   # i-th base predictor
    q = pt_q = R[i, j]   # q
    ##################################
    Nu, Ni = R.shape

    # --- rating/probability ---
    
    # target variables: 
    #    1. bias
    #    2. value 
    #    3. user/classifier name
    vn = 'bias'
    fvn.append(vn)  
    fv.append(1.0)
    tags[i][vn] = 1.0

    vn = 'value' # index_var('value', i)
    # ... need to add user index (timestamp) to distinguish itself from the same feature but at different indices
    fvn.append(vn)
    fv.append(q)
    tags[i][vn] = q

    cur_user = 'n/a'
    if U is not None: 
        # vn = 'user'  # user/classifier
        vn, bnum = base_name(U[i])  # classifier, bag number
        cur_user = vn 

        # this is perhaps not needed because eventually we will consider all users/classifiers activated, which includes this
        # fvn.append(vn); fv.append(1.0)
        # tags[i][vn] = 1.0

    # ... filter outliers according to the quartile? (done prior to this call)

    # --- KDE signature ---
    # target variables 
    #   1. raw deltas of (1-CDF) (e.g. delta(tp-tn) as defined by vterms)
    #   2. raw deltas of ks.statistics 
    #   3. raw deltas of SE
    #   1a. indicator features derived from 1 
    #   2a. 
    #   3a. 

    vmin, vmax = -100, 100
    sv = []  # P(X>=x) for TP, TN, FP, FN
    tMaxPooling = False
    
    scopes = p_model   # polarity model derived from polarity_feature_extraction() (e.g. KDE models for the 4 sample types)
    vterms = ['tptn', 'fpfn' ] # + ['tpfn', 'fptn'] 
    hterms = ['tpfp', 'tnfn'] # + ['tpfn', 'tnfp', ]
    kde_v = ['tail', ]  
    kde_h = ['ks.statistic',  'se', 'kurtosis', 'skew',  ]  # ks.pvalue,  'se', 'kurtosis', 'skew', 'range'
    if 'kde' in scopes[sample_types[0]][0]: 

        diffs = {stype: {} for stype in sample_types}
        for stype in sample_types:
            vn = 'kde-{}'.format(stype) # variable name

            # has_value = True
            pm = 0.0
            if i in scopes[stype]:  # first need to check if a given sample type even exist
                    
                # ib = int(np.digitize(i, bins=bagging_bins))
                kde =  scopes[stype][i]['kde']  # kde_processed[stype][ib]
                # ... it's possible that i-th BP does not reference some flavors of the particle (e.g. no TPs ...)

                # compute P(X>=x) using survival function
                ii = np.digitize(pt_q, bins=kde.support)  # which interaval does the query point fall into
                if ii >= kde.support.size: ii = kde.support.size - 1   # max index

                # np.isnan(kde.sf).any()
                if not np.isnan(kde.sf[ii]): pm = kde.sf[ii]
            else: 
                msg += "(get_seq_stats) {}-th BP does not have sample type: {}\n".format(i, stype) # should rarely occur 
                # has_value = False

            diffs[stype]['tail'] = pm
            
            # if not tMaxPooling: 
            #     fv.append( pm )
            #     fvn.append( vn )
            sv.append( pm )
        ### end foreach flavor
        
        # collecting the vertical differentials
        #  tp  fp
        #  tn  fn 
        kde_v = ['tail', ] 
        vterms = ['tptn', 'fpfn' ] # + ['tpfn', 'fptn']   # term
        dx = kde_v # ['tail' , ]  # 'cde'
        delv = {t: {} for t in vterms}   
        for d in dx:  
            # vertical
            delv['tptn'] = diffs['tp'][d] - diffs['tn'][d]
            delv['fpfn'] = diffs['fp'][d] - diffs['fn'][d]

            # cross terms 
            delv['tpfn'] = diffs['tp'][d] - diffs['fn'][d]
            delv['fptn'] = diffs['fp'][d] - diffs['tn'][d]
            
            # a. raw features
            # [del_tptn, del_fpfn, del_tpfn, del_fptn ]
            for term in vterms: 
                vn = '{}-{}'.format(d, term)
                fvn.append(vn)  # add variable name
                fv.append(delv[term])  # add the value
                tags[i][vn] = delv[term]
            # ... del_fpfn is likely to be redundant as it's similar to del_tptn

            # b. inferred features
            vn = '{}:tp>=tn'.format(d)
            val = 1.0 if delv['tptn'] >= 0 else 0.0   # tail: TP > TN  i.e. more likely to be TP than TN if tp dominates tn in 1-CDF
            fvn.append(vn); fv.append(val)
            tags[i][vn] = val

            vn = '{}:fp>=fn'.format(d)
            val = 1.0 if delv['fpfn'] >= 0 else 0.0
            fvn.append(vn); fv.append(val)
            tags[i][vn] = val

            msg += "... Differential KS on {}-example | metric: {} | R[i,j]: {} | KS(tp-tn, fp-fn): {}, {}\n".format(name, 
                d, pt_q, delv['tptn'], delv['fpfn']) 

        # if verbose: print("(get_vars_hstats) KDE signature: {} | {}".format(fv, name))
      
        
        # ... other KDE signature HERE 
    ### end KDE section

    # --- local distribution (expensive; use nearest neighbors instead) ---
    # q = R[i, j] 
    N = R.shape[1]
    rk = -1   # rank of the query point
    
    wsize_min, wsize_max = N//100, wsize   # e.g. 20 
    wsize = min(wsize_max, max(wsize_min, wsize))

    sv = []  # reset sv
    if Rm is not None: 
        # ... Rm[i, :] must be sorted
        rk = np.searchsorted(Rm[i, :], pt_q, side='left')  # rank, q's position in R[i, :]
        k = wsize # min(min_size, N//100)  # no less than 50 points
     
        # search k nearest neighbor
        bl, br = rk-k, rk+k   

        # boundary conditions
        if bl < 0: bl = 0
        if br >= N: br = N-1 

        pts_lower = Rm[i, bl:rk]
        pts_higher = Rm[i, rk:br]
        assert np.all(pts_lower <= pt_q) and np.all(pts_higher >= pt_q), "query: {} | max(low): {}, min(high): {} | sorted? {}".format(pt_q, np.max(pts_lower), np.min(pts_higher), scipy.stats.rankdata(Rm[i, :][:20]))
        
        # SE wrt q
        pts_knn = np.hstack([pts_lower, pts_higher])
        
        se_local = np.std(pts_knn) # mean_squared_error(pts_knn, np.full_like(pts_knn, pt_q))
        range_local = stats.iqr(pts_knn) # np.max(pts_knn)-np.min(pts_knn)
        # fv.append(se_local)  # <<< 

        if verbose: 
            n = 10
            pts_knn_subset = np.hstack([pts_lower[:n], pts_higher[-n:]])
            msg += "(get_seq_stats) {} | local(SE): {} | R[i,j]: {}, rank: {} | pts_knn (subset:{}, total: {} vs N/100:{}): {}\n".format(name,
                se_local, pt_q, rk, len(pts_knn_subset), len(pts_knn), N//100, pts_knn_subset)

        # compare distribution with neighoring points of different flavors
        vars_stats = ['ks.statistic', 'ks.pvalue', 'median', 'skew', 'kurtosis', 'se', 'range', ]
        diffs = {stype: {v: 0.0 for v in vars_stats} for stype in sample_types}
        for stype in sample_types:
            kss = np.max(pts_knn-0.0)   # assumed to be very large (as large as max abs distance to zero vectors)
            if i in scopes[stype]: 
                pts = scopes[stype][i]['sample']
                # ... assuming that pts has been pre-sorted
                n = len(pts)

                # rank wrt to this particular sample type
                rk = np.searchsorted(pts, pt_q, side='left')

                # search k nearest neighbor
                bl, br = rk-k, rk+k

                # boundary conditions
                if bl < 0: bl = 0
                if br >= N: br = n-1 

                pts_lower = pts[bl:rk]
                pts_higher = pts[rk:br]
                assert np.all(pts_lower <= pt_q) and np.all(pts_higher >= pt_q), "query: {} wrt flavor: {} | max(low): {}, min(high): {}".format(pt_q, stype, np.max(pts_lower), np.min(pts_higher))
                
                pts_knn_flavored = np.hstack([pts_lower, pts_higher])
                range_flavored = stats.iqr(pts_knn_flavored) # np.max(pts_knn_flavored)-np.min(pts_knn_flavored)
                se_flavored = np.std(pts_knn_flavored)

                # K-S tes
                ks_stats = ks_2samp(pts_knn, pts_knn_flavored)
                kss = -10*np.log(ks_stats.pvalue) # ks_stats.statistic, -10*np.log(ks_stats.pvalue) 
                msg += "(get_vars_hstats) Local structure: {} | flavor: {} | R[i,j]: {}, rank: {} | KS statistic: {}\n".format(name, stype, pt_q, rk, kss)

                diffs[stype]['ks.statistic'] = ks_stats.statistic
                diffs[stype]['ks.pvalue'] = -10*np.log(ks_stats.pvalue) # ks_stats.pvalue 
                diffs[stype]['median'] = np.median(pts_knn_flavored)/(np.median(pts_knn)+1e-4)
                diffs[stype]['skew'] = skew(pts_knn_flavored)/(skew(pts_knn)+1e-4)
                diffs[stype]['kurtosis'] = kurtosis(pts_knn_flavored)/(kurtosis(pts_knn)+1e-4)
                diffs[stype]['se'] = se_flavored/(se_local+1e-4)
                diffs[stype]['range'] = range_flavored/(range_local+1e-4)
            else: 
                msg += "... !!!{}-th point in R does not have sample type: {}\n".format(i, stype)                

            # sv.append(kss) # ... use differential variables instead of the values themselves

        # horizontal differential variables 
        #  tp  fp
        #  tn  fn 
        kde_h = ['ks.statistic',  'se', 'kurtosis', 'skew',  ]  # ks.pvalue,  'se', 'kurtosis', 'skew', 'range'
        hterms = ['tpfp', 'tnfn'] # + ['tpfn', 'tnfp', ]
        dx = kde_h 
        delh = {t: {} for t in hterms}
        for d in dx:  

            # if d in ['se', ]:  # ratio
            #     del_tpfp = diffs['tp'][d]/(diffs['fp'][d]+1e-4)
            #     del_tnfn = diffs['tn'][d]/(diffs['fn'][d]+1e-4)
            # else:  # difference

            # horizontal terms
            delh['tpfp'] = diffs['tp'][d] - diffs['fp'][d]
            delh['tnfn'] = diffs['tn'][d] - diffs['fn'][d]

            # cross terms
            delh['tpfn'] = diffs['tp'][d] - diffs['fn'][d]
            delh['tnfp'] = diffs['tn'][d] - diffs['fp'][d]

            # raw features
            for term in hterms: 
                vn = '{}-{}'.format(d, term)
                fvn.append(vn)  # add variable name
                fv.append(delh[term])  # add the value
                tags[i][vn] = delh[term]
            # ... del_tpfp can still be 'noisy'

            # inferred features
            vn = '{}:tp>=fp'.format(d)
            val = 1.0 if delh['tpfp'] >= 0 else 0.0
            fvn.append(vn); fv.append(val)
            tags[i][vn] = val

            vn = '{}:tn>=fn'.format(d)
            val = 1.0 if delh['tnfn'] >= 0 else 0.0
            fvn.append(vn); fv.append(val)
            tags[i][vn] = val

            msg += "... Differential KS on {}-example | metric: {} | R[i,j]: {}, rank: {} | KS(tp-fp, tn-fn): {}, {}\n".format(name, d, pt_q, rk, delh['tpfp'], delh['tnfn']) #  del_tpfn, del_tnfp

        # collect all the related differential variables
        # fv.extend(sv); fvn.extend(dx)
    ### end if Rm 

    # --- threshold dependent statistics ---
    # ... delta p_th
    vn = 'c-label'  # class label
    delta = pt_q - p_th[i]
    cl = pos_label if delta >= 0 else neg_label
    fvn.append(vn); fv.append( cl )
    tags[i][vn] = cl

    # --- rank statistics ----
    if Rm is not None:
        vn = 'h-rank'   # horizontal rank
        rkh = np.searchsorted(Rm[i, :], pt_q, side='left')  # rank, q's position in R[i, :]
        rkh = rkh / (Ni+0.0)

        fvn.append(vn); fv.append( rkh )
        tags[i][vn] = rkh

        vn = 'v-rank' # vertical rank
        rkv = stats.rankdata(R[:, j])[i]  # /(n_users+0.0)   
        rkv = rkv / (Nu+0.0)

        fvn.append(vn); fv.append( rkv )
        tags[i][vn] = rkv  
        
        # ... threshold-specific rank
        # if delta >= 0: 
        #     pts = Rm[i, :][Rm >= p_th[i]]
        #     Np = len(pts)
        #     rk = np.searchsorted(pts, pt_q, side='left')

        #     # standardize
        #     rk = rk/(Np+0.0)
        # else: 
        #     # negative rank
        #     pts = Rm[i, :][Rm < p_th[i]]
        #     Nn = len(pts)
        #     rkc = np.searchsorted(pts, pt_q, side='left')
        #     rk = -((Nn+1)-rkc)

        #     # standardize
        #     rk = rk/(Nn+0.0)

        # N = Rm.shape[1]
        # Rm: either a sorted array (or a rank array)
        # r = np.searchsorted(R[i, :], q, side='left')  

    # --- (raw) confidence score ---
    if C is not None: 
        vn = 'c-score'
        conf_score = C[i, j]
        # fv.append(conf_score)
        # fvn.append(index_var(vn, i))
        # tags[i][vn] = conf_score

    # --- sequence statistics --- 
    if include_chain: 
        if i == 0: 
            vn = 'bp_start'  # beginning of the ensemble
            fvn.append(vn)
            fv.append(1.0)

        lags = [1, 2, ]
        chain_vars = ['value', 'c-label', ]
        for lag in lags: 
            if i_cur >= lag:   # i_cur alias of i
                # --- previous values ---
                for vn in chain_vars:  # KDE variables: tail-tptn, tail-fpfn, ... 
                    if vn in tags[i-lag]: 
                        vn_t = index_var(vn, -lag) # '{}(t-{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i-lag][vn])

                # rank statistics already are of sequential nature, no need to include

                # user/classifier 
                if U is not None:
                    prev_user, bnum = base_name(U[i-lag]) # resolve_var(tags, i-lag, prefix='user', exception=True)
                    vn_t = index_var(prev_user, -lag)
                    fvn.append(vn_t)
                    fv.append(1.0)

                # KDE derivatives
                for d in kde_v:  # ['tail']
                    vn = '{}:tp>=tn'.format(d)
                    assert vn in tags[i-lag], "missing feature {} at index={} | tags: {}".format(vn, i-lag, tags)
                    if vn in tags[i-lag]: 
                        vn_t = index_var(vn, -lag) # '{}(t-{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i-lag][vn])
                    vn = '{}:fp>=fn'.format(d)
                    if vn in tags[i-lag]: 
                        vn_t = index_var(vn, -lag) # '{}(t-{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i-lag][vn])
                        
                for d in kde_h:  # ['ks.statistic', 'se', ]
                    vn = '{}:tp>=fp'.format(d)
                    if vn in tags[i-lag]: 
                        vn_t = index_var(vn, -lag) # '{}(t-{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i-lag][vn])
                        
                    vn = '{}:tn>=fn'.format(d)
                    if vn in tags[i-lag]: 
                        vn_t = index_var(vn, -lag) # '{}(t-{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i-lag][vn])
                        
            # --- future values ---
            if (i_cur+lag < Nu) and ((i_cur+lag) in tags): 
                for vn in chain_vars:  # KDE variables: tail-tptn, tail-fpfn, ... 
                    if vn in tags[i+lag]: 
                        vn_t = index_var(vn, +lag) # '{}(t+{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i+lag][vn])
                        
                # rank statistics already are of sequential nature, no need to include

                # user/classifier 
                if U is not None:
                    next_user, bnum = base_name(U[i+lag]) # resolve_var(tags, i+lag, prefix='user', exception=True)
                    vn_t = index_var(next_user, +lag)
                    fvn.append(vn_t)
                    fv.append(1.0)

                # KDE derivatives
                for d in kde_v:  # ['tail']
                    vn = '{}:tp>=tn'.format(d)
                    if vn in tags[i+lag]: 
                        vn_t = index_var(vn, +lag) # '{}(t+{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i+lag][vn])
                    vn = '{}:fp>=fn'.format(d)
                    if vn in tags[i+lag]: 
                        vn_t = index_var(vn, +lag) #'{}(t+{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i+lag][vn])
                        
                for d in kde_h:  # ['ks.statistic', 'se', ]
                    vn = '{}:tp>=fp'.format(d)
                    if vn in tags[i+lag]: 
                        vn_t = index_var(vn, +lag) # '{}(t+{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i+lag][vn])
                        
                    vn = '{}:tn>=fn'.format(d)
                    if vn in tags[i+lag]: 
                        vn_t = index_var(vn, +lag) # '{}(t+{})'.format(vn, lag)
                        fvn.append(vn_t)
                        fv.append(tags[i+lag][vn])

    # last position
    if i_cur == Nu-1: 
        # note: in 'sequence' mode, don't set to true for all i's; redundant 

        # ... majority vote
        nbp = R.shape[0]
        r_maxvote = 0.0
        neg_label, pos_label = 0, 1
        
        lhv = np.zeros(nbp)
        idx_pos = np.where(R[:, j] >= p_th)[0]
        lhv[idx_pos] = 1

        counter = collections.Counter(lhv)
        # top_vote = counter.most_common(1)[0]
        n_neg, n_pos = counter[neg_label], counter[pos_label]
        # maxvote, n_maxvote = top_vote

        idx_activated = []
        if n_pos > n_neg: 
            maxvote = pos_label
            minvote = neg_label
            n_maxvote, n_minvote = counter[pos_label], counter[neg_label]
            idx_activated = idx_pos
        else: 
            maxvote = neg_label 
            minvote = pos_label
            n_maxvote, n_minvote = counter[neg_label], counter[pos_label]
            idx_activated = np.where(R[:, j] < p_th)[0]
        r_maxvote, r_minvote = n_maxvote/(nbp+0.0), n_minvote/(nbp+0.0) 
            
        # fv.extend( [r_minvote, r_maxvote] )
        vn = 'maxvote'   # maxvote and its proportional (degree of belief)
        fvn.append(vn)
        fv.append(maxvote)
        tags[i][vn] = maxvote

        vn = 'r_maxvote'
        fvn.append(vn)
        fv.append(r_maxvote)
        tags[i][vn] = r_maxvote

        # include the classifiers/users that are 'activated' (i.e. members of majority votes) for this column/datatum
        if U is not None: 
            for u in np.asarray(U)[idx_activated]: 
                vn, bnum = base_name(u)
                vn = decorate_var(vn, 'activated')
                fvn.append(vn)
                fv.append(1.0)   

    # assert len(fv.shape) == 1
    assert len(fv) == len(fvn), "dim(fv): {} <> dim(fvn): {}".format(len(fv), len(fvn))
    if verbose: 
        # for vn, v in zip(fvn, fv): 
        msg += "(get_vars_hstats) vars name values ({}):\n... {}\n".format(name, list(zip(fvn, fv)))
        # print("... q: {}, topk_th: {} | r_min: {}".format(q, topk_th, r_min))   # ... ok
        print(msg)

    if tagging_only: 
        return tags
    elif to_dict: 
        return dict(zip(fvn, fv))

    return np.array(fv)

def get_vars_hstats(R, i, j, p_th, Rm=None, C=None, Lh=None, U=None, Uc=None, encoder=None, p_model={}, r_min=0.1, name='', 
        index=0, verbose=False, wsize=20, to_dict=False, neg_label=0, pos_label=1):  
    """

    Params
    ------


    Memo
    ----

    """
    def base_name(cls_name, prefix='', sep='.'):
        bn = 0
        cn, *x = cls_name.split(sep)
        if prefix: cn = "{}-{}".format(prefix, cn) 
        if len(x) > 0: bn = x[0]
        return cn, bn

    # get BP prediction vector statistics as variables
    import scipy.stats as stats
    from scipy.stats import kurtosis, skew, ks_2samp
    import numpy as np

    # sample_types = ['tp', 'tn'] + ['fp', 'fn']
    # codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
    #         'unk': 0, 't': 3, 'f': -3}
    sample_types = Polarity.sample_types
    codes = Polarity.codes

    # settings for KDE features
    tDummyCoding = False

    tKDERawF = True
    tKDEDerivedF = False
    # neg_label, pos_label = 0, 1
    Nu, Ni = R.shape

    msg = ""
    # query point 
    q = pt_q = R[i, j]   # q

    fv = []  # features 
    fvn = []  # feature names

    # vn = 'value'
    # fvn.append(vn)
    # fv.append(q)
    # ... subsumed by horizontal rank

    if encoder is None: 
        vn = 'user'
        assert Uc is not None

        # le = preprocessing.LabelEncoder()  # should have been integer-encoded prior to the call
        fvn.append(vn)
        assert isinstance(Uc[i], (int, np.int64, np.int32))
        fv.append(Uc[i])  
    else:
        # need the encoder to map classifier to its right one-hot encoding 
        if tDummyCoding: 
            nu = len(encoder.classes_)
            vnx = [encoder.classes_[u] for u in range(nu)]
            fvx = [0] * nu
            fvx[Uc[i]] = 1   # Uc[i]: numerical value for i-th classifier/user according to 'u_encoder'

            fvn.extend(vnx)
            fv.extend(fvx)
        else: 
            vn = 'user'
            # note: if we wish to consider all the 'activated' classifiers/users, then perhaps dummy coding will introduce too many variables? 
            fvn.append(vn)
            assert isinstance(Uc[i], (int, np.int64, np.int32))
            fv.append(Uc[i])
        
        # inverse 
        # [u_encoder.classes_[u] for u in Uc]

    # 5-number summary
    # for stype in sample_types: 
    #     fv.extend(scopes[stype][i]['summary'])  

    # beta distribution parameters
    # for stype in sample_types: 
    #     fv.extend(scopes[stype][i]['beta'])  

    # ... filter outliers according to the quartile? (done prior to this call)

    # --- KDE signature ---
    vmin, vmax = -100, 100
    sv = []  # P(X>=x) for TP, TN, FP, FN
    tMaxPooling = False
    
    scopes = p_model   # polarity model derived from polarity_feature_extraction() (e.g. KDE models for the 4 sample types)
    vterms = ['tptn', 'fpfn' ] # + ['tpfn', 'fptn'] 
    hterms = ['tpfp', 'tnfn'] # + ['tpfn', 'tnfp', ]
    kde_v = ['tail', ]  
    kde_h = ['ks.pvalue',  'se',  'kurtosis', 'skew', ]  # ks.pvalue,  'se', 'kurtosis', 'skew', 'range'
    if 'kde' in scopes[sample_types[0]][0]: 

        diffs = {stype: {} for stype in sample_types}
        for stype in sample_types:
            vn = 'kde-{}'.format(stype) # variable name

            # has_value = True
            pm = 0.0
            if i in scopes[stype]:  # first need to check if a given sample type even exist
                    
                # ib = int(np.digitize(i, bins=bagging_bins))
                kde =  scopes[stype][i]['kde']  # kde_processed[stype][ib]
                # ... it's possible that i-th BP does not reference some flavors of the particle (e.g. no TPs ...)

                # qv = np.array([pt_q, ])
                # print("(get_vars_hstats) dim(qv): {} => {}".format(qv.shape, qv[:, None].shape))
                # score = kde.score_samples(qv[:, None])[0]  # for sklearn 
                # if score < vmin: score = vmin
                # if score > vmax: score = vmax 

                # pval = kde.evaluate(pt_q)[0]  # kde.evaluate() returns an array
                # if np.isnan(pval): pval = 0.0
                # fv.append( pval )  # probability of R[i,j] in the small neighbborhood

                # compute P(X>=x) using survival function
                ii = np.digitize(pt_q, bins=kde.support)  # which interaval does the query point fall into
                if ii >= kde.support.size: ii = kde.support.size - 1   # max index

                # np.isnan(kde.sf).any()
                if not np.isnan(kde.sf[ii]): pm = kde.sf[ii]
            else: 
                msg += "(get_vars_hstats) {}-th BP does not have sample type: {}\n".format(i, stype) # should rarely occur 
                # has_value = False

            diffs[stype]['tail'] = pm
            
            # if not tMaxPooling: 
            #     fv.append( pm )
            #     fvn.append( vn )
            sv.append( pm )
        ### end foreach flavor
        
        # collecting the vertical differentials
        #  tp  fp
        #  tn  fn 
        kde_v = ['tail', ]
        vterms = ['tptn', 'fpfn' ] 
        dx = kde_v  # 'cde'
        delv = {t: {} for t in vterms}
        for d in dx:  
            # vertical
            delv['tptn'] = diffs['tp'][d] - diffs['tn'][d]
            delv['fpfn'] = diffs['fp'][d] - diffs['fn'][d]

            # cross terms 
            delv['tpfn'] = diffs['tp'][d] - diffs['fn'][d]
            delv['fptn'] = diffs['fp'][d] - diffs['tn'][d]
            
            # a. raw features
            # [del_tptn, del_fpfn, del_tpfn, del_fptn ]
            if tKDERawF: 
                for term in vterms: 
                    vn = '{}-{}'.format(d, term)
                    fvn.append(vn)  # add variable name
                    fv.append(delv[term])  # add the value
            # ... del_fpfn is likely to be redundant as it's similar to del_tptn

            # b. inferred features 
            if tKDEDerivedF: 
                vn = '{}:tp>=tn'.format(d)
                val = 1.0 if delv['tptn'] >= 0 else 0.0   # tail: TP > TN  i.e. more likely to be TP than TN if tp dominates tn in 1-CDF
                fvn.append(vn); fv.append(val)

                vn = '{}:fp>=fn'.format(d)
                val = 1.0 if delv['fpfn'] >= 0 else 0.0
                fvn.append(vn); fv.append(val)

            msg += "... Differential KS on {}-example | metric: {} | R[i,j]: {} | KS(tp-tn, fp-fn): {}, {}\n".format(name, 
                d, pt_q, delv['tptn'], delv['fpfn']) 

        # if verbose: print("(get_vars_hstats) KDE signature: {} | {}".format(fv, name))
      
        
        # ... other KDE signature HERE 
    ### end KDE section

    # ... finding P(X>=x) is good at distigushing TP from TN (and FP from FN) but no so much for separting TP from FP or TN from FN
    # ... TP <> TN, FP <> FN
    
    
    # --- other horizontal statistics ---

    # ... quartile (4-quantile) regardless of the particle flavor
    # sv = common.five_number(R[i, :])  # (x_min, quartiles[0], quartiles[1], quartiles[2], x_max)
    # q_digitized = int(np.digitize(pt_q, bins=sv))
    # fv.append(q_digitized) # if 2, smaller than median, if 3, larger than median

    ### flavor dependent statistics
    # ... larger than TP's median? 
    # p_high = np.float(pt_q - scopes['tp'][i]['median']) if i in scopes['tp'] else 1.0  # max distance is 1.0
    # fv.append( p_high )

    # # smaller than TN's median? 
    # p_low = np.float(pt_q - scopes['tn'][i]['median']) if i in scopes['tn'] else 1.0
    # fv.append( p_low )

    # ... q vs median(TP) and median(FP)
    max_gap = 1.0
    # gap_tp = abs(pt_q - scopes['tp'][i]['median']) if i in scopes['tp'] else max_gap
    # gap_fp = abs(pt_q - scopes['fp'][i]['median']) if i in scopes['fp'] else max_gap
    # affinity_tp = float(gap_tp > gap_fp)   # closer to TP's median or FP's median

    # gap_tn = abs(pt_q - scopes['tn'][i]['median']) if i in scopes['tn'] else max_gap
    # gap_fn = abs(pt_q - scopes['fn'][i]['median']) if i in scopes['fn'] else max_gap
    # affinity_tn = float(gap_tn > gap_fn)
    # fv.extend([affinity_tp, affinity_tn])

    # --- local distribution (expensive; use nearest neighbors instead) ---
    # q = R[i, j] 
    N = R.shape[1]
    rk = -1   # rank of the query point
    
    wsize_min, wsize_max = N//100, 20 
    wsize = min(wsize_max, max(wsize_min, wsize))

    sv = []  # reset sv
    if Rm is not None: 
        # ... Rm[i, :] must be sorted
        rk = np.searchsorted(Rm[i, :], pt_q, side='left')  # rank, q's position in R[i, :]
        k = wsize # min(min_size, N//100)  # no less than 50 points
     
        # search k nearest neighbor
        bl, br = rk-k, rk+k   

        # boundary conditions
        if bl < 0: bl = 0
        if br >= N: br = N-1 

        pts_lower = Rm[i, bl:rk]
        pts_higher = Rm[i, rk:br]
        assert np.all(pts_lower <= pt_q) and np.all(pts_higher >= pt_q), "query: {} | max(low): {}, min(high): {} | sorted? {}".format(pt_q, np.max(pts_lower), np.min(pts_higher), scipy.stats.rankdata(Rm[i, :][:20]))
        
        # SE wrt q
        pts_knn = np.hstack([pts_lower, pts_higher])
        
        se_local = np.std(pts_knn) # mean_squared_error(pts_knn, np.full_like(pts_knn, pt_q))
        range_local = stats.iqr(pts_knn) # np.max(pts_knn)-np.min(pts_knn)
        # fv.append(se_local)  # <<< 

        if verbose: 
            n = 10
            pts_knn_subset = np.hstack([pts_lower[:n], pts_higher[-n:]])
            msg += "(get_vars_hstats) {} | local(SE): {} | R[i,j]: {}, rank: {} | pts_knn (subset:{}, total: {} vs N/100:{}): {}\n".format(name,
                se_local, pt_q, rk, len(pts_knn_subset), len(pts_knn), N//100, pts_knn_subset)

        # compare distribution with neighoring points of different flavors
        vars_stats = ['ks.statistic', 'ks.pvalue', 'median', 'skew', 'kurtosis', 'se', 'range', ]
        diffs = {stype: {v:0.0  for v in vars_stats} for stype in sample_types}
        for stype in sample_types:
            kss = np.max(pts_knn-0.0)   # assumed to be very large (as large as max abs distance to zero vectors)
            if i in scopes[stype]: 
                pts = scopes[stype][i]['sample']
                # ... assuming that pts has been pre-sorted
                n = len(pts)

                # rank wrt to this particular sample type
                rk = np.searchsorted(pts, pt_q, side='left')

                # search k nearest neighbor
                bl, br = rk-k, rk+k

                # boundary conditions
                if bl < 0: bl = 0
                if br >= N: br = n-1 

                pts_lower = pts[bl:rk]
                pts_higher = pts[rk:br]
                assert np.all(pts_lower <= pt_q) and np.all(pts_higher >= pt_q), "query: {} wrt flavor: {} | max(low): {}, min(high): {}".format(pt_q, stype, np.max(pts_lower), np.min(pts_higher))
                
                pts_knn_flavored = np.hstack([pts_lower, pts_higher])
                range_flavored = stats.iqr(pts_knn_flavored) # np.max(pts_knn_flavored)-np.min(pts_knn_flavored)
                se_flavored = np.std(pts_knn_flavored)

                # K-S tes
                ks_stats = ks_2samp(pts_knn, pts_knn_flavored)
                kss = -10*np.log(ks_stats.pvalue) # ks_stats.statistic, -10*np.log(ks_stats.pvalue) 
                msg += "(get_vars_hstats) Local structure: {} | flavor: {} | R[i,j]: {}, rank: {} | KS statistic: {}\n".format(name, stype, pt_q, rk, kss)

                diffs[stype]['ks.statistic'] = ks_stats.statistic
                diffs[stype]['ks.pvalue'] = -10*np.log(ks_stats.pvalue) if ks_stats.pvalue != 0 else 0.0 # ks_stats.pvalue 
                diffs[stype]['median'] = np.median(pts_knn_flavored)/(np.median(pts_knn)+1e-4)
                diffs[stype]['skew'] = skew(pts_knn_flavored)/(skew(pts_knn)+1e-4)
                diffs[stype]['kurtosis'] = kurtosis(pts_knn_flavored)/(kurtosis(pts_knn)+1e-4)
                diffs[stype]['se'] = se_flavored/(se_local+1e-4)
                diffs[stype]['range'] = range_flavored/(range_local+1e-4)
            else: 
                msg += "... !!!{}-th point in R does not have sample type: {}\n".format(i, stype)

            # sv.append(kss) # ... use differential variables instead of the values themselves

        # horizontal differential variables 
        #  tp  fp
        #  tn  fn 
        kde_h = ['ks.pvalue', 'se', 'kurtosis', 'skew', ]
        hterms = ['tpfp', 'tnfn'] # + ['tpfn', 'tnfp', ]
        dx = kde_h  # ['ks.statistic',  'se',  ]  # ks.pvalue,  'se', 'kurtosis', 'skew', 'range', 
        delh = {t: {} for t in hterms}
        for d in dx:  

            # if d in ['se', ]:  # ratio
            #     del_tpfp = diffs['tp'][d]/(diffs['fp'][d]+1e-4)
            #     del_tnfn = diffs['tn'][d]/(diffs['fn'][d]+1e-4)
            # else:  # difference

            # horizontal terms
            delh['tpfp'] = diffs['tp'][d] - diffs['fp'][d]
            delh['tnfn'] = diffs['tn'][d] - diffs['fn'][d]

            # cross terms
            delh['tpfn'] = diffs['tp'][d] - diffs['fn'][d]
            delh['tnfp'] = diffs['tn'][d] - diffs['fp'][d]

            if tKDERawF: 
                for term in hterms: 
                    vn = '{}-{}'.format(d, term)
                    fvn.append(vn)  # add variable name
                    fv.append(delh[term])  # add the value

            # inferred features
            if tKDEDerivedF: 
                vn = '{}:tp>=fp'.format(d)
                val = 1.0 if delh['tpfp'] >= 0 else 0.0
                fvn.append(vn); fv.append(val)

                vn = '{}:tn>=fn'.format(d)
                val = 1.0 if delh['tnfn'] >= 0 else 0.0
                fvn.append(vn); fv.append(val)

            msg += "... Differential KS on {}-example | metric: {} | R[i,j]: {}, rank: {} | KS(tp-fp, tn-fn): {}, {}\n".format(name, d, pt_q, rk, delh['tpfp'], delh['tnfn']) #  del_tpfn, del_tnfp

        # collect all the related differential variables
        # fv.extend(sv); fvn.extend(dx)
    ### end if Rm 

    # ... delta median 
    # tstats = 'median'  # target statistic
    # for stype in sample_types: 
    #     if i in scopes[stype]:  # first need to check if a given sample type even exist
    #         val = scopes[stype][i][tstats]  
    #     else: 
    #         print("(get_vars_hstats) {}-th BP does not have sample type: {} | when comparing to {}".format(i, stype, tstats)) # should rarely occur 
    #         val = 0.5 
    #     fv.append(q - val)  

    # --- threshold dependent statistics ---
    # ... delta p_th
    vn = 'c-label'  # class label
    delta = pt_q - p_th[i]
    cl = pos_label if delta >= 0 else neg_label
    fvn.append(vn)
    fv.append(cl)
    # ... case q > p_th, likely TP if L = 1, or FP if L = 0
    # ... case q <= p_th, likely FN if L = 1, or TN if L = 0

    ### rank?  can also use q-
    if Rm is not None:
        # a. ranking via quantiles
        # vn = 'h-rank'   # horizontal rank
        # sv = common.five_number(R[i, :])  # (x_min, quartiles[0], quartiles[1], quartiles[2], x_max)
        # q_digitized = int(np.digitize(pt_q, bins=sv))
        # fvn.append(vn); fv.append(q_digitized)
        # ... this info can be subsummed by the query-point feature 

        # b. ranking via sorting
        vn = 'h-rank'   # horizontal rank
        nh = Rm.shape[1]
        rkh = np.searchsorted(Rm[i, :], pt_q, side='left')  # rank, q's position in R[i, :]
        fvn.append(vn); fv.append( rkh /(Ni+0.0) )
        # -------------------------------------------------- 
    
        # a. ranking via quantiles
        vn = 'v-rank'  # vertical rank
        # sv = common.five_number(R[:, j])  # (x_min, quartiles[0], quartiles[1], quartiles[2], x_max)
        # q_digitized = int(np.digitize(pt_q, bins=sv))
        # fvn.append(vn); fv.append(q_digitized)

        # b. ranking via sorting
        nv = Rm.shape[0]
        rkv = stats.rankdata(R[:, j])[i]  # /(n_users+0.0)   
        fvn.append(vn); fv.append( rkv/(Nu+0.0) ) 
        
        # ... threshold-specific rank
        # if delta >= 0: 
        #     pts = Rm[i, :][Rm >= p_th[i]]
        #     Np = len(pts)
        #     rk = np.searchsorted(pts, pt_q, side='left')

        #     # standardize
        #     rk = rk/(Np+0.0)
        # else: 
        #     # negative rank
        #     pts = Rm[i, :][Rm < p_th[i]]
        #     Nn = len(pts)
        #     rkc = np.searchsorted(pts, pt_q, side='left')
        #     rk = -((Nn+1)-rkc)

        #     # standardize
        #     rk = rk/(Nn+0.0)

        # N = Rm.shape[1]
        # Rm: either a sorted array (or a rank array)
        # r = np.searchsorted(R[i, :], q, side='left')  
        
    
    # top k: does the query point has a value larger than min(top k)?
    # topk_th = percentile(R[i, :], 100.0-r_min)
    # fv.append( int(q >= topk_th) )
    # lowerk_th = percentile(R[i, :], 10)
    # fv.append( int(q <= lowerk_th) )

    # --- (raw) confidence score ---
    vn = 'c-score'
    if C is not None: 
        fvn.append(vn)
        fv.append(C[i, j])

    # --- vertical statistics ---
    # ... majority vote
    nbp = R.shape[0]
    r_maxvote = 0.0
    neg_label, pos_label = 0, 1
    
    lhv = np.zeros(nbp)
    idx_pos = np.where(R[:, j] >= p_th)[0]
    lhv[idx_pos] = 1

    counter = collections.Counter(lhv)
    # top_vote = counter.most_common(1)[0]
    n_neg, n_pos = counter[neg_label], counter[pos_label]
    # maxvote, n_maxvote = top_vote

    idx_activated = []
    if n_pos > n_neg: 
        maxvote = pos_label
        minvote = neg_label
        n_maxvote, n_minvote = counter[pos_label], counter[neg_label]
        idx_activated = idx_pos
    else: 
        maxvote = neg_label 
        minvote = pos_label
        n_maxvote, n_minvote = counter[neg_label], counter[pos_label]
        idx_activated = np.where(R[:, j] < p_th)[0]
    r_maxvote, r_minvote = n_maxvote/(nbp+0.0), n_minvote/(nbp+0.0) 
        
    # fv.extend( [r_minvote, r_maxvote] )
    vn = 'maxvote'   # maxvote and its proportional (degree of belief)
    fvn.append(vn)
    fv.append(maxvote)

    vn = 'r_maxvote'
    fvn.append(vn)
    fv.append(r_maxvote)

    # include the classifiers/users that are 'activated' (i.e. members of majority votes) for this column/datatum
    if U is not None: 
        assert isinstance(U[0], str), "Diagnosis: Mixed U and Uc? U:{}".format(U[:10])
        # note: pass U instead of Uc so that we can use user/classifer's names as feature names

        nu = len(encoder.classes_)
        vnx = [encoder.classes_[u] for u in range(nu)]
        active_counts = {vn: 0 for vn in vnx}
        
        for u in np.asarray(U)[idx_activated]:
            vn, bnum = base_name(u)
            active_counts[vn] += 1
        
        fvx = [active_counts[vn] for vn in vnx]

        fvn.extend(vnx)
        fv.extend(fvx)
              
    # assert len(fv.shape) == 1
    assert len(fv) == len(fvn), "dim(fv): {} <> dim(fvn): {}".format(len(fv), len(fvn))
    if verbose: 
        # for vn, v in zip(fvn, fv): 
        msg += "(get_vars_hstats) vars name values ({}):\n... {}\n".format(name, list(zip(fvn, fv)))
        # print("... q: {}, topk_th: {} | r_min: {}".format(q, topk_th, r_min))   # ... ok
        print(msg)

    # ensure that none of the features is too extremem 
    assert np.where(np.isnan(fv))[0].size == 0, "dubious fv: {}".format(dict(zip(fvn, fv)))

    if to_dict: 
        return dict(zip(fvn, fv))

    return np.array(fv)

def get_vars_vstats(R, i, j, p_th, Rm=None, C=None, Lh=None, p_model={}, 
        r_min=0.1, name='', index=0, verbose=False, wsize=10, to_dict=False, neg_label=0, pos_label=1):
    fv = []  # features 
    fvn = []  # feature names
    if to_dict: 
        return dict(zip(fvn, fv))
    return np.array(fv)

def polarity_correction(Po, p_th=[], p_model={}, n_symbols_col=2, pos_label=1, neg_label=0):

    sample_types = Polarity.sample_types # ['tp', 'tn'] + ['fp', 'fn']
    codes = Polarity.codes
    inv_codes = {v: k for k, v in codes.items()}

    ntp0 = np.sum(Po == codes['tp'])
    ntn0 = np.sum(Po == codes['tn'])
    nfp0 = np.sum(Po == codes['fp'])
    nfn0 = np.sum(Po == codes['fn'])

    Nu, Ni = Po.shape 
    Pc = np.copy(Po)

    n_mixed = 0
    delta_tp = delta_tn = delta_fp = delta_fn = 0
    for j in range(Ni):
        counts = collections.Counter(Pc[:, j])
        if len(counts) > n_symbols_col:  # not a 'pure' sequence (either should be TP-FN seuqence or TN-FP sequence)
            n_mixed += 1
            n_tpfn = counts[ inv_codes['tp'] ] + counts[ inv_codes['fn'] ]
            n_tnfp = counts[ inv_codes['tn'] ] + counts[ inv_codes['fp'] ]
            if n_tpfn > n_tnfp: 
                # TP-FN predominent

                # TN -> FN
                idx_tn = np.where(Pc[:, j] == inv_codes['tn'])[0]
                if len(idx_tn) > 0: 
                    Pc[idx_tn, j] = inv_codes['fn']
                    delta_fn += len(idx_tn)

                # FP -> TP
                idx_fp = np.where(Pc[:, j] == inv_codes['fp'])[0]
                if len(idx_fp) > 0: 
                    Pc[idx_fp, j] = inv_codes['tp']
                    delta_tp += len(idx_fp)

            elif n_tpfn < n_tnfp: 
                # TN-FP document

                # TP -> FP
                idx_tp = np.where(Pc[:, j] == inv_codes['tp'])[0]
                if len(idx_tp) > 0: 
                    Pc[idx_tp, j] = inv_codes['fp']
                    delta_fp += len(idx_tp)

                # FN -> TN
                idx_fn = np.where(Pc[:, j] == inv_codes['fn'])[0]
                if len(idx_fn) > 0: 
                    Pc[idx_fn, j] = inv_codes['tn']
                    delta_tn += len(idx_fn)
    
    msg = "(polarity_correction) delta_tp: {}, delta_fp: {}\n".format(delta_tp, delta_fp)

    ntp = np.sum(Pc == codes['tp'])
    ntn = np.sum(Pc == codes['tn'])
    nfp = np.sum(Pc == codes['fp'])
    nfn = np.sum(Pc == codes['fn'])

    msg += "(polarity_correction) n(tp): {} -> {}, n(fn): {} -> {} | n(tn): {} -> {}, n(fp): {} -> {}\n".format(ntp0, ntp,
        nfn0, nfn, ntn0, ntn, nfp0, nfp)
    print(msg)

    return Pc

def polarity_sample_bootstrap(R, C, p_th=[], Lr=[], p_model={}, n_samples=100, Ro=None, target_label=1, pos_label=1, neg_label=0): 
    """

    Memo
    ----
    1. flatten a 2D (n-by-1 or 1-by-n) array to 1D 

       np.ndarray.flatten(x)


    """
    # precondition: polarity_feature_extraction() has been involved
    n_users = R.shape[0]

    # Mc, Lh = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
    # predict_pos = (Lh == pos_label)  # Given BP's prediction Lh, select entries ~ target label
    # predict_neg = (Lh == neg_label)
    # cells_tp = (Mc == 1) & predict_pos   # estimated
    # cells_tn = (Mc == 1) & predict_neg
    # cells_fp = (Mc == 0) & predict_pos
    # cells_fn = (Mc == 0) & predict_neg

    if n_samples < 1: n_samples = 1
    Ra = np.zeros((n_users, n_samples))
    Ca = np.zeros((n_users, n_samples))

    low, high = 0.0, 1.0
    if target_label == pos_label: 
        idx_pos = np.random.choice(np.where(Lr == pos_label)[0], n_samples, replace=True)

        for jc, j in enumerate(idx_pos):  # foreach column index ~ positive class
            
            # identiy TPs and FNs in this TP-FN sequence (across users)
            sv = []
            for i in range(n_users):  
                stype = 'tp' if R[i, j] >= p_th[i] else 'fn'  # if the prediction >= threshold, then TP otherwise FN
                sv.append(stype)
            # ... polarity vector + + - + - ... 
            
            # now generate probabilty/rating values consistent with the sample type
            pvj = []
            cvj = []
            for i in range(n_users): 
                flavor = sv[i]      

                # if Ro is not None: low, high = Ro[i]  # outlier bound
                
                # p_model[flavor][i]['sample']
                # pvh = R[i, :]
                # idx = np.where(pvh >= p_th[i])[0]
                c = C[i, j]  # assuming that the confidence score remains similar

                pv = p_model[flavor][i]['sample'] if i in p_model[flavor] else []  # scopes['tp'][i]
                
                # [test]
                if flavor == 'tp' and len(pv) > 0: 
                    assert np.all(pv >= p_th[i]), "Found {}-inconsistent sample in pv against p_th[i]: {} | {}".format( p_th[i], pv[pv < p_th[i]] )                   

                if len(pv) > 0: 
                    r = np.random.choice(pv, 1)[0]
                else: 
                    # in this case, we can only hope to generate an unconstrained sample (flavor-free sample)
                    print("(polarity_sample_bootstrap) {}-th classifier does not have probabilities of sample type: {}".format(i, flavor))
                    pvh = R[i, :]
                    idx = np.where(pvh >= p_th[i])[0]
                    pv_j = np.random.choice(idx, 1)[0]
                    r = R[i, pv_j]  # np.random.choice(pvh[pvh >= p_th[i]], 1)[0]

                # if r < low: r = low
                # if r > high: r = high
                
                pvj.append(r)
                cvj.append(c)
            # ... geneated whole column worth of sample
            # np.asarray(cvj) > p_th[i]
            cvj = np.array(cvj); pvj = np.array(pvj)
            Ca[:, jc] = cvj
            Ra[:, jc] = pvj

            # [test]
            # it's possbile that due to the cutoff from outlier bounds, the probability value may not be consistent with p_th
            idx_tp = np.where(np.asarray(sv)=='tp')[0]
            for i in idx_tp: 
                if not (pvj[i] >= p_th[i]):
                    msg = "... flavor-inconsistent sample found | pvj[i]: {}, sv[i]: {} | p_th[i]: {}\n".format(pvj[i], sv[i], p_th[i])
                    raise ValueError(msg)
            # assert np.all(pvj[idx_tp] >= p_th[i]), \
            #     "Found non-TPs | pvj[np.where(np.asarray(sv)=='tp')[0]]: {} | p_th[i]: {} | outlier bound: low: {}, high:{}".format(pvj[np.where(np.asarray(sv)=='tp')[0]], p_th[i], low, high)
           
    else: 
        idx_neg = np.random.choice(np.where(Lr == neg_label)[0], n_samples, replace=True)

        for j in idx_neg: 
            
            # identiy TNs and FPs in this TN-FP sequence (across users) 
            sv = []
            for i in range(n_users):  
                stype = 'tn' if R[i, j] < p_th[i] else 'fp'  # if the prediction >= threshold, then TP otherwise FN
                sv.append(stype)
            # ... polarity vector + + - + - ... 
            
            # now generate probabilty/rating values consistent with the sample type
            pvj = []
            cvj = []
            for i in range(n_users): 
                flavor = sv[i]      

                # if Ro is not None: low, high = Ro[i]  # outlier bound
                
                # p_model[flavor][i]['sample']
                # pvh = R[i, :]
                # idx = np.where(pvh >= p_th[i])[0]
                c = C[i, j]  # assuming that the confidence score remains similar

                pv = p_model[flavor][i]['sample'] if i in p_model[flavor] else []  # scopes['tp'][i]
                if len(pv) > 0: 
                    r = np.random.choice(pv, 1)[0]
                else: 
                    # in this case, we can only hope to generate an unconstrained sample (flavor-free sample)
                    print("(polarity_sample_bootstrap) {}-th classifier does not have probabilities of sample type: {}".format(i, flavor))
                    pvh = R[i, :]
                    idx = np.where(pvh < p_th[i])[0]
                    pv_j = np.random.choice(idx, 1)[0]
                    r = R[i, pv_j]  # np.random.choice(pvh[pvh >= p_th[i]], 1)[0]

                # if r < low: r = low
                # if r > high: r = high
                
                cvj.append(c)
                pvj.append(r)
            # ... geneated whole column worth of sample

            Ca[:, jc] = cvj
            Ra[:, jc] = pvj

    return Ra, Ca

def polarity_feature_extraction(R, Lr, p_th, T, Lt=None, C=None, U=None, 
        max_size_kde=1000, max_size_user=10000,
        pos_label=1, neg_label=0, bag_count=10, fold_count=5, index=0): 
    """
 
    Params
    ------
    max_size_kde: 
    max_size_user: max sample size used to keep track of the rating/probability values for each user/classifier.
    
    """
    def get_bins(n_classifiers=5, bag_count=10):
        # e.g. n_classifiers = 5 
        #      => [0, 10, 20, 30, 40, 50]
        return [i * bag_count for i in range(n_classifiers+1)]
    def range_bags(bn, n_classifiers=5, bag_count=10):  # given
        # bn: bin number starting from 1
        assert bn <= n_classifiers and bn >= 1
        imax = bn * bag_count
        return range(imax-bag_count, imax)

    def random_bags(n_classifiers=5, bag_count=10): 
        bns = []  # bag number sequence
        for e in get_bins(n_classifiers)[:-1]:
            bns.append( np.random.randint(e, e+bag_count,1)[0] )
        return bns
    def base_name(cls_name, sep='.'):
        cn, *bn = cls_name.split(sep) 
        return cn

    # import common
    from itertools import chain
    import analyzer
    from sampling import bootstrap_resample
    # import scipy.stats
    from scipy.stats import kurtosis, skew, ks_2samp

    sample_types = Polarity.sample_types # ['tp', 'tn'] + ['fp', 'fn']
    codes = Polarity.codes
    # codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
    #         'unk': 0, 't': 3, 'f': -3}

    Mc, Lh = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
    n_users = R.shape[0]

    # other parameters 
    # bag_count = 10
    classifiers = ['u{}'.format(i) for i in range( int(n_users/bag_count) )]
    if U is not None: classifiers, bag_count = common.infer_bag_count(U, sep='.', verify=False)
    n_classifiers = n_uniq_users = len(classifiers)
    print("(polarity_feature_extraction) n_classifiers: {}\n... {}\n".format(n_classifiers, classifiers))

    predict_pos = (Lh == pos_label)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == neg_label)
    cells_tp = (Mc == 1) & predict_pos   # estimated
    cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    cells_fn = (Mc == 0) & predict_neg
    Ntp = np.sum(cells_tp)
    Ntn = np.sum(cells_tn)
    Nfp = np.sum(cells_fp)
    Nfn = np.sum(cells_fn)
    print("(polarity_feature_extraction) n(TP): {}, n(TN): {}, n(FP): {}, n(FN): {} | Nu: {}, Ni: {}".format(Ntp, Ntn, Nfp, Nfn, R.shape[0], R.shape[1]))

    # estimate proba range for each classifier based on sample type (TP, TN, FP, FN)
    # max_size_kde = 1000  # used for the sampling in KDE
    # max_size_user = 10000
    kernel = 'gaussian' # 'epanechnikov'

    kde_processed = {st: {} for st in sample_types}   # user_processed
    sample_processed = {st: {} for st in sample_types}
    bagging_bins = get_bins(n_classifiers=n_uniq_users, bag_count=bag_count)

    # >>> since computing KDE for each bagged classifier is too expensive, we want to build one KDE model for each classifier
    tAgg = False  # if True, then for each classifier, aggregate all prediction vectors (PVs) from all bags to form a giant vector
    scopes = {st: {} for st in sample_types}   # scope['tp'][0]: to be true positive, 0th classifier must have this proba range
    for i in range(R.shape[0]):  # foreach classifier
        scopes['tp'][i] = {}

        # find sample size and the flavor/sample type with the max sample size (usually TNs)

        # TNs 
        v = R[i, :][cells_tn[i, :]]  # remove outliers?
        if len(v) > 0: 
            flavor = 'tn'

            # downsampling if too large
            if len(v) > max_size_user: v = np.random.choice(v, min(max_size_user, len(v)))
            assert np.all(v < p_th[i]), "(polarity_feature_extraction) Found {}-inconsistent sample: {}".format(flavor, v[v >= p_th[i]][:50])   # ... ok

            Ntn = len(v)
            scopes[flavor][i] = {'min': np.min(v), 'max': np.max(v), 'mean': np.mean(v), 'median': np.median(v), 
                'summary': common.five_number(v), 'sample': np.sort(v) }   
        
            # ... positive polarity candidates 
            # assert scopes['tp'][i]['median'] != scopes['tn'][i]['median'] 

            # a, b, loc, scale = analyzer.fit_beta(v, name='TN', output_path=None, dpi=300, verbose=True, ext='pdf', index=0, save=True, color='blue')
            # scopes['tn'][i]['beta'] = np.array([a, b, loc, scale])

            ### KDE
            # kernels: epanechnikov, gaussian
            uname = U[i] if U is not None else i
            # rbn = random_bags(n_classifiers=n_uniq_users, bag_count=bag_count)  # random bag numbers
            ib = int(np.digitize(i, bins=bagging_bins))
            if not ib in kde_processed[flavor]: 
                # train only one KDEstimator per classifier (instead of per bagged classifier)

                # collect all the bagged prediction vectors
                pv = list(chain.from_iterable([R[u, :][cells_tn[i, :]] for u in range_bags(ib, n_classifiers=n_uniq_users, bag_count=bag_count)])) 
                
                # KDE
                scopes[flavor][i]['kde'] = kde = analyzer.fit_kd2(pv, kernel=kernel, size=max_size_kde, cv=20, name="{}-{}".format(flavor.upper(), base_name(uname)), 
                    output_path=None, dpi=300, verbose=ib==1, ext='pdf', save=True)  # use bag_number
                kde_processed[flavor][ib] = kde

                if tAgg: 
                    # downsampling?
                    # if max_size_user > 0: x = np.random.choice(pv, min(max_size_user, len(x)))
                    scopes[flavor][i]['sample'] = sample_processed[flavor][ib] = np.sort(pv)

            else: 
                scopes[flavor][i]['kde'] = kde_processed[flavor][ib] # used the already computed KD estimator
                if tAgg: scopes[flavor][i]['sample'] = sample_processed[flavor][ib]  # scopes[flavor][ sample_processed[flavor][ib] ]['sample']
        
        # TPs
        v = R[i, :][cells_tp[i, :]]
        if len(v) > 0: 
            flavor = 'tp'

            if len(v) > max_size_user: v = np.random.choice(v, min(max_size_user, len(v)))
            Ntp = len(v)
            if Ntn > Ntp: 
                v, _ = bootstrap_resample(v, n=Ntn); Ntp = Ntn 
            assert np.all(v >= p_th[i]), "(polarity_feature_extraction) Found {}-inconsistent sample: {}".format(flavor, v[v < p_th[i]][:50])

            scopes[flavor][i] = {'min': np.min(v), 'max': np.max(v), 'mean': np.mean(v), 'median': np.median(v), 
               'summary': common.five_number(v), 'sample': np.sort(v)}  # memory intensive

            # a, b, loc, scale = analyzer.fit_beta(v, name='TP', output_path=None, dpi=300, verbose=True, ext='pdf', index=0, save=True, color='blue')
            # scopes['tp'][i]['beta'] = np.array([a, b, loc, scale])

            ### KDE
            # kernels: epanechnikov, gaussian
            uname = U[i] if U is not None else i
            # rbn = random_bags(n_classifiers=n_uniq_users, bag_count=bag_count)  # random bag numbers
            ib = int(np.digitize(i, bins=bagging_bins))
            if not ib in kde_processed[flavor]:  # flavor: 'tp'
                # train only one KDEstimator per classifier (instead of per bagged classifier)

                pv = list(chain.from_iterable([R[u, :][cells_tp[i, :]] for u in range_bags(ib, n_classifiers=n_uniq_users, bag_count=bag_count)]))
                
                # KDE
                scopes[flavor][i]['kde'] = kde = analyzer.fit_kd2(pv, kernel=kernel, size=max_size_kde, cv=20, name="{}-{}".format(flavor.upper(), base_name(uname)),
                    output_path=None, dpi=300, verbose=ib==1, ext='pdf', save=True)  # use bag_number
                kde_processed[flavor][ib] = kde   # KDE shared by i-th bin

                if tAgg: 
                    # downsampling?
                    # if max_size_user > 0: x = np.random.choice(pv, min(max_size_user, len(x)))
                    scopes[flavor][i]['sample'] = sample_processed[flavor][ib] = np.sort(pv)
            else: 
                scopes[flavor][i]['kde'] = kde_processed[flavor][ib] # used the already computed KD estimator
                if tAgg: scopes[flavor][i]['sample'] = sample_processed[flavor][ib]  # scopes[flavor][ sample_processed[flavor][ib] ]['sample']

        # ... negative polarity candidates ... 

        # FPs ~ TPs
        v = R[i, :][cells_fp[i, :]] # v3
        if len(v) > 0: 
            flavor = 'fp'

            if len(v) > max_size_user: v = np.random.choice(v, min(max_size_user, len(v)))
            Nfp = len(v)       # label: 0 => TN, FP
            if Ntp > Nfp:      # match sample size of TP
                v, _ = bootstrap_resample(v, n=Ntp); Nfp = Ntp  # try to get more FP examples since TP-FP discerning signals are weaker
            # assert np.all(v >= p_th[i]), "(polarity_feature_extraction) Found {}-inconsistent sample: {}".format(flavor, v[v < p_th[i]][:50])

            scopes[flavor][i] = {'min': np.min(v), 'max': np.max(v), 'mean': np.mean(v), 'median': np.median(v), 
                'summary': common.five_number(v), 'sample': np.sort(v) }

            # a, b, loc, scale = analyzer.fit_beta(v, name='FP', output_path=None, dpi=300, verbose=True, ext='pdf', index=0, save=True, color='blue')
            # scopes['fp'][i]['beta'] = np.array([a, b, loc, scale])

            ### KDE
            # kernels: epanechnikov, gaussian
            uname = U[i] if U is not None else i
            # rbn = random_bags(n_classifiers=n_uniq_users, bag_count=bag_count)  # random bag numbers
            ib = int(np.digitize(i, bins=bagging_bins))
            if not ib in kde_processed[flavor]: 
                # train only one KDEstimator per classifier (instead of per bagged classifier)
                
                pv = list(chain.from_iterable([R[u, :][cells_fp[i, :]] for u in range_bags(ib, n_classifiers=n_uniq_users, bag_count=bag_count)]))
                
                # KDE 
                scopes[flavor][i]['kde'] = kde = analyzer.fit_kd2(pv, kernel=kernel, size=max_size_kde, cv=20, name="{}-{}".format(flavor.upper(), base_name(uname)), 
                    output_path=None, dpi=300, verbose=ib==1, ext='pdf', save=True)  # use bag_number
                kde_processed[flavor][ib] = kde
                
                if tAgg: 
                    # downsampling?
                    # if max_size_user > 0: x = np.random.choice(pv, min(max_size_user, len(x)))
                    scopes[flavor][i]['sample'] = sample_processed[flavor][ib] = np.sort(pv)
            else: 
                scopes[flavor][i]['kde'] = kde_processed[flavor][ib] # used the already computed KD estimator
                if tAgg: scopes[flavor][i]['sample'] = sample_processed[flavor][ib]  # scopes[flavor][ sample_processed[flavor][ib] ]['sample']

        # FNs ~ TNs
        v = R[i, :][cells_fn[i, :]]  # v4
        if len(v) > 0: 
            flavor = 'fn'

            if len(v) > max_size_user: v = np.random.choice(v, min(max_size_user, len(v)))
            Nfn = len(v)     
            if Nfp > Nfn:   # match sample size of FP 
                v, _ = bootstrap_resample(v, n=Nfp); Nfn = Nfp  # usually n(fp) > n(fn) being positive class is the minority i.e. less TP-FN pairs
            # assert np.all(v < p_th[i]), "(polarity_feature_extraction) Found {}-inconsistent sample: {}".format(flavor, v[v >= p_th[i]][:50])

            scopes[flavor][i] = {'min': np.min(v), 'max': np.max(v), 'mean': np.mean(v), 'median': np.median(v), 
                'summary': common.five_number(v), 'sample': np.sort(v)}  

            # a, b, loc, scale = analyzer.fit_beta(v, name='FN', output_path=None, dpi=300, verbose=True, ext='pdf', index=0, save=True, color='blue')
            # scopes['fn'][i]['beta'] = np.array([a, b, loc, scale]) 

            ### KDE
            # kernels: epanechnikov, gaussian
            uname = U[i] if U is not None else i
            # rbn = random_bags(n_classifiers=n_uniq_users, bag_count=bag_count)  # random bag numbers
            ib = int(np.digitize(i, bins=bagging_bins))
            if not ib in kde_processed[flavor]: 
                # train only one KDEstimator per classifier (instead of per bagged classifier)

                pv = list(chain.from_iterable([R[u, :][cells_fn[i, :]] for u in range_bags(ib, n_classifiers=n_uniq_users, bag_count=bag_count)]))
                
                # KDE
                scopes[flavor][i]['kde'] = kde = analyzer.fit_kd2(pv, kernel=kernel, size=max_size_kde, cv=20, name="{}-{}".format(flavor.upper(), base_name(uname)), 
                    output_path=None, dpi=300, verbose=ib==1, ext='pdf', save=True)  # use bag_number
                kde_processed[flavor][ib] = kde

                if tAgg: 
                    # downsampling?
                    # if max_size_user > 0: x = np.random.choice(pv, min(max_size_user, len(x)))
                    scopes[flavor][i]['sample'] = sample_processed[flavor][ib] = np.sort(pv)
            else: 
                scopes[flavor][i]['kde'] = kde_processed[flavor][ib] # used the already computed KD estimator
                if tAgg: scopes[flavor][i]['sample'] = sample_processed[flavor][ib]  # scopes[flavor][ sample_processed[flavor][ib] ]['sample']

        if i % 10 == 0: print("(polarity_matrix) {}-th classifier | n(tp): {}, n(tn): {}, n(fp): {}, n(fn): {}".format(i, Ntp, Ntn, Nfp, Nfn))
            
    ### end foreach classifier/user index (i)

    # [test]
    for flavor in sample_types: 
        # this may not hold true when there's insufficient flavored sample size to begin with
        assert len(kde_processed[flavor]) == n_classifiers, "n(kde_processed[{}]): {}, n_classifiers: {}".format(flavor, len(kde_processed[flavor]), n_classifiers)

    msg = '(polarity_feature_extraction) K-S tests on hard cases: 1) TP vs FP 2) TN vs FN -- 3) TP vs FN, 4) TN vs FP\n'
    for flavor, entry in kde_processed.items(): 
        if flavor == 'tp':  # how does it compare to fp? 
            for ib in entry.keys(): 
                tps = scopes[flavor][ib]['sample']
                fps = scopes['fp'][ib]['sample']
                kss = ks_2samp(tps, fps)
                msg += "... TP vs FP | n(tp): {}, n(fp): {} | KS stat: {}, p-value: {}\n".format(len(tps), len(fps), kss.statistic, kss.pvalue)
        elif flavor == 'tn': 
            for ib in entry.keys(): 
                tns = scopes[flavor][ib]['sample']
                fns = scopes['fn'][ib]['sample']
                kss = ks_2samp(tns, fns)
                msg += "... TN vs FN | n(tn): {}, n(fn): {} | KS stat: {}, p-value: {}\n".format(len(tns), len(fns), kss.statistic, kss.pvalue)
        else: 
            continue
    print(msg)

    # [test]
    msg = '(polarity_feature_extraction) Test | probability values\n'
    for st in sample_types: 
        msg += '\n--- Sample Type: {} ---\n'.format(st.upper())
        for i in range(R.shape[0]):  # foreach classifier
            if i % 2 == 0: 
                uname = U[i] if U is not None else i
                if i in scopes[st] and (len(scopes[st][i]) > 0): 
                    msg += '... {}-{} | min: {}, max: {} | mean: {}, median: {}\n'.format(st.upper(), base_name(uname), scopes[st][i]['min'], scopes[st][i]['max'], scopes[st][i]['mean'], scopes[st][i]['median'])
    print(msg)

    return scopes

def polarity_sequence_model(R, Lr, p_th, T, Lt=None, C=None, U=None, p_classifier='crf', constrained=True, 
    pos_label=1, neg_label=0, bag_count=10, fold_count=5, index=0): 
    pass

def define_outliers(R, p_th): 
    """

    Memo
    ----
    1. outiler detection in sklearn

       https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_housing.html#sphx-glr-auto-examples-applications-plot-outlier-detection-housing-py
    
    """
    Ro = {}
    vmin, vmax = 0.01, 0.99
    for i in range(R.shape[0]): 

        # A. label-specific 
        pv_low = R[i, :][R[i, :] < p_th[i]]
        pv_high = R[i, :][R[i, :] >= p_th[i]] 

        Ro[i] = [0.0, 1.0]  # default
        if len(pv_low) > 0 and len(pv_high) > 0: 
            q1_low = np.percentile(pv_low, 25)
            q3_low = np.percentile(pv_low, 75)

            q1_high = np.percentile(pv_high, 25)
            q3_high = np.percentile(pv_high, 75)

            iqr_low = q3_low - q1_low
            iqr_high = q3_high - q3_high
            # Ro[i] = [q1_low-0.5*iqr_low, q3_high+0.5*iqr_high]
             
            # B. 
            # q3, q1 = np.percentile(R[i, :], [75 ,25])
            # iqr = q3-q1
            bl = max(vmin, q1_low-1.5*iqr_low)  # >= vmin
            bh = min(vmax, q3_high+1.5*iqr_high)  # <= vmax
            Ro[i] = [bl, bh]

        if i % 10 == 0: print("(define_outliers) outlier bound | classifier #{} | {}".format(i, Ro[i]))

    return Ro

def polarity_modeling(R, Lr, p_th, T, Lt=None, C=None, U=None, policy='sequence', p_classifier='rf', constrained=True, 
    pos_label=1, neg_label=0, bag_count=10, fold_count=5, index=0): 
    """
    
    Params
    ------
    Lt: test labels; used only for testing and evaluation

    Reference 
    ---------
    1. sorting 2D arrays 
       https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html

    2. KDE (see analyzer.py)

    3. CRF 
        a. http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/
        
           trainer = pycrfsuite.Trainer(verbose=True)

        b. CRFsuite tutorial 
        
           http://www.chokkan.org/software/crfsuite/

        c. hyperparameter optimization 

           https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb  

    4. categorical variables 

       a. https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63 
    """
    def is_within(v, scope): 
        if not scope: return False
        return (v >= scope['min']) and (v <= scope['max'])
    def is_within_or_above(v, scope): 
        if not scope: return False
        return ((v >= scope['min']) and (v <= scope['max'])) or (v > scope['max'])
    def is_within_or_below(v, scope): 
        if not scope: return False
        return ((v >= scope['min']) and (v <= scope['max'])) or (v < scope['min'])
    def is_above(v, scope, k='mean'):
        if not scope: return False
        return v >= scope[k]
    def is_below(v, scope, k='mean'):
        if not scope: return False
        return v <= scope[k]  
    def get_vars_query(q, p_th): 
        pass
    def get_bins(n_classifiers=5, bag_count=10):
        # e.g. n_classifiers = 5 
        #      => [0, 10, 20, 30, 40, 50]
        return [i * bag_count for i in range(n_classifiers+1)]
    def range_bags(bn, n_classifiers=5, bag_count=10):  # given
        # bn: bin number starting from 1
        assert bn <= n_classifiers and bn >= 1
        imax = bn * bag_count
        return range(imax-bag_count, imax)

    def random_bags(n_classifiers=5, bag_count=10): 
        bns = []  # bag number sequence
        for e in get_bins(n_classifiers)[:-1]:
            bns.append( np.random.randint(e, e+bag_count,1)[0] )
        return bns
    def numeric_to_str(labels, codes={}): 
        if len(codes) == 0: codes = Polarity.codes
        
        # inverse the codes 
        inv_codes = {num: stype for stype, num in codes.items()}
        return [inv_codes[l] for i, l in enumerate(labels)]
    def str_to_numeric(labels, codes={}): # 
        if len(codes) == 0: codes = Polarity.codes
        return [codes[l] for i, l in enumerate(labels)]
    def to_numeric(Yh, codes={}): 
        if len(codes) == 0: codes = Polarity.codes
        
        Yhn = []
        for y in Yh: # foreach label sequence/list
            Yhn.append( [codes[e] for e in y] )
        return Yhn
    def serialize_labels(Yh, to_numeric=True, codes={}):
        if len(codes) == 0: codes = Polarity.codes
        
        lh = []
        if to_numeric: 
            for yj in Yh: # foreach label sequence/list
                lh.extend( [codes[e] for e in yj] )  
        else: 
            for yj in Yh: # foreach label sequence/list
                lh.extend( yj )
        return lh
    def base_name(cls_name, prefix='', sep='.'):
        bn = 0
        cn, *x = cls_name.split(sep)
        if prefix: cn = "{}-{}".format(prefix, cn) 
        if len(x) > 0: bn = x[0]
        return cn, bn  

    # import collections
    from numpy import percentile
    from itertools import chain
    import analyzer
    from sampling import bootstrap_resample
    from sklearn.neighbors import KernelDensity
    from sklearn.metrics import mean_squared_error, classification_report
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier   # [test]
    from classifier_util import optimize_crf_params
    # from boruta import BorutaPy
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

    # resampling
    from imblearn.pipeline import make_pipeline
    from imblearn.under_sampling import AllKNN, NeighbourhoodCleaningRule
    from imblearn.under_sampling import NearMiss # doctest: +NORMALIZE_WHITESPACE
    from imblearn.combine import SMOTETomek # doctest: +NORMALIZE_WHITESPACE 

    np.set_printoptions(precision=3)
    sample_types = Polarity.sample_types # ['tp', 'tn'] + ['fp', 'fn']
    codes = Polarity.codes
    # codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
    #         'unk': 0, 't': 3, 'f': -3, 
    #         'pos': 1, 'neg': -1, '+': 1, '-': -1}

    n_users = R.shape[0]

    # other parameters 
    # bag_count = 10
    classifiers = ['u{}'.format(i) for i in range( int(n_users/bag_count) )]
    if U is not None: classifiers, bag_count = common.infer_bag_count(U, sep='.', verify=False)
    n_classifiers = n_uniq_users = len(classifiers)
    print("(polarity_modeling) policy: {}, n_classifiers: {}\n... {}\n".format(policy, n_classifiers, classifiers))

    scopes = polarity_feature_extraction(R, Lr, p_th, T, U=U, max_size_kde=1000, 
        bag_count=bag_count, fold_count=fold_count, index=index)
    # tConstrained = True if constrained and ((k_upper > 0) or (k_lower > 0)) else False
    # now scan through T while looking up table scope to determine if entries in T should be considered as positive or negative or neutral/unknown
    
    M = np.zeros(T.shape)
    M_bar = None  # not all methods return M_bar
    if policy in ('mean', 'median', ): 
        M_bar = np.zeros(T.shape)

        for j in range(T.shape[1]):  # foreach item/datum
            for i in range(T.shape[0]):  # foreach user/classifier
                # positive? 
                is_tp = is_above(T[i, j], scopes['tp'][i], k=policy) if i in scopes['tp'] else False    # if the proba value is greater than the mean or median of TPs, then it is likely a TP example
                is_tn = is_below(T[i, j], scopes['tn'][i], k=policy) if i in scopes['tn'] else False

                ########## 
                # ... depending on the labeling, they can be just as equally likely to be opposite 
                is_fp = is_above(T[i, j], scopes['fp'][i], k=policy) if i in scopes['fp'] else False
                is_fn = is_below(T[i, j], scopes['fn'][i], k=policy) if i in scopes['fn'] else False

                ########## 
                # ... now suppose that the label is positive
                Li = 1
                if is_tp != is_fn: 
                    M[i, j] = 1 if is_tp else -1
                elif is_tp and is_fn: 
                    M[i, j] = 0   # ambiguous
                else: # neither
                    # Li = 0?
                    M[i, j] = 0  # but shouldn't be possible

                ##########
                # ... suppose that the label is negative
                Li = 0 
                if is_tn != is_fp:  
                    M_bar[i, j] = 1 if is_tn else -1
                elif is_tn and is_fp: 
                    M_bar[i, j] = 0   # ambiguous
                else: # neither
                    # Li = 1?
                    M_bar[i, j] = 0

    elif policy.startswith('interv'):  # interval 
        M_bar = np.zeros(T.shape)

        for j in range(T.shape[1]): 
            for i in range(T.shape[0]):

                # positive? 
                is_tp = is_within_or_above(T[i, j], scopes['tp'][i]) if i in scopes['tp'] else False
                is_tn = is_within_or_below(T[i, j], scopes['tn'][i]) if i in scopes['tn'] else False

                is_fp = is_within_or_above(T[i, j], scopes['fp'][i]) if i in scopes['fp'] else False
                is_fn = is_within_or_below(T[i, j], scopes['fn'][i]) if i in scopes['fn'] else False

                if is_tp != is_tn: # only one of them is true
                    M[i, j] = codes['tp'] if is_tp else codes['tn']

                ########## 
                # ... now suppose that the label is positive
                Li = 1
                if is_tp != is_fn: 
                    M[i, j] = 1 if is_tp else -1
                elif is_tp and is_fn: 
                    M[i, j] = 0   # ambiguous
                else: # neither
                    # Li = 0?
                    M[i, j] = 0
                ##########
                # ... suppose that the label is negative
                Li = 0 
                if is_tn != is_fp:  
                    M_bar[i, j] = 1 if is_tn else -1
                elif is_tn and is_fp: 
                    M_bar[i, j] = 0   # ambiguous
                else: # neither
                    # Li = 1?
                    M_bar[i, j] = 0

    elif policy.startswith('seq'):  # sequence model 
        import sklearn_crfsuite
        from sklearn_crfsuite import scorers
        from sklearn_crfsuite import metrics

        tSubset = True
        
        tBootstrap = False  # note: boostrapping causes degradation in precision (but increases recalls) => tends to produce more FPs, which seems to always reduce fmax performance
        bootstrap_factor = 10   # generate this many extra examples for the positive: (n_neg-n_pos)//bootstrap_factor
        tResample = tResamplePrior = True
        tResamplePost = False

        tMulticlass = True   # note: binary class mode does not work well
        tFilterOutlier = True
        tStandardize = False
        tModelSelection = True
        print("(polarity_modeling) algorithm setting: {} | subset(majority)? {}, [ tResamplePrior: {}, tResamplePost: {} ], multiclass? {}, standardize? {}, ms? {}".format(policy, 
            tSubset, tResamplePrior, tResamplePost, tMulticlass, tStandardize, tModelSelection))

        # ... too low is not good
        gamma = 1.5  # scale for the minority class
        save_data = False

        min_size, max_size = 1000, 4000
        max_size_ms = 3000   # max sample size for model selection
        # create training examples

        # Initialization
        #########################################################

        msg = ''
        # define (probabilty) bounds delineating outliers for each classifier 
        Ro = define_outliers(R, p_th=p_th)

        #########################################################
        # ... resampling to balance the classes
        if tResamplePrior: 
            msg += '... original dataset shape: %s\n' % collections.Counter(Lr)
            Xr = R.T

            resampling_method = 'SMOTETomek'
            msg += "... resampling method applied (prior to polarity sample generation): {}\n".format(resampling_method)

            rs = int(time.time()+random.randint(1, 1000)+index)
            sampler = SMOTETomek(random_state=rs)  # sampling_stategy: 'auto' by default, resample all classes but the majority class
            Xr, Lr = sampler.fit_resample(Xr, Lr)

            R = Xr.T
            msg += '... resampled dataset shape: %s\n' % collections.Counter(Lr)
        #########################################################
        print(msg)

        # top k 
        ret = classPrior(Lr, labels=[0, 1], ratio_ref=0.1, verbose=False)
        r_min = ret['r_min']

        # subsample the negative to match sample size of the positive
        pos_sample = np.where(Lr == pos_label)[0]  # <<< 
        neg_sample = np.where(Lr == neg_label)[0]
        n_neg = len(neg_sample)
        n_pos = len(pos_sample)

        if tSubset: 
            if n_neg > max_size: 
                neg_sample = np.random.choice(neg_sample, max_size, replace=False)
                n_neg = len(neg_sample)
            
            # ... perhaps do not down sample minority class
            if n_pos > max_size: 
                pos_sample = np.random.choice(pos_sample, max_size, replace=False)
                n_pos = len(pos_sample)

        Ra = None  # augmented ratings/probabilities
        if tBootstrap: 
            # if n_pos < n_neg:   # bootstrap minority class
            #     n_delta = (n_neg - n_pos)//bootstrap_factor   # full bootstrap usually does not help
            #     Ra, Ca = polarity_sample_bootstrap(R, C=C, p_th=p_th, Lr=Lr, p_model=scopes, n_samples=n_delta, Ro=Ro, target_label=pos_label)
            #     Lpos = np.repeat(pos_label, n_delta)

            #     Lr = np.hstack([Lr, Lpos])
            #     R = np.hstack([R, Ra])
            #     C = np.hstack([C, Ca])

            #     pos_sample = np.where(Lr == pos_label)[0]
            #     n_pos = len(pos_sample)
                # assert n_pos == n_neg
            # ... this will not affect the indices of the negative examples because we are padding only positive examples in the back R | Ra(+)
            pass 
        else: 
            # make the dataset more balanced
            neg_sample = np.random.choice(neg_sample, min(n_neg, n_pos * gamma), replace=False)
            n_neg = len(neg_sample)
        # ... now (neg_sample, pos_sample) are determined

        Mc, Lh = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
        Lhr = Lh # estimateLabelMatrix(R, p_th=p_th)
        #########################################################
        # ... R -> R', C -> C', Lr -> Lr' 

        # [test]
        #########################################################
        predict_pos = (Lh == pos_label)  # Given BP's prediction Lh, select entries ~ target label
        predict_neg = (Lh == neg_label)
        cells_tp = (Mc == 1) & predict_pos   # estimated
        cells_tn = (Mc == 1) & predict_neg
        cells_fp = (Mc == 0) & predict_pos
        cells_fn = (Mc == 0) & predict_neg
        Ntp = np.sum(cells_tp)
        Ntn = np.sum(cells_tn)
        Nfp = np.sum(cells_fp)
        Nfn = np.sum(cells_fn)
        print("(polarity_modeling) Sample statistics after sampling process | n(+): {}, n(-): {} | n(TP): {}, n(TN): {}, n(FP): {}, n(FN): {} | Nu: {}, Ni: {}".format(
            np.sum(Lh == pos_label), np.sum((Lh == neg_label)), Ntp, Ntn, Nfp, Nfn, R.shape[0], R.shape[1]))
        #########################################################

        n_users = nbp = R.shape[0]
        n_train = n_items = R.shape[1]
        k = int(R.shape[0]/2)   

        Cr = Ct = None
        if C is not None: 
            Cr, Ct = C[:, :n_train], C[:, n_train:]

            if tResamplePrior: 
                # need to recompute confidence score ... todo: include other methods
                Wu = confidence_brier(R, Lr, mode='user')
                Wi = confidence_brier(R, Lr, mode='item')
                Cr = np.outer(Wu, Wi)

                Wu = confidence_brier(T, Lt, mode='user')
                Wi = confidence_brier(T, Lt, mode='item')
                Ct = np.outer(Wu, Wi)

            assert Cr.shape == R.shape, "dim(Cr): {}, dim(R): {}".format(Cr.shape, R.shape)
            assert Ct.shape == T.shape 

        # Lht = estimateLabelMatrix(T, p_th=p_th)
        n_polarity_pos =  n_polarity_neg = 0  # note: this is not the same as 'n_pos': number of positive-class examples

        Xset, yset = [], [] # fset
        nTP = nTN = nFP = nFN = 0

        # ranked version of R 
        # Rm = rank(R, method='average')
        Rs = np.sort(R, axis=1)

        polarity_labels = ['tp', 'tn', 'fp', 'fn'] if tMulticlass else ['+', '-', ]
        seq_type = 'TN-FP-seq'
        for j in neg_sample: # foreach negative-class example 

            # get feature repr for j-th item by varying user (i) while holding item (j) fixed
            fseq = get_feature_sequence(R, j, p_th, Rm=Rs, C=Cr, U=U, p_model=scopes, name=seq_type, index=index, verbose=False, wsize=20) # feature sequence
            # ... a list of feature dictionaries
            Xset.append(fseq)

            # each element of the label sequence is either a TN or an FP
            ls_j = []  # label sequence for j-th item
            for i in range(n_users):  # foreach user/classifeir
                if Mc[i, j] == 1: 
                    polarity = '+'
                    label = 'tn' if tMulticlass else polarity # codes['tn'] if tMulticlass else polarity
                elif Mc[i, j] == 0: 
                    polarity = '-'
                    label = 'fp' if tMulticlass else polarity # codes['fp'] if tMulticlass else polarity
                else: 
                    raise ValueError
                ls_j.append(label)

            assert len(fseq) == len(ls_j)
            yset.append(ls_j)

        #########################

        seq_type = 'TP-FN-seq'
        for j in pos_sample:  # foreach positive-class example

            fseq = get_feature_sequence(R, j, p_th, Rm=Rs, C=Cr, U=U, p_model=scopes, name=seq_type, index=index, verbose=False, wsize=20) # feature sequence
            Xset.append(fseq)

            # each element of the label sequence is either a TP or an FN
            ls_j = []  # label sequence for j-th item
            for i in range(n_users):  # foreach user/classifeir
                if Mc[i, j] == 1: 
                    polarity = '+'
                    label = 'tp' if tMulticlass else polarity # codes['tp'] if tMulticlass else polarity
                elif Mc[i, j] == 0: 
                    polarity = '-'
                    label = 'fn' if tMulticlass else polarity # codes['fn'] if tMulticlass else polarity
                else: 
                    raise ValueError
                ls_j.append(label)

            assert len(fseq) == len(ls_j)
            yset.append(ls_j)

        # now train a sequence classifier
        model = sklearn_crfsuite.CRF(
            algorithm='lbfgs', 

            c1=0.5,  # coefficient for L1 penalty
            c2=0.02,  # coefficient for L2 penalty
            max_iterations=300, 
            all_possible_transitions=True
        )

        # 1a. feature transformation 
        #############################################
        scaler = None
        if tStandardize: 
            Xset, scaler = standardize(Xset, target_vars=['ks.pvalue', ])

        msg = ""

        # 1b. model fitting
        #############################################
        if tModelSelection: 
            model = optimize_crf_params(Xset, yset, model, labels=polarity_labels, max_size=max_size_ms, verfiy=True)
        
        model.fit(Xset, yset) # remember to take transpose
        msg += "(polarity_modeling) labels: {}\n".format( list(model.classes_) )
        
        polarity_labels = list(model.classes_)
        sorted_labels = sorted(polarity_labels, key=lambda x: (x[1:], x[0]))

        # [test]
        Xset, yset = [], []  # reset X and y
        Mct, Lht = probability_filter(T, Lt, p_th)  # Mc is a (0, 1)-matrix
        # Lht = estimateLabelMatrix(T, p_th=p_th) 
        # ... Lht does not require true labels
        Ts = np.sort(T, axis=1) # rank(T, method='average')

        test_j = np.random.choice(range(T.shape[1]), 10)
        for j in range(T.shape[1]):

            # get feature repr for j-th item by varying user (i) while holding item (j) fixed
            fseq = get_feature_sequence(T, j, p_th, Rm=Ts, C=Ct, U=U, p_model=scopes, name='predict-T', index=index, verbose=False, wsize=20) # feature sequence
            # ... a list of feature dictionaries
            Xset.append(fseq)  # Xset[j] -> feature sequence for the j-th column/item

            # [test]
            ls_j = []  # label sequence for j-th item
            for i in range(n_users):  # foreach user/classifeir
                ################################################################################
                if Lht[i, j] == pos_label and Mct[i, j] == 1: 
                    ls_j.append('tp' if tMulticlass else '+' )
                elif Lht[i, j] == pos_label and Mct[i, j] == 0:
                    ls_j.append('fp' if tMulticlass else '-' )  # 'fp'
                elif Lht[i, j] == neg_label and Mct[i, j] == 1:
                    ls_j.append('tn' if tMulticlass else '+'  )
                elif Lht[i, j] == neg_label and Mct[i, j] == 0:
                    ls_j.append('fn' if tMulticlass else '-' ) # 'fn'
                ################################################################################
            # [test]
            assert len(fseq) == len(ls_j) == T.shape[0], \
                "size(feature seq): {}, size(label seq): {}, n_classifiers: {}".format(len(fseq), len(ls_j), T.shape[0])
            yset.append(ls_j)

        assert set(model.classes_) == set(polarity_labels)

        # prediction on T
        ################################################################################
        Xset = transform(Xset, scaler)
        y_pred = model.predict( Xset )  # transform(Xset, scaler)
        ################################################################################

        # convert to np.array format 
        for j, yj in enumerate(y_pred): 
            if j == 0: assert len(yj) == T.shape[0]

            # yj: sequence/list of strings as labels
            M[:, j] = str_to_numeric(yj) 

        y_test = yset
        f1 = metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=polarity_labels)
        msg += "(polarity_modeling) flat F1 score on T: {}\n".format(f1)
        msg += metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3) + '\n'

        ################################
        atype = 'micro'  # 'macro'   
        ys_test = serialize_labels(yset, to_numeric=False)
        ys_pred = serialize_labels(y_pred, to_numeric=False)
        # ... micro averaging is useful when classes different in sizes
        m_auc = common.multiclass_roc_auc_score(ys_test, ys_pred, average=atype) # Multioutput target data is not supported with label binarization
        msg += "(polarity_modeling) overall {} AUC on T: {}\n".format(atype, m_auc)

        # ... M determined 
        #     M_bar = None
        Po_t = M   # if just binary {-1, 1}, then the job is done 
        if tMulticlass: 
            # convert to polarity repr {-1, 0, 1} for testing
            Po_t = from_color_to_polarity(M, codes, verify=True)
        npol = len(np.unique(Po_t)); assert npol == 2, "Expecting 2 polarities but got {}".format(npol)
        # ... Po_t in {-1, 1}-repr 

        print(msg)

        # [test] evaluate preference score predictions
        msg = ""
        if Lt is not None: 
            assert len(Lt) == T.shape[1] 
            # Mct, Lht = probability_filter(T, Lt, p_th)  # Mc is a (0, 1)-matrix
            # ... correctness and label matrix using true labels Lt
            ret = eval_polarity(Po_t, Mct, Lht, verbose=True, name='T', neg_po=-1, title='(polarity_modeling) -- T given Po_t --')

            # fmax metric for the estimated polarity matrix (M)
            msg += '(polarity_modeling) Compare estimated polarity matrix (M) with majority vote-induced preference matrix ...\n'
            pvt_max = predict_by_importance_weights(T, to_preference(Po_t), aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
            fmax_t = common.fmax_score(Lt, pvt_max, beta = 1.0, pos_label = 1)
            msg += "... fmax(Po_t): {}\n".format(fmax_t)
            # ... polarity performance of M 

            # how does it compare to majority votes? 
            # ... to_polarity(lh, Lh)
            ############################################################
            name = 'T'
            lh = estimateLabels(T, p_th=p_th, pos_label=pos_label) 
            Mct_max, Lht_max = probability_filter(T, lh, p_th)  # Mc is a (0, 1)-matrix
            
            # how many entries are different compared to True polarities? 
            n_agreed = np.sum(Mct_max == Mct)
            n_agreed_labeling = np.sum(Lt == lh)
            msg += '... {}max | n_agreed: {}, ratio: {} | lh vs Lt? n_agreed: {}, ratio: {}\n'.format(name, n_agreed, n_agreed/(Mct_max.shape[0] * Mct_max.shape[1]+0.0), 
                n_agreed_labeling, n_agreed_labeling/(len(lh)+0.0))
            pvt_max = predict_by_importance_weights(T, Mct_max, aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
            fmax_t = common.fmax_score(Lt, pvt_max, beta = 1.0, pos_label = 1)
            msg += "... fmax({}, majority): {}\n".format(name, fmax_t)

            # Mct_max = (Lht == lh).astype(int)  # use estimated labels and label matrix to compute polarity matrix
            ret = eval_polarity(preference_to_polarity(Mct_max), Mct, Lht, verbose=True, name='Tmax', neg_po=-1, title='(polarity_modeling) -- {}: majority votes --'.format(name))
            # ... majority vote (T)   
            ############################################################
        print(msg)

        # [test] pretend to predict the training data
        msg = "" 
        Xset, yset = [], []
        Mcr, Lhr = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
        Mr = np.zeros(R.shape)
        Rm = np.sort(R, axis=1) # rank(R, method='average')

        for j in range(R.shape[1]):
            # get feature repr for j-th item by varying user (i) while holding item (j) fixed
            fseq = get_feature_sequence(R, j, p_th, Rm=Rs, C=Cr, U=U, p_model=scopes, name='predict-R', index=index, verbose=False, wsize=20) # feature sequence
            # ... a list of feature dictionaries
            Xset.append(fseq)  # Xset[j] -> feature sequence for the j-th column/item

            ls_j = []  # label sequence for j-th item
            for i in range(n_users):  # foreach user/classifeir
                ################################################################################
                if Lhr[i, j] == pos_label and Mcr[i, j] == 1: 
                    ls_j.append('tp' if tMulticlass else '+' )
                elif Lhr[i, j] == pos_label and Mcr[i, j] == 0:
                    ls_j.append('fp' if tMulticlass else '-' )  # 'fp'
                elif Lhr[i, j] == neg_label and Mcr[i, j] == 1:
                    ls_j.append('tn' if tMulticlass else '+'  )
                elif Lhr[i, j] == neg_label and Mcr[i, j] == 0:
                    ls_j.append('fn' if tMulticlass else '-' ) # 'fn'
                ################################################################################
            # [test]
            assert len(fseq) == len(ls_j) == R.shape[0], \
                "size(feature seq): {}, size(label seq): {}, n_classifiers: {}".format(len(fseq), len(ls_j), R.shape[0])
            yset.append(ls_j)

        # prediction on R
        ################################################################################
        Xset = transform(Xset, scaler)
        y_pred = model.predict( Xset )  # transform(Xset, scaler)
        ################################################################################

        # convert to np.array format 
        for j, yj in enumerate(y_pred): 
            if j == 0: assert len(yj) == R.shape[0]

            # yj: sequence/list of strings as labels
            Mr[:, j] = str_to_numeric(yj)

        y_test = yset
        f1 = metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=polarity_labels)
        msg += "(polarity_modeling) flat F1 score on R: {}\n".format(f1)
        msg += metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3) + '\n'
        print(msg)
        
        msg = ""
        Po_r = Mr
        if tMulticlass: 
            # maps tp, tn to 1; fp, fn to -1
            Po_r = from_color_to_polarity(Mr, codes, verify=True)
        npol = len(np.unique(Po_r)); assert npol == 2, "Expecting 2 polarities but got {}".format(npol)
        
        ret = eval_polarity(Po_r, Mcr, Lhr, verbose=True, name='R', neg_po=-1, title="(polarity_modeling) -- R given Po_r --")

        # Py1_x = model.predict_proba(scaler.transform(Xset))[:, 1]
        atype = 'micro'  # 'macro'   
        # ... micro averaging is useful when classes different in sizes
        ys_test = serialize_labels(yset, to_numeric=False)
        ys_pred = serialize_labels(y_pred, to_numeric=False)
        m_auc = common.multiclass_roc_auc_score(ys_test, ys_pred, average=atype)
        msg += "(polarity_modeling) overall {} AUC on R: {}\n".format(atype, m_auc)

        name = 'R'
        lh = estimateLabels(R, p_th=p_th, pos_label=pos_label) 
        Mcr_max, Lhr_max = probability_filter(R, lh, p_th)  # Mc is a (0, 1)-matrix
        # ... correctness and labeling matrix obtained via majority vote
        
        # how many entries are different compared to True polarities? 
        n_agreed = np.sum(Mcr_max == Mcr)
        n_agreed_labeling = np.sum(Lr == lh)
        msg += '... {}max | n_agreed: {}, ratio: {} | lh vs Lr? n_agreed: {}, ratio: {}\n'.format(name, n_agreed, n_agreed/(Mcr_max.shape[0] * Mcr_max.shape[1]+0.0), 
            n_agreed_labeling, n_agreed_labeling/(len(lh)+0.0))
        pvr_max = predict_by_importance_weights(R, Mcr_max, aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
        fmax_r = common.fmax_score(Lr, pvr_max, beta = 1.0, pos_label = 1)
        msg += "... fmax({}, majority): {}\n".format(name, fmax_r)

        # if we were to use preference matrix as a polarity matrix
        ret = eval_polarity(preference_to_polarity(Mcr_max), Mcr, Lhr, verbose=True, name='Rmax', neg_po=-1, title='(polarity_modeling) -- {}: majority votes --'.format(name))  
        # ... majority vote (R)   
        print(msg) 

    elif policy.startswith('class'):  # polarity classification
        import stacking
        from sklearn import preprocessing
        from sklearn_crfsuite import metrics  
        # from imblearn.pipeline import make_pipeline
        # from imblearn.under_sampling import AllKNN, NeighbourhoodCleaningRule
        # from imblearn.under_sampling import NearMiss # doctest: +NORMALIZE_WHITESPACE
        # from imblearn.combine import SMOTETomek # doctest: +NORMALIZE_WHITESPACE 

        # variables: 
        # top k proba in + and bottom k proba in - + tp-tn statistics
        tMulticlass = True
        tFilterOutlier = True
        tAdditionalVars = True
        tSubset = True
        
        tBootstrap = False
        bootstrap_factor = 10   # generate this many extra examples for the positive: (n_neg-n_pos)//bootstrap_factor
        tResample = tResamplePrior = True  # apply resampling prior to polarty sample generation (used to balance class samples)
        tResamplePost = False  # apply resampling after polarity sample generation (used to balance polarity samples in terms of TP, TN, FP, FN)
        # tBigData = True

        tStandardize = True
        tModelSelection = False

        print("(polarity_modeling) algorithm setting: {} | subset(majority)? {}, [ tResamplePrior: {}, tResamplePost: {} ], multiclass? {}, standardize? {}, ms? {}".format(policy, 
            tSubset, tResamplePrior, tResamplePost, tMulticlass, tStandardize, tModelSelection))
 
        # ... too low is not good
        gamma = 1.5  # scale for the minority class
        save_data = False
        max_size = 3000
        # create training examples
        
        # Initialization
        #########################################################

        
        # ... define (probabilty) bounds for outliers for each classifier 
        Ro = define_outliers(R, p_th=p_th)

        # top k 
        ret = classPrior(Lr, labels=[0, 1], ratio_ref=0.1, verbose=False)
        r_min = ret['r_min']

        # subsample the negative to match sample size of the positive
        pos_sample = np.where(Lr == pos_label)[0]  # <<< 
        neg_sample = np.where(Lr == neg_label)[0]
        n_neg = len(neg_sample)
        n_pos = len(pos_sample)

        # subsetting the input training data
        # a. 
        if tSubset: 
            if n_neg > max_size: 
                neg_sample = np.random.choice(neg_sample, max_size, replace=False)
                n_neg = len(neg_sample)
            
            # ... perhaps do not down sample minority class
            if n_pos > max_size: 
                pos_sample = np.random.choice(pos_sample, max_size, replace=False)
                n_pos = len(pos_sample)

        Ra = None  # augmented ratings/probabilities
        if tBootstrap: 
            # [note] this will almost always reduce precision, which ultimately leads to a decrease in fmax performance

            # if n_pos < n_neg:   # bootstrap minority class
            #     n_delta = (n_neg - n_pos)//bootstrap_factor
            #     Ra, Ca = polarity_sample_bootstrap(R, C=C, p_th=p_th, Lr=Lr, p_model=scopes, n_samples=n_delta, Ro=Ro, target_label=pos_label)
            #     Lpos = np.repeat(pos_label, n_delta)

            #     Lr = np.hstack([Lr, Lpos])
            #     R = np.hstack([R, Ra])
            #     C = np.hstack([C, Ca])

            #     pos_sample = np.where(Lr == pos_label)[0]
            #     n_pos = len(pos_sample)
            #     # assert n_pos == n_neg
            # ... this will not affect the indices of the negative examples because we are padding only positive examples in the back R | Ra(+)
            pass
        else: 
            # make the dataset more balanced
            neg_sample = np.random.choice(neg_sample, min(n_neg, int(n_pos * gamma)), replace=False)
            n_neg = len(neg_sample)
        # ... now (neg_sample, pos_sample) are determined

        msg = ''
        #########################################################
        # ... resampling to balance the classes
        if tResamplePrior: 
            msg += '... original dataset shape: %s\n' % collections.Counter(Lr)
            Xr = R.T

            resampling_method = 'SMOTETomek'
            msg += "... resampling method applied (prior to polarity sample generation): {}\n".format(resampling_method)

            rs = int(time.time()+random.randint(1, 1000)+index)
            sampler = SMOTETomek(random_state=rs)  # sampling_stategy: 'auto' by default, resample all classes but the majority class
            Xr, Lr = sampler.fit_resample(Xr, Lr)

            R = Xr.T
            msg += '... resampled dataset shape: %s\n' % collections.Counter(Lr)
        #########################################################
        print(msg)

        Mc, Lh = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
        Lhr = Lh # estimateLabelMatrix(R, p_th=p_th)
        #########################################################
        # ... R -> R', C -> C', Lr -> Lr' 
        
        n_users = nbp = R.shape[0]
        n_train = n_items = R.shape[1]
        k = int(R.shape[0]/2)   

        Cr = Ct = None
        if C is not None: 
            Cr, Ct = C[:, :n_train], C[:, n_train:]

            if tResamplePrior: 
                # need to recompute confidence score ... [todo] include other methods; see confidence2D()
                Wu = confidence_brier(R, Lr, mode='user')
                Wi = confidence_brier(R, Lr, mode='item')
                Cr = np.outer(Wu, Wi)

                Wu = confidence_brier(T, Lt, mode='user')
                Wi = confidence_brier(T, Lt, mode='item')
                Ct = np.outer(Wu, Wi)

            assert Cr.shape == R.shape, "dim(Cr): {}, dim(R): {}".format(Cr.shape, R.shape)
            assert Ct.shape == T.shape 

        # Lht = estimateLabelMatrix(T, p_th=p_th)
        n_polarity_pos =  n_polarity_neg = 0  # note: this is not the same as 'n_pos': number of positive-class examples

        Xset, yset = [], [] # fset
        nTP = nTN = nFP = nFN = 0

        # ranked version of R 
        # Rm = rank(R, method='average')
        Rm = Rs = np.sort(R, axis=1)

        # encode user/classifier 
        Uc = None
        if U is not None: 
            Uc = []
            uset = set()
            for i in range(n_users):
                bp, bn = base_name(U[i])
                # if not (bp in uset): 
                Uc.append(bp)
                uset.add(bp)
            u_encoder = preprocessing.LabelEncoder()
            Uc = u_encoder.fit_transform(Uc)
            # now each user/classifier has a numerical representation

            # apply one-hot encoding to the numerically encoded classifiers 
            u_ohe = preprocessing.OneHotEncoder()
            Ub = u_ohe.fit_transform(Uc[:, None])  # input to the encoder must be a 2D array
            # ... Ub is in a sparse matrix format

            print("(polarity_modeling) numerically encoded classifiers: {} | n(Uc): {}, dim(Ub): {}".format(u_encoder.classes_, 
                len(Uc), Ub.shape))

            assert len(np.unique(Uc)) == len(uset)
            assert Ub.shape[0] == len(U), "dim(Ub): {}, len(U): {}".format(Ub.shape, len(U))
        # ... U/users/classifiers now have numeric repr 

        for j in neg_sample: 
            pos_i = np.where(Mc[:, j] == 1)[0] # TNs
            neg_i = np.where(Mc[:, j] == 0)[0]  # FPs

            pos_i_sorted = np.argsort(R[:, j])  # low to high

            # polarity positive: TN examples
            polarity = 1
            npp = npn = 0
            if len(pos_i) > 0:  # foreach positive-polarity particle  
                flavor = 'TN'

                if tSubset: 
                    # use n(TP) to control n(TN)
                    pos_i = np.random.choice(pos_i, min(k, len(pos_i)), replace=False)  # "dropout" subsampling
                    # pos_i = [i for i in pos_i_sorted if i in pos_i][:k]  # top k lowest (low to high)

                test_pos_j = np.random.choice(pos_i, 10)  # [test]
                for i in pos_i:  # foreach TN examples
                    # assert (Lh[i, j] == neg_label) and (Mc[i, j] == 1), "flavor: {}".format(flavor)  # ... ok
                    tVerbose = True if (j in test_pos_j) and (npp < 2) else False

                    # if R[i, j] is an 'outlier' or of extreme values (e.g. 1.0, 0.0), then the statistics become less reliable
                    if tFilterOutlier and ((R[i,j] < Ro[i][0]) or (R[i,j] > Ro[i][1])):
                        # outlier
                        continue 
                    else: 

                        fvh = get_vars_hstats(R, i, j, p_th, C=Cr, Rm=Rm, U=U, Uc=Uc, encoder=u_encoder, p_model=scopes, r_min=r_min,  # ratio of minority class
                                name='{}-h'.format(flavor), index=index, verbose=tVerbose)
                        # fvv = get_vars_vstats(R, i, j, p_th, C=Cr, p_model=scopes, name='{}-v'.format(flavor), index=index)

                        label = codes['tn'] if tMulticlass else polarity
                        yset.append(label) 

                        # define feature vector
                        #########################
                        fv = fvh # np.hstack([fvh, fvv])
                        Xset.append( fv ) 
                        #########################

                        n_polarity_pos += 1
                        npp += 1
                        nTN += 1

            ### end if 

                # polarity negative: FP examples
                # introudce negative polarity examples only when pp examples exist
                polarity = -1
                if len(neg_i) > 0: 
                    flavor = 'FP'

                    if tSubset: 
                        npp_eff = int(npp * gamma)
                        neg_i = np.random.choice(neg_i, min(npp_eff, len(neg_i)), replace=False)  # subsampling
                        # neg_i = [i for i in pos_i_sorted if i in neg_i][:npp_eff]   # choose those that are relatively lower i.e. border cases that could have been negative 

                    for i in neg_i:
                        # assert (Lh[i, j] == pos_label) and (Mc[i, j] == 0), "flavor: {}".format(flavor)  # ... ok
                        tVerbose = True if (j in test_pos_j) and (npn < 2) else False

                        if tFilterOutlier and ((R[i,j] < Ro[i][0]) or (R[i,j] > Ro[i][1])):
                            # outlier
                            continue
                        else: 
                            fvh = get_vars_hstats(R, i, j, p_th, C=Cr, Rm=Rm, U=U, Uc=Uc, encoder=u_encoder, p_model=scopes, r_min=r_min,  # ratio of minority class
                                name='{}-h'.format(flavor), index=index, verbose=tVerbose)
                            # fvv = get_vars_vstats(R, i, j, p_th, C=Cr, p_model=scopes, name='{}-v'.format(flavor), index=index)

                            # define feature vector
                            #########################
                            fv = fvh # np.hstack([fvh, fvv])
                            Xset.append( fv )  
                            #########################

                            label = codes['fp'] if tMulticlass else polarity   # 'fp'
                            yset.append(label) 

                            n_polarity_neg += 1
                            npn += 1 
                            nFP += 1

            else: 
                # no positive polarity examples
                # ... then the entire column is ambiguous 
                pass

        #------------------------------------------------------------------------------------------- 
        # pos_sample = np.where(Lr == pos_label)[0]  # <<< 
        # n_pos = len(pos_sample)

        for j in pos_sample:  # foreach positive examples

            pos_i = np.where(Mc[:, j] == 1)[0] # TPs
            neg_i = np.where(Mc[:, j] == 0)[0]  # FNs

            pos_i_sorted = np.argsort(-R[:, j])  # high to low
 
            # polarity positive: TP examples
            polarity = 1
            # ... it's possible that we do not have positive-polarity examples 
            npp = npn = 0
            if len(pos_i) > 0: # positive polarity
                flavor = 'TP'
                 
                if tSubset: 
                    # pos_i = np.random.choice(pos_i, min(k, len(pos_i)), replace=False)  # subsampling
                    # pos_i = [i for i in pos_i_sorted if i in pos_i][:k]  # top k (high to low)
                    pass  # ... don't downisze TP examples

                test_pos_j = np.random.choice(pos_i, 20)  # [test]
                for i in pos_i:
                    # assert (Lh[i, j] == pos_label) and (Mc[i, j] == 1), "TP"   # ... ok
                    tVerbose = True if (j in test_pos_j) and (npp < 2) else False

                    if tFilterOutlier and ((R[i,j] < Ro[i][0]) or (R[i,j] > Ro[i][1])):
                        # outlier
                        continue
                    else:  

                        fvh = get_vars_hstats(R, i, j, p_th, C=Cr, Rm=Rm, U=U, Uc=Uc, encoder=u_encoder, p_model=scopes, r_min=r_min,  # ratio of minority class
                                name='{}-h'.format(flavor), index=index, verbose=tVerbose)
                        # fvv = get_vars_vstats(R, i, j, p_th, C=Cr, name='{}-v'.format(flavor), index=index)

                        label = codes['tp'] if tMulticlass else polarity
                        yset.append(label) 

                        # define feature vector
                        #########################
                        fv = fvh # np.hstack([fvh, fvv])
                        Xset.append( fv )  
                        #########################

                        n_polarity_pos += 1  # number of positive polarity exmaples overall
                        npp += 1 # number of positive polarity exmaples for j-th data point
                        nTP += 1  # number of TPs
            ### end if
                
                # if there's no correct prediction for this positive example, then it becomes ambiguous
                # ... consider negative-polarity examples only when positive-polarity examples exist for this data point j

                # polarity negative: FN examples
                polarity = -1
                if len(neg_i) > 0:  # negative polarity
                    flavor = 'FN'

                    if tSubset: 
                        # npp_eff = int(npp * gamma)
                        # neg_i = np.random.choice(neg_i, min(npp_eff, len(neg_i)), replace=False)  # subsampling
                        # neg_i = [i for i in pos_i_sorted if i in neg_i][:npp_eff]   # choose those that are higher i.e. border cases that could have been positive 
                        pass

                    for i in neg_i:  # foreach negative-polarity example
                        # assert (Lh[i, j] == neg_label) and (Mc[i, j] == 0), "flavor: {}".format(flavor)  # ... ok
                        tVerbose = True if (j in test_pos_j) and (npn < 2) else False

                        if tFilterOutlier and ((R[i,j] < Ro[i][0]) or (R[i,j] > Ro[i][1])):
                            # outlier
                            continue
                        else: 
                        
                            fvh = get_vars_hstats(R, i, j, p_th, C=Cr, Rm=Rm, U=U, Uc=Uc, encoder=u_encoder, p_model=scopes, r_min=r_min,  # ratio of minority class
                                      name='{}-h'.format(flavor), index=index, verbose=tVerbose)
                            # fvv = get_vars_vstats(R, i, j, p_th, C=Cr, p_model=scopes, name='{}-v'.format(flavor), index=index)

                            # define feature vector
                            #########################
                            fv = fvh # np.hstack([fvh, fvv])
                            Xset.append( fv )  
                            #########################

                            label = codes['fn'] if tMulticlass else polarity   # 'fn'
                            yset.append(label) 
                            
                            n_polarity_neg += 1  # number of positive polarity exmaples overall
                            npn += 1   # number of positive polarity exmaples for j-th data point
                            nFN += 1

            else: 
                # no positive polarity examples
                # ... then the entire column is ambiguous 
                pass

        #------------------------------------------------------------------------------------------- 
        # ... training data generation complete

        msg = ""
        X, y = np.array(Xset), np.array(yset)
        nf = n_features = X.shape[1]

        # 1a. feature transformation
        #############################################
        scaler = StandardScaler() # MinMaxScaler(), StandardScaler()
        X = scaler.fit_transform(X)
        #############################################

        # 2a. resampling to keep particle colors balanced
        #############################################
        if tResamplePost: 
            ver = 3
            resampling_method = 'NearMiss(v{})'.format(ver)
            sampler = NearMiss(version=ver)    # undersampling based on KNNs (version 3 is less affected by noise)
            X, y = sampler.fit_resample(X, y)
            msg += "(polarity_modeling) post-resampling method applied: {}\n".format(resampling_method)
        # ... does not seem to be effective to apply resampling here

        # assert X.shape[0] > len(pos_sample)+len(neg_sample), "sample size for polarity detection should be far greater than that of classification | n(polarity): {} >? n(cls): {}".format(X.shape[0], 
        #     len(pos_sample)+len(neg_sample))
        msg += "(polarity_modeling) number of features: {}\n".format(n_features)
        
        nX = X.shape[0]
        labels = np.unique(y) 

        msg += "(polarity_modeling) n(pos): {}, n(neg): {} | max: {}\n".format(n_pos, n_neg, max_size)
        msg += "...                 n_polarity(pos):{}, n_polarity(neg):{} | n(TP): {}, n(TN): {}, n(FP): {}, n(FN): {}\n".format(n_polarity_pos, n_polarity_neg, nTP, nTN, nFP, nFN)
        msg += "...                 n(tset): {} | dim(X): {}, dim(y): {}\n".format(nX, X.shape, y.shape)
        msg += "...                 n(label): {} | {}\n".format(len(labels), list(labels))

        n_polarity_sample = nTP+nTN+nFP+nFN
        assert n_polarity_sample == n_polarity_pos + n_polarity_neg
        prior_polarity = np.array([nTP/(n_polarity_sample+0.0), nTN/(n_polarity_sample+0.0), nFP/(n_polarity_sample+0.0), nFN/(n_polarity_sample+0.0)])
        msg += "...                 polarity prior | tp: {}, tn: {}, fp: {}, fn: {}\n".format(*prior_polarity)
        print(msg)

        # now train a classifier
        if tMulticlass: 
            # note: the smaller the C, the stronger the regularization

            # sampler = NeighbourhoodCleaningRule()  # still imbalanced but better 
            stacker = stacker0 = RandomForestClassifier(n_estimators=500, max_depth=8, bootstrap=True, random_state=0, class_weight='balanced')
            # stacker = make_pipeline(sampler, stacker0)
            
            # ... Boruta feature selection
            # stacker = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
            # LogisticRegression(penalty='l2', C=1, tol=1e-4, max_iter=200, solver='saga', multi_class='multinomial', class_weight='balanced')

            # candidates
            # LogisticRegression(penalty='l2', tol=1e-4, solver='sag', multi_class='multinomial', class_weight='balanced')
            # RandomForestClassifier(n_estimators=150, max_depth=8, bootstrap=True, random_state=0, class_weight='balanced')
            # QuadraticDiscriminantAnalysis(store_covariance=True)
            # ... variables are collinear
            # LinearDiscriminantAnalysis()
        else: 
            stacker = LogisticRegression(penalty='l2', C=0.1, tol=1e-4, solver='sag', class_weight='balanced') 
            # stacking.choose_classifier(p_classifier)  # e.g. log, enet, knn
        
        # 1b. model fitting
        #############################################
        model = stacker.fit(X, y) # remember to take transpose

        # model.support_
        #############################################
        # ... how well does it fit the data?
        X = y = Rs = None
        #------------------------------------------------------------------------------------------- 
        # ... training complete, now create test data 

        ###########################
        Xset, yset = [], []
        ###########################
        # M = np.zeros(T.shape)   # [note] defined upfront 

        # [test]
        Mct, Lht = probability_filter(T, Lt, p_th)  # Mc is a (0, 1)-matrix
        # Lht = estimateLabelMatrix(T, p_th=p_th) 
        # ... Lht does not require true labels
        Tm = np.sort(T, axis=1) # rank(T, method='average')

        test_j = np.random.choice(range(T.shape[1]), 10)
        for j in range(T.shape[1]):

            Xset_j = []
            npt = 0
            for i in range(T.shape[0]):   # foreach row in T[:, j]
                tVerbose = True if (j in test_j) and (npt < 2) else False

                fvh = get_vars_hstats(T, i, j, p_th, C=Ct, Rm=Tm, U=U, Uc=Uc, encoder=u_encoder, p_model=scopes, r_min=r_min,  # ratio of minority class
                            name='{}-horizontal'.format('predict-T'), index=index, verbose=tVerbose)
                # fvv = get_vars_vstats(T, i, j, p_th, C=Ct, p_model=scopes, name='{}-vertical'.format('predict-T'), index=index)

                # define feature vector
                #########################
                fv = fvh # np.hstack([fvh, fvv])
                Xset_j.append( fv )  
                npt += 1 
                #########################

                # [test]
                ################################################################################
                Xset.append(fv)
                if Lht[i, j] == pos_label and Mct[i, j] ==1: 
                    yset.append(codes['tp'])
                elif Lht[i, j] == pos_label and Mct[i, j] ==0:
                    yset.append(codes['fp'])  # 'fp'
                elif Lht[i, j] == neg_label and Mct[i, j] == 1:
                    yset.append(codes['tn'])
                elif Lht[i, j] == neg_label and Mct[i, j] == 0:
                    yset.append(codes['fn']) # 'fn'
                ################################################################################
            
            fv_tj = np.array(Xset_j)  # 2D array; feature vectors associated with a column
            if j < 10: 
                assert fv_tj.shape[0] == T.shape[0] and fv_tj.shape[1] == n_features, \
                    "column-with of fv | dim(fv_tj): {} =?= T.shape[0]: {} | dim(fv): {} =?= nf: {}".format(fv_tj.shape[0], T.shape[0], fv_tj.shape[1], n_features)

            # print("> fv <- T[:, j] | dim(fv): {}".format(fv.shape))
            # pvj = model.predict(fv) # model.predict_proba(fv)[:, 1]  # 1/foldCount worth of data

            # np.nan_to_num( scaler.transform(fv_tj) )
            pvj = model.predict( scaler.transform(fv_tj) )
            M[:, j] = pvj   # predict polarity column by column (item by item)
        ### end foreach test datum/item

        # [test] #1. AUC and F1
        # -------------------------------------------------------------------
        msg = ""
        # Py1_x = model.predict_proba(scaler.transform(Xt))[:, 1]

        # add np.nan_to_num to replace nan by 0s
        # np.nan_to_num( scaler.transform(fv_tj) )
        y_pred = model.predict( scaler.transform(Xset) )  # todo: numerical problem: Input contains infinity or a value too large for dtype('float64')
        atype = 'micro'  # 'macro'   
        # ... micro averaging is useful when classes different in sizes
        m_auc = common.multiclass_roc_auc_score(yset, y_pred, average=atype)
        msg += "(polarity_modeling) overall {} AUC on T: {}\n".format(atype, m_auc)

        # another way to analyze the performance
        # inv_codes = {v: k for k, v in codes.items()}
        # polarity_labels = [inv_codes[l] for l in np.unique(yset)]  # labels are coded numerics
        # sorted_labels = sorted(polarity_labels, key=lambda x: x[1:], reverse=True)
        # y_test = np.array([inv_codes[l] for l in yset])
        msg += classification_report(yset, y_pred, target_names=['fp', 'fn', 'tn', 'tp']) + '\n'

        # ... M determined 
        #     M_bar = None
        Po_t = M   # if just binary {-1, 1}, then the job is done 
        if tMulticlass: 
            # convert to polarity repr {-1, 0, 1} for testing
            Po_t = from_color_to_polarity(M, codes, verify=True)

        npol = len(np.unique(Po_t)); assert npol == 2, "Expecting 2 polarities but got {}".format(npol)
        print(msg)
        # convert to -1 and 1 repr 

        # -------------------------------------------------------------------

        # [test] #2. evaluate preference score predictions
        msg = ""
        if Lt is not None: 
            assert len(Lt) == T.shape[1] 
            # Mct, Lht = probability_filter(T, Lt, p_th)  # Mc is a (0, 1)-matrix
            # ... correctness and label matrix using true labels Lt
            ret = eval_polarity(Po_t, Mct, Lht, verbose=True, name='T', neg_po=-1, title='(polarity_modeling) -- T given Po_t --')

            # fmax metric for the estimated polarity matrix (M)
            msg += '(polarity_modeling) Compare estimated polarity matrix (Po_t) with majority vote-induced preference matrix ...\n'
            pvt_max = predict_by_importance_weights(T, to_preference(Po_t), aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
            fmax_t = common.fmax_score(Lt, pvt_max, beta = 1.0, pos_label = 1)
            msg += "... fmax(Po_t): {}\n".format(fmax_t)
            # ... polarity performance of M 

            # introduce polarity correction? 
            Po_c = polarity_correction(Po_t, n_symbols_col=2, pos_label=1, neg_label=0)
            msg += '(polarity_modeling) Compare corrected polarity matrix (Po_c) with majority vote-induced preference matrix ...\n'
            pvc_max = predict_by_importance_weights(T, to_preference(Po_c), aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
            fmax_c = common.fmax_score(Lt, pvc_max, beta = 1.0, pos_label = 1)
            msg += "... fmax(Po_c): {}\n".format(fmax_c)

            # how does it compare to majority votes? 
            # ... to_polarity(lh, Lh)
            ############################################################
            name = 'T'
            lh = estimateLabels(T, p_th=p_th, pos_label=pos_label) 
            Mct_max, Lht_max = probability_filter(T, lh, p_th)  # Mc is a (0, 1)-matrix
            
            # how many entries are different compared to True polarities? 
            n_agreed = np.sum(Mct_max == Mct)
            n_agreed_labeling = np.sum(Lt == lh)
            msg += '... {}max | n_agreed: {}, ratio: {} | lh vs Lt? n_agreed: {}, ratio: {}\n'.format(name, n_agreed, n_agreed/(Mct_max.shape[0] * Mct_max.shape[1]+0.0), 
                n_agreed_labeling, n_agreed_labeling/(len(lh)+0.0))
            pvt_max = predict_by_importance_weights(T, Mct_max, aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
            fmax_t = common.fmax_score(Lt, pvt_max, beta = 1.0, pos_label = 1)
            msg += "... fmax({}, majority): {}\n".format(name, fmax_t)

            # Mct_max = (Lht == lh).astype(int)  # use estimated labels and label matrix to compute polarity matrix
            ret = eval_polarity(preference_to_polarity(Mct_max), Mct, Lht, verbose=True, name='Tmax', neg_po=-1, title='(polarity_modeling) -- {}: majority votes --'.format(name))
            # ... majority vote (T)   
            ############################################################
        print(msg)

        # -------------------------------------------------------------------
         # [test] #3. pretend to predict the training data

        msg = ""

        Xset, yset = [], []

        Mcr, Lhr = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
        Mr = np.zeros(R.shape)
        Rm = np.sort(R, axis=1) # rank(R, method='average')

        for j in range(R.shape[1]):

            Xset_j = []
            for i in range(T.shape[0]):   # foreach row in T[:, j]

                fvh = get_vars_hstats(R, i, j, p_th, C=Cr, Rm=Rm, U=U, Uc=Uc, encoder=u_encoder, p_model=scopes, r_min=r_min,  # ratio of minority class
                            name='{}-horizontal'.format('predict-R'), index=index, verbose=tVerbose)
                # fvv = get_vars_vstats(R, i, j, p_th, C=Cr, p_model=scopes, name='{}-vertical'.format('predict-R'), index=index)

                # define feature vector
                #########################
                fv = fvh # np.hstack([fvh, fvv])
                Xset_j.append( fv )  
                #########################

                # [test]
                ################################################################################
                Xset.append(fv)
                if Lhr[i, j] == pos_label and Mcr[i, j] ==1: 
                    yset.append(codes['tp'])
                elif Lhr[i, j] == pos_label and Mcr[i, j] ==0:
                    yset.append(codes['fp'])  # 'fp'
                elif Lhr[i, j] == neg_label and Mcr[i, j] == 1:
                    yset.append(codes['tn'])
                elif Lhr[i, j] == neg_label and Mcr[i, j] == 0:
                    yset.append(codes['fn']) # 'fn'
                ################################################################################
            
            fv_rj = np.array(Xset_j)  # 2D array; feature vectors associated with a column

            # np.nan_to_num( scaler.transform(fv_rj) )
            pvj = model.predict( scaler.transform(fv_rj) ) # model.predict_proba(fv_rj)[:, 1]  # 1/foldCount worth of data
            Mr[:, j] = pvj
        
        Po_r = Mr
        if tMulticlass: 
            # maps tp, tn to 1; fp, fn to -1
            Po_r = from_color_to_polarity(Mr, codes, verify=True)
        npol = len(np.unique(Po_r)); assert npol == 2, "Expecting 2 polarities but got {}".format(npol)
        
        ret = eval_polarity(Po_r, Mcr, Lhr, verbose=True, name='R', neg_po=-1, title="(polarity_modeling) -- R given Po_r --")
        # Py1_x = model.predict_proba(scaler.transform(Xset))[:, 1]

        # np.nan_to_num( scaler.transform(fv_rj) )
        y_pred = model.predict(scaler.transform(Xset))
        atype = 'micro'  # 'micro' or 'macro'  ... micro averaging is useful when classes different in sizes
        m_auc = common.multiclass_roc_auc_score(yset, y_pred, average=atype)
        msg += "(polarity_modeling) overall {} AUC on R: {}\n".format(atype, m_auc)

        # another way to analyze the performance on R
        # y_test = np.array([inv_codes[l] for l in yset])
        msg += classification_report(yset, y_pred, target_names=['fp', 'fn', 'tn', 'tp']) + '\n'

        name = 'R'
        lh = estimateLabels(R, p_th=p_th, pos_label=pos_label) 
        Mcr_max, Lhr_max = probability_filter(R, lh, p_th)  # Mc is a (0, 1)-matrix
        # ... correctness and labeling matrix obtained via majority vote
        
        # how many entries are different compared to True polarities? 
        n_agreed = np.sum(Mcr_max == Mcr)
        n_agreed_labeling = np.sum(Lr == lh)
        msg += '... {}max | n_agreed: {}, ratio: {} | lh vs Lr? n_agreed: {}, ratio: {}\n'.format(name, n_agreed, n_agreed/(Mcr_max.shape[0] * Mcr_max.shape[1]+0.0), 
            n_agreed_labeling, n_agreed_labeling/(len(lh)+0.0))
        pvr_max = predict_by_importance_weights(R, Mcr_max, aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
        fmax_r = common.fmax_score(Lr, pvr_max, beta = 1.0, pos_label = 1)
        msg += "... fmax({}, majority): {}\n".format(name, fmax_r)

        # if we were to use preference matrix as a polarity matrix
        ret = eval_polarity(preference_to_polarity(Mcr_max), Mcr, Lhr, verbose=True, name='Rmax', neg_po=-1, title='(polarity_modeling) -- {}: majority votes --'.format(name))  
        # ... majority vote (R)   
        print(msg)    

    else:
        raise NotImplementedError
    # ... M encodes an estimate of sample types (e.g. TP, TN, negative, unknown)

    # [test] output
    polarities = np.unique(M)
    if policy.startswith('class'): 
        assert len(polarities) >= 4, "Expecting >= 4 polarities but got n={} | {}".format(len(polarities), polarities)
    else: 
        pass

    msg = ''
    msg += "(polarity_modeling) positive model M | n(pos): {}, n(neg): {}, n(neutral): {}\n".format(np.sum(M > 0), np.sum(M < 0), np.sum(M==0)) 
    if M_bar is not None: 
        msg += "...                 negative model M_bar | n(pos): {}, n(neg): {}, n(neutral): {}\n".format(np.sum(M_bar > 0), np.sum(M_bar < 0), np.sum(M_bar==0))
    print(msg)

    # sys.exit(0)
    return M, M_bar

def make_cn(C, Po, is_unweighted=False, weight_neutral=0.0, weight_negative=-1.0, sparsify=True, verbose=1):
    """
    Given polarity matrix (Po), mask the neutral and negative entries in the confidence matrix so that 
    they do not enter the optimization objective (i.e. latent factors will not be made to approximate 
    these entries well because these entries do not matter). 

    Note
    ----
    1. this routine is effectively the same as make_cp() now ... 02.07.22
    """
    import scipy.sparse as sparse

    if not is_unweighted: 
        if verbose: print("(make_cn) Using UNWEIGHTED Cw, non-weighted MF to approximate ratings ...")
        Cn = np.ones(C.shape) 

        if sparse.issparse(Po): 
            mask = Po.toarray()
            Cn[mask == 0] = weight_neutral    # marking neutral (if 0, won't enter optimization objective)
            Cn[mask < 0] = weight_negative    # marking negative
            # Cn = sparse.csr_matrix(Cn)
        else: 
            Cn[Po == 0] = weight_neutral    # marking neutral
            Cn[Po < 0]  = weight_negative   # marking negative
    else: 
        if verbose: print("(make_cn) Using WEIGHTED Cw, weighted MF to approximate ratings ...")
        # otherwise, we retain the weight but masking the neutral and negative examples so that they do not enter the cost function when approximating "ratings"
        
        # ... If Cx is sparse, then Cn+Cx is no longer sparse but of matrix type (if without .toarray())
        if sparse.issparse(C):
            Cn = C.toarray()  # Cx.toarray()  # copying
        else: 
            Cn = np.copy(C)

        mask_neutral = Po.toarray()==0 if sparse.issparse(Po) else Po == 0
        Cn[mask_neutral] = Cn[mask_neutral] * weight_neutral  # masking neutral  
        
        mask_negative = Po.toarray() < 0.0 if sparse.issparse(Po) else Po < 0
        Cn[mask_negative] = Cn[mask_negative] * weight_negative  # masking negative
    # ... Cn is dense at this point

    # dtype(Cn) must be consistent with dtype(Po)
    if sparse.issparse(Po) or sparsify: # then Cn must also be sparse to be consistent
        Cn = sparse.csr_matrix(Cn)
    return Cn
# [alias]
mask_neutral_and_negative = make_cn

def make_cp(C, Po, is_unweighted=False, sparsify=True):
    """
    Similar to make_cn() but mask only the nuetral (i.e. entries with so much uncertainty 
    that we do not know if they are TP, TN or FP, FN)
    """

    import scipy.sparse as sparse

    if is_unweighted:  
        print("(make_cp) Using UNWEIGHTED Cw, non-weighted MF to approximate ratings ...")
        Cp = np.ones(C.shape) 
        
        if sparse.issparse(Po): 
            mask = Po.toarray()
            Cp[mask==0] = 0.0
            # Cp = sparse.csr_matrix(Cp)
        else: 
            Cp[Po==0] = 0.0  # masking neutral
    else: 
        print("(make_cp) Using WEIGHTED Cw, weighted MF to approximate ratings ...")
        # otherwise, we retain the weight but masking the neutral and negative examples so that they do not enter the cost function when approximating "ratings"
        
        # ... If Cx is sparse, then Cp+Cx is no longer sparse but of matrix type (if without .toarray())
        if sparse.issparse(C):
            Cp = C.toarray()  # Cx.toarray()  # copying
        else: 
            Cp = np.copy(C)

        mask_neutral = Po.toarray()==0 if sparse.issparse(Po) else Po == 0
        Cp[mask_neutral] = 0.0 
    # ... Cp is dense at this point

    # dtype(Cn) must be consistent with dtype(Po)
    if sparse.issparse(Po) or sparsify: # then Cn must also be sparse to be consistent
        Cp = sparse.csr_matrix(Cp)
    return Cp
# [alias]
mask_neutral_only = make_cp


def polarity_to_preference(**kargs): 
    return to_preference(**kargs)
def to_preference_matrix(**kargs):
    return to_preference(**kargs)
def to_preference(Po, neutral=0.0):
    import scipy.sparse as sparse 
    # polarity matrix to preference matrix 
    # assert neutral < 1 and neutral >= 0.0

    P = np.ones(Po.shape)  
    if sparse.issparse(Po): 
        Pa = Po.toarray()
        P[Pa==0] = neutral      # masking neutral
        P[Pa > 0] = 1.0
        P[Pa < 0] = 0.0    # masking negative
        P = sparse.csr_matrix(P)
    else: 
        P[Po==0] = neutral  # masking neutral
        P[Po > 0] = 1.0
        P[Po < 0] = 0.0  # masking negative
    return P # {0, 1}
def preference_to_polarity(M):
    return to_polarity(M)
def to_polarity(M, verify=False): 
    # from preference matrix to polarity matrix

    # if verify: 
    #     vmin, vmax = np.min(M), np.max(M)

    import scipy.sparse as sparse
    P = np.ones(M.shape)  
    if sparse.issparse(M):      
        P[M.toarray() == 0] = -1    # incorrect predictions (FP, FN) => negative polarity 
        P = sparse.csr_matrix(P)
    else: 
        P[M == 0] = -1 
    return P
def to_polarity_matrix(M):
    # preference matrix {0, 1} to polarity matrix 
    return to_polarity(M)

def from_color_to_preference(M, codes={}, verify=True): 
    import scipy.sparse as sparse
    
    # [test]
    colors = set(np.unique(M))
    vmin, vmax = min(colors), max(colors)
    if vmin == 0 and vmax == 1 and len(colors) == 2: 
        print("(from_color_to_polarity) Input M is already a preference matrix! | colors: {}".format(colors))
        return M  

    # M: color matrix
    # use: for approximating ratings
    Po = np.zeros(M.shape)
    if sparse.issparse(M):
        mask = M.toarray()
    else:
        mask = M # np.copy(M)

    Po[mask > 0] = 1 # representing preferred entries (e.g. reliable, correct probabilities)
    Po[mask < 0] = 0 # representing non-preferable entries (e.g. unreliable, incorrect probabilities)

    return Po

def from_color_to_reduced_color(M, codes={}, verify=True):
    """
    Convert from color (polarity) matrix to reduced color polarity matrix {-1, 0, 1, 2}
    """
    import scipy.sparse as sparse
    
    # [test]
    colors = set(np.unique(M))
    vmin, vmax = min(colors), max(colors)
    if colors == set([-1, 0, 1, 2]): 
        print("(from_color_to_reduced_color) Input M is already a reduced-color matrix! | colors: {}".format(colors))
        return M
    
    # M: color matrix
    # use: for approximating ratings
    if sparse.issparse(M):
        mask = M.toarray()
    else:
        mask = M # np.copy(M)

    # Po[M > 0] does not change values
    Po[M < 0] = -1

    return Po

def from_color_to_polarity(M, codes={}, verify=False): 
    """
    Convert from color (polarity) matrix to regular polarity matrix, whose non-zero entries 
    can only be either 1 or -1
    
    Converting color matrix to polarity matrix is useful for preference score approximation

    Note that both color matrix and polarity matrix can have 0s, whose corresponding probabilities 
    will not enter the optimization objective. 
    """
    import scipy.sparse as sparse
    
    # [test]
    if verify:
        if sparse.issparse(M): 
            colors = set(np.unique(M.toarray()))  # np.unique() can only apply to dense matrix
        else: 
            colors = set(np.unique(M))
        vmin, vmax = min(colors), max(colors)
        if vmin == -1 and vmax == 1: 
            print("(from_color_to_polarity) Input M is already a polarity matrix! | colors: {}".format(colors))
            return M
    
    # M: color matrix
    # use: for approximating ratings
    Po = np.zeros(M.shape)
    if sparse.issparse(M):
        mask = M.toarray()
    else:
        mask = M # np.copy(M)

    Po[mask > 0] = 1 
    Po[mask < 0] = -1

    return Po

def to_colored_preference_matrix(**kargs): 
    # use: for approximating ratings
    return to_colored_preference(**kargs)
def to_colored_preference(M, codes):
    import scipy.sparse as sparse
    # M: colored polarity
    # use: for approximating ratings
    if sparse.issparse(M):
        Po = M.toarray()
    else:
        Po = np.copy(M)

    # no-op for TP
    # Po[(Po == codes['tp'])]

    # no-op for TN
    # Po[Po == codes['tp'])]

    # FP, FN all considered negative
    Po[(Po == codes['fp']) | (Po == codes['fn'])] = -1
    Po[ Po == codes['unk'] ] = 0 

    return Po

def test_polarity(T, labels, Pref=None, p_th=[], lh=[], name='T', pos_label=1, neg_label=0, title=''):
    import scipy.sparse as sparse

    msg = title+'\n' if title else '' # Po
    if len(p_th) == 0: 
        msg += "(test_polarity) proba threshold default to 0.5 ...\n"
        p_th = np.array([0.5] * T.shape[0])

    if Pref is None: 
        # then we need p_th and lh to estimate Pref
        if len(lh) == 0: 
            msg += "(test_polarity) label estimates default to majority votes ...\n"
            lh = estimateLabels(T, p_th=p_th, pos_label=pos_label) 
        # ... p_th, lh determined 

    # using true labels 
    Mc, Lh = probability_filter(T, labels, p_th)  # Mc is a (0, 1)-matrix

    # using estimated labels to estimate polarity matrix and labeling matrix
    Mct = Pref
    if Mct is None:
        Mct, Lht = probability_filter(T, lh, p_th)  # Mc is a (0, 1)-matrix
    
        # how many entries are different compared to True polarities? 
        n_agreed = np.sum(Mct == Mc)
        n_agreed_labeling = np.sum(Lh == Lht)
        msg += '(test_polarity) {}max | n_agreed: {}, ratio: {} | lh vs Lt? n_agreed: {}, ratio: {}\n'.format(name, n_agreed, n_agreed/(Mct.shape[0] * Mct.shape[1]+0.0), 
            n_agreed_labeling, n_agreed_labeling/(len(lh)+0.0))
    
    pvt = predict_by_importance_weights(T, Mct, aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
    fmax_t = common.fmax_score(labels, pvt, beta = 1.0, pos_label = 1)
    msg += "... fmax(majority): {}\n".format(fmax_t)
    print(msg)
    return

def eval_polarity(Po, Mc, Lh, pos_po=1, neg_po=-1, verbose=False, name='X', title=''): 
    # adapted from ratio_of_alignment() and ratio_of_alignment2()
    cells_positive = cells_preferred = (Po == pos_po)
    cells_negative = (Po == neg_po)
    # cells_miss = Po == 0

    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos
    cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    cells_fn = (Mc == 0) & predict_neg

    n_tp_hit = n_tp_pref = np.sum( cells_preferred & cells_tp ) # correctly aligned TPs
    n_tn_hit = n_tn_pref = np.sum( cells_preferred & cells_tn )
    n_fp_hit = n_fp_pref = np.sum( cells_preferred & cells_fp )
    n_fn_hit = n_fn_pref = np.sum( cells_preferred & cells_fn )

    ret = {}
    ret['precision_pref'] = precision_pref = n_tp_pref/(n_tp_pref+n_fp_pref+1e-3)
    ret['recall_pref'] = recall_pref = n_tp_pref/(n_tp_pref+n_fn_pref+1e-3)
    ret['npv_pref'] = npv_pref = n_tn_pref/(n_tn_pref+n_fn_pref+1e-3)
    ret['specificity_pref'] = specificity_pref = n_tn_pref/(n_tn_pref+n_fp_pref+1e-3)

    # combined measure 
    n_hit = n_tp_hit + n_tn_hit
    n_miss = n_fp_hit + n_fn_hit
    ret['accuracy_pref'] = accuracy = n_hit/(n_hit+n_miss)

    # if polarity = 1 corresponds to tp or tn, then it's a TP for polarity
    n_tp_polarity = np.sum( cells_positive & (cells_tp | cells_tn) )
    # if polarity = 1 ~              fp or fn, then it's a FP for polarity 
    n_fp_polarity = np.sum( cells_positive & (cells_fp | cells_fn) )
    # if polarity = 0 ~              fp or fn, then it's a TN for polarity 
    n_tn_polarity = np.sum( cells_negative & (cells_fp | cells_fn) )
    # if poarlity = 0 ~              tp or tn, then it's a FN for polarity
    n_fn_polarity = np.sum( cells_negative & (cells_tp | cells_tn) )

    ret['precision_polarity'] = n_tp_polarity/(n_tp_polarity+n_fp_polarity+1e-3)
    ret['recall_polarity'] = n_tp_polarity/(n_tp_polarity+n_fn_polarity+1e-3)
    ret['npv_polarity'] = n_tn_polarity/(n_tn_polarity+n_fn_polarity+1e-3)
    ret['specificity_polarity'] = n_tn_polarity/(n_tn_polarity+n_fp_polarity+1e-3)

    if verbose: 
        msg = title+'\n' if title else ''
        msg += "... Quality of preferred_entries({}) | precision:   {}, recall:   {}, npv:   {}, specificity:   {}\n".format(
            name, ret['precision_pref'], ret['recall_pref'], ret['npv_pref'], ret['specificity_pref'])
        msg += "... Quality of          polarity({}) | p-precision: {}, p-recall: {}, p-npv: {}, p-specificity: {} | accuracy: {}\n".format(
            name, ret['precision_polarity'], ret['recall_polarity'], ret['npv_polarity'], ret['specificity_polarity'], accuracy)
        print(msg)

    return ret

def estimate_polarity(R, Lr, p_th, T, Lt=None, C=None, U=None, policy='sequence', 
        labeling_model='simple', p_classifier='rf', 
        constrained=True, stochastic=True, estimate_sample_type=True,
        k_upper=-1, k_lower=-1, k_max=-1, k_min=2, verbose=True, pos_label=1, neg_label=0, index=0):

    if policy.startswith( 'class' ):
        M, M_bar = polarity_modeling(R, Lr, p_th, T, Lt=Lt, C=C, U=U, policy='classification', constrained=constrained, p_classifier=p_classifier, index=index)  
        # then M is already the desired polarity matrix (that does not depend on label estimation)
        return M 
    if policy.startswith( 'seq' ):
        if not p_classifier.lower() in ('crf', ): 
            # raise NotImplementedError()
            p_classifier = 'crf'
        M, M_bar =  polarity_modeling(R, Lr, p_th, T, Lt=Lt, C=C, U=U, policy='sequence', constrained=constrained, p_classifier=p_classifier, index=index)  
        # then M is already the desired polarity matrix (that does not depend on label estimation)
        return M 

    ############################################
    # ... methods below require label estimation
    if labeling_model.startswith('sim'):
        return estimate_polarity_simple(R, Lr, p_th, T, policy=policy, 
                   constrained=constrained, stochastic=stochastic, 
                       p_classifier=p_classifier,
                       k_upper=k_upper, k_lower=k_lower, k_max=k_max, k_min=k_min)
    if estimate_sample_type: 
        return estimate_polarity_stacker(R, Lr, p_th, T, policy=policy, 
                   labeling_model=labeling_model, p_classifier=p_classifier,
                   constrained=constrained, stochastic=stochastic, k_upper=k_upper, k_lower=k_lower, k_max=k_max, k_min=k_min)
    return estimate_polarity_stacker2(R, Lr, p_th, T, policy=policy, labeling_model=labeling_model,
                   constrained=constrained, stochastic=stochastic, k_upper=k_upper, k_lower=k_lower, k_max=k_max, k_min=k_min)
    # output: polarity matrix

def estimate_polarity_simple(R, Lr, p_th, T, policy='median', labeling_model='simple',
        p_classifier='rf',
        constrained=1, stochastic=True, 
        k_upper=-1, k_lower=-1, k_max=-1, k_min=1, verbose=True, pos_label=1, neg_label=0):

    import collections
    from scipy.stats import nbinom, binom

    sample_types = Polarity.sample_types
    codes = Polarity.codes

    # identify entry type (TP, TN, FP, FN, '?') wrt positive label (M) or negative label (M_bar) 
    Mc, Lh = probability_filter(R, Lr, p_th)
    M, M_bar = polarity_modeling(R, Lr, p_th, T, policy=policy, constrained=constrained, 

        # only relevant when policy = 'classification'
        p_classifier=p_classifier) 
    # typically M
    
    # [test]
    npol = 0 
    polarities = []
    if policy.startswith('class'): 
        assert M_bar is None 
        polarities = np.unique(M)
        npol = len(polarities)
        assert npol >= 2
    else: 
        polarities = np.unique(M)
        npol = len(polarities)
        assert npol == 3
    print("(estimate_polarity_simple) unique polarities: {} (n={}) | policy={}".format(polarities, npol, policy))

    ############################################################
    # weight M[:,j], M_bar[:,j] according to Lh statistics 
    # n_positive_m_negagive => label
    labels = [neg_label, pos_label, ]

    ### priors: p(y=1), p(y=0)
    counter = collections.Counter(Lr)
    n = sum(counter.values())
    priors = {l: counter.get(l, 0.0)/(n+0.0) for l in labels}

    # Py_c = CPT = {}
    # for i in range(n_users+1): 
    #     cc = (i, n_users-i)
    #     if not cc in CPT: CPT[cv] = {l:0 for l in labels}
    # for j in range(Lh.shape[1]): 
    #     label = Lr[j]
    #     counts = collections.Counter(Lh[:, j])
    #     cv = tuple([counts.get(l, 0) for l in labels]) # order: -,+ 
    #     if not cv in CPT: CPT[cv] = {l:0 for l in labels}
    #     CPT[cv][label] += 1

    Pc_y = {l:{} for l in labels}
    for j in range(Lh.shape[1]): 
        label = Lr[j]
        counts = collections.Counter(Lh[:, j])

        cv = tuple([counts.get(l, 0) for l in labels]) # order: -,+ 
        if not cv in Pc_y[label]: Pc_y[label][cv] = 0
        Pc_y[label][cv] += 1

    n_users = R.shape[0]
    # fit a binomial model for the voting counts
    # ... all possible count configurations and their correspondig counts (sum of '+' and '-')
    # ... or just counts of 1's (predicting positive)
    F = {l: {i: 0 for i in range(n_users+1)} for l in labels}  # frequnecy counts
    # among all cases that predict positive, what the distribution of their counts c|y=1
    for y_true, pred in Pc_y.items():  # foreach entry in P(y_predict=1 | y_true)
        for cc, c in pred.items(): 

            ##################
            f_pos = cc[pos_label]  # label count for label = l: {0, 1}
            ##################
            # ... focus on predicting '+'

            F[y_true][f_pos] = c   # frequency table for true label: y_true, in which freq(predicting '+') = c 
    # ... labels: {0, 1} each has it own frequency distribution for predicting '+'

    # estimate P(c|y=1)
    ws = sum([f * F[pos_label].get(f, 0) for f in range(n_users+1)])  # 0/0, 1/5, 2/10, ... 39/1, ... 49/0, 50/0
    w = sum([F[pos_label].get(f, 0) for f in range(n_users+1)])  # 0, 5, 10, ... 1, ... 0, 0
    Ef1 = ws/(w+0.0)  # expected freq
    p_y1 = Ef1/n_users  # probability predicting '+' | y=1

    # estimate P(c|y=0)
    ws = sum([f * F[neg_label].get(f, 0) for f in range(n_users+1)])  
    w = sum([F[neg_label].get(f, 0) for f in range(n_users+1)])
    Ef0 = ws/(w+0.0)
    p_y0 = Ef0/n_users # probability of predicting '+' | y = 0

    msg = ''
    msg += '(polarity_simple) E[f]:{}, N={} => p(predicting +): {} | y_true = 1\n'.format(Ef1, n_users, p_y1)
    msg += '...               E[f]:{}, N={} => p(predicting +): {} | y_true = 0\n'.format(Ef0, n_users, p_y0)
    msg += '...               prior: P(y=0): {}, P(y=1): {}'.format(priors[neg_label], priors[pos_label])
    ############################################################
    print(msg)

    ### resolve sample types 
    # ... M eventually should only consists of 3 types of polarity: 0/neutral, 1/poistive, -1/negative
    #     positive => high confidence of being in {TP, TN}, negative: incorrect {FP, FN}, neutral: unknown 
    tStochastic = stochastic # stochastic
    tConstrained = constrained
    print('(estimate_polarity_simple) Constrained? {}, Stochastic? {}'.format(constrained, True))

    test_pts = set(np.random.choice(range(T.shape[1]), 20))
    n_conflict_evidence = n_no_positive = 0
    n_tp_dominant = n_tn_dominant = 0
    tHasConflict = False
    n_null = 0 
    M2 = np.zeros(T.shape)
    pt = {}
    if not tConstrained: 
        for j in range(T.shape[1]):  # foreach item/datum
            counts = collections.Counter(Lh[:, j])
            n_pos = counts.get(pos_label, 0)
            n_neg = counts.get(neg_label, 0)
            cv = (n_neg, n_pos) # order: -,+

            # default method: majority vote 
            P_tp_dominant = pos_label if n_pos > n_neg else neg_label
            P_tn_dominant = 1 - P_tp_dominant

            # method A: simple counts on Lh[:, j]
            # P_tp_dominant = n_pos/(Lh.shape[0]+0.0)

            # method B: estimate conditional proba  P( y | count)
            # ... problem: more prone to have higher votes for negative classes
            # Z = CPT[cv][pos_label] + CPT[cv][neg_label]+0.0  # normalization constant
            # P_tp_dominant = CPT[cv][pos_label]/Z
            # P_tn_dominant = CPT[cv][neg_label]/Z

            # method C: fit a binomial model for P(cv|y=1) and P(cv|y=0), then use bayes rule to infer p(y=1|cv), where cv is a count vector 
            # ... P(y=1|c) = P(y, c)/P(c) = P(c|y=1)P(y=1)/(P(c|y=1)P(y=1)+P(c|y=0)P(y=0))
            if tStochastic: 
                pcy1 = pt['P(c(+)=n|l=1)'] = binom.pmf(n_pos, n=n_users, p=p_y1)   # given that true label is positiev, proba of predicting positive
                pcy0 = pt['P(c(+)=n|l=0)'] = binom.pmf(n_pos, n=n_users, p=p_y0)
                py1  = pt['P(l=1)'] = priors[pos_label]
                py0 = pt['P(l=0)'] = priors[neg_label]
                # assert pcy1 <= 1.0 and pcy0 <= 1.0 
                # assert py1 <= 1.0 and py0 <= 1.0

                P_tp_dominant = pt['P(l=1|c(+)=n)'] = pt['P(c(+)=n|l=1)'] * pt['P(l=1)'] / (pt['P(c(+)=n|l=0)']*pt['P(l=0)'] + pt['P(c(+)=n|l=1)']*pt['P(l=1)'])
                P_tn_dominant = pt['P(l=0|c(+)=n)'] = 1. - P_tp_dominant
            # assert P_tp_dominant <= 1.0
            # assert P_tn_dominant <= 1.0

            # [test]
            if (j in test_pts) or (P_tp_dominant >= P_tn_dominant):  # (j in test_pts) 
                print("... mode: unconstrained | P(y|count) | [{}] counts: (-:{}, +:{}) => P_tp_dominant: {}, P_tn_dominant: {}".format(j, cv[0], cv[1], P_tp_dominant, P_tn_dominant))

            if P_tp_dominant > 0 and P_tn_dominant > 0: 
                n_conflict_evidence += 1   # this is almost always true

            # choose table M vs M_bar
            ###########################################################################
            p = np.random.uniform(0, 1, 1)[0]
            is_tp_dominant = (True if p <= P_tp_dominant else False) if tStochastic else (True if P_tp_dominant == pos_label else False)    
            ###########################################################################

            # choose a polarity matrix 
            Mct = M if is_tp_dominant else M_bar

            tPartialSupport = tNullSupport = False
            if is_tp_dominant: 
                # index into M (otherwise M_bar)
                tp_i = np.where(Mct[:, j] == 1)[0]   # codes['tp']
                fn_i = np.where(Mct[:, j] == -1)[0]   # codes['fn']

                support_pos = tp_i
                support_neg = fn_i 
                # ... the rest of the entries remain neutral 

                n_tp_dominant += 1
            else: # is_tn_dominent or P(y=0|count) > P(y=1|count)
                # index into M_bar
                tn_i = np.where(Mct[:, j] == 1)[0]  #  codes['tn']
                fp_i = np.where(Mct[:, j] == -1)[0]  # codes['fp']

                support_pos = tn_i
                support_neg = fp_i 

                n_tn_dominant += 1 
            
            # ... it's possible that item[j] does not have eithre positive or negative support
            if len(support_pos) == 0 or len(support_neg) == 0: tPartialSupport = True

            # assert len(support_pos) > 0 or len(support_neg) > 0 
            if tPartialSupport:  # include the case where neither positive support nor negative support were found
                if is_tp_dominant: 
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '+' examples
                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # if len(support_neg) == 0, then choose k_min

                            p_th = np.median(T[support_neg, j])
                            # p_th = np.max(T[support_neg, j])  # max of the negative 

                            rows = np.argsort(-T[:, j])  # high to low
                            support_pos = [r for r in rows if T[r, j] > p_th][: n_neg]  # [: n_neg] to get a balanced set
                    if len(support_neg) == 0: 
                        # devoid of '-' examples
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.min(T[support_pos, j])  # min of the positive

                            rows = np.argsort(T[:, j])  # low to high
                            support_neg = [r for r in rows if T[r, j] < p_th][: n_pos]
                else:  # is_tn_dominent
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '-' examples

                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # '+' examples are the negative
                            
                            p_th = np.median(T[support_neg, j])
                            # p_th = np.min(T[support_neg, j])  # min proba of the positive examples
                            rows = np.argsort(T[:, j])  # low to high

                            # '+' polarity examples are those ~ negative examples (having low proba values)
                            #  ... which should not go above the min proba of '+' examples
                            support_pos = [r for r in rows if T[r, j] < p_th][: n_neg]
                    if len(support_neg) == 0:  # '+' examples serve as the negative polarity 
                        # devoid of '+' examples 
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))  # TNs

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.max(T[support_pos, j])  # max of '-' examples

                            rows = np.argsort(-T[:, j])  # high to low
                            support_neg = [r for r in rows if T[r, j] > p_th][: n_pos]

            if len(support_pos) == 0 or len(support_neg) == 0: 
                tNullSupport = True
                n_null += 1

                # then pick k_min for each polarity anyway 
                if is_tp_dominant:
                    # positive support
                    rows = np.argsort(-T[:, j])  # high to low
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]
                else: 
                    rows = np.argsort(T[:, j])  # low to high
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]

            assert len(support_pos) >= k_min, "There should be at least {} positive-polarity examples but got {}".format(k_min, len(support_pos)) 
            assert len(support_neg) >= k_min, "There should be at least {} negative-polarity examples but got {}".format(k_min, len(support_neg)) 

            # ... negative and positive support determined
            M2[support_pos, j] = 1
            M2[support_neg, j] = -1
        ### end foreach datum

    else: # constrained        

        # test_pts = set(np.random.choice(range(T.shape[1]), 100))
        for j in range(T.shape[1]):  # foreach item/datum
            counts = collections.Counter(Lh[:, j])
            n_pos = counts.get(pos_label, 0)
            n_neg = counts.get(neg_label, 0)
            cv = (n_neg, n_pos)

            # method 0: majority vote 
            P_tp_dominant = pos_label if n_pos > n_neg else neg_label
            P_tn_dominant = 1 - P_tp_dominant
            # method A: simple counts on Lh[:, j]
            # P_tp_dominant = n_pos/(Lh.shape[0]+0.0)

            # method B: estimate conditional proba  P( y | count)
            # Z = CPT[cv][pos_label] + CPT[cv][neg_label]+0.0  # normalization constant
            # P_tp_dominant = CPT[cv][pos_label]/Z
            # P_tn_dominant = CPT[cv][neg_label]/Z

            if tStochastic: 
                # method C: fit a binomial model for P(cv|y=1) and P(cv|y=0), then use bayes rule to infer p(y=1|cv), where cv is a count vector 
                # ... P(y=1|c) = P(y, c)/P(c) = P(c|y=1)P(y=1)/(P(c|y=1)P(y=1)+P(c|y=0)P(y=0))
                pt['P(c(+)=n|l=1)'] = binom.pmf(n_pos, n=n_users, p=p_y1)   # given that true label is positiev, proba of predicting positive
                pt['P(c(+)=n|l=0)'] = binom.pmf(n_pos, n=n_users, p=p_y0)
                pt['P(l=1)'] = priors[pos_label]
                pt['P(l=0)'] = priors[neg_label]

                P_tp_dominant = pt['P(l=1|c(+)=n)'] = pt['P(c(+)=n|l=1)'] * pt['P(l=1)'] / (pt['P(c(+)=n|l=0)']*pt['P(l=0)'] + pt['P(c(+)=n|l=1)']*pt['P(l=1)'])
                P_tn_dominant = pt['P(l=0|c(+)=n)'] = 1 - P_tp_dominant

            # [test]
            if (j in test_pts) or (P_tp_dominant >= P_tn_dominant): 
                print("... mode: constrained | P(y|count) | [{}] counts: (-:{}, +:{}) => P_tp_dominant: {}, P_tn_dominant: {}".format(j, cv[0], cv[1], P_tp_dominant, P_tn_dominant))

            if P_tp_dominant > 0 and P_tn_dominant > 0: 
                n_conflict_evidence += 1

            ###########################################################################
            p = np.random.uniform(0, 1, 1)[0]
            is_tp_dominant = (True if p <= P_tp_dominant else False) if tStochastic else (True if P_tp_dominant == pos_label else False)    
            ###########################################################################

            # choose polarity matrix 
            Mct = M if is_tp_dominant else M_bar

            tPartialSupport = tNullSupport = False
            if is_tp_dominant: 
                # index into M (otherwise M_bar)
                tp_i = np.where(Mct[:, j] == 1)[0]   # codes['tp']
                fn_i = np.where(Mct[:, j] == -1)[0]  # codes['fn']

                support_pos = tp_i
                support_neg = fn_i 
                # ... the rest of the entries remain neutral 

                # assert len(tp_i) > 0  
                rows = np.argsort(-T[:, j])  # np.argsort(R[:, j])[:-k-1: -1] # choose indices of k highest probs
                # ... high to low
                
                # positive examples: the larger the better
                support_pos = [i for i in rows if i in tp_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tp_i][:k_upper]
                # ... if k_upper_tp > 0, then we have conflicting evidence

                # negative examples: the smaller the better
                support_neg = [i for i in rows[::-1] if i in fn_i][:k_lower]

                # [test]
                # if j % 100 == 0: assert np.min(T[support_pos,j]) >= np.max(T[support_neg,j]), "min(pos): {} <? max(neg): {}".format(np.min(T[support_pos,j]), np.max(T[support_neg,j]))

                n_tp_dominant += 1
            else:  
                # index into M_bar
                tn_i = np.where(Mct[:, j] == 1)[0]  # codes['tn']
                fp_i = np.where(Mct[:, j] == -1)[0]  # codes['fp']

                rows = np.argsort(T[:, j])
                # ... low to high

                # assert len(tn_i) > 0

                # positive examples: the smaller the better
                support_pos = [i for i in rows if i in tn_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tn_i][:k_upper]

                # negative examples: the larger the better 
                support_neg = [i for i in rows[::-1] if i in fp_i][:k_lower]
                    
                n_tn_dominant += 1
            # ... negative and positive support determined

            # ... it's possible that item[j] does not have eithre positive or negative support
            if len(support_pos) == 0 or len(support_neg) == 0: tPartialSupport = True

            if tPartialSupport:  # include the case where neither positive support nor negative support were found
                if is_tp_dominant: 
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '+' examples
                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # if len(support_neg) == 0, then choose k_min
                            
                            p_th = np.median(T[support_neg, j])
                            # p_th = np.max(T[support_neg, j])  # max of the negative 

                            rows = np.argsort(-T[:, j])  # high to low
                            support_pos = [r for r in rows if T[r, j] > p_th][: n_neg]
                    if len(support_neg) == 0: 
                        # devoid of '-' examples
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.min(T[support_pos, j])  # min of the positive

                            rows = np.argsort(T[:, j])  # low to high
                            support_neg = [r for r in rows if T[r, j] < p_th][: n_pos]
                else:  # is_tn_dominent
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '-' examples

                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # '+' examples are the negative
                            
                            p_th = np.median(T[support_neg, j])
                            # p_th = np.min(T[support_neg, j])  # min proba of the positive examples
                            rows = np.argsort(T[:, j])  # low to high

                            # '+' polarity examples are those ~ negative examples (having low proba values)
                            #  ... which should not go above the min proba of '+' examples
                            support_pos = [r for r in rows if T[r, j] < p_th][: n_neg]
                    if len(support_neg) == 0:  # '+' examples serve as the negative polarity 
                        # devoid of '+' examples 
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))  # TNs

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.max(T[support_pos, j])  # max of '-' examples

                            rows = np.argsort(-T[:, j])  # high to low
                            support_neg = [r for r in rows if T[r, j] > p_th][: n_pos]

            # still missing supports? 
            if len(support_pos) == 0 or len(support_neg) == 0: 
                tNullSupport = True
                n_null += 1

                # then pick k_min for each polarity anyway 
                if is_tp_dominant:
                    # positive support
                    rows = np.argsort(-T[:, j])  # high to low
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]
                else: 
                    rows = np.argsort(T[:, j])  # low to high
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]

            assert len(support_pos) >= k_min, "There should be at least {} positive-polarity examples but got {}".format(k_min, len(support_pos)) 
            assert len(support_neg) >= k_min, "There should be at least {} negative-polarity examples but got {}".format(k_min, len(support_neg)) 

            M2[support_pos, j] = 1
            M2[support_neg, j] = -1

        ### ... end foreach item

    if verbose: 
        msg = ''
        r = n_conflict_evidence/(T.shape[1]+0.0)
        msg += "(polarity_simple) tContrained: True | Found n_conflict_evidence: {}, n_no_positive: {} | N={}, ratio_conflict_evidence: {}\n".format(n_conflict_evidence, n_no_positive, T.shape[1], r)
        msg += "...               n_tp_dominant: {}, n_tn_dominant: {}\n".format(n_tp_dominant, n_tn_dominant)
        msg += "...               n(pos): {}, n(neg): {}, n(neutral): {} | n(null): {}\n".format(np.sum(M2>0), np.sum(M2<0), np.sum(M2==0), n_null)
        print(msg)

    return M2

def estimate_polarity_stacker(R, Lr, p_th, T, policy='median', labeling_model='logistic', 
       constrained=True, stochastic=True, p_classifier='rf',
       k_upper=-1, k_lower=-1, k_max=-1, k_min=2, verbose=True, pos_label=1, neg_label=0): 
    import stacking 

    sample_types = Polarity.sample_types
    codes = Polarity.codes

    Mc, Lh = probability_filter(R, Lr, p_th)
    M, M_bar = polarity_modeling(R, Lr, p_th, T, policy=policy, constrained=constrained, 
        # only relevant when policy = 'classification'
        p_classifier=p_classifier)

    # [test]
    npol = 0 
    polarities = []
    if policy.startswith('class'): 
        assert M_bar is None 
        polarities = np.unique(M)
        npol = len(polarities)
        assert npol >= 2
    else: 
        polarities = np.unique(M)
        npol = len(polarities)
        assert npol == 3
    print("(estimate_polarity_stacker) unique polarities: {} (n={}) | policy={}".format(polarities, npol, policy))

    stacker = stacking.choose_classifier(labeling_model)  # e.g. log, enet, knn
    model = stacker.fit(R.T, Lr) # remember to take transpose
    pv = model.predict_proba(T.T)[:, 1]  # 1/foldCount worth of data

    ### resolve sample types 
    # ... M eventually should only consists of 3 types of polarity: 0/neutral, 1/poistive, -1/negative
    #     positive => high confidence of being in {TP, TN}, negative: incorrect {FP, FN}, neutral: unknown 
    tConstrained, tStochastic = constrained, stochastic # stochastic
    print('(estimate_polarity_stacker) Constrained? {}, Stochastic? {} | k_upper: {}, k_lower: {}'.format(constrained, stochastic, k_upper, k_lower))
    if not tStochastic: 
        metric = 'fmax'
        # turn probabilities into labels
        # ... can only use training data to estimate the best threshold
        pv_r = model.predict_proba(R.T)[:, 1]
        f1, p_threshold = common.fmax_score_threshold(Lr, pv_r, beta = 1.0, pos_label = 1)
        labels = [int(pv[j] >= p_threshold) for j, pvj in enumerate(pv)]
        pv = labels
        print("(estimate_polarity_stacker) p_threshold(R): {}, metric: {} example labels:\n... {}\n".format(p_threshold, metric, labels[:10]))

    test_pts = set(np.random.choice(range(T.shape[1]), 20))
    n_conflict_evidence = n_no_positive = 0
    n_tp_dominant = n_tn_dominant = 0
    tHasConflict = False
    n_null = 0 

    M2 = np.zeros(T.shape)
    if not tConstrained: 

        for j in range(T.shape[1]):  # foreach item/datum
            counts = collections.Counter(Lh[:, j])
            n_pos = counts.get(pos_label, 0)
            n_neg = counts.get(neg_label, 0)
            cv = (n_neg, n_pos) # order: -,+

            P_tp_dominant = pv[j]
            P_tn_dominant = 1.0 - P_tp_dominant

            # [test]
            if (j in test_pts) or (P_tp_dominant >= P_tn_dominant):  # (j in test_pts) 
                print("... mode: unconstrained | P(y|count) | [{}] counts: (-:{}, +:{}) => P_tp_dominant: {}, P_tn_dominant: {}".format(j, cv[0], cv[1], P_tp_dominant, P_tn_dominant))

            if P_tp_dominant > 0 and P_tn_dominant > 0: 
                n_conflict_evidence += 1   # this is almost always true

            p = np.random.uniform(0, 1, 1)[0]
            is_tp_dominant = (True if p <= P_tp_dominant else False) if tStochastic else (True if P_tp_dominant == pos_label else False)

            # choose polarity matrix 
            Mct = M if is_tp_dominant else M_bar

            tPartialSupport = tNullSupport = False
            if is_tp_dominant: # if likely a positive example ... 
                # index into M (otherwise M_bar)
                tp_i = np.where(Mct[:, j] == 1)[0]  # codes['tp']
                fn_i = np.where(Mct[:, j] == -1)[0]  # codes['fn']

                support_pos = tp_i
                support_neg = fn_i 
                # ... the rest of the entries remain neutral 

                n_tp_dominant += 1
            else: # is_tn_dominent or P(y=0|count) > P(y=1|count)
                # index into M_bar
                tn_i = np.where(Mct[:, j] == 1)[0]  # codes['tn']
                fp_i = np.where(Mct[:, j] == -1)[0]  # codes['fp']

                support_pos = tn_i
                support_neg = fp_i 

                n_tn_dominant += 1 
            
            # ... it's possible that item[j] does not have eithre positive or negative support
            if len(support_pos) == 0 or len(support_neg) == 0: tPartialSupport = True

            # assert len(support_pos) > 0 or len(support_neg) > 0 
            if tPartialSupport:  # include the case where neither positive support nor negative support were found
                if is_tp_dominant: 
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '+' examples
                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # if len(support_neg) == 0, then choose k_min

                            p_th = np.median(T[support_neg, j])
                            # p_th = np.max(T[support_neg, j])  # max of the negative 

                            rows = np.argsort(-T[:, j])  # high to low
                            support_pos = [r for r in rows if T[r, j] > p_th][: n_neg]
                    if len(support_neg) == 0: 
                        # devoid of '-' examples
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.min(T[support_pos, j])  # min of the positive

                            rows = np.argsort(T[:, j])  # low to high
                            support_neg = [r for r in rows if T[r, j] < p_th][: n_pos]
                else:  # is_tn_dominent
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '-' examples

                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # '+' examples are the negative

                            p_th = np.median(T[support_neg, j])
                            # p_th = np.min(T[support_neg, j])  # min proba of the positive examples
                            rows = np.argsort(T[:, j])  # low to high

                            # '+' polarity examples are those ~ negative examples (having low proba values)
                            #  ... which should not go above the min proba of '+' examples
                            support_pos = [r for r in rows if T[r, j] < p_th][: n_neg]
                    if len(support_neg) == 0:  # '+' examples serve as the negative polarity 
                        # devoid of '+' examples 
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))  # TNs

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.max(T[support_pos, j])  # max of '-' examples

                            rows = np.argsort(-T[:, j])  # high to low
                            support_neg = [r for r in rows if T[r, j] > p_th][: n_pos]

            if len(support_pos) == 0 or len(support_neg) == 0: 
                tNullSupport = True
                n_null += 1

                # then pick k_min for each polarity anyway 
                if is_tp_dominant:
                    # positive support
                    rows = np.argsort(-T[:, j])  # high to low
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]
                else: 
                    rows = np.argsort(T[:, j])  # low to high
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]

            assert len(support_pos) >= k_min, "There should be at least {} positive-polarity examples but got {}".format(k_min, len(support_pos)) 
            assert len(support_neg) >= k_min, "There should be at least {} negative-polarity examples but got {}".format(k_min, len(support_neg))
            
            # ... negative and positive support determined
            M2[support_pos, j] = 1
            M2[support_neg, j] = -1

    else: # constrained

        # test_pts = set(np.random.choice(range(T.shape[1]), 100))
        for j in range(T.shape[1]):  # foreach item/datum
            counts = collections.Counter(Lh[:, j])
            n_pos = counts.get(pos_label, 0)
            n_neg = counts.get(neg_label, 0)
            cv = (n_neg, n_pos)

            P_tp_dominant = pv[j]
            P_tn_dominant = 1.0 - P_tp_dominant

            # [test]
            if (j in test_pts) or (P_tp_dominant >= P_tn_dominant): 
                print("... mode: constrained | P(y|count) | [{}] counts: (-:{}, +:{}) => P_tp_dominant: {}, P_tn_dominant: {}".format(j, cv[0], cv[1], P_tp_dominant, P_tn_dominant))

            if P_tp_dominant > 0 and P_tn_dominant > 0: 
                n_conflict_evidence += 1

            p = np.random.uniform(0, 1, 1)[0]
            is_tp_dominant = (True if p <= P_tp_dominant else False) if tStochastic else (True if P_tp_dominant == pos_label else False)    

            # choose polarity matrix 
            Mct = M if is_tp_dominant else M_bar

            tPartialSupport = tNullSupport = False
            if is_tp_dominant: 
                # index into M (otherwise M_bar)
                tp_i = np.where(Mct[:, j] == 1)[0]  # codes['tp']
                fn_i = np.where(Mct[:, j] == -1)[0] # codes['fn']

                support_pos = tp_i
                support_neg = fn_i 
                # ... the rest of the entries remain neutral 

                # assert len(tp_i) > 0  
                rows = np.argsort(-T[:, j])  # np.argsort(R[:, j])[:-k-1: -1] # choose indices of k highest probs
                # ... high to low
                
                # positive examples: the larger the better
                support_pos = [i for i in rows if i in tp_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tp_i][:k_upper]
                # ... if k_upper_tp > 0, then we have conflicting evidence

                # negative examples: the smaller the better
                support_neg = [i for i in rows[::-1] if i in fn_i][:k_lower]

                # [test]
                # if j % 100 == 0: assert np.min(T[support_pos,j]) >= np.max(T[support_neg,j]), "min(pos): {} <? max(neg): {}".format(np.min(T[support_pos,j]), np.max(T[support_neg,j]))

                n_tp_dominant += 1
            else:  
                # index into M_bar
                tn_i = np.where(Mct[:, j] == 1)[0]  # codes['tn']
                fp_i = np.where(Mct[:, j] == -1)[0] # codes['fp']

                rows = np.argsort(T[:, j])
                # ... low to high

                # assert len(tn_i) > 0

                # positive examples: the smaller the better
                support_pos = [i for i in rows if i in tn_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tn_i][:k_upper]

                # negative examples: the larger the better 
                support_neg = [i for i in rows[::-1] if i in fp_i][:k_lower]
                    
                n_tn_dominant += 1
            # ... negative and positive support determined

            # ... it's possible that item[j] does not have eithre positive or negative support
            if len(support_pos) == 0 or len(support_neg) == 0: tPartialSupport = True

            # assert len(support_pos) > 0 or len(support_neg) > 0 
            if tPartialSupport:  # include the case where neither positive support nor negative support were found
                if is_tp_dominant: 
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '+' examples
                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # if len(support_neg) == 0, then choose k_min

                            p_th = np.median(T[support_neg, j])
                            # p_th = np.max(T[support_neg, j])  # max of the negative 

                            rows = np.argsort(-T[:, j])  # high to low
                            support_pos = [r for r in rows if T[r, j] > p_th][: n_neg]
                    if len(support_neg) == 0: 
                        # devoid of '-' examples
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.min(T[support_pos, j])  # min of the positive

                            rows = np.argsort(T[:, j])  # low to high
                            support_neg = [r for r in rows if T[r, j] < p_th][: n_pos]
                else:  # is_tn_dominent
                    if len(support_pos) == 0:  # create positive support
                        # devoid of '-' examples

                        if len(support_neg) > 0: 
                            n_neg = max(k_min, len(support_neg))  # '+' examples are the negative

                            p_th = np.median(T[support_neg, j])
                            # p_th = np.min(T[support_neg, j])  # min proba of the positive examples
                            rows = np.argsort(T[:, j])  # low to high

                            # '+' polarity examples are those ~ negative examples (having low proba values)
                            #  ... which should not go above the min proba of '+' examples
                            support_pos = [r for r in rows if T[r, j] < p_th][: n_neg]
                    if len(support_neg) == 0:  # '+' examples serve as the negative polarity 
                        # devoid of '+' examples 
                        if len(support_pos) > 0: 
                            n_pos = max(k_min, len(support_pos))  # TNs

                            p_th = np.median(T[support_pos, j])
                            # p_th = np.max(T[support_pos, j])  # max of '-' examples

                            rows = np.argsort(-T[:, j])  # high to low
                            support_neg = [r for r in rows if T[r, j] > p_th][: n_pos]

            if len(support_pos) == 0 or len(support_neg) == 0: 
                tNullSupport = True
                n_null += 1

                # then pick k_min for each polarity anyway 
                if is_tp_dominant:
                    # positive support
                    rows = np.argsort(-T[:, j])  # high to low
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]
                else: 
                    rows = np.argsort(T[:, j])  # low to high
                    support_pos = rows[:k_min]
                    support_neg = rows[::-1][:k_min]

            assert len(support_pos) >= k_min, "There should be at least {} positive-polarity examples but got {}".format(k_min, len(support_pos)) 
            assert len(support_neg) >= k_min, "There should be at least {} negative-polarity examples but got {}".format(k_min, len(support_neg))

            M2[support_pos, j] = 1
            M2[support_neg, j] = -1
        ### ... end foreach item

    if verbose: 
        msg = ''
        r = n_conflict_evidence/(T.shape[1]+0.0)
        msg += "(polarity_stacker) tContrained: True | Found n_conflict_evidence: {}, n_no_positive: {} | N={}, ratio_conflict_evidence: {}\n".format(n_conflict_evidence, n_no_positive, T.shape[1], r)
        msg += "...                n_tp_dominant: {}, n_tn_dominant: {}\n".format(n_tp_dominant, n_tn_dominant)
        msg += "...                n(pos): {}, n(neg): {}, n(neutral): {} | n(null): {}\n".format(np.sum(M2>0), np.sum(M2<0), np.sum(M2==0), n_null)
        print(msg)

    return M2

def estimate_polarity_stacker2(R, Lr, p_th, T, policy='median', labeling_model='logistic', 
       constrained=True, stochastic=True, 
       k_upper=-1, k_lower=-1, k_max=-1, k_min=1, verbose=True, pos_label=1, neg_label=0): 
    import stacking 

    sample_types = Polarity.sample_types
    codes = Polarity.codes

    Mc, Lh = probability_filter(R, Lr, p_th)  # Mc is a (0, 1)-matrix
    # M, M_bar = polarity_modeling(R, Lr, p_th, T, policy=policy, constrained=constrained)

    stacker = stacking.choose_classifier(labeling_model)  # e.g. log, enet, knn
    model = stacker.fit(R.T, Lr) # remember to take transpose
    pv = model.predict_proba(T.T)[:, 1]  # 1/foldCount worth of data

    ### resolve sample types 
    # ... M eventually should only consists of 3 types of polarity: 0/neutral, 1/poistive, -1/negative
    #     positive => high confidence of being in {TP, TN}, negative: incorrect {FP, FN}, neutral: unknown 
    tConstrained, tStochastic = True, stochastic # stochastic
    print('(estimate_polarity_stacker2) Constrained? {}, Stochastic? {}, Estimate sample type: False'.format(tConstrained, True))
    if not tStochastic: 
        metric = 'fmax'
        # turn probabilities into labels
        # ... can only use training data to estimate the best threshold
        pv_r = model.predict_proba(R.T)[:, 1]
        f1, p_threshold = common.fmax_score_threshold(Lr, pv_r, beta = 1.0, pos_label = 1)
        labels = [int(pv[j] >= p_threshold) for j, pvj in enumerate(pv)]
        pv = labels
        print("(estimate_polarity_stacker2) p_threshold(R): {}, metric: {} example labels:\n... {}\n".format(p_threshold, metric, labels[:10]))

    test_pts = set(np.random.choice(range(T.shape[1]), 20))
    n_conflict_evidence = n_no_positive = 0
    n_tp_dominant = n_tn_dominant = 0
    tHasConflict = False
    n_null = 0 
    assert k_upper > 0 and k_lower > 0, "(estimate_polarity_stacker2) k_upper and k_lower need to be given!"

    M2 = np.zeros(T.shape)

    for j in range(T.shape[1]):  # foreach item/datum
        counts = collections.Counter(Lh[:, j])
        n_pos = counts.get(pos_label, 0)
        n_neg = counts.get(neg_label, 0)
        cv = (n_neg, n_pos) # order: -,+

        P_tp_dominant = pv[j]
        P_tn_dominant = 1.0 - P_tp_dominant

        # [test]
        if (j in test_pts) or (P_tp_dominant >= P_tn_dominant):  # (j in test_pts) 
            print("... mode: unconstrained | P(y|count) | [{}] counts: (-:{}, +:{}) => P_tp_dominant: {}, P_tn_dominant: {}".format(j, cv[0], cv[1], P_tp_dominant, P_tn_dominant))

        if P_tp_dominant > 0 and P_tn_dominant > 0: 
            n_conflict_evidence += 1   # this is almost always true

        p = np.random.uniform(0, 1, 1)[0]
        is_tp_dominant = (True if p <= P_tp_dominant else False) if tStochastic else (True if P_tp_dominant == pos_label else False)

        support_pos, support_neg = [], []
        # choose polarity matrix 
        # Mct = M if is_tp_dominant else M_bar

        tPartialSupport = tNullSupport = False
        # if is_tp_dominant: 
        #     # index into M (otherwise M_bar)
        #     tp_i = np.where(Mct[:, j] == codes['tp'])[0] 
        #     fn_i = np.where(Mct[:, j] == codes['fn'])[0]

        #     support_pos = tp_i
        #     support_neg = fn_i 
        #     # ... the rest of the entries remain neutral 

        #     n_tp_dominant += 1
        # else: # is_tn_dominent or P(y=0|count) > P(y=1|count)
        #     # index into M_bar
        #     tn_i = np.where(Mct[:, j] == codes['tn'])[0] 
        #     fp_i = np.where(Mct[:, j] == codes['fp'])[0] 

        #     support_pos = tn_i
        #     support_neg = fp_i 

        #     n_tn_dominant += 1 
        
        if len(support_pos) == 0 or len(support_neg) == 0: tPartialSupport = True

        if tPartialSupport:  # include the case where neither positive support nor negative support were found
            if is_tp_dominant: 
                if len(support_pos) == 0:  # create positive support
                    # devoid of '+' examples
                    if len(support_neg) > 0: 
                        n_neg = max(k_min, len(support_neg))  # if len(support_neg) == 0, then choose k_min
                        p_th = np.max(T[support_neg, j])  # max of the negative 

                        rows = np.argsort(-T[:, j])  # high to low
                        support_pos = [r for r in rows if T[r, j] > p_th][: n_neg]
                if len(support_neg) == 0: 
                    # devoid of '-' examples
                    if len(support_pos) > 0: 
                        n_pos = max(k_min, len(support_pos))
                        p_th = np.min(T[support_pos, j])  # min of the positive

                        rows = np.argsort(T[:, j])  # low to high
                        support_neg = [r for r in rows if T[r, j] < p_th][: n_pos]
            else:  # is_tn_dominent
                if len(support_pos) == 0:  # create positive support
                    # devoid of '-' examples

                    if len(support_neg) > 0: 
                        n_neg = max(k_min, len(support_neg))  # '+' examples are the negative
                        p_th = np.min(T[support_neg, j])  # min proba of the positive examples
                        rows = np.argsort(T[:, j])  # low to high

                        # '+' polarity examples are those ~ negative examples (having low proba values)
                        #  ... which should not go above the min proba of '+' examples
                        support_pos = [r for r in rows if T[r, j] < p_th][: n_neg]
                if len(support_neg) == 0:  # '+' examples serve as the negative polarity 
                    # devoid of '+' examples 
                    if len(support_pos) > 0: 
                        n_pos = max(k_min, len(support_pos))  # TNs
                        p_th = np.max(T[support_pos, j])  # max of '-' examples

                        rows = np.argsort(-T[:, j])  # high to low
                        support_neg = [r for r in rows if T[r, j] > p_th][: n_pos]

        if len(support_pos) == 0 or len(support_neg) == 0: 
            tNullSupport = True
            n_null += 1

            # then pick k_min for each polarity anyway 
            if is_tp_dominant:
                # positive support
                rows = np.argsort(-T[:, j])  # high to low
                support_pos = rows[:k_upper]
                support_neg = rows[::-1][:k_lower]
            else: 
                rows = np.argsort(T[:, j])  # low to high
                support_pos = rows[:k_upper]
                support_neg = rows[::-1][:k_lower]

        assert len(support_pos) > 0, "There should be at least {} positive-polarity examples but got {}".format(k_min, len(support_pos)) 
        assert len(support_neg) > 0, "There should be at least {} negative-polarity examples but got {}".format(k_min, len(support_neg))
        
        # ... negative and positive support determined
        M2[support_pos, j] = 1
        M2[support_neg, j] = -1

    ### ... end foreach item

    if verbose: 
        msg = ''
        r = n_conflict_evidence/(T.shape[1]+0.0)
        msg += "(polarity_stacker2) tContrained: True | Found n_conflict_evidence: {}, n_no_positive: {} | N={}, ratio_conflict_evidence: {}\n".format(n_conflict_evidence, n_no_positive, T.shape[1], r)
        msg += "...                 n_tp_dominant: {}, n_tn_dominant: {}\n".format(n_tp_dominant, n_tn_dominant)
        msg += "...                 n(pos): {}, n(neg): {}, n(neutral): {} | n(null): {}\n".format(np.sum(M2>0), np.sum(M2<0), np.sum(M2==0), n_null)
        print(msg)

    return M2

def preference_matrix(**kargs):
    return probability_filter(**kargs) 
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
        M = ((Lh == L[None, :]) & (Lh == target_label)).astype(int)
    else: 
        M = (Lh == L[None, :]).astype(int)

    return (M, Lh)
### alias
correctness_matrix = probability_filter

def polarity_matrix(X, L, p_th, reduced_negative=-1, pos_label=1, neg_label=0): 
    Mc, Lh = probability_filter(X, L, p_th)
    return to_polarity(Mc), Lh

def is_color_matrix(Pc):
    codes = np.unique(Pc) 
    tval = True
    for code in codes:
        if not code in Polarity.codes: 
            tval = False
            break 
    return tval
def color_matrix(X, L, p_th, reduced_negative=False, pos_label=1, neg_label=0): 
    """

    Parameters
    ----------
    X: Probability/rating matrix (this could be R, T, [R|T] or any arbitrary matrix)
    L: True labels

    """
    sample_types = Polarity.sample_types
    codes = Polarity.codes

    Mc, Lh = probability_filter(X, L, p_th)  
    # ... Mc is a (0, 1)-matrix where 1: {TP, TN} and 0: {FP, FN}
    # ... Lh is an label matrix estimated via `p_th`
    
    n_users = X.shape[0]

    predict_pos = (L[None, :] == pos_label)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (L[None, :] == neg_label)

    cells_tp = (Mc == 1) & predict_pos  # probability is correct and it predicts positive => TP
    cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos  # probability is wrong while attempt to predict positive => FP
    cells_fn = (Mc == 0) & predict_neg

    Pc = np.zeros(X.shape)
    Pc[cells_tp] = codes['tp']
    Pc[cells_tn] = codes['tn']

    if reduced_negative: 
        Pc[cells_fp | cells_fn] = -1
    else: 
        Pc[cells_fp] = codes['fp']
        Pc[cells_fn] = codes['fn']

    return Pc, Lh

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

def softmax(X, axis=0): 
    # if scipy.sparse.issparse(X): X = X.toarray()
    # column-wise softmax
    X = X.astype(float)
    if X.ndim==1:
        S=np.sum(np.exp(X))
        return np.exp(X)/S
    elif X.ndim==2:
        Xw= np.zeros_like(X)
        m,n = X.shape
        for j in range(n):
            S=np.sum(np.exp(X[:,j])) # column sum-of-exp
            Xw[:,j]=np.exp(X[:,j])/S
        return Xw
    else:
        print("The input array is not 1- or 2-dimensional.")
    return X
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

# ref: demo_ALS
def toImplicitUserItem(fold, merge_=False, save_=False, fill=0): 
    pass

# utils_cf
def toUserItem(fold, to_surprise_format=True, merge_=False, save_=False, fill=0): 
    """
    Convert level-1 training data to user-item dataframe format consisting of the following attributes (columns): 

    ['user_id', 'item_id', 'prediction', 'label'], 

    where user_id corresponds to classifiers 
          item_id corresponds to data points 


    Params
    ------
    fill: placeholder for missing values (e.g. used when masking FPs and FNs)

    """
    # from surprise import BaselineOnly
    from surprise import Dataset, Reader
    import pandas as pd

    # As we're loading a custom dataset, we need to define a reader. In the
    # movielens-100k dataset, each line has the following format:
    # 'user item rating timestamp', separated by '\t' characters.
    delimit = ','
    # header = ['user_id', 'item_id', 'rating']
    # A reader is still needed but only the rating_scale param is required.
    # The Reader class is used to parse a file containing ratings.
    reader = Reader(rating_scale=(0, 1))

    # src_path = utils_sys.getProjectPath(domain='diabetes_cf')
    print('(verify) project path:\n%s\n' % ProjectPath)
    train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
    # print('(toUserItem) dim(train_df):%s' % str(train_df.shape))
    
    # get all data IDs 
    users_ref = train_df.columns.values
    
    splits = ['train', 'test', ] 
    dataset = [train_df, test_df, ]

    # treat classifiers as users, data points as items 
    # dataframe format: 
    #   user_id, item_id, rating
    header = ['user_id', 'item_id', 'rating', ]
    D = []
    for split, ts in zip(splits, dataset): 
        users = ts.columns.values

        ts = ts.reset_index() # convert multilevel index to flat index
        idx = ts['id'].values
        assert len(idx) == len(set(idx)), "Data IDs are not unique!"
        labels = ts['label'].values

        assert set(users) == set(users_ref)

        nU = nUsers = len(users) # number of users/classifiers
        nI = nItems = len(idx)  # number of items/data points

        adict = {h: [] for h in header}
        for i, user in enumerate(users_ref): 
            predictions = ts[user].values
            if i == 0: assert len(idx) == len(predictions)

            adict['user_id'].extend([user] * len(idx)) # repeated
            adict['item_id'].extend(idx)
            adict['rating'].extend(predictions)

        ## mask FP, FN    
        ts = DataFrame(adict, columns=header)   # level 1 training data
        # print('(toUserItem) sample set=%s | n_ids: %d, n_users: %d, dim(ts): %s' % (split, len(idx), nU, str(ts.shape)))
        # print('... ts(n=5):\n%s\n' % ts.head(5))
        
        if save_: 
            l1_data_path = os.path.join(src_path, 'level1')
            fpath = os.path.join(l1_data_path, 'cf-%s-f%d-b%d.csv' % (split, fold, bag_count))  # naming: test-b3-f1-s1.csv.gz
            print('(toUserItem) Saving level-1 CF %s set (dim=%s) to .csv: %s' % (split, str(ts.shape), fpath))
            ts.to_csv(fpath, sep=delimit, index=False, header=True)
            # data = Dataset.load_from_file(fpath, reader=reader, rating_scale=(0, 1))
        D.append(ts)
    ### end foreach split
    
    # individual split (e.g. train, test)
    ret = {}
    for split, ts in zip(splits, D): 
        if to_surprise_format: 
            ret['X_%s' % split] = Dataset.load_from_df(ts[['user_id', 'item_id', 'rating']], reader)
        else:
            # [design] we actually only need the item_ids that distiguishes train and test splits
            ret['X_%s' % split] = ts  # unprocessed dataframe   
        
    # if split == 'test': 
    #     ret['test_offset'] = min(ts['item_id'].values)  # keep track of raw ids to access their predictive values later on
            
    if merge_: 
        D = pd.concat(D, ignore_index=True)
        if to_surprise_format: 
            ret['X'] = Dataset.load_from_df(D[['user_id', 'item_id', 'rating']], reader)
        else: 
            ret['X'] = D

    # [design] may be more convenient to return tuple consisting (combined data, training split, test split)
    #          because we don't want to get test split via train_test_split, instead we have a predefined test split
    return ret 

def t_predict(**kargs): 
    """

    Memo
    ----
    1. Surprise 
       a. Trainset class 
           https://surprise.readthedocs.io/en/stable/trainset.html

       b. prediction 
          https://surprise.readthedocs.io/en/stable/building_custom_algo.html

    2. Usage note
       assign a new rating matrix (R) to the exsiting Trainset object
           ts_total.raw_ratings = R_minus 

    """
    def inspect(ts, message=''):
        # todo: how to view the raw user ids? 

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
    def predict_items(algo, data): # data: a dictionary 
        # [note] Surprise doesn't seem to have an existing interface that allows for a query for the predictive values 
        # in a test split that results in a nice matrix format (T)

        assert 'X_test' in data
        X_test = data['X_test']

        n_users, n_items = len(X_test['user_id']), len(X_test['item_id'])
        print('(predict_items) test split | n_users: %d, n_items: %d' % (n_users, n_items))

        # [todo]
        T = np.zeros((n_users, n_items))
        for i, uid in enumerate(X_test.all_users()): # foreach internal id
            user = str(X_test.to_raw_uid(uid))
            for j, iid in X_test.all_items(): 
                item = str(ts.to_raw_iid(iid))
                T[i, j] = algo.predict(user, item, r_ui=None, clip=False, verbose=True)  # r_ui: true rating

        return T

    from surprise import KNNBasic, NMF
    from surprise import Dataset, accuracy
    from surprise.model_selection import train_test_split, GridSearchCV

    # from surprise.model_selection import cross_validate

    ### Load data 
    #   > Also see demo_surprise.load_from_predefined_folds()
    n_fold = 5
    n_factors = 15
    n_epochs = 300
    tRandomSplit = True

    for fold in [1, ]: # range(n_fold): 
        data = toUserItem(fold, to_surprise_format=True, merge_=True)
        # assert 'X' in data, "No combined data (train + test) generated."
   
        # div(message="I. Biased estimate...", symbol='*', border=2)
        # ## Retrieve the trainset.
        ts_total = data['X'].build_full_trainset()  # total wrt to the train, test split for ensemble leanring
        # inspect(ts_total, message='Trainset instance containing all users and items')

        # ## Build an algorithm, and train it.
        # algo = NMF(n_factors=n_factors, n_epochs=n_epochs) # KNNBasic()
        # algo.fit(ts_total)
        # print('... dim(pu): %s, dim(qi): %s' % (str(algo.pu.shape), str(algo.qi.shape)))  # ... ok
        
        # P, Q = applyMF(fold, **kargs)
        P, Q = applyCoCluster(fold, n_cltr_u=5, n_cltr_i=5, n_epochs=n_epochs)

        # we can now query for specific predicions
        # uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
        # iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

        # # get a prediction for specific users and items.
        # pred = algo.predict(uid, iid, r_ui=4, verbose=True)
        # T = predict_items(algo, data)
        T = np.dot(P, Q.T)

        # use the total training set to build a test set => biased 
        predictions = algo.test(ts_total.build_testset())
        
        # print('Biased accuracy on A,', end='   ') # syntax
        print("Biased accuracy: using a subset of the training data for testing ...")
        accuracy.rmse(predictions)

        div(message="II. Unbiased estimate...", symbol='*', border=2)

        ## unbiased estimate 
        #  usage: data.raw_ratings # where data has to be a Dataset instance
        R = data['X'].raw_ratings  # all ratings including both train and test split
        
        # [note] R is a list
        print('... type(R): %s, size(R):%d, R:\n%s\n' % (type(R), len(R), R[:50] ))
 
        if tRandomSplit: 
            # shuffle ratings if you want
            random.shuffle(R)

            # A = 90% of the data, B = 10% of the data
            ratio = 0.9
            threshold = int(ratio * len(R))
            R_minus = R[:threshold]
            R_test = R[threshold:]

            # use: create a 
            ts_total.raw_ratings = R_minus  # data is now the set A

            # Compute unbiased accuracy on B
            testset = ts_total.construct_testset(R_test)  # testset is now the set B
            predictions = algo.test(testset)
            
            # print('+ Split: Random shuffling > Unbiased accuracy on test split,', end=' ') # syntax
            print('+ Split: Random shuffling > Unbiased accuracy on test split ...')
            accuracy.rmse(predictions)
        else: # pre-defined splits
            R_minus = data['X_train'].raw_ratings
            T = data['X_test'].raw_ratings  # T is a list
            inspect(T, message='Trainset object containing only test split')

            testset = ts_total.construct_testset(T)
            predictions = algo.test(testset)

            # print('+ Split=Predefined > Unbiased accuracy on test split,', end=' ')  # syntax 
            print('+ Split=Predefined > Unbiased accuracy on test split ...')
            accuracy.rmse(predictions)

    return 

def t_load_from_dataframe(**kargs): 
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

def t_memory_based(**kargs): 
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

    # de-mean? 
    # R = demean(R, kind='user')

    train, test = train_test_split(R)
    print('... dim(R): %s, dim(train): %s, dim(test): %s' % (str(R.shape), str(train.shape), str(test.shape)))
    print('... nU: %d, nI: %d' % (n_u, n_m))

    ### compute similarity matrix
    fast_similarity(train, kind='user')
    user_similarity = fast_similarity(train, kind='user')
    item_similarity = fast_similarity(train, kind='item')
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

def t_rating_matrix(**kargs):
    import common
    # global ProjectPath, Domain
    
    ProjectPath = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    Domain = os.path.basename(ProjectPath) 

    kargs['p_threshold'] = 0.5
    for fold in [0, ]: # range(n_fold): 
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold)
        R, T, L_train, L_test, U = toRatingMatrix(fold, **kargs)
        assert train_df.shape == R.T.shape 
        print('dim(train_df): {0} vs dim(R.T): {1}'.format(train_df.shape, R.T.shape))

    return 

def t_cluster(**kargs):
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

def t_confidence_measure(**kargs): 
    from analyze_performance import Analysis
    from stacking import read
    import collections
    import getpass
    from numpy import linalg as LA

    # debugging 
    np.set_printoptions(precision=3)

    domain = 'pf2' # 'diabetes_cf'
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
    print("> Domain: {d} | dim(df): {dim}, n(train): {ntr}, n(test): {nt}".format(d=domain, dim=df.shape, ntr=R.shape[1], nt=T.shape[1]))

    ret = classPrior(L_train, labels=[0, 1], ratio_ref=0.1)  # ratio_ref: if minority class ratio is small than this, then the classes are considered imbalanced
    print("> class ratios: {r}".format(r=ret))

    print("> classifier: {list}".format(list=U))

    tBalanceClassWeights = tBalanceClass = True
    div("... Balance class | balance sample distribution? {t_sample}, balance class conf scores? {t_conf}".format(
        t_sample=tBalanceClass, t_conf=tBalanceClassWeights))


    policy_conf = 'user'
    policy_threshold = 'fmax'
    CR = evalConfidenceMatrix(R, L=L_train, U=U,  
            ratio_users=0.5, 
            ratio_small_class=ret[1], factor_small_class=1.0, 

            policy=policy_conf, # determine filtering dimension (e.g. user, item)
            policy_opt='rating',  # determine optimization type (e.g. rating, tradeoff)
            policy_threshold=policy_threshold, 

                supervised=True, # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure='brier', alpha=100, 

                        balance_and_scale=tBalanceClassWeights,
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],
                        fill=fill_marker, fold=1) # project_path=System.projectPath 
    Cr, Cr_bar, p_threshold, *rest = CR

    CR2 = evalConfidenceMatrix(R, L=L_train, U=U,  
            ratio_users=0.5, 
            ratio_small_class=ret[1], factor_small_class=1.0, 

            policy=policy_conf, # determine filtering dimension (e.g. user, item)
            policy_opt='rating',  # determine optimization type (e.g. rating, tradeoff)

                supervised=True, # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure='brier', alpha=100, 

                        balance_and_scale=False,
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],
                        fill=fill_marker, fold=1) # project_path=System.projectPath 
    Cr2, Cr_bar2, p_threshold2, *rest = CR2 

    # Cr, Cr_bar = evalConfidenceMatrix(T, L=[], U=U,  
    #         ratio_users=0.5, 
    #         ratio_small_class=ret[1], factor_small_class=1.0, 

    #         policy='user', # determine filtering dimension (e.g. user, item)
    #         policy_opt='rating',  # determine optimization type (e.g. rating, tradeoff)
    #             supervised=True, # applicable to all policies (determines how p_threshold is computed if applicable)
    #                 conf_measure='brier', alpha=100, 

    #                     balance_and_scale=tBalanceClassWeights, 
    #                     # masked=params['masked'], mask_all_test=params['mask_all_test'],
    #                     fill=fill_marker, fold=1, L_true=L_test) # project_path=System.projectPath 
    
    print('... delta(norm) on balance_and_scale: %f' % LA.norm(Cr.todense()-Cr2.todense(), 1))

    # pv = to_mean_vector(R, L=L_train, **kargs)
    
    div('Add meta users (R) | create new prediction vectors as a funciton of R (e.g. mean, masked mean) ... ')  
    cutoff = 10
    for policy_conf in ['user', 'item', 'none']: 
        pv = to_mean_vector(R, L=L_train, 
                ratio_users=0.5,  # filtering in the item direction 
                ratio_small_class=ret[1], factor_small_class=1.0,  # used for unsupervised mode 

                policy=policy_conf,  # determining filtering dimension
                policy_threshold=policy_threshold, # determining proba threshold

                    supervised=True, # applicable to all policies (determines how p_threshold is computed if applicable)
                    fill=fill_marker, fold=1)
        assert len(pv) == R.shape[1]
        d_pv = len(pv)
        pv = pv[:cutoff]
        row = R[1][:cutoff]
        print('>>>>>> policy={policy} | R[1][:10]: {row} -> PV (dim: {dim}): {pv}'.format(policy=policy_conf, row=row, dim=d_pv, pv=pv))
    
    div('Add meta users (T) | create new prediction vectors as a funciton of T (e.g. mean, masked mean) ... ') 
    for policy_conf in ['user', 'item', 'none']: 
        pv = to_mean_vector(T, L=[], 
                ratio_users=0.5,  # filtering in the item direction 
                ratio_small_class=ret[1], factor_small_class=1.0,  # used for unsupervised mode 

                policy=policy_conf,  # determining filtering dimension
                policy_threshold=policy_threshold, # determining proba threshold

                    supervised=True, # applicable to all policies (determines how p_threshold is computed if applicable)
                    fill=fill_marker, fold=1)
        assert len(pv) == T.shape[1]
        d_pv = len(pv)
        pv = pv[:cutoff]
        row = T[1][:cutoff]
        print('>>>>>> policy={policy} | T[1][:10]: {row} -> PV (dim: {dim}): {pv}'.format(policy=policy_conf, row=row, dim=d_pv, pv=pv))    
   
    div('Estimate P(y=1|Lh), from which to estimte lh in T ...')
    pos_label = 1
    p_threshold = estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=policy_threshold)
    # jointModel = estimateProbaByLabelMatrix(R, L_train, p_th=p_threshold, pos_label=pos_label)
    Lt = lh = estimateLabels(T, L=[], p_th=p_threshold, pos_label=pos_label, M=(R, L_train)) # joint_model=jointModel
    accuracy = sum(L_test==Lt)/len(Lt)
    print("> Policy='joint model' | accuracy(Lt): {score}".format(score=accuracy))  # pf2: 0.8417085427135679

    # accuracy on minority class
    minority_class = ret['min_class'] 
    precision = sum( (lh == minority_class) & (L_test == minority_class) )/ (sum(lh == minority_class)+0.0)
    recall = sum( (lh == minority_class) & (L_test == minority_class) )/ (sum(L_test == minority_class)+0.0)
    if sum(lh == minority_class) == 0: print("!!! never detected a minority class !!!")
    print('> ... precision: {P}, recall: {R} | minority_class: {l}'.format(P=precision, R=recall, l=minority_class))
    print('> ...... precision: {n}/{N} | recall: {nr}/{Nr}'.format(n= sum( (lh == minority_class) & (L_test == minority_class) ), N=sum(lh == minority_class), 
        nr=sum( (lh == minority_class) & (L_test == minority_class)), Nr=sum(L_test == minority_class) ))

    print('-' * 100)

    # compared to majority vote ...
    lh = estimateLabels(T, L=[], p_th=p_threshold, pos_label=pos_label, joint_model=None)
    accuracy = sum(L_test==lh)/len(lh)
    print("> Policy='majority vote' | accuracy(Lt): {score}".format(score=accuracy))  # pf2: 0.8417085427135679 

    precision = sum( (lh == minority_class) & (L_test == minority_class) )/ (sum(lh == minority_class)+0.0)
    recall = sum( (lh == minority_class) & (L_test == minority_class) )/ (sum(L_test == minority_class)+0.0)
    print('> ... precision: {P}, recall: {R}'.format(P=precision, R=recall))
    print('> ...... precision: {n}/{N} | recall: {nr}/{Nr}'.format(n= sum( (lh == minority_class) & (L_test == minority_class) ), N=sum(lh == minority_class), 
        nr=sum( (lh == minority_class) & (L_test == minority_class)), Nr=sum(L_test == minority_class) ))

    return

def t_factor_similarity(**kargs):
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
    S = evaCrossSimilarity(T, R, kind='item', epsilon=1e-9, unbiased=True)
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
    
    ### demo
    # demo()

    ### data processing
    # t_load_from_dataframe()

    # t_rating_matrix() 

    ### Surprise interface  
    # t_predict()

    ### Memory-based Recommender (no parametric ML models are used; also include clustering-based, non-parametric methods)
    # t_memory_based()

    ### Utililties 
    # t_cluster()

    ### Confidence matrix 
    t_confidence_measure()

    ### similarity and factor transfer 
    # t_factor_similarity()
   
    return

if __name__ == "__main__": 
    test()



