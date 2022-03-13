
import os, sys, re, math, random, time
import collections
import scipy

import pandas as pd
import numpy as np

# Scikit-learn 
from sklearn.preprocessing import normalize

from scipy.stats import entropy
from utilities import normalize
import scipy.sparse as sparse
# from sklearn.preprocessing import normalize

# import knn_models # NOTE: importing this module requires 

# General kNN utilitis 
################################################################
def predict_by_knn(model, indices, Pc): 
    """
    
    Parameters 
    ----------
    `model`: A CFNet-based model that supports `predict(X)` method call, where `X` must be 
             in the user-item-pair format: (user_id, item_id)
    `T`: Probability/rating matrix of the test set 
   
    """
    import data_pipeline as dp
    from tqdm import tqdm

    def predict_core(T, eps=1e-5): 

        n_users, n_test = T.shape 

        # T_avg = np.zeros_like(T)
        Th = T_masked_avg = np.zeros_like(T)
        # T_adj_masked_avg = np.zeros_like(T)
        for i in tqdm(range(n_test)):  # foreach position in the test split (T)
            knn_idx = knn_indices[i] # test point (i)'s k nearest neighbors in R (in terms of their indices)
            # knn_idx = top_indices[i]

            Pc_i = Pc[:, knn_idx].astype(int) # subset the color matrix at kNN indices

            # Compute the mask within these kNN part of the training data
            M = np.zeros_like(Pc_i)
            M[Pc_i > 0] = 1 # polarity > 0 => correct predictions (either TP or TN) => keep their re-estimated values by setting these entries to 1s
      
            # Get re-estimated values for the kNN (of test instance)
            X_knn = dp.make_user_item_pairs(T, item_ids=knn_idx) # structure k-NN in user-item-pair format for CFNet-based models
            # assert X_knn.shape[0] == Pc_i.size 
            
            y_knn = model.predict(X_knn)
            T_knn = y_knn.reshape((n_users, len(knn_idx))) # use len(knn_idx) instead of `k` for the flexibility of selecting fewer candidates
            
            # if i == 10: print(f"[test] knn_idx: {knn_idx}"); print(f"[test] X_knn:\n{X_knn}\n"); print(f"[test] T_knn:\n{T_knn}\n") 
            # assert T_knn.shape[1] <= k, f"T_knn[1] == k(NN): {k} but got {T_knn.shape[1]}"
            # assert T_knn.shape == Pc_i.shape, f"T_knn is a n_users-by-k matrix but got shape: {T_knn.shape}"
            
            # Take column-wise average: use the average across the re-estimated kNNs
            # ti_knn_avg = np.mean(T_knn, axis=1) # take column-wise average (i.e. for each user, take the average among kNNs)
            # T_avg[:, i] = ti_knn_avg

            ti_knn_masked_avg = (M * T_knn).sum(1)/(M.sum(1)+eps) # average from non-zero entries only
            T_masked_avg[:, i] = ti_knn_masked_avg

            # Adjusted Masked Average: Consider degenerative cases in which, for a given base classifier, 
            #           NONE of its predictions in these kNNs are correct
            #           - It's possible that some classifiers never made correct predictions in the context of these kNNs
            #           - Set a default value if that's the case (e.g. average)
            # Th[:, i] = np.where(ti_knn_masked_avg == 0, ti_knn_avg, ti_knn_masked_avg)

        return Th
    return predict_core


# Count-based methods 
################################################################

def most_common_element_and_position(x, pos_key_only=True):
    """

    Parameters 
    ----------
    `x`: a 1D nd.array (e.g. a row vector of a color matrix)

    """

    if len(x) == 0: 
        return (None, -1)

    # It's cool to use the np.argmax( np.bincount() ) idiom but it's not necessarily fast
    # u = np.unique(x) # `x` can be negative (which cannot be handled by bincount)
    # umap = dict(zip(u, range(len(u))))
    # umap_inv = dict(zip(range(len(u)), u))
    # elem = umap_inv[np.argmax(np.bincount([umap[e] for e in x]))] # most common element
    
    counter = collections.Counter(x)
    
    elem = counter.most_common(1)[0][0]
    if pos_key_only: 
        for k, v in counter.most_common(): 
            if k > 0: 
                elem = k
                break

    # position
    pos = np.argmax(np.array(x)==elem)
    return elem, pos
def most_common_element(x, pos_key_only=True): 
    if len(x) == 0: 
        return (None, -1)

    counter = collections.Counter(x)
    elem = counter.most_common(1)[0][0]
    if pos_key_only: 
        for k, v in counter.most_common(): 
            if k > 0: 
                elem = k
                break
    return elem  
def conditional_majority_vote(x, condition=None, pos_key_only=True): 
    # Find the target element by majority vote excluding those with negative values (e.g. those that do not satisfy a desirable condition)
    
    if condition is None: 
        return most_common_element(x, pos_key_only=pos_key_only)

    if isinstance(condition, (list, tuple, np.ndarray)): 
        cx = np.ones_like(condition)
        x = np.array(x) * cx
        return most_common_element(x, pos_key_only=pos_key_only)

    # Otherwise conditions must be a callable (predicate) used to evaluate elements in x 
    # to True or False: If True, assign 1, if False, assign -1 (so that majority vote won't count these elements)
    assert callable(condition), f"Invalid condition given: {condition}"

    t = np.array([conditions(e) for e in x]).astype(int)
    cx = np.where(t==0, -1, 1)
    x = np.array(x) * cx
    return most_common_element(x, pos_key_only=pos_key_only)

# Error Analysis
###############################################################

def analyze_knn(fknn, X_test, L_test, Pc, target_label=1): 
    """

    Parameters 
    ----------
    fknn: An instance of FaissKNN that supports .search() method call
          See `knn_models` for details as for how to create an instance of FaissKNN 
    """
    distances, indices = fknn.search(X_test) # fknn must have been fit
    # NOTE: Reminder if X_test is assigned to a rating matrix (like T), remember to take transpose!

    target_examples = np.where(L_test == target_label)[0]
    Nt = len(target_examples)
    target_examples = np.random.choice(target_examples, min(Nt, 50))
    if sparse.issparse(Pc): Pc = Pc.A

    n_diff_knns = 0
    n_users = Pc.shape[0]
    ptype = 'Positive' if target_label == 1 else 'Negative'
    for i, example in enumerate(target_examples): 
        print(f"> {ptype} example #{i+1}")
        knn_indices = indices[example]
        Pc_i = Pc[:, knn_indices].astype(int)
        print(f"> Pc_{i}:\n{Pc_i}")
        
        max_colors, max_indices = [], []
        for u in range(n_users): 
            color, pos = most_common_element_and_position(Pc_i[u, :], pos_key_only=True)
            max_colors.append(color)
            max_indices.append(knn_indices[pos]) # we want the knn index
        if len(set(max_indices)) > 1: n_diff_knns += 1

        print('> colors: ', max_colors)
        print('> indices:', max_indices)
        print('-' * 50)
    print(f"[info] Found {n_diff_knns} cases for which the majority 'color' does not come from the same training instance")
    
    return

def estimate_ratios(fknn, R, Pc, n_samples=30, codes={}, pos_label=1, neg_label=0, verbose=0, eps=1e-3):
    """
    Given the (fitted) kNN model, find the count ratio of colors among the subset of color matrix (Pc) 
    associated the kNNs. Specifically, this function attempts to answer the following questions: 

    For each training instance in (R) labeled positive, look into its kNNs and find the ratio of TP entries 
    within the kNN-related part of the color matrix (i.e. column-subset of `Pc`). Similiarly, 
    for each negative example, look into its kNN-associated color matrix and find the ratio of TNs. 

    Whenever base classifiers predict positive, do we observe a TP-majority in color matrix (associated with the kNNs)? 
    Whenever base classifiers predict negative, do we observe a TN-majority? 

    Returns
    -------
    A dictionary that maps from labels (0 for negative, 1 for positive) to (mean) ratios. 

    """

    import polarity_models as pmodel 
    import utils_cf as uc

    if sparse.issparse(Pc): Pc = Pc.A
    if len(codes) == 0: codes = pmodel.Polarity.codes

    distances, indices = fknn.search(R.T) # fknn must have been fit

    L_train = pmodel.color_matrix_to_labels(Pc)
    indices_pos = np.where(L_train == pos_label)[0]
    indices_neg = np.where(L_train == neg_label)[0]
    indices_pos = np.random.choice(indices_pos, min(len(indices_pos), n_samples))
    indices_neg = np.random.choice(indices_neg, min(len(indices_neg), n_samples))
   
    n_users, n_train = R.shape 
    ratios = dict.fromkeys([pos_label, neg_label], [])

    # Estimate ratio of colors in order to predict positive
    for i in indices_pos: # foreach training instance labeled positive
        self_knn_i = indices[i] # test point (i)'s k nearest neighbors in R (in terms of their indices)
        Pc_i = Pc[:, self_knn_i]

        counts = pmodel.count_colors(Pc_i)    
        counts_positive = {color:count for color, count in counts.items() if color > 0} # Count only correct predictions TPs, TNs
        
        r = counts_positive[codes['tp']]/(counts_positive[codes['tn']]+counts_positive[codes['tp']]+eps)
        ratios[pos_label].append(r)

    # Estimate ratio of colors in order to predict negative
    for i in indices_neg: # foreach training instance labeled positive
        self_knn_i = indices[i] # test point (i)'s k nearest neighbors in R (in terms of their indices)
        Pc_i = Pc[:, self_knn_i]

        counts = pmodel.count_colors(Pc_i)    
        counts_positive = {color:count for color, count in counts.items() if color > 0} # Count only correct predictions TPs, TNs
        
        r = counts_positive[codes['tn']]/(counts_positive[codes['tn']]+counts_positive[codes['tp']]+eps)
        ratios[neg_label].append(r)

    ratios[pos_label] = np.mean(ratios[pos_label])
    ratios[neg_label] = np.mean(ratios[neg_label])

    return ratios

def estimate_labels_by_matching(fknn, R, Pc, p_threshold,
                                pos_label=1, neg_label=0, verbose=0):
    import polarity_models as pmodel 
    import utils_cf as uc
    from scipy.spatial import distance

    if sparse.issparse(Pc): Pc = Pc.A
    assert R.shape == Pc.shape

    def vector_to_matrix_distance(v, X, distance_fn): 
        return np.sum(distance_fn(v, X[:, j]) for j in range(X.shape[1]))
    def vector_to_matrix_match(v, X): 
        # return positions of match if it exists; return empty array if not
        return np.where(np.all(X == v.reshape(-1, 1),  axis=0))[0] 
            
    def matching_core(T): 
        k_knn = fknn.k # the constant `k` of the kNN
        distances, indices = fknn.search(T.T) # fknn must have been fit
        n_users, n_test = T.shape

        test_points = set(np.random.choice(range(n_test), 10)) # [test]
        y_estimated = []
        for i in range(n_test): # foreach test instance (i.e. each column in T => T[:, i])
            idx_knn_i = indices[i] # test point (i)'s k nearest neighbors in R (in terms of their indices)
            Pc_i = Pc[:, idx_knn_i].astype(int) # column-subset of the color matrix at kNN indices (at positions in R similar to this current test point)

            # If the label were known, how good would the match be in terms of colors (according to a given distance measure)? 

            y_i = 1 # default label
            d_i = np.inf # default distance
            n_i = 0 # default number of exact matches
            history = [] # for testing only
            for label in [1, 0, ]: 
                # Get color vector reprsentation for each column vector of T, given the current labeling guess
                ti_color = pmodel.color_vector(T[:, i], label=label, p_th=p_threshold) # if the label is ..., then its colors are ... 

                # Exact match(es) take precedence if they exist 
                indices_exact_color_match = vector_to_matrix_match(ti_color, Pc_i)
                n_matches = len(indices_exact_color_match)
                if n_matches > 0: 
                    d_label = 0 # set distance to 0 as exact matches when existent, takes precedence than partial matches
                    if n_matches > n_i: # match existent but a small number may be a better sign 
                        n_i = n_matches
                        y_i = label 
                else: 
                    # Compute sum of hamming distances
                    d_label = vector_to_matrix_distance(ti_color, Pc_i, distance_fn=distance.hamming)
                    if d_label < d_i: # sum-of-distance the smaller, the better
                        d_i = d_label
                        y_i = label
                history.append({'color': ti_color, 'label': label, 'distance': d_label, 'n_matches': n_matches})

            if verbose and (i in test_points): 
                print(f"[info] Pc_i:\n{Pc_i}\n")
                msg = ''
                for h in history: 
                    msg += f"... Label = {h['label']}\n"
                    msg += f"... Color(ti): {h['color']}\n" 
                    msg += f"... N_matches(ti): {h['n_matches']}\n"
                    msg += f"...... sum distances: {h['distance']}\n" 
                print(msg); print('-' * 50)

            y_estimated.append(y_i)
        return y_estimated

    return matching_core

def estimate_labels_by_rank(fknn, T, Pc, topn=3, rank_fn=None, 
                    larger_is_better=True, 
                    pos_label=1, neg_label=0, 
                    verbose=0):
    import polarity_models as pmodel 

    if sparse.issparse(Pc): Pc = Pc.A

    # if T.shape[1] == Pc.shape[0]: # T wasn't being transposed 
    #     T = T.T

    if rank_fn is None: 
        rank_fn = compute_entropy
        larger_is_better = False
    
    k_knn = fknn.k # the constant `k` of the kNN
    topn = min(k_knn, topn) # top `n` of the kNN (i.e. top of the top :)
    
    n_users, n_test = T.shape
    distances, indices = fknn.search(T.T) # fknn must have been fit

    test_points = set(np.random.choice(range(n_test), 10))
    top_indices = []
    lh = []
    for i in range(n_test): 
        idx_knn_i = indices[i] # test point (i)'s k nearest neighbors in R (in terms of their indices)
        Pc_i = Pc[:, idx_knn_i].astype(int) # subset the color matrix at kNN indices

        # Choose among top kNNs      
        # [todo] Remove classifiers and kNNs that do not contribute to useful information (i.e. negative row-wise polarities and column-wise polarities)
        
        # Sort each kNN according to their entropy values
        sorted_knn_i = sorted([(compute_entropy(Pc_i[:, j]), j) for j in range(k_knn)], 
                                    reverse=larger_is_better)[:topn] # sort from small to large, entropy-wise

        # [todo] Assign (normalized) weights to each kNN based on the rank function

        # Get the global indices (wrt Pc) of the top N kNNs
        top_knn_i = [ idx_knn_i[knn_ij[1]] for knn_ij in sorted_knn_i]  #  idx_knn_i[j]
        top_indices.append(top_knn_i)
        # Note: knn_ij[1] is the index within Pc_i => idx_knn_i[knn_ij[1]] is the position in Pc

        top_knn_ij = [knn_ij[1] for knn_ij in sorted_knn_i]
        L_knn = pmodel.color_matrix_to_labels( Pc_i[:, top_knn_ij] )
        assert len(L_knn) == len(sorted_knn_i)

        if verbose and (i in test_points): 
            print(f"[info] Pc_{i}:\n{Pc_i}\n")
            print(f"[info] sorted_knn_i (n={topn}):\n{sorted_knn_i}\n")
            assert set(top_knn_i) <= set(idx_knn_i)
            print(f"[info] top_knn_i:\n{top_knn_i}\n")
            print(f"[info] L_knn(n={topn}): {L_knn}")
            print(f"..... top_knn_ij: {top_knn_ij}")
            print(f"..... Pc_{i} local:\n{Pc_i[:, top_knn_ij]}\n")
  
        # [todo] Weighted voting? 
        lh.append(most_common_element(L_knn, pos_key_only=True))


    # for u in range(n_users): 
    #     color, pos = uknn.most_common_element_and_position(Pc_i[u, :], pos_key_only=True)
    #     max_colors.append(color)
    #     max_indices.append(knn_idx[pos]) # we want the knn index
    #     X_knn_best = dp.zip_user_item_pairs(T, item_ids=max_indices)

    return np.array(lh), np.array(top_indices)


# Information-theoretic utilities
################################################################

def compute_impurity(feature, impurity_criterion='entropy', base=2):
    """
    This function calculates impurity of a feature.
    Supported impurity criteria: 'entropy', 'gini'
    input: feature (this needs to be a Pandas series)
    output: feature impurity
    """
    probs = pd.Series(feature).value_counts(normalize=True)
    
    if impurity_criterion == 'entropy':
        impurity = entropy(probs, base=base) # -1 * np.sum(np.log2(probs) * probs)
        # base: default (when base=None) is np.e

    elif impurity_criterion == 'gini':
        impurity = 1. - np.sum(np.square(probs))
    else:
        raise ValueError('Unknown impurity criterion')
        
    return(round(impurity, 3))


def compute_feature_information_gain(df, target, descriptive_feature, split_criterion='entropy', verbose=0):
    """
    This function calculates information gain for splitting on 
    a particular descriptive feature for a given dataset (df[target])
    and a given impurity criteria.
    Supported split criterion: 'entropy', 'gini'

    Reference 
    ---------
    1. https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation/
    """
    if verbose: 
        print('target feature:', target)
        print('descriptive_feature:', descriptive_feature)
        print('split criterion:', split_criterion)
            
    target_entropy = compute_impurity(df[target], split_criterion)

    # we define two lists below:
    # entropy_list to store the entropy of each partition
    # weight_list to store the relative number of observations in each partition
    entropy_list = list()
    weight_list = list()
    
    # loop over each level of the descriptive feature
    # to partition the dataset with respect to that level
    # and compute the entropy and the weight of the level's partition
    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_impurity(df_feature_level[target], split_criterion)
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(round(weight_level, 3))

    if verbose: 
        print('impurity of partitions:', entropy_list)
        print('weights of partitions:', weight_list)

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    if verbose: 
        print('remaining impurity:', feature_remaining_impurity)
    
    information_gain = target_entropy - feature_remaining_impurity
    if verbose: 
        print('information gain:', information_gain)
        print('====================')

    return(information_gain)

def compute_entropy(v, base=2): # [todo] efficiency
    """

    Reference 
    ---------
    1. https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation/
    """
    # from scipy.stats import entropy

    # Method 1
    # probs = np.array([np.sum(v==e)/len(v) for e in np.unique(v)])

    # Method 2: Use pandas 
    probs = pd.Series(v).value_counts(normalize=True)

    return entropy(probs, base=base)

# Similarity measure-related utilities
################################################################

def pairwise_similarity0(ratings, kind='user'):
    """
    Slow version of pairwise_similarity() without vectorization. 
    """
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

def pairwise_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

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
def eval_correlation(R, kind='user', epsilon=1e-9, to_distance=False): 
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

def eval_similarity(R, kind='user', centering=False, epsilon=1e-9): 
 
    if centering: 
        R = center(R, kind=kind) 
    
    if kind.startswith(('u', 'r')):  # user- or row- direction
        # User similarities are evaluated according to how they rate items

        # each user rates items => user: row vectors
        Ru = normalize(R, axis=1, norm='l2')

        # R = (R - user_bias[:, np.newaxis]).copy()   # np.newaxis turns user_bias into a column vector
        sim = np.dot(Ru, Ru.T) # R.dot(ratings.T) + epsilon

    else: 
        # Item similarities are evaluated accodring to how they are being rated by users

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

def eval_cross_similarity(T, R, kind='item', unbiased=True, epsilon=1e-9): 
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
            Tc = center(T, kind='item')
            Rc = center(R, kind='item')

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
        #     Tc = center(T, kind='user')  # 3 * 50
        #     Rc = center(R, kind='user')  # 3 * 100 => 3 * 100, 50  = > 5 users_train vs 2 users_test
        #     D = Rc.dot(Tc.T) + epsilon 
        # else: 
        #     D = T.dot(R.T) + epsilon # leads to the similarity in terms of item_train by item_test 
        
        # # D: users_train vs users_test
        # assert D.shape[0] == R.shape[0]
        # assert D.shape[1] == T.shape[0]
        msg = "Invalid dimension: {kind} unless each user in R and T represent exactly the same number of items (rare to be useful).".format(kind=kind)
        raise ValueError(msg)

    return D 

def center(A, kind='user'): 
    """
    Center input matrix `A` along the row/user or column/item direction

    Memo
    ----
    1. use outer product or numpy broadcasting 

       https://bic-berkeley.github.io/psych-214-fall-2016/subtract_means.html
    """
    # nu, ni = A.shape[0], A.shape[1]
    # broadcast_centered = np.zeros((nu, ni))

    if kind.startswith( ('u', 'r') ):  # user- or row- direction
        row_bias = row_means = np.mean(A, axis=1) 
        
        # row_means_col_vec = row_means.reshape((ratings.shape[0], 1))  # Better: np.newaxis
        # broadcast_centered = ratings - row_means_col_vec
        broadcast_centered = (A - row_bias[:, np.newaxis]).copy() # turns row_means into a column vector

        # [test] should be all 0s
        # assert sum(broadcast_centered.mean(axis=1)) < 1e-9
    else:  # item- or column-direction
        col_bias = A.mean(axis=0)
        broadcast_centered = (A - col_bias[np.newaxis, :]).copy()

        # assert sum(broadcast_centered.mean(axis=0)) < 1e-9

    return broadcast_centered

def predict_by_similarity(R, T, S=None, kind='user', topk=None, canonicalize=True, epsilon=1e-9):
    """

    args
        T: rating matrix for the test set 
        S: similarity matrix

    """ 
    # kind = kargs.get('kind', 'user')
    test_offset = R.shape[1]
    Ra = np.hstack((R, T))

    if S is None: # then use cosine similarity by default 
        Rc = center(Ra, kind=kind) # Rc: mean centered R 
        S = eval_similarity(Rc, kind=kind)  # mean-adjusted cosine similarity
    
    if kind.startswith('i'):     
        assert S.shape[0] == Ra.shape[1], "Similarity metrics (dim=%d) should take on the dimension of items: %d" % (S.shape[0], T.shape[1])
    else: 
        assert S.shape[0] == Ra.shape[0]
    
    if topk: 
        Rt = predict_topk(Ra, S, kind=kind, k=topk)  # users * (items_train + items_test)
        # Note: Slow due to for-loop
    else: 
        Rt = predict_debiased(Ra, S, kind=kind) 

    Rh = Rt[:, :test_offset]
    if canonicalize: Rh = canonicalize_prob(Rh, name='Rh')  

    Th = Rt[:, test_offset: ]
    if canonicalize: Th = canonicalize_prob(Th, name='Th')  

    assert Rh.shape[1] == R.shape[1]
    assert Th.shape== T.shape, "dim(T):{0} but dim(Th):{1}".format(T.shape, Th.shape)

    return (Rh, Th)
 
def predict_new_items(R, T, S=None, kind='item', topk=None, canonicalize=True, epsilon=1e-9): 
    """
    Same as predict_by_similarity() but return only the predictions of T (test split)
    """
    _, Th = predict_by_similarity(R, T, S=S, kind=kind, topk=topk, canonicalize=canonicalize)
    return Th

def predict_topk(ratings, similarity, kind='user', k=40):
    """

    Memo
    ----
    1. suppose a = [0, 1, 2, ... 9], then

       a[:-5:-1] => the last 4 (-1, -2, -3, -4) elements ~> 9, 8, 7, 6 
       # i.e. take the last elements (counting backward due to -1) up until the 4th to last (exclusing the fifth)

       the last k elements in general: 
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
    from utils_cf import confidence_corr
    labels = L_train 

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
    from sklearn.preprocessing import normalize

    A = normalize(A, axis=1, norm='l2')

    # Below is NOT recommmended see Memo [1]
    # sim = np.dot(A, A.T) # A.dot(A.T) # + epsilon
    # norms = np.array([np.sqrt(np.diagonal(sim))])
    # return (sim / norms / norms.T) 

    return np.dot(A, A.T)

def demo_basics(): 

    x = [-1, 1, -2, 3, -2, 10, 3, -1, 2, -2, 3, -2, -2, 4, -5, 3, 1, -2, 7, -5, -5, -5, 7, -5, -5]
    x_freq, x_pos = most_common_element_and_position(x)
    print(f"> x:\n{x}\n")
    print(f"> freq(x): {x_freq}, pos_freq(x): {x_pos}")


def test(): 

    # Basic utilities 
    demo_basics()

    return

if __name__ == "__main__": 
    test()