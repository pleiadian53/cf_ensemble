
import os, sys, re, math, random, time
import collections
import scipy

import numpy as np

# Scikit-learn 
from sklearn.preprocessing import normalize


from utilities import normalize
import scipy.sparse as sparse
# from sklearn.preprocessing import normalize

class FaissKNN:
    def __init__(self, k=5, normalize=False):
        self.index = None
        self.y = None
        self.y_tag = None # other meta data for the label/target such as polarities, colors
        self.k = k
        self.normalize_input = normalize

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1]) # Each x in X is in row-vector format i.e. X has shape  (n_instances, n_dim)
        # Note: Rating matrix (X), however, is in column-vector format; therefore, we need to remember to take transpose before using it as an input
  
        if self.normalize_input: 
            X = normalize(X, axis=1) # X is in row-vector format

        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        # shape(distances): (n_instances, k)
        # shape(indices):   (n_instances, k)

        votes = self.y[indices] # note: shape(votes)=shape(indices)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        # np.bincount([1, 1, 1, 0, 1, 0, 0, 0, 1, 1]) 
        # ~> array([4, 6]) because index 0 occurs 4 times, and 1 occurs 6 times
        return predictions
    def search(self, X): 
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return distances, indices


# Count-based methods 
################################################################
def most_common_element_and_position(x, pos_key_only=True):
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