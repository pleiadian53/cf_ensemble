import os, sys, re, math, random, time
from sys import argv
import scipy
import scipy.io
import scipy.sparse as sparse
# import scipy.stats as stats
import numpy as np
from pandas import DataFrame, Series

# Scikit-learn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

# CF modules
# from analyzer import is_sparse
import cf_spec
from cf_spec import System
import common, utilities

import utils_knn as uknn
import utils_sys

class Polarity(object): 
    sample_types = ['tp', 'tn'] + ['fp', 'fn']
    codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
            'unk': 0, 't': 3, 'f': -3, 
            'pos': 1, 'neg': -1, '+': 1, '-': -1}

# Polarity and color matrices
######################################################################

def is_hard_filter(P, n_codes_ref=None):

    if sparse.issparse(P): P = P.A
    n_codes = np.unique(P)
    
    if n_codes_ref is None: # how many unique codes do we expect to observe in a hard filter?
        n_codes_ref = len(set(Polarity.codes.values()))

    if len(n_codes) > n_codes_ref:
        return False 
  
    return True
# [alias] 
def is_mask(P, n_codes_ref=None): 
    # A hard filter IS a "mask" with 0s and 1s (instead of continous values in [0, 1] like those in soft filters)
    return is_hard_filter(P, n_codes_ref=n_codes_ref)

def to_hard_filter(P, r_th, n_codes_ref=None, dtype='int', inplace=False):
    if is_hard_filter(P, n_codes_ref): 
        return P # no-op

    if sparse.issparse(P): P = P.A

    if isinstance(r_th, float): 
        pass # no-op
    else:
        if isinstance(r_th, list): 
            if len(r_th) != P.shape[0]: raise ValueError(f"Incompatible number of thresholds: {len(r_th)} with shape(X)={P.shape}")
            r_th = np.array(r_th)
        else: 
            if not isinstance(r_th, np.ndarray): 
                raise ValueError(f"Invalid r_th: {r_th}")
        r_th = r_th.reshape(-1, 1) # convert r_th to a column vector
    
    P_new = P if inplace else P.copy()
    P_new[P_new >= r_th] = 1
    P_new[P_new < r_th] = 0

    return P_new.astype(dtype)

def infer_reliability_threshold(X, L, P, p_th, *, policy_threshold='balanced', min_r_th=1e-3, verbose=0):
    """
    Given the probability/rating matrix (X), labels (L) and predicted filter (P) (and additionally the probability threshold 
    `p_th` as we need it to infer the label predictions from X), 

    find, for each BP/user, 

    their optimal reliability thresholds individually according to `policy_threshold`

    Note that reliability threshold has a similar meanings and implications as probability threshold except that 
    their derivations are different. Reliability of a probability (or rating in general) 
    is also bounded within [0, 1] where 0s represent unreilable (BP-generated, conditional) probabilities 
    and 1s represent reliabble probabilities. 


    Parameters
    ----------
    P: A "soft" probability filter estimated via a polarity model (e.g. a polarity classifier), in which 
       the reliability of a rating (in R) is represented by a "soft" score between 0 and 1 (i.e. a scalar in [0, 1]); 
       this is in contrast to a "hard" probaiilty filter where reliable entries are typically encoded by a 
       "hard" 1 and unreliabe entries are encoded by a "hard" 0. 

    X: A rating matrix 
    L: The class label associated with the rating matrix X
    p_th: probability thresholds 
    """
    import utils_classifier as uclf

    R = T = None
    if isinstance(X, (tuple, list)): 
        R, T = X
    else: 
        R = X # assume that X contains only the training data
    n_train = R.shape[1]

    if sparse.issparse(P): P = P.A

    ###############################
    Pr = Pt = None
    if isinstance(P, (tuple, list)):
        Pr, Pt = P
        assert Pt.shape == T.shape
    else: 
        Pr = P
    assert Pr.shape == R.shape
    ###############################

    if len(L) > n_train: L = L[:, n_train]

    # The input `P` must be a soft filter
    assert not is_hard_filter(Pr), "Input probabilty filter is a hard filter! Please provide a soft filter."

    # Infer proportion of reliable entries from the training set (i.e. R, rating matrix derived from the training set)
    Po, Lh = probability_filter(R, L, p_th) # Note that probability_filter() gives 0-1 encoding
    # Note: `Po` takes on values {0, 1}; `Po` is the ground truth while `Pr` is the predicted filter

    # Infer, for each user/classifier, a reliability threshold using the training data
    # NOTE: 
    #   1. Reliability threshold may depend on the class label (can we really decouple reliability threshold from class label?)
    r_th = []
    if verbose > 1: print(f"[threshold] Using policy: {policy_threshold}")
    if policy_threshold == 'prior': # A. Global prior
        r_th_global = (Po == 1).sum()/(Po.size+0.0)
    
        for i in range(R.shape[0]):
            # r_th_tp = Pr[i][(Po[i] == 1) & (L == 1)].min() # reliability threshold for TPs
            # r_th_tn = Pr[i][(Po[i] == 1) & (L == 0)].min() # reliability threshold for TNs
            
            idx_reliable = (Po[i] == 1)
            if idx_reliable.sum() > 0: 

                # A. Minimum
                # r_th_i = Pr[i][ idx_reliable ].min() # minimum reliability degree in order to be consider reliable
                # ... min doesn't seem to be a reliable threshold

                # B. Prior
                r_th_i = idx_reliable.sum()/(n_train+0.0) # use prior as the threshold

            else: 
                r_th_i = r_th_gobal # global prior
                
                # idx_unreliable = (Po[i] == 0)
                # if idx_unreliable.sum() > 0:
                #     r_th_i = Pr[i][ idx_unreliable ].max() # could this be a lower bound on the reliabilty threshold? Not necessarily
                # else: 
                #     raise ValueError(f"Found ill-formed polarity at i={i}:\n{Po[i]}\n")

            r_th.append(r_th_i)

    else: # B. Balanced accuracy
        threshold_fn = lambda x, y: 0.5
        if policy_threshold.startswith('bal'): # B. Balanced accuracy
            thresholds_fn = uclf.acc_max_threshold
        elif policy_threshold.startswith('f'): # C. fmax
            thresholds_fn = uclf.fmax_threshold
        else: 
            raise NotImplementedError

        labels = np.ravel(Po) # true filter value (reliable entries (1s) are those corresponding to TPs and TNs) 
        predictions = np.ravel(Pr) # predicted filter value
        r_th_global = threshold_fn (labels, predictions)

        for i in range(R.shape[0]): # foreach BP/user row i, find its threshold individually

            idx_reliable = (Po[i] == 1)
            if idx_reliable.sum() > 0: 
                labels = Po[i] # i-th row P
                predictions = Pr[i]
                r_th_i = threshold_fn (labels, predictions)
            else: 
                r_th_i = r_th_gobal # global prior
                
            r_th.append(r_th_i)

    ##########################################################################
    r_th = np.array(r_th)

    return Po, r_th

def infer_probability_filter(X, L, P, p_th, *, policy_threshold='balanced', 
                                use_ground_truth_fitler=True, 
                                polarity_encoding=False,
                                min_r_th=1e-3, verbose=0): 
    """

    Parameters
    ----------
    P: A "soft" probability filter estimated via a polarity model (e.g. a polarity classifier), in which 
       the reliability of a rating (in R) is represented by a "soft" score between 0 and 1 (i.e. a scalar in [0, 1]); 
       this is in contrast to a "hard" probaiilty filter where reliable entries are typically encoded by a 
       "hard" 1 and unreliabe entries are encoded by a "hard" 0. 

    X: A rating matrix 
    L: The class label associated with the rating matrix X
    p_th: probability thresholds  
    """
    Pr = Pt = None
    if isinstance(P, (tuple, list)):
        Pr, Pt = P
    else: 
        Pr = P

    ###############################################
    Po, r_th = infer_reliability_threshold(X, L, P, p_th, 
                        policy_threshold=policy_threshold, 
                        min_r_th=min_r_th, verbose=verbose)
    assert len(r_th) == Pr.shape[0]
    ###############################################
    # NOTE: `Po` holds the true filter value for the training split (R)

    # [test] 
    # For training data, we already have the "reliability matrix" (Po) (which is the same as the probability filter) 
    # but let's double check if the estimate is consistent with the one inferred from the data (Po_pred)
    # Objective: want `Po_pred` ~ `Po`

    Pr_hard = (Pr >= r_th.reshape((-1, 1))).astype(int) # note: .reshape turns r_th into a column vector 
    if not np.array_equal(Pr_hard, Po): 
        n_diff = (Pr_hard != Po).sum()
        print(f"Conflict in reliability matrix estimate: { n_diff } entries are different")
        print(f"Error rate: {n_diff/(Po.size+0.0)}")

    # Infer filter values for the test set if given
    if Pt is None: 
        Po_pred = Po if use_ground_truth_fitler else Pr_hard
        # Po_pred = Pr_hard # the predicted hard filter is the just the one inferred from the training set
    else: 
        Pt_hard = (Pt >= r_th.reshape((-1, 1))).astype(int) # note that `r_th` is the thresholds inferred only from training data
        
        if use_ground_truth_fitler: 
            Po_pred = np.hstack((Po, Pt_hard)) 
        else:
            Po_pred = np.hstack((Pr_hard, Pt_hard)) # combine "hard" filters for training and test sets
        
        # [Q] Should we use the "true" filter values for the training data instead of the predicted values?
        # Po_pred = np.hstack((Po, Pt_hard))

    if polarity_encoding: 
        Po_pred = to_polarity(Po_pred, verify=verbose > 1)
        
    return Po_pred, r_th

def filter_by_majority_vote(X, p_th, **kargs): 
    import utils_cf as uc

    # Parameters
    p_threshold = p_th

    if isinstance(p_threshold, float): p_threshold = np.repeat(p_threshold, X.shape[0])
    if len(p_threshold) != X.shape[0]: raise ValueError(f"The size of `p_threshold` must equal nrow(X): {X.shape[0]} but given {len(p_threshold)}")
    pos_label = kargs.get('pos_label', 1)
    dtype = kargs.get('dtype', int)

    lh_maxvote = uc.estimateLabels(X, p_th=p_threshold, pos_label=pos_label) 
    Lh = uc.estimateLabelMatrix(X, p_th=p_threshold, pos_label=pos_label)
    
    P = (Lh == lh_maxvote).astype(dtype) # if the label prediction matches the majority vote, then assign weight of 1 o.w. 0

    return P

# source: utils_cf
def probability_filter(X, L, p_th, *, to_polarity=False, target_label=None, verbose=0): 
    """
    Compute a binary matrix in which 1 represents a correct prediction (i.e. TP or TN), 
    and 0 represents a false prediction (i.e. FP or FN). Predicted labels (Lh) are determined by the given probability threshold (p_th). 
    
    Lh is a function of `X` and `p_th`

    Lh are compared with the "ground truth" L to determine the correctness. 

    Parameters
    ----------
    `X`: Probability/rating matrix generated by base classifiers (or users)
    `L`: A vector of class labels associated with `X`

    Returns 
    -------
    A 2-tuple, where 

    First matrix (M) is a probability filter comprising 0s and 1s (1 for TPs and TNs and 0 for FPs and FNs)
    Second matrix (Lh) is an estimated label matrix given proba thresholds
    """
    import utils_cf as uc
    Lh = uc.estimateLabelMatrix(X, p_th=p_th) # this is a label matrix NOT an estimate for true labels (lh); no 'majority vote' involved
    
    L = np.asarray(L) # ensure L is of nd.array type
    # Check if the labeling matrix is consistent with the true/guessed labels (L)
    # - If consistent, then mark as 1; if not, mark by 0
    if target_label is not None: 
        M = ((Lh == L[None, :]) & (Lh == target_label)).astype(int)
        # Note: 
        # Check the consistency between "true/guess" (L) labels and predicted labels (as determined by X and p_th)
        # E.g. Lh: (5 x 10), L: (10, )
        #      L[None, :] => (10, 1), by adding an extra dimension at axis=0
        #      Lh == L[None, :] compares row-wise between Lh[i] and L due to the broadcasting rule
    else: 
        M = (Lh == L[None, :]).astype(int)

    if to_polarity: 
        M = preference_to_polarity(M, verify=verbose > 1)

    return (M, Lh)
def preference_matrix(X, L, p_th, **kargs):
    return probability_filter(X, L, p_th, **kargs) 
def correctness_matrix(X, L, p_th, **kargs):
    return probability_filter(X, L, p_th, **kargs)
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
def color_matrix(X, L, p_th, reduced_negative=False, codes={}, pos_label=1, neg_label=0): 
    """

    Parameters
    ----------
    X: Probability/rating matrix (this could be R, T, [R|T] or any arbitrary matrix)
    L: True labels

    """
    sample_types = Polarity.sample_types
    if len(codes)==0: codes = Polarity.codes

    Pf, Lh = probability_filter(X, L, p_th)  
    # ... Pf is a (0, 1)-matrix where 1: {TP, TN} and 0: {FP, FN}
    # ... Lh is an label matrix estimated via `p_th`
    
    n_users = X.shape[0]

    predict_pos = (Lh == pos_label)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == neg_label)

    cells_tp = (Pf == 1) & predict_pos  # probability is correct and it predicts positive => TP
    cells_tn = (Pf == 1) & predict_neg
    cells_fp = (Pf == 0) & predict_pos  # probability is wrong while attempt to predict positive => FP
    cells_fn = (Pf == 0) & predict_neg

    Pc = np.zeros(X.shape)
    Pc[cells_tp] = codes['tp']
    Pc[cells_tn] = codes['tn']

    if reduced_negative: 
        Pc[cells_fp | cells_fn] = -1
    else: 
        Pc[cells_fp] = codes['fp']
        Pc[cells_fn] = codes['fn']

    return Pc, Lh

def color_vector(col_vec, label, p_th, reduced_negative=False, pos_label=1, neg_label=0):
    """

    """
    col_vec = np.asarray(col_vec)
    # if col_vec.ndim == 1: 
    #     pass # no-op
    if col_vec.ndim == 2: 
        assert np.squeeze(col_vec).ndim == 1
    col_vec = col_vec.reshape(-1, 1) # turn into a column vector

    Pc_i, Lh_i = color_matrix(col_vec, np.array([label, ]), p_th=p_th)
    colors = np.squeeze(Pc_i)
    # assert colors.ndim == 1 
    
    return colors

def color_matrix_to_labels(Pc, codes={}, pos_label=1, neg_label=0): 

    P_is_sparse = False
    if sparse.issparse(Pc): 
        Pc = Pc.A
        P_is_sparse = True
    if len(codes)==0: codes = Polarity.codes
    
    n_users, n_items = Pc.shape 

    L = []
    for j in range(n_items): 
        colors = set(Pc[:, j])
        if (codes['tp'] in colors) or (codes['fn'] in colors): 
            L.append(pos_label)
        else: 
            L.append(1-pos_label)

    return np.array(L)
def color_vector_to_label(colors, codes={}, pos_label=1, neg_label=0):
    # colors: colors associated with a particular column of the color matrix (Pc)

    if len(codes)==0: codes = Polarity.codes
    if (codes['tp'] in colors) or (codes['fn'] in colors):
        return pos_label
    return neg_label

def count_colors(Pc, codes={}):
    if Pc.size == 0: 
        return {}
    
    if len(codes) == 0: codes = Polarity.codes

    if sparse.issparse(Pc): Pc.A = Pc

    (unique, count) = np.unique(Pc, return_counts=True)
    counts = dict(zip(unique, count)) # map unique element to its count

    for color in codes.values(): 
        if not color in counts: 
            counts[color] = 0
    return counts

def sort_colors(Pc, reverse=True, codes={}):
    if Pc.size == 0: 
        return np.array([])

    if sparse.issparse(Pc): Pc.A = Pc

    counts = count_colors(Pc, codes=codes) # map unique element to its count
    unique, count = zip(*counts.items())

    if reverse:
        elem_sorted = unique[np.argsort(-count)] # Sort in descending order according to count
    else: 
        elem_sorted = unique[np.argsort(count)] # Sort in count-ascending order 

    # target, target_count = elem_sorted[0], cmap[elem_sorted[0]]
    # if pos_key_only: 
    #     for elem in elem_sorted: 
    #         if elem > 0: 
    #             target = elem
    #             target_count = cmap[target]
    #             break 
    return elem_sorted, lookup

def verify_colors(Pc, X=None, codes={}):
    # Foreach data point (column vector): 
    #    TP must pairs with FN (if any)
    #    TN must pairs with FP (if any)

    if len(codes)==0: codes = Polarity.codes

    P_is_sparse = False
    if sparse.issparse(Pc): 
        Pc = Pc.A
        P_is_sparse = True
    if X is not None: 
        assert Pc.shape == X.shape

    n_users, n_items = Pc.shape
    max_colors = []
    for j in range(n_items):
        colors = np.unique(Pc[:, j])
        max_colors.append(max(colors)) # TP or TN exists? 

        nc = len(colors)
        assert nc <= 2, f"There can be at most 2 colors in one instance but got n={nc}"
        if nc == 2: 
            if codes['tp'] in colors: 
                assert codes['fn'] in colors, f"TP(2) can only pair with FN (-1) but got: {colors}"
            elif codes['tn'] in colors: 
                assert codes['fp'] in colors, f"TN(1) can only pair with FP (-2) but got: {colors}"
            else: 
                assert not (codes['fp'] in colors) or (codes['fn'] not in colors)
                assert not (codes['fn'] in colors) or (codes['fp'] not in colors)
                raise ValueError(f"Invalid color mixture: {colors} at position col={j}")

    # Convert back to sparse matrix format if necessary
    # if P_is_sparse: 
    #     Pc = sparse.csr_matrix(Pc)
    # ... it turns out that this is unnecssary because Pc's sparseness will remain unchanged after the call
    return np.array(max_colors)

def polarity_to_preference(**kargs): 
    return to_preference(**kargs)
def to_preference_matrix(**kargs):
    return to_preference(**kargs)
def to_preference(Po, neutral=0.0):
    # import scipy.sparse as sparse 
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
    """ 
    Convert 0-1 encoding to {-1, 1} encoding
    """
    # from preference matrix to polarity matrix

    # if verify: 
    #     vmin, vmax = np.min(M), np.max(M)
    P = np.ones(M.shape)  
    if sparse.issparse(M):  
        M = M.A    
        P[M == 0] = -1    # incorrect predictions (FP, FN) => negative polarity 
        P = sparse.csr_matrix(P)
    else: 
        P[M == 0] = -1 
    return P
def to_polarity_matrix(M):
    # preference matrix {0, 1} to polarity matrix 
    return to_polarity(M)

def to_colored_preference_matrix(**kargs): 
    # use: for approximating ratings
    return to_colored_preference(**kargs)
def to_colored_preference(M, codes):
    # import scipy.sparse as sparse
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

def from_color_to_polarity(M, codes={}, verify=False): 
    """
    Convert from color (polarity) matrix to regular polarity matrix, whose non-zero entries 
    can only be either 1 or -1
    
    Converting color matrix to polarity matrix is useful for preference score approximation

    Note that both color matrix and polarity matrix can have 0s, whose corresponding probabilities 
    will not enter the optimization objective. 
    """
    # import scipy.sparse as sparse
    
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

def from_color_to_preference(M, codes={}, verify=True): 
    # import scipy.sparse as sparse
    
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
    # import scipy.sparse as sparse
    
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

# Training data creation
#########################################################################

def make_seq2seq_training_data(R, Po=None, L=None, include_label=True, **kargs): 
    """

    Parameters
    ----------
    R:  Rating matrix where rows correspond to classifers/users and columns correspond to data points
    Po: A Probability filter where 1 represents reliable entries and 0 represents unreliable entries), whose 
        column vectors serve as "labels" for R

        That is Po[:, i] is the label for the corresponding column vector R[:, i] for all i in [0, R.shape[1]]

    L: class labels associated with R

    filter_type: 'mask', 'polarity', 'color'
       - mask (probability filter): uses 0-1 encoding, where 0 represents incorrect predictions (FPs, FNs) and 1 represents correct predictions (TPs, TNs)
       - preference: uses 0-1 encoding but, unlike a mask (or a probability filter), 0 represents "not preferred" whereas 1 "preferred" 
         
         Although 'mask' and 'preference matrix' both uses 0-1 encoding, they are derived differently

       - polarity: similar to "mask" but uses {-1, 1}-encoding

                   if we consider "neutral" particles (e.g. ratings with high uncertainty), 
                   then 0 can be a valid value as well, which generalizes to a {-1, 0, 1}-encoding


       - color: color matrix that generalizes polarity matrix and further distingues between: 
                TPs (2), TNs (1), FPs (-2), FNs (-1)

                neutral particles are also possible and represented by 0s 


    Returns
    -------
    A 2-tuple (X, Y), where 
    X: 
    Y: 
    """
    import utils_cf as uc

    verbose = kargs.get('verbose', 0)
    filter_type = kargs.get('filter_type', 'mask') # Options: 'mask' ({0, 1}-encoding), 'polarity', ''

    p_threshold = kargs.get('p_threshold', [])
    if Po is None: # If the probability filter is not provided externally, we need to estimate using a default method
        assert len(p_threshold) > 0 and L is not None
        assert (len(p_threshold), len(L)) == R.shape

        if filter_type.startswith( ("mask", "proba", "filter") ): # 0-1 encoding
            Po, _ = probability_filter(R, L, p_threshold)
        elif filter_type.startswith( "po" ):
            Po, _ = polarity_matrix(R, L, p_threshold) # {-1,1} encoding, possibly including 0
        elif filter_type == "color":
            Po, _ = color_matrix(R, L, p_threshold) # {-2, -1, 1, 2} encoding; possibly including 0 and more

        elif filter_type.startswith("pref"):
            raise NotImplementedError("For preference filter, please provide a pre-computed `Po`")
        else: 
            raise ValueError(f"Unrecognized filter type: {filter_type}")
        
    if L is not None: assert len(L) == R.shape[1]

    assert R.shape == Po.shape
    
    if sparse.issparse(Po): Po = Po.A

    X = [] # input sequences (one training instance consists of `n_users` ratings per data point, where `n_users` = R.shape[0])
    Y = [] # labels or output sequences

    if not include_label: 
        for j in range(R.shape[1]):

            X.append( R[:, j].reshape(-1, 1) ) # number of "timesteps" is the number of classifiers/users, where ... 
            # ... each object/rating is represented by 1 feature value, which is the rating itself

            Y.append(Po[:, j].reshape(-1, 1) ) # `X -> Y` maps a sequence of ratings to a sequence of polarities 
    else: 
        assert len(p_threshold) > 0 and L is not None

        # Either use zero-padding or compute a label place older for the training instances 
        # (LSTM requires that the input and output sequence have fixed, pre-specified lengths)

        # Why do we use a placeholder (or a heuristic label) instead of the true label? 
        L_heuristic = kargs.get('L_heuristic', "majority_vote")

        # Indeed we know `L_train` but we may not want to rely on it too much. After the model is trained at the prediction/test time, 
        # we will no longer have the labeling information (i.e. when calling model.predict(X_test)); 
        # this means that we cannot make the test instances with the label in it (because ultimately the class label for T is what 
        # we are aiming to predict)
        # 
        # What we could do instead is to infer the labeling using the ratings and probability thresholds, which are equally accessible 
        # during the prediction/test time

        # Heusristic #1: Majority vote
        if isinstance(L_heuristic, str): 
            if L_heuristic.startswith( ('maj', 'max') ): # majority vote
                L_heuristic = uc.estimateLabels(R, p_th=p_threshold) # this heuristically guessed labeling by default uses majority vote
            elif L_heuristic.startswith( 'zero' ): # zero-padding
                L_heuristic = np.zeros(R.shape[1]) # [design] we could instead just assign 0 in the loop
            else: 
                raise NotImplementedError(f"Unrecognized label heuristic: {L_heuristic}")
        else: 
            msg = f"Invalid `L_heuristic` (hint: L_heuristic should be a list/ndarray of same size as R.shape[1]): {L_heuristic}"
            assert isinstance(L_heuristic, (list, np.ndarray)) and len(L_heuristic) == R.shape[1], msg

        for j in range(R.shape[1]):
            X.append( np.vstack( (R[:, j].reshape(-1, 1), L_heuristic[j]) )) # Append either a heustically guessed label or zero-pad at the end of the rating sequence ... 
            # Note: 
            # Why don't we add class label to X? Because at the prediction time, we will not know the class label (which is to be predicted) 
            
            Y.append( np.vstack( (Po[:, j].reshape(-1, 1), L[j]) )) # Also append the class label L[j] and assume positive class as having positive polarity

    X = np.array(X).astype('float')
    Y = np.array(Y).astype('float')
    if verbose: 
        print(f"[info] shape(X): {X.shape}, shape(Y): {Y.shape}")

    return (X, Y) # X and Y should be in 3D, which is required by TF's LSTM 


# Models
##########################################################################


def mask_by_filter(C, Po, is_unweighted=False, weight_neutral=0.0, weight_negative=0.0, sparsify=True, verbose=1):
    """
    Given polarity matrix (Po), mask the neutral and negative entries in the confidence matrix so that 
    they do not enter the optimization objective (i.e. latent factors will not be made to approximate 
    these entries well because these entries do not matter). 

    Note
    ----
    1. this routine is effectively the same as make_cp() now ... 02.07.22
    """
    # import scipy.sparse as sparse

    if sparse.issparse(Po): Po = Po.A

    C_is_sparse = False
    if is_unweighted: 
        if verbose: print("(make_cn) Using UNWEIGHTED confidence matrix (with all C[i][j] having equal weights) to approximate ratings ...")
        Cn = np.ones(C.shape) 
    else: 
        if verbose: print("(make_cn) Using WEIGHTED confidence matrix to approximate ratings ...")
        if sparse.issparse(C):
            Cn = C.toarray()  # Cx.toarray()  # copying
            C_is_sparse = True
        else: 
            Cn = np.copy(C)

    # Given a rating matrix (R), entries with zero confidence score won't enter optimization objective (i.e. these entries 
    # will not contribute to the derviation of latent factors)
    Cn[Po == 0] = weight_neutral    
    Cn[Po < 0] = weight_negative    
        
    # NOTE: A simpler statement would be the following but this is a re-assignment operation, which is not the same as "re-weighting"
    #       because then we'll have no control over the weight
    # Cn = Cn * (Po > 0).astype('float32') 

    # At this point, Cn is in "dense" format 

    if C_is_sparse: # then Cn must also be sparse to be consistent
        Cn = sparse.csr_matrix(Cn)
    return Cn

# [alias] For backward compatibility
make_cn = mask_by_filter 

def make_cp(C, Po, is_unweighted=False, weight=0.0, sparsify=True):
    """
    Similar to make_cn() but mask only the nuetral (i.e. entries with so much uncertainty 
    that we do not know if they are TP, TN or FP, FN)
    """
    if sparse.issparse(Po): Po = Po.A

    C_is_sparse = False
    if is_unweighted: 
        if verbose: print("(make_cp) Using UNWEIGHTED confidence matrix (with all C[i][j] having equal weights) to approximate ratings ...")
        Cp = np.ones(C.shape) 
    else: 
        if verbose: print("(make_cp) Using WEIGHTED confidence matrix to approximate ratings ...")
        if sparse.issparse(C):
            Cp = C.toarray()  # Cx.toarray()  # copying
            C_is_sparse = True
        else: 
            Cp = np.copy(C)

    Cp[Po == 0] = weight

    # dtype(Cn) must be consistent with dtype(Po)
    if sparse.issparse(Po) or sparsify: # then Cn must also be sparse to be consistent
        Cp = sparse.csr_matrix(Cp)
    return Cp

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
    cells_tp = (Mc == 1) & predict_pos  # estimated
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
    raise NotImplementedError

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

# [todo]
def get_vars_vstats(R, i, j, p_th, Rm=None, C=None, Lh=None, p_model={}, 
        r_min=0.1, name='', index=0, verbose=False, wsize=10, to_dict=False, neg_label=0, pos_label=1):
    fv = []  # features 
    fvn = []  # feature names
    if to_dict: 
        return dict(zip(fvn, fv))
    return np.array(fv)


# Evaluation Metrics
#########################################################

def test_polarity(T, labels, Pref=None, p_th=[], lh=[], name='T', pos_label=1, neg_label=0, title=''):
    # import scipy.sparse as sparse

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

def ratio_of_alignment2(Xpf, Mc, Lh, verify=True, verbose=True, message=''):
    """

    Parameters 
    ----------
    `Xpf`: A predicted preference matrix comprising 0s and 1s.

           "1s" correspond to preferred entries in X[i][j], where 
           we've used X to denote a probaility/rating matrix
           "0s" correspond to those not preferred (or "bad" probabilities). 

           `Xpf` is a "predicted" preference matrix because it's an predictive 
           output from a CF ensemble algorithm applied to X. 

    `Mc`:  A probability filter determined by: 

           1. True (or guessed) labels (L or Lh)
           2. probability thresholds (of the base classifiers)

    `Lh`:  A label matrix (as determined by rating matrix X and probability threshold p_threshold)
           
           `Mc` is a derived quantity of `Lh`; i.e. 

           with `X` and `p_threshold` we get `Lh`: 
           (X, p_th) => Lh

           with `Lh` and `L` we get `Mc`: 
           (Lh, L) => Mc, by checking the consistency between Lh (predicted labels) and true labels (L)


    """
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

    # Among all preferred entries, what's the fraction of them that are TPs and TNs? 
    accuracy = (n_tp_hit + n_tn_hit)/(n_pref+0.0+1e-3)

    ret['precision_pref'] = precision_pref = n_tp_pref/(n_tp_pref+n_fp_pref+1e-3)
    ret['recall_pref'] = recall_pref = n_tp_pref/(n_tp_pref+n_fn_pref+1e-3)
    ret['npv_pref'] = npv_pref = n_tn_pref/(n_tn_pref+n_fn_pref+1e-3)
    ret['specificity_pref'] = specificity_pref = n_tn_pref/(n_tn_pref+n_fp_pref+1e-3)
    ret['accuracy'] = accuracy # (n_tp_hit + n_tn_hit)/(n_pref+0.0+1e-3)

    # B. aligned
    aligned = (Xpf == Mc)  # predicted preference vs probability filter (true "labels")
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

def eval_estimated_probability_filter(P, T, L_test, p_threshold, eps=1e-3):
    """

    Parameters 
    ----------
    `P`: Can be: 
        1. A polarity matrix (Po), where -1 corresponds to {FP, FN} and 1 corresponds to {TP, TN}
            Neutrals are encoded by 0
        2. A color matrix (Pc)
        3. A probabilty filter (aka preference matrix M), where 1s correspond to desirable entries and 
           0s correspond to undesirable entries. 
         
         Basically any matrices with the semantics of positive numbers representing reliable entries and 
         negative numbers representing unreliable entries. 
        
    """
    ret = {} # Output

    # Predicted filter
    Pf_predicted = to_preference(P) # `Pf` as in preference

    cells_positive = cells_predict_reliable = (Pf_predicted == 1)
    cells_negative = cells_predict_unreliable = (Pf_predicted == 0) # Note: negative could also include the neutral particles (e.g. entries with high uncertainty)
    # cells_not_preferred = (Pf_predicted == 0)
    
    n_reliable = np.sum(cells_predict_reliable)
    n_unreliable = np.sum(cells_predict_unreliable)

    # True filter (i.e. true probability filter, whose 1s correspond to {TP, TN} and 0s correspond to {FP, PN})
    Pf_true, Lh = probability_filter(T, L_test, p_threshold) 
    
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)
    cells_tp = (Pf_true == 1) & predict_pos # Entry's prediction is correct and I also predict positive => TP
    cells_tn = (Pf_true == 1) & predict_neg # Correct and negative => TN
    cells_fp = (Pf_true == 0) & predict_pos # Incorrect and positive => FP
    cells_fn = (Pf_true == 0) & predict_neg # Incorrect and negative => FN

    n_tp_hit = n_tp_aligned = np.sum( cells_predict_reliable & cells_tp ) # aligned TPs (i.e. prediction and truth are aligned)
    n_tn_hit = n_tn_aligned = np.sum( cells_predict_reliable & cells_tn ) # aligned TNs   
    
    recall_tp = n_tp_hit/(np.sum(cells_tp)+eps) # The fraction of correctly identified TPs (as being reliable) among all TPs
    recall_tn = n_tn_hit/(np.sum(cells_tn)+eps) # The fraction of correctly identified TNs (as being reliable) among all TNs 
    recall = (n_tp_hit+n_tn_hit)/(np.sum(cells_tp)+np.sum(cells_tn)+eps)    

    precision_tp = p_tp_reliable = n_tp_hit/(n_reliable+eps)  # P(tp|reliable): the fraction of TPs among those predicted reliable
    precision_tn = p_tn_reliable = n_tn_hit/(n_reliable+eps)  # P(tn|reliable): the fraction of TNs among those predicted reliable
    
    # Among all preferred entries (those predicted to be reliable), what's the fraction of them that are 
    # actually reliable (i.e. TPs and TNs)? 
    precision = (n_tp_hit + n_tn_hit)/(n_reliable+eps) # P(correct | reliable): the fraction of correct entries among those predicted reliable

    n_missed_fp = n_misaligned_fp = np.sum( cells_predict_reliable & cells_fp ) # predict reliable but it's FP => reliablility mis-aligned
    n_missed_fn = n_misaligned_fn = np.sum( cells_predict_reliable & cells_fn ) # predict reliable but it's FN
    
    p_missed_fp = n_misaligned_fp/(n_reliable+eps) # Probability of predicting reliable but hitting FPs
    p_missed_fn = n_misaligned_fn/(n_reliable+eps) # Probability of predicting reliable but hitting FNs
    p_missed = (p_missed_fp+p_missed_fn)/(n_reliable+eps) # Probability of predicting reliable but hitting either FPs or FNs

    ret['n_tp_hit'] = n_tp_hit
    ret['n_tn_hit'] = n_tn_hit
    ret['precision_tp'] = precision_tp
    ret['precision_tn'] = precision_tn
    ret['precision'] = precision
    ret['recall_tp'] = recall_tp 
    ret['recall_tn'] = recall_tn
    ret['recall'] = recall
    ret['p_missed']  = p_missed    

    # Overlap: Fraction of entries predicted reliable and are actually correct
    ret['p_overlap'] = p_overlap = np.sum(Pf_predicted == Pf_true)/(Pf_true.size+0.0)

    return ret

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


def demo_data_pipeline(): 
    n_users = 5
    n_items = 10

    # mock rating matrix
    R = np.random.rand(n_users, n_items)
    Po = np.random.choice([0, 1], (n_users, n_items))
    L = np.random.choice([0, 1], n_items)

    X, Y = make_seq2seq_training_data(R, Po=Po, L=L, include_label=False)

    print(f"[demo] shape(X): {X.shape}, shape(Y): {Y.shape}")

    return
 
def test(): 

    demo_data_pipeline()

    return

if __name__ == "__main__": 
    test()