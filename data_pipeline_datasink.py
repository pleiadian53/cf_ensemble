
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import common
import evaluate
import os

import utils_sys

# Plotting 
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('ggplot')  # values: {'seaborn', 'ggplot', }

###################################################################################
# 
# This module contains utility funcitons for accessing (pre-computed) training data 
# and perform data transformations. The pretrained data sets used in this module 
# are assumed to have obtained via the datasink pipeline. 
# 
#   
# Notes
# -----
# 1. Some of these functions were already implemented throughout parts of 
#    CF ensemble modules; however, they tended to work specifically with datasink 
#    but not for general use. This module structures these functions in a way that 
#    works in more general settings. 
#    
# 
###################################################################################


def toUserItem(fold=None, split='train', **kargs): 
    """
    Convert level-1 training data to user-item dataframe format consisting of the following attributes (columns): 

    ['user_id', 'item_id', 'prediction', 'label'], 

    where user_id corresponds to classifiers 
          item_id corresponds to data points 

    """
    save_= kargs.get('save', False)
    input_path = kargs.get('input_path', os.path.join(os.getcwd(), 'data')) 
    output_path = kargs.get('output_path', input_path) 

    # Interpret the input
    ###############################################
    # Extract `users`, `dataframe`, `ID`, `labels`

    # 1. If `fold` is given, then assume that we have (level-1) training data pre-computed and stored according to 
    #    datasink's format (i.e. dataframes)
    if fold is not None: 
        train_df, train_labels, test_df, test_labels = common.read_fold(input_path, fold) # [todo] single out this part
        print('(toUserItem) dim(train_df):%s' % str(train_df.shape))
        
        # get all data IDs 
        users = train_df.columns.values
        ts = train_df if split.startswith('tr') else test_df

        ts = ts.reset_index() # convert multilevel index to flat index
        idx = ts['id'].values
        assert len(idx) == len(set(idx)), "Data IDs are not unique!"
        labels = ts['label'].values

    ###############################################

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
        output_file = kargs.get("output_file", 'user_item-%s-f%d.csv' % (split, fold))
        fpath = os.path.join(output_path, output_file) 
        print('(toUserItem) Saving level-1 CF %s set (dim=%s) to .csv: %s' % (split, str(ts.shape), fpath))
        ts.to_csv(fpath, sep=',', index=False, header=True)

    return (ts, nUsers, nItems)

def dataframe_to_rating_matrix(ts, n_users, n_items, p_threshold=0.5, users=[], select_all=False, missing_value=-1): # I: indicator, R: ratings
    """
    Convert rating dataframe format (transformed via toUserItem()) to the rating matrix format: users/classifiers vs items/data.  

    Memo
    ----
    1. Call chain: 
            to_rating_matrix_train -> toUserItem -> dataframe_to_rating_matrix -> R 
            to_rating_matrix_test  -> toUserItem -> dataframe_to_label_matrix -> L

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
def dataframe_to_label_matrix(ts, n_users, n_items, p_threshold=0.5, users=[], soft_label=False, missing_value=-1):
    """
    Convert rating dataframe (transformed via toUserItem()) to label matrix. 
    Each entry in the label matrix corresponds to the True label. 


    Memo
    ----
    1. Assuming that all classifiers/users predict items/data correctly and perfectly (positive: 1, negative: 0). 

    2. Mask all FP and FN by considering them as missing. 

    3. Call chain: 
        to_rating_matrix_train -> toUserItem -> dataframe_to_rating_matrix -> R 
        to_label_matrix        -> toUserItem -> dataframe_to_label_matrix -> L

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

def to_label_matrix(fold, p_threshold=0.5, save_=False, missing_value=-1, merge_=False):
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

    L = dataframe_to_label_matrix(ts_train, n_users, n_items, missing_value=missing_value)

    ts_test, n_users, n_items = toUserItem(fold, split='test', save_=save_)
    Lt = dataframe_to_label_matrix(ts_test, n_users, n_items, missing_value=missing_value)

    # merge 
    print('... dim(L): %s, dim(Lt): %s' % (str(L.shape), str(Lt.shape)))
    if merge_: 
        L = np.hstack((L, Lt))
        print('... dim(L_combined): %s' % str(L.shape))

    return (L, Lt)

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
    R = dataframe_to_rating_matrix(ts_train, n_users, n_items, p_threshold=p_threshold, select_all=False, missing_value=missing_value)
    verify(R)

    # test set needs to come in because we need to know all the items at test time ... [todo]
    ts_test, nu_test, ni_test = toUserItem(fold, split='test', save_=save_)
    assert nu_test == n_users, "n_classifiers @ training: %d, in test: %d" % (n_users, nu_test)

    # set select_all to True, because we cannot use label info to mask FP and FN
    print('to_rating_matrix_train> 2. include all test data but without using the ground truth information ...')
    Rt = dataframe_to_rating_matrix(ts_test, nu_test, ni_test, p_threshold=0.5, select_all=True, missing_value=missing_value)
    verify(Rt)

    # merge
    print('... dim(R): %s, dim(Rt): %s' % (str(R.shape), str(Rt.shape)))
    if merge_:  
        R = np.hstack((R, Rt))
        print('... dim(R_combined): %s' % str(R.shape))

    return (R, Rt)  # R: combines train and test split, Rt: test only

def to_rating_matrix(fold, **kargs):
    """


    Memo
    ----
    1. train-dev-test split 
       <ref> https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213

       train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))]

    """
    # from cf_spec import System  # [note] the value configured in cf does not propogate here
    input_path = kargs.get('input_path', os.path.join(os.getcwd(), 'data'))
    
    # tDev = kargs.get('include_devset', False)  # if True, return a train-dev-test split (instead of just train-test split)
    train_df, train_labels, test_df, test_labels = common.read_fold(input_path, fold) # [todo] single out this part
    if kargs.get('unbag', False):
        bag_count = kargs['bag_count'] if 'bag_count' in kargs else None
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        test_df = common.unbag(test_df, bag_count)

    R = train_df.values.T  # R: users vs items
    T = test_df.values.T 
    U = train_df.columns.values

    return (R, T, train_labels, test_labels, U)

# Adapted from utils_cf.to_rating_matrix_dev
def to_rating_matrix_dev(**kargs):
    # consider dev set
    fold = kargs.get('fold', 0)
    dev_ratio = kargs.get('dev_ratio', 1./5)
    input_path = kargs.get('input_path', os.path.join(os.getcwd(), 'data'))
    train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.read_fold2(input_path, fold, dev_ratio=dev_ratio)        
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

def to_rating_matrix_random_subsampling(**kargs):
    """
    An extension to to_rating_matrix() by supporting random subsampling. 

    The premise of using this routine to construct rating matrices is to incorporate model selection. 
    This function will assume, by default, that the return value should include dev set, which means that the return value 
    consider rating matrices structured as (R, Td, Tt) instead of (R, T), where 

    Td: the rating matrix for the dev set (for hyperparameter tunning)
    Tt: the rating matrix for the test set (for model evaluation)

    Use
    ---
        to_rating_matrix_random_subsampling(dev_ratio=1/5., fold_count=5, policy='random_cv_fold')
    
    """
    # from cf_spec import System
    input_path = input_path = kargs.get('input_path', os.path.join(os.getcwd(), 'data'))

    fold = kargs.pop('fold', 0)
    dev_ratio = kargs.get('dev_ratio', 1./5)
    test_ratio = kargs.get('test_ratio', 1./5)

    # kargs['include_devset'] = True
    if fold > 0: return to_rating_matrix_dev(**kargs) # (R, Td, Tt, train_labels, dev_labels, test_labels, U)

    # ... otherwise, proceed with the random subsampling 
    fold_count = kargs.get('fold_count', 5)
    
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
    train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.read_random_fold(input_path, fold_count=fold_count, dev_ratio=dev_ratio, test_ratio=test_ratio, shuffle=shuffle)

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
    return (Xr.T, Lr)

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
def to_rating_matrix0(fold, **kargs): 
    return toPredictiveScores(fold, **kargs)

# evaluate
def evalTestSet(labels, Th, **kargs): # labels: true labels, Th: estimates, T: 'true' rating matrix
    # return metrics  # a dictionary: metric -> score
    return evaluate.evalTestSet(labels, Th, **kargs)

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

def demo_cf_ensemble(**kargs): 
    from cf import sgd

    input_path = kargs.get('input_path', os.path.join(os.getcwd(), 'data')) 

    ### configurations
    # project_path = os.path.abspath('../data/diabetes_cf')
    # try: 
    #     project_path = path = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    # except: 
    #     pass 
    # assert os.path.exists(path)
    domain = kargs.get('domain', 'test')

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

    T, Tt  = to_label_matrix(fold, p_threshold=0.5, merge_=False)

    # assuming training set is missing, so that we do not include rmse for the training part
    # this is ok because we are not trying to fit P and Q for T
    Tm = np.full((T.shape[0], T.shape[1]), missing_value)
    T = np.hstack((Tm, Tt))
    print('... T:\n%s\n' % T[:10])

    assert R.shape[0] == T.shape[0] and R.shape[1] == T.shape[1]

    # CF
    P, Q = sgd(R, T, n_features=20, n_epochs=1000, plot_name='cf_sgd_rmse-%s' % domain)
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

    train_df, train_labels, test_df, test_labels = common.read_fold(input_path, fold) # [todo] single out this part
    evalTestSet0(Pt, Qt, Tt, labels = test_labels)

    return

def to_rating_matrix(df, users=[], p_threshold=0.5, mask_false_predictions=False, fill=-1): # I: indicator, R: ratings
    """
    Convert a rating dataframe (in movielens format) to a rating matrix whose rows correspond to users (or classifiers)
    and columns correspond to items (or data points). 
 
    Notes
    -----
    Input dataframe the following format: 

    userId  movieId rating  timestamp
       1       1      4.0    964982703
       1       3      4.0    964981247
       1       6      4.0    964982224 

    """

    # Get all unique users/classifiers
    if not users: 
        users = np.unique(df['userId'].values) # sorted

    N = df.shape[0]


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


def test(): 

    return

if __name__ == "__main__": 
    test()