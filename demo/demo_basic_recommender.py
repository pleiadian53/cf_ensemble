
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
import utils_sys
from utils_sys import div
from evaluate import visualizeCoeffs
import cluster


def pairwise_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

# [data_pipeline_datasink]
def toUserItem(fold, **kargs): 
    import data_pipeline_datasink as dsp
    return dsp.toUserItem(fold, **kargs)

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

def demo_recommender(**kargs): 
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
        algo = NMF(n_factors=n_factors, n_epochs=n_epochs) # KNNBasic()
        algo.fit(ts_total)
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

def test(): 

    return 


if __name__ == "__main__": 
    test()