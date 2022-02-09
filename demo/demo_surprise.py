
# from __future__ imports must occur at the beginning of the file
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) 
import utils_sys # only works if utils_sys is a package

import pandas as pd
import numpy as np
import random

# http://surprise.readthedocs.io/en/stable/getting_started.html
# I believe in loading all the datasets from pandas df 
# you can also load dataset from csv and whatever suits

#################################################################
#
#
#
#   Ref
#   ---
#   1. https://github.com/NicolasHug/Surprise
#

### reproducible experiments
my_seed = 53
random.seed(my_seed)
np.random.seed(my_seed)


def load_data(): 
    from surprise import Reader, Dataset

    # e.g. /Users/chiup04/Documents/work/data/recommender/ml-latest-small
    # src_path = '/Users/chiup04/Documents/work/data/recommender/ml-latest-small/ratings.csv'
    src_path = utils_sys.getProjectPath(domain='recommender/ml-latest-small', dataset='ratings.csv')

    ratings = pd.read_csv(src_path) # reading data in pandas df
    print('(load_data) dim(ratings): %s' % str(ratings.shape))

    # to load dataset from pandas df, we need `load_fromm_df` method in surprise lib

    ratings_dict = {'itemID': list(ratings.movieId),
                    'userID': list(ratings.userId),
                    'rating': list(ratings.rating)}
    df = pd.DataFrame(ratings_dict)

    # A reader is still needed but only the rating_scale param is required.
    # The Reader class is used to parse a file containing ratings.
    reader = Reader(rating_scale=(0.5, 5.0))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    return data

def t_predict(): 
    """
    This module descibes how to train on a full dataset (when no testset is
    built/specified) and how to use the predict() method.

    Memo
    ----
    1. Referemce demo/demo_surprise_predict.py
    """
    from surprise import KNNBasic, NMF
    from surprise import Dataset
    from surprise.model_selection import train_test_split
    # from surprise.model_selection import GridSearchCV
    # from surprise.model_selection import cross_validate

    # Load the movielens-100k dataset
    data = Dataset.load_builtin('ml-100k')  # data is an instance of DatasetAutoFolds, which has no attribute 'shape'
    
    R = raw_ratings = data.raw_ratings  
    trainset, testset = train_test_split(data, test_size=.15)
    print('... type(R): %s, type(trainset): %s' % (type(R), type(trainset)))  # R is a list, trainset is an instance

    # Retrieve the trainset.
    trainset = data.build_full_trainset()
    # print('... dim(trainset): %s' % str(trainset.shape))

    # Build an algorithm, and train it.
    algo = NMF(n_factors=15, n_epochs=300) # KNNBasic()
    algo.fit(trainset)

    print('... dim(pu): %s, dim(qi): %s' % (str(algo.pu.shape), str(algo.qi.shape)))

    # we can now query for specific predicions
    uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
    iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

    # get a prediction for specific users and items.
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)

    return

def load_from_predefined_folds(): 
    """

    Memo
    ----
    1. the problem is that this requires loading from the predefined files 
       but what if we only dataframes, each of which contains data from each fold? 
    """

    from surprise import SVD, NMF
    from surprise import Dataset
    from surprise import Reader
    from surprise import accuracy
    from surprise.model_selection import PredefinedKFold

    # path to dataset folder
    files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')

    # This time, we'll use the built-in reader.
    reader = Reader('ml-100k')

    # folds_files is a list of tuples containing file paths:
    # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
    train_file = files_dir + 'u%d.base'
    test_file = files_dir + 'u%d.test'
    folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()
    algo = SVD()

    for trainset, testset in pkf.split(data):

        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        accuracy.rmse(predictions, verbose=True)
    return

def t_unbiased_estimate(): 
    from surprise import SVD
    from surprise import Dataset
    from surprise import accuracy
    from surprise.model_selection import GridSearchCV


    # Load the full dataset.
    data = Dataset.load_builtin('ml-100k')
    raw_ratings = data.raw_ratings

    # shuffle ratings if you want
    random.shuffle(raw_ratings)

    # A = 90% of the data, B = 10% of the data
    threshold = int(.9 * len(raw_ratings))
    A_raw_ratings = raw_ratings[:threshold]
    B_raw_ratings = raw_ratings[threshold:]

    data.raw_ratings = A_raw_ratings  # data is now the set A

    # Select your best algo with grid search.
    print('Grid Search...')
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)

    algo = grid_search.best_estimator['rmse']

    # retrain on the whole set A
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Compute biased accuracy on A
    predictions = algo.test(trainset.build_testset())
    print('Biased accuracy on A,', end='   ')
    accuracy.rmse(predictions)

    # Compute unbiased accuracy on B
    testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
    predictions = algo.test(testset)
    print('Unbiased accuracy on B,', end=' ')
    accuracy.rmse(predictions)

    return

def t_basic(): 
    from surprise import SVD, evaluate
    from surprise import NMF

    data = load_data()

    # Split data into 5 folds

    data.split(n_folds=5)

    # svd
    algo = SVD()
    evaluate(algo, data, measures=['RMSE'])

    # nmf
    algo = NMF()
    evaluate(algo, data, measures=['RMSE'])

    return

def test(): 

    # basic demo 
    # t_basic()

    # prediction 
    t_predict()

    return


if __name__ == "__main__": 
    test()


