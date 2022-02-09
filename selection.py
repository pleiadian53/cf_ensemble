#!/usr/bin/env python

"""
    datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
    Copyright (C) 2013 Sean Whalen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
"""

from os import mkdir
from os.path import abspath, exists
from sys import argv
import sys, os, getpass

from numpy import array, column_stack
from numpy.random import choice, seed
from pandas import DataFrame, concat
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
import common

def get_cluster_performance(labels, predictions, n_clusters, fold, seedval):
    return {'fold': fold, 'seed': seedval, 'score': common.score(labels, predictions), 'n_clusters': n_clusters}


def get_performance(df, ensemble, fold, seedval):
    labels          = df.index.get_level_values('label').values
    predictions     = df[ensemble].mean(axis = 1)  # mean score across all classifier.bag (each of which is a column)
    # if verbose: print('(get_performance) %s' % {'fold': fold, 'seed': seedval, 'score': common.score(labels, predictions), 'ensemble': ensemble[-1], 'ensemble_size': len(ensemble)})
    
    # performance of an ensemble is defined as ... 
    # ... the score of the mean predictions (of all classifiers) (i.e. score(labels, mean_predictions))
    return {'fold': fold, 'seed': seedval, 'score': common.score(labels, predictions), 'ensemble': ensemble[-1], 'ensemble_size': len(ensemble)}


def get_predictions(df, ensemble, fold, seedval):
    ids             = df.index.get_level_values('id')
    labels          = df.index.get_level_values('label')
    predictions     = df[ensemble].mean(axis = 1)   # select all classifiers/columns and take their means for each data point/row
    diversity       = common.diversity_score(df[ensemble].values)

    # predictions of an ensemble is their mean predictions
    return DataFrame({'fold': fold, 'seed': seedval, 'id': ids, 'label': labels, 'prediction': predictions, 'diversity': diversity, 'ensemble_size': len(ensemble)})


def select_candidate_greedy(train_df, train_labels, best_classifiers, ensemble, i):

    # [params]
    # train_df: (id+) label vs classifer.bag
    # best_classifiers: a Series: classifier.bag -> score (e.g. auc)
    return best_classifiers.index.values[i]  # alwasy pick the one with the best score regardless of the ensemble set


def select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:  # randomly select the classifier such that it has the best combined performance in terms of the mean score

        # sample without replacement from a list of classifiers
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)

        # merge existing ensemble with selected candidate
        candidate_scores = [common.score(train_labels, train_df[ensemble + [candidate]].mean(axis = 1)) for candidate in candidates]
        best_candidate = candidates[common.argbest(candidate_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def select_candidate_drep(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)

        candidate_diversity_scores = [abs(common.diversity_score(train_df[ensemble + [candidate]].values)) for candidate in candidates]
        candidate_diversity_ranks = array(candidate_diversity_scores).argsort()  # ~> indices of the diversity score
        diversity_candidates = candidates[candidate_diversity_ranks[:max_diversity_candidates]] # candidates with higher diversity scores
        candidate_accuracy_scores = [common.score(train_labels, train_df[ensemble + [candidate]].mean(axis = 1)) for candidate in diversity_candidates]

        # among the set of classifiers with higher divesity scores, choose the one that leads to the best combined score wrt their mean
        best_candidate = candidates[common.argbest(candidate_accuracy_scores)]  
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def select_candidate_sdi(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_diversity_scores = [1 - abs(common.diversity_score(train_df[ensemble + [candidate]].values)) for candidate in candidates] # 1 - kappa so larger = more diverse
        candidate_scores = [accuracy_weight * best_classifiers.ix[candidate] + (1 - accuracy_weight) * candidate_diversity_scores[candidate_i] for candidate_i, candidate in enumerate(candidates)]
        best_candidate = candidates[common.argbest(candidate_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def stack_intra(n_clusters, distances, fit_df, fit_labels, predict_df):
    cluster_labels = MiniBatchKMeans(n_clusters).fit_predict(distances)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        model = stacker.fit(fit_df.ix[:, mask], fit_labels)
        predictions = model.predict_proba(predict_df.ix[:, mask])[:, 1]
        cols.append(predictions)
    values = column_stack(cols)
    predictions = values.mean(axis = 1)
    return values, predictions


def stack_inter(n_clusters, distances, fit_df, fit_labels, predict_df):
    """

    Memo
    ----
    1.  >>> np.vstack(([1,2,3],[4,5,6]))
        array([[1, 2, 3],
              [4, 5, 6]])
        >>> np.column_stack(([1,2,3],[4,5,6]))
        array([[1, 4],
              [2, 5],
              [3, 6]])
        >>> np.hstack(([1,2,3],[4,5,6]))
        array([1, 2, 3, 4, 5, 6])

        ~ 
        >>> np.hstack(([[1],[2],[3]],[[4],[5],[6]]))
        array([[1, 4],
              [2, 5],
               [3, 6]])
    """
    cluster_labels = MiniBatchKMeans(n_clusters).fit_predict(distances)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        predictions = fit_df.ix[:, mask].mean(axis = 1)
        cols.append(predictions)
    model = stacker.fit(column_stack(cols), fit_labels)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        predictions = predict_df.ix[:, mask].mean(axis = 1)
        cols.append(predictions)
    values = column_stack(cols)
    predictions = model.predict_proba(values)[:, 1]
    return values, predictions


def stacked_selection(fold):
    """

    Memo
    ----
    stack_function: 
        a. stack_intra 
        b. stack_inter
    """
    seed(seedval)
    train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold)
    train_distances = 1 - train_df.corr().abs()
    train_performance = []
    test_performance = []
    for n_clusters in range(1, max_clusters + 1):
        train_values, train_predictions = stack_function(n_clusters, train_distances, train_df, train_labels, predict_df = train_df)
        test_values, test_predictions = stack_function(n_clusters, train_distances, train_df, train_labels, predict_df = test_df)
        train_performance.append(get_cluster_performance(train_labels, train_predictions, n_clusters, fold, seedval))
        test_performance.append(get_cluster_performance(test_labels, test_predictions, n_clusters, fold, seedval))
    best_cluster_size = common.get_best_performer(DataFrame.from_records(train_performance)).n_clusters.values
    test_values, test_predictions = stack_function(best_cluster_size, train_distances, train_df, train_labels, predict_df = test_df)
    return DataFrame({'fold': fold, 'seed': seedval, 'id': test_df.index.get_level_values('id'), 'label': test_labels, 'prediction': test_predictions, 'diversity': common.diversity_score(test_values), 'metric': common.score.__name__}), DataFrame.from_records(test_performance)

def selection(fold):
    """

    Memo
    ----
    1. train_df, test_df 

            label vs classifier.bag
    """
    def view_df(df, n=3):
        print('(selection) dim: %s' % str(df.shape))  # e.g. 3 classifier, 10-fold => 30 columns, NaiveBayes.0 ... NaiveBayes.9, SGD.0 ... SGD.9, ...
        print('... df.columns:\n%s\n' % df.columns)
        print('... df.index:\n%s\n' % df.index[:10])

        print('... show first %d records ...' % n)
        print('######\n%s\n######' % df.head(n))
        return  

    seed(seedval)
    train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold) # train_labels is a numpy array
    # view_df(train_df, n=3) 

    # [note] 'Series' object has no attribute 'order'
    # best_classifiers = train_df.apply(lambda x: common.score(train_labels, x)).order(ascending = not common.greater_is_better)
    # ... score: sklearn's roc_auc_score <- y_true, y_score
    #         x: takes in entire column (axis=0 by default, row values as input)
    #         => train_df.apply(...) => a series where classifiers are the indices and AUC are the values 
    # ... sort classifier.bag according to AUC scores
    best_classifiers = train_df.apply(lambda x: common.score(train_labels, x)).sort_values(ascending = not common.greater_is_better)
    # => sorted classifier.bag from best to worst

    # ensemble selection loop
    train_performance = []
    test_performance = []
    ensemble = []
    for i in range(min(max_ensemble_size, len(best_classifiers))): # select from among (the subset) of best classifiers
        best_candidate = select_candidate(train_df, train_labels, best_classifiers, ensemble, i) # <<< this is the main function
        
        ensemble.append(best_candidate)
        if i < 2: 
            # e.g. added ['SGD.6', 'SGD.3', ...]
            print ('... added %s\n... perf: %s ##' % (ensemble, get_performance(train_df, ensemble, fold, seedval)))

        train_performance.append(get_performance(train_df, ensemble, fold, seedval))  # fold is fixed
        test_performance.append(get_performance(test_df, ensemble, fold, seedval))

    train_performance_df = DataFrame.from_records(train_performance)
    # columns: {'fold', 'seed', 'score', 'ensemble', 'ensemble_size'} | ensemble: always the latest one i.e. best candidate in the ensemble list

    # [note] common.get_best_performer(train_performance_df) ~> a dataframe with one row (best by ensemble size) ... one_se = False), not considering SE
    # best_ensemble_size = common.get_best_performer(train_performance_df).ensemble_size.values 
    best_ensemble_size = common.get_best_performer(train_performance_df).ensemble_size.values[0]  # 1-by-1 array array([19])
    
    # [test]
    # train_performance_df.ensemble is a Series (e.g. SGD.1 SGD.2 .. )
    # print('(selection) train_performance_df.ensemble (type:%s):\n%s\n' % (type(train_performance_df.ensemble), train_performance_df.ensemble.head(10)))
    # print('... best_ensemble_size (type: %s): %s' % (type(best_ensemble_size), best_ensemble_size))
    # print('... test_df:\n%s\n' % test_df.head(2)) # use: test_df[best_ensemble].mean(axis = 1)
    # print('... train_performance_df.ensemble [%s]:\n%s\n##' % (type(train_performance_df.ensemble), train_performance_df.ensemble))
    
    # train_performance_df.ensemble ~> a Series of classifiers e.g. [SGD.1, SGD.2, LogitBoost.1, NaiveBayes.9, ...]
    best_ensemble = train_performance_df.ensemble[:best_ensemble_size + 1]  # [note] so the classifiers added earlier has higher channce of being selected? 
    
    return get_predictions(test_df, best_ensemble, fold, seedval), DataFrame.from_records(test_performance)


### configuration
user = getpass.getuser() # 'pleiades' 
prefix = '/Users/%s/Documents/work/' % user  # /Users/<username>/Documents/work/data/recommender
domain = dataset_name = 'diabetes_cf'
project_path = abspath(argv[1]) if len(argv) > 1 else os.path.join(prefix, 'data/%s' % dataset_name) # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
assert exists(project_path)

# [datasink]
# project_path = path = abspath(argv[1])
# assert exists(path)

if not exists('%s/analysis' % project_path):
    mkdir('%s/analysis' % project_path)

default_method = 'greedy'
try: 
    method = argv[2]
except: 
    msg = "Missing method parameter: 'greedy', 'enhanced', 'drep', 'sdi', 'inter', 'intra' ... etc."
    print('(main) Use method=%s by default ...' % default_method)
    method = default_method

    # raise ValueError, msg
assert method in ['greedy', 'enhanced', 'drep', 'sdi', 'inter', 'intra', ]  # [todo] 'cf' once cf.py is completed 

# two categories
if method in ['inter', 'intra']:
    stack_function = eval('stack_' + method)
    method_function = stacked_selection
else:
    select_candidate = eval('select_candidate_' + method)  # this defines the function ptr: e.g. select_candidate_greedy
    print('(selection) Using method: %s' % select_candidate.__name__)
    method_function = selection

p = common.load_properties(project_path)
fold_count = int(p['foldCount'])
initial_ensemble_size = 2
max_ensemble_size = 100 # 50
max_candidates = 50
max_diversity_candidates = 5
accuracy_weight = 0.5
max_clusters = 20

def t_distance(**kargs): 

    fold = 1
    train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold)

    # df.corr() 
    #   > Compute pairwise correlation of columns, excluding NA/null values
    train_distances = 1 - train_df.corr().abs()   # classifier_i vs classifier_j 
    print('... dim(train_df): %s > cols: \n%s\n' % (str(train_df.shape), ' '.join(train_df.columns.values)))
    print('... pairwise correlation of columns:\n%s\n' % train_df.corr().abs())

    return 

def t_select_best(**kargs):

    fold = 1
    train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold)

    # select the best classifiers based on a. train b. test 
    # for split in ['train', 'test']: 
    best_classifiers = train_df.apply(lambda x: common.score(train_labels, x)).sort_values(ascending = not common.greater_is_better) 
    print('(test) best_classifiers in train:\n%s\n' % best_classifiers)  # a Series

    best_classifiers = test_df.apply(lambda x: common.score(test_labels, x)).sort_values(ascending = not common.greater_is_better) 
    print('(test) best_classifiers in test:\n%s\n' % best_classifiers)

    return

def test(): 

    ### correlation, distance, diversity 
    # t_distance()

    ### best base classifeirs
    t_select_best()

    return

def run(): 
    # use shallow non-linear stacker by default
    stacker = RandomForestClassifier(n_estimators = 200, max_depth = 2, bootstrap = False, random_state = 0)
    if len(argv) > 3 and argv[3] == 'linear':
        stacker = SGDClassifier(loss = 'log', n_iter = 50, random_state = 0)

    predictions_dfs = []
    performance_dfs = []
    seeds = [0] if method == 'greedy' else range(10)
    for seedval in seeds:
        results = Parallel(n_jobs = -1, verbose = 1)(delayed(method_function)(fold) for fold in range(fold_count))
        for predictions_df, performance_df in results:
            predictions_dfs.append(predictions_df)
            performance_dfs.append(performance_df)
    performance_df = concat(performance_dfs)
    performance_df.to_csv('%s/analysis/selection-%s-%s-iterations.csv' % (project_path, method, common.score.__name__), index = False)
    predictions_df = concat(predictions_dfs)
    predictions_df['method'] = method
    predictions_df['metric'] = common.score.__name__
    predictions_df.to_csv('%s/analysis/selection-%s-%s.csv' % (project_path, method, common.score.__name__), index = False)
    print '%.3f %i' % (predictions_df.groupby(['fold', 'seed']).apply(lambda x: common.score(x.label, x.prediction)).mean(), predictions_df.ensemble_size.mean())


if __name__ == "__main__": 
    test()
    # run()

