from os.path import abspath, exists, isdir
from os import makedirs
import os

from sys import argv
from glob import glob
import random
import gzip
import pickle
from numpy import array
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score
from utilities import load_properties, f_score, fmax_score
from pandas import concat, read_csv, DataFrame, Series
from collections import OrderedDict

#################################################################
#
#  Memo
#  ----
# 
#  1. debug
#     https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/ranking.py
# 


print "\nStarting. . ."

# ensure project directory exists
project_path = abspath(argv[1])   # e.g. /Users/chiup04/Documents/work/data/diabetes_cf
assert exists(project_path)

if not exists("%s/ENSEMBLES/" % project_path):
    makedirs("%s/ENSEMBLES/" % project_path)

# load and parse project properties
p          	= load_properties(project_path)
fold_count 	= int(p['foldCount'])
seeds 	   	= int(p['seeds'])
bags 	   	= int(p['bags'])
metric		= p['metric']
assert (metric in ['fmax', 'auROC'])

def order(save_=False, verify_=False): 
    """
    Order base predictors by their performance scores (e.g. fmax)    

    Params
    ------
    save_: 
    verify_: if True, verify CV partitions => use verify()

    """
    def get_clf_basename(dirname, delimit='.'):
        clf_name = os.path.basename(dirname)
        clf_name = clf_name.split(delimit)[-1].lower() # e.g. weka.classifiers.trees.RandomForest
        return clf_name

    # sort by classifier directories
    dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

    print "Decreasing order of performance of the base predictors for the seeds (seed in [0-%i]) on the validation set:" % seeds
    seed_list = range(seeds)
    bag_list  = range(bags)

    for seed in seed_list:  # foreach experiment 
        dir_dict = {}
    
        for dirname in dirnames:  # foreach bp 
            for bag in bag_list:
                    
                labels = Series() # DataFrame(columns = ["label"])
                predictions = Series() # DataFrame(columns = ["prediction"])
                for fold in range(fold_count):  # foreach CV fold
                    filename = '%s/valid-b%i-f%s-s%i.csv.gz' % (dirname, bag, fold, seed)
                    df = read_csv(filename, skiprows = 1, compression = 'gzip')  # columns: ['id' 'label' 'prediction']

                    # y_true = df.iloc[:,1:2]  # df[ ['label'] ].values, a dataframe with single column = 'label'
                    # y_score = df.iloc[:,2:3] # df[ ['prediction'] ].values, a dataframe with single column = 'prediction'
                    
                    labels = labels.append(df['label'], ignore_index=True)  # append is not inplace! 
                    predictions = predictions.append(df['prediction'], ignore_index=True)

                # condition: now we have all the (y_true, y_score) pairs for all folds
                if metric == "fmax":
                    assert len(labels) == len(predictions) and len(labels) > 0
                    dir_dict["%s_bag%i" % (dirname, bag)] = fmax_score(labels.values, predictions.values)      
                if metric == "auROC":
                    dir_dict ["%s_bag%i" % (dirname, bag)] = roc_auc_score(labels.values, predictions.values)
            
        # sort according to x[1] in descending order, and then x[0] in ascending order
        d_sorted_by_value = OrderedDict(sorted(dir_dict.items(), key=lambda x: (-x[1], x[0])))

        order_fn = '%s/ENSEMBLES/order_of_seed%i_%s.txt' % (project_path, seed, metric) 
        if save_: 
            with open(order_fn, 'wb') as order_file:
                for key, v in d_sorted_by_value.items():
                    order_file.write("%s, %s \n" % (key, v))   # classifier, performance score
            order_file.close()
            print order_fn
        else: 
            scores = []
            for key, v in d_sorted_by_value.items():
                clf_name = get_clf_basename(key, delimit='.')
                print("%s, %s \n" % (clf_name, v))   # classifier, performance score

                scores.append(v)
            print('> min: %f, max: %f' % (scores[-1], scores[0]))
    
    print('... Complete!')
    return

def verify(): 
    def get_clf_basename(dirname, delimit='.'):
        clf_name = os.path.basename(dirname)
        clf_name = clf_name.split(delimit)[-1].lower() # e.g. weka.classifiers.trees.RandomForest
        return clf_name

    # sort by classifier directories
    dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

    print "Decreasing order of performance of the base predictors for the seeds (seed in [0-%i]) on the validation set:" % seeds
    seed_list = range(seeds)
    bag_list  = range(bags)

    for seed in seed_list:  # foreach experiment 
        dir_dict = {}
    
        valid_files, test_files = [], []
        bpx = []

        Nv, Nt = [], []  # size of validation set (per bp)
        for dirname in dirnames:  # foreach bp 
            for bag in bag_list:
                    
                # idx = Series()
                labels = Series() # DataFrame(columns = ["label"])
                predictions = Series() # DataFrame(columns = ["prediction"])

                bp_name = get_clf_basename("%s_bag%i" % (dirname, bag))
                bpx.append(bp_name)

                # validation split 
                nValidPerFold = nTestPerFold = 0
                vidx, tidx = Series(), Series()

                # [test] Each bp has exactly the same number of validation and test splits
                # sample sizes from CV folds sum to the total (N)
                for i, fold in enumerate(range(fold_count)):  # foreach CV fold

                    # validation
                    filename = '%s/valid-b%i-f%s-s%i.csv.gz' % (dirname, bag, fold, seed)
                    valid_files.append(os.path.basename(filename))
                    dfv = read_csv(filename, skiprows = 1, compression = 'gzip')  # columns: ['id' 'label' 'prediction']
                    nValidPerFold += dfv.shape[0]

                    vidx = dfv['id'].values # vidx.append(df['id'], ignore_index=True)  # append is not inplace! 

                    # test
                    filename = '%s/test-b%i-f%s-s%i.csv.gz' % (dirname, bag, fold, seed)
                    test_files.append(os.path.basename(filename))
                    dft = read_csv(filename, skiprows = 1, compression = 'gzip')  # columns: ['id' 'label' 'prediction']
                    nTestPerFold += dft.shape[0]

                    tidx = dft['id'].values # tidx.append(df['id'], ignore_index=True)  # append is not inplace!  
                    # labels = labels.append(df['label'], ignore_index=True)  # append is not inplace! 
                    # predictions = predictions.append(df['prediction'], ignore_index=True)
                    
                    n_intersect = len(set(vidx).intersection(tidx))
                    # if n_intersect > 0: 
                    #     r = len(vidx)/(n_intersect+0.0)
                    #     print("... n(vidx):%d, n(tidx):%d, n(intersection):%d, r(intersect): %f" % (len(vidx), len(tidx), n_intersect, r))
                    assert len(set(vidx).intersection(tidx)) == 0, "Overlapped instances in valid and test splits"
                 
                nv = len(vidx)  # all CV partitions combined
                nt = len(tidx)
                Nv.append(nv); Nt.append(nt)

                # test statistis for BP 
                if len(bpx) % 10 == 0: 
                    print('(verify) BP=%s, nValidPerFold: %d nTestPerFold: %d' % (bp_name, nValidPerFold, nTestPerFold))

                ### alternative loop for test split
                # n_test_samples = 0
                # for i, fold in enumerate(range(fold_count)):  # foreach CV fold
                #     filename = '%s/test-b%i-f%s-s%i.csv.gz' % (dirname, bag, fold, seed)
                #     test_files.append(os.path.basename(filename))

                #     df = read_csv(filename, skiprows = 1, compression = 'gzip')  # columns: ['id' 'label' 'prediction']
                #     n_test_samples += df.shape[0]
                #     # labels = labels.append(df['label'], ignore_index=True)  # append is not inplace! 
                #     # predictions = predictions.append(df['prediction'], ignore_index=True)

        ### end each bp
        assert len(set(Nv)) == 1, "Inconsistent validation set sizes across CV folds:\n%s\n" % Nv

        # [log] diabetes data: size(valid):154, size(test):153 in each CV fold
        print('(verify) size(valid):%d, size(test):%d in each CV fold.' % (Nv[0], Nt[0]))  
        
        # assert len(valid_files) == len(test_files) 
        # e.g. n_classifier = 5, bags = 10 => n_bp = 50, fold =5 => 50 * 5 = 250 validation and test splits
        print('(verify) N(validation splits): %d =?= N(test splits): %d | summing over all bps ...' % (len(valid_files), len(test_files)))

    return

### Author's code ... 

# # sort by classifier directories
# dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

# print "Decreasing order of performance of the base predictors for the seeds (seed in [0-%i]) on the validation set:" % seeds
# seed_list = range(seeds)
# bag_list  = range(bags)

# for seed in seed_list:  # foreach experiment 
#     dir_dict = {}
#     order_fn = '%s/ENSEMBLES/order_of_seed%i_%s.txt' % (project_path, seed, metric) 
#     with open(order_fn, 'wb') as order_file:
#         for dirname in dirnames:  # foreach classifier (and its data)
#             for bag in bag_list:
#                 # x1 = DataFrame(columns = ["label"])
#                 # x2 = DataFrame(columns = ["prediction"])

#                 # or use np.hstack()
#                 x1, x2 = [], []

#                 for fold in range(fold_count):
#                     filename = '%s/valid-b%i-f%s-s%i.csv.gz' % (dirname, bag, fold, seed)
#                     df = read_csv(filename, skiprows = 1, compression = 'gzip')  # columns: ['id' 'label' 'prediction']

#                     # 
#                     y_true = df.ix[:,1:2]  # a dataframe with single column = 'label'
#                     y_score = df.ix[:,2:3] # a dataframe with single column = 'prediction'
                    
#                     # print('... fold=%d | y_true (%s)> %s, y_score(%s)> %s' % (fold, type(y_true), str(y_true.shape), type(y_score), str(y_score.shape)))
#                     # print('...... y_true: %s, y_score: %s' % (y_true.columns, y_score.columns))
                    
#                     # [debug] somehow this leads to ValueError: unknown format is not supported
#                     #        in anaconda2/lib/python2.7/site-packages/sklearn/metrics/ranking.py
#                     # x1 = concat([x1, y_true], axis = 0)  # dataframe | concatenated vertically, along the row
#                     # x2 = concat([x2, y_score], axis = 0) 

#                     x1.append(y_true.label.values)
#                     x2.append(y_score.prediction.values)

#                 # print('(after concat) x1: %s' % (x1[:5]))
#                 x1 = np.hstack(x1)
#                 x2 = np.hstack(x2)

#                 if metric == "fmax":
                    
#                     # [test]
#                     # x1, x2 = x1['label'].values, x2['prediction'].values
#                     # print('(test) x1 (%s): %s\n... x2 (%s): %s\n' % (type(x1), x1[-10:], type(x2), x2[-10:]))
#                     # print('(test) uniq(x1): %s\n' % np.unique(x1))
#                     # assert len(x1) == len(x2)
                    
#                     # this int() conversion makes it work! 
#                     # print('(test) fmax: %f' % fmax_score([int(e) for e in x1[-10:]] ,x2[-10:]))

#                     dir_dict["%s_bag%i" % (dirname, bag)] = fmax_score(x1 ,x2)    	
#                 if metric == "auROC":
#                     dir_dict ["%s_bag%i" % (dirname, bag)] = roc_auc_score(x1,x2)
	        
#         # sort according to x[1] in descending order, and then x[0] in ascending order
#         d_sorted_by_value = OrderedDict(sorted(dir_dict.items(), key=lambda x: (-x[1], x[0])))

#         # print('... d_sorted_by_value ') 
#         for key, v in d_sorted_by_value.items():
#             # print('... key=%s' % key) # e.g. .../weka.classifiers.functions.SimpleLogistic_bag0
#             order_file.write("%s, %s \n" % (key, v))   # classifier, performance score
	    
#         order_file.close()
#         print order_fn

# # [output] generates ENSEMBLE directory, 
# print "Done!\n"

def run(): 
    
    # order bp performance without savinng 
    order(save_=True)

    # verify datasink validation and test splits
    # verify()

    return

if __name__ == "__main__": 
    run()






