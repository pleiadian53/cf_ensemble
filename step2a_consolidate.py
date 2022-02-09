
import os
from os.path import abspath, exists, isdir
from os import makedirs
from sys import argv
from glob import glob
import random
import gzip
import pickle
from numpy import array
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score
from utilities import load_properties, f_score, fmax_score
from pandas import concat, read_csv, DataFrame
from collections import OrderedDict

#################################################################
#
#  Memo
#  ----
# 
#  1. debug
#     https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/ranking.py
#
#  2. after base predictors are generated via step1_generate.py 
#     
#     python step2a_consolidate.py /Users/chiup04/Documents/work/data/diabetes_cf
# 
#  3. Use stacking.py in the case of nested CV
# 

print "\nStarting. . ."

# ensure project directory exists
try: 
    project_path = abspath(argv[1])
except: 
    msg = '(hint) missing project path (e.g. /Users/chiup04/Documents/work/data/diabetes_cf)\n'
    raise ValueError, msg
assert exists(project_path)

l1_data_path = os.path.join(project_path, 'LEVEL1')
if not exists(l1_data_path):
    makedirs(l1_data_path)

# load and parse project properties
p          	= load_properties(project_path)
fold_count 	= int(p['foldCount'])
seeds 	   	= int(p['seeds'])
bags 	   	= int(p['bags'])
metric		= p['metric']
assert (metric in ['fmax', 'auROC'])

# refactor to utilties? 
def define_level1_tset(dirnames=[],  merge_bags=False, split='valid'): 
    """
    Define training sets for model stacking. 

    N-fold CV on validation set. Use (N-1)-fold portion for training and the remaining fold for tuning 
    hyperparamters of the meta learner (e.g. regularization strength)

    
    Params
    ------
    merge_bags: set merge_bags to True to merge training data from all bags
    split: {'test', 'valid'}

    Use 
    ---
    1. provide as sources for visualize_coeffs(ts, features, label, ts_test=None)


    """
    def get_clf_basename(dirname, bag, delimit='.', verbose=False):
        clf_name = os.path.basename(dirname) # remove prefix
        clf_name = clf_name.split(delimit)[-1].lower() # e.g. weka.classifiers.trees.RandomForest
        clf_name  = '%s-b%i' % (clf_name, bag)
        if verbose: print '... BP: %s' % clf_name
        return clf_name
    def get_bps(seed=0, use_full_path=False, reverse=False): 
        bpx = []
        # seed = seed_list[0]  # foreach experiment 
        if use_full_path: 
            # todo
            pass
        else: 
            for dirname in dirnames:  # foreach classifier (and its data)
                for bag in bag_list: 
                    bpx.append(get_clf_basename(dirname, bag=bag)) # [reduce]

        bpx.sort(reverse=reverse) # ascending order if reverse <- False
        return bpx

    # create level1 training data directory if it doesn't exist 
    l1_dir = l1_data_path # e.g. e.g. /Users/chiup04/Documents/work/data/diabetes_cf/LEVEL1

    if not dirnames: dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))
    seed_list = range(seeds)
    bag_list  = range(bags)

    # assume that bags are merged
    # [todo]

    # find all the possible base predictors 
    base_predictors = get_bps()
    nBP = len(base_predictors)
    header = base_predictors + ['label', ]

    # assume that we do not merge different experiments (seeds)
    for seed in seed_list:  # foreach experiment 

        adict = {h:[] for h in header}
        # each fold has its own (train, validation, test)-split
        for fold in range(fold_count):  # foreach fold, look into all classifiers indexed by (classifier, bag), i.e. all base predictors (bps)

            yT_fold = [] # all true labels within the same fold
            N = nGrand = 0  # number of instances

            ### foreach bp
            adict_local = {h:[] for h in header}  # collect level-1 training data separately for each CV fold
            for ic, dirname in enumerate(dirnames):  # foreach classifier (and its data)
                for bag in bag_list:
                
                    clf_name = get_clf_basename(dirname, bag=bag, verbose=True if fold == 0 else False) # [reduce]
                    # if not adict.has_key(clf_name): adict[clf_name] = []
                    assert clf_name in adict, "Unknown bp: %s" % clf_name
               
                    ### aggregate validation and test data 
                    # read bp predictive results
                    filename = '%s/%s-b%i-f%s-s%i.csv.gz' % (dirname, split, bag, fold, seed)   # split: {'valid', 'test', }
                    df = read_csv(filename, skiprows = 1, compression = 'gzip')  # columns: ['id' 'label' 'prediction']
                    y_true = df['label'].values # df.ix[:,1:2]   # [check] note that within the same fold, y_true should be consistent across bps
                    y_score = df['prediction'].values # df.ix[:,2:3] 
                    
                    # collect global data (by merging prob scores from each fold)
                    adict[clf_name].append(y_score)  # predictive scores within this CV fold
                    # collect local data (for training meta-classifiers using only a single CV-fold worth of data)
                    adict_local[clf_name] = y_score
                    
                    if len(yT_fold) == 0: 
                        yT_fold = np.array(y_true) # create a copy
                        adict['label'].append(yT_fold); adict_local['label'] = yT_fold  # this only gets updated per fold (not per bp)
                    else: 
                        N = len(y_score)
                        assert all(yT_fold == y_true), "Not all labels consistent within a CV-partition (fold=%d, bag=%d)!" % (fold, bag)
            ### end foreach bp

            assert sum(1 for k, vx in adict.items() if len(vx[-1]) != N) == 0, "Each BP should have equal number of predictive scores: %d" % N

            ## local/inner-loop stacker code here (or just collect data for this inner stacker)
            print('(local) collected %d scores per bp per fold in a total of %d bps | split=%s' % (len(adict_local['label']), len(adict_local)-1, split))
            ts = DataFrame(adict_local, columns=header)   # level 1 training data
            fpath = os.path.join(l1_dir, '%s-nbp%d-f%d-s%i.csv' % (split, nBP, fold, seed))  # naming: test-b3-f1-s1.csv.gz
            print('(define_level1_tset) Saving level-1 LOCAL %s set (dim=%s) to .csv: %s' % (str(ts.shape), split, fpath))
            ts.to_csv(fpath, sep=',', index=False, header=True)

            nGrand += N
        ### end foreach fold 
        
        # global dataset: consolidate scores
        for col, scores in adict.items(): 
            adict[col] = np.hstack(scores) # merge multiple np.arrays (scores) into one long score vector

        print('(global) collected %d scores per bp, combining data from all CV folds, n_bps=%d | split=%s' % (len(adict['label']), len(adict)-1, split))
        ts = DataFrame(adict, columns=header)   # level 1 training data

        # naming: number of base predictors, seed number
        fpath = os.path.join(l1_dir, '%s-nbp%d-s%i.csv' % (split, nBP, seed))

        # [log] 10 * 5 = 50 bps => dim:  (767, 51)  
        print('(define_level1_tset) Saving level-1 GLOBAL %s set (dim=%s) to .csv: %s' % (str(ts.shape), split, fpath))
        ts.to_csv(fpath, sep=',', index=False, header=True)
        
    return

def test(): 

    # create level-1 training data 
    for split in ['valid', 'test', ]: 
        define_level1_tset(split=split)

    return

if __name__ == "__main__": 
    test()






