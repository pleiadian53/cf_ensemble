from glob import glob
import gzip
from os.path import abspath, exists, isdir
from sys import argv

from numpy import array, random
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed
from os import makedirs
from utilities import load_properties, fmax_score, get_set_preds, get_num_cores, get_path_bag_weight, get_bps, aggregate_predictions
from pandas import concat, read_csv, DataFrame, Series
from itertools import product
import math


def get_max_predictions(predictors, seed, fold, set): 
    """

    Memo
    ----
    1. a parser
    """
    max_p = ''
    max_w = 0

    path, bag, weight = get_path_bag_weight(predictors[0])  # e.g. '/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.functions.SimpleLogistic_bag0, 0.6982055464926591 ',
    if weight > max_w:
        max_w = weight
        max_p = path

    for bp in predictors[1:]:
        path, bag, weight = get_path_bag_weight(bp)
        if weight > max_w:
            max_w = weight
            max_p = path
    
    set = 'test'
    #print 'GET_MAX_PREDICTIONS FOR THE BEST BP, I.E., %s_bag%s (based on the order file obtained form the validation set)\n' % (max_p, bag)
    y_true, y_score = get_set_preds(max_p, set, bag, fold, seed)   # looks into the classifier directory and fetch the test split result (e.g. test-b5-f2-s1.csv.gz)
    perf = fmax_score(y_true, y_score)  
    return (y_true, y_score, ('%s_bag%s' % (max_p, max_w)))


def BEST_bp0(parameters):
    size, seed = parameters   
    
    # y_true = DataFrame(columns = ["label"], dtype=np.int8)
    # y_score = DataFrame(columns = ["prediction"])
    y_true, y_score = Series(), Series()

    string = ""
    for fold in range(fold_count):
        ensemble_bps = get_bps(project_path, seed, metric, size)[0]  # a list of classifiers (in terms of paths to their weka directories)
        
        inner_y_true, inner_y_score, bp = get_max_predictions(ensemble_bps, seed, fold, "test")  # inner_y_true: N-by-1 df
        # print('... fold=%d | inner_y_true(%s):\n%s\n' % (fold, inner_y_true.columns.values, str(inner_y_true.shape)))
        
        # y_true = concat([y_true, inner_y_true], axis = 0)
        # y_score = concat([y_score, inner_y_score], axis = 0)
        # y_true.append(inner_y_true.label.values)
        # y_score.append(inner_y_score.prediction.values) 

        y_true = y_true.append(inner_y_true.label, ignore_index=True)  # append is not inplace! 
        y_score = y_score.append(inner_y_score.prediction, ignore_index=True)

        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))

    # y_true = np.hstack(y_true) # y_true['label'].values 
    # y_score = np.hstack(y_score) # y_score['prediction'].values

    string += ("final,%f\n" % fmax_score(y_true, y_score))
    filename = '%s/%s/%s%i/BP_bp%i_seed%i.fmax' % (project_path, directory, subdirectory, seed, size, seed)

    with open(filename, 'wb') as f:
    	f.write(string)
    f.close()
    print filename 

def BEST_bp(parameters):
    size, seed = parameters   
    
    # y_true = DataFrame(columns = ["label"], dtype=np.int8)
    # y_score = DataFrame(columns = ["prediction"])
    y_true, y_score = Series(), Series()

    string = ""
    for fold in range(fold_count):
        ensemble_bps = get_bps(project_path, seed, metric, size)[0]  # a list of classifiers (in terms of paths to their weka directories)
        
        # [note] inner_y_true is a Series (original was a dataframe)
        inner_y_true, inner_y_score, bp = get_max_predictions(ensemble_bps, seed, fold, "test")  # inner_y_true: N-by-1 df

        y_true = y_true.append(inner_y_true, ignore_index=True)  # append is not inplace! 
        y_score = y_score.append(inner_y_score, ignore_index=True)

        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true.values, inner_y_score.values)))

    string += ("final,%f\n" % fmax_score(y_true.values, y_score.values))
    filename = '%s/%s/%s%i/BP_bp%i_seed%i.fmax' % (project_path, directory, subdirectory, seed, size, seed)

    with open(filename, 'wb') as f:
        f.write(string)
    f.close()
    print filename
    

def FULL_ens0(parameters):
    size, seed = parameters

    # y_true = DataFrame(columns = ["label"])
    # y_score = DataFrame(columns = ["prediction"])
    y_true, y_score = [], []
    
    string = ""
    for fold in range(fold_count):
        ensemble_bps = get_bps(project_path, seed, metric, size)[0]
        inner_y_true, inner_y_score = aggregate_predictions(ensemble_bps, seed, fold, "test", RULE)
        
        # y_true = concat([y_true, inner_y_true], axis = 0)
        # y_score = concat([y_score, inner_y_score], axis = 0)
        y_true.append(inner_y_true.label.values)
        y_score.append(inner_y_score.prediction.values) 
        
        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))

    y_true = np.hstack(y_true) # y_true['label'].values 
    y_score = np.hstack(y_score) # y_score['prediction'].values
    
    string += ("final,%f\n" % fmax_score(y_true, y_score))
    filename = '%s/%s/%s%i/FE_bp%i_seed%i_%s.fmax' % (project_path, directory, subdirectory, seed, size, seed, RULE)

    with open(filename, 'wb') as f:
        f.write(string)
    f.close()
    print filename

def FULL_ens(parameters):
    size, seed = parameters

    # y_true = DataFrame(columns = ["label"])
    # y_score = DataFrame(columns = ["prediction"])
    y_true, y_score = Series(), Series()
    
    string = ""
    for fold in range(fold_count):
        ensemble_bps = get_bps(project_path, seed, metric, size)[0]
        inner_y_true, inner_y_score = aggregate_predictions(ensemble_bps, seed, fold, "test", RULE)
        
        # y_true = concat([y_true, inner_y_true], axis = 0)
        # y_score = concat([y_score, inner_y_score], axis = 0)
        y_true = y_true.append(inner_y_true)
        y_score = y_score.append(inner_y_score) 
        
        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true.values, inner_y_score.values)))
    
    string += ("final,%f\n" % fmax_score(y_true.values, y_score.values))
    filename = '%s/%s/%s%i/FE_bp%i_seed%i_%s.fmax' % (project_path, directory, subdirectory, seed, size, seed, RULE)

    with open(filename, 'wb') as f:
        f.write(string)
    f.close()
    print filename

def baselines(parameters):
    BEST_bp(parameters)
    FULL_ens(parameters)

print "\nStarting . . ."

# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)
directory    = 'BASELINE'
subdirectory = 'ORDER'
dirnames     = sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))

# load and parse project properties
p          = load_properties(project_path)
fold_count = int(p['foldCount'])
seeds      = int(p['seeds'])
metric     = p['metric']
RULE       = p['RULE']


max_num_clsf = len(dirnames) * seeds
sizes        = range(1,max_num_clsf+1)

# generate BASELINE directory
if not exists('%s/%s/' % (project_path, directory)):
    makedirs('%s/%s/' % (project_path, directory))

# generate ORDER directory
for o in range(seeds):
    if not exists("%s/%s/%s%i" % (project_path, directory, subdirectory, o)):
        makedirs("%s/%s/%s%i" % (project_path, directory, subdirectory, o))

def run(): 
    all_parameters = list(product(sizes, range(seeds)))
    Parallel(n_jobs = get_num_cores(), verbose = 50)(delayed(baselines)(parameters) for parameters in all_parameters)

    print "Done!\n"

def test(): 

    seed = 0
    fold = 1
    predictors = ['/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.functions.SimpleLogistic_bag0, 0.6982055464926591 ', '/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.trees.RandomForest_bag1, 0.69375 ',]
    get_max_predictions(predictors, seed, fold, 'test')

    return

if __name__ == "__main__": 
    run()




