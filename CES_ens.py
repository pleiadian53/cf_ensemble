from glob import glob
import gzip, time
from os.path import abspath, exists, isdir
from sys import argv
from random import randrange

from numpy import array
import numpy as np 

from sklearn.metrics import roc_auc_score
from os import makedirs
from utilities import load_properties, fmax_score, get_set_preds, get_fold_ens, get_path_bag_weight, aggregate_predictions
from pandas import concat, read_csv, DataFrame


def get_ens_bps(ensemble, filename_fold):
    with open(filename_fold, 'r') as f:
        content = [line.strip() for line in f]
        for line in content:
            if "Base predictors and their weight (performance on the validation, over 5 folds):" in line:
                index = content.index(line)
                break
        dirnames = [bp for bp in content[index+1:index+1+size]]
        ensemble_bps = [dirnames[bp-1] for bp in ensemble]
    f.close()
    return ensemble_bps


def CES_ens():
    start_time = time.time()

    # y_true = DataFrame(columns = ["label"])
    # y_score = DataFrame(columns = ["prediction"])
    y_true, y_score = [], []

    string = ""
    for fold in range(fold_count):
        filename_fold = '%s/CES_OUTPUT/ORDER%s/bp%s_fold%s_seed%s_%s_start-%s.%s' % (project_path, seed, size, fold, seed, RULE, start, metric)
        ensemble = get_fold_ens(filename_fold)
        ensemble_bps = get_ens_bps(ensemble, filename_fold)
        inner_y_true, inner_y_score = aggregate_predictions(ensemble_bps, seed, fold, "test", RULE)
        
        # y_true = concat([y_true, inner_y_true], axis = 0)
        # y_score = concat([y_score, inner_y_score], axis = 0)

        y_true.append(inner_y_true.label.values)
        y_score.append(inner_y_score.prediction.values) 

        string += ("fold_%i,%f\n" % (fold, fmax_score(inner_y_true, inner_y_score)))

    y_true = np.hstack(y_true) # y_true['label'].values 
    y_score = np.hstack(y_score) # y_score['prediction'].values
    string += ("final,%f\n" % fmax_score(y_true, y_score))

    dst = '%s/CES_RESULTS/ORDER%i/CES_bp%i_seed%i_%s_start-%s.%s' % (project_path, seed, size, seed, RULE, start, metric) 
    with open(dst, 'wb') as f:
	f.write(string)
    f.close()
    print "\t%s (%s)"  % (dst, (time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))))



project_path = abspath(argv[1])
assert exists(project_path)
size = int(argv[2])
seed = int(argv[3])
RULE = argv[4]
start = argv[5]
metric = argv[6]
p = load_properties(project_path)
fold_count = int(p['foldCount'])
CES_ens()



