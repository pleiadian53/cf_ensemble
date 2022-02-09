import time, gzip, copy
from glob import glob
from os.path import abspath, exists, isdir
from os import makedirs
from sys import argv
from numpy import array, argmax, random
from numpy.random import choice
from sklearn.metrics import roc_auc_score
from utilities import load_properties, fmax_score, get_set_preds, get_bps, bps2string, get_path_bag_weight, aggregate_predictions
from pandas import concat, read_csv, DataFrame
from shutil import copyfile
from copy import deepcopy


def select_top_classifier(classifiers, seed, fold, RULE):
    scores = [fmax_score(*aggregate_predictions([classifiers[i]], seed, fold, "valid", RULE)) for i in range(len(classifiers))]
    top_classifier = classifiers[argmax(scores)]
    classifiers.remove(top_classifier)
    return top_classifier


def get_potential_ensembles(ensemble, classifiers):
    #print "\tPOTENTIAL ENSEMBLES checking complementarity:"
    #for c in classifiers:
    #    print "\t %s" % c
    all_ens = []
    for c in classifiers:
        current = list(ensemble)
        current.append(c)
        all_ens.append(list(current))
    return all_ens        


def find_ensemble(ensemble, classifiers, seed, fold, RULE, set_ensembles):
    if len(classifiers) == 0 or len(ensemble) == max_ens_size:
        return ensemble
    else:
        potential_ensembles = get_potential_ensembles(ensemble, random.choice(classifiers, len(classifiers), replace=False))
        scores = [fmax_score(*aggregate_predictions(pe, seed, fold, "valid", RULE)) for pe in potential_ensembles]
        ensemble.append(potential_ensembles[argmax(scores)][-1])
        #print "\t adding CURRENT ENSEMBLE:"
        #for c in ensemble:
        #    print "\t - %s" % c
        #print "\t ==> $%s" % max(scores)
        set_ensembles[tuple(deepcopy(ensemble))] = max(scores)
        classifiers.remove(potential_ensembles[argmax(scores)][-1])
        find_ensemble(ensemble, classifiers, seed, fold, RULE, set_ensembles)


def CES_run():
    start_time = time.time()
    classifiers, bps_weight = get_bps(project_path, seed, metric, size)
    original = deepcopy(classifiers)
    ensemble = []
    set_ensembles = {}
    for i in range(init_ens_size):
        ensemble.append(select_top_classifier(classifiers, seed, fold, RULE))
    find_ensemble(ensemble, classifiers, seed, fold, RULE, set_ensembles)

    if len(set_ensembles) > 0: 
        sel_ens = max(set_ensembles, key=set_ensembles.get)
    else:
        sel_ens = ensemble
    actual = []
    for o in range(len(original)):
        if original[o] in sel_ens:
            actual.append(o+1)

    val_score = fmax_score(*aggregate_predictions(sel_ens, seed, fold, "valid", RULE))
    test_score = fmax_score(*aggregate_predictions(sel_ens, seed, fold, "test", RULE))
    seconds  = time.time() - start_time
    string = "Fold_%i (val = %f) (test = %f) :: (%s) [%s]\n%s" % (fold, val_score, test_score, ", ".join(str(a) for a in actual), time.strftime('%H:%M:%S', time.gmtime(seconds)), bps2string(original))
    dst = '%s/CES_OUTPUT/ORDER%i/bp%i_fold%i_seed%i_%s_start-%s.fmax' % (project_path, seed, size, fold, seed, RULE, init_ens_size)
    with open(dst, 'wb') as f:
        f.write('%s' % string)
    f.close()
    print "\t%s (%s)"  % (dst, (time.strftime('%H:%M:%S', time.gmtime(seconds))))




  

project_path = abspath(argv[1])
assert exists(project_path)
size = int(argv[2])
fold = int(argv[3])
seed = int(argv[4])
RULE = argv[5]
init_ens_size = int(argv[6])
metric = argv[7]
max_ens_size = float(size)
CES_run()




