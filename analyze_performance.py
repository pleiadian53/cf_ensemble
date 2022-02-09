# encoding: utf-8

import pandas as pd
from pandas import DataFrame, Series 
import numpy as np

import os, shutil, subprocess, sys, collections
import getpass
import random, re
from glob import glob

from tabulate import tabulate  # pip install tabulate // not in standard library

import common
from utils_sys import div, parse_params_list, format_list, format_sort_dict
from cf_spec import MFEnsemble

# import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
# import matplotlib.pyplot as plt

# # select plotting style 
# plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

# sklearn forces warning
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")

import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support


#################################################
p_file_base = re.compile(r"(?P<method>\w+)\.S-prediction\.csv")
p_file_cf = re.compile(r"(?P<method>\w+)\.S-(?P<mf>\w+)-prediction\.csv")
#################################################

class Job(object): 
    options = ''
    args = '' 

    # domains = ['thaliana', 'drosophila', 'elegans', 'pacificus', 'remanei', 'sl', ]

    domain = ''
    settings = [] 

class Stacker(object): 
    columns = ['fold', 'method', 'label', 'prediction']
    types = ['wmf_stacker', ]
    pdict = {'bp_stacker': p_file_base, 'wmf_stacker': p_file_cf}  

    # NOTE
    # bp.C-prediction: combined base predictors
    # bp_stacker.C-prediction: combined stacker predictions

class Analysis(object): 
    """

    Memo
    ----
    1. Examples of candidate models: 

    ['ada' 'ada+F100_A100_CFclassifier' 'ada+F100_A100_CFsample' 'enet'
 'enet+F100_A100_CFclassifier' 'enet+F100_A100_CFsample' 'knn'
 'knn+F100_A100_CFclassifier' 'knn+F100_A100_CFsample' 'log'
 'log+F100_A100_CFclassifier' 'log+F100_A100_CFsample'
 'mean+F100_A100_CFclassifier' 'mean+F100_A100_CFsample' 'naive'
 'naive+F100_A100_CFclassifier' 'naive+F100_A100_CFsample' 'qda'
 'qda+F100_A100_CFclassifier' 'qda+F100_A100_CFsample' 'rf'
 'rf+F100_A100_CFclassifier' 'rf+F100_A100_CFsample' 'svm'
 'svm+F100_A100_CFclassifier' 'svm+F100_A100_CFsample']

    """
    # columns of 'performance_table_threshold_{t}.csv' 
    columns = ['seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model'] # 'model' NOT 'method' here

    working_dir = os.getcwd()
    project_path = os.getcwd()
    analysis_path = os.path.join(os.getcwd(), 'analysis')

    models = ['ada' 'enet' 'knn' 'log' 'naive' 'qda' 'rf' 'svm', 'mean'] + ['latent_mean', 'masked_latent_mean', ]
    modelDict = {'log': 'Logistic', 'qda': 'QDA', 'enet': 'ElasticNet', 'svm': 'SVM', 
                   'naive': 'NaiveBayes', 'rf': 'RandomForest', 'ada': 'AdaBoost', 'knn': 'kNN', 
                   'mean': 'mean', 'latent_mean': 'LatentMean', 'latent_mean_masked': 'MaskedLatentMean'}
    orders = ['Logistic', 'ElasticNet', 'SVM', 'QDA', 'NaiveBayes', 'kNN',  'AdaBoost', 'RandomForest', 'mean', ] # ['AdaBoostM1','LogitBoost','NaiveBayes','Logistic','SMO','VotedPerceptron','IBk','PART','J48','RandomForest']

    name_map = replacement = {'user': 'classifier', 'item': 'sample'}  # e.g. replace 'user' by 'classifier' in method naming
    inv_name_map = {v: k for k, v in name_map.items()}

    @staticmethod
    def config(domain='', analysis_dn='analysis', create_dir=False):
        if domain: 
            # resolve project path e.g. /Users/<user>/work/data/pf1
            # home_dir = os.path.expanduser('~')
            # working_dir_default = '/'.join([home_dir, 'work/data', ])
            parentdir = os.path.dirname(os.getcwd())
            Analysis.working_dir = os.path.join(parentdir, 'data')  # e.g. /Users/<user>/work/data
            Analysis.project_path = os.path.join(Analysis.working_dir, domain) # e.g. /Users/<user>/work/data/pf1
            if not os.path.exists(Analysis.project_path): 
                if create_dir: 
                    os.mkdir(Analysis.project_path)
                else: 
                    msg = "Invalid project path (which includes domain): {data_path}".format(data_path=Analysis.project_path)
                    raise ValueError(msg)
            Analysis.analysis_path = os.path.join(Analysis.project_path, analysis_dn)  # dn: directory name
            if not os.path.exists(Analysis.analysis_path):
                os.mkdir(Analysis.analysis_path)
        else: 
            # use default 
            pass 
        # print('(verify) analysis_path: {p}'.format(p=Analysis.analysis_path))
    
    @staticmethod
    def rename(models, delimit='+'):  
        # delimit: separator between stacker name and the meta-learning method (an MF method of particular parameter setting)
        pass 
    @staticmethod
    def order(models, delimit='+'):
        def to_index(name):
            delimit_index = name.find(delimit)
            if delimit_index > 0: 
                return delimit_index
            return len(name) 

        rankmap = {model: i for i, model in enumerate(Analysis.orders)}

        versions = {}
        for model in models: 
            stacker, meta = model, ''
            if model.find(delimit) > 0: 
                # this is a stacker method applied to a transformed dataset 
                stacker, meta = model.split(delimit)
            if not stacker in versions: versions[stacker] = 1
            versions[stacker] += 1  # how many meta model does this stacker have? 

        rankmap2 = {}
        for model in models:
            stacker, meta = model, ''
            if model.find(delimit) > 0: 
                # this is a stacker method applied to a transformed dataset 
                stacker, meta = model.split(delimit)
            if not meta: 
                rankmap2[model] = rankmap.get(model, len(models)) # should already be in the rankmap
            else: 
                rankmap2[model] = rankmap.get(stacker,len(models)) # + 1/(version[stacker]+0.0)
        
        return sorted(models, key=lambda e: rankmap2[e], reverse=False)

# Given true labels, regression predictions, predicting class, get precision, recall and f_score at the F-max point
def precision_recall_fscore(labels, predictions, predict_label,beta=1.0):
    thresholds = np.arange(0,1,0.05)
    precisions = []
    recalls = []
    f1_scores = []
    for thr in thresholds:
        bin_pred = np.digitize(predictions, [thr])
        pre, rec, f1,_ = precision_recall_fscore_support(labels, bin_pred, beta=1.0, labels=None, pos_label=predict_label, average = 'binary')
        precisions.append(pre)
        recalls.append(rec)
        f1_scores.append(f1)
    ind = np.nanargmax(f1_scores)
    return thresholds[ind], precisions[ind], recalls[ind], f1_scores[ind]

def fmax_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)
    precision += 0.0001
    recall += 0.0001
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return thresholds[np.argmax(f1)]

# Given the prediction dataframe contains labels and predictions
# get the threshold that yields the best (pre + rec)/2 score
def get_opt_balanced_acc_thershold(labels, predictions, model=''):  # arguments: prediction_df,model=''
    # labels = prediction_df.label.tolist()
    # predictions = prediction_df.prediction.tolist()
    thres = np.arange(0,1,0.05)
    average_scores = []
    sens = []
    spec = []
    for thr in thres: # foreach threshold
        bin_pred = np.digitize(predictions,[thr])
        sen = calculate_sensitivity(bin_pred,labels)
        spe = calculate_specificity(bin_pred,labels)
        sens.append(sen)
        spec.append(spe)
        score = float(sen + spe) / 2.0
        average_scores.append(score)       
    ind = np.argmax(average_scores)
    opt_thr = thres[ind]
    return opt_thr

def get_score_by_average_threshold(prediction_df,model):
    labels = prediction_df.label.tolist()
    predictions = prediction_df.prediction.tolist()
    
    thres = np.arange(0,1,0.05)
    average_scores = []
    sens = []
    spec = []
    for thr in thres:
        bin_pred = np.digitize(predictions,[thr])
        sen = calculate_sensitivity(bin_pred,labels)
        spe = calculate_specificity(bin_pred,labels)
        sens.append(sen)
        spec.append(spe)
        
        score = float(sen + spe) / 2.0
        average_scores.append(score)       
    ind = np.argmax(average_scores)
    opt_thr = thres[ind]
    opt_sen = sens[ind]
    opt_spe = spec[ind]
    auc = sklearn.metrics.roc_auc_score(labels,predictions)
    df = pd.DataFrame({'threshold': [opt_thr],'sensitivity':[opt_sen],'specificity':[opt_spe],'auc':[auc],'model':[model]})
    return df

def get_scores(prediction_df,model):
    label = prediction_df.label.tolist()
    prediction = prediction_df.prediction.tolist()
    threshold = fmax_threshold(label,prediction,beta=1.0)
#    threshold = 0.5
#     print threshold
    binary_prediction = np.digitize(prediction,[threshold])
    sens = calculate_sensitivity(binary_prediction,label)
    spec = calculate_specificity(binary_prediction,label)
    auc = sklearn.metrics.roc_auc_score(label,prediction)
    df = pd.DataFrame({'threshold': [threshold],'sensitivity':[sens],'specificity':[spec],'auc':[auc],'model':[model]})
    return df

def get_scores_fix_thr(prediction_df,model,threshold=0.5):
    label = prediction_df.label.tolist()
    prediction = prediction_df.prediction.tolist()
    binary_prediction = np.digitize(prediction,[threshold])
    sens = calculate_sensitivity(binary_prediction,label)
    spec = calculate_specificity(binary_prediction,label)
    auc = sklearn.metrics.roc_auc_score(label,prediction)
    df = pd.DataFrame({'threshold': [threshold],'sensitivity':[sens],'specificity':[spec],'auc':[auc],'model':[model]})
    return df

# Given true labels, regression predictions, predicting class, threshold, get precision, recall and f_score.
def get_precision_recall_scores(prediction_df,model,predict_label,threshold):
    label = prediction_df.label.tolist()
    prediction = prediction_df.prediction.tolist()
    binary_prediction = np.digitize(prediction,[threshold])
    
    precision = calculate_precision(binary_prediction, label,predict_label)
    recall = calculate_recall(binary_prediction, label,predict_label)
    f1 = calculate_fscore(binary_prediction, label,predict_label)
    df = pd.DataFrame({'threshold': [threshold],'precision':[precision],'recall':[recall],'f_score':[f1],'model':[model]})

    return df

'''
    Input for calculating sensitivity, specificity, precision, recall 
    is the binary prediction vector and true binary label vector
''' 
def calculate_sensitivity(predictions,labels):
    tn, fp, fn, tp =  sklearn.metrics.confusion_matrix(labels,predictions).ravel()
    return float(tp)/(float(tp) + float(fn))

def calculate_specificity(predictions,labels):
    tn, fp, fn, tp =  sklearn.metrics.confusion_matrix(labels,predictions).ravel()
    #print tn,fp,fn,tp
    return float(tn)/(float(tn) + float(fp))

def calculate_precision(predictions,labels,predict_label):
    return  sklearn.metrics.precision_score(labels, predictions,pos_label=predict_label)
    
def calculate_recall(predictions,labels,predict_label):
    return  sklearn.metrics.recall_score(labels, predictions,pos_label=predict_label)

def calculate_fscore(predictions,labels,predict_label):
    return  sklearn.metrics.f1_score(labels, predictions,pos_label=predict_label)


# Tasks
#########################################################################################################

"""
AB.S-prediction.csv  KNN.S-prediction.csv  LR.S-prediction.csv  RF.S-prediction.csv               
SVM.S-prediction.csv  DT.S-prediction.csv  LB.S-prediction.csv   NB.S-prediction.csv  

RandomForest-base-prediction.csv  
ces-prediction.csv    performance.csv
mean-prediction.csv  selection-enhanced-fmax-iterations.csv
"""

# combined base predictions across multiple classifiers and multiple runs

###################################
base_fn = 'A1C_r10_preds.csv'   # attribute: label,method,prediction,fold (or round)
f_baseline = "baselines_A1C.txt"
###################################

def mean_stacker(test_df, test_labels=None, bag_count=10): 
    return aggregate(test_df, op=np.mean)
def median_stacker(test_df, test_labels=None, bag_count=10):
    return aggregate(test_df, op=np.median) 
def aggregate(test_df, test_labels=None, op=np.mean, unbag_=True, bag_count=10):
    # import common
    if unbag_:  
        test_df = common.unbag(test_df, bag_count)
    T = test_df.values   # format: n_data by n_features/n_classifiers
    assert hasattr(op, '__call__')
    test_predictions = op(T, axis=1)
    return test_predictions

def load_performance_dataframe_baseline(domain, policy_threshold='fmax', sep=','): 
    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # best baseline 
    fpath = os.path.join(analysis_path, 'high.B-{th}.csv'.format(th=policy_threshold))
    df_best_base = pd.read_csv(fpath, sep=sep, header=0, index_col=False)
    print('... (verify) loaded best baseline profile:\n{df}\n'.format(df=df_best_base))

    # all baseline performance (per seed, per model): seed,predict_label,threshold,precision,recall,f_score,model
    fpath = os.path.join(analysis_path, 'performance_baseline_threshold_{t}.csv'.format(t=policy_threshold))
    df_all_base = pd.read_csv(fpath, sep=sep, header=0, index_col=False)

    return df_best_base, df_all_base 
def prepare_performance_dataframe_baseline(domain, topk=-1, **kargs): 
    """

    Input
    -----
        metric: the metric used to select the top K model (when topk > 0)

    Related 
    -------
    prepare_performance_dataframe()


    Use 
    ---
    prepare_performance_dataframe_baseline(domain, policy_threshold='fmax', policy_iter='subsampling', n_runs=10)

    """
    from evaluate import Metrics, plot_roc
    from evaluate import PerformanceMetrics, analyzeBasePerf # as perfm
    import utils_cf as uc
    import cf_spec

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined
    print('(base_predictors) project_path: {p}'.format(p=project_path))

    policy_iter = kargs.get('policy_iter', 'subsampling')  # train_dev_test
    policy_threshold = kargs.get('policy_threshold', 'fmax') # {'balanced', 'fmax', }
    sep = kargs.get('sep', ',')  # separator for output dataframes (there are 2)

    agg = kargs.get('agg', 10)
    n_runs = kargs.get('n_runs', 10)  
    fold_count = kargs.get('fold_count', 5)

    n_labels = 2 
    labelSet = [0, 1]

    df_scores = []
    metrics = ['precision','recall','f_score']

    # fmax_scores, balanced_scores = {}, {}  # to be sorted
    pos_scores, neg_scores = {}, {}
    indices = kargs.get('indices', range(fold_count) if policy_iter == 'cv' else range(n_runs))
    models = set()
    # meta_models = ['mean', ]

    for index in indices:

        # if policy_iter == 'cv': 
        #     R, T, L_train, L_test, models = uc.to_rating_matrix(fold=index, project_path=project_path, verbose=True)
        # else: 
        #     R, T, L_train, L_test, models = uc.to_rating_matrix_random_subsampling(project_path=project_path, dev_ratio=0.2, test_ratio=0.2, 
        #         train_dev_test=False, shuffle=True, fold_count=5)

        # n_users, n_items = R.shape[0], R.shape[1]
        # labelSet = np.unique(L_train)
        # n_labels = len(labelSet)
        # print('... cycle = {id} | n_train: {ntr}, n_test: {nt} | dim(T): {dt}'.format(id=index, dt=T.shape, nt=len(L_test), ntr=len(L_train) ))

        if policy_iter.startswith(('cv', 'cross')): 
            _,_,test_df,label = common.read_fold(project_path, index)
        else: 
            _,_,test_df,label = common.shuffle_split(project_path, split_number=2, dev_ratio=0.2, test_ratio=0.2)

        test_df = common.unbag(test_df, agg)  # if agg = 1: average all bags
        if index == 0: 
            print('... base models (n={n}): {mset}'.format(n=len(test_df.columns.values), mset= test_df.columns.values))   # ['NaiveBayes' 'Logistic' 'SMO' 'AdaBoostM1' 'RandomForest']

        models.update(test_df.columns.values)
        labelSet = np.unique(label)
        n_labels = len(labelSet)

        for j, model in enumerate(test_df.columns): 
           
            prediction = test_df[model].values  # each column references a prediction vector for a given method

            if policy_threshold == 'fmax': 
                # warning: Precision and f_score are ill-defined and being set to 0.0 due to no predicted samples.
                pos_thr,pos_pre,pos_rec,pos_f1 =  precision_recall_fscore(label, prediction, 1, beta=1.0)  # Given true labels, regression predictions, predicting class, get precision, recall and f_score at the F-max point
                neg_thr,neg_pre,neg_rec,neg_f1 =  precision_recall_fscore(label, prediction, 0, beta=1.0)
        
                # >>> pos_thr vs neg_thr
                performance_df_pos = pd.DataFrame({'seed':index,'predict_label':[1],'threshold': [pos_thr],'precision':[pos_pre],'recall':[pos_rec],'f_score':[pos_f1],'model':[model]})
                performance_df_neg = pd.DataFrame({'seed':index,'predict_label':[0],'threshold': [neg_thr],'precision':[neg_pre],'recall':[neg_rec],'f_score':[neg_f1],'model':[model]}) 

            elif policy_threshold.startswith('bal'):

                thr = get_opt_balanced_acc_thershold(label, prediction, model)

                bin_pred = np.digitize(prediction,[thr])
                pos_pre, pos_rec, pos_f1,_ = precision_recall_fscore_support(label, bin_pred, beta=1.0, labels=None, pos_label=1, average = 'binary') 
                neg_pre, neg_rec, neg_f1,_ = precision_recall_fscore_support(label, bin_pred, beta=1.0, labels=None, pos_label=0, average = 'binary') 

                # index_col for this new dataframe will be 'seed'
                # >>> both labels share the same threshold 
                performance_df_pos = pd.DataFrame({'seed':index,'predict_label':[1],'threshold': [thr],'precision':[pos_pre],'recall':[pos_rec],'f_score':[pos_f1],'model':[model]})
                performance_df_neg = pd.DataFrame({'seed':index,'predict_label':[0],'threshold': [thr],'precision':[neg_pre],'recall':[neg_rec],'f_score':[neg_f1],'model':[model]})

            # baseline
            if not model in pos_scores: pos_scores[model] = []
            if not model in neg_scores: neg_scores[model] = []
            pos_scores[model].append( [pos_pre, pos_rec, pos_f1] ) # {'precision': pos_pre, 'recall': pos_rec, 'f_score': pos_f1} | [pos_pre, pos_rec, pos_f1]
            neg_scores[model].append( [neg_pre, neg_rec, neg_f1] )

            # one score set per model
            df_scores.append(performance_df_pos)
            df_scores.append(performance_df_neg) 

        # add additional aggregate models here 
        # columns: fold,id,label,prediction,diversity,method

        # for j, model in enumerate(meta_models): 
        #     routine = '{name}_stacker'.format(model) 
        #     try: 
        #         prediction = globals()[routine](test_df, label)  # or? test_predictions = getattr(stacking, '{name}_stacker'.format(stacker))(test_df, test_labels)
        #     except: 
        #         raise NameError("Unknown special stacker: {name}".format(name=model))


    ### end foreach cycle
    print('... gathering {n} sets of predictive scores (verify)'.format(n = len(df_scores)))
    df_scores = pd.concat(df_scores)  # each model has n(indices) number of score sets

    print('... dim(df_scores): {dim} <n_rows = n(rounds) * n(models) * n(labels) = {nr} * {nm} * {nl}>\n'.format(dim=df_scores.shape, nr=len(indices), nm=len(models), nl=n_labels))
    print( tabulate(df_scores.head(10), headers='keys', tablefmt='psql') )  

    # find the best baseline 
    for method, scores in pos_scores.items(): 
        pos_scores[method] = np.mean(scores, axis=0)  # mean across rounds
    for method, scores in neg_scores.items(): 
        neg_scores[method] = np.mean(scores, axis=0)  # mean across rounds

    target_metric = 'f_score'  # use this metric as a gauge 
    order = {metric: i for i, metric in enumerate(metrics)}
    pos_sorted_mean = sorted( [(m, s[order[target_metric]]) for m, s in pos_scores.items()], key=lambda x: x[1], reverse=True)
    neg_sorted_mean = sorted( [(m, s[order[target_metric]]) for m, s in neg_scores.items()], key=lambda x: x[1], reverse=True)
    for i, (m, s) in enumerate(pos_sorted_mean): 
        print('... rank #{r} | method: {m}, score: {s}'.format(r=i, m=m, s=s if i > 0 else '%f (metric=%s)' % (s, target_metric)))
    best_method, best_score = pos_sorted_mean[0]
    div(message="(verify) Best method: {method}, best score: {score} using '+ fmax' scores " .format(method=best_method, score=best_score), symbol='#', border=2)

    # build baseline table
    header = ['domain', 'best', 'label', 'precision', 'recall', 'f_score']
    sdict = {h: [] for h in header}
    for label in labelSet:     
        # best_predictions = predictions.iloc[:,ith] 
        # ret = common.fmax_precision_recall_scores(labels, best_predictions, beta = 1.0, pos_label = label)  # the point of precision, recall that reaches fmax    
        # ...  key:  id, precision, recall, f/fmax
        method_scores = pos_scores if label in (1, '+', 'pos',) else neg_scores
            
        sdict['label'].append(label) 
        sdict['precision'].append( method_scores[best_method][order['precision']] )
        sdict['recall'].append(method_scores[best_method][order['recall']] )
        sdict['f_score'].append(method_scores[best_method][order['f_score']] )
    sdict['domain'] = [domain] * n_labels
    sdict['best'] = [best_method] * n_labels
    df_best_base = pd.DataFrame(sdict, index=range(n_labels), columns=header)

    if kargs.get('save', True): 
        fpath = os.path.join(analysis_path, 'performance_baseline_threshold_{t}.csv'.format(t=policy_threshold))  # n=n_cycles
        df_scores.to_csv(fpath, index=False, sep=sep)
        print('(output) saved performance dataframe ({t}) to:\n{p}\n'.format(t=policy_threshold, p=fpath))

        # baseline reference 
        fpath = os.path.join(analysis_path, 'high.B-{th}.csv'.format(th=policy_threshold))
        df_best_base.to_csv(fpath, sep=sep, index=False, header=True)  # baselines_A1C.txt
        print("(output) baseline dataframe (best predictor='{bp}'):\n{df}\n".format(bp=best_method, df=tabulate(df_best_base, headers='keys', tablefmt='psql')))
    
    return (df_best_base, df_scores)

def meta_stackers(domain, **kargs):
    """
    Create additional stackers on baseline data but whose outputs are not considered as level-1 data. 

    """
    from evaluate import Metrics, plot_roc
    from evaluate import PerformanceMetrics, analyzeBasePerf # as perfm
    import utils_cf as uc
    import cf_spec

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined
    print('(meta_stackers) project_path: {p}'.format(p=project_path))

    policy_iter = kargs.get('policy_iter', 'subsampling')  # train_dev_test
    policy_threshold = kargs.get('policy_threshold', 'fmax') # {'balanced', 'fmax', }
    sep = kargs.get('sep', ',')  # separator for output dataframes (there are 2)

    agg = kargs.get('agg', 10)
    n_runs = kargs.get('n_runs', 10)  
    fold_count = kargs.get('fold_count', 5)

    indices = kargs.get('indices', range(fold_count) if policy_iter == 'cv' else range(n_runs))
    models = kargs.get('models', ['mean', ])

    header = ['fold', 'method', 'label', 'prediction' ]
    adict = {h: [] for h in header}
    for i, model in enumerate(models): 

        dfs = []
        for index in indices: 
            if policy_iter.startswith(('cv', 'cross')): 
                _,_,test_df,label = common.read_fold(project_path, index)
            else: 
                _,_,test_df,label = common.shuffle_split(project_path, split_number=2, dev_ratio=0.2, test_ratio=0.2)
            test_df = common.unbag(test_df, agg)  # if agg = 1: average all bags

            routine = '{name}_stacker'.format(name=model)
            print('(verify) globals()[routine]: {0}'.format(globals()[routine]))
            try: 
                prediction = globals()[routine](test_df, label)  # or? test_predictions = getattr(stacking, '{name}_stacker'.format(stacker))(test_df, test_labels)
            except: 
                raise NameError("Unknown special stacker: {name}".format(name=model))

            df = pd.DataFrame({'fold': index, 'id': test_df.index.get_level_values('id'), 'label': label, 
                               'prediction': prediction, 
                               'diversity': common.diversity_score(test_df.values)})
            dfs.append(df)
        df = pd.concat(dfs)  # each model has n(indices) number of score sets  
        df['method'] = model
        df['label'] = df['label'].astype(int)  

        dset_type = kargs.get('dtype', 'prediction')  # values: {'prior', 'posterior', }
        fpath = os.path.join(analysis_path, '{stacker}.S-{suffix}.csv'.format(stacker=model, suffix=dset_type))
        print('(verify) saving predictions from {name} on original level-1 data to:\n{output}\n'.format(name=model, output=fpath))
        df.to_csv(fpath, index = False)    

    return # resulting dataframe must have columns: ['fold', 'method', 'label', 'prediction', 'method', ]

def bestbase(domain, project_path='', policy_iter='cv', fold_count=5, agg=None, analysis_dn='analysis', sep=','):
    """
    
    Memo
    ----
    1. Reference module: 
       
       largeGOPred.ensemble

    2. Alternatively, run cf.base_predictors() which returns sorted methods according to a given metric


    """
    # from tabulate import tabulate  # pip install tabulate

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined
    
    labels, predictions = combine_baselines(domain=domain, project_path=project_path, policy_iter=policy_iter, fold_count=fold_count, agg=agg, sep=sep)
    labelSet = np.unique(labels)
    n_labels = len(labelSet)

    # load precomputed combined base predicitons
    fpath = os.path.join(analysis_path, 'bp.C-prediction.csv')  # fold,method,label,prediction
    assert os.path.exists(fpath), "Could not find combined, serialized base predictions at {path}".format(path=fpath)
    df_base = pd.read_csv(fpath, sep=sep, header=0, index_col=False)
    print("(bestbase) combined base bps dim: {dim} <n_instances * n_classifiers *by* 'prediction,label,method'> ... ".format(dim=df_base.shape))

    # prepare baseline performance data 
    # format 
    #     domain,cls,precision,recall,f
    #     a1c,1,0.109,0.541,0.182
    #     a1c,0,0.949,0.660,0.778

    header = ['domain', 'best', 'label', 'precision', 'recall', 'f_score', ]
    sdict  = {h:[] for h in header} # 'domain': [domain] * n_labels 

    # find the best fmax score and the corresponding base predictor
    ##########################################
    ref_label = 1   # use this label's score as a measure

    # [note] this fmax is usually slightly higher than taking the average of fmax across CV folds
    # method_scores = base_predictors(domain=domain, topk=-1, metric='fmax', policy_iter='subsampling', n_runs=10)
    scores = [common.fmax_score(labels,predictions.iloc[:,i], pos_label=ref_label) for i in range(len(predictions.columns))]
    median_score = np.median(scores)
    best_score = np.nanmax(scores)
    ith = np.nanargmax(scores)
    best_predictor = predictions.columns[ith]

    low_score = np.nanmin(scores)
    jth = np.nanargmin(scores)
    low_predictor = predictions.columns[jth]

    ##########################################
    div('... F score: (min: %f, max: %f, median: %f) | min algorithm: %s, max algorithm: %s' % \
        (low_score, best_score, median_score, low_predictor, best_predictor), symbol='#', border=2)

    # build baseline table
    for label in labelSet:     
        best_predictions = predictions.iloc[:,ith] 
        ret = common.fmax_precision_recall_scores(labels, best_predictions, beta = 1.0, pos_label = label)  # the point of precision, recall that reaches fmax    
        # ...  key:  id, precision, recall, f/fmax

        sdict['label'].append(label) 
        sdict['precision'].append(ret['precision'])
        sdict['recall'].append(ret['recall'])
        sdict['f_score'].append(ret['f_score'])
    sdict['domain'] = [domain] * n_labels
    sdict['best'] = [best_predictor] * n_labels
    df_best_base = pd.DataFrame(sdict, index=range(n_labels))

    fpath = os.path.join(analysis_path, 'high.B-{domain}.csv'.format(domain=domain))
    df_best_base.to_csv(fpath, sep=sep, index=False, header=True)  # baselines_A1C.txt

    print("(output) baseline dataframe (best predictor='{bp}'):\n{df}\n".format(bp=best_predictor, df=tabulate(df_best_base, headers='keys', tablefmt='psql')))

    return df_best_base

### format 
# label,method,prediction,round
# 0.0,SMO,1.0,r1
# 0.0,SMO,0.0,r1
# 1.0,SMO,1.0,r1
# ...
# 0.0,VotedPerceptron,0.1004948,r2
# 0.0,VotedPerceptron,1.0,r2
# 0.0,VotedPerceptron,0.3135426,r2
# 0.0,VotedPerceptron,0.0,r2
def read_baselines(df_baseline=None, f_baseline='', metrics=['precision','recall','f_score']): 
    """

    Memo
    ----
    1. format

        Data,cls,precision,recall,f
        a1c,1,0.109,0.541,0.182
        a1c,0,0.949,0.660,0.778

    """
    # columns: domain,cls,precision,recall,f
    if df_baseline is None: 
        assert os.path.exists(f_baseline), "Invalid path to the baseline performance dataframe: {p}".format(p=f_baseline)
        df_baseline = pd.read_csv(f_baseline, sep=sep, header=0, index_col=False) # 'baselines.txt'
    # metrics = ['precision','recall','f_score']
    pos_scores = df_baseline.loc[df_baseline.label==1,metrics].values[0]
    neg_scores = df_baseline.loc[df_baseline.label==0,metrics].values[0]
    # print( list(zip(metrics, pos_scores)) )
    # print( list(zip(metrics, neg_scores)) )
    print( tabulate(df_baseline, headers='keys', tablefmt='psql') )

    return (pos_scores, neg_scores)

### threshold by fmax 

# output: 
#   i) performance_table_r10_fmax_threshold.csv
#   ii) performance_boxplot_thre_fmax.pdf

def combine_baselines(domain, project_path='', policy_iter='cv', fold_count=5, agg=None, sep=',', **kargs):
    import collections

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined
    
    if not agg: 
        # try to infer bagCount
        rfold = random.choice(range(fold_count)) 
        _,_,test_df,label = common.read_fold(project_path,rfold)

        basenames = []
        for cn in test_df.columns.values: 
            if cn.find('.'): 
                basenames.append(cn.split('.')[0])
            else: 
                basenames.append(cn)
        for i, (cn, bag_count) in enumerate(collections.Counter(basenames).items()): # cn: classifier name 
            # print('... cn: %s, bag_count: %d' % (cn, bag_count))
            if i == 0: agg = bag_count
            assert agg == bag_count, "Inconsistent bag count at (classifier={cn}, bag_count={bn}) while agg={bn0}".format(cn=cn, bn=bag_count, bn0=agg)
        # print('... inferred bag count = {bn}'.format(bn=agg))
    # ... bag count determined

    predictions = []
    labels = []

    n_runs = kargs.get('n_runs', 10)  
    fold_count = kargs.get('fold_count', 5)
    indices = kargs.get('indices', range(fold_count) if policy_iter == 'cv' else range(n_runs))

    for index in indices:
        if policy_iter.startswith(('cv', 'cross')): 
            _,_,test_df,label = common.read_fold(project_path,index)
        else: 
            _,_,test_df,label = common.shuffle_split(project_path, split_number=2, dev_ratio=0.2, test_ratio=0.2)

        test_df = common.unbag(test_df, agg)  # if agg = 1: average all bags
        predictions.append(test_df)
        labels = np.append(labels,label)  # labels is an numpy.ndarray
    predictions = pd.concat(predictions)  # columns: classifiers
    
    print('(combine_bps) combined performance data dim: {dim} (n_instances by n_classifiers), base predictors: {cols}'.format(dim=predictions.shape, cols= np.unique(predictions.columns.values)) )

    # ||| => |
    #        | 
    #        |
    dfs = []
    # for i in range(len(predictions.columns)):  # foreach prediction vector of a classifier/column
    #     preds = predictions.iloc[:,i].tolist()
    #     method = predictions.columns.tolist()[i]  
    #     print('... method: {m}, preds: {pv}'.format(m=method, pv=preds[:10]))
    #     preds_df = pd.DataFrame({'prediction':preds,'label':labels, 'method': method},index=range(len(preds)))
    #     preds_df.to_csv(path + '/analysis/%s-base-prediction.csv' % method, index=False)
    #     dfs.append(preds_df)

    ##########################################
    for i, method in enumerate(predictions.columns): # foreach "classifier PV", where classifier is a column, PV: prediction vector
        preds = predictions[method].values
        preds_df = pd.DataFrame({'prediction':preds,'label':labels, 'method': method}, index=range(len(preds)))
        # ...  [note] the label becomes 'float'
        preds_df['label'] = preds_df['label'].astype(int)

        fpath = os.path.join(analysis_path, '{m}.B-prediction.csv'.format(m=method))  # B: base predictors
        preds_df.to_csv(fpath, sep=',', index=False, header=True)  # path + '/analysis/%s-base-prediction.csv' % method
        dfs.append(preds_df)
    ##########################################

    df_base = pd.concat(dfs)
    df_base['label'] = df_base['label'].astype(int)
    fpath = os.path.join(analysis_path, 'bp.C-prediction.csv')  # C: combined (S: Stacker)
    df_base.to_csv(fpath, index=False, sep=sep, header=True)

    ##################################################################
    # Output 
    # ------
    #    a. individual base predictor 
    # 
    #       {method}.B-prediction.csv  e.g. Logistic.B-prediction.csv
    #       
    #       columns: prediction,label,method
    #    
    #    b. combined 
    #       
    #       bp.C-prediction.csv
    #    
    ##################################################################

    return labels, predictions  # fold-combined labels and fold-combined predictions (where each classifier's prediction organized in a column)
### alias 
combine_bps = combine_baselines
######################################


def extract_name(fn, sep='_', identifiers=[]): 
    from cf_spec import MFEnsemble
    # e.g. wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior
    #      set prefix to 'F' to extract number of factors
    if not identifiers: identifiers = MFEnsemble.segment_ids  # ['F', 'A', 'X', 'CF', 'OPT', 'RE', 'PT', 'S', ]
    # if prefix: assert prefix in identifiers, "Unrecognized segment ID: {id}".format(id=prefix)

    adict = {}
    segments = fn.split(sep)
    for segment in segments: 
        for ID in identifiers:
            if ID in adict: continue # already processed
            if segment.startswith(ID): 
                adict[ID] = segment.replace(ID, '') # extract the value

    return adict

def match(fn, criteria={}, verbose=False): 
    from sklearn.model_selection import ParameterGrid
    if not criteria: return True 

    # e.g. log.S-wmf_F75_A100_XCFuser_S2-posterior.csv
    p = re.compile(r"(?P<method>\w+)\.S-(?P<mf>\w+)-(?P<dtype>\w+)")
    m = p.match(fn)

    # match methods
    # target_stackers = criteria['stacker'] if 'stacker' in criteria else []

    stacker_name = '?'
    if m: 
        fn = m.group('mf')
        stacker_name = m.group('method')

    # if len(target_stackers) > 0: 
    #     # only focus on given stackers ['mean', 'log', 'latent_mean', 'latent_mean_masked', ]
    #     if not stacker_name in target_stackers: 
    #         return False
    ##################################
    # ... stacker matched
    
    lookup = extract_name(fn, identifiers=list(criteria.keys()))

    # normalize data structure 
    for k, v in criteria.items():
        if isinstance(v, (list, tuple, np.ndarray, )):
            pass 
        else: 
            criteria[k] = [v, ]

    isMatched = True
    for i, cr in enumerate(ParameterGrid(criteria)): # each cr is a dictionary

        matched_params = {}
        for k, v in cr.items(): 
            k, v = str(k), str(v)
            if not (k in lookup and lookup[k] == v): 
                isMatched = False
                break
            else: 
                matched_params[k] = v
        if not isMatched: 
            break  # any match

    if isMatched: 
        print('... Found a matched configuration: %s' % matched_params)
    else: 
        if verbose: 
            print("... Could not find a match for %s given %s | lookup: %s" % (fn, criteria, lookup))
    return isMatched

def combine_stackers(domain, rename_method=True, criteria={},  project_path='', analysis_dn='analysis', sep=',', method_params=[], n_cycles=-1, exception_no_data=True, target_stackers=[], paired=True): 
    """
    Combine performance data from all baseline stackers. 
    Combine performance data from all CF stackers. 

    Focus only on top N? 
    
    e.g. top 3 stackers vs top 3 CF stackers

    Params
    ------
    criteria: if specified, only consider files containing any of the strings
        e.g. criteria_str: F100_A100 will match

             wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior

             but not 

             wmf_F50_A100_Xbrier_CFitem_OPTrating_PTprior

    rename_method: if True, convert the short name (e.g. ada) to its canonical name (e.g. AdaBoost) defined by the 
                  mapping in Analysis.modelDict

    Memo
    ----
    1. baseline stacker data layout: 

        fold,id,label,prediction,diversity,method
        0,1,0,0.48845510602573955,0.4014045360034943,ada
        0,2,0,0.49487475657472996,0.4014045360034943,ada
        0,3,0,0.4951036636404069,0.4014045360034943,ada

    2. usage  

        for domain in domains: 
            combine(domain)

    """
    def match0(fn, criteria, settings):  # does the given file name (fn) match a set of criteria and ALSO in the given (algorithmic) settings? 
        # params: criteria, settings
        if isinstance(criteria, str):
            criteria = [criteria, ]
        if isinstance(settings, str): 
            settings = [settings, ]

        # [todo]
        # mvec = []
        # for rules in [criteria, settings, ]:
        #     criteraMatched = False
        #     for rule in rules:  # each set of rules is such that 'any match is a match'
        #         criteraMatched = fn.find(rule) >= 0   # e.g. we may want to consider models: F100A100, F100A10, F10A100, ... 
        criteriaMatched = settingsMatched = True
        if criteria: 
            # print('>>> match | criteira: {c}'.format(c=criteria))
            criteriaMatched = False
            if isinstance(criteria, list):  # any match is a match
                for cr in criteria: 
                    criteriaMatched = fn.find(cr) >= 0   # e.g. we may want to consider models: F100A100, F100A10, F10A100, ... 
                    if criteriaMatched: break 
            else: 
                msg = "Invalid criteria: {0}".format(criteria)
                raise ValueError(msg)
        else: 
            criteriaMatched = True

        if settings: 
            settingsMatched = False
            if isinstance(settings, list):
                for setting in settings: 
                    settingsMatched = fn.find(str(setting)) >= 0   # e.g. we may want to only consider settings 7 and 8 (see cf.System.descriptions)
                    if settingsMatched: break 
            else: 
                msg = "Invalid settings: {0}".format(settings)
                raise ValueError(msg) 
        else: 
            settingsMatched = True           

        return criteriaMatched and settingsMatched
    def parameter_distribution(df, col_params='params', col_index='fold'): # prediction,label,method,fold,params
        assert col_params in df.columns, "Missing 'params' column"
        indices = df[col_index].unique()
        
        params_counter = {}
        for i, dfi in df.groupby([col_index, ]):
            p = dfi[col_params].unique()
            assert len(p) == 1, "Within each cycle, there should be only one 'best' params but got: {val}".format(val=p)
            if not p[0] in params_counter: params_counter[p[0]] = []
            params_counter[p[0]].append(i)

        print('(parameter_distribution) paramters to indices:\n{adict}\n'.format(adict=params_counter))
        best_params = sorted(params_counter, key=lambda k: len(params_counter[k]), reverse=True)[0]
        print('... best params by frequency: {params}'.format(params=best_params))

    import pandas as pd
    columns = ['fold', 'method', 'label', 'prediction'] # params: best params in nth cycle (by comparing parameters in the model selection loop)

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # Given the path to analysis ... 
    ##########################################
    # e.g. rf.S-wmf_F50_A100_XCFitem_S3-posterior.csv

    p_file_base = re.compile(r"(?P<method>\w+)\.S-prediction\.csv")
    p_file_cf = re.compile(r"(?P<method>\w+)\.S-(?P<mf>\w+)-prediction\.csv")
    ps_base = "(?P<method>\w+)\.S-{dtype}\.csv"
    ps_cf = "(?P<method>\w+)\.S-(?P<mf>\w+)-(?P<dtype>{dtype})\.csv"   # .format(dtype='posterior')
    ##########################################

    if not method_params: method_params = list(criteria.keys())

    # pdict = {'bp_stacker': ps_base, 'wmf_stacker': ps_cf}
    ldict = {'pos': 1, 'neg': 0}
    file_types = ['prior', 'posterior', ]
    # for p_type, p_file in pdict.items():  # foreach type of performance data
    
    p_type = 'wmf_stacker'  # baseline stacker (level-1) vs wmf_stacker (level-2)
    fdict = {dt:[] for dt in file_types}   # file dictionary by dataset type
    ndict = {}
    for file_type in file_types: 
        dfs = []
        imin = imax = -1
        n_eff_baselines = 0
        print("##### processing file type: {ft} with criteria: {c}".format(ft=file_type, c=criteria))

        for ith, fp in enumerate(glob("{prefix}/*.csv".format(prefix=analysis_path))):  # foreach stacker performance dataframe
            prefix, fn = os.path.dirname(fp), os.path.basename(fp) 

            ps = ps_cf.format(dtype=file_type) 
            p_file = re.compile(r"%s" % ps)

            m = p_file.match(fn)

            if m:   # 1. file naming pattern matched
                #############################################
                # ... filters
                method = stacker = method0 = m.group('method')
                print('...  fn: {name} | stacker name: {stacker}'.format(name=fn, stacker=stacker))

                # e.g. wmf_F150_A100_Xbrier_CFitem_OPTrating_PTprior_S3 => criteria: F150_A100, setting: S3
                criteraMatched = match(fn, criteria) # match hyperparameters 
                if len(target_stackers) > 0: # then only focus on these stackers 
                    if not stacker in target_stackers: # e.g. ['mean', 'log', 'latent_mean', 'latent_mean_masked', ]
                        print("... stacker {name} is not in the set {set}".format(name=stacker, set=target_stackers))
                        criteraMatched = False 
                
                # if not criteraMatched: 
                #     print('... name: {name} NOT matched'.format(name=fn))
                #############################################
                foldConsistent = True   # the baseline coming from the same (type of) experiment? and therefore has the same fold/index range?
                foldAdjustable = False  

                if criteraMatched:  # 2. criteria matched: e.g. wmf_F150_A100_Xbrier_CFitem_OPTrating_PTprior_S3 => criteria: F150_A100, setting: S3

                    ############################################################################
                    if rename_method: method = stacker = Analysis.modelDict.get(method, method)

                    meta = m.group('mf') # if not p_type.startswith(('bp', 'base')) else ''  # meta: meta layer
                    if file_type.startswith( ('pri', 'post') ):   # basically, we want to include parameter information in order to enable cross referencing later on
                        ### use a more interpretable naming convention
                        #   a. may also want to incorporate the transformation info 
                        if meta: 
                            print('... detected meta parameter string: {meta} | stacker: {method}'.format(meta=meta, method=stacker))
                            # nmap = MFEnsemble.interpret_name(fn) # nmap: name map
                            replacement = {'user': 'classifier', 'item': 'sample'}  # CFuser -> CFmodels i.e. on classifier dimension

                            # use which parameters to name/index the method? 
                            # e.g. {'F': '250', 'A': '100', 'X': 'CFuser', 'XCF': 'user', 'S': '2'}
                            #      and method_params: ['F', 'A']
                            #      => meta: F250_A100
                            meta = MFEnsemble.abridge(meta, identifiers=method_params, replace=replacement, verify=True)  # wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior  -> F100_A100_CFitem
                            method = "{stacker}{delimit}{transformation}".format(stacker=stacker, delimit='+', transformation=meta)
                        else: 
                            pass 
                    div('...... (verify) matched stacker {id}: {m} | data type: {s} | transformation: {tr}'.format(id=n_eff_baselines, m=stacker, s=file_type, tr=meta if meta else 'n/a'))
                    ############################################################################

                    # read the file only if both the file naming pattern and criteria matched
                    df = pd.read_csv(fp, sep=sep, header=0, index_col=False) # error_bad_lines=True
                    assert not df.empty
                    if method0.startswith('mean'): parameter_distribution(df, col_params='params', col_index='fold')

                    df = df[columns]  # prediction,label,method,fold,params
                    # check the method 
                    print("... identified method='{mn}' in file: {fn} | mehtods unique? {methods}".format(mn=method, fn=fn, methods=np.unique(df['method'])))
                    # if not 'method' in df.columns: 
                    df['method'] = method 

                    # check the fold/index range; should be consistent among different stackers 
                    indices = df['fold'].values
                    if n_cycles > 0: 
                        foldConsistent = True if max(indices)+1 == n_cycles else False
                        foldAdjustable = True if max(indices)+1 > n_cycles else False
                    else: 
                        if imin == -1:  # the first one
                            imin, imax = min(indices), max(indices)
                            foldConsistent = True # need to put into a buffer, don't know if consistent yet
                        else: 
                            foldConsistent = True if (imin == min(indices) and imax == max(indices)) else False 

                    if foldConsistent: 
                        dfs.append(df)
                        fdict[file_type].append(fp)
                        n_eff_baselines += 1
                        print("...... adding file: {fn} -> method: {mn} | fold range: {min}~{max} | n_cycles: {nc}".format(fn=fn, mn=method, min=min(indices), max=max(indices), nc=n_cycles))
                    else: 
                        if foldAdjustable: 
                            assert n_cycles > 0
                            # fold_numbers = range(n_cycles)
                            df = df.loc[df['fold'] < n_cycles]  # subset data by cycle/fold

                            dfs.append(df)
                            fdict[file_type].append(fp)
                            n_eff_baselines += 1
                            print("... adding file: {fn} -> method: {mn} | fold range: {min}~{max} | n_cycles: {nc}".format(fn=fn, mn=method, min=min(indices), max=max(indices), nc=n_cycles))
                            print("...... file: {fn} has conflicting fold numbers but can be subset to match n_cycle={nc}".format(fn=fn, nc=n_cycles))
                        else: 
                            print("...... file: {fn} has conflicting fold numbers > fold range({min}, {max})".format(fn=fn, min=imin, max=imax))
           
        ### end foreach file type
        ndict[file_type] = n_eff_baselines

        # assert len(dfs) > 0, 
        err_msg = "No performance data found of data type: {t} | domain: {d}".format(t=file_type, d=domain)
        if len(dfs) == 0: 
            if exception_no_data: 
                raise FileNotFoundError(err_msg)
            else: 
                err_msg += ' > Skipping ...'
                print(err_msg)

                #############
                # ... premature return
                return None
                #############

        dfc = pd.concat(dfs, ignore_index=True)
        print('(combine_stackers) columns: {cols}'.format(cols=dfc.columns.values))  # ['fold' 'method' 'label' 'prediction']
        for col in ['label', 'fold', ]: 
            dfc[col] = dfc[col].astype(int)

        if n_cycles > 0: 
            indices = dfc['fold']
            assert max(indices)+1 == n_cycles, "max(indices): %d != n_cycles: %d" % (max(indices), n_cycles)

        # some sanity check 
        n_pos = dfc.loc[dfc.label==ldict['pos']].shape[0]
        n_neg = dfc.loc[dfc.label==ldict['neg']].shape[0]
        minority_class = 'positive' if n_pos < n_neg else 'negative'
        minority_ratio = min(n_pos, n_neg)/(dfc.shape[0]+0.0)

        div(message="(combine) Found {n} valid stackers on data type {t} | dim: {d} | minority class: {cls}, ratio: {r}  ... (verify)".format(
            n=n_eff_baselines, t=file_type, d=dfc.shape, cls=minority_class, r=minority_ratio), symbol='#', border=1)
        
        # save 
        dset_type = file_type
        fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=p_type, suffix=dset_type)
        print('... saving {f} ... (verify) #'.format(f=fp))
        dfc.to_csv(fp, index = False)  # .C: combined, .S: stacker
    print()

    # files considered 
    div('(combine_stackers) Listing all stacking-specific performance results associated with data types: {alist}'.format( alist=list(fdict.keys())), symbol='#') 
    for i, (k, v) in enumerate(fdict.items()): 
        print("[{dtype}]\n{files} (n={n})\n".format(dtype=k, files=format_list(v, mode='v', sep=', ', padding=0), n=len(v)))
    
    if paired: 
        assert len(set(ndict.values())) == 1, "conflicting number of datasets across file_types: {alist}".format(alist=ndict)
    ##################################################################
    # Output 
    # ------
    #   
    #   wmf_stacker.C-prior.csv
    #   wmf_stacker.C-posterior.csv # columns: fold,method,label,prediction
    #  
    #   <outdated> 
    #      bp_stacker.C-prediction.csv
    #      wmf_stacker.C-prediction.csv
    # 
    #   columns: ['fold', 'method', 'label', 'prediction']
    ##################################################################

    return dfc

def find_best_params(df, metric='fmax', stacker='mean', topn=1, scoring=None, greater_is_better=True, **kargs):
    """
    Find the best params wrt to the stacker (e.g. mean) and the metric (e.g. fmax)

    Memo
    ----
    1. use 'mean' stacker to pick the best params
       then fixed the params, and make comparisons in the following cases: 

           mean vs mean 
           stacker vs stacker 
           stacker vs mean

    2. example outputs: 

        ################################################################################
        (result) Domain: pf2 | rank by params (metric: fmax)
        ################################################################################
        [1]  mean+F100_A100: 0.3214285714285714
        [2]  mean+F120_A100: 0.30303030303030304
        [3]  mean+F75_A100: 0.3018867924528302

        ... stacker fixed: mean with params set: ['F100_A100']
    """

    def method_components(df, sep='+'): # closure: sep
        methods = np.unique(df[col_method].values) 
        # print('(find_best_params) methods: {alist}'.format(alist=methods))  # e.g. ['AdaBoost+F100_A100' 'AdaBoost+F120_A100', ...
        classifiers, hyperparams = set(), set()
        for method in methods: 
            classifier, params, *rest = method.split(sep)
            assert params.startswith('F'), "Invalid params in method: %s" % method
            hyperparams.add(params) 
            classifiers.add(classifier)
        return classifiers, hyperparams

    # kargs
    ################################
    sep = kargs.get('sep', '+')
    domain = kargs.get('domain', '?')
    ################################

    # >>> this only works on *.C-prediction.csv files
    columns = ['fold', 'method', 'label', 'prediction']
    col_method = 'method'
    col_index = 'fold'

    classifiers, hyperparams = method_components(df, sep=sep)
    print('(find_best_params) all possible classifiers: {alist}'.format(alist=classifiers))
    
    assert isinstance(stacker, str) and stacker in classifiers, "Input stacker must be a specific, known classifier but given %s" % stacker
    method_scores = rank_performance(df, metric=metric, stacker=stacker, by='params', topn=topn, scoring=scoring, **kargs) 
    # ... output: [('mean+F120_A100', 0.24372759856630824)]
    methods = [m for m, s in method_scores]

    # print('(find_best_params) best methods (topn: {tn}, ref_stacker: {rs}): {alist}'.format(tn=topn, rs=stacker, alist=methods))
    hyperparams = set()
    for method in methods: 
        classifier, params, *rest = method.split(sep)
        hyperparams.add(params)
    hyperparams = list(hyperparams)
    print('... stacker fixed: {model} with params set: {alist}'.format(model=stacker, alist=hyperparams))
    return hyperparams

def rank_performance_wrt_params(df, metric='fmax', ref_stacker='mean', stacker={}, scoring=None, greater_is_better=True, **kargs): 
    """
    Find the best parameter setting wrt a reference stacker (e.g. mean) and a given metric (e.g. fmax). 
    Fixing this (best) parameter setting, rank performances between 'prior' and 'posterior' datasets while allowing the stacker ensemble to
    very within respsective datasets (specified via 'stacker'). 

    """
    params_set = find_best_params(df, metric=metric, stacker=ref_stacker, topn=1, scoring=scoring, greater_is_better=greater_is_better, **kargs) # **kargs: sep, domain
    assert len(params_set) == 1 
    print('(rank_performance_wrt_params) The best parameter setting wrt to {model}: {p}'.format(model=ref_stacker, p=params_set[0]))
    return rank_performance(df, metric=metric, stacker=stacker, the_params=params_set[0], scoring=scoring, **kargs)

# [helper]
def rank_performance(df, metric='fmax', by='params', stacker=None, the_params=None, topn=-1, scoring=None, greater_is_better=True, **kargs):
    """
    Rank performance and return top N methods
 
    Memo
    ----
    a. select rows that match a string
       <ref> https://davidhamann.de/2017/06/26/pandas-select-elements-by-string/

       df[df['model'].str.match('svm')]    ... startswith 
       df[df['model'].str.contains('F100A100')]
    """
    def method_components(df, sep='+'): # closure: sep
        methods = np.unique(df[col_method].values) 
        classifiers, hyperparams = set(), set()
        for method in methods: 
            classifier, params, *rest = method.split(sep)
            assert params.startswith('F'), "Invalid params in method: %s" % method
            hyperparams.add(params) 
            classifiers.add(classifier)
        return classifiers, hyperparams

    import utils_sys as us 

    # kargs
    ################################
    sep = kargs.get('sep', '+')
    domain = kargs.get('domain', '?')
    file_type = kargs.get('file_type', '?')
    ################################

    # >>> this only works on *.C-prediction.csv files
    columns = ['fold', 'method', 'label', 'prediction']
    col_method = 'method'
    col_index = 'fold'

    # get individual attributes
    classifiers, hyperparams = method_components(df, sep=sep)
    hyperparams = sorted(hyperparams)
    indices = np.unique(df[col_index].values)  # every method should have same number of runs/folds (where 'fold' in CV is more than just an iteration index)

    print("(rank_performance) classifiers: {set}".format(set=classifiers))
    print("... parameter settigs: {set}".format(set=hyperparams))      

    # select corresponding subroutine for the given performance metric
    if metric.lower().startswith('f'):
        scoring_func = common.fmax_score
    else: 
        if scoring is not None: scoring_func = scoring
        assert hasattr(scoring, '__call__')

    method_scores = []
    isSingleStacker = False
    if by.startswith('p'): # focus on parameter combination, agnostic to classification algorithms
        if stacker is not None:  # e.g. focus only on mean
            # find qualified methods first 
            if isinstance(stacker, str): 
                if stacker.startswith('-'):  # '-mean': all but mean
                    stacker = stacker[1:]
                    df_cls = df.loc[~df[col_method].str.contains(stacker)]
                else: 
                    df_cls = df.loc[df[col_method].str.contains(stacker)]
                    isSingleStacker = True  # one stacker + parameters
            else: 
                # note that str also has '__iter__'
                assert hasattr(stacker, '__iter__'), "Invalid stacker input (either a string or a sequence type): %s" % stacker 
                dfs = []
                for s in stacker: 
                    dfs.append( df.loc[df[col_method].str.contains(s)] )  
                df_cls = pd.concat(dfs, ignore_index=True)
        else: 
            df_cls = df  # consider all stackers

        stackers, hyperparams = method_components(df_cls, sep=sep)
        n_stackers = len(stackers)
        if n_stackers == 1: 
            isSingleStacker = True
        table_ex = tabulate(df_cls[df_cls[col_index]==1].head(10), headers='keys', tablefmt='psql')
        print('... Domain: %s, ftype: %s | fixing (best) params? %s | focused stackers (n=%d): %s =>\n%s\n' % (domain, file_type, True if the_params is not None else False, 
            n_stackers, stackers, table_ex))
    
        i = 0
        for _, hp in enumerate(hyperparams): 
            if (the_params is not None) and hp != the_params: 
                # skip this params
                continue  # the_params: None => pass 
            else: 
                i+=1

            df_cls_p = df_cls.loc[df_cls[col_method].str.contains(hp)]  # fixing (classifier, params)
            
            if i == 1: 
                methods_fixed_params = df_cls_p[col_method].unique()
                print('..... params: {p} | method (fixed params, n={n} =?= {nh}): {alist}'.format(p=hp, n=len(methods_fixed_params), nh=n_stackers, alist=methods_fixed_params ))

            scores = []
            for index, dfm in df_cls_p.groupby(col_index): 
                score = scoring_func(dfm.label.values, dfm.prediction.values, pos_label=1) 
                scores.append(score)
            print('...... n(scores): %d' % len(scores))
            score = np.median(scores)
            
            method = df_cls_p[col_method].iloc[0] if isSingleStacker else hp
            method_scores.append((method, score))
    else:  # by specific methods
        methods = np.unique(df[col_method].values) 

        for method in methods: # e.g. SVM+F100_A100
            scores = []
            for index in indices: 
                cond_method = df.method == method 
                dfm = df.loc[ (df.method == method) & (df.fold == index) ]
                score = scoring_func(dfm.label.values, dfm.prediction.values, pos_label=1)
                scores.append(score)
            print('... method: {m} > scores (n={n}) > {s} => mean: {avg}'.format(m=method, n=len(scores), s=scores, avg=np.mean(scores)))
            score = np.median(scores)
            method_scores.append((method, score))
    
    method_ranked = sorted(method_scores, key=lambda x: x[1], reverse=greater_is_better)

    s = us.format_sort_dict(dict(method_ranked), reverse=True, padding=4, title="(result) Domain: {dom} | rank by {cond} (metric: {metric})".format(dom=domain, cond=by, metric='fmax'))
    print('... method ranked:\n%s\n' % s)  # mean+F100A100, F100A100

    if topn <= 0: 
        return method_ranked
    return method_ranked[:topn] # a (sorted) list of 2-tuples: (method, score)

def rank_settings(domain, criteria, topn=1, topn_setting=1, rank_by='params', stacker='mean', method_params=['F', 'A'], n_cycles=5, sep=',', minority_class=1, greater_is_better=True, target_metric='f_score'):
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    ######################################################
    columns = ['fold', 'method', 'label', 'prediction']
    classifiers = Analysis.modelDict
    #  e.g. {'log': 'Logistic', 'qda': 'QDA', 'enet': 'ElasticNet', 'svm': 'SVM', 
    #                'naive': 'NaiveBayes', 'rf': 'RandomForest', 'ada': 'AdaBoost', 'knn': 'kNN', }
    orders = ['Logistic', 'ElasticNet', 'SVM', 'QDA', 'NaiveBayes', 'kNN',  'AdaBoost', 'RandomForest', ] # ['AdaBoostM1','LogitBoost','NaiveBayes','Logistic','SMO','VotedPerceptron','IBk','PART','J48','RandomForest']
    ######################################################
    col_index = 'fold'
    col_method = 'method'

    settings = criteria['S']
    if len(settings) == 1: 
        # job is done, no-op 
        return settings[0]

    target_file_type = 'posterior'
    settings_to_scores = {}
    for setting in settings: 
        combine_stackers(domain, criteria=criteria, method_params=method_params, n_cycles=n_cycles)  # set to [] to bypass 
        ##################################################################
        # Output 
        # ------
        #   
        #   wmf_stacker.C-prior.csv
        #   wmf_stacker.C-posterior.csv # columns: fold,method,label,prediction
        # 
        #   columns: ['fold', 'method', 'label', 'prediction']
        ##################################################################
            
        qualified_models = set()
        for stype in Stacker.types:  # foreach stacker type:  ['bp_stacker', 'wmf_stacker', ] and possible other algorithmic types
            fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=stype, suffix=target_file_type) # e.g. wmf_stacker.C-posterior.csv
            if not os.path.exists(fp): 
                print("Could not find {t}-specific combined dataset: {fn} > Skipping ...".format(t=stype, fn=os.path.basename(fp)))
                continue 
            dfe = pd.read_csv(fp, sep=sep, header=0, index_col=False)  # fold, method, label, prediction

            # stacker: a string (one stacker) or a list (a list of stackers)
            methods = [m for m, _ in rank_performance(dfe, metric='fmax', topn=topn, by=rank_by, stacker=stacker, domain=domain)]  # topn stacker methods on this parituclar date type (e.g prior vs posterior)
            # print('(verify) topn: {n} | prior dim(dfe): {d0} => posterior dim(dfe): {dp}'.format(n=topn, d0=prior_dim, dp=dfe.shape))
            qualified_models.update(methods)
        ##################################################################
        # ... best methods by, say, parameters, are selected

        perfs = []  
        for stype in Stacker.types:  # foreach stacker type:  ['bp_stacker', 'wmf_stacker', ] and possible other algorithmic types
            fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=stype, suffix=target_file_type) # e.g. wmf_stacker.C-posterior.csv
            if not os.path.exists(fp): 
                print("Could not find {t}-specific combined dataset: {fn} > Skipping ...".format(t=stype, fn=os.path.basename(fp)))
                continue 

            dfe = pd.read_csv(fp, sep=sep, header=0, index_col=False)  # fold, method, label, prediction
            prior_dim = dfe.shape
                
            # for each type, select only top N? 
            ################################
            if len(qualified_models) > 0:  # only focus on these methods
                if list(qualified_models)[0] in dfe[col_method].unique(): 
                    dfe = dfe.loc[dfe[col_method].isin(qualified_models)]
                    print('... stacker+params given | topn: {n} | prior dim(dfe): {d0} => posterior dim(dfe): {dp}'.format(n=topn, d0=prior_dim, dp=dfe.shape))
                else: # method_selected contains partial strings (e.g. F100_A100 but without stacker name specified)
                    dfex = []
                    for method in qualified_models: 
                        dfex.append(dfe.loc[dfe[col_method].str.contains(method)]) # isin(qualified_models) may not work because it can contain only partial strings
                    dfe = pd.concat(dfex, ignore_index=True)
                    print('... partial method given | topn: {n} | prior dim(dfe): {d0} => posterior dim(dfe): {dp}'.format(n=topn, d0=prior_dim, dp=dfe.shape))
            #################################

            # >>> need to distinguish between file types 
            assert col_method in dfe.columns
            dfe[col_method] = dfe[col_method] + '+%s' % file_type
            dfe[col_method] = dfe[col_method].astype(str)
            perfs.append(dfe)

        perf_all = pd.concat(perfs, ignore_index=True)  

        ##################################################################
        th = 'fmax'
        df_scores = threshold_by_fmax(perf_all, index_col=col_index)  # adds: ['precision', 'recall', 'f_score']
        # ... columns: 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score'

        col_method = 'model'  # not method 
        col_index = 'seed'
        col_label = 'predict_label'
        ##################################################################

        # df_scores has columns: 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score',
        df_scores = average_and_rename(df_scores, method_params=method_params, sep='+', file_types=[target_file_type, ], by=col_index)

        # best vs best 
        target_metrics = [target_metric, ] # ['f_score', ]
        target_labels = [minority_class, ]
        # for i, metric in enumerate(target_metrics): 
        #     for j, label in enumerate(target_labels):  
                # primary key: seed + model + label
        df = df_scores.loc[df_scores[col_label]==minority_class] # [df_pos, df_neg][j]  # select the dataframe from the list
        methods = df[col_method].unique()
        scores = df[target_metric].values
        print('... n(methods): {n} => {alist}'.format(n=len(methods), alist=methods ))
        print('... n(scores): {n}, avg: {avg}, std: {std}'.format(n=len(scores), avg=np.mean(scores), std=np.std(scores) ))
        settings_to_scores[setting] = np.mean(scores)

    ### end foreach setting 
    settings_sorted = sorted(settings_to_scores, key=settings_to_scores.__getitem__, reverse=greater_is_better)
    s = us.format_sort_dict(dict(settings_to_scores), reverse=True, padding=4, title="(rank_settings) rank by {cond} ".format(cond=target_metric))
    print('... method ranked:\n%s\n' % s)

    # best settings 
    return [k for k in settings_sorted[topn_setting]]

def get_sample_size(domain, sep=',', exception_=False): 
    col_index = 'fold'
    col_method = 'method'

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined
    col_index = 'fold'
    col_method = 'method'

    # load combined stacker performance dataframe
    ###################################################

    # first, find out which methods to include according to 'posterior'
    qualified_models = set()
    target_file_type = 'posterior'
    N = 0
    for stype in Stacker.types:  # foreach stacker type:  ['bp_stacker', 'wmf_stacker', ] and possible other algorithmic types
        if stype.find('wmf') < 0: continue
        fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=stype, suffix=target_file_type) # e.g. wmf_stacker.C-posterior.csv
        if not os.path.exists(fp): 
            print("Could not find {t}-specific combined dataset: {fn} > Skipping ...".format(t=stype, fn=os.path.basename(fp)))
            continue 
        dfe = pd.read_csv(fp, sep=sep, header=0, index_col=False)  # fold, method, label, prediction
        methods = dfe[col_method].unique()
        indices = dfe[col_index].unique()
        max_index = max(indices)

        sizes = {}
        tSizeInconsistent = False
        for i, (entry, df) in enumerate(dfe.groupby([col_index, col_method,])): 
            index, method = entry
            if not index in sizes: sizes[index] = []
            sizes[index].append(df.shape[0])

            if i == 0: 
                N = df.shape[0]
            else: 
                msg = "index: {i}, method: {m} => size: {n} =?= ref size: {nr}".format(i=index, m=method, n=df.shape[0], nr=N)
                # assert N == df.shape[0], "index: {i}, method: {m} => size: {n} =?= ref size: {nr}".format(i=index, m=method, n=df.shape[0], nr=N)
                if exception_: 
                    raise ValueError(msg)
                else: 
                    tSizeInconsistent = True
                    if index == max_index: print('... Inconsistent sample size in method: {m} => sizses: {val}'.format(m=method, val=sizes[index]))
                
    return N

def load_performance_dataframe(domain, policy_threshold, sep=','):
    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # load baseline 
    df_best_base, df_all_base = load_performance_dataframe_baseline(domain, policy_threshold=policy_threshold, sep=sep)
    print('... (verify) loaded BASE performance profile:\n{df}\n'.format(df=df_all_base.head(10)))

    fpath = os.path.join(analysis_path, 'performance_table_threshold_{t}.csv'.format(t=policy_threshold))  # n=n_cycles
    df_scores = pd.read_csv(fpath, sep=sep, header=0, index_col=False)
    print('... (verify) loaded performance profile:\n{df}\n'.format(df=df_scores.head(10)))

    return (df_best_base, df_all_base, df_scores)

def prepare_performance_dataframe(domain, perf_fn='', topn=1, rank_by='params', stacker={}, save=True, project_path='', sep=',', target_params=''):
    """

    Memo
    ----
    1. Treat each regular stacker as a baseline 
       
       columns: 'fold,id,label,prediction,diversity,method' 
       want: fold, method, label, prediction


    """
    def method_components(df, sep='+'): # closure: sep
        methods = np.unique(df[col_method].values) 
        classifiers, hyperparams = set(), set()
        for method in methods: 
            classifier, params, *rest = method.split(sep)
            assert params.startswith('F'), "Invalid params in method: %s" % method
            hyperparams.add(params) 
            classifiers.add(classifier)
        return classifiers, hyperparams

    import utils_sys as us
    from utils_cf import classPrior

    ######################################################
    columns = ['fold', 'method', 'label', 'prediction']
    classifier_map = Analysis.modelDict
    # e.g. {'log': 'Logistic', 'qda': 'QDA', 'enet': 'ElasticNet', 'svm': 'SVM', 
    #                'naive': 'NaiveBayes', 'rf': 'RandomForest', 'ada': 'AdaBoost', 'knn': 'kNN', }
    orders = ['Logistic', 'ElasticNet', 'SVM', 'QDA', 'NaiveBayes', 'kNN',  'AdaBoost', 'RandomForest', ] # ['AdaBoostM1','LogitBoost','NaiveBayes','Logistic','SMO','VotedPerceptron','IBk','PART','J48','RandomForest']
    ######################################################
    col_index = 'fold'
    col_method = 'method'
    col_label = 'label'

    dset_types = ['prior', 'posterior', ]

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # load combined stacker performance dataframe
    ###################################################

    ftype_models = {'prior': None, 'posterior': None}
    qualified_models = {'prior': set(), 'posterior': set()}
    n_qualified = sum(len(v) for v in qualified_models.values())

    # example stacker performance dataframe under analyis path 
    # e.g. mean.S-wmf_F75_A100_XCFuser_S2-posterior.csv

    if perf_fn:  # a specific file is provided (e.g. knn.S-prediction.csv)
        if perf_fn.find('.C') < 0: print('(verify) Input performance dataframe is not a combined dataframe: {fn}'.format(fn=perf_fn))
        fp = os.path.join(analysis_path, perf_fn)
        assert os.path.exists(fp), "Could not find {fn} in analysis directory: {dir}".format(fn=perf_fn, dir=analysis_path)
        perf_all = pd.read_csv(perf_fn)  # perf_n: bp_stacker, wmf_stacker files 
    
    else: # infer from stacker type (which includes simple mean aggregate from cf.run_combiner())

        # first, find out which methods to include according to 'posterior'
        # ftype_models = {'prior': None, 'posterior': None}  # different file types may reference different stackers; None => consider all stackers
        # qualified_models = {'prior': set(), 'posterior': set()}  # methods (stacker+params) actually selected after ranking their performances via rank_performance*
        
        ### search for the best params 
        if n_qualified == 0 and (topn > 0 or rank_by.startswith('param')): 

            # find the 'best' parameter setting in the posterior data
            #########################################################
            target_file_type = 'posterior'
            ref_stacker, ref_metric = 'mean', 'fmax'
            stype = 'wmf_stacker'  # stacker type:  ['bp_stacker', 'wmf_stacker', ] and possible other algorithmic types
            #########################################################

            div("(prepare_performance_dataframe) 1. Find the 'test' parameter setting wrt stacker: {s}, metric: {m}".format(s=ref_stacker, m=ref_metric), symbol='#')
            classifiers = []
            the_params = '?'
            
            fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=stype, suffix=target_file_type) # e.g. wmf_stacker.C-posterior.csv
            assert os.path.exists(fp), "Could not find {t}-specific combined dataset: {fn}".format(t=stype, fn=os.path.basename(fp))
            dfe = pd.read_csv(fp, sep=sep, header=0, index_col=False)  # fold, method, label, prediction
            classifiers, hyperparams = method_components(dfe, sep='+')
            print('... all classifiers: {c}\n... params: {p}'.format(c=classifiers, p=hyperparams))

            # best with respect to a stacker (e.g. mean) and a metric (e.g. fmax)
            if not target_params: 
                params_set = find_best_params(dfe, metric=ref_metric, stacker=ref_stacker, topn=1, greater_is_better=True, domain=domain, file_type=target_file_type)
                the_params = params_set[0]
            else: 
                the_params = target_params
            print('(result) Domain: {dom} > best params: {p}\n'.format(dom=domain, p=the_params))  # F75_A100
            # ... best parameter setting is determined

            # stacker: None, a string, or a dictionary   
            print("\n(prepare_performance_dataframe) 2. Combine classifier name with the parameter setting ...")
            if stacker is None: 
                # use all for both 'prior' and 'posterior'
                for file_type in dset_types: 
                    qualified_models[file_type] = ['{stacker}+{params}'.format(stacker=c, params=the_params) for c in classifiers] 
            else: 
                if isinstance(stacker, str): 
                    assert stacker in classifiers, "Unrecognized stacker: %s (options: %s)" % (stacker, classifiers)
                    for file_type in dset_types: 
                        qualified_models[file_type] = ['{stacker}+{params}'.format(stacker=stacker, params=the_params)]
                else: 
                    assert isinstance(stacker, dict)

                    # stacker is a dictionary mapping dataset type (file type) to a set of stackers
                    assert all([ft in stacker for ft in dset_types]), "Invalid stacker dictionary: %s" % stacker
                    # ... e.g. {'prior': 'mean', 'posterior': '-mean'}

                    for file_type in dset_types: 
                        if stacker[file_type] is None: 
                            qualified_models[file_type] = ['{stacker}+{params}'.format(stacker=c, params=the_params) for c in classifiers]
                        elif isinstance(stacker[file_type], str): 
                            if stacker[file_type].startswith('-'):  # all but 
                                sn = stacker[file_type][1:]
                                assert sn in classifiers, "Unrecognized stacker: %s (options: %s)" % (sn, classifiers)
                                complement = set(classifiers)-set([sn])
                                qualified_models[file_type] = ['{stacker}+{params}'.format(stacker=c, params=the_params) for c in complement]
                            else: 
                                assert stacker[file_type] in classifiers, "Unrecognized stacker: %s (options: %s)" % (stacker[file_type], classifiers)
                                qualified_models[file_type] = ['{stacker}+{params}'.format(stacker=stacker[file_type], params=the_params)]
                        else: 
                            assert isinstance(stacker[file_type], list)
                            # ... e.g. {'prior': [log'], 'posterior': ['mean']}

                            candidates = stacker[file_type]
                            candidates = [classifier_map.get(c, c) for c in candidates]
                            assert len(set(candidates)-set(classifiers))==0, "Specified: {set0}, available: {set1}".format(set0=candidates, set1=classifiers)
                            qualified_models[file_type] = ['{stacker}+{params}'.format(stacker=c, params=the_params) for c in candidates]

            # the following block: the best parameter settings in prior may not be the same as those for posterior 
            # if hasOneTargetStacker: # same for both 'prior' and 'posterior'
            #     for file_type in dset_types:
            #         for stype in Stacker.types:  # foreach stacker type:  ['bp_stacker', 'wmf_stacker', ] and possible other algorithmic types
            #             fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=stype, suffix=file_type) # e.g. wmf_stacker.C-posterior.csv
            #             if not os.path.exists(fp): 
            #                 print("Could not find {t}-specific combined dataset: {fn} > Skipping ...".format(t=stype, fn=os.path.basename(fp)))
            #                 continue 
            #             dfe = pd.read_csv(fp, sep=sep, header=0, index_col=False)  # fold, method, label, prediction

            #             # stacker: None, a string (one stacker) or a list (a list of stackers)
            #             methods = [m for m, _ in rank_performance(dfe, metric='fmax', topn=topn, by=rank_by, stacker=ftype_models[file_type], domain=domain)]  # topn stacker methods on this parituclar date type (e.g prior vs posterior)
            #             # print('(verify) topn: {n} | prior dim(dfe): {d0} => posterior dim(dfe): {dp}'.format(n=topn, d0=prior_dim, dp=dfe.shape))
            #             qualified_models[file_type].update(methods)
            #         div("(performance dataframe) Domain: {d} | topn={tn} | only focus on methods (n={n}):\n{methods}\n ... (verify)".format(d=domain, tn=topn, n=len(qualified_models[file_type]), methods=format_list(list(qualified_models[file_type]), mode='v', sep=', ', padding=0)), symbol='#')
            # else: # each file type is 'potentially' associated with multiple, different stackers (e.g. prior: a set of stackers, posterior: mean)
            #     for file_type in dset_types:
            #         for stype in Stacker.types:  # foreach stacker type:  ['bp_stacker', 'wmf_stacker', ] and possible other algorithmic types
            #             fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=stype, suffix=file_type) # e.g. wmf_stacker.C-posterior.csv
            #             if not os.path.exists(fp): 
            #                 print("Could not find {t}-specific combined dataset: {fn} > Skipping ...".format(t=stype, fn=os.path.basename(fp)))
            #                 continue 
            #             dfe = pd.read_csv(fp, sep=sep, header=0, index_col=False)  # fold, method, label, prediction

            #             # stacker: None, a string (one stacker) or a list (a list of stackers)
            #             print('... stacker <- %s' % ftype_models[file_type])
            #             methods = [m for m, _ in rank_performance_wrt_params(dfe, metric='fmax', ref_stacker='mean', stacker=ftype_models[file_type], 
            #                                                 domain=domain, file_type=file_type)]  # topn stacker methods on this parituclar date type (e.g prior vs posterior)
            #             # print('(verify) topn: {n} | prior dim(dfe): {d0} => posterior dim(dfe): {dp}'.format(n=topn, d0=prior_dim, dp=dfe.shape))
            #             qualified_models[file_type].update(methods)
            #         div("(performance dataframe) Domain: {d} | topn={tn} | ftype: {ft} | focus on methods (n={n}):\n{methods}\n ... (verify)".format(d=domain, tn=topn, ft=file_type, n=len(qualified_models[file_type]), methods=format_list(list(qualified_models[file_type]), mode='v', sep=', ', padding=0)), symbol='#')
          
        ################################################################################################
        # at this point, all methods only reference a single parameter setting
        div('(prepare_performance_dataframe) After performance ranking (domain: {dom}), we found qualified models:\n{adict}\n'.format(dom=domain, adict=qualified_models), symbol='#', border=2)
        # ... e.g. {'prior': ['Logistic+F75_A100'], 'posterior': ['mean+F75_A100']}

        # qualified model examples: 
        # a. mean+F100A100  => classifier+params 
        # b. F100A100       => params only i.e. classifiers are 'marginalized'/averaged
        ################################################################################################
        # stacker: {'prior': '-mean', 'posterior': mean}
        #   =>     {'prior': ['AdaBoost', 'SVM', ...], 'posterior': mean }  # todo
        

        # now collect data based on the selected methods
        dfs = []
        for file_type in dset_types: 
            print("### processing file type: {t}".format(t=file_type))

            for stype in Stacker.types:  # foreach stacker type:  ['bp_stacker', 'wmf_stacker', ] and possible other algorithmic types
                fp = '{path}/{pt}.C-{suffix}.csv'.format(path=analysis_path, pt=stype, suffix=file_type) # e.g. wmf_stacker.C-posterior.csv
                if not os.path.exists(fp): 
                    print("Could not find {t}-specific combined dataset: {fn} > Skipping ...".format(t=stype, fn=os.path.basename(fp)))
                    continue 

                dfe = pd.read_csv(fp, sep=sep, header=0, index_col=False)  # fold, method, label, prediction
                prior_dim = dfe.shape
                print('  + all methods: {alist}'.format(alist=dfe[col_method].unique()))

                # [test]
                for m, dfi in dfe.groupby([col_method, ]):
                    print('    -> method: {m}, dim(dfi): {dim}'.format(m=m, dim=dfi.shape))

                # FILTERING: for each type, select only top N (a. exact match, b. partial match)
                ################################
                target_methods = qualified_models[file_type]
                print('... file_type: {type} => target methods: {alist}'.format(type=file_type, alist=target_methods))
                if len(target_methods) > 0:  # only focus on these methods
                    if len(set(target_methods)-set(dfe[col_method].unique())) == 0: # exact match  
                        dfe = dfe.loc[dfe[col_method].isin(target_methods)]
                        print('...... stacker+params given (ftype: {ft}) | topn: {n} | prior dim(dfe): {d0} => dim(dfe): {dp}'.format(ft=file_type, n=topn, d0=prior_dim, dp=dfe.shape))
                        print('...... attributes(dfe): {cols} | n_runs: {n}'.format(cols=dfe.columns.values, n=len(dfe[col_index].unique())))
                        

                        # [test]
                        # example_method = np.random.choice(target_methods, 1)[0]
                        # example_scores = []
                        # for entry, dfe_i in dfe.groupby([col_index, ]): 
                        #     print('... %s' % dfe_i.loc[ (dfe_i[col_method]==example_method) & (dfe_i[col_label]==1) ].values) # ['fold', 'method', 'label', 'prediction']
                        
                    else: # partial match | method_selected contains partial strings (e.g. F100_A100 but without stacker name specified)
                        dfex = []
                        for method in target_methods: 
                            dfex.append(dfe.loc[dfe[col_method].str.contains(method)]) # isin(target_methods) may not work because it can contain only partial strings
                        dfe = pd.concat(dfex, ignore_index=True)
                        print('... partial method given (ftype: {ft}) | topn: {n} | prior dim(dfe): {d0} => posterior dim(dfe): {dp}'.format(ft=file_type, n=topn, d0=prior_dim, dp=dfe.shape))
                #################################

                # >>> need to distinguish between file types 
                #################################
                assert col_method in dfe.columns
                dfe[col_method] = dfe[col_method] + '+%s' % file_type
                #################################

                # dfe[col_method] = dfe[col_method].astype(str)
                # print('... AFTER updating dfe[col_method] < file_type: {type} | methods: {alist}'.format(type=file_type, alist=dfe[col_method].unique()))
                dfs.append(dfe)

        perf_all = pd.concat(dfs, ignore_index=True)

    # input: perf_all with columns = ['fold', 'method', 'label', 'prediction']
    assert len(set(columns)-set(perf_all.columns)) == 0, "Missing a subset of attributes in combined dataframe: {cols}".format(cols=perf_all.columns.values)
    ############################################################

    # [test]
    if len(qualified_models['posterior']) > 0: 
        n_qualified = sum(len(qualified_models[ft]) for ft in dset_types)
        final_models = perf_all[col_method].unique()
        assert len(final_models) == n_qualified, "size(final_models): %d vs size(qualified_models): %d (best params is not available in a subset of qualified models?)" % (len(final_models), n_qualified)

    n_cycles = len(np.unique(perf_all[col_index]))
    print('(prepare_performance_dataframe) combined performance dataframe > dim(perf_all): {d} | fold_count/n_cycles: {n}  ... (verify)'.format(d=perf_all.shape, n=n_cycles))
    div('(prepare_performance_dataframe) Domain: %s | Final list of methods:\n%s\n' % (domain, final_models), symbol='#', border=2)
    print( tabulate(perf_all.iloc[:4,:4], headers='keys', tablefmt='psql') )
    # ... S'pose that we are comparing stacker (original) vs stacker (transformed)
    #     then list of methods should look sth like: 
    #          ['Logistic+F75_A100+prior' 'Logistic+F75_A100+posterior']
    
    ret = {}
    th = 'fmax'
    ret[th] = df_scores = threshold_by_fmax(perf_all, index_col='fold')  # adds columns: ['precision', 'recall', 'f_score']
    # df_scores has columns: 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 

    # [test] scores from different runs should be different ... (verify)
    ########################################################################
    print('(prepare_performance_dataframe) after getting fmax ...')
    models = df_scores['model'].unique()
    seeds = df_scores['seed'].unique()
    test_model = np.random.choice(models, 1)[0]
    method_scores = {}
    for entry, dfe in df_scores.groupby(['model', 'predict_label',]):
        model, label = entry 
        if model == test_model and label == 1: 
            print('(test) model: {name}, scores (n={n}): {s} | label: {l}'.format(name=model, n=len(seeds), s=dfe['f_score'].values, l=label))
            method_scores[model] = dfe['f_score'].values
            assert len(method_scores[model]) == len(seeds)
    title = '(prepare_performance_dataframe) models vs scores in {n} runs'.format(n=len(seeds))
    print('%s' % format_sort_dict(method_scores, key='len', reverse=True, padding=0, title=title, symbol='#', border=1))
    ########################################################################

    # meta stackers derived from baseline predicitons 
    # meta_stackers(domain, **kargs)  # columns: fold,method,label,prediction

    if save: 
        fpath = os.path.join(analysis_path, 'performance_table_threshold_{t}.csv'.format(t=th))  # n=n_cycles
        df_scores.to_csv(fpath, index=False)
        print('(output) saved performance dataframe ({t}) to:\n{p}\n'.format(t=th, p=fpath))

    # th = 'balanced'
    # ret[th] = df_balanced = threshold_by_balanced_measure(perf_all, index_col='fold')
    # if save: 
    #     fpath = os.path.join(analysis_path, 'performance_table_threshold_{t}.csv'.format(t=th)) # n=n_cycles 
    #     df_balanced.to_csv(fpath, index=False)
    #     print('(output) saved performance dataframe ({t}) to:\n{p}\n'.format(t=th, p=fpath))
    ############################################################
    
    # Output: 2 performance dataframes

    return ret 

def extract_model_params(df, method_params=['F', 'A']):
    def is_posterior(m):
        if m.find('+') < 0: return False 
        for p in method_params: 
            if m.find(p) < 0:
                return False 
        return True
    def n_records(adict):
        n0 = 0
        for i, (k, v) in enumerate(adict.items()): 
            if i == 0: 
                n0 = len(v)
            else:
                assert n0 == len(v), "adict=\n%s\n" % adict
        return n0

    from pandas import DataFrame
    import collections

    # columns(df): 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 
    col_index = 'seed'
    col_label = 'predict_label'
    col_method = 'model'

    methods = df[col_method].unique()
    file_types = ['prior', 'posterior']
    file_types_canonical = {'prior': 'original', 'posterior': 'transformed', }
    fmap = {file_type:[] for file_type in file_types}

    # collect params
    params = []
    for method in methods: 
        if is_posterior(method): 
            # print('(extract) ... method: %s' % method)
            params.append(method.split('+')[1])
    return list(set(params))

def average_and_rename(df, method_params=['F', 'A'], sep='+', **kargs):
    """
    Input dataframe 'df' is a performance dataframe (e.g. generated by prepare_performance_dataframe()) with the following attributes: 
    
    'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', (... other metrics)


    Memo
    ----
    1. layout of performance dataframe: 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 

    
    fixing seed, file_type => 4 entries 

    |  0 |      0 | AdaBoost               |               1 |        0.45 |   0.0997151 | 0.921053 |  0.179949 |
    |  0 |      0 | AdaBoost               |               0 |        0.55 |   0.904523  | 1        |  0.949868 |
    |  0 |      0 | AdaBoost+F100_A100     |               1 |        0.4  |   0.102426  | 1        |  0.185819 |
    |  0 |      0 | AdaBoost+F100_A100     |               0 |        0.55 |   0.904523  | 1        |  0.949868 |


    """
    def is_posterior0(m):
        if m.find('+') < 0: return False 
        for p in method_params: 
            if m.find(p) < 0:
                return False 
        return True
    def is_posterior(m):
        if m.find('post') > 0: return True 
        return False 
    def n_records(adict):
        n0 = 0
        for i, (k, v) in enumerate(adict.items()): 
            if i == 0: 
                n0 = len(v)
            else:
                assert n0 == len(v), "adict=\n%s\n" % adict
        return n0
    def get_params(m): # closure: sep
        if isinstance(m, str):
            classifier, params, file_type, *rest = m.split(sep)
        else: 
            assert hasattr(m, '__iter__')
            classifier, params, file_type, *rest = m[0].split(sep)  
        return params
    def interpret_params(fn, sep='_', identifiers=[]): 
        # e.g. wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior
        #      set prefix to 'F' to extract number of factors
        if not identifiers: identifiers = ['F', 'A', 'X', 'CF', 'OPT', 'RE', 'PT', 'S', ]
        # if prefix: assert prefix in identifiers, "Unrecognized segment ID: {id}".format(id=prefix)

        adict = {}
        segments = fn.split(sep)
        for segment in segments: 
            for ID in identifiers:
                if ID in adict: continue # already processed
                if segment.startswith(ID): 
                    adict[ID] = segment.replace(ID, '') # extract the value
        return adict


    from pandas import DataFrame

    # columns(df): 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', (params)
    col_index = 'seed'
    col_label = 'predict_label'
    col_method = 'model'
    col_params = 'params'

    ########################################
    indices = df[col_index].unique()
    labels = df[col_label].unique()
    methods = df[col_method].unique()
    print('(average_and_rename) n(methods): %d =>\n%s\n' % (len(methods), methods))
    ########################################
    # **kargs
    domain = kargs.get('domain', '?')
    comparison_mode = kargs.get('comparison_mode', '?')
    file_types = kargs.get('file_types', ['prior', 'posterior'])  
    ########################################
    
    
    # get parameter settings
    hyperparams = set()
    classifiers = set()
    the_params = {}
    for method in methods: 
        classifier, params, *rest = method.split(sep)
        hyperparams.add(params)
        file_type = 'posterior'
        if len(rest) > 0: 
            file_type = rest[0]
        classifiers.add(classifier)
    if len(hyperparams) == 1: 
        the_params = interpret_params(list(hyperparams)[0], sep='_', identifiers=[])
        print('... input dataframe has a unique parameter setting: %s' % the_params)
    else: 
        the_params = ''
        msg = "Input dataframe has multiple params (n=%d): %s" % (len(hyperparams), hyperparams)
        raise ValueError(msg)  # don't allow this for now 

    print('...... classifiers: {cls}'.format(cls=classifiers))
    ########################################

    file_types_canonical = {'prior': 'Original', 'posterior': 'Transformed', }

    if len(the_params) > 0: 
        # only 1 set of params, we can specify what it is 
        # file_types_canonical['posterior'] = 'Transformed({nf},{a})'.format(nf=the_params.get('F', '?'), a=the_params.get('A', '?'))
        
        # this could create multiple category in a grouped barplot! 
        # file_types_canonical['posterior'] = 'Transformed(d={nf})'.format(nf=the_params.get('F', '?'))
        pass

    ########################################
    # ... separate 'prior' and 'posterior'
    fmap = {file_type:[] for file_type in file_types}
    for method in methods: 
        if is_posterior(method): 
            fmap['posterior'].append(method)
        else: 
            fmap['prior'].append(method)
    ########################################
    # [test]
    div('(average_and_rename) prior models vs posterior models ...')
    for file_type, methods in fmap.items(): 
        print('... [%s] methods: %s' % (file_type, methods))

    metrics = ['precision', 'recall', 'f_score', ]
    new_cols = [col_index, col_method, col_label, ] + metrics
    dfs = []
    file_type_scores = {ft:[] for ft in file_types}
    # the_params = '?'

    metric_scores = {}
    for cycle, dfi in df.groupby([col_index, ]): # foreach run
        
        for file_type in file_types: # fix dfi
 
            # dfj will contain only methods that match the file type
            dfj = dfi.loc[dfi[col_method].isin(fmap[file_type])] # fmap: file_type -> methods
            assert not dfj.empty
            print('### cycle: %d, ftype: %s => n_methods? %s | n_entries: %d' % (cycle, file_type, len(dfj[col_method].unique()), dfj.shape[0]))

            for label in labels: # fix dfj
                dfk = dfj.loc[dfj[col_label]==label]  # cycle, file_type, label
                assert not dfk.empty
                
                if len(the_params) == 1: # we definitely have an unique parameter setting 
                    # the_params = get_params(models)
                    # adict = interpret_params(the_params)
                    # the_params = the_params['F']  # <<< customize here
                    pass 
                 
                print('... fixing label > how many methods? (n=%d) > %s' % (dfk.shape[0], dfk[col_method].unique())) # e.g. 'mean+F120_A100+prior'
                dfn = DataFrame(columns=new_cols)
                for metric in metrics: 
                    dfn[metric] = [np.mean(dfk[metric]), ]  # has to be a list

                    # [test]
                    if label==1 and metric == 'f_score': 
                        # dfk: fixed file_type, label
                        file_type_scores[file_type].append(np.mean(dfk[metric].values))  # mean across methods
                        print('... (verify) metric={metric} | cycle: {c}, ftype: {ft}, label: {l} | averaging over {n} scores (mode: {mode})'.format(metric=metric, 
                            c=cycle, ft=file_type, l=label, n=dfk[metric].shape[0], mode=comparison_mode)) # e.g. 'mean+F120_A100+prior'
                        if dfk[metric].shape[0] > 1: 
                            print('+++ methods: {methods}'.format(methods=dfk[col_method].unique()))
                            print('+++ scores: {alist}'.format(alist=dfk[metric].values))  # ... ok. has variability in test but not in splice site data, why? 

                    # entry[metric].append( np.mean(dfk[metric]) )
                # cannot average threshold
                dfn[col_label] = label 

                # >>> renaming
                dfn[col_method] = file_types_canonical[file_type]  # either prior or posterior

                try: 
                    dfn[col_params] = the_params['F']  # <<< customize here
                except: 
                    raise ValueError('multiple parameters exist: {alist}'.format(hyperparams))

                dfn[col_index] = cycle
                assert not dfn.empty
                dfs.append(dfn) 
            
        # [test]
        if cycle == 0: # 4 entries
            dfn = pd.concat(dfs, ignore_index=True)
            assert dfn.shape[0] == len(file_types) * len(labels)
            print('(average_and_rename) example new dataframe ...')
            print( tabulate(dfn, headers='keys', tablefmt='psql') )
    print('(average_and_rename) gathered %d records (%d per cycle)' % (len(dfs), len(dfs)/len(indices)) )  # 2 labels * 2 file_types = 4/cycle
    div('> Domain: {dom} | comparison: {mode} | file_type (n={nft}) vs score (n={n}, metric: f_score) ... '.format(dom=domain, mode=comparison_mode, nft=len(file_types), n=len(file_type_scores['posterior']) ), symbol='#')
    for file_type in file_types: 
        scores = file_type_scores[file_type]
        print("... File type: {ft} => scores: {scores}\n...... size: {n} | avg: {avg} | min: {min}, max: {max} | std: {std}".format(ft=file_type, 
            scores=scores, n=len(scores), avg=np.mean(scores), std=np.std(scores), min=np.min(scores), max=np.max(scores) ))

    df2 = pd.concat(dfs, ignore_index=True)

    return df2, file_type_scores

# test routine
def verify_nruns(df_perf, index_col='fold'): 

    models = np.unique(df_perf.method)
    rounds = np.unique(df_perf.fold)

    # [test]
    print('(verify) all models: {m}\n...    runs: {n}'.format(m=models, n=rounds))
    print('...    dim(df_perf): {dim}'.format(dim=df_perf.shape))
    
    n_rounds_models = []
    for model in models: 
        df = df_perf.loc[df_perf.method == model]
        rounds = sorted( df[index_col].unique().tolist() )
        n_rounds_models.append(len(rounds))
        # print('(verify) model: {m} => indices: {i}'.format(m=model, i=rounds))

    return n_rounds_models

def threshold_by_fmax(df_perf, index_col='fold', max_round=None, verbose=False):
    # Input: df_perf with columns = ['fold', 'method', 'label', 'prediction'], where 'fold' may correspond to 'index' or 'round' in other formats
    div(message='Thresholding by fmax ...', symbol="#", border=1)
    models = np.unique(df_perf.method)
    rounds = np.unique(df_perf.fold)

    # [test]
    rounds_foreach_model = verify_nruns(df_perf, index_col=index_col) # sometimes different models may be associated with different number of cycles

    df_scores = [] 
    n_labels = len(np.unique(df_perf.label))
    if max_round is None: max_round = min(rounds_foreach_model)

    model_scores = {model: [] for model in models}  # debug 
    test_models = np.random.choice(models, 1)
    for r in rounds:
        if r >= max_round: break

        print ('### processing experiment round: %s' % r)
        perf_round = df_perf.loc[df_perf[index_col] == r,:]
        
        for model in models:
            # print '####processing model: %s' %model
            prediction_df = perf_round.loc[perf_round.method == model,:]  # groupby
            
            labels = prediction_df.label.values  # tolist()
            predictions = prediction_df.prediction.values # tolist()

            # [test]
            if model in test_models: print('(threshold_by_fmax) predictions (different?): %s' % predictions[:10])
        
            pos_thr,pos_pre,pos_rec,pos_f1 =  precision_recall_fscore(labels, predictions, 1, beta=1.0)
            neg_thr,neg_pre,neg_rec,neg_f1 =  precision_recall_fscore(labels, predictions, 0, beta=1.0)

            # [test]
            model_scores[model].append(pos_f1)
        
            performance_df_pos = pd.DataFrame({'seed':r, 'model':[model], 'predict_label':[1],'threshold': [pos_thr],'precision':[pos_pre],'recall':[pos_rec],'f_score':[pos_f1], })
            performance_df_neg = pd.DataFrame({'seed':r, 'model':[model], 'predict_label':[0],'threshold': [neg_thr],'precision':[neg_pre],'recall':[neg_rec],'f_score':[neg_f1], }) 
        
            df_scores.append(performance_df_pos)
            df_scores.append(performance_df_neg)
    df_scores = pd.concat(df_scores)
    
    print('... dim(df_scores): {dim} <n_rows = n(rounds) * n(models) * n(labels) = {nr} * {nm} * {nl}>\n'.format(dim=df_scores.shape, nr=len(rounds), nm=len(models), nl=n_labels))
    title = '(threshold_by_fmax) models vs scores in {n} runs'.format(n=len(rounds))
    print('%s' % format_sort_dict(model_scores, key='len', reverse=True, padding=0, title=title, symbol='#', border=1))

    print( tabulate(df_scores.head(10), headers='keys', tablefmt='psql') )
    # df_scores.to_csv('performance_table_r{n}_fmax_threshold.csv'.format(n=len(rounds)),index=False)
    
    return df_scores # columns: 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score',

def threshold_by_balanced_measure(df_perf, index_col='fold', max_round=None): # measure: sensitivity (TPR) & specificity (TNR)
    # Input: df_perf with columns = ['fold', 'method', 'label', 'prediction'], where 'fold' may correspond to 'index' or 'round' in other formats
    div(message='Thresholding by balanced sensitivity and specificity ...', symbol="#")
    models = np.unique(df_perf.method)
    rounds = np.unique(df_perf.fold)
    # print('... all models: {m}'.format(m=models)) # all models: ['ada' 'enet' 'knn' 'log' 'mean' 'naive' 'qda' 'rf' 'svm']
    # print('... runs: {n}'.format(n=rounds))

    # [test]
    rounds_foreach_model = verify_nruns(df_perf, index_col=index_col)

    ### Thresholding by balanced specificity/sensitivity
    # output: 
    #   i) performance_table_balanced_threshold.csv
    #   ii) 
    balanced_scores = []
    n_labels = len(np.unique(df_perf.label))
    if max_round is None: max_round = min(rounds_foreach_model)

    for r in rounds:
        if r >= max_round: break

        print('### processing experiment round: %s' %r)
        perf_round = df_perf.loc[df_perf[index_col] == r, :]
        assert not perf_round.empty
        for model in models:
            # print('###### processing model: %s' % model)
            prediction_df = perf_round.loc[perf_round.method == model,:]  # groupby
            assert not prediction_df.empty

            label = prediction_df.label.tolist()
            prediction = prediction_df.prediction.tolist()
        
            thr = get_opt_balanced_acc_thershold(label, prediction, model)

            bin_pred = np.digitize(prediction,[thr])
            pos_pre, pos_rec, pos_f1,_ = precision_recall_fscore_support(label, bin_pred, beta=1.0, labels=None, pos_label=1, average = 'binary') 
            neg_pre, neg_rec, neg_f1,_ = precision_recall_fscore_support(label, bin_pred, beta=1.0, labels=None, pos_label=0, average = 'binary') 
        
            # index_col for this new dataframe will be 'seed'
            performance_df_pos = pd.DataFrame({'seed':r,'predict_label':[1],'threshold': [thr],'precision':[pos_pre],'recall':[pos_rec],'f_score':[pos_f1],'model':[model]})
            performance_df_neg = pd.DataFrame({'seed':r,'predict_label':[0],'threshold': [thr],'precision':[neg_pre],'recall':[neg_rec],'f_score':[neg_f1],'model':[model]})

            balanced_scores.append(performance_df_pos)
            balanced_scores.append(performance_df_neg)

    df_balanced = pd.concat(balanced_scores)

    print('... dim(df_balanced): {dim} <n_rows = n(rounds) * n(models) * n(labels) = {nr} * {nm} * {nl}>\n'.format(dim=df_balanced.shape, nr=len(rounds), nm=len(models), nl=n_labels))
    print( tabulate(df_balanced.head(10), headers='keys', tablefmt='psql') )
    # df_balanced.to_csv('performance_table_balanced_threshold.csv',index=False)

    return df_balanced  # columns: 'seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model'

### performance plot 

def rank_performance_table(df_scores, label=None, metric='f_score', topn=-1, scoring=None, greater_is_better=True):

    # >>> this only works on thresholded performance data: 'performance_table_threshold_{t}.csv'.format(t=th)
    columns = ['seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model']

    if label is not None: 
        df_scores = df_scores.loc[df_scores.predict_label == label, :]

    seeds = sorted(df_scores['seed'].unique())
    models = df_scores['model'].unique()
    
    # primary key: ['seed', 'model'] -> a set of scores
    model_scores = {}
    for index in seeds: 
        df_round = df_scores.loc[df_scores.seed == index]
        for model in models: 
            df_model = df_round.loc[df_round.model]
            if not model in model_scores: model_scores[model] = []  # collect the score per seed/round
            model_scores[model].extend(df_model[metric].values.tolist())
    
    n_points = -1
    for i, (model, scores) in enumerate(model_scores.items()): 
        if i == 0: 
            n_points = len(scores)
        else: 
            assert len(scores) == n_points
    print('... Found {n} data points in each model'.format(n=n_points))

    M = Metrics(model_scores, op=np.mean)
    method_scores = M.sort(by='aggregate')
    
    print('(verify) sorted method vs score (metric={m}):\n{list}\n'.format(m=metric))

    return 

def read(fold, path, dataset='bp', reconstructed_testset=True, policy_iter='cv'):
    """
    
    Params
    ------
    subsampling: 

    """
    import common
    project_path = path

    # global var: project_path
    # if fold == 0: print('(verify) stacking.read() in policy_iter={t}'.format(t=policy_iter))
    if policy_iter.startswith('subs'):  # subsampling 
        # can do subsampling arbitrary number of times
        if dataset == 'bp': 
            return common.shuffle_split(project_path, split_number=2)
        return common.shuffle_split_reconstructed(project_path, method=dataset, split_number=2)
    else:  # policy_iter == cv: cross validation
        # max number of fold is defined by fold count (e.g. 5)
        if dataset == 'bp': 
            return common.read_fold(project_path, fold)
        return common.read_fold_reconstructed(project_path, fold, method=dataset, reconstructed_testset=reconstructed_testset)

def to_rating_matrix(index, domain, dataset, mode='agggregate', bag_count=10): 
    import common
    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # train-dev-test split
    # fold_count=-1 => infer fold count from data
    
    # train_df, train_labels, dev_df, dev_labels, test_df, test_labels = common.read_random_fold(project_path, fold_count=-1, dev_ratio=0.2, shuffle=True)
    train_df, train_labels, test_df, test_labels = read(index, path=project_path, dataset=dataset, reconstructed_testset=True, policy_iter='subsampling')

    # method is global
    if mode.startswith('agg'): 
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        test_df = common.unbag(test_df, bag_count)

    # Convert input data to rating matrix format
    ########################################################################################
    R = train_df.values.T
    # Td = dev_df.values.T
    T = test_df.values.T
    U = train_df.columns.values
    L_train,  L_test = train_labels, test_labels

    # combine train and dev split for model selection (so that each run has its own separate random splits between train and dev)
    # df = pd.concat([train_df, dev_df])
    # labels = np.hstack((train_labels, dev_labels))
    # D_minus = (df, labels)
    # Ix = (df.index, test_df.index)  # i.e. index of the combined training set (train+dev) and index of the test set

    # D = [np.hstack((R, Td)), T, np.hstack((L_train, L_dev)), L_test, U] # + Ix   # in (R, T, L, Lt, U)-format
    # return [np.hstack((R, Td)), T, np.hstack((L_train, L_dev)), L_test, U]
    return (R, T, L_train, L_test, U)

def verify_reconstruction(domain, params, n_runs=10, canonicalize=True, fill=0, predict_func=None, 
        output_path='', dist_func=None, n_sample=None, n_sample_per_run=None, smoothing=True, n_sample_smoothed=100, ext='pdf'):
    """
    
    Params
    ------
    X: (R, T)
    Xh: (Rh, Th)


    Memo
    ----
    1. multiple subplots 

       https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

    """
    def manhatan(x, y):
        return abs(x-y) 

    from scipy.spatial import distance
    from sklearn.metrics import mean_squared_error, brier_score_loss
    from numpy import mean, sqrt, square

    import matplotlib.backends.backend_pdf
    from matplotlib import pyplot as plt
    import seaborn as sns

    # brier score: average sum of square distance 
    #              np.sum((y_true-y_prob)**2) / (N+0.0)

    # distances to the true label: 
    # 1) (R vs L_train), (T vs L_test) 
    # 2) (Rh vs L_train), (Th vs L_test)
    # rms = sqrt(mean(square(Rh-R))) # root mean square 

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # [todo]
    cf_dim_map = Analysis.inv_name_map # classifier -> user, sample -> item
    hyperparams = params['params']   # F100_A100
    cf_dim = cf_dim_map[params['cf_dim']]  # classifier -> user, sample -> item
    setting = params.get('setting', 4)

    # template: wmf_F100_A100_Xbrier_CFuser_OPTrating_PTprior_S10
    tset_id = 'wmf_{params}_Xbrier_CF{cf_dim}_OPTrating_PTprior_S{case}'.format(params=hyperparams, cf_dim=cf_dim, case=setting)  # e.g. 'wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior'
    print('(verify_reconstruction) dataset: {id}'.format(id=tset_id))

    header = ['seed', 'method', 'dist', 'dist_cf'] # 'score_h'?  # method: a base method | score: brier score
    if dist_func is None: dist_func = distance.cityblock  # manhattan, abs
    # dist: distance to label on original probablity (ratings)
    # dist_cf: distance to label on collaboratively filtered (ratings)
    
    Rdict = {h:[] for h in header}; # Rhdict = {h:[] for h in header}
    Tdict = {h:[] for h in header}; # Thdict = {h:[] for h in header}

    dfRx, dfTx = [], []
    methods = []
    
    for index in range(n_runs): 
        R, T, L_train, L_test, U = to_rating_matrix(index, domain=domain, dataset='bp')
        Rh, Th, Lh_train, Lh_test, Uh = to_rating_matrix(index, domain=domain, dataset=tset_id)

        n_users, n_items = R.shape
        assert R.shape == Rh.shape
        if index == 0: 
            print('... (verify) n_classifiers: {nc}, n_data: {nd} | methods: {alist}'.format(nc=n_users, nd=n_items, alist=U))
            methods = U

        ### Training (R) 
        dfRs = []
        for i in range(n_users):
            pv = R[i, :]  # prediction vector from a classifier
            for j in range(n_items):
                Rdict['dist'].append( dist_func(pv[j], L_train[j]) )  # <<< define distance here
            
            pvh = Rh[i, :]
            for j in range(n_items):
                Rdict['dist_cf'].append( dist_func(pvh[j], Lh_train[j]) )
             
            dfR = pd.DataFrame(Rdict, columns=['dist', 'dist_cf', ])
            dfR['method'] = U[i]
            dfR['seed'] = index
            dfRs.append(dfR)
        dfR = pd.concat(dfRs, ignore_index=True) # one round worth of data
        
        ########################################
        # subsampling
        if n_sample_per_run and n_sample_per_run < dfR.shape[0]: dfR = dfR.sample(n=n_sample_per_run, replace=False)
        if smoothing: 
            Nt = n_sample_smoothed//n_runs  # want this many points in total
            dfgs, ns = [], []
            for method, dfg in dfR.groupby(['method', ]): 
                n = dfg.shape[0]
                ng = max(int(n/Nt), 1)  # every ng points are grouped together
                dfe = dfg.groupby(np.arange(len(dfg))//ng).mean()

                print('> method={m} | n_smoothed/run: {Nt} | n: {n} => ng: {ng} => dim(dfe): {de}'.format(m=method, Nt=Nt, n=n, ng=ng, de=dfe.shape)) # method is gone
                print('... dist: {d1} (mean {Ed1}), dist_cf: {d2} (mean {Ed2}'.format(d1=dfe['dist'].values[:10], Ed1=np.mean(dfe['dist'].values), 
                    d2=dfe['dist_cf'].values[:10], Ed2=np.mean(dfe['dist_cf'].values)))
                
                dfe['method'] = method
                dfe['seed'] = index
                
                ns.append( dfe.shape[0] )
                dfgs.append( dfe )
            dfR = pd.concat(dfgs, ignore_index=True) 
            print('... (verify) TRAIN SET > after smoothing, we get {nt} points/method, total: {nT} | n_methods: {nm} | columns: {cols}'.format(nt=ns, 
                nT=dfR.shape[0], nm=len(methods), cols=dfR.columns))
        ########################################
        dfRx.append( dfR )

        ### Testing (T)
 
        dfTs = []
        n_users, n_items = T.shape
        for i in range(n_users):
            pv = T[i, :]  # prediction vector from a classifier
            for j in range(n_items):
                Tdict['dist'].append( dist_func(pv[j], L_test[j]) )  # <<< define distance here
            
            pvh = Th[i, :]
            for j in range(n_items):
                Tdict['dist_cf'].append( dist_func(pvh[j], Lh_test[j]) )
             
            dfT = pd.DataFrame(Tdict, columns=['dist', 'dist_cf', ])
            dfT['method'] = U[i]
            dfT['seed'] = index
            dfTs.append(dfT)
        dfT = pd.concat(dfTs, ignore_index=True) # one round worth of test data
        
        ########################################
        # subsampling
        if n_sample_per_run and n_sample_per_run < dfT.shape[0]: dfT = dfT.sample(n=n_sample_per_run, replace=False) # replace=True if n_sample_per_run > dfT.shape[0] else False
        if smoothing: 
            Nt = n_sample_smoothed//n_runs  # want this many points in total (e.g. 100/10=10)
            dfgs, ns = [], []
            for method, dfg in dfT.groupby(['method', ]): 
                n = dfg.shape[0]
                ng = max(int(n/Nt), 1)  # every ng points are grouped together (e.g. ng = 10000/100=100, take avg every 100 points)
                dfe = dfg.groupby(np.arange(len(dfg))//ng).mean()
                dfe['method'] = method
                ns.append( dfe.shape[0] )
                dfgs.append( dfe )
            dfT = pd.concat(dfgs, ignore_index=True) 
            print('... (verify) TEST SET > after smoothing, we get {nt} points/method, total: {nT} | n_methods: {nm}'.format(nt=ns, 
                nT=dfT.shape[0], nm=len(methods)))
        ########################################
        dfTx.append( dfT )

    ### end foreach round 
    dfR = pd.concat(dfRx, ignore_index=True)   
    dfT = pd.concat(dfTx, ignore_index=True)

    div("Complete {n} runs, dim(dfR):{dim}, dim(dfT):{dimT} ...".format(n=n_runs, dim=dfR.shape, dimT=dfT.shape))

    D = {'train': dfR, 'test': dfT}

    subset = []
    # nrows = 1 
    # ncols = len(methods)

    plt.clf()
    for dtype, df in D.items(): 
        print('... PLOTTING distance | dtype: {d} | dim(df): {dim} ~ ({n}/method)'.format(d=dtype, dim=df.shape, n=df.shape[0]/len(methods)))
        
        # method 1
        # g = sns.FacetGrid(df, col="method",  col_wrap=2, height=4)  # stratify by method
        # g = g.map(plt.scatter, "dist", "dist_cf", edgecolor="w", color="b")

        # method 2
        g = sns.FacetGrid(df, col="method",  col_wrap=2, height=4)  # stratify by method
        g = g.map(sns.regplot, x='dist', y='dist_cf', data=df, 
                    fit_reg=False, marker='o', scatter_kws={"marker": "D", 's': 2}, 
                    color='g') # color='b', 
        # sns.regplot(x='dist', y='dist_cf', data=df, marker='o', color='red', scatter_kws={'s':2})

        # g = sns.FacetGrid(dfR, col="method", hue="tset_id")
        # g.map(plt.scatter, "score", "score", alpha=.7)
        # g.add_legend();

        # xticks=[0.1, 0.3, 0.5, 0.7], yticks=[0.1, 0.3, 0.5, 0.7]
        g = g.set(xlim=(0, 1), ylim=(0, 1)) # scatter_kws={"color":"darkred","alpha":0.3,"s":2}
        # g = g.fig.subplots_adjust(wspace=.05, hspace=.05)

        # diagonal line 
        g.map_dataframe(plt.plot, [0,1], [0,1], 'b:')

        # axis labels
        g.set_axis_labels("Distance to label (before)", "Distance to label (after)")

        # hyperparams, cf_dim
        # output_file = 'proba_calibration_M{method}_P{params}.pdf'.format(method=method, params='{hp}_{cf_dim}'.format(hp=hyperparams, cf_dim=cf_dim))
        output_file = 'proba_dist_D{dtype}_X{params}.{ext}'.format(dtype=dtype, params=tset_id, ext=ext)
        output_path = os.path.join(analysis_path, output_file)
        print('(verify_reconstruction) saved plot to:\n%s\n' % output_path)
        # pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
        # pdf.savefig(g, bbox_inches = 'tight')
        g.savefig(output_path)

        ### density plot
        # g = sns.FacetGrid(df, col="method",  col_wrap=2, height=4)  # stratify by method
        # g = g.map(sns.kdeplot, x='dist', y='dist_cf', data=df, cmap="Reds", shade=True)
        # g = g.set(xlim=(0, 1), ylim=(0, 1)) # scatter_kws={"color":"darkred","alpha":0.3,"s":2}
        # g.set_axis_labels("Distance to label (before)", "Distance to label (after)")

        # output_file = 'proba_dist_density_D{dtype}_X{params}.{ext}'.format(dtype=dtype, params=tset_id, ext='tif')
        # output_path = os.path.join(analysis_path, output_file)
        # g.savefig(output_path)

        # pdf.close()

    # plt.show()
    plt.close()

    return 

def verify_reconstruction_brier(domain, params, n_runs=10, canonicalize=True, fill=0, predict_func=None, 
        output_path='', dist_func=None, n_sample=None, n_sample_per_run=None, ext='pdf'): 
    from sklearn.metrics import mean_squared_error, brier_score_loss
    from numpy import mean, sqrt, square

    import matplotlib.backends.backend_pdf
    from matplotlib import pyplot as plt
    import seaborn as sns

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # [todo]
    cf_dim_map = Analysis.inv_name_map # classifier -> user, sample -> item
    hyperparams = params['params']   # F100_A100
    cf_dim = cf_dim_map[params['cf_dim']]  # classifier -> user, sample -> item
    setting = params.get('setting', 4)

    # template: wmf_F100_A100_Xbrier_CFuser_OPTrating_PTprior_S10
    tset_id = 'wmf_{params}_Xbrier_CF{cf_dim}_OPTrating_PTprior_S{case}'.format(params=hyperparams, cf_dim=cf_dim, case=setting)  # e.g. 'wmf_F100_A100_Xbrier_CFitem_OPTrating_PTprior'
    print('(verify_reconstruction) dataset: {id}'.format(id=tset_id))
    ##########################################

    header = ['seed', 'method', 'score', 'score_cf']
    Rdict = {h:[] for h in header}; # Rhdict = {h:[] for h in header}
    Tdict = {h:[] for h in header}; # Thdict = {h:[] for h in header}
    methods = []  # ['NaiveBayes' 'Logistic' 'SMO' 'AdaBoostM1' 'RandomForest']
    dfRs, dfTs = [], []
    for index in range(n_runs): 
        R, T, L_train, L_test, U = to_rating_matrix(index, domain=domain, dataset='bp')
        Rh, Th, Lh_train, Lh_test, Uh = to_rating_matrix(index, domain=domain, dataset=tset_id)

        # [test]
        n_users, n_items = R.shape
        assert R.shape == Rh.shape
        if index == 0: 
            print('... (verify) n_classifiers: {nc}, n_data: {nd} | methods: {alist}'.format(nc=n_users, nd=n_items, alist=U))
            methods = U
        
        for i in range(n_users): 

            ### Training set
            pv = R[i, :]  # prediction vector from a classifier
            Rdict['score'].append(  brier_score_loss(L_train, pv) )

            pvh = Rh[i, :]
            Rdict['score_cf'].append( brier_score_loss(Lh_train, pvh) )
             
            dfR = pd.DataFrame(Rdict, columns=['score', 'score_cf', ])
            dfR['method'] = U[i]
            dfR['seed'] = index
            dfRs.append(dfR)

            ### Test set
            pv = T[i, :]
            Tdict['score'].append( brier_score_loss(L_test, pv) )
            
            pvh = Th[i, :]
            Tdict['score_cf'].append( brier_score_loss(Lh_test, pvh) ) 

            dfT = pd.DataFrame(Tdict, columns=['score', 'score_cf', ])
            dfT['method'] = U[i]
            dfT['seed'] = index
            dfTs.append(dfT) 
        
    dfR = pd.concat(dfRs, ignore_index=True) # one round worth of data
    dfT = pd.concat(dfTs, ignore_index=True)

    D = {'train': dfR, 'test': dfT}
 
    subset = []
    nrows = 1 
    ncols = len(methods)

    plt.clf()
    for dtype, df in D.items(): 
        print('### processing dataset type: {dtype}, dim(df): {dim}  ...'.format(dtype=dtype, dim=df.shape))  # per method how many points
        g = sns.FacetGrid(df, col="method",  col_wrap=2, height=4)  # stratify by method
        g = g.map(plt.scatter, "score", "score_cf", edgecolor="w", color="g") # s=12

        # g = sns.FacetGrid(dfR, col="method", hue="tset_id")
        # g.map(plt.scatter, "score", "score", alpha=.7)
        # g.add_legend();

        g = g.set(xlim=(0.05, 0.5), ylim=(0.05, 0.5)) # xticks=[0.1, 0.3, 0.5, 0.7], yticks=[0.1, 0.3, 0.5, 0.7]
        g.set_axis_labels("Brier score", "Collaboratively Filetered Brier score")

        # put a diagonal line? 
        g.map_dataframe(plt.plot, [0,1], [0,1], 'b:')

        # hyperparams, cf_dim
        # output_file = 'proba_calibration_M{method}_P{params}.pdf'.format(method=method, params='{hp}_{cf_dim}'.format(hp=hyperparams, cf_dim=cf_dim))
        output_file = 'proba_calib_D{dtype}_X{params}.{ext}'.format(dtype=dtype, params=tset_id, ext=ext)
        output_path = os.path.join(analysis_path, output_file)
        # pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
        # pdf.savefig(g, bbox_inches = 'tight')
        
        g.savefig(output_path)
        print('(verify_reconstruction_brier) saved plot to:\n%s\n' % output_path)
        # pdf.close()

    # plt.show()
    plt.close()

    return

def plot_boxplot(df_scores, df_baseline, domain='', metrics=['precision','recall','f_score'], output_file='performance_boxplot_threshold_fmax.pdf', sorted=True): 
    """
    
    Params
    ------
    df_scores: 
       performance score dataframe associated with: 
           performance_table_{metric}_threshold.csv, where metric = {'fmax', 'balanced', } and possibly other metrics
           columns: 'seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model'

    Memo
    ----
    1. need to get baseline scores (see bestbase()) prior to this call

    """

    # Input: df_scores is the output from 
    #   threshold_by* subroutines 
    #     threshold_by_fmax 
    #     threshold_by_balanced_measure

    # example output files: 
    #    threshold by fmax: 
    #        performance_boxplot_threshold_fmax.pdf
    #    threshold by balanced sensitivity and specificity: 
    #        performance_boxplot_threshold_balanced.pdf

    import matplotlib.backends.backend_pdf
    from matplotlib import pyplot as plt
    import seaborn as sns

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # orders = Analysis.orders
    models = np.unique(df_scores.model) # note: use 'model' instead of 'method' here
    orders = Analysis.order(models, delimit='+')
    print('(verify) classifier ordering: {ord} (n={n})'.format(ord=orders, n=len(models)))

    # sort: todo
    # df_scores format: # columns: 'seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model'

    pos_scores, neg_scores = read_baselines(df_baseline=df_baseline)

    pos_df = df_scores.loc[df_scores.predict_label == 1, :]
    neg_df = df_scores.loc[df_scores.predict_label == 0, :]

    n_labels = len(np.unique(df_scores.predict_label.values))
    n_metrics = len(metrics)

    plt.clf()
    f,axs = plt.subplots(nrows=n_metrics,ncols=n_labels,sharex=True,sharey=False,figsize=(15,18))  # three by two

    for i in range(n_metrics):
        # pos_score, neg_score are from the baseline (base predictors)
        metric, pos_score, neg_score = metrics[i],pos_scores[i],neg_scores[i]
        
        # boxplot for positive class

        # ordering according to performance? 


        sns.boxplot(data=pos_df,x='model',y=metric,ax=axs[i][0],order=orders)  # stratify the data (label='+') by 'model' in the order specified by 'order'
        axs[i][0].tick_params(axis='both',labelsize=15)
        axs[i][0].set_ylabel(metric,fontsize=17)
        axs[i][0].set_xlabel('')
        axs[i][0].axhline(y=pos_score,color='red')
        axs[i][0].set_title('Positive Class', fontsize=17)
        axs[i][0].tick_params(axis='x', labelrotation=90)

        # boxplot for negative class
        sns.boxplot(data=neg_df,x='model',y=metric,ax=axs[i][1],order=orders)
        axs[i][1].tick_params(axis='both',labelsize=15)
        axs[i][1].tick_params(axis='x', labelrotation=90)
        axs[i][1].set_ylabel(metric,fontsize=17)
        axs[i][1].set_xlabel('Model',fontsize=17)
        axs[i][1].axhline(y=neg_score,color='red')
        axs[i][1].set_title('Negative Class', fontsize=17)
        plt.subplots_adjust(wspace=0.3, hspace=0.2)
        
    output_path = os.path.join(analysis_path, output_file)
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
    pdf.savefig(f, bbox_inches = 'tight')
    print('(plot_boxplot) saved plot to:\n%s\n' % output_path)
    pdf.close()
    # plt.show()
    plt.close()

    return

def categorize(df, col_src='model', col_target=['model_group', 'model_class', ], sep='+', mapper={}): 
    """
    Take a dataframe (e.g. df_scores with columns = 'seed' ... 'f_score', ... 'model')
    break a given source column into groups and classes (in preparation for grouped barplots). 

    Use 
    ---
    df <- df_scores: 'seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model'
    source <- 'model'
    target <- ['model_group', 'model_class', ]

    """
    assert col_src in df.columns
   
    # either use a separator or a mapper to make down the source column into its constituents
    # 'kNN+F100_A100_CFclassifier' 'kNN+F100_A100_CFsample' => [kNN, kNN], [F100_A100_CFclassifier, kNN+F100_A100_CFsample]
    default = 'original'
    adict = {col: [] for col in col_target}
    for elem in df[col_src]: 
        # model_group, model_class = elem.split(sep)
        components = elem.split(sep)
        
        # adict['model_group'].append(components[0])
        # adict['model_class'].append(components[1] if len(components) > 1 else default)
        for i, col in enumerate(col_target): 
            adict[col_target[i]].append(components[i] if len(components) > i else default)
    
    for col in col_target: 
        df[col] = adict[col]

    print('(categorize) result:\n{0}\n'.format( tabulate(df.sample(n=10), headers='keys', tablefmt='psql') ) )

    # verfiy
    div('(categorize) verifying group statistics ...')
    cols = ['model', 'predict_label']
    for entry, group in df.groupby(cols):
        # assert len(entry) == len(cols)
        name, label = entry
        if label == 1: 
            # print('... entry: {e}'.format(e=entry))
            sv = group['f_score'].values 
            print("...... model: {g} (n={n}, l='{l}') | mean: {m}, median: {M}, std: {std}".format(g=name, n=len(sv), l='+' if label >= 1 else '-', m=np.mean(sv), M=np.median(sv), std=np.std(sv)))

    return df

def plot_grouped_barplot(df_scores, df_baseline=None, domain='splice_site', orient='v', target_metrics=['f_score', ], target_labels=[1, ], 
        output_file='performance_boxplot_threshold_fmax.pdf', sorted=True, **kargs): 
    """

    Memo
    ----
    1. Annotate bars in barplot 

       https://stackoverflow.com/questions/39444665/add-data-labels-to-seaborn-factor-plot

       https://github.com/mwaskom/seaborn/issues/1582

        plt.figure(figsize=(6, 8))
        splot = sns.barplot(data=df, x = 'sex', y = 'total_bill', ci = None)
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    """
    from matplotlib import pyplot as plt
    import seaborn as sns

    # output format 
    ext = kargs.get('ext', 'pdf')

    ### input df_scores
    #         'domain', 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', ('params')
    
    ##########################################
    col_dataset = 'domain'
    col_index = 'seed'
    col_label = 'predict_label'
    col_method = 'model'
    col_params = 'params'
    ##########################################

    domains = kargs.get('domains', [])  # ordered according to a given criterion (e.g. sample size)
    if len(domains) == 0:
        if col_dataset in df_scores: 
            domains = df_scores[col_dataset].unique()
            assert not domain in domains, "Output domain name: {odn} already exists among those in the input dataframe: {alist}".format(odn=domain, alist=domains)

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain, create_dir=True) # collective directory combining a set of domains, create dir when non-existent
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    models = np.unique(df_scores[col_method]) # note: use 'model' instead of 'method' here
    print('(plot_grouped_barplot) models: %s' % models)

    # orders = Analysis.order(models, delimit='+')
    # print('(verify) classifier ordering: {ord} (n={n})'.format(ord=orders, n=len(models)))

    metrics = ['precision','recall','f_score', ]

    # user-defined
    mmap = metrics_display = {'f_score': 'Fmax', }
    ##########################################
    n_labels = len(np.unique(df_scores.predict_label.values))
    n_metrics = len(metrics)
    metric_map = {metric: i for i, metric in enumerate(metrics)} # metrics to indices

    # stratify by model through its derived attributes: ['model_group', 'model_class', ]
    # df_scores = categorize(df_scores, col_src='model', col_target=['model_group', 'model_class', ], sep='+', mapper={})

    # check availability of the target metrics
    for metric in target_metrics: 
        assert metric in df_scores.columns, "Unrecognized metric: {m}".format(m=metric)

    # >>> plotting
    ##########################################
    # font attributes
    plt.rcParams.update({'font.size': 11})
    plt.rcParams['font.family'] = "sans-serif"

    # plotting style 
    sns.set(style="whitegrid")
    sns.set_color_codes("pastel")
    ##########################################
    bl_score = -1

    # a4_dims = (11.7, 8.27)
    # f, axs = plt.subplots(nrows=len(target_metrics),ncols=len(target_labels), sharex=True, sharey=False, figsize=(20,15))
    tSubplots = True if len(target_metrics) > 1 or len(target_labels) > 1 else False

    plt.clf()

    # print('(verify) df_baseline: {df}'.format(df=df_baseline.head(5))


    # Note: 
    #    df_scores format 
    #        'seed', 'model', 'predict_label', ('threshold'), 'precision', 'recall', 'f_score', 
    domains_to_params = {}
    for d, dfd in df_scores.groupby([col_dataset, ]): 
        param_set = dfd[col_params].unique()
        assert len(param_set) == 1, "More than one parameter setting (n={n}) within the same dataset ({domain})?".format(n=len(param_set), domain=d) 
        domains_to_params[d] = param_set[0]
    print('(verify) domains to parameters:\n{adict}\n'.format(adict=domains_to_params))
    
    col_order = kargs.get('col_order', domains) # ordered
    for i, metric in enumerate(target_metrics): 
        for j, label in enumerate(target_labels):  

            if df_baseline is not None: 
                # sometimes we may want to precompute this score by averaging the best baseline scores from different datasets
                bl_score = df_baseline.loc[df_baseline['label']==label][metric].values[0]  # should resolve to only a single value  
                print('(verify) baseline(label={l}, metric={m}): {s}'.format(l=label, m=metric, s=bl_score))

            # sns.set_color_codes("pastel")

            # primary key: seed + model + label
            df = df_scores.loc[df_scores[col_label]==label] # [df_pos, df_neg][j]  # select the dataframe from the list
            # n_factors = df[col_params].unique()[0]  # this is domain-dependent! 

            print('### processing metric: {metric}, label: {label} | n_data: {n}'.format(metric=metric, label=label, n=df.shape[0]))

            # rename columns | df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
            # df.rename(columns={metric: mmap.get(metric, metric), }, inplace=True)

            ############################################################
            # ... plot configuration: 1D or 2D? 
            col_wrap = kargs.get('col_wrap', 3)
            # is2DSubplot = False if col_wrap is None else True
            ############################################################
            if tSubplots:
                print('... Coming soon!')
            else: 
                if orient.startswith('v'):  # vertical  model (x) by metric score (y)
                    print("(plot_grouped_barplot) orientation: vertical ... ")

                    ########################################
                
                    # sns.catplot(x="model_group", y=metric, 
                    #     hue="model_class", kind="bar", data=df_scores, ci='sd', orient=orient);  # legend_out
 
                    # sns.catplot(x="model_class", y=metric, col="model_group", col_wrap=4, data=df_scores,
                    #                     kind='bar', height=2.5, aspect=.8)

                    # sns.color_palette("husl", 8)

                    ################################################################################
                    # https://seaborn.pydata.org/generated/seaborn.catplot.html
                    g = sns.catplot(data=df,  # df.drop(col_params, axis=1),
                                        x='model', y=metric,   # each model is a bar .. two bars (original vs transformed) ... 
                                        col='domain',    # ... stratified by domain/dataset
                                        col_order=col_order,  # ... order of columns; important for annotations
                                        # hue="model", 
                                            # hue_order=,
                                            col_wrap=col_wrap,    
                                            saturation=.5,
                                            aspect=.6,

                                            # height=8, 
                                            # capsize=.1,  # add "caps" to error bars
                                            # palette: pastel, coolwarm
                                            kind="bar", palette='pastel', ci='sd', orient=orient, legend_out=True)  # palette="Set1"

                    # xticks_label_set = []
                    # for j, dom in enumerate(domains):
                    #     n_factors = domains_to_params[dom]
                    #     xticks_label_set.append(["Original", "Transformed(d={nf})".format(nf=n_factors)])

                    # other options 
                    #    set(ylim=(0, 1)).
                    # g.set_axis_labels("", "Fmax Score").set_xticklabels(["Original", "Transformed"]).set_titles("{col_name} {col_var}").despine(left=True)
                    g.set_axis_labels("", "Fmax Score")
                    g.set_xticklabels(["Original", "Transformed"], fontsize=9)  # fontsize=8, ["Original", "Transformed"], ["Original", "Transformed(d={nf})".format(nf=n_factors)] 
                    g.set_titles("{col_name}") # e.g. pf1, pf2   ... other options: {col_var}  
                    g.despine(left=True)
                    ################################################################################
                    # annotate the bars 
                    # plt.annotate()
                    
                    # a FacetGrid, which stores all of its axes in an axes property as a 2D numpy array; 
                    # Note that if your FacetGrid only has one Axes object in it, you can access it directly with g.ax
                    # for p in g.axes[0].patches:  
                        # g.axes[0].annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                    xpos = kargs.get('xpos', 'center')
                    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
                    offset = {'center': 0, 'right': 1, 'left': -1}

                    # g.ax? easy access to single axes
                    # print('... g.ax? %s, type: %s' % (g.ax, type(g.ax))) # You must use the '.axes' attribute (an array) when there is more than one plot.
                    
                    # >>> use FacetGrid's .axes' attribute (an array) when there is more than one plot.
                    dim_subplot = len(g.axes.shape) 
                    print('... g.axes (1D or 2D)? dim: {dim} | col_wrap: {cw} | g.axes: {ax}'.format(dim=g.axes.shape, cw=col_wrap, ax=g.axes))

                    # e.g. if |domains| = 6 as in splice site datasets 
                    #      then, 
                    #          g.axes (1D or 2D)? dim: (1, 6) when col_wrap is None (i.e. subplots arranged row-wise)
                    #                             dim: (6, )  when col_wrap is 3  (hmmm, counterintuitive!)
                    for j, dom in enumerate(domains):  # domains are ordered! 
                        ax = g.axes[0][j] if dim_subplot == 2 else g.axes[j]

                        n_factors = domains_to_params[dom]
                        print('...... domain: {dom} | params: {p}'.format(dom=dom, p=n_factors))

                        n_patches = len(ax.patches)  # number of bars
                        for i, rect in enumerate(ax.patches):  # g.axes[0] is a numpy array
                            print('... patch #%d => %s' % (i, rect))
                            height = rect.get_height() 
                            text = "d={nf}".format(nf=n_factors)
                            print('... height of plot: %d' % height)  # 0? why? 

                            # annotate only the bar corresponding to the transformed dataset
                            if i == n_patches - 1:  
                                print('... x: {x}, width: {w} > x + width/2: {pos}'.format(x=rect.get_x(), w=rect.get_width(), pos=rect.get_x()+rect.get_width()/2 ))
                                x_offset = rect.get_width()/6
                                y_offset = height/100
                                ax.text(rect.get_x()+rect.get_width()/2+x_offset, height+y_offset, text, fontsize=8, color='grey')  # colors: dimgrey

                        # this doesn't work (all results in the same value on the plot)      
                        # ax.set_xticklabels(["Original", "Transformed(d={nf})".format(nf=n_factors)], fontsize=7)

                    # print('... list attribute g.axes[0]: %s' % dir(g.axes[0]))
                    # height = g.axes[0].get_height()
                    # # print('... FacetGrid has axes? {t} | how many? {n} | height: {h}'.format(t=g.axes, n=len(g.axes), h=g.axes[0].get_height()) )
                    
                    # g.axes[0].annotate('{}'.format(height),
                    #     xy=(g.axes[0].get_x() + g.axes[0].get_width() / 2, height),
                    #     xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    #     textcoords="offset points",  # in both directions
                    #     ha = ha[xpos], va = 'bottom')
                    # g.axes[0].text(g.get_x() + g.get_width()/2, height, the_params , fontsize=9)


                    ################################################################################

                    # ... set legend_out=True in order to use g._legend
                    # ... g is a FacetGrid object (ref: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)
                    
                    # g.despine(left=True)  # https://seaborn.pydata.org/tutorial/aesthetics.html

                    # g.tick_params(labelsize=8, axis='both')

                    # [error] FacetGrid does not have set_xlabel()
                    # g.set_xlabel('Model', fontsize=10)  # only model names, no additional label
                    # g.set_ylabels(metrics_display.get(metric, metric), fontsize=10)

                    # plt.xlabel('Dataset', fontsize=10)  # only model names, no additional label
                    # plt.ylabel(metrics_display.get(metric, metric), fontsize=10) 
                    # plt.tick_params(labelsize=9)  # ... 'FacetGrid' object has no attribute 'tick_params'
                    
                    # g.set_title('Positive Class' if label == 1 else 'Negative Class', fontsize=12)  

                    # g._legend.get_texts()
                    if g._legend is not None: 
                        plt.setp(g._legend.get_texts(), fontsize='8') # for legend text
                        plt.setp(g._legend.get_title(), fontsize='10') # for legend title

                    ########################################
                    
                    # if bl_score > -1: axs.axhline(y=bl_score,color='red')   # each set of baseline scores is a line
                    if bl_score > -1: plt.axhline(bl_score, color='tomato')

                    # plt.title('Performance comparison (before vs after transformation) in {metric}'.format(metric=mmap.get(metric, metric)), fontsize=14)
                else: 
                    # params: 
                    #    palette = sn.color_palette(palette = ["SteelBlue" , "Salmon"]
                    sns.catplot(x=metric, 
                        y="model_group", hue="model_class", kind="bar", data=df_scores, 
                        ci='sd', legend_out = True); 

                    ### multifaceted plot 

                    # adjust the legend 
                    # axs.legend(frameon=False, loc='lower center', ncol=2)
                    axs.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5))
                    # plt.setp(f.get_legend().get_texts(), fontsize='8') # for legend text
                    # plt.setp(f.get_legend().get_title(), fontsize='10') # for legend title  

                    axs.tick_params(labelsize=8, axis='both')
                    axs.set_ylabel('')
                    axs.set_xlabel(metrics_display.get(metric, metric),fontsize=10)
                    if bl_score > -1: axs.axvline(x=bl_score,color='red')   # each set of baseline scores is a line
                    axs.tick_params(rotation=0,axis='y')
                    axs.set_title('Positive Class' if label == 1 else 'Negative Class', fontsize=12)   
       

    if not output_file: output_file = '{domain}_grouped_barplot.{ext}'.format(domain=domain, ext=ext)
    fpath = os.path.join(analysis_path, output_file)
    plt.savefig(fpath, bbox_inches='tight')
    print('(plot_grouped_barplot) domain: %s | saved plot to:\n%s\n' % (domain, fpath))
    # plt.show()
    plt.close()

    return

def plot_barplot(df_scores, df_baseline=None, domain='', orient='v', target_metrics=['f_score', ], target_labels=[1, ], output_file='', sorted=True, ext='pdf'): 
    """

    Memo
    ----
    1. available metrics: ['precision','recall','f_score']

    """

    ### errorbar plot for balanced
    from matplotlib import pyplot as plt
    import seaborn as sns

    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    # orders = Analysis.orders
    models = np.unique(df_scores.model) # note: use 'model' instead of 'method' here
    print('(verify) all models:\n{m}\n'.format(m=models))
    orders = Analysis.order(models, delimit='+')
    print('(verify) classifier ordering: {ord} (n={n})'.format(ord=orders, n=len(models)))

    # pos_scores, neg_scores = read_baselines(df_baseline=df_baseline)
    # basemap = {1: pos_scores, 0: neg_scores}

    ##########################################
    metrics = ['precision','recall','f_score', ]

    # user-defined
    mmap =metrics_display = {'f_score': 'Fmax', }
    ##########################################

    # plotting style 
    sns.set_color_codes("pastel")

    # adjust axis label size 
    plt.rcParams["axes.labelsize"] = 9

    ##########################################
    n_labels = len(np.unique(df_scores.predict_label.values))
    n_metrics = len(metrics)
    metric_map = {metric: i for i, metric in enumerate(metrics)} # metrics to indices

    tSubplots = True if len(target_metrics) > 1 or len(target_labels) > 1 else False

    plt.clf()
    # for i in range(n_metrics):
    #     for j in range(n_labels) 
    bl_score = -1
    # print('(verify) df_baseline: {df}'.format(df=df_baseline.head(5)))
    for i, metric in enumerate(target_metrics): 
        for j, label in enumerate(target_labels):  
            # metric = metrics[metric_map[metric]]  # ith metric

            # dataframe indexed by (model, metric)
            df = df_scores.loc[df_scores.predict_label==label][[metric,'model']] # [df_pos, df_neg][j]  # select the dataframe from the list
            
            # rename columns | df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
            df.rename(columns={metric: mmap.get(metric, metric), }, inplace=True)

            if df_baseline is not None: 
                print('... (verify) baseline cols: {cols}'.format(cols=df_baseline.columns.values))
                # baseline scores: each corresponds to a line
                # bl_score = [pos_scores,neg_scores][j][i]  # jth label, ith metric score

                # ith metric in the order defined in baseline performance dataframe
                #  ... columns: domain,best,cls,precision,recall,f
                # bl_score = df_baseline.loc[df_baseline['label']==label].loc[0, metric] # ~> 'the label [0] is not in the [index]'
                bl_score = df_baseline.loc[df_baseline['label']==label][metric].values[0]  # should resolve to only a single value

                #         bl_score_df = pd.DataFrame(data=[[bl_score,'baseline']],columns=[metric,'model'],index=[0])
                #         df = df.append(bl_score_df,ignore_index=True)

            if tSubplots: 
                f,axs = plt.subplots(nrows=len(target_metrics),ncols=len(target_labels), sharex=True,sharey=False,figsize=(15,18))

                if orient.startswith('v'):  # vertical  model (x) by metric score (y)
                    sns.barplot(data=df, x='model', y=mmap.get(metric, metric), order=orders, ci='sd', ax=axs[i][j])  # ci
                    
                    if bl_score > -1: axs[i][j].axhline(y=bl_score,color='red')   # each set of baseline scores is a line
                    axs[i][j].tick_params(labelsize=15,axis='both')
                    axs[i][j].set_xlabel('')
                    axs[i][j].set_ylabel(mmap.get(metric, metric),fontsize=18)
                    axs[i][j].tick_params(rotation=90,axis='x')
                    axs[i][j].set_title('Positive Class' if label == 1 else 'Negative Class', fontsize=20)

                else: 
                    sns.barplot(data=df, x=mmap.get(metric, metric), y='model', order=orders, ci='sd', ax=axs[i][j])  # ci
                    if bl_score > -1: axs[i][j].axhline(x=bl_score,color='red')   # each set of baseline scores is a line
                
                    axs[i][j].tick_params(labelsize=15,axis='both')
                    axs[i][j].set_ylabel('')
                    axs[i][j].set_xlabel(mmap.get(metric, metric),fontsize=18)
                    axs[i][j].tick_params(rotation=0, axis='y')
                    axs[i][j].set_title('Positive Class' if label == 1 else 'Negative Class', fontsize=20)
            else: 
                # sns.barplot(data=df, x='model', y=metric, order=orders, ci='sd', orient=orient)  # ci
                # # if bl_score > -1: axs.axhline(y=bl_score,color='red')   # each set of baseline scores is a line

                # axs.tick_params(labelsize=15,axis='both')
                # axs.set_xlabel('')
                # axs.set_ylabel(mmap.get(metric, metric),fontsize=18)
                # axs.tick_params(rotation=90,axis='x')
                # axs.set_title('Positive Class' if label == 1 else 'Negative Class', fontsize=20)
                if orient.startswith('v'):  # vertical  model (x) by metric score (y)
                    g = sns.barplot(data=df, x='model', y=mmap.get(metric, metric), order=orders, ci='sd', orient=orient)  # ci         

                    # if bl_score > -1: axs.axhline(y=bl_score,color='red')   # each set of baseline scores is a line
                    # axs.tick_params(labelsize=15,axis='both')
                    # axs.set_xlabel('')
                    # axs.set_ylabel(mmap.get(metric, metric),fontsize=18)
                    # axs.tick_params(rotation=90,axis='x')
                    # axs.set_title('Positive Class' if label == 1 else 'Negative Class', fontsize=20)
                else: 
                    print('(plot_barplot) horizontal ...')
                    g = sns.barplot(data=df, x=mmap.get(metric, metric), y='model', order=orders, ci='sd')  # ci

                    g.set_xlabel(mmap.get(metric, metric), fontsize=12)
                    g.set_ylabel('Model', fontsize=12) 
                    g.tick_params(labelsize=8)

                    # g.set(xlabel=mmap.get(metric, metric), ylabel='Model', fontsize=18)
                    # plt.xlabel(mmap.get(metric, metric), fontsize=11)
                    # plt.ylabel('Model', fontsize=11)
                    plt.title('Performance comparison in {metric}'.format(metric=mmap.get(metric, metric)), fontsize=14)

                    # plt.setp(g._legend.get_texts(), fontsize='8') # for legend text
                    # plt.setp(g._legend.get_title(), fontsize='10') # for legend title
                                           
                    g.set_yticklabels(g.get_yticklabels(), rotation=0)
                    if bl_score > -1: plt.axvline(bl_score, color='tomato')   
            
                    # if bl_score > -1: axs.axvline(x=bl_score, color='red')   # each set of baseline scores is a line
                    # axs.tick_params(labelsize=15,axis='both')

                    # # no effect on the barplot
                    # axs.set_ylabel('Model')
                    # axs.set_xlabel('Fmax',fontsize=18) # mmap.get(metric, metric)
                    # axs.tick_params(rotation=0, axis='y')
                    # axs.set_title('Positive Class' if label == 1 else 'Negative Class', fontsize=20)

    if not output_file: output_file = '{domain}_barplot.{ext}'.format(domain=domain, ext=ext)
    fpath = os.path.join(analysis_path, output_file)
    plt.savefig(fpath, bbox_inches='tight')
    print('(plot_barplot) saved plot to:\n%s\n' % fpath)
    # plt.show()
    plt.close()

    return


##### CD Plots 

def prepare_cd_plot(scores, domain=''): 
    """
    Input
    -----
    scores: a dictionary with keys representing keywords and values holding the outputs from threshold_by* subroutines: 

           threshold_by_fmax 
           threshold_by_balanced_measure
      
    df_scores has the following columns: 
            'seed', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 'model'
    """
    # resolve project path e.g. /Users/<user>/work/data/pf1
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined
    

    scores['fmax'] = df_scores 
    scores['bal'] = df_balanced

    ### CD input of F-max threshold
    fmax_pos = df_scores.loc[df_scores.predict_label == 1,:]
    fmax_neg = df_scores.loc[df_scores.predict_label == 0,:]

    dir_cdplot = os.path.join(analysis_path, 'CD')
    if not os.path.exists(dir_cdplot): os.mkdir(dir_cdplot)

    for metric,pos_score,neg_score in zip(['precision','recall','f_score'],pos_scores,neg_scores):
        fmax_pos_cd = fmax_pos.pivot(index='seed',columns='model',values=metric)
        fmax_pos_cd['baseline'] = pos_score
        del fmax_pos_cd.index.name
        del fmax_pos_cd.columns.name

        fpath = os.path.join(dir_cdplot, '%s_pos_cd_fmax_input.csv' % metric)
        fmax_pos_cd.to_csv(fpath)
        
        fmax_neg_cd = fmax_neg.pivot(index='seed',columns='model',values=metric)
        del fmax_neg_cd.index.name
        del fmax_neg_cd.columns.name
        fmax_neg_cd['baseline'] = neg_score

        fpath = os.path.join(dir_cdplot, '%s_neg_cd_fmax_input.csv' %metric)
        fmax_neg_cd.to_csv(fpath)

    ### CD input of balanced threshold

    bal_pos = balanced_df.loc[balanced_df.predict_label == 1,:]
    bal_neg = balanced_df.loc[balanced_df.predict_label == 0,:]

    for metric,pos_score,neg_score in zip(['precision','recall','f_score'],pos_scores,neg_scores):
        bal_pos_cd = bal_pos.pivot(index='seed',columns='model',values=metric)
        bal_pos_cd['baseline'] = pos_score
        del bal_pos_cd.index.name
        del bal_pos_cd.columns.name
        bal_pos_cd.to_csv('CD/%s_pos_cd_bal_input.csv' %metric)
        
        bal_neg_cd = bal_neg.pivot(index='seed',columns='model',values=metric)
        del bal_neg_cd.index.name
        del bal_neg_cd.columns.name
        bal_neg_cd['baseline'] = neg_score
        bal_neg_cd.to_csv('CD/%s_neg_cd_bal_input.csv' %metric)

    return

def t_baseline(**kargs):

    domains = ['pf2', ]

    ret = {}
    for domain in domains: 
        for th in ['fmax', 'balanced', ]: 
            df_best_base, df_scores = prepare_performance_dataframe_baseline(domain, policy_threshold=th, policy_iter='subsampling', n_runs=30, agg=10, fold_count=5) 
            ret[th] = df_best_base
    
    return

def isClassifierDim(params_dict):  # params_dict: cf_dim, setting, params (hyperparams string e.g. F100_A100)
    
    # find equivalent keywords: 
    cf_dim_map = Analysis.inv_name_map # classifier -> user, sample -> item
    cf_dim = cf_dim_map.get(params_dict['cf_dim'], params_dict['cf_dim'])

    is_cls = cf_dim.startswith(('u', 'cl'))  # user/classifier
    setting = params_dict.get('setting', -1)
    if setting > 0: 
        return is_cls and setting % 2 == 0   # e.g. cases in 2, 4, 8, 10
    return is_cls
def isDataDim(params_dict): 

    # find equivalent keywords: 
    cf_dim_map = Analysis.inv_name_map # classifier -> user, sample -> item
    cf_dim = cf_dim_map.get(params_dict['cf_dim'], params_dict['cf_dim'])

    is_sample = cf_dim.startswith(('i', 'sa', 'da'))   # item/sample/data
    setting = params_dict.get('setting', -1)
    if setting > 0: 
        return is_sample and setting % 2 == 1   # e.g. cases in 1, 3, 5, 7
    return is_sample

def prepare_data(domain, criteria, 
        policy_threshold='fmax', policy_params='homo',
        method_params=['F', 'A'],
        policy_iter='subsampling', n_runs=20, agg=10, fold_count=5, 

        topn=1,  # 1 to select the best
        n_cycles=10, 
        rank_by='params', 
        stacker={},   
        target_params=[]):  
    """
    Input
    -----
    stacker: a string or a dictionary that specifies which classifiers to serve as a basis for the following copmarisons: 
            mean vs mean 
            stacker vs mean 
            stacker vs stacker

    target_params: if specified, only focus on the given parameters instead of performaning search for the best one

    """
    def criteria_to_target_params():
        assert len(target_params) == 0 or (len(target_params) == len(method_params)) 
        if len(target_params) == 0: 
            return ''

        for k, v in zip(method_params, target_params):
            assert v in criteria[k], "target_params must specify a subset choice of criteria ({k}={v} but criteria: {c}".format(k=k, v=v, c=criteria)
        return '_'.join(['{id}{val}'.format(id=k, val=target_params[i]) for i, k in enumerate(method_params)]) 

    # method_params: used for naming the method
    dfc = combine_stackers(domain, criteria=criteria, method_params=method_params, n_cycles=n_cycles, exception_no_data=False)  # set to [] to bypass
    
    target_params = criteria_to_target_params()
    ret = {}
    if dfc is not None: 
        # combine baseline performance dataframes and determine the best base predictor and its score (e.g. fmax)
        # >>> concatenating predictive vectors from across CV folds tend to give higher or more optiministic performance scores? 
        # df_best_base = bestbase(domain, fold_count=5)  # best score and best predictor
        th_type = policy_threshold
        df_best_base, df_scores_base = prepare_performance_dataframe_baseline(domain, policy_threshold=th_type, 
                                           policy_iter=policy_iter, n_runs=n_runs, agg=agg, fold_count=fold_count) 

        ret = prepare_performance_dataframe(domain, topn=topn, rank_by=rank_by, stacker=stacker, target_params=target_params) # params: perf_fn='', topn=-1, file_type='bp_stacker', project_path='', sep=','
        df_scores_stacker = ret[th_type]
        
        # print('(prepare_data) dtype(df_scores_stacker): %s => \n%s\n' % (type(df_scores_stacker), df_scores_stacker.head(5)))
        ret['best_base'] = df_best_base
        ret['scores_base'] = df_scores_base
        ret['scores_stacker'] = df_scores_stacker

    return ret  # keys: 'fmax', 'balanced'


def t_plot(**kargs):
    """

    Memo
    ----
    1. best base
          domain          best  label  precision    recall   f_score
        0    pf1  RandomForest      0   0.904916  0.999931  0.950053
        1    pf1  RandomForest      1   0.160195  0.571053  0.241917
    """
    def performance_summary(file_types=[]):  # closure: domain_scores, domain_params, domains_to_settings, sizes
        if not file_types: file_types = ['prior', 'posterior', ]
        print('(t_plot) Performance Summary ###')
        for i, domain in enumerate(domains): 
            if not domain in domain_scores: 
                # print('... No statistics for domain: {dom}'.format(dom=domain))
                div('[{id}] Domain: {dom} | No statistics ...'.format(id=i+1, dom=domain), symbol='#')
                continue 
            
            the_setting = domains_to_settings[domain]
            file_type_scores = domain_scores[domain]

            # summary taken from average_and_rename()
            div('[{id}] Domain: {dom} | setting: {s}, comparison: {mode} | file_type (n={nft}) vs score (n={n}, metric: f_score) ... '.format(id=i+1, 
                dom=domain, s=the_setting, mode=comparison_mode, nft=len(file_types), n=len(file_type_scores['posterior']) ), symbol='#')
            for file_type in file_types: 
                scores = file_type_scores[file_type]
                if file_type.startswith(('post', 'trans')): 
                    print("... Transformed (params: {params}) | size: {n} | avg: {avg} | min: {min}, max: {max} | std: {std} | sample_size: {N}".format(params=domain_params[domain],
                            n=len(scores), avg=np.mean(scores), std=np.std(scores), min=np.min(scores), max=np.max(scores), N=sizes[domain] ))  # n=len(scores) 
                else: 
                    print("... Original (params: {params})    | size: {n} | avg: {avg} | min: {min}, max: {max} | std: {std} | sample_size: {N}".format(params=domain_params[domain],
                        n=len(scores), avg=np.mean(scores), std=np.std(scores), min=np.min(scores), max=np.max(scores), N=sizes[domain] )) 
        return
    def list_latent_models(): 
        models = kargs.get('latent_model', ['latent_mean_masked', ]) # e.g. ['latent_mean_masked', 'latent_mean', ]
        if isinstance(models, str):
            models = [models, ]
        return models 

    import copy

    tPrepareData = kargs.get('prepare_data', True)
    tDomainSize = kargs.get('domain_size', 'small')
    tTestOnly = kargs.get('test', False)
    n_cycles = kargs.get('n_cycles', 10)
    topn = kargs.get('topn', 1) # set to 1 to choose the best

    # classifiers_prior = kargs.get('classifiers_prior', None)  # '-' all but 
    # classifiers = focused_methods = kargs.get('classifiers', 'mean')

    # cases: 
    # 1) mean vs mean 
    # classifiers = {'prior': 'mean', 'posterior': 'mean'} 
    # 2) stackers vs stackers 
    # classifiers = {'prior': '-mean', 'posterior': '-mean'}
    # 3) all vs all 
    # classifiers = {'prior': None, 'posterior': None}
    # 4) stackers vs mean
    # 

    comparison_mode = kargs.get('comparison_mode', 'customized')
    classifiers = {'prior': None, 'posterior': None}  # None: all | '-mean': all but 'mean' | '-logistic': all but 'logistic'
    target_stackers = ['log', ]

    if comparison_mode in ('mm', 'mean_vs_mean'): 
        classifiers = {'prior': ['mean', ], 'posterior': ['mean', ]} 

        # if the setting is between [60, 69], then we want to focus on latent models
        if Job.settings[0] >= 60 and Job.settings[0] < 70: classifiers = {'prior': list_latent_models(), 'posterior': list_latent_models() }

    if comparison_mode == 'stacker_vs_mean': 
        classifiers = {'prior': ['log', ], 'posterior': ['mean', ]}  

        if Job.settings[0] >= 60 and Job.settings[0] < 70: classifiers['posterior'] = list_latent_models()

    elif comparison_mode == 'stacker_vs_stacker': 
        # classifiers = {'prior': '-mean', 'posterior': '-mean'}
        classifiers = {'prior': ['log', ], 'posterior': ['log', ]}

    elif comparison_mode == 'mean_vs_stacker': 
        # mean vs stacker: hmmm 
        # classifiers = {'prior': 'mean', 'posterior': '-mean'}
        classifiers = {'prior': ['mean', ], 'posterior': ['log', ]}

        if Job.settings[0] >= 60 and Job.settings[0] < 70: classifiers['prior'] = list_latent_models()
    else: 
        # assert comparison_mode == 'mean_vs_mean', "Unrecognized mode: %s" % comparison_mode
        print('Comparison mode: %s => %s' % (comparison_mode, classifiers))
    div('Comparison mode: {mode}, classifiers config: {adict} | settings: {alist}'.format(mode=comparison_mode, adict=classifiers, alist=Job.settings), symbol='#', border=2)
 
    rank_by = kargs.get('rank_by', 'params')
    n_factors = kargs.get('n_factors', [100, ]) # [10, 50, 100, 150, 200, 250]
    alpha = kargs.get('alpha', [100, ])

    # combine baseline and compute baseline scores 
    base_criteria = criteria = {'F': n_factors, 'A': alpha, 'S': Job.settings, } # ['F10_A100', 'F50_A100', 'F100_A100', 'F150_A100', 'F200_A100', ] 
    
    if Job.domains:  # --domains
        # individual domain
        domains = Job.domains
        domain_group = 'generic'
    else: 
        # group domain
        if tDomainSize.startswith('s'): 
            domains = ['pf1', 'pf2', 'pf3' ]  # ['pf1', 'pf2', 'pf3'] | ['thaliana', 'drosophila', 'elegans', 'pacificus', 'remanei', 'sl']
            domain_group = 'protein_function'
        else: 
            domains = ['thaliana', 'drosophila', 'elegans', 'pacificus', 'remanei', ]  # 'sl': similar to 'gl' but with missing values removed
            domain_group = 'splice_site'
            print("(t_plot) Consider domain group {g}: {alist}".format(g=domain_group, alist=domains))
    
    # run test only? then overwrite the default settings above however appropriate
    ###################################
    test_domain = 'pf1'
    if tTestOnly: 
        n_factors = kargs.get('n_factors', [100, ])

        test_setting = kargs.get('test_setting', 6)
        Job.settings = [test_setting, ]
        base_criteria = criteira = {'F': n_factors, 'A': alpha, 'S': Job.settings, }
        # domains = ['pf1',]  # ['pf2', 'pf3', 'thaliana']  
        test_domain = kargs.get('test_domain', 'thaliana')  # <<< configure
        
        domains = [test_domain, ]  # kargs.get('domains', [test_domain, ])    
        domain_group = 'test'  # 'splice_site'

        n_cycles = 5
    ####################################
    # ... defined domains, domain_group, criteria 
    print("(t_plot) domains: {}, group: {}, criteria: {}".format(domains, domain_group, criteria))

    ####################################
    # ... customize 

    method_params = ['F', 'A']  # which file IDs are used to distinguish methods? F: factor, A: alpha | other optoins: 'S': setting
    policy_threshold = 'fmax'   # metric used to compare performances
    policy_iter = 'subsampling'
    policy_params = 'hetero'
    n_runs = 20  # subsampling how many times? 
    agg = 10     # number of bags 
    fold_count = 5    # fold number of the CV used in ensemble generation
    ####################################

    # chooing single algorithmic setting can greatly reduce the complexity of analysis
    domains_to_settings = {}
    for domain in domains: 
        the_setting = criteria['S'][0]
        if len(criteria['S']) > 1: 
            # rank which setting is the best? 
            the_setting = rank_settings(domain, criteria=criteria, topn=1, topn_setting=1, rank_by='params', stacker='mean', n_cycles=n_cycles)  # domain, criteria, topn=1, topn_setting=1, rank_by='params', stacker='mean', method_params=['F', 'A'], n_cycles=5, sep=',', minority_class=1, greater_is_better=True, target_metric='f_score'
            div('The best setting is {s}'.format(s=the_setting))
            # criteria['S'] = [the_setting, ]
        else: 
            div('Domain: {dom} > the target setting is {s}'.format(dom=domain, s=the_setting))
        domains_to_settings[domain] = [the_setting, ]

    ###################################
    # domain to criteria
    domainToCriteria = {
       'pf1': {'F': n_factors, 'A': alpha, 'S': Job.settings, }, 
       'pf2': {'F': n_factors, 'A': alpha, 'S': Job.settings, }, 
       'pf3': {'F': n_factors, 'A': alpha, 'S': Job.settings, }, 

        # in general, parameter settings will be different
       'thaliana': {'F': [200, ], 'A': alpha, 'S': Job.settings, },
       'drosophila': {'F': [50, ], 'A': alpha, 'S': Job.settings, }, 
       'elegans': {'F': [50, ], 'A': alpha, 'S': Job.settings, }, 
       'pacificus': {'F': [50, ], 'A': alpha, 'S': Job.settings, }, 
       'remanei': {'F': [250, ], 'A': alpha, 'S': Job.settings, },  
    }
    
    for domain in domains: 
        print("> active domain: {}:\n... {}\n".format(domain, domainToCriteria[domain]))

    # criteria = ['F100_A100', ]
    # settings = [7, 8, ]
    # domains = ['pf1', ]
    df = DataFrame() # dummy 
    dfs, dfs_base = [], []
    hyperparams_set = set()
    sizes = {}
    domain_scores, domain_params = {}, {}  # result
    for domain in domains: 
        print('###### processing domain: %s' % domain)

        ##########################################
        # ... if mulitiple algorithmic settings are given, then choose the 'best' one to analyze and plot
        criteria = domainToCriteria[domain] if not tTestOnly else base_criteria  # copy.deepcopy(base_criteria)

        # criteria['S'] = domains_to_settings[domain]
        print('... final criteira: {c} | domain: {dom}'.format(c=criteria, dom=domain))
        ##########################################

        Analysis.config(domain=domain)

        ### box plot 
        # fpath = os.path.join(Analysis.analysis_path, 'high.B-{domain}.csv'.format(domain=domain))
        hasData = False
        for th in ['fmax', ]: # 'balanced' 

            ################################################
            df_best_base = df_all_base = df = None
            if tPrepareData: 
                curated_data = \
                    prepare_data(domain, criteria=criteria, policy_threshold=policy_threshold, 
                        method_params=method_params,  # only use these fields to name the method (e.g. ['F', 'A'], where 'F': n_factors, 'A': alpha)
              
                        policy_iter='subsampling',
                        policy_params='hetero',   # homo(geneous), hetero(geneous)
                        
                        n_runs=20, agg=10, fold_count=5, 
                            n_cycles=n_cycles, topn=topn, rank_by=rank_by, stacker=classifiers)
                if len(curated_data) > 0: 
                    (df_best_base, df_all_base, df) = curated_data['best_base'], curated_data['scores_base'], curated_data['scores_stacker']
                    hasData = True
            else: 
                (df_best_base, df_all_base, df) = load_performance_dataframe(domain, policy_threshold=th, sep=',')
                if df is not None and not df.empty: hasData = True
            
            if hasData: 
                assert not df.empty
                ################################################
                #  columns(df): 'seed', 'model', 'predict_label', 'threshold', 'precision', 'recall', 'f_score', 
                
                # extract hyperparameters; has to come before average_and_rename
                hyperparams = extract_model_params(df, method_params=method_params)
                hyperparams_set.update(hyperparams)
                domain_params[domain] = hyperparams

                ################################################
                df, scores = average_and_rename(df, method_params=method_params, by='seed', 
                        domain=domain, comparison_mode=comparison_mode) # **kargs: domain, comparison_mode
                domain_scores[domain] = scores # scores: a dictionary from file_type to scores (n=# runs)
                ################################################

                df['domain'] = domain
                dfs.append(df)

                div('... domain: {d} | params: {alist} | show dataframe ... '.format(d=domain, alist=hyperparams_set))
                print( tabulate(df, headers='keys', tablefmt='psql') )


                print('... best baseline:')
                print( tabulate(df_best_base, headers='keys', tablefmt='psql') )
                # baseline data already has 'domain'
                dfs_base.append(df_best_base)

                # plot_boxplot(df, df_baseline=df_best_base, domain=domain, metrics=['precision','recall','f_score'], 
                #     output_file='performance_table_threshold_{t}.pdf'.format(t=th), sorted=True)

                # metrics=['precision','recall','f_score'], 
               
                # plot_barplot(df, df_baseline=df_best_base, domain=domain, orient='h', 
                #     target_metrics=['f_score', ], target_labels=[1, ], 
                #     output_file='performance_barplot_threshold_{t}.pdf'.format(t=th), sorted=True)
                sizes[domain] = get_sample_size(domain, sep=',')
                print('... finishing domain: %s' % domain)
            else: 
                div('No data found for domain: {dom} with threshold: {metric}'.format(dom=domain, metric=th)) 
    # end foreach domain 

    if len(dfs_base) > 0 and len(dfs) > 0: # at least one of the domain has data

        df_best_base = pd.concat(dfs_base, ignore_index=True)
        df = pd.concat(dfs, ignore_index=True)

        # double check 
        n_domains = len(df_best_base['domain'].unique())
        assert n_domains == len(df['domain'].unique()), "base-level domains and meta-level domains not consistent!"

        # df = consolidate_baseline(df, df_best_base)

        # df: contain all performance scores from all domains
        #     average fmax from all stackers: prior vs posterior 
        the_params = collections.Counter(hyperparams_set).most_common(1)[0][0]
        the_setting = Job.settings[0]
        plot_name = 'performance_barplot_T{t}-P{p}-S{s}.pdf'.format(t=th, p=the_params, s=the_setting)
        if rank_by.startswith('param'):
            plot_name = 'performance_barplot_T{t}-P{p}-S{s}.pdf'.format(t=th, p=comparison_mode, s=the_setting) 
        if tTestOnly: 
            plot_name = 'performance_barplot_D{dom}_T{t}-C{c}-P{p}-S{s}.pdf'.format(dom=test_domain, t=th, c=comparison_mode, p=the_params, s=the_setting) 
            
        # >>> determine column order 
        print('... sample sizes: {adict}'.format(adict=sizes))
        col_order = [dom for dom in sorted(sizes, key=sizes.__getitem__, reverse=False)] # reverse=False => ascending order 
        print('... column order by sample size:\n      {alist}\n'.format(alist=col_order))
        for col_wrap in [3, None]: 
            if col_wrap is None: 
                plot_name = 'performance_barplot_T{t}-P{p}-S{s}-wide.pdf'.format(t=th, p=comparison_mode, s=the_setting) if not tTestOnly else \
                                'performance_barplot_D{dom}_T{t}-C{c}-P{p}-S{s}-wide.pdf'.format(dom=test_domain, t=th, c=comparison_mode, p=the_params, s=the_setting)

            plot_grouped_barplot(df, domain=domain_group, orient='v', 
                target_metrics=['f_score', ], target_labels=[1, ], 
                output_file=plot_name, sorted=True, col_wrap=col_wrap, domains=col_order)

        performance_summary()
    print('(t_plot) Comparison mode: {mode} completed --- #'.format(mode=comparison_mode))
    return

def t_reconstruction(**kargs):
    from sklearn.model_selection import ParameterGrid
    import traceback

    # e.g. wmf_F100_A100_Xbrier_CFuser_OPTrating_PTprior_S8
    param_grid = {'params': ['F10_A100',  'F50_A100', 'F100_A100', 'F150_A100', 'F200_A100', 'F250_A100'], 
                  'cf_dim': ['classifier',  'sample'], 
                  'setting': Job.settings}
    if Job.domain: 
        domains = [Job.domain, ]
    else: 
        domains = ['elegans', ] # 'pf1', 'pf2', 'pf3' | 'thaliana', 'drosophila', 'elegans', 'pacificus', 'remanei', 'sl'

    # param_grid = {'params': ['F100_A100', ], 
    #               'cf_dim': ['classifier',  ],
    #               'setting': [8, ]}
    # domains = ['pf1', ]

    cf_dim_map = Analysis.inv_name_map # classifier -> user, sample -> item
    processed, processed_brier = [], []
    for domain in domains: 
        Analysis.config(domain=domain)
        for i, hyperp in enumerate(ParameterGrid(param_grid)):
            params = {'params': hyperp['params'], 'cf_dim': hyperp['cf_dim'], 'setting': hyperp['setting']}

            # filter impossible combinations: if the params does not agree with classifier dim or data dim, then it's not consistent with the naming convention, skip
            if not isClassifierDim(params) and not isDataDim(params): 
                print('... invalid params: {p}, skipping ...'.format(p=params))
                continue
                
            print("... processing domain: {domain} | params: {params}".format(domain=domain, params=params))
            # verify_reconstruction(domain=domain, params=params, n_runs=10, 
            #     # n_sample_per_run=10000,
            #     smoothing=True, 
            #     n_sample_smoothed=50,   # this many data points per method
            #     output_path='')
            
            # verify_reconstruction_brier(domain=domain, params=params, n_runs=10, output_path='') 
            try: 
                verify_reconstruction(domain=domain, params=params, n_runs=10, 
                    n_sample_per_run=None, # upper limit of total number of particles per subsampling iteration
                    smoothing=True, 
                    n_sample_smoothed=Job.n_sample_smoothed,   # this many data points per method (default 50)
                    output_path='', ext='pdf')
                processed.append(frozenset(params.items()))
            except Exception as e: 
                print('Distance to label: skipping {case}... Error: {e}... \n{trace}>>> \n{err}'.format(case=params, 
                    e=e, trace=traceback.print_exc(), err=traceback.format_exc()))
                continue

            try: 
                verify_reconstruction_brier(domain=domain, params=params, n_runs=10, 
                    output_path='', ext='pdf') 
                processed_brier.append(frozenset(params.items()))
            except Exception as e: 
                print('Brier scores: skipping {case}... Error: {e}... \n{trace}>>> \n{err}'.format(case=params, 
                    e=e, trace=traceback.print_exc(), err=traceback.format_exc()))
                continue
    
    # double check 
    print(); print('#' * 100)
    for i, proc in enumerate(processed): 
        print('[{ord}] params: {p} done ... :)'.format(ord=i+1, p=proc))
    return

def parse_args(): 
    import time, os
    from optparse import OptionParser

    timestamp = now = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time())) # time()
    parentdir = os.path.dirname(os.getcwd())

    # home_dir = os.path.expanduser('~')
    # working_dir_default = '/'.join([home_dir, 'work/data', ])
    working_dir_default = os.path.join(parentdir, 'data')  # e.g. /Users/<user>/work/data
    # ... cf: project_path, e.g. /Users/<user>/work/data/pf1, which includes the domain or dataset name

    parser = OptionParser()
    parser.add_option('-d', '--domain', dest='domain', default='')
    parser.add_option('--domains', dest='domains', default='')
    parser.add_option('-s', '--settings', dest = 'settings', default='6')
    parser.add_option('--n-sample', dest = 'n_sample_smoothed', type='int', default=50)
    parser.add_option('-n', '--nfact', '--n-factors', '--factor', dest='n_factors')
    parser.add_option('-a', '--alpha', dest='alpha')  # can be a comma separated list of values or just a single value
    # parser.add_option('--domains')

    ### wrapper options (wrapping around cf.py)
    # [note] cannot use short-hand switches -a, -s because they are being taken
    # parser.add_option('--nfact', '--n-factors', dest='n_factors')  # can be a comma separated list of values or just a single value; None if not specified
    # parser.add_option('--alpha', dest='alpha')  # can be a comma separated list of values or just a single value
    # a more generalized format 
    # parser.add_option('--params', dest='params')  # comma separated, assignment via equal sign e.g. "n_factors=5, alpha=6, policy='rating'"
    # parser.add_option('--runs', dest='n_runs', type='int', default=10)
    # parser.add_option('--runs-model-select', dest='n_runs_modelselect', type='int', default=1)
    # parser.add_option('--dryrun', action="store_true", dest="dryrun", default=False)

    ### use command line options to configure the job spec and system variables
    (options, args) = parser.parse_args()
    Job.options = vars(options)  # to dictionary object
    Job.args = args

    if options.domain: 
        Job.domains = [options.domain, ]
    elif len(options.domains) > 0: 
        Job.domains = parse_params_list(options.domains, sep=',')

    if len(options.settings) > 0: 
        Job.settings = parse_params_list(options.settings, sep=',', dtype=int) # each case is an integer

    Job.n_sample_smoothed = options.n_sample_smoothed

    div("Command line> domain: {dom}, settings: {cases}, n_sample_smoothed: {n}".format(dom=Job.domain, cases=Job.settings, n=Job.n_sample_smoothed), symbol="#", border=2)

    # resolve project path e.g. /Users/<user>/work/data/pf1
    # Analysis.config(domain=Job.domain)
    # project_path = Analysis.project_path
    # analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    return options, args

def t_raw_data(**kargs):
    """

    Memo
    ----
        > domain: pf1 | size: 3979 | n(labels): 3979 => labels: [0 1]
        > domain: pf2 | size: 3979 | n(labels): 3979 => labels: [0 1]
        > domain: pf3 | size: 3979 | n(labels): 3979 => labels: [0 1]
    """
    from stacking import read
    import collections
    import getpass
    from numpy import linalg as LA

    # debugging 
    np.set_printoptions(precision=3)

    domains = ['pf1', 'pf2', 'pf3'] # 'diabetes_cf'
    for domain in domains: 
        Analysis.config(domain=domain)
        project_path = Analysis.project_path
        analysis_path = Analysis.analysis_path
        ##########################################
        # ... project path and analysis path defined

        # load R, T 
        dev_ratio = 0.2
        max_dev = None
        df, labels = common.get_data(project_path, dataset='bp', fold_count=5) 
        print('> domain: {dom} | size: {size} | n(labels): {nl} => labels: {aseq}'.format(dom=domain, size=df.shape[0], nl=len(labels), aseq=np.unique(labels)))

    return

def t_data(**kargs):
    import sys
    # combine baseline and compute baseline scores 
    # criteria = ['F100_A100', ] # ['F10_A100', 'F50_A100', 'F100_A100', 'F150_A100', 'F200_A100', ]  # ['F120_A100', 'F100_A100', ] # ['F10_A100', 'F150_A100', 'F200A100', 'F50A100']
    # settings = Job.settings

    n_factors = kargs.get('n_factors', [75, 100, 120, ])
    alpha = kargs.get('alpha', [100, ])
    criteria = {'F': n_factors, 'A': alpha, 'S': Job.settings}
    # print('> settings: {0}'.format(Job.settings))
    target_stackers = ['mean', 'log', 'latent_mean', 'latent_mean_masked', ]

    domain = 'pf1'

    topn = 1
    n_cycles = 10
   
    dfc = combine_stackers(domain, criteria=criteria, method_params=['F', 'A'], n_cycles=5, target_stackers=target_stackers) # method_params: used to name the final model
    dfcs = []
    for fold, dfi in dfc.groupby(['fold', ]): 
        dfcs.append(dfi.head(3))
    print('> methods: {alist}'.format(alist=dfc['method'].unique()))
    print( tabulate(pd.concat(dfcs, ignore_index=True), headers='keys', tablefmt='psql') )
    # sys.exit(0)

    classifiers = {'prior': ['log', ], 'posterior': ['log', ]} 
    prepare_performance_dataframe(domain, topn=1, rank_by='params', stacker=classifiers)  # performance_table_threshold_fmax.csv

    sys.exit(0)

    (df_best_base, df_all_base, df) = \
        prepare_data(domain, criteria=criteria, policy_threshold='fmax', 
                        method_params=['F', 'A'],

                        # parameter for baseline methods
                        policy_iter='subsampling',
                        n_runs=20, agg=10, fold_count=5, 
                        n_cycles=n_cycles, topn=topn, rank_by='params', stacker='mean')
    # (df_best_base, df_all_base, df) = load_performance_dataframe(domain, policy_threshold='fmax', sep=',')
    # assert not df.empty

    print('... before averaging ...')
    print( tabulate(df, headers='keys', tablefmt='psql') )

    df = average_and_rename(df, method_params=['F', 'A'], by='seed')
    
    print('... after averging ...')
    df = df.loc[df['predict_label'] == 1]
    print( tabulate(df, headers='keys', tablefmt='psql') )

    return  

def t_match(**kargs):
    method_id = 'wmf_F100_A100_XCFuser_S2'

    criteria = {'F': [100, 200], 'A': [100, ], 'S': [1, 3, 2]}
    tval = match(method_id, criteria=criteria)
    print('> method: {id} | criteria: {c} => matched? {t}'.format(id=method_id, c=criteria, t=tval))
    
    return 

def test(**kargs): 
    np.set_printoptions(precision=3)
    options, args = parse_args()
    # baseline performance dataframe and best baseline method 
    # t_baseline(**kargs)

    # matching by criteria
    # t_match(**kargs)

    # prepare input data 
    # t_raw_data(**kargs)
    # t_data(n_factors=[75, ], )


    # box plot, bar plot, etc. 
    comparison_modes = [ 'mean_vs_mean', 'stacker_vs_mean',  'stacker_vs_stacker', ]  # options: 'mean_vs_mean', 'stacker_vs_mean', 'stacker_vs_stacker' |  'mean_vs_stacker'
    for mode in comparison_modes: 
        t_plot(prepare_data=True, domain_size='small', classifiers='mean', topn=1, n_cycles=10, 
                    comparison_mode=mode, test=False, 

                    # overwrite default configurations for testing; otherwise, critera will be specified through domainsToCritera with pre-determined values
                    # test_domain='pacificus',
                    # test_setting=6,
                    n_factors=[100, ], 
                    alpha=[10, ], 

                    # domains=['thaliana', ], 
                    latent_model='latent_mean')  # # if the setting is between [60, 69], then we want to focus on latent models (e.g. latent_mean_masked, latent_mean)

    # reconstruction quality 
    # t_reconstruction(**kargs)

    return 

def main(**kargs): 

    return

if __name__ == "__main__": 
    # main()
    test()



