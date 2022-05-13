#!/usr/bin/env python
# encoding: utf-8

"""

    Use
    ---
    Do combine first: 
       python combine.py /Users/chiup04/Documents/work/data/diabetes_cf

    python stacking.py <project dir> <method> <stacker>
    python stacking.py /Users/pleiades/Documents/work/data/diabetes_cf standard enet

    Memo
    ----
    1. functions that plot coeffs is available in evaluate module

    Reference
    ----------

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
import os, sys, re
import collections
import numpy as np  # R[line[1]-1, line[2]-1] = line[3]

from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import joblib
from joblib import Parallel, delayed
# from sklearn.externals.joblib import Parallel, delayed # deprecated

from sklearn.linear_model import SGDClassifier, LogisticRegression, Lasso, LassoCV, Perceptron
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

###############################
import utils_classifier as uclf

from nnls import NNLS
import common, utilities
from utils_sys import div

from evaluate import visualizeCoeffs
import getpass

#####################################
class SpecialStacker(object): 
    # mean, CES, etc. that do not necessarily have a functional form
    names = ['CES', ]

    @staticmethod
    def isDefined(name):
        import inspect
        return name in dict(inspect.getmembers(sys.modules['utils_classifier'], inspect.isclass)) 

def mean_stacker(test_df, test_labels=None): 
    return aggregate(test_df, op=np.mean)
def median_stacker(test_df, test_labels=None):
    return aggregate(test_df, op=np.median) 
def aggregate(test_df, test_labels=None, op=np.mean, unbag_=True):
    # import common
    if unbag_:  
        # bag_count is global
        test_df = common.unbag(test_df, bag_count)
    T = test_df.values   # format: n_data by n_features/n_classifiers
    assert hasattr(op, '__call__')
    test_predictions = op(T, axis=1)
    return test_predictions
#####################################

def read(fold, dataset='bp', reconstructed_testset=True, policy_iter='cv', file_type='', path='', mode='train-test-split'):
    """
    
    Params
    ------
    mode: {pairwise, train-test-split, }
        'pairwise' mode is used to conduct pariwise study, comparing predictive strengths between the original and transformed datasets
        'train-test-split'
    
    subsampling: 

    """
    if not path: path = project_path # use global variable by default 

    # global var: project_path
    if fold == 0: print('(verify) stacking.read() in policy_iter={t}'.format(t=policy_iter))
    if policy_iter.startswith('subs'):  # subsampling 
        # can do subsampling arbitrary number of times
        if dataset == 'bp': 
            return common.shuffle_split(path, split_number=2, index=fold)
            
        ##################################################################
        # ... dataset can be a string or a dictionary (mapping from an index/fold to a dataID)
        dataID = dataset[fold] if isinstance(dataset, dict) else dataset
        assert isinstance(dataID, str)

        if mode.startswith('pair'): 
            # note: use shuffle_split_reconstructed if file_type in ['prior', 'posterior', ]
            #       'fold' in this call is really just a dataset index
            
            return common.shuffle_split_reconstructed(path, method=dataID, split_number=2, file_type=file_type, 
                        index=fold, fold_count=fold_count)
        else: 
            # in this case, training and test splits have been made available by the algorithm
            #  ... should also allow for a mixture of datasets of different params

            print('(stacking.read) Reading training and test splits for data index: {index} | dataset ID: {method_id}'.format(index=fold, 
                method_id=dataID))
            return common.read_subsampled(path, index=fold, method=dataID, file_type=file_type) # exception_=False

    else:  # policy_iter == cv: cross validation
        # max number of fold is defined by fold count (e.g. 5)
        if dataset == 'bp': 
            return common.read_fold(path, fold)
        return common.read_fold_reconstructed(path, fold, method=dataset, reconstructed_testset=reconstructed_testset)

def name_stacker(stacker):
    name = 'stacker'
    if not isinstance(stacker, str): # special stacker
        try: 
            name = stacker.__name__ 
        except: 
            name = stacker.__class__.__name__
            # dot separated 
            name = name.split('.')[-1] 
    else: 
        name = stacker
    return name

def stacked_generalization(fold, stacker, **kargs):
    """

    **kargs

        dataset: training set type
        
        analyze_coeffs 
        output_file

    Memo
    ----
    1. train_df: 
       a two-level indexed dataframe, where

       index: (id, label)

       columns: <classifier.bag> 
        e.g. 3 classifiers, 10 bags => n(col) = 30 

             ['NaiveBayes.0' 'NaiveBayes.1' 'NaiveBayes.2' 'NaiveBayes.3'
 'NaiveBayes.4' 'NaiveBayes.5' 'NaiveBayes.6' 'NaiveBayes.7'
 'NaiveBayes.8' 'NaiveBayes.9' 'Logistic.0' 'Logistic.1' 'Logistic.2'
 'Logistic.3' 'Logistic.4' 'Logistic.5' 'Logistic.6' 'Logistic.7'
 'Logistic.8' 'Logistic.9' 'SMO.0' 'SMO.1' 'SMO.2' 'SMO.3' 'SMO.4' 'SMO.5'
 'SMO.6' 'SMO.7' 'SMO.8' 'SMO.9']


       rows: probability scores

    """
    mode_evaluation = kargs.get('mode', 'train-test-split')
    dataset = kargs.get('dataset', 'bp')
    train_df, train_labels, test_df, test_labels = read(fold, 
            dataset=dataset, 
            file_type=kargs.get('file_type', ''),  # values: {'prior', 'posterior', 'train' }
            policy_iter=kargs.get('policy_iter', 'cv'),   # values: {'cv', 'subsampling'}
            reconstructed_testset=True, mode=mode_evaluation) # common.read_fold(project_path, fold)

    # [test] In subsampling mode, data are randomly drawn each time?
    # idx = train_df.index.get_level_values('id').values[:10]

    # method is global
    if method == 'aggregate':
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        test_df = common.unbag(test_df, bag_count)

    # stacker is a regular stacker or a specialized stacker (e.g. mean, CES, etc.)
    test_predictions = None
    if isinstance(stacker, str): 
        div('Running a specialized stacker {name} ... '.format(name=stacker))
        routine = '{name}_stacker'.format(stacker)

        try: 
            test_predictions = globals()[routine](test_df, test_labels)  # or? test_predictions = getattr(stacking, '{name}_stacker'.format(stacker))(test_df, test_labels)
        except: 
            raise NameError("Unknown special stacker: {name}".format(name=stacker))
        
    else: 
        model = stacker.fit(train_df, train_labels)
        test_predictions = model.predict_proba(test_df)[:, 1]  # 1/foldCount worth of data

        if kargs.get('analyze_coeffs', False):  # analyze local coeffs (per CV fold)
            file_name = kargs.get('output_file', '%s_coeffs-f%d' % (name_stacker(stacker), fold)) # kind: stacking algoirthm (e.g. enet, random_forest)
            
            # set exception_ to False, to let through those models that do not have coef_ (e.g. RandomForestClassifier)
            visualizeCoeffs(model, features=train_df.columns.values, file_name=file_name, exception_=False)

    # [note]
    # 1. diversity_score: average_pearson_score
    # 2. test_df.values -> design matrix X: classfiers vs predictive values
    # 3. columns: ['fold', 'id', 'label', 'prediction', 'diversity']
    
    # [test]
    test_ids = test_df.index.get_level_values('id').values
    if mode_evaluation.startswith('pair'): 
        print('(stacked_generalization) Fold: {id} test data ids: {alist}'.format(id=fold, alist=test_ids[:10])) # should be different each time
    else: 
        # the training split and test split were pre-determined rather than sampled on the fly
        n = 10
        col = np.random.choice(test_df.columns, 1)[0]
        y_pred = test_df[col].head(n).values
        y_true = test_df.index.get_level_values('label').values[:n]
        print('(stacked_generalization) Fold: {id} | n(train): {ntr}, n(test): {nt}'.format(id=fold, ntr=train_df.shape[0], nt=test_df.shape[0]))
        # print('... predictors: {alist} | choose {col}'.format(alist=test_df.columns.values, col=col))
        print('... y_pred (n={n}): {scores}'.format(n=n, scores=y_pred))
        print('... y_label(n={n}): {labels}'.format(n=n, labels=y_true))

        # diversity score 
        # ds0 = common.diversity_score(test_df.values)  # cols(test_df): classifiers 
        
    # diversity score uses average_pearson_score() by default
    return DataFrame({'fold': fold, 'id': test_ids, 'label': test_labels, 'prediction': test_predictions, 'diversity': common.diversity_score(test_df.values)})

def evaluate(fold, stacker, **kargs): 
    """
    A stacker based on collaborative filtering.

    Memo
    ----
    1. related modules: cf, utils_cf

    2. Typical use: 

        cf_stacker(stacker, fold=fold, Rh=Rh, Th=Th, p_threshold=p_th, logger=logger)
           : use 'fold' to access the original R (rating matrix) obtained from BPs 

    """
    def info(mse):  # result for paper
        result_mse = '(stacking.evaluate) Mean Square Error (Rh|Th vs R|T): {mse} | CV fold: {fold}, method: {method}'.format(mse=mse, fold=fold, method=kargs.get('method', 'cf_stacker')) 
        print(result_mse)  # or use logger.debug(msg) ~> stderr or stdout 
        logger = kargs.get('logger', None) 
        if logger is not None: 
            logger.info(result_mse)
        return

    import utils_cf as uc

    # global: method, bag_count
    fold = kargs.get('fold', -1)
    p_th = kargs.get('p_threshold', 0.5)
    R = T = None

    # train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold)
    unbag = True if method.startswith('aggr') else False
    R, T, train_labels, test_labels, U = uc.to_rating_matrix2(fold, p_threshold=p_th, 
            missing_value=kargs.get('missing_value', 0), verbose=True, unbag=unbag, bag_count=bag_count)
    n_train, n_test = len(train_labels), len(test_labels)

    # reconstructed R and T 
    Rh = kargs['Rh'] if 'Rh' in kargs else np.zeros(R.shape)
    Th = kargs['Th'] if 'Th' in kargs else np.zeros(T.shape)

    # test 
    assert R.shape == Rh.shape and T.shape == Th.shape
    assert R.T.shape[0] == len(train_labels)

    # MSE between Rh|Th and R|T 
    mse = mean_squared_error(np.hstack((R, T)).ravel(), np.hstack((R, T)).ravel())
    info(mse)  # print + log  ... result 1 

    model = stacker.fit(Rh.T, train_labels)
    test_predictions = model.predict_proba(test_labels)[:, 1]  # 1/foldCount worth of data

    if kargs.get('analyze_coeffs', False):  # analyze local coeffs (per CV fold)
        file_name = kargs.get('coeffs_%s-f%d' % (kargs.get('kind', 'stacker'), fold)) # kind: stacking algoirthm (e.g. enet, random_forest)
        # set exception_ to False, to let through those models that do not have coef_ (e.g. RandomForestClassifier)
        visualizeCoeffs(model, features=U, file_name=file_name, exception_=False)  # result 2

    # [note]
    # diversity_score: average_pearson_score
    # test_df.values -> design matrix X: predictive values (rows) vs classfiers (columns)
    return DataFrame({'fold': fold, 'id': range(n_train, n_train+n_test) , 
        'label': test_labels, 'prediction': test_predictions, 'diversity': common.diversity_score(Th.T)}) # test_df ~ T.T

# [refactor]
def base_models(fold, **kargs):
    train_df, train_labels, test_df, test_labels = common.read_fold(project_path, fold)

    if method == 'aggregate':
        train_df = common.unbag(train_df, bag_count) # mean aggregates (average over all bags)
        test_df = common.unbag(test_df, bag_count)

    return

def choose_classifier(name, **kargs): 
    return uclf.choose_classifier(name, **kargs)

# use non-negative least squares for regression
def customize_stacker(stacker=None, **kargs): 
    """

    Ref
    ---
    1. regularization path
       plot_logistic_path

    2. Gradient Boosting 
       subsample: The fraction of samples to be used for fitting the individual base learners. 
                  If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. 
                  Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.

    Memo
    ----
    1. Perceptron 
       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    2. Logistic Regression 
       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    3. SGD Classifier 
       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        - used for large datasets 

    4. Gradient Boosting 
       https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

    5. Bayesian Ridge Regression 
       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge

       
       The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
    
    * hyperparameter tuning using parfit 
      https://towardsdatascience.com/how-to-make-sgd-classifier-perform-as-well-as-logistic-regression-using-parfit-cc10bca2d3c4

    6. Candidate stackers 

       Naive Bayes: sklearn.naive_bayes.GaussianNB
       AdaBoost:    sklearn.ensemble.AdaBoostClassifier
            ref: https://chrisalbon.com/machine_learning/trees_and_forests/adaboost_classifier/
                 https://www.datacamp.com/community/tutorials/adaboost-classifier-python    # customizing base estimator

       Decision Tree: sklearn.tree.DecisionTreeClassifier 
            ref: http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/
                 http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/

            Defining some of the attributes like max_depth, max_leaf_nodes, min_impurity_split, and min_samples_leaf can help prevent overfitting the model to the training data.

            + plot tree using GraphViz (http://www.graphviz.org/)
                pip install graphviz
                pip install pygraphviz
                tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)

       KNN:          sklearn.neighbors.KNeighborspredictor

       logistic:     sklearn.linear_model.LogisticRegression
       SGD:          sklearn.linear_model.SGDpredictor
       RF:           sklearn.ensemble.RandomForestpredictor

    """
    if 'predictClassValue' not in p:
        stacker = NNLS()
    else:
        if len(argv) > 3: 
            stacker = argv[3]
            # ... args: project_path, method ('aggregate', 'standard'), stacker name (...)

        # specify a stacker either by name or by providing a stacker function
        if isinstance(stacker, str): 

            if stacker in SpecialStacker.names:
                print('(customize_stacker) special stacker: {name}'.format(stacker))
                raise NotImplementedError("Coming soon!")
            else: 
                stacker = choose_classifier(stacker)
        else: 
            # assert stacker is not None
            assert callable(getattr(stacker, 'fit', None)), "Unsupported classifier: {name}".format(name=get_stacker_name(stacker))
    
    # assert hasattr(stacker, 'predict_proba'), "Given stacker is not a probabilistic classifier: %s" % stacker

    return stacker # usually a classifier object with fit() and predict_proba() but can be a special stacker as a string as well

def run(stacker=None, **kargs): 
    """

    Params
    ------
        **kargs reserved for model parameters (e.g. n_estimators for RandomForestClassifier)

    """
    def get_stacker_name(): 
        if isinstance(stacker, str): 
            return stacker  # e.g. ['log', 'qda', 'enet', 'svm', 'naive', 'rf', 'ada', 'knn', ]

        assert stacker is not None 
        if hasattr(stacker, '__class__'): 
            return stacker.__class__.__name__
        if hasattr(stacker, '__name__'): 
            return stacker.__name__ 
        return str(stacker)

    global fold_count  # may need to changethe default defined in 'config.txt'
    
    #######################################################
    mode_module = 'cf'  # values: {'datasink', 'cf', } | this determines the output format
    mode_evaluation = kargs.get('mode', 'train-test-split')
    dataset = kargs.get('dataset', 'bp')
    performance_id = kargs.get('performance_id', '')
    if not performance_id: 
        if isinstance(dataset, dict): 
            performance_id = collections.Counter(dataset.values()).most_common(1)[0][0]
        else: 
            performance_id = dataset
    print('(run) performance_id: {id}'.format(id=performance_id))

    file_type = kargs.get('file_type', '?')
    #######################################################
    stacker_name = stacker if isinstance(stacker, str) else get_stacker_name() # use the original input string as its name whenever possible
    stacker = customize_stacker(stacker=stacker, **kargs)

    if 'fold_count' in kargs: fold_count = kargs['fold_count']

    indices = kargs.get('indices', [])
    if len(indices) == 0: 
        if isinstance(dataset, dict): 
            indices = sorted(dataset.keys()) 
        else: 
            indices = range(fold_count)
    assert len(indices) > 0, "Number of iterations has to be a least 1!"

    div("(run) stacker: {name} | data type: {dataset}, file_type: {dtype} | n_cycles: {nc} | eval mode: {mode}".format(name=name_stacker(stacker), 
        dataset=dataset, dtype=file_type, nc=len(indices), mode=mode_evaluation))
    # ... dataset can be a dictionary: e.g. {0: 'wmf_F100_A100_XCFuser_S2', 2: 'wmf_F75_A100_XCFuser_S2', ... }

    if kargs.pop('parallelize', True): 
        # [note] this many not work when fold/index are not contiguous due to random subsampling process
        # predictions_dfs = Parallel(n_jobs = -1, verbose = 1)(delayed(stacked_generalization)(fold, stacker, **kargs) for fold in range(fold_count))

        predictions_dfs = Parallel(n_jobs = -1, verbose = 1)(delayed(stacked_generalization)(index, stacker, **kargs) for index in indices)
    else: 
        print('(run) Parallelization disabled ...')
        predictions_dfs = []
        for index in indices:
            predictions_dfs.append(stacked_generalization(index, stacker, **kargs))
    
    predictions_df = concat(predictions_dfs)
 
    # save prediction results
    if mode_module == 'datasink': 
        predictions_df['method'] = method   # standard vs aggregate (combining all bags)
        predictions_df.to_csv('%s/analysis/stacking-%s.csv' % (project_path, method), index = False)
    else: 
        # have ['fold', 'id', 'label', 'prediction', 'diversity']
        # need ['label', 'prediction', 'method', 'round']

        print('(verify) inserting column method: {m} ...'.format(m=stacker_name))
        predictions_df['method'] = stacker_name    # standard vs aggregate (combining all bags)

        # naming convention: 
        #  e.g. AB.S-prediction.csv for the regular data set (generated by base predictors)
        #       AB.S-wmf_F10_A100_Xbrier_rating-prediction.csv (generated by CF reconstruction)
        dset_type = kargs.get('file_type', 'prediction')  # values: {'prior', 'posterior', }
        if dataset in ('', None, 'bp'): 
            fpath = '{path}/analysis/{stacker}.S-{suffix}.csv'.format(path=project_path, stacker=stacker_name, suffix=dset_type)
            print('(verify) saving predictions from {name} on original level-1 data to:\n{output}\n'.format(name=stacker_name, output=fpath))
            predictions_df.to_csv(fpath, index = False)
        else: 
            fpath = '{path}/analysis/{stacker}.S-{method_id}-{suffix}.csv'.format(path=project_path, 
                stacker=stacker_name, method_id=performance_id, suffix=dset_type)
             
            print('(verify) saving predictions from {name} on reconstructed data {data} to:\n{output}\n'.format(name=stacker_name, data=dataset, output=fpath))
            predictions_df.to_csv(fpath, index = False)

    # print('(run) predictions_df:\n%s\n' % predictions_df.head(10))

    # common.score <- sklearn.metrics.roc_auc_score
    print( '%.3f' % predictions_df.groupby('fold').apply(lambda x: common.score(x.label, x.prediction)).mean() )

    return predictions_df

def app(prefix=None, project_path=None, stacker=None, **kargs): 
    ########################################################################################
    # ... Global configuration
    if prefix is None: 
        user = getpass.getuser() # 'pleiades' 
        prefix = '/Users/%s/Documents/work/' % user  # /Users/<username>/Documents/work/data/recommender
    assert exists(prefix), f"Invalid prefix: {prefix}"

    if not project_path: 
        dataset_name = 'diabetes_cf'
        project_path = abspath(argv[1]) if len(argv) > 1 else os.path.join(prefix, 'data/%s' % dataset_name) # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    assert exists(project_path), "sys.argv: %s" % argv
    
    if not exists('%s/analysis' % project_path):
        mkdir('%s/analysis' % project_path)

    method = argv[2] if len(argv) > 2 else 'standard'
    assert method in ['aggregate', 'standard']  # aggregate all bags? 

    p = common.load_properties(project_path, config_file='config.txt')  # parse config.txt (instead of weka.properties)
    fold_count = int(p['foldCount'])
    bag_count = int(p['bagCount']) if 'bagCount' in p else int(p['bags']) 
    bags = bag_values = range(bag_count) if bag_count > 1 else [0]

    # level 1 data (e.g. for training CF)
    l1_data_path = os.path.join(project_path, 'LEVEL1')
    if not exists(l1_data_path):
        # os.makedirs(l1_data_path) # creating multiple directories at once
        os.mkdir(l1_data_path)
    ########################################################################################

    run(stacker, **kargs)

    return

def t_stacker(**kargs):
    from analyze_performance import Analysis
    import collections
    import getpass
    from numpy import linalg as LA
    global project_path

    # debugging 
    np.set_printoptions(precision=3)
    domain = 'pf1' # 'diabetes_cf'
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    user = getpass.getuser() # 'pleiades'
    print('> user: {u}, domain: {dom}'.format(u=user, dom=domain))

    method = 'log'
    dataset = 'wmf_F75_A100_XCFuser_S2'
    indices = range(5)
    policy_iter = 'subsampling'
    file_type = 'posterior'
    mode_evaluation = 'train-test-split'

    # stacker = choose_classifier(method)
    predictions_df = run(stacker=method, dataset=dataset, parallelize=kargs.pop('parallelize', True), 
            indices=indices, policy_iter=policy_iter, file_type=file_type, mode=mode_evaluation) 
    print('> cols(predictions_df): {alist}'.format(alist=predictions_df.columns.values))

    return 
def t_read(**kargs): 
    # read data set
    if 'fold_count' in kargs: fold_count = kargs['fold_count']
    for fold in range(fold_count): 
        train_df, train_labels, test_df, test_labels = read(fold, dataset='wmf', reconstructed_testset=False)
        print('(read) fold={0} > dim(train_df):{1}'.format(fold, train_df.shape))

def test(**kargs): 

    ### reading data
    # t_read()

    t_stacker(**kargs)

    return

if __name__ == "__main__": 
    # run()
    test()


