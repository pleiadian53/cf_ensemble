import numpy as np
import pandas as pd
# import common

import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.style.use('seaborn')

import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Some basic classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # AdaBoostClassifier, StackingClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

"""

Reference 
---------
1. sklearn-crfsuite
   pip install sklearn-crfsuite
   https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
"""

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian generative classification based on KDE.
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        """
        predict_proba() returns an array of class probabilities of shape [n_samples, n_classes]

        Entry [i, j] of this array is the posterior probability that sample i is a member of class j, 
        computed by multiplying the likelihood by the class prior and normalizing.
        """
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class CESClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self): 
        pass 

# Utility function to report best scores
def report(results, n_top=3):
    """
    Params
    ------
    results: am output dictionary from a GridSearchCV or RandomizedSearchCV
             as in grid_search.cv_results_ (which is a dictionary)

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def is_label_prediction(y_pred, n_classes=2):
    if str(np.array(y_pred).dtype).startswith('int'): 
        return True
    values = np.unique(y_pred)
    if len(values) <= n_classes: 
        return True
    return False

def optimize_crf_params(X, y, model, labels, max_size=3000, verfiy=True): 
    import scipy.stats as stats
    from sklearn.metrics import make_scorer
    # import sklearn_crfsuite
    from sklearn_crfsuite import scorers
    from sklearn_crfsuite import metrics

    params_space = {
        'c1': stats.expon(scale=0.5),
        'c2': stats.expon(scale=0.05),
    }

    # use f1 for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, 
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(model, params_space, 
                            cv=3, 
                            verbose=1, 
                            n_jobs=-1, 
                            n_iter=50, 
                            scoring=f1_scorer)

    X_train, y_train = X, y 
    N = len(X_train)
    if N > max_size: 
        indices = np.random.choice(range(N), max_size)
        X_train = list( np.asarray(X_train)[indices] )
        y_train = list( np.asarray(y_train)[indices] )

    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    # if verify: validate_crf_params(rs, output_path=None, dpi=300)

    return rs.best_estimator_

def validate_crf_params(rs, output_path=None, dpi=300, save=True, verbose=True): 
    from utils_plot import saveFig

    _x = [s.parameters['c1'] for s in rs.cv_results_]
    _y = [s.parameters['c2'] for s in rs.cv_results_]
    _c = [s.mean_validation_score for s in rs.cv_results_]

    plt.clf() 

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

    print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    if save: 
        basedir = os.path.join(os.getcwd(), 'analysis')
        if not os.path.exists(basedir) and create_dir:
            print('(validate_crf_params) Creating analysis directory:\n%s\n' % basedir)
            os.mkdir(basedir) 

        if output_path is None: 
            if not name: name = 'generic'
            fname = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix='{}-crf'.format(kernel), suffix=name, index=index, ext=ext)
            output_path = os.path.join(basedir, fname)  # example path: System.analysisPath
        else: 
            # output_path can be either a file name OR a full path including the file name
            prefix, fname = os.path.dirname(output_path), os.path.basename(output_path)
            if not prefix: 
                prefix = basedir
                output_path = os.path.join(basedir, fname)
            assert os.path.exists(output_path), "Invalid output path: {}".format(output_path)

        if verbose: print('(validate_crf_params) Saving distribution plot at: {path}'.format(path=output_path))
        saveFig(plt, output_path, dpi=dpi, verbose=verbose) 
    else: 
        # pass
        try: 
            plt.show()
        except: 
            pass
    return


def choose_classifier(name, **kargs): 
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier, LogisticRegression, Lasso, LassoCV, Perceptron
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.svm import SVC

    verbose = kargs.get("verbose", 0)
    if verbose > 1: print('> Classifier name: %s' % name)

    model = None
    if name.startswith('percep'):  # perceptron
        model = SGDClassifier(loss = 'perceptron', n_iter = 50, random_state = 0)   # loss = 'log' => logistic regression
        # model = Perceptron(tol=1e-3, random_state=0, penalty='l2') # does not have predict_probs()
    elif name.startswith('log'):
        model = LogisticRegression(penalty='l2', tol=1e-4, warm_start=False, solver='sag') # max_iter=int(1e4), class_weight='balanced'
    elif name == 'enet':
        params = {'penalty': 'elasticnet', 'alpha': 0.01, 'loss': 'modified_huber', 
                     'fit_intercept': True, 'random_state': 0, 'tol': 1e-4}   # 'tol': 1e-3, 
        model = SGDClassifier(**params)
        
    elif name == 'lasso':
        # max_iter is useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
        model = LogisticRegression(penalty='l1', solver='saga',
                              tol=1e-4, 
                              warm_start=False) # set warm_start to reuse learned coeffs

    elif name.startswith(('qda', 'quad')):  # QDA 
        # default: priors=None, reg_param=0.0, store_covariance=False, tol=0.0001
        model = QuadraticDiscriminantAnalysis() 
    
    elif name.startswith(('svm', 'svc', )):  # SVM with linear kernel 
        # default: C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001,
        model = svc = SVC(probability=True, kernel='linear', class_weight='balanced') 

    elif name.startswith(('rf', 'randomf','random_f')):
        model = RandomForestClassifier(n_estimators = kargs.get('n_estimators', 150), 
                    max_depth = 2, bootstrap = False, random_state = 0)

    elif name.startswith(('nai', ) ):   # naive bayes
        model = GaussianNB()          # default: priors=None, var_smoothing=1e-09

    elif name.startswith('ada'): # AdaBoost with base_estimator as decision tree 
        model = AdaBoostClassifier(n_estimators=kargs.get('n_estimators', 100), random_state=0) 

        ### using other base estimator 
        # svc = SVC(probability=True, kernel='linear')
        # model = AdaBoostClassifier(base_estimator=svc, n_estimators=100, random_state=0)
    elif name.startswith( ('tree', 'deci') ):  # decision tree classifier
        # default: criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None
        model = DecisionTreeClassifier(class_weight='balanced', criterion = "gini", random_state = 100,
                       max_depth=4, min_samples_leaf=5)

    elif name.startswith(('grad', 'gb')):  # gradient boost tree 
        model = GradientBoostingClassifier(n_estimators=kargs.get('n_estimators', 150), 
                    learning_rate=0.1, 
                    # random_state=53, 
                    min_samples_split=250, min_samples_leaf=5, max_depth=4,  # prevent overfitting
                    max_features = 'sqrt', # Its a general thumb-rule to start with square root.
                    subsample=0.8) # fraction of samples to be used for fitting the individual base learners
    elif name.startswith(('knn', 'neighbor')): 
        model =  KNeighborsClassifier(n_neighbors=5)

    # [test]
    assert callable(getattr(model, 'fit', None)), "Unsupported classifier: {name}".format(name=name)
    # a valid classifier should provide the following operations
    #     model = model.fit(train_df, train_labels)
    #     test_predictions = model.predict_proba(test_df)[:, 1]  # 1/foldCount worth of data

    return model

###############################################################################################
# Probability Thresholding Utilities
###############################################################################################

def auc_threshold(labels, predictions, verbose=0, pos_label=1): 
    """
    Identify that threshold that gives us the upper-left corner of the curve. 
    Mathematically speaking, we want to find

    p* = argmin< p > | tpr(p) + fpr(p) - 1 |

    Memo
    ----
    1. roc_curve(): 
       - drop_intermediate
          Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve. 
          This is useful in order to create lighter ROC curves.
    """
    # from sklearn.metrics import roc_curve, plot_roc_curve 

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, drop_intermediate=False, pos_label=pos_label)

    if verbose: 
        # Plot the objective function with respect to the threshold and see where its minimum is
        plt.scatter(thresholds, np.abs(fpr+tpr-1))
        plt.xlabel("Threshold")
        plt.ylabel("|FPR + TPR - 1|")
        plt.show()

    return thresholds[np.argmin(np.abs(fpr+tpr-1))]

def auc_score_threshold(labels, predictions, verbose=0, pos_label=1):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, drop_intermediate=False, pos_label=pos_label)
    if verbose: 
        # Plot the objective function with respect to the threshold and see where its minimum is
        plt.scatter(thresholds, np.abs(fpr+tpr-1))
        plt.xlabel("Threshold")
        plt.ylabel("|FPR + TPR - 1|")
        plt.show()

    score = metrics.auc(fpr, tpr)
    threshold = thresholds[np.argmin(np.abs(fpr+tpr-1))]

    return (score, threshold)

def acc_max_threshold(labels, predictions, verbose=0, pos_label=1):
    # from sklearn.metrics import balanced_accuracy_score

    thresholds = []
    accuracies = []

    for p in np.unique(predictions):
        thresholds.append(p)
        y_pred = (predictions >= p).astype(int)
        accuracies.append(metrics.balanced_accuracy_score(labels, y_pred))

    if verbose: 
        plt.scatter(thresholds, accuracies)
        plt.xlabel("Threshold")
        plt.ylabel("Balanced accuracy")
        plt.show()

    return thresholds[np.argmax(accuracies)]

def acc_max_score_threshold(labels, predictions, verbose=0, pos_label=1): 
    thresholds = []
    accuracies = []

    for p in np.unique(predictions):
        thresholds.append(p)
        y_pred = (predictions >= p).astype(int)
        accuracies.append(metrics.balanced_accuracy_score(labels, y_pred))

    if verbose: 
        plt.scatter(thresholds, accuracies)
        plt.xlabel("Threshold")
        plt.ylabel("Balanced accuracy")
        plt.show()

    imax = np.argmax(accuracies)
    # score = np.max(accuracies)
    # threshold = thresholds[np.argmax(accuracies)]
    return accuracies[imax], thresholds[imax]

def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative

    """
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    # i = np.nanargmax(f1)

    # return (f1[i], threshold[i])
    return np.nanmax(f1)

def fmax_threshold(labels, predictions, beta = 1.0, pos_label = 1): 
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == np.nanmax(f1)
    return th

def fmax_score_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    """

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative
       
       precision: Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1. 
       recall: Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.

    2. example 

    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision  
    array([0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])

    precision[1] = 0.5, for any prediction >= thresholds[1] = 0.4 as positive (assuming that pos_label = 1)

    """
    # from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == np.nanmax(f1)
    return (f1[i], th)

def fmax_precision_recall_scores(labels, predictions, beta = 1.0, pos_label = 1):
    ret = {}
    precision, recall, threshold = metrics.precision_recall_curve(labels, predictions, pos_label=pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 

    ret['id'] = i 
    ret['threshold'] = th
    ret['precision'] = precision[i] # element i is the precision of predictions with score >= thresholds[i]
    ret['recall'] = recall[i]
    ret['f'] = ret['fmax'] = f1[i]
    
    return ret  # key:  id, precision, recall, f/fmax 


###############################################################################################
# Model selection utilities 
###############################################################################################


def hyperparameter_template(model='rf'):
    
    if model.lower() in ('rf', 'random forest'): 
        n_estimators = [200, 300 ] # [int(x) for x in np.linspace(start = 100, stop = 700, num = 50)]
        max_features = ['auto', ] # ['auto', 'log2']  # Number of features to consider at every split
        
        max_depth = [8, 10] # [int(x) for x in np.linspace(10, 110, num = 11)]   # Maximum number of levels in tree
        max_depth.append(None)

        min_samples_split = [2, 4]  # Minimum number of samples required to split a node
        min_samples_leaf = [1, ]    # Minimum number of samples required at each leaf node
        max_leaf_nodes = [None,  ] # + [10, 25, 50] # [None] + list(np.linspace(10, 50, 500).astype(int)), # 10 to 50 "inclusive"
        bootstrap = [True, False]       # Method of selecting samples for training each tree
        # ... NOTE: Out of bag estimation only available if bootstrap=True

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'max_leaf_nodes': max_leaf_nodes, 
                       'bootstrap': bootstrap}
    elif model.lower().startswith(('logis', 'logit', )): 
        solvers = ['lbfgs', ] # ['newton-cg', 'lbfgs', 'liblinear'] 
        # Note: newton-cg and lbfgs solvers support only l2 penalties.

        penalty = ['l2', ] # 'l1'
        c_values = np.logspace(-3, 2, 6) # [100, 10, 1.0, 0.1, 0.01]
        random_grid = dict(solver=solvers, penalty=penalty, C=c_values)
    else: 
        raise NotImplementedError(f"{model.capitalize()} not supported. Coming soon :)")
    return random_grid

def tune_model(model, grid, cv=None, **kargs): 
    """

    Parameters 
    ---------- 
    model: a classifier to tune for its best hyperparameter settings
    grid: a parameter dictionary
    """ 
    scoring = kargs.get('scoring', 'f1') # Evaluation metric
    verbose = kargs.get('verbose', 1)

    def fit_on_data(X, y): 
        nonlocal cv 
        if cv is None: 
            random_state = kargs.get('random_state', 53)
            n_splits = kargs.get('n_splits', 5)
            n_repeats = kargs.get('n_repeats', 2)
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring, error_score=0, verbose=verbose)
        model_tuned = grid_result = grid_search.fit(X, y)
        
        # summarize results
        if verbose: 
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        
        return model_tuned # to make predictions: call .predict(X_test) or grid_result.best_estimator_.predict(X_test)
    return fit_on_data

# Dfine grid search
def demo_logistic_regression_tuning(): 
    """

    Reference 
    ---------
    1. How to use GridSearchCV ouptut
       https://stackoverflow.com/questions/35388647/how-to-use-gridsearchcv-output-for-a-scikit-prediction
    """

    # example of grid searching key hyperparametres for logistic regression
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    # define dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
    
    # define models and parameters
    #################################
    model = LogisticRegression()
    solvers = ['lbfgs', ] # ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2', 'l1', ]
    c_values = [100, 10, 1.0, 0.1, 0.01]
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    #################################

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result

def demo_knn_tuning(): 
    # example of grid searching key hyperparametres for KNeighborsClassifier
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    
    # define dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)

    # define models and parameters
    model = KNeighborsClassifier()

    # define grid search
    #################################
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
    #################################
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result

def demo_gradient_boosting(): 
    # example of grid searching key hyperparameters for GradientBoostingClassifier
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    
    # define dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
    
    # define models and parameters
    model = GradientBoostingClassifier()

    # define grid search
    #################################
    n_estimators = [10, 100, 1000]
    learning_rate = [0.001, 0.01, 0.1]
    subsample = [0.5, 0.7, 1.0]
    max_depth = [3, 7, 9]
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    #################################
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result

###############################################################################################
# Performance evaluation utilities
# 
# Related Modules
# ---------------
#   1. evaluate
#   2. common
# 
###############################################################################################

def evaluate_model(train, test):
    y_train, train_predictions, train_probs = train # order: true labels, predicted labels, probability scores
    y_test, y_pred, probs = test

    baseline = {}
    baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5

    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)

    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)

    for metric in ['recall', 'precision', 'roc']:  
          print(f'{metric.capitalize()} \
                 Baseline: {round(baseline[metric], 2)} \
                 Test: {round(results[metric], 2)} \
                 Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();
    

# Data generation utilities
######################################################################

def generate_simple_data(n_samples=5000, noise=False):
    if noise: 
        X,y = make_classification(n_samples=n_samples, n_features=15, n_informative=9, 
                        n_redundant=3, n_repeated=2, n_classes=2, n_clusters_per_class=1,
                            class_sep=2,
                    flip_y=0.2, # <<< 
                    weights=[0.5,0.5], random_state=17)
    else: 
        X,y = make_classification(n_samples=n_samples, n_features=15, n_informative=9, 
                            n_redundant=3, n_repeated=2, n_classes=2, n_clusters_per_class=1,
                                class_sep=2,
                        flip_y=0,weights=[0.5,0.5], random_state=17)
    return X, y

def generate_gaussian_quantiles(n_samples=5000, verbose=0): 
    """
    Generate synthetic data for binary classification. 

    Reference 
    ---------
    1. https://towardsdatascience.com/https-medium-com-faizanahemad-generating-synthetic-classification-data-using-scikit-1590c1632922
    """
    from sklearn.datasets import make_gaussian_quantiles

    n_neg = n_samples//2
    n_pos = n_samples - n_neg

    # Construct dataset
    # Gaussian 1
    X1, y1 = make_gaussian_quantiles(cov=3.,
                                     n_samples=n_neg, n_features=2,
                                     n_classes=2, random_state=1)
    X1 = pd.DataFrame(X1,columns=['x','y'])
    y1 = pd.Series(y1)

    # Gaussian 2
    X2, y2 = make_gaussian_quantiles(mean=(4, 4), cov=1,
                                     n_samples=n_pos, n_features=2,
                                     n_classes=2, random_state=1)
    X2 = pd.DataFrame(X2,columns=['x','y'])
    y2 = pd.Series(y2)

    # Combine the gaussians
    if verbose: 
        print('> shape(X1)', X1.shape)
        print('> shape(X2)', X2.shape)

    X = pd.DataFrame(np.concatenate((X1, X2)))
    y = pd.Series(np.concatenate((y1, - y2 + 1)))
    print(X.shape)

    X = X.values
    # visualize_2d(X,y)

    return (X, y)


######################################################################

# Convert y comprising >=3 classes into y_prime with only two classes
def to_binary_classification(y, target_class): 
    """
    Label encode y followed by reducing the encoded labels into a binary vector
    where 1 represents positive classes and 0 represents negative classes

    Params
    ------
    target_class: the label of the positive class
    """
    from sklearn import preprocessing

    # First label-encode y
    le = preprocessing.LabelEncoder()
    y_prime = le.fit_transform(y)

    y_map = dict(zip(le.classes_, le.transform(le.classes_))) # label -> encoded label
    target_encoded = y_map[target_class]
    y_prime = np.where(y_prime == target_encoded, 1, 0)

    return y_prime, y_map, le


def is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all, raise_exception=False):
    """
    Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify if the
    estimator is fitted or not.

    Parameters
    ----------
    estimator : estimator instance
        estimator instance for which the check is performed.
    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``
        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.
    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    True if `estimator` is fitted, False otherwise
    ------
    NotFittedError
        If the attributes are not found.

    Reference 
    ---------
    [1] sklearn's check_is_fitted(): 
        https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/utils/validation.py#L1153
    """
    from inspect import signature, isclass, Parameter

    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        if raise_exception: 
            raise NotFittedError(msg % {"name": type(estimator).__name__})
        return False

    return True

def load_iris(): 
    """

    Memo
    ----
    1. n_samples: 100, n_features: 804, n_classes: 2
    """
    from sklearn import datasets
    # Data IO and generation

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape
    print('(data) n_samples: {n}, n_features: {nf}'.format(n=n_samples, nf=n_features))

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    return (X, y)

def demo_kde_classifier(): 
    from sklearn.datasets import load_digits
    from sklearn.grid_search import GridSearchCV

    digits = load_digits()

    bandwidths = 10 ** np.linspace(0, 2, 100)
    grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
    grid.fit(digits.data, digits.target)

    scores = [val.mean_validation_score for val in grid.grid_scores_]

    plt.semilogx(bandwidths, scores)
    plt.xlabel('bandwidth')
    plt.ylabel('accuracy')
    plt.title('KDE Model Performance')
    print(grid.best_params_)
    print('accuracy =', grid.best_score_)

    return

def demo_mean_classifier(): 
    from sklearn.model_selection import train_test_split

    estimator = MeanClassifier()
    X, y = load_iris()
    print('> dim(X): {0} | dim(y): {1}'.format(X.shape, y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    print('> dim(X_train): {0} | dim(y_train): {1}'.format(X_train.shape, y_train.shape))
    estimator.fit(X_train, y_train)
    print('> test example: {0}'.format(X_test[:10]))

    y_score = estimator.predict_proba(X_test, y_test)
    print('> scores:       {0}'.format(y_score[:10]))

    return

def demo_model_selection(): 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    # Generate data
    X, y = generate_gaussian_quantiles(n_samples=5000, verbose=0)

    # Train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    # Define model
    model = RandomForestClassifier()

    n_estimators = [10, 100, 500, ]
    max_features = ['sqrt', 'log2']
    grid = dict(n_estimators=n_estimators,max_features=max_features)

    tuner = tune_model(model, grid, scoring='f1', verbose=1)
    model_tuned = tuner(X_train, y_train)
    y_pred = model_tuned.predict(X_test)
    print(f"> shape(y_pred): {y_pred.shape}")

    perf_score = f1_score(y_test, y_pred)
    print(f'[result] F1 score:  {perf_score}')

    #####################################
    
    # Define another model
    model = LogisticRegression()
    grid = hyperparameter_template('logistic')

    tuner = tune_model(model, grid, scoring='roc_auc', verbose=1)
    model_tuned = tuner(X_train, y_train)
    y_pred = model_tuned.predict(X_test)
    print(f"> shape(y_pred): {y_pred.shape}")

    perf_score = f1_score(y_test, y_pred)
    print(f'[result] F1 score:  {perf_score}')

    return

def test(**kargs): 
    
    # Model selection-related utilities
    # demo_logistic_regression_tuning()

    demo_model_selection()

    return

if __name__ == "__main__":
    test() 



