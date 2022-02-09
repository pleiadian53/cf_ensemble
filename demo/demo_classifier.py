import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) # include parent directory to the module search path

from utils_sys import div

import numpy as np
from scipy import interp

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

### classifiers 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier


"""

    Memo
    ----
    1. related modules: 

       generate_data

"""

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

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    return (X, y)

def inspect(data): 
    X, y = data[0], data[1]
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    print('(inspect) n_samples: %d, n_features: %d, n_classes: %d' % (n_samples, n_features, n_classes))

    return 

def demo_classifier(**kargs): 
    # from sklearn.metrics import accuracy_score
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.svm import SVC
    # from sklearn.gaussian_process import GaussianProcessClassifier
    # import matplotlib.pyplot as plt
    from sklearn.gaussian_process.kernels import RBF
    import evaluate
    from evaluate import Metrics, PerformanceMetrics

    X, y = load_iris()
    inspect(data=(X,y))


    perf = PerformanceMetrics()   # rows: metrics, cols: bps

    C = 10
    kernel = 1.0 * RBF([1.0] * X.shape[1])  # for GPC

    # Create different classifiers.
    classifiers = {
        'L1 logistic': LogisticRegression(C=C, penalty='l1',
                                      solver='saga',
                                      multi_class='multinomial',
                                      max_iter=10000),
        'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
                                                    solver='saga',
                                                    multi_class='multinomial',
                                                    max_iter=10000),
        'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
                                            solver='saga',
                                            multi_class='ovr',
                                            max_iter=10000),
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                      random_state=0),
        # 'GPC': GaussianProcessClassifier(kernel)
    }

    n_classifiers = len(classifiers)
    metrics = kargs.get('metrics_tracked', Metrics.tracked)  # ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', ] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

        mdict = evaluate.scores(y_test, y_pred, metrics=metrics) # compute all (tracked) performance at once
        if index == 0: print('... tracked metrics: %s' % mdict.keys())
        # add column by column (each method is a column in this performance metrics dataframe)
        perf.add(scores=mdict, method=name)  # use a dataframe to keep tracck of performance scores, which then faciliates plotting
        # or merge the above two lines as 
        # perf.add_and_eval(y, y_pred, method=name)

    # best performance? 
    div(message=perf.table, symbol='=', border=2)
    sorted_bps = perf.sort(metric='specificity', verbose=True)
    div(message=sorted_bps, symbol='=', border=2)
    
    return

def test(**kargs): 
    # X, y = load_iris()
    # inspect(data=(X,y)) 

    ## basic demo of classifiers + performance evaluation
    demo_classifier()
    

    return

if __name__ == "__main__": 
    test()
