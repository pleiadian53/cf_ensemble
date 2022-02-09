import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,os.getcwd()) # include parent directory to the module search path
sys.path.insert(0,parentdir)

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

# select plotting style 
plt.style.use('ggplot')  # values: {'seaborn', 'ggplot', }

from utils_plot import saveFig, plot_path
import utils_sys as utils


import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

import sklearn.metrics
from sklearn.metrics import roc_curve, auc

"""


Reference
---------
   1. data: 
   
      https://stats.idre.ucla.edu/r/dae/logit-regression/
   
   2. Q&A: 
 
      https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python

"""



# utils_plot
def plot_path(name='heatmap', basedir=None, ext='tif', create_dir=True):
    import os
    # create the desired path to the plot by its name
    if basedir is None: basedir = os.path.join(os.getcwd(), 'plot') 
    if not os.path.exists(basedir) and create_dir:
        print('(plot) Creating plot directory:\n%s\n' % basedir)
        os.mkdir(basedir) 
    return os.path.join(basedir, '%s.%s' % (name, ext))

# read the data in
def demo(df=None): 

    # log: pandas.errors.ParserError: Error tokenizing data. C error
    df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv", error_bad_lines=False, engine='python')

    # rename the 'rank' column because there is also a DataFrame method called 'rank'
    df.columns = ["admit", "gre", "gpa", "prestige"]
    # dummify rank
    dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
    # create a clean data frame for the regression
    cols_to_keep = ['admit', 'gre', 'gpa']
    data = df[cols_to_keep].join(dummy_ranks.iloc[:, 'prestige_2':])

    # manually add the intercept
    data['intercept'] = 1.0

    train_cols = data.columns[1:]
    # fit the model
    result = sm.Logit(data['admit'], data[train_cols]).fit()
    print(result.summary())

    # Add prediction to dataframe
    data['pred'] = result.predict(data[train_cols])

    fpr, tpr, thresholds =roc_curve(data['admit'], data['pred'])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    ####################################
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    ####################################
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.ix[(roc.tf-0).abs().argsort()[:1]]

    # Plot tpr vs 1-fpr
    fig, ax = pl.subplots()
    pl.plot(roc['tpr'])
    pl.plot(roc['1-fpr'], color = 'red')
    pl.xlabel('1-False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    ax.set_xticklabels([])

    saveFig(pl, plot_path(name='optimal_auc_cutoff'), dpi=300)

    return

def findOptimalCutoff(y_true, y_score, metric='auc'):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate
    
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    from sklearn.metrics import roc_curve, auc

    if metric == 'auc': 
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})

        # The optimal cut off would be where tpr is high and fpr is low => tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  

        return list(roc_t['threshold'])[0]
    elif metric == 'fmax': 
        pass 

    return

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

    saveFig(plt, plot_path(name='precision_recall_vs_threshold'), dpi=300)

    return

def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)

    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    plt.clf()

    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)

    saveFig(plt, plot_path(name='precision_recall_curve'), dpi=300)

    return

def fmax_score_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    """

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative

    """
    import numpy as np
    # import sklearn.metrics
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)

    return (f1[i], threshold[i])

def findOptimalCutoffFmax(y_true, y_score, beta = 1.0, pos_label=1):
    # import sklearn.metrics

    precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_true, y_score, pos_label) 
    print('... precision:\n%s\n' % precision[:10])
    print('... recall:\n%s\n' % recall[:10])
    print('... threshold:\n%s\n' % threshold[:10])

    fmax, p_th = fmax_score_threshold(y_true, y_score, beta = beta, pos_label = pos_label)
    print('... fmax: {0}, threshold: {1}'.format(fmax, p_th))
    # fmax score
    # f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # i = np.arange(len(precision)) 
    # tradeoff = pd.DataFrame({'tf' : pd.Series(precision-recall, index=i), 'threshold' : pd.Series(threshold, index=i)})

    # # The optimal cut off would be where tpr is high and fpr is low => tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    # tradeoff_t = tradeoff.iloc[(tradeoff.tf-0).abs().argsort()[:1]]  

    return p_th

def get_data():

    # log: pandas.errors.ParserError: Error tokenizing data. C error
    df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv", error_bad_lines=False, engine='python') 

    # rename the 'rank' column because there is also a DataFrame method called 'rank'
    df.columns = ["admit", "gre", "gpa", "prestige"]
    
    # dummify rank
    dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
    
    # create a clean data frame for the regression
    cols_to_keep = ['admit', 'gre', 'gpa']
    data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

    return data

def demo2(): 
    data = get_data()

    # manually add the intercept
    data['intercept'] = 1.0

    train_cols = data.columns[1:]
    # fit the model
    result = sm.Logit(data['admit'], data[train_cols]).fit()
    print(result.summary())

    # Add prediction to dataframe
    # data['pred'] = result.predict(data[train_cols])

    # Add prediction probability to dataframe
    data['pred_proba'] = result.predict(data[train_cols])

    # Find optimal probability threshold
    threshold = findOptimalCutoff(data['admit'], data['pred_proba'])
    print('... optimal threshold: %f' % threshold)
    # [0.31762762459360921]

    threshold_fmax = findOptimalCutoffFmax(data['admit'], data['pred_proba'], beta = 1.0, pos_label=1)

    # Find prediction to the dataframe applying threshold
    data['pred'] = data['pred_proba'].map(lambda x: 1 if x > threshold else 0)
    data['pred2'] = data['pred_proba'].map(lambda x: 1 if x > threshold_fmax else 0)

    # Print confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    M = confusion_matrix(data['admit'], data['pred'])
    print('... confusion_matrix (auc):\n{0}\n'.format(M))
    # array([[175,  98],
    #        [ 46,  81]])

    M2 = confusion_matrix(data['admit'], data['pred2'])
    print('... confusion_matrix (fmax):\n{0}\n'.format(M2))


def test(): 

    demo2()

    return

if __name__ == "__main__":
    test() 