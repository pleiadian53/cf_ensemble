import sklearn.metrics
from numpy import random, nanmax
from pandas import concat, read_csv, DataFrame
import math



def softmax(X, axis=0): 
    # if scipy.sparse.issparse(X): X = X.toarray()
    # column-wise softmax
    X = X.astype(float)
    if X.ndim==1:
        S=np.sum(np.exp(X))
        return np.exp(X)/S
    elif X.ndim==2:
        Xw= np.zeros_like(X)
        m,n = X.shape
        for j in range(n):
            S=np.sum(np.exp(X[:,j])) # column sum-of-exp
            Xw[:,j]=np.exp(X[:,j])/S
        return Xw
    else:
        print("The input array is not 1- or 2-dimensional.")
    return X

# Note: For demo only
def pearson_correlation(object1, object2):
    values = range(len(object1))
    
    # Summation over all attributes for both objects
    sum_object1 = sum([float(object1[i]) for i in values]) 
    sum_object2 = sum([float(object2[i]) for i in values])

    # Sum the squares
    square_sum1 = sum([pow(object1[i],2) for i in values])
    square_sum2 = sum([pow(object2[i],2) for i in values])

    # Add up the products
    product = sum([object1[i]*object2[i] for i in values])

    #Calculate Pearson Correlation score
    numerator = product - (sum_object1*sum_object2/len(object1))
    denominator = ((square_sum1 - pow(sum_object1,2)/len(object1)) * (square_sum2 - 
        pow(sum_object2,2)/len(object1))) ** 0.5
        
    # Can"t have division by 0
    if denominator == 0:
        return 0

    result = numerator/denominator
    return result

def load_properties(dirname, config_file='config.txt'):
    """
    Configuration parser. 

    Memo
    ----
    1. datasink uses weka.properties instead of config.txt
    """

    # [todo] add remove comments
    properties = [line.split('=') for line in open(dirname + '/%s' % config_file).readlines() 
                    if len(line) > 0 and not line.strip().startswith('#')]
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d

def cluster_cmd(queue='expressalloc', allocation='acc_pandeg01a'): 
    """

    Memo
    ----
    1. If the 'cmd' in return cmd is left out 
       ~> python: can't open file 'None': [Errno 2] No such file or directory 
    """
    cmd = 'rc.py --cores 1 --walltime 00:10 --queue %s' % queue
    if allocation: 
        cmd += ' ' + '--allocation %s' % allocation   # job name
    return cmd  

def get_num_cores():
    #return -1 #spawning processes on all the available cores of the local machine
    return 1

def get_fold_ens(fileName):
    with open(fileName, 'r') as f:
        content = f.readline()
        ens = content.split(':: (',1)[1].split(')')[0]
        ens = (ens[:-1] if ens[-1] == ',' else ens)
        predictors = map(int, ens.split(","))
    f.close()
    return predictors

def load_arff_headers(filename):
    dtypes = {}
    for line in open(filename):
        if line.startswith('@data'):
            break
        if line.startswith('@attribute'):
            _, name, dtype = line.split()
            if dtype.startswith('{'):
                dtype = dtype[1:-1]
            dtypes[name] = set(dtype.split(','))
    return dtypes


def f_score(labels, predictions, beta = 1.0, pos_label = 1):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f1

def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """

        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.

    Memo
    ----
    1. nanmax(): Return the maximum of an array or maximum along an axis, ignoring any NaNs.
    2. fmax is: A function of precision and recall values. 

    3. runtime warning

       RuntimeWarning: invalid value encountered in divide
       f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    """
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return nanmax(f1) 


def get_set_preds(dirname, set, bag, fold, seed):
    filename = '%s/%s-b%s-f%s-s%s.csv.gz' % (dirname, set, bag, fold, seed)
    df  = read_csv(filename, skiprows = 1, compression = 'gzip')
    y_true = df['label'] # df.ix[:,1:2]   
    y_score = df['prediction'] # df.ix[:,2:3]  
    return y_true, y_score   # y_true: a series of labels


def get_bps(project_path, seed, metric, size): #to vary the ensemble size, select randomly w/o replacement, an equal number of good, medium, and weak performance bps
    """

    Memo
    ----
    1. run step2_order.py to get ENSEMBLES

    2. project_path: /Users/chiup04/Documents/work/data/diabetes_cf
       seed
       metric
       size: [1, max_num_clsf+1], where max_num_clsf = number of bps * n_seeds
    """
    # a list of (classifier, perf score) (specified by metric), e.g. order_of_seed0_fmax.txt
    order_file = "%s/ENSEMBLES/order_of_seed%s_%s.txt" % (project_path, seed, metric) 

    with open(order_file, 'r') as f:
        content = f.read().splitlines()
    f.close()

    max_num_clsf = len(content)    
    bps_weight = {}
    subset = []
    random.seed(int(seed))

    num_bins = 3  # good, medium, weak
    interval = int(math.floor(max_num_clsf/num_bins)) 
    rem = max_num_clsf%num_bins

    for bin in range(num_bins):
        num_sel  = int(math.floor(size/num_bins) + 1 if (size % num_bins > bin) else int(math.floor(size/num_bins)))
        if bin == 0:
            start = 0
            end   = (interval + 1 if rem else interval)
        elif bin == 1:
            start = (interval + 1 if rem else interval)
            end   = (2*interval + rem if rem else 2*interval)
        else:
            start = (2*interval + rem if rem else 2*interval)
            end   = max_num_clsf

        # print('... bin: %d | start: %d, end: %d, num_sel: %d' % (bin, start, end, num_sel))
        selected = random.choice(range(start, end), num_sel, replace=False)
        subset.extend([content[bp] for bp in selected])
        # print('... selected: %s => subset:\n      %s\n' % (selected, subset)) 

    for i in range(len(subset)):
        index = i + 1
        bps_weight[index] = float(subset[i].split(",")[1])

    return subset, bps_weight  # subset: a list of classifiers, bps_weights: performance scores as weights

def bps2string(predictors):
    str = "\n* * * * *\nBase predictors and their weight (performance on the validation, over 5 folds):\n"
    index = 1
    for p in predictors:
        str += "%i :: %s\n" % (index, p)
        index += 1
    str += "* * * * *\n"
    return str

def get_path_bag_weight(predictor):
    path = (predictor.split("_bag")[0].split(":: ")[1] if "::" in predictor else predictor.split("_bag")[0])
    bag  = int(predictor.split("_bag")[1].split(",")[0])
    weight = float(predictor.split(",")[1].strip())
    return path, bag, weight

def aggregate_predictions(predictors, seed, fold, set, RULE):
    denom = 0
    path, bag, weight = get_path_bag_weight(predictors[0])

    denom = ((denom + weight) if RULE == 'WA' else (denom + 1))
    y_true, y_score = get_set_preds(path, set, bag, fold, seed)
    y_score = weight * y_score

    for bp in predictors[1:]:
        path, bag, weight = get_path_bag_weight(bp)
        denom  += weight
        y_true, y_score_current = get_set_preds(path, set, bag, fold, seed)
        y_score = (y_score.add(weight * y_score_current) if RULE =='WA' else y_score.add(y_score_current))

    y_score = y_score/denom
    perf    = fmax_score(y_true, y_score)
    return (y_true, y_score)  # y_true and y_score are now both a Series (previously a dataframe)

def test(**kargs):
    import numpy as np

    ### performance metrics
    labels = y_true = np.array([0, 0, 1.0, 1.0])
    predictions = y_score = np.array([0.1, 0.4, 0.35, 0.8])
    print('... labels: %s' % str(labels.shape))
    s = fmax_score(labels, predictions, beta = 1.0, pos_label = 1)
    print('(test) fmax_score: %f' % s)

    ### get bp 
    project_path = "/Users/chiup04/Documents/work/data/diabetes_lens"
    metric = 'fmax'  # 
    seed = 0 
    size = 5
    subset, bps_weight = get_bps(project_path, seed, metric, size)
    print('(get_bps) subset:     %s' % subset)
    print('...       bps_weight: %s' % bps_weight)  # fmax as weights

    # ... start: 0, end: 4, num_sel: 2
    # ... start: 4, end: 7, num_sel: 2
    # ... start: 7, end: 10, num_sel: 1

    # subset
    # ['/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.functions.SimpleLogistic_bag0, 0.6982055464926591 ', '/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.trees.RandomForest_bag1, 0.69375 ', '/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.functions.Logistic_bag0, 0.689922480620155 ', '/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.meta.AdaBoostM1_bag1, 0.6754772393538914 ', '/Users/chiup04/Documents/work/data/diabetes_lens/weka.classifiers.bayes.NaiveBayes_bag0, 0.6286472148541113 ']
    
    
    return 

if __name__ == "__main__": 
    test()



