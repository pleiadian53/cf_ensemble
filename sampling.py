import statistics 
import numpy as np 
import random, math, re
import pandas as pd
from pandas import Series, DataFrame
from bisect import bisect

"""

Reference
---------
1. sampling methods in python
    https://people.duke.edu/~ccc14/sta-663/ResamplingAndMonteCarloSimulations.html

"""

def weighted_choice(choices):
    """
    choices: 
       [("WHITE",90), ("RED",8), ("GREEN",2)]
    """
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]

def estimate_ci(x, n_rep=1000, estimate_ci=True, lower=None, upper=97.5): 
    n = len(x)  # dim(x)
    xb = np.random.choice(x, (n, n_rep), replace=True)  # sampling with replacement n times
    mb = xb.mean(axis=0)

    # estimate confidence interval
    mb.sort()
    if lower is None: lower = 100.0-upper
    return np.percentile(mb, [lower, upper])

def bootstrap_resample(x, n=None, random_state=1, verbose=False): 
    """

    Reference
    ---------
    1. https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
    """
    from sklearn.utils import resample   # scikit-learn bootstrap
    # data sample
    data = x
    # prepare bootstrap sample
    boot = resample(data, replace=True, n_samples=n, random_state=random_state)
    if verbose: print('> Bootstrap Sample: %s' % boot)
    
    # out of bag observations
    oob = [x for x in data if x not in boot]
    if verbose: print('> OOB Sample: %s' % oob)

    return (boot, oob)

def bootstrap_resample2(X, n=None, y=None, all_labels_present=True, n_cycles=20):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    import collections
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    else: 
         X = np.array(X) # need to use array/list to index elements
    if n is None: n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int) # e.g. array([ 8410, 11437, 87128, ..., 75103,  5866, 44852])
    X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important

    if y is not None:
        labels = np.unique(y)
        n_labels = len(labels)
        assert len(y) == len(X)

        y_resample = np.array(y[resample_i]) 
        if all_labels_present: 
            n_labels_resample = len(np.unique(y_resample))
            while True:
                if n_labels_resample == n_labels: break 
                # need to resample again 
                if j > n_cycles: 
                    print('bootstrap> after %d cycles of resampling, still could not have all labels present.')
                    ac = collections.Counter(y)
                    print('info> class label counts:\n%s\n' % ac) 
                    break

                resample_i = np.floor(np.random.rand(n)*len(X)).astype(int) # e.g. array([ 8410, 11437, 87128, ..., 75103,  5866, 44852])
                X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important
                y_resample = np.array(y[resample_i]) 

                j += 1 
            # assert np.unique(y_resample) == n_labels

        return (X_resample, y_resample)

    return X_resample

def ci(scores, low=0.05, high=0.95):
    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 
    return (confidence_lower, confidence_upper)

def ci2(scores, low=0.05, high=0.95, mean=None):
    std = statistics.stdev(scores) 
    mean_score = np.mean(scores)  # bootstrap sample mean
    if mean is None: mean = mean_score

    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    middle = (confidence_upper+confidence_lower)/2.0  # assume symmetric

    print('ci2> mean score: %f, middle: %f' % (mean_score, middle))
    # mean = sorted_scores[int(0.5 * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 

    if confidence_upper > 1.0: 
        print('ci2> Warning: upper bound larger than 1.0! %f' % confidence_upper)
        confidence_upper = 1.0

    # this estimate may exceeds 1 
    delminus, delplus = (mean-confidence_lower, confidence_upper-mean)

    return (confidence_lower, confidence_upper, delminus, delplus, std)

def ci3(scores, low=0.05, high=0.95):
    if isinstance(scores[0], int): 
        scores = [float(e) for e in scores]
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    mean_score = np.mean(scores)  # bootstrap sample mean
    se = statistics.stdev(scores) # square root of sample variance, standard error

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 
    return (mean_score, se, confidence_lower, confidence_upper)

def ci4(scores, low=0.05, high=0.95):
    if isinstance(scores[0], int): 
        scores = [float(e) for e in scores]

    ret = {}
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    ret['mean'] = np.mean(scores)  # bootstrap sample mean
    ret['median'] = np.median(scores)
    ret['se'] = ret['error'] = se = statistics.stdev(scores) # square root of sample variance, standard error

    ret['ci_low'] = ret['confidence_lower'] = confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    ret['ci_high'] = ret['confidence_upper'] = confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 

    return ret

def sorted_interval_sampling(l, npar, reverse=False):
    """
    Arguments
    ---------
    npar: n partitions 
    """ 
    l.sort(reverse=reverse)
    avg = len(l)/float(npar)
    slist, partitions = [], []
    last = 0.0

    while last < len(l):
        partitions.append(l[int(last):int(last + avg)])
        last += avg    
    
    npar_eff = len(partitions) # sometimes 1 extra 
    # print('info> n_par: %d' % len(partitions))
    # print('\n%s\n' % partitions)
    for par in partitions:
        slist.append(random.sample(par, 1)[0])
        
    # 0, 1, 2, 3, 4, 5 => n=6, 6/2=3, 6/2-1=2 
    # 0, 1, 2, 3, 4    => n=5, 5/2=2 
    if npar_eff > npar: 
        assert npar_eff - npar == 1
        del slist[npar_eff/2]

    assert len(slist) == npar
    # for par in [l[i:i+n] for i in xrange(0, len(l), n)]: 
    #     alist.append(random.sample(par, 1)[0])
    return slist

# sampling with datastruct 
def sample_dict(adict, n_sample=10): 
    """
    Get a sampled subset of the dictionary. 
    """
    import random 
    keys = adict.keys() 
    n = len(keys)
    keys = random.sample(keys, min(n_sample, n))
    return {k: adict[k] for k in keys} 

def sample_subset(x, n_sample=10):
    if len(x) == 0: return x
    if isinstance(x, dict): return sample_dict(x, n_sample=n_sample)
    
    # assume [(), (), ] 
    return random.sample(x, n_sample)

def sample_cluster(cluster, n_sample=10): 
    """
    Input
    -----
    cluster: a list of cluster indices 
             e.g. 3 clusters 7 data points [0, 1, 1, 2, 2, 0, 0] 

    """
    n_clusters = len(set(cluster))
    hashtb = {cid: [] for cid in cluster}

    for i, cid in enumerate(cluster): 
        hashtb[cid].append(i)      # cid to positions        
 
    return sample_hashtable(hashtb, n_sample=n_sample)

def sample_hashtable(hashtable, n_sample=10):
    import random, gc, copy
    from itertools import cycle

    n_sampled = 0
    tb = copy.deepcopy(hashtable)
    R = tb.keys(); random.shuffle(R) # shuffle elements in R inplace 
    nT = sum([len(v) for v in tb.values()])
    print('sample_hashtable> Total keys: %d, members: %d' % (len(R), nT))

    if nT < n_sample:
        print('warning> size of hashtable: %d < n_sample: %d' % (nT, n_sample))
        n_sample = nT
    
    n_cases = n_sample 
    candidates = set()

    for e in cycle(R):
        if n_sampled >= n_cases or len(candidates) >= nT: break 
        entry = tb[e]
        if entry: 
            v = random.sample(entry, 1)
            candidates.update(v)
            entry.remove(v[0])
            n_sampled += 1

    return candidates

def t_ci(scores): 
    m, se, cil, cih = ci3(scores)
    print('mean: %f, stderr: %f, CI:(%f, %f)' % (m, se, cil, cih))
    return 

def t_resample(): 
    # X = np.array(range(10000)) # arange(10000)
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import roc_curve, auc

    mu, sigma = 1, 2 # mean and standard deviation
    X = np.random.normal(mu, sigma, 1000)
    nX = len(X)
    X_resample = bootstrap_resample(X, n=5000)
    nXs = len(X_resample)
    print('size of X: %d, Xs: %d' % (nX, nXs))
    print('X: %s' % X_resample[:10])
    print('original mean:', X.mean())
    print('resampled mean:', X_resample.mean())

    reg = np.logspace(-3, 3, 7)
    penalty = 'l1'
    clf0 = LogisticRegression(tol=0.01, penalty='l1')
    params = [{'C': reg }] 
    estimator = GridSearchCV(clf0, params, cv=10, scoring='roc_auc')

    print('> test resampling training examples')
    X, y = datasets.make_classification(n_samples=10000, n_features=20,
                                    n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    estimator.fit(X_train, y_train)
    probas_ = estimator.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    auc_score = auc(fpr, tpr)

    print('> prior to resample dim X_train: %s, dim y_train: %s => auc: %f' % (str(X_train.shape), str(y_train.shape), auc_score))

    scores = []
    Xr, yr = X_train[:], y_train[:]
    for i in range(30): 
        assert Xr.shape == X_train.shape and len(y_train) == len(yr)
        estimator = GridSearchCV(clf0, params, cv=10, scoring='roc_auc')
        estimator.fit(Xr, yr)
        probas_ = estimator.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        auc_score = auc(fpr, tpr)
        print('  + resampled => auc: %f' %  auc_score)
        scores.append(auc_score)
        Xr, yr = bootstrap_resample(Xr, yr)

    print('  + mean auc: %f, median auc: %f' % (np.mean(scores), np.median(scores)))

    return X_resample 

def t_resample2(**kargs): 
    from scipy.stats import kurtosis, skew, ks_2samp

    x = np.random.random(10)
    xr, xoob = bootstrap_resample(x, n=1000)

    print("> mean(x): {} vs mean(xr): {}\n".format(np.mean(x), np.mean(xr)))
    print("> skew |     {} ~=? {}".format(skew(x), skew(xr)))
    print("> kurtosis | {} ~=? {}".format(kurtosis(x), kurtosis(xr)))
    print("> x: {}".format(x))
    print("> xr: {}".format(xr[:20]))

    return

def test_bsr_mean():
    # test that means are close
    np.random.seed(123456)  # set seed so that randomness does not lead to failed test
    X = arange(10000)
    X_resample = bootstrap_resample(X, n=5000)
    assert abs(X_resample.mean() - X.mean()) / X.mean() < 1e-2, 'means should be approximately equal'

def t_weighted_sampling():
    x = [("WHITE",90), ("RED",8), ("GREEN",2)] 
    for i in range(100): 
        o = weighted_choice(x)
        print(o)

def t_interval_sampling(): 
    alist = range(0, 20)
    elements = sorted_interval_sampling(alist, 3)
    print('> sampled(n=%d):\n%s\n' % (len(elements), elements))

    return 

def t_cluster(): 
    cluster = [0, 1, 1, 2, 2, 0, 0, 2, 1]
    candidate_indices = sample_cluster(cluster, n_sample=4)
    candidates = ['_'] * len(cluster)
    for i in candidate_indices: 
        candidates[i] = 'x'

    cstr = ' '.join(str(e) for e in cluster)
    sstr = ' '.join(str(e) for e in candidates)

    print(cstr)
    print(sstr)

    return 

def test(): 
    # Xs = t_resample()
    # t_ci(Xs)

    # t_weighted_sampling()
    # t_interval_sampling()

    # resampling 
    # t_resample()
    t_resample2()

    # sample cluster
    # t_cluster()

    return

if __name__ == "__main__": 
    test()