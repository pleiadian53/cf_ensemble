# encoding: utf-8

import os
from pandas import DataFrame

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }
from utils_plot import saveFig

import seaborn as sns
import numpy as np 
from tabulate import tabulate

import scipy
from scipy.stats import kde
import scipy.sparse as sparse

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

# import contextlib
import warnings
warnings.filterwarnings("ignore")


# Matrix diagnosis utilties
######################################################
def box_sampler(arr, 
                loc_sampler_fn, 
                loc_dim_param, 
                loc_params, 
                shape_sampler_fn, 
                shape_dim_param,
                shape_params):
    """
    Extracts a sample cut from `arr`.

    Parameters
    ----------
    loc_sampler_fn : function
        The function to determine the where the minimum coordinate
        for each axis should be placed.
    loc_dim_param : string or None
        The parameter in `loc_sampler_fn` that should use the axes
        dimension size
    loc_params : dict
        Parameters to pass to `loc_sampler_fn`.
    shape_sampler_fn : function
        The function to determine the width of the sample cut 
        along each axis.
    shape_dim_param : string or None
        The parameter in `shape_sampler_fn` that should use the
        axes dimension size.
    shape_params : dict
        Parameters to pass to `shape_sampler_fn`.

    Returns
    -------
    (slices, x) : A tuple of the slices used to cut the sample as well as
    the sampled subsection with the same dimensionality of arr.
        slice :: list of slice objects
        x :: array object with the same ndims as arr

    Examples 
    --------
    # 1. A uniform cut on a 2D array with widths between 3 and 9:
    a = np.random.randint(0, 1+1, size=(100,150))
    box_sampler(a, 
                np.random.uniform, 'high', {'low':0}, 
                np.random.uniform, None, {'low':3, 'high':10})
    Output
    ------
    ([slice(49, 55, None), slice(86, 89, None)], 
     array([[0, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 0]]))

    # 2. Taking 2x2x2 chunks from a 10x20x30 3D array

    a = np.random.randint(0,2,size=(10,20,30))
    box_sampler(a, np.random.uniform, 'high', {'low':0}, 
                   np.random.uniform, None, {'low':2, 'high':2})
    # returns:
    ([slice(7, 9, None), slice(9, 11, None), slice(19, 21, None)], 
     array([[[0, 1],
             [1, 0]],
            [[0, 1],
             [1, 1]]]))

    Reference 
    ---------
    1. https://stackoverflow.com/questions/47373311/randomly-sample-sub-arrays-from-a-2d-array-in-python
    """
    if sparse.issparse(arr): arr = arr.A
    # [todo] Add out-of-bound adjustments

    slices = []
    for dim in arr.shape:
        if loc_dim_param:
            loc_params.update({loc_dim_param: dim})
        if shape_dim_param:
            shape_params.update({shape_dim_param: dim})
        start = int(loc_sampler_fn(**loc_params))
        stop = start + int(shape_sampler_fn(**shape_params))
        slices.append(slice(start, stop))
    return slices, arr[tuple(slices)]

def uniform_box_sampler(arr, min_width, max_width):
    """
    Extracts a sample cut from `arr`.

    Parameters
    ----------
    arr : array
        The numpy array to sample a box from
    min_width : int or tuple
        The minimum width of the box along a given axis.
        If a tuple of integers is supplied, it my have the
        same length as the number of dimensions of `arr`
    max_width : int or tuple
        The maximum width of the box along a given axis.
        If a tuple of integers is supplied, it my have the
        same length as the number of dimensions of `arr`

    Returns
    -------
    (slices, x) : A tuple of the slices used to cut the sample as well as
    the sampled subsection with the same dimensionality of arr.
        slice :: list of slice objects
        x :: array object with the same ndims as arr

    Examples
    --------
    # 1. Generate a box cut that starts uniformly anywhere in the array, 
        the height is a random uniform draw from 1 to 4 and the width is a random uniform draw from 2 to 6 (just to show). 
        In this case, the size of the box was 3 by 4, starting at the 66th row and 19th column.

    x = np.random.randint(0,2,size=(100,100))
    uniform_box_sampler(x, (1,2), (4,6))
    # returns:
    ([slice(65, 68, None), slice(18, 22, None)], 
     array([[1, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 0]]))
    """
    if sparse.issparse(arr): arr = arr.A

    if isinstance(min_width, (tuple, list)):
        assert len(min_width)==arr.ndim, 'Dimensions of `min_width` and `arr` must match'
    else:
        min_width = (min_width,)*arr.ndim
    if isinstance(max_width, (tuple, list)):
        assert len(max_width)==arr.ndim, 'Dimensions of `max_width` and `arr` must match'
    else:
        max_width = (max_width,)*arr.ndim

    # [todo] Add out-of-bound adjustments

    slices = []
    for dim, mn, mx in zip(arr.shape, min_width, max_width):
        fn = np.random.uniform
        start = int(np.random.uniform(0,dim))
        stop = start + int(np.random.uniform(mn, mx+1))
        slices.append(slice(start, stop))
    return slices, arr[tuple(slices)]


#######################################################
# KDE utilities

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)
    
def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

kde_funcs = [kde_statsmodels_u, kde_statsmodels_m, kde_scipy, kde_sklearn]
kde_funcnames = ['Statsmodels-U', 'Statsmodels-M', 'Scipy', 'Scikit-learn']

# print "Package Versions:"
# import sklearn; print "  scikit-learn:", sklearn.__version__
# import scipy; print "  scipy:", scipy.__version__
# import statsmodels; print "  statsmodels:", statsmodels.__version__
##############################################################################################################

def kde_pdf(data, kernel_func, bandwidth):
    """
    Generate kernel density estimator over data.

    Reference
    ---------
    1. http://www.jtrive.com/kernel-density-estimation-in-python.html

    Examples
    --------

    vals = [5, 12, 15, 20]

    eval_kde = kde_pdf(data=vals, kernel_func=uniform_pdf, bandwidth=1)

    # pass `eval_kde` points to evaluate =>
    eval_kde(10)    # 0
    eval_kde(15.5)  # .125
    eval_kde(20)    # .125

    """
    kernels = dict()
    n = len(data)
    for d in data:
        kernels[d] = kernel_func(d, bandwidth)
    def evaluate(x):
        """Evaluate `x` using kernels above."""
        pdfs = list()
        for d in data: pdfs.append(kernels[d](x))
        return(sum(pdfs)/n)
    return(evaluate)

def kde_cdf(data, kernel_func, bandwidth):
    """Generate kernel distribution estimator over data."""
    kernels = dict()
    n = len(data)
    for d in data:  # foreach sample point
        kernels[d] = kernel_func(d, bandwidth)

    def evaluate(x):
        """Evaluate x using kernels above."""
        cdfs = list()
        for d in data: cdfs.append(kernels[d](x))
        return (sum(cdfs)/n)
    return(evaluate)

def gaussian_kde(data, width=1, gridsize=100, normalized=True, bounds=None):
    """
    Compute the gaussian KDE from the given sample.

    Args:
        data (array or list): sample of values
        width (float): width of the normal functions
        gridsize (int): number of grid points on which the kde is computed
        normalized (bool): if True the KDE is normalized (default)
        bounds (tuple): min and max value of the kde

    Returns:
        The grid and the KDE

    Reference
    ---------
    1. https://gsalvatovallverdu.gitlab.io/python/kernel_density_estimation/

    """
    import scipy as sp
    from scipy.stats import gaussian_kde
    from scipy.stats import norm
    import pandas as pd
    
    # boundaries
    if bounds:
        xmin, xmax = bounds
    else:
        xmin = min(data) - 3 * width
        xmax = max(data) + 3 * width

    # grid points
    x = np.linspace(xmin, xmax, gridsize)

    # compute kde
    kde = np.zeros(gridsize)
    for val in data:
        kde += norm.pdf(x, loc=val, scale=width)

    # normalized the KDE
    if normalized:
        kde /= sp.integrate.simps(kde, x)

    return x, kde

# sliding window
def window_stack(a, stepsize=1, width=3):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )

def sample_kd0(kde):
    from scipy.optimize import brentq
    # sample
    u = np.random.random()

    # 1-d root-finding
    def func(x):
        return kde.cdf([x]) - u
    sample_x = brentq(func, -99999999, 99999999)
    return sample_x

def sample_kd(data, bandwidth, n_samples=10, kernel='gaussian', sample_weight=None, kde=None, random_state=None):
    """
    
    Reference
    ---------
    1. sklearn 
       https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/kde.py

    """
    import utils_sys as us 
    
    rng = us.check_random_state(random_state)
    u = rng.uniform(0, 1, size=n_samples)

    if sample_weight is None:
        i = (u * data.shape[0]).astype(np.int64)
    else:
        cumsum_weight = np.cumsum(np.asarray(sample_weight))
        sum_weight = cumsum_weight[-1]
        i = np.searchsorted(cumsum_weight, u * sum_weight)

    if kernel.startswith('gau'):
        # return np.atleast_2d(rng.normal(data[i], self.bandwidth))
        return rng.normal(data[i], bandwidth)

    raise NotImplementedError()

def fit_kd2(x, name='TP', kernel='gaussian', output_path=None, dpi=300, verbose=False, ext='pdf', 
        index=0, save=False, create_dir=True, color='blue', size=-1, cv=None, **kargs): 
    """

    Reference
    ---------
    1. implementation
       https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
       https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    2. KDE 
       https://mathisonian.github.io/kde/

        - statsmodels
            https://www.statsmodels.org/dev/examples/notebooks/generated/kernel_density.html

    3. random variables 
    
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

    Memo
    ----
    np.full_like(): Return a full array with the same shape and type as a given array.

    scatter()
       zorder: The default drawing order for axes is patches, lines, text.
    """
    def map_name(kernel): 
        if kernel.startswith('gau'):  # gaussian
            return 'gau'
        if kernel.startswith('epa'): # epanechnikov
            return 'epa'
        return kernel
    def get_label(dt='Histogram'):
        cls_name = kargs.get('cls_name', '')
        label = "{} from {} sample with {}".format(dt, name, cls_name) if cls_name else "{} from {} sample".format(dt, name)
        return label

    # from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeaveOneOut, KFold
    from scipy.stats import norm
    import statsmodels.api as sm
    import utils_sys as us

    # which library to use? 
    # library = 'statsmodels'  # statsmodels, sklearn

    msg = ''
    x = np.array(x)
    msg += "(fit_kd2) input dim(x): {}\n".format(x.shape)

    # down-sampling 
    if size > 0: x = np.random.choice(x, min(size, len(x)))
    if cv is None: 
        loo = LeaveOneOut()
        msg += "> number of splitting iterations: {}\n".format(loo.get_n_splits(x))
        cv = loo.split(x)

    # select bandwidth 
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    params = {'bandwidth': bandwidths}
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                        params,
                        cv=cv, iid=False)  # iid=False

    # x = x.reshape((len(x), 1))
    try: 
        grid.fit(x[:, None]);  # grid expect the data to have a 2D shape
    except Exception as e:
        kf = KFold(n_splits=10 if len(x) > 10 * 2 else 5) 
        cv = kf.split(x)
        grid = GridSearchCV(KernelDensity(kernel=kernel),
                        params,
                        cv=cv, iid=False)  # iid=False
        grid.fit(x[:, None])

    # best bandwith
    bandwidth = grid.best_params_['bandwidth']

    ################################################################
    # ... now optimal bandwith is determined 
    height = 6
    eps = 0.05
    xmin, xmax = 0.0-eps, 1.0+eps
    # x_d = np.linspace(xmin, xmax, 1000)

    kc = map_name(kernel)  # kernel
    kde = sm.nonparametric.KDEUnivariate(x)
    n_support = 2**10
    with us.stdout_redirected():
        kde.fit(kernel=kc, fft=False, gridsize=n_support)  # FFT is more efficient but only implemented for gaussian kernel

    ### plotting
    plt.clf() 

    plt.hist(x, bins=50, density=True, edgecolor='k', zorder=4, alpha=0.5)  # label=get_label(dt='Histogram')
    plt.plot(kde.support, kde.density, lw=3, zorder=7, label=get_label(dt='KDE'))

    plt.plot(x, np.full_like(x, -0.2), '|k', markeredgewidth=1)  # placing 'bars' for each sample point in x

    # Plot the samples
    # plt.scatter(x, np.abs(np.random.randn(x.size))/50,
    #        marker='x', color='red', zorder=20, label='Data samples', alpha=0.5)
    
    plt.legend(loc = 'best')
    plt.grid(True, zorder=-5)

    msg += "(fit_kd2) sample size: {},  n(support): {} | kernel: {} | ID: {}\n".format(len(x), len(kde.support), kernel, name)
    if verbose: print(msg)
    
    if save: 
        basedir = os.path.join(os.getcwd(), 'analysis')
        if not os.path.exists(basedir) and create_dir:
            print('(fit_kd) Creating analysis directory:\n%s\n' % basedir)
            os.mkdir(basedir) 

        if output_path is None: 
            if not name: name = 'generic'
            fname = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix='{}-kde'.format(kernel), suffix=name, index=index, ext=ext)
            output_path = os.path.join(basedir, fname)  # example path: System.analysisPath
        else: 
            # output_path can be either a file name OR a full path including the file name
            prefix, fname = os.path.dirname(output_path), os.path.basename(output_path)
            if not prefix: 
                prefix = basedir
                output_path = os.path.join(basedir, fname)
            assert os.path.exists(output_path), "Invalid output path: {}".format(output_path)

        if verbose: print('(fit_kd) Saving distribution plot at: {path}'.format(path=output_path))
        saveFig(plt, output_path, dpi=dpi, verbose=verbose) 
    else: 
        pass
        # plt.show()

    return kde 


def fit_kd(x, name='TP', kernel='gaussian', output_path=None, dpi=300, verbose=False, ext='pdf', 
        index=0, save=False, create_dir=True, color='blue', size=-1, cv=None, **kargs): 
    """

    Reference
    ---------
    1. implementation
       https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
       https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    2. KDE 
       https://mathisonian.github.io/kde/

        - statsmodels
            https://www.statsmodels.org/dev/examples/notebooks/generated/kernel_density.html

    3. random variables 
    
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

    Memo
    ----
    np.full_like(): Return a full array with the same shape and type as a given array.

    scatter()
       zorder: The default drawing order for axes is patches, lines, text.
    """
    def map_name(kernel): 
        if kernel.startswith('gau'):  # gaussian
            return 'gau'
        if kernel.startswith('epa'): # epanechnikov
            return 'epa'
        return kernel

    # from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeaveOneOut
    from scipy.stats import norm

    # which library to use? 
    # library = 'statsmodels'  # statsmodels, sklearn

    msg = ''
    x = np.array(x)
    msg += "(fit_kd) input dim(x): {}\n".format(x.shape)

    # down-sampling 
    if size > 0: x = np.random.choice(x, min(size, len(x)))

    msg += "(fit_kd) sample size: {}, kernel: {} | ID: {}\n".format(len(x), kernel, name)
    if verbose: print(msg)

    if cv is None: 
        loo = LeaveOneOut()
        print("> number of splitting iterations: {}".format(loo.get_n_splits(x)))
        cv = loo.split(x)

    # select bandwidth 
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    params = {'bandwidth': bandwidths}
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                        params,
                        cv=cv, iid=False)  # iid=False

    # x = x.reshape((len(x), 1))
    grid.fit(x[:, None]);  # grid expect the data to have a 2D shape

    # best bandwith
    bandwidth = grid.best_params_['bandwidth']

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(x[:, None])

    ### plotting
    plt.clf() 
    
    height = 6
    eps = 0.05
    xmin, xmax = 0.0-eps, 1.0+eps
    x_d = np.linspace(xmin, xmax, 1000)  

    logprob = kde.score_samples(x_d.reshape((len(x_d), 1))) # score_samples returns the log of the probability density
    probabilities = np.exp(logprob)

    plt.fill_between(x_d, probabilities, alpha=0.5, label='KDE from samples')   # 

    ### mark input samples (x)
    # a.
    plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)  # placing 'bars' for each sample point in x
    # b. scatter plot 
    #    ... does not visualize well with too large of a sample
    # plt.scatter(probabilities, np.abs(np.random.randn(probabilities.size))/40,
    #             zorder=20, color='red', marker='x', alpha=0.5, label='Samples')  # zorder: plotting order
    
    plt.ylim(-0.2, height)
    # plt.axis([xmin, xmax, -0.2, height]);

    # plot the histogram and pdf
    bw = 0.01   # bin width
    nbins = 100 # int(((xmax-xmin)/bw))
    # plt.hist(x, bins=nbins, zorder=5, density=True, edgecolor='k', label='Histogram from samples')
    # plt.plot(x_d[:], probabilities, lw=3, label='True distribution', zorder=15)

    plt.legend(loc='best')
    plt.grid(True, zorder=-5)

    if save: 
        basedir = os.path.join(os.getcwd(), 'analysis')
        if not os.path.exists(basedir) and create_dir:
            print('(fit_kd) Creating analysis directory:\n%s\n' % basedir)
            os.mkdir(basedir) 

        if output_path is None: 
            if not name: name = 'generic'
            fname = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix='{}-kde'.format(kernel), suffix=name, index=index, ext=ext)
            output_path = os.path.join(basedir, fname)  # example path: System.analysisPath
        else: 
            # output_path can be either a file name OR a full path including the file name
            prefix, fname = os.path.dirname(output_path), os.path.basename(output_path)
            if not prefix: 
                prefix = basedir
                output_path = os.path.join(basedir, fname)
            assert os.path.exists(output_path), "Invalid output path: {}".format(output_path)

        if verbose: print('(fit_kd) Saving distribution plot at: {path}'.format(path=output_path))
        saveFig(plt, output_path, dpi=dpi, verbose=verbose) 
    else: 
        pass
        # plt.show()

    return kde 

def fit_beta(x, name='TP', output_path=None, dpi=300, verbose=True, ext='pdf', 
        index=0, save=False, create_dir=True, color='blue'):
    """

    Memo
    ----
    1. https://github.com/scipy/scipy/issues/9754

    2. fitting a distribution 
       
       http://danielhnyk.cz/fitting-distribution-histogram-using-python/

    3. histogram features (using gaussian as an example)
       https://matplotlib.org/3.1.1/gallery/statistics/histogram_features.html

    """
    # distribution_func = getattr(stats, 'beta') 
    observations = x 
    sample_size = num_bins = len(x)
    a, b, loc, scale = scipy.stats.beta.fit(observations)
    print("(fit_beta) a: {} b: {}, loc: {}, scale: {}".format(a, b, loc, scale))

    plt.clf()
    # ax = plt.subplot(111)
    fig, ax = plt.subplots()
    
    # ax.plot(np.linspace(0, 1, 100), stats.beta.pdf(np.linspace(0, 1, 100), a, b))
    xmin, xmax = np.min(observations), np.max(observations)
    lnspc = np.linspace(xmin, xmax, len(observations))
    
    # the histogram of the data
    n, bins, patches = ax.hist(observations, alpha=0.75, color=color, bins=sample_size, density=True)  # 'normed' is a deprecated param for density

    # add density curve
    # plt.hist(observations, density=True)
    pdf_beta = scipy.stats.beta.pdf(lnspc, a, b, loc, scale)  
    plt.plot(lnspc, pdf_beta, label="beta")

    # customize x-label, y-label and title 
    ax.set_xlabel("Base predictor's probability scores")
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of fitted beta distribution (a: {}, b: {})'.format(a, b))

    if save: 
        basedir = os.path.join(os.getcwd(), 'analysis')
        if not os.path.exists(basedir) and create_dir:
            print('(fit_data) Creating analysis directory:\n%s\n' % basedir)
            os.mkdir(basedir) 

        if output_path is None: 
            if not name: name = 'generic'
            fname = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix='beta-distribution', suffix=name, index=index, ext=ext)
            output_path = os.path.join(basedir, fname)  # example path: System.analysisPath
        else: 
            # output_path can be either a file name OR a full path including the file name
            prefix, fname = os.path.dirname(output_path), os.path.basename(output_path)
            if not prefix: 
                prefix = basedir
                output_path = os.path.join(basedir, fname)
            assert os.path.exists(output_path), "Invalid output path: {}".format(output_path)

        if verbose: print('(fit_beta) Saving distribution plot at: {path}'.format(path=output_path))
        saveFig(plt, output_path, dpi=dpi) 
    else: 
        plt.show()
 
    return (a, b, loc, scale)

#############################################################################################
def is_binary(X):
    M = X
    if sparse.issparse(var): M = X.A
    return True if len(np.unique(M)) == 2 else False

def is_multivalued(X, max_n=10):
    """
    Check if a matrix is multi-valued (e.g. color matrix), containing at least 2 unique 
    elements but less than or equal to a maximum number `max_n` of unqiue values (why? 
    because more than this quantity, the matrix is likely an artibrary numeric-valued matrix)
    """ 
    M = X
    if sparse.issparse(var): M = X.A
    n = len(np.unique(M))
    return True if (n >= 2 and n <= max_n) else False

def is_sparse(X): 
    return True if sparse.issparse(X) else False

def analyze_matrix_type(*M, **K):
    # import scipy.sparse as sparse

    mats = {}
    if len(M) > 0: 
        for i in range(len(M)): 
            mats[i] = M[i]
    if len(K) > 0:
        for k, v in K.items(): 
            mats[k] = v 

    for varname, var in mats.items(): 
        msg = ''
        mtype = 'dense'

        mat = var
        if sparse.issparse(var): 
            mtype = 'sparse'
            mat = var.A # most matrices in our use cases are probably not very sparse

        msg += f"[info] Matrix {varname}: shape={mat.shape}, mtype={mtype}, dtype={type(var)} \n" 

        N = mat.size # np.prod(M.shape)
        n_zeros = np.sum(mat == 0.0) 
        msg += f"...    Number of zeros: {n_zeros}, ratio: {n_zeros/(N+0.0)}\n"
        
        # Simple statistics on non-zero elements
        m_nonzeros = np.mean(mat[mat != 0.0])
        med_nonzeros = np.median(np.mean(mat[mat != 0.0]))
        msg += f"...    Mean(non-zeros)={m_nonzeros}, median(non-zeros)={med_nonzeros}\n"

        # Is it a binary matrix? e.g. polarity matrix, probability filter (or preference matrix)
        unique_elements = np.unique(mat)
        nE = len(unique_elements)
        msg += f"...    `{varname}` is a binary matrix? {True if nE==2 else False}\n"
        msg += f"...    Number of unique elements: {nE}\n"

        if len(unique_elements) <= 10: 
            msg += f"...    Elements:\n{unique_elements}\n"

        print(msg); print('-' * 50)

    return 


#############################################################################################

# Create data: 200 points
def plot_relation(**kargs): 
    
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    x, y = data.T
     
    # Create a figure with 6 plot areas
    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))
     
    # Everything sarts with a Scatterplot
    axes[0].set_title('Scatterplot')
    axes[0].plot(x, y, 'ko')
    # As you can see there is a lot of overplottin here!
     
    # Thus we can cut the plotting window in several hexbins
    nbins = 20
    axes[1].set_title('Hexbin')
    axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)
     
    # 2D Histogram
    axes[2].set_title('2D Histogram')
    axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
     
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
     
    # plot a density
    axes[3].set_title('Calculate Gaussian KDE')
    axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)
     
    # add shading
    axes[4].set_title('2D Density with shading')
    axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
     
    # contour
    axes[5].set_title('Contour')
    axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[5].contour(xi, yi, zi.reshape(xi.shape) )

    return

def plot_confidence_matrix(C, X, L, p_th, U=[], n_max=100, measure='fmax', target_label=None, alpha=10, beta=2, index=0, 
        ext='pdf', verbose=1, dpi=300, file_id='', output_dir=None, output_file=None): 
    """

    Memo
    ----
    1. file naming convention: 

       e.g. heatmap 

            dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)
            fname = '{prefix}.P-{dataset}-{suffix}-{index}.{ext}'.format(prefix='similarity', dataset=dset_id, suffix=file_type, index=index, ext=ext)

    Reference
    ---------
    * See analysis-experiments.ipynb for plotting options ... 

    * binning a 2D array
       https://scipython.com/blog/binning-a-2d-array-in-numpy/

    * digitize
       https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html

    """
    import utils_cf as uc
    # from analyze_similarity import plot_heatmap

    plt.clf()
    if output_dir is None: 
        output_dir = './data'
    assert os.path.exists(output_dir), f"Invalid output directory:\n{output_dir}\n"

    if verbose: print('(plot_confidence_matrix) dim(C): {}'.format(C.shape))
    assert C.shape == X.shape
    # plot_heatmap(data=df, output_path=output_path, dpi=300, annot=tAnnotateSimDeg, mask_upper=False)

    # processing C 
    n_users, n_items = C.shape
    # print(dfX)

    # correctness matrix 
    Mc, Lh = uc.correctness_matrix(X, L, p_th, target_label=None) # X, pth -> Lh | L -> Mc
    # ... correctness matrix (Mc), label matrix (Lh)
    assert C.shape == Lh.shape

    # w_min = np.min(C)
    # assert all(C[~Mc] == w_min), "All confidence scores corresponding to FP and FN should have minimum weight w={} but got values: {}".format( w_min, np.random.choice(C[~Mc], 10) )

    # use Lh to select entries according to the target_label (e.g. positive only)
    if target_label is not None: 
        C[  ~(Mc & (Lh == target_label)) ] = 0  # all entries are suppressed as long as they are not correct or do not belong to the target label

    # random select columns 
    if n_max < n_items: 
        col_subset = np.random.choice(range(C.shape[1]), n_max)
        Csub = C[:, col_subset]
    else: 
        Csub = C

    plt.matshow(Csub, cmap=plt.cm.Blues); 
    # plt.colorbar()
    # plt.show()

    if not output_file: 
        if not file_id: file_id = "M{m}-a{a}b{b}".format(m=measure, a=alpha, b=beta)
        output_file = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix='confidence_matrix', suffix=file_id, index=index, ext=ext)
    output_path = os.path.join(output_dir, output_file)  # example path: System.analysisPath

    if verbose: print('(plot_confidence_matrix) Saving confidence matrix plot at: {path}'.format(path=output_path))
    saveFig(plt, output_path, dpi=dpi)

    return

def t_kde(**kargs):
    
    return 

def t_fit_beta(**kargs): 
    x = [0.294, 0.2955, 0.235, 0.2536, 0.2423, 0.2844, 0.2099, 0.2355, 0.2946, 0.3388,
        0.2202, 0.2523, 0.2209, 0.2707, 0.1885, 0.2414, 0.2846, 0.328, 0.2265, 0.2563,
        0.2345, 0.2845, 0.1787, 0.2392, 0.2777, 0.3076, 0.2108, 0.2477, 0.234, 0.2696,
        0.1839, 0.2344, 0.2872, 0.3224, 0.2152, 0.2593, 0.2295, 0.2702, 0.1876, 0.2331,
        0.2809, 0.3316, 0.2099, 0.2814, 0.2174, 0.2516, 0.2029, 0.2282, 0.2697, 0.3424,
        0.2259, 0.2626, 0.2187, 0.2502, 0.2161, 0.2194, 0.2628, 0.3296, 0.2323, 0.2557,
        0.2215, 0.2383, 0.2166, 0.2315, 0.2757, 0.3163, 0.2311, 0.2479, 0.2199, 0.2418,
        0.1938, 0.2394, 0.2718, 0.3297, 0.2346, 0.2523, 0.2262, 0.2481, 0.2118, 0.241,
        0.271, 0.3525, 0.2323, 0.2513, 0.2313, 0.2476, 0.232, 0.2295, 0.2645, 0.3386,
        0.2334, 0.2631, 0.226, 0.2603, 0.2334, 0.2375, 0.2744, 0.3491, 0.2052, 0.2473,
        0.228, 0.2448, 0.2189, 0.2149]
    x_tp = np.random.uniform(0.65, 0.95, 100)
    x_tn = np.random.uniform(0.07, 0.40, 100)
    x = np.hstack([x_tp, x_tn])

    fit_beta(x, save=True)

    kde = fit_kd(x, name='TN', kernel='epanechnikov', output_path=None, save=True, size=1000, cv=10)

    q = np.array([0.11, ])
    logprob = np.exp(kde.score_samples(q[:, None]))
    print('> logprob at {} = {}'.format(q, logprob))

    return

def test(**kargs): 
    # run_example(**kargs)

    # fit a probability distribution 
    t_fit_beta()

    return

if __name__ == "__main__": 
    test()