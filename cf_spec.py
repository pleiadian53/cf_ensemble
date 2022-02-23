import os, sys
import re
import utils_sys

# import logging
# import common
from pandas import DataFrame

### configurations
Domain = 'dummy'
ProjectPath = utils_sys.getProjectPath(domain=Domain, verify_=False) # default

### datasink  

# try: 
#     ProjectPath = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
#     Domain = os.path.basename(ProjectPath)
#     utils_cf.Domain = Domain
#     utils_cf.ProjectPath = ProjectPath
# except: 
#     pass 
# assert os.path.exists(ProjectPath)

class System(object): 
    domain = 'TBD'
    projectPath = os.getcwd() # getProjectPath(domain=domain, verify_=False) # default
    projectPrefix = prefix = os.path.dirname(projectPath) 
    analysisPath = os.path.join(projectPath, 'analysis')

    foldCount = 5 
    nestedFoldCount = 5
    configFile = 'config.txt'
    bagCount = 10

    stacker_method = 'standard'

    n_factors = None   # applicable to all MF methods; set to None to use the module default unless it's specified through the command line
    alpha = None      # applicable to WMF

    ### command line options
    options = None
    args = None

    ### experimental settings
    test_wmf_probs_suite = True 

    ### general info 
    startTime = '?'

    @staticmethod
    def config(project_path):   
        System.domain = os.path.basename(project_path)


def getProjectPath(domain='recommender/ml-100k', dataset='', verify_=False): 
    """

    dataset: a relative path to prefix hosting particular dataset associated with the given domain

    e.g. domain = 'pf2'
         data dir: /Users/galaxy/Documents/work/data
         => prefix = /Users/galaxy/Documents/work/data/pf2

         dataset <- /a/particular/protein_function
         => src_path = /Users/galaxy/Documents/work/data/pf2/a/particular/protein_function

    Memo
    ----
    1. domain, by default,  is a specific dataset directory under data/
    2. example project path (i.e. path to the data source where project configuration and classifier outputs are kept)
           /Users/<username>/Documents/work/data/recommender/ml-latest-small

    """
    import getpass
    user = getpass.getuser() # 'pleiades' 

    parentdir = os.path.dirname(os.getcwd())  # /sc/orga/work/chiup04/cluster_ensemble
    datadir = os.path.join(parentdir, 'data') 
    prefix = os.path.join(datadir, domain)

    # domain can be a relative path: e.g. recommender/ml-latest-small
    # prefix = '/Users/%s/Documents/work/data/%s' % (user, domain)  # /Users/pleiades/Documents/work/data/recommender
    
    # dataset can be a relative path to the 'prefix'
    src_path = prefix if not dataset else os.path.join(prefix, dataset)

    if verify_: 
        if not os.path.exists(src_path): raise RuntimeError ( "(getProjectPath) Invalid data source: %s" % src_path )

    return src_path


def config(**kargs):
    import utils_cf as uc
    import utils_als as ua
    from evaluate import PerformanceMetrics

    global ProjectPath, Domain

    if 'project_path' in kargs: 
        ProjectPath = uc.ProjectPath = System.projectPath = kargs['project_path'] 
    if 'domain' in kargs: 
        Domain = uc.Domain = System.domain = kargs['domain']
    if 'fold_count' in kargs: 
        System.foldCount = kargs['fold_count']
    
    if 'nested_fold_count' in kargs: 
        System.nestedFoldCount = kargs['nested_fold_count']
    else: 
        System.nestedFoldCount = System.foldCount  # use outer CV fold count as default

    if 'bag_count' in kargs: 
        System.bagCount = kargs['bag_count']

    PerformanceMetrics.set_path(prefix=System.projectPath) 
    MFEnsemble.set_path(prefix=System.projectPath)

### logging 
# create logger with 'spam_application'

# design
# class CFSystem(object): 
#     params = {}
#     params['domain'] = Domain 
#     params['project_path'] = ProjectPath

# Properties = common.load_properties(ProjectPath, config_file='config.txt')  # parse config.txt (instead of weka.properties)
# FoldCount = int(Properties['foldCount'])
# BagCount = int(Properties['bagCount']) if 'bagCount' in Properties else int(Properties['bags']) 

class MFEnsemble(object): 
    n_factors = 20    # subsumed by System.n_factors
    n_epochs = n_iter = 30  
    alpha_val = alpha = 100  # subsumed by System.alpha
    lambda_val = 0.8

    # method annotation
    delimiter = '_'

    # matrix data delimiter
    sep = ','
    n_max = 5000  # don't save matrix with n(rows) >= n_max

    log_dir = os.path.join(os.getcwd(), 'log')
    data_dir = os.path.join(os.getcwd(), 'data')

    # meta parameters
    meta_keys = [
        'supervised', 
        # 'masked', 
        # 'augmented', 
        'predict_probs', # prediction mode or replacement mode (i.e. use the latent factors to replace bad rating values in R and/or T)

        'conf_measure', 
        'policy',  # filtering dimension 'user', 'item'
        'policy_threshold', 
        'policy_opt', 
        'policy_replace', 
        'setting']  
    meta_value_default = default = {
                'conf_measure': 'brier', 
                
                'predict_probs': False,  # replace is the default 

                # 'policy': 'user',   # no default
                'policy_threshold': 'prior', 
                'policy_opt': 'rating',
                'policy_replace': 'rating', 

                # 'setting': 4, 
                'supervised': True, 
                'masked': True } 

    p_method = re.compile(r"(?P<method>[a-zA-Z0-9]+)(?P<specific>.*)_F(?P<n_factors>\d+)(_A(?P<alpha>\d+))?")
    segment_ids = ['F', 'A', 'X', 'XCF', 'CF', 'OPT', 'RE', 'PT', 'S']

    params_to_ids = {'n_factors': 'F', 'alpha': 'A', 'policy_opt': 'OPT', 'policy_replace': 'RE'}

    codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
            'unk': 0, 't': 3, 'f': -3}

    @staticmethod 
    def is_baseline(n_factors, alpha): 
        return n_factors == MFEnsemble.n_factors and alpha == MFEnsemble.alpha 

    @staticmethod
    def is_preference_data(name):
        # name.find('RE') < 0 to ensure that the preference score isn't used as a meta data to replace rating scores
        return name.find('pref') > 0 and name.find('RE') < 0

    @staticmethod
    def _name_tset(method, **params):
        name = '{method}'.format(method=method)  # base method (e.g. wmf, nmf)
        if params: 
            # if not MFEnsemble.is_baseline(params['n_factors'], alpha=params['alpha']): 
            if 'n_factors' in params and params['n_factors'] > 0: 
                name += '_F%s' % params.pop('n_factors')
            if 'alpha' in params and params['alpha'] > 0: 
                name += '_A%s' % params.pop('alpha')

            # everything else is meta info (e.g. mode, policy)
            if 'meta' in params and len(params['meta']) > 0: 
                name += '_X%s' % params.pop('meta') 

        return name
    @staticmethod
    def name_tset(method='wmf', params={}, meta_params={}): 
        """
        Used to distinguish different WMF-generated training data with different parameter settings. 

        cf: get_id() 
        
        e.g. In wmf_F10_A100_Xbrier_preference-validation-3.csv.gz

             wmf_F10_A100_Xbrier_preference is the dset_id (or training set ID)

        e.g. wmf_F10_A100_Xbrier_CFitem_RErating_PTprior-validation-1.csv.gz

        F, A, X: extension/meta 

        Memo
        ----
        1. Meta parametres (meta_params)

           meta_params do not go into the model selection loop for the following reasons: 
                1) if included, there'll be too many parameters to tune 
                2) these parameters may not play a key role in the performance or other research questions we care about
                3) we wish to analyze these parameter settings separately and independently

        2. how to utilize preference scores? 

        3. MF training data naming convention 

                dset_id provides a parameterized file ID that precedes {validation, prediction} in the training data file nameing
                
                tset stands for training set

                e.g. wmf_F10_A100_Xbrier_preference-validation-4.csv.gz     
                
                    prefix = wmf_F10_A100_Xbrier_preference
                    validation: combined 'test data' in the inner loop of the nested CV 
                    4: fold count 

                General form: 
                <dset_id>_{validation, prediction}_<fold>.csv.gz

        """
        ### naming convention
        #   key model parameters (e.g. n_factors, alpha), followed by meta parameters (algorithm descriptors e.g. confidence measure)

        ### A. meta-parameters
        # 1. meta parameters that typically are not considered in model tuning
        if len(meta_params) == 0: 
            # default = {'conf_measure': 'brier', 'policy': 'rating', 'supervised': True, 'masked': True} # masked: True
            for k in MFEnsemble.meta_keys: 
                if k in params: 
                    meta_params[k] = params[k]
                else: 
                    if k in MFEnsemble.default: 
                        meta_params[k] = MFEnsemble.default[k]
            # meta_params = {k: params.get(k, MFEnsemble.default[k]) for k in MFEnsemble.meta_keys}  # meta_params are a subset of params that do not go into model selection loop

        # meta_params include: 
        #     1. policy_opt
        #        policy_probs
        #        policy_replace 
        #     2. policy_threshold
        #     3. setting 
        # 
        #     4. auxiliary params (subsumed by 3)
        #        supervised 
        #        mask 
        #        augmented

        # >>> the choice of confidence matrix and its corresponding optimization objective may be specified separately
        policy_filter = policy_conf = meta_params['policy']
        policy_opt = meta_params['policy_opt'] if 'policy_opt' in meta_params else 'rating'

        # >>> two major top categories: the new estimate of the rating matrices, (Rh, Th), represent 'ratings' or preference scores? 
        # if policy_opt.startswith('pref') and not meta_params['predict_probs']: 
        #     if 'policy_replace' in meta_params: 
        #         # >>> using preference scores to select entries to keep and replace the rest with the reconstructed ratings using latent factors
        #         meta = '{confidence_type}_CF{policy}_RE{policy_replace}'.format(confidence_type=meta_params['conf_measure'], 
        #             policy=policy_filter, policy_replace=meta_params['policy_replace'])
        #     else: 
        #         meta = '{confidence_type}_CF{policy_conf}_OPT{policy_opt}'.format(confidence_type=meta_params['conf_measure'], 
        #             policy_conf=policy_filter, policy_opt=policy_opt)
        # else: # all the other cases, name the training data as usual
        #     if policy_opt == policy_filter: 
        #         meta = '{confidence_type}_CF{policy}'.format(confidence_type=meta_params['conf_measure'], policy=policy_filter) 
        #     else: 
        #         # the policy for constructing confidence matrix and for the optimization method are different
        # meta = '{confidence_type}_CF{policy_conf}_OPT{policy_opt}'.format(confidence_type=meta_params['conf_measure'], policy_conf=policy_filter, policy_opt=policy_opt)

        if policy_opt.startswith('trade'):  # trade-off between labels and probas
            meta = 'CF{policy_conf}_OPT{policy_opt}'.format(policy_conf=policy_filter, policy_opt=policy_opt) 
        else: 
            meta = 'CF{policy_conf}'.format(policy_conf=policy_filter)  # default is always 'rating'
           
        # if 'policy_threshold' in meta_params: meta = '%s_PT%s' % (meta, meta_params['policy_threshold']) # e.g. 'prior'

        # include the algorithm setting to distinguish combinatorial parameter differences 
        if 'setting' in meta_params: 
            meta = '%s_S%s' % (meta, meta_params['setting'])
        else: 
            msg = "Without the setting, the training set name may become ambiguous!"
            raise ValueError(msg) 

        #######################################################################################
        # Note: use the 'setting' parameter to summarize the following auxiliary parameters for simplicity 

        # # 2. supervised or unsupervised method (that determines Cui)? 
        # if not meta_params.get('supervised', True): meta = '{meta}_UnSpv'.format(meta=meta)
        
        # # 3. sparsify the confidence matrix? masking FP, FN (and missing data if applicable)
        # if not meta_params.get('mask', True): meta = '{meta}_dense'.format(meta=meta)

        # # 4. augmented? 
        # if not meta_params.get('augmented', True): meta = '{meta}_ReCons'.format(meta=meta) 

        #######################################################################################
        ### parameters (likely to be in the model selection loop)

        # B. MF model parameters (e.g. number of latent factors: n_factors)
        # assert len(params) > 0
        if len(params) == 0: 
            params = {'n_factors': 10, 'alpha': -1, }  # alpha only applicable to WMF 
            print('(name_tset) Warning: MF algorithmic parameters not specified. Use default:\n{0}\n'.format(params))

        dset_id = MFEnsemble._name_tset(method=method, n_factors=params.get('n_factors', 10), 
                    alpha=params.get('alpha', -1), meta=meta) # meta: extra info needed to differentiate training data
        return dset_id
    ##################################################
    #--- alias --- 
    get_dset_id = get_long_id = name_tset
    ##################################################

    @staticmethod
    def interpret_name(fn, sep='_', identifiers=[]): 
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
    @staticmethod
    def abridge(fn, identifiers, replace={}, sep='_', verify=True): 
        """
        Name the (training) data by only a subset of the identifiers. 

        Params
        ------
        replace: a dictionary that maps the original value (associated with an ID) to a new value. 

        Use
        ---
        analyze_performance module

        """
        # sort input identifiers 
        idmap = {e: i for i, e in enumerate(MFEnsemble.segment_ids)}  # where i indicates the ordering
        identifiers = sorted(identifiers, key=lambda e: idmap.get(e, -1), reverse=False)

        adict = MFEnsemble.interpret_name(fn, sep=sep)

        # print('(abridge) resulted name-value pairs: %s' % adict)
        id_values = []
        for ID in identifiers: 
            # if verify: assert ID in adict, "Unrecognized segment ID: {id}".format(id=ID)
            if not (ID in adict):
                if verify: print("Unrecognized segment ID: {id}".format(id=ID))
                continue
            if replace: adict[ID] = replace.get(adict[ID], adict[ID])  # replace whenever possible
            id_values.append("%s%s" % (ID, adict.get(ID, '')))

        return sep.join(id_values)

    # utils_cf
    @staticmethod
    def save_meta_tset(X, fold, params, method='wmf', verbose=1): # module: tset
        # import utils_cf as uc # [design] maybe it's more appropriate to define this routine in utils_cf? 

        # # MFEnsemble.meta_keys: ['conf_measure', 'policy', 'supervised', 'masked']
        # Rh, Th = X
        # meta_params = {k: params[k] for k in MFEnsemble.meta_keys}  # meta_params are a subset of params that do not go into model selection loop
        
        # # MFEnsemble.name_tset(method='wmf', params=params, meta_params=meta_params) 
        # dset_id = MFEnsemble.get_dset_id(method=method, params=params, meta_params=meta_params) # meta: extra info needed to differentiate training data
        
        # isAugmented = params.get('augmented', True)
        # if not isAugmented: 
        #     assert Th is None
        # else: 
        #     assert Th is not None
        #     print('(test) dim(Rh): {0}, dim(Th): {1}, n_train: {2}, n_test: {3}'.format(Rh.shape, Th.shape, L_train.size, L_test.size))
        
        # if verbose > 1: print('(save_meta_tset) Saving reconstructed data > Rh? {0}, Th? {1}  #'.format(Rh is not None, Th is not None))

        # # >>> (Rh, Th) represents new rating scores or not? 
        # isRating = True

        # if params['policy_opt'].startswith('pref') and not params['replace']: isRating=False
        
        # if verbose > 0: print('... (Rh, Th) represent new rating scores? {0}'.format(isRating))
        # uc.save_reconstructed_probs((Rh, Th), labels=(L_train, L_test), fold=fold, method=dset_id, verify=True, U=U, is_rating=isRating)
        raise NotImplementedError("Use the implementation in utils_cf module :)")

    @staticmethod
    def name_tsets(param_grid, meta_params={}):
        from sklearn.model_selection import ParameterGrid

        if not meta_params: meta_params = {'conf_measure': 'brier', 'policy': 'rating', 'supervised': True, 'policy_threshold': 'prior'}

        datasets = []
        for params in list(ParameterGrid(param_grid)): 
            # meta: extra info needed to differentiate training data 
            # 'target_method' is not to be confused with the combiner method specified by 'method', 
            datasets.append(MFEnsemble.name_tset(method=kargs.get('target_method', 'wmf'),  
                params=params, meta_params=meta_params)) 
        return datasets

    @staticmethod
    def name_method(method='wmf', kind='', params={}, meta=''):  # default values
        """
        Name the method derived from a given matrix factorization (e.g. wmf)
        """

        # use meta to distinguish among variations of the same algorithm ...
        # ... e.g. when using WMF as a means to construct similarity matrix, we set meta='sim' to distinguish itself from the method using latent factors alone

        # need to differentiate multiple parameter settings 
        name = method
        if kind: name = '{prefix}_{kind}'.format(prefix=name, kind=kind)  # base method + specific kind
        if meta: name = '{prefix}_{meta}'.format(prefix=name, meta=meta)  # e.g. meta='sim'

        if params: 

            # if not MFEnsemble.is_baseline(params['n_factors'], alpha=params['alpha']): 

            if 'n_factors' in params: 
                name += '_F%s' % params['n_factors']
            if 'alpha' in params: 
                name += '_A%s' % params['alpha']
            if 'conf_measure' in params: 
                name += '_CF%s' % params['conf_measure']
            if 'setting' in params: 
                name += '_S%s' % params['setting']
            # if 'lambda' in params: 
            #     name += '_L%s' % params['lambda'] 

            # name = '{basename}_F{n_factors}_A{alpha}'.format(basename=name, method=method, kind=kind, 
            #         n_factors=params['n_factors'], alpha=params['alpha'])

        return name
    ##################################################
    # -- alias --
    get_method_id = get_short_id = name_method
    ##################################################

    @staticmethod
    def toDict(name):
        pass 

    @staticmethod
    def isAMethodName(name):
        # todo 
        return MFEnsemble.p_method.match(name) is not None

    @staticmethod
    def name_sim_method(method, kind, params={}):
        return MFEnsemble.name_method(method=method, kind=kind, params=params, meta='sim') 

    @staticmethod
    def name_performance_metric(): 
        # see evaluate.PerformanceMetrics.my_shortname()
        pass 

    @staticmethod
    def name_performance_plot(method, metric, size, aspect='comparison', domain='test'):
        # assert isinstance(params, 'dict')

        file_name = '{method}_{metric}_{aspect}-N{size}-D{domain}'.format(method=method, 
                metric=metric, aspect=aspect, size=size, domain=domain)

        return file_name

    @classmethod
    def set_path(cls, prefix, basedirs=['log', 'data', ], create_dir=True):
        for basedir in basedirs:
            attr = '{base}_dir'.format(base=basedir)

            path = os.path.join(prefix, basedir)
            setattr(cls, attr, path)
            print('(MFEnsemble.set_path) attr %s -> %s | check: %s' % (attr, os.path.join(prefix, basedir), getattr(MFEnsemble, attr)))

            if not os.path.exists(path) and create_dir: 
                print('(MFEnsemble.set_path) Creating %s directory:\n%s\n' % (basedir, path))
                os.mkdir(path) 
        return

    @staticmethod
    def save_nmf_factors(P, Q, U, **kargs):
        # P: users, Q: items
        # n_max: save data only when number of instances is <= n_max
        assert P.shape[1] == Q.shape[1]   # n_factors consistent? 
        assert P.shape[0] == len(U)
        
        file_name = 'P_%dby%d.csv' % (P.shape[0], P.shape[1])
        if P.shape[0] <= MFEnsemble.n_max: 
            dfp = DataFrame(P.T, columns=U)  # format: rows <- latent factors, cols <- attributes 
            fpath = os.path.join(MFEnsemble.data_dir, file_name)
            dfp.to_csv(fpath, sep=MFEnsemble.sep, index=False, header=True)  # MFEnsemble.data_dir()
        
        file_name = 'Q_%dby%d.csv' % (Q.shape[0], Q.shape[1])
        if Q.shape[0] <= MFEnsemble.n_max: 
            cols = cols = ['item_%d' % i for i in range(Q.shape[0])]
            dfq = DataFrame(Q.T, columns=cols)  # format: rows <- latent factors, cols <- attributes 
            fpath = os.path.join(MFEnsemble.data_dir, file_name)
            dfq.to_csv(fpath, sep=MFEnsemble.sep, index=False, header=True)  # MFEnsemble.data_dir()   
        return
    @staticmethod
    def load_nmf_factors(**kargs):
        pass          

    @staticmethod
    def save_factors(A, **kargs):
        cols = kargs.get('cols', [])
        if len(cols) == 0: cols = ['item_%d' % i for i in range(A.shape[0])]
        assert A.shape[0] == len(cols)

        file_name = kargs.get('file_name', 'factors_D%d-%d.csv' % (A.shape[0], A.shape[1]))
        if A.shape[0] <= MFEnsemble.n_max: 
            df = DataFrame(A.T, columns=cols)  # format: rows <- latent factors, cols <- attributes 
            fpath = os.path.join(MFEnsemble.data_dir, file_name)
            
            if kargs.get('verbose', True): print('(save_factors) Saving input matrix of dim: {0} ...'.format(A.shape))
            df.to_csv(fpath, sep=MFEnsemble.sep, index=False, header=True)  # MFEnsemble.data_dir()
        return
    @staticmethod
    def load_factors(file_name, verbose=True):
        import pandas as pd
        fpath = os.path.join(MFEnsemble.data_dir, file_name)
        return pd.read_csv(fpath, sep=MFEnsemble.sep, header=0, index_col=False, error_bad_lines=True)
    @staticmethod
    def save_array(A, **kargs):
        """

        Use 
        ---
        1. save similarity matrix
        """
        # save 2d array to .npz format? 
        if kargs.get('format', 'csv'): 
            df = DataFrame(A)  # format: rows <- latent factors, cols <- attributes 

            file_name = kargs.get('file_name', '2darray_D%d-%d.csv' % (A.shape[0], A.shape[1]))
            fpath = os.path.join(MFEnsemble.data_dir, file_name)
            
            if kargs.get('verbose', True): print('(save_array) Saving input array of dim: {0} ...'.format(A.shape))
            df.to_csv(fpath, sep=MFEnsemble.sep, index=False, header=True)  # MFEnsemble.data_dir()
        else: 
            raise NotImplementedError

    @staticmethod
    def load_array(file_name, **kargs):
        import pandas as pd
        fpath = os.path.join(MFEnsemble.data_dir, file_name)
        A = pd.read_csv(fpath, sep=MFEnsemble.sep, header=0, index_col=False, error_bad_lines=True)
        if kargs.get('verbose', True): print('(load_array) Loading 2D array of dim: {0} ...'.format(A.shape))

        return A.values

### end class MFEnsemble



