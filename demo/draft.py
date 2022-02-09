

# common 
def save_reconstructed_probs(matrics, labels, fold, method, **kargs):
    isRating = kargs.get('is_rating', True) # (Rh, Th) are not rating scores but other meta data (e.g. preference scores)
    Rh = Th = None
    if isRating: 
        try: 
            Rh, Th = matrics
        except: 
            raise ValueError("matrics is a 2-tuple (R, T) but given: %s" % matrics)
    else: 
        if method.find('pref'):
            try: 
                R, Rh, T, Th = matrics
            except: 
                raise ValueError('In preference mode, matrices is a 4-tuple (R, R_pref, T, T_pref)')

    try: 
        L_train, L_test = labels
    except: 
        raise ValueError("labels is a 2-tuple (L_train, L_test) but given: %s" % labels)

    # dataframe index 
    indices = kargs.pop('indices', [])
    Rx = Tx = None
    if len(indices) > 0: 
        Rx, Tx = indices  # indices must a 2-tuple (or higher order tuples?)
        print('... (verify) unpacking indices:\n... Rx: {rx}\n... Tx: {tx}'.format(rx=Rx.names, tx=Tx.names))

    if not isRating: # if (Rh, Th) do not represent rating scores, then method must have related keyword 
        
        if method.find('pref'): # treat (Rh, Th) as preference scores
            # may need to pad additional feature set (e.g. using preferences as indicator features)
            if Rh is not None: 
                if Rx is not None: kargs['index'] = Rx
                save_preference_training_data(Rh, L_train, fold, method, **kargs)
            if Th is not None: 
                if Tx is not None: kargs['index'] = Tx
                save_preference_test_data(Th, L_test, fold, method, **kargs)
        else: 
            msg = 'Unrecognized method (where Rh, Th do not represent ratings): %s' % method 
            raise ValueError(msg)
    else: 
        if Rh is not None: 
            if Rx is not None: kargs['index'] = Rx
            save_reconstructed_training_data(Rh, L_train, fold, method, **kargs)
        if Th is not None: 
            if Tx is not None: kargs['index'] = Tx
            save_reconstructed_test_data(Th, L_test, fold, method, **kargs) 

    return

def save_preference_training_data(R, L_train, fold, method, **kargs):
    import pandas as pd
 
    tRandomSubsampling = kargs.get('subsampling', False)
    if tRandomSubsampling: 
        # in subsampling mode, we cannot use read_fold() to recover the labeling
        index = kargs.get('index', None)
        if index is None: 
            index = pd.MultiIndex.from_tuples([(i, L_train[i]) for i in range(len(L_train))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new pref-TRAIN set (method: %s)' % method)
        assert hasattr(R, '__iter__') and len(R) == 2, "Need both rating and preference matrices"
        Rt, R_pref = R

        assert 'U' in kargs
        augmented_cols = list(U) + ['t_%s' % c for c in U]
    else: 
        R_pref = R
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
        index = train_df.index
        # level-1 training data: validation-<fold>.csv.gz

        Rt = train_df.values  # R: users vs items but train_df has users as columns
        augmented_cols = list(train_df.columns) + ['t_%s' % c for c in train_df.columns]
    # T = test_df.values.T 
    # U = train_df.columns.values

    # R, R_pref = R_pair
    # Ra = np.vstack((R, R_pref))  # users/classifier followed by indicators
    augmented_cols = list(train_df.columns) + ['t_%s' % c for c in train_df.columns]
    df = DataFrame(np.hstack( (Rt, R_pref.T) ), index=index, columns=augmented_cols)  # datasink convention> rows: data points, columns: classifiers

    # save 
    assert method.find('pref') > 0, "Questionable method: %s" % method

    # >>> cannot use 'fold' as an index (because then the resulting set may not have continous indices)
    df_path = '%s/%s-validation-%s.csv.gz' % (ProjectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving preference-augmented TRAIN set (dim: {0}) to:\n{path}\n'.format(df.shape, path=df_path))

    return

def save_preference_test_data(T, L_test, fold, method, **kargs): 
    import pandas as pd 
    if kargs.get('subsampling', False): 
        # in subsampling mode, we cannot use read_fold() to recover the labeling
        index = kargs.get('index', None)
        if index is None: 
            index = pd.MultiIndex.from_tuples([(i, L_test[i]) for i in range(len(L_test))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new pref-TEST set (method: %s)' % method)
        assert hasattr(T, '__iter__') and len(T) == 2, "Need both rating and preference matrices"
        Tt, T_pref = T

        assert 'U' in kargs
        augmented_cols = list(U) + ['t_%s' % c for c in U]

    else: 
        T_pref = T
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
        index = test_df.index
        
        Tt = test_df.values 
        augmented_cols = list(test_df.columns) + ['t_%s' % c for c in test_df.columns]

    df = DataFrame(np.hstack( (Tt, T_pref.T) ), index=index, columns=augmented_cols)  # datasink convention> rows: data points, columns: classifiers
    print('(save) new test set dim: {0} vs original: {1}'.format(df.shape, test_df.shape))

    df_path = '%s/%s-predictions-%s.csv.gz' % (ProjectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving preference-augmented TEST set (dim: {dim}) to:\n{path}\n'.format(dim=df.shape, path=df_path))

    return

# common
def save_reconstructed_training_data(Rh, L_train, fold, method, **kargs):
    """

    Memo
    ----
    1. MultiIndex: 
       
       L_train
       index = [(i, L_train[i]) for i in range(len(L_train))] 
       pd.MultiIndex.from_tuples(index)

    """
    import pandas as pd
    tRandomSubsampling = kargs.get('subsampling', False)
    if tRandomSubsampling: 
        index = kargs.get('index', None)
        if index is None: 
            # in subsampling mode, we cannot use read_fold() to recover the labeling
            index = pd.MultiIndex.from_tuples([(i, L_train[i]) for i in range(len(L_train))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new TRAIN set (method: %s)' % method)
        else: 
            # if 'names' is not part of the index attribute => FrozenList([None])
            assert index.names[0] is not None

        if not 'U' in kargs: 
            train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, 0) # [todo] single out this part
            cols = train_df.columns
        else: 
            cols = kargs['U']
    else:
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, 0) # [todo] single out this part
        index = train_df.index
        cols = train_df.columns 

        # verify 
        if kargs.get('verify', True):
            # train_df = train_df.reset_index() # convert multilevel index to flat index
            labels = train_df.index.get_level_values('label').values # ground truth labels
            assert all(L_train == labels)

            if 'U' in kargs: 
                assert all(kargs['U'] == train_df.columns.values)
    
    # level-1 training data: validation-<fold>.csv.gz
    df = DataFrame(Rh.T, index=index, columns=cols)  # datasink convention> rows: data points, columns: classifiers

    # save 
    df_path = '%s/%s-validation-%s.csv.gz' % (ProjectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving new training set (dim: {dim}) to:\n{path}\n'.format(dim=df.shape, path=df_path))

    return

# common
def save_reconstructed_test_data(Th, L_test, fold, method, **kargs):
    import pandas as pd
    tRandomSubsampling = kargs.get('subsampling', False)
    if tRandomSubsampling: 
        index = kargs.get('index', None)
        if index is None: 
            # in subsampling mode, we cannot use read_fold() to recover the labeling
            index = pd.MultiIndex.from_tuples([(i, L_test[i]) for i in range(len(L_test))])
            index.names = kargs.get('index_names', ['id', 'label'])
            print('(verify) created new index for the new TEST set (method: %s)' % method)
        else: 
            # if 'names' is not part of the index attribute => FrozenList([None])
            assert index.names[0] is not None

        if not 'U' in kargs: 
            train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, 0) # [todo] single out this part
            cols = test_df.columns
        else: 
            cols = kargs['U']
    else:
        train_df, train_labels, test_df, test_labels = common.read_fold(ProjectPath, fold) # [todo] single out this part
        index = test_df.index
        cols = test_df.columns

        # verify 
        if kargs.get('verify', True):
            # test_df = test_df.reset_index() # convert multilevel index to flat index
            labels = test_df.index.get_level_values('label').values
            assert all(L_test == labels)

            if 'U' in kargs: 
                assert all(kargs['U'] == test_df.columns.values)

    # level-1 training data: validation-<fold>.csv.gz
    df = DataFrame(Th.T, index=index, columns=cols)  # datasink convention> rows: data points, columns: classifiers
    
    # save   # note that regular level-1 test data has the name: predictions-<fold>.csv.gz
    df_path = '%s/%s-predictions-%s.csv.gz' % (System.projectPath, method, fold)
    df.to_csv(df_path, compression='gzip')
    print('(verify) saving new TEST set (dim: {0}) to:\n{path}\n'.format(df.shape, path=df_path))

    return 


    if not params['policy'].startswith('trade'):  # <<< any policies other than 'tradeoff' using this procedure
        # verify > (determine_iter_routine) Use the prediction-label tradeoff routine ...'
        Pr, Qr = ua.implicit_als(Cr, ratings=R, labels=L_train,
                    iterations=params['n_iter'], features=params['n_factors'], 
                        policy=params['policy_opt'], message=piggyback_msg)
    else: # tradeoff between R and L
        # Cui, Cui_bar are not in sparse format
        assert Cr_bar is not None
        Pr, Qr = ua.implicit_als_johnson(Cr, label_confidence=Cr_bar, ratings=R, labels=L_train,
                        iterations=params['n_iter'], features=params['n_factors'], 
                            policy=params['policy_opt'], message=piggyback_msg)

# subsampling 

def subsample(data, users=[], ratio=0.5, max_train=None, max_test=None, replace=False):
    # from common import subsample_array

    n_tuple = len(data)
    assert n_tuple >= 4, "Input data is either a 4-tuple or 5-tuple"
    if n_tuple == 4: 
        R, T, L_train, L_test = data
        U = kargs.get('users', np.array(['u%d' % i for i in range(R.shape[0])]) )
    else: 
        R, T, L_train, L_test, U = data 

    n_train, n_test = R.shape[1], T.shape[1]
    rtt = n_train/(n_test+0.0)  # train-to-test ratio 

    if not max_train: 
        # use ratio
        max_train = int( np.ceil(n_train * ratio) )
    if max_train > n_train: max_train = n_train

    indices = np.random.choice(n_train, max_size, replace=replace)

    return

# solution 1
def nmf_ensemble(base_perf=None, **kargs): 
    """

    Memo
    ----
    1. related modules: 
        selection 


    """
    from evaluate import Metrics, plot_roc
    from evaluate import analyzePerfStacker
    import utils_cf as uc

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    
    params = {}
    params['n_factors'] = n_factors = kargs.get('n_factors', MFEnsemble.n_factors)
    params['n_epochs'] = kargs.get('n_epochs', MFEnsemble.n_epochs)

    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    if base_perf is None: 
        base_perf = base_predictors() # consolidated PerformanceMetrics object across CV fold

    # bpMetrics, mfMetrics = Metrics(), Metrics() # matrix factorization metrics
    perfMetrics = []

    # topk = 30
    method = 'nmf'
    kinds = ['mean_aggregate', ] 
    stackers = ['enet', 'rf', ]  # 'cf_stacker_{0}'.format(kind)
    kinds = kinds + stackers

    nmfMetrics = {k: [] for k in kinds}
    nmfCV = {k: [] for k in kinds}

    # keep track of reproduced probabilities
    nmfStacker = {stacker:[] for stacker in ['enet', 'rf', ]} 

    for fold in range(n_fold): 
        # different than toPredictiveScores(), toRatingMatrix() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        R, T, L_train, L_test, U = uc.toRatingMatrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        n_users, n_items = R.shape[0], R.shape[1]
        # print('[nmf_ensemble] dim(T): %s, n_test: %d' % (str(T.shape), len(L_test)))

        # [note] Surprise does not take inputs in the form of R (rating matrix)
        P, Q = uc.applyMF(fold, n_factors=params['n_factors'], n_epochs=params['n_epochs'], fill=missing_value)  # P and Q
        print('(nmf_ensemble) Fold: {0} | dim(P): {1}, dim(Q): {2}'.format(fold, P.shape, Q.shape))

        Rh, Th = uc.predict_by_factors(P, Q, test_offset=len(L_train), test_set_only=True) # Rh <- None if test_set_only: True
        if kargs.get('save', True): 
            uc.save_reconstructed_training_data(Rh, L_train, fold, method, verify=True, U=U)
            # uc.save_reconstructed_test_data(Th, L_test, fold, method, verify=True, U=U)

        for kind in kinds: 

            # metrics = compareEstimates0(T, L_test, Th=Th, R=None, L_train=None)

            # optinal params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
            method_specific = MFEnsemble.name_method(method, kind, params=params)  # '{method}_{kind}'.format(method=method, kind=kind)
            
            # also consider stacking on top of the reproduced probabilities
            if kind in stackers: # is a kind of stacker => need special performance Handler
                
                # **kargs: classifier hyperparams
                perf, df_prediction = analyzePerfStacker(fold, Rh, Th, method=method_specific)  # run stacker on top of the reproduced probabilities
                y_true, y_score = df_prediction['label'], df_prediction['prediction']
                assert all(L_test == y_true)
                nmfMetrics[kind].append(perf)
                nmfCV[kind].append((L_test, y_score))

            else:  # regular kind 
                nmfMetrics[kind].append(analyzePerf(L_test, Th, method=method_specific, aggregate_func=np.mean, T=T, fold=fold))  # analyzePerf -> { compare* } where compare* is a set of analysis functions (e.g.  compareEstimates_
                nmfCV[kind].append( (L_test, uc.combiner(Th, aggregate_func=np.mean)) )


        # bpMetrics.add(metrics['bp'])
        # mfMetrics.add(metrics['cf'])

    ## evaluation 
    for kind in kinds: 
        method_specific = MFEnsemble.name_method(method, kind, params=params)
        plot_roc(nmfCV[kind], file_name='roc-{method}-{project}'.format(method=method_specific, project=Domain))  # an import from evaluate
    
    # Q1: does the reconstructed prob "better? 

    perfAll = PerformanceMetrics.merge([PerformanceMetrics.consolidate(nmfMetrics[kind]) for kind in kinds]) # merge all CV-consolidated PerformanceMetrics objects

    # ret: output dictionary
    #      key: {methods}, {metrics}
    ret['metrics'] = perfAll  # foreach metric, take average over CV folds
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(perfAll, metric=metric, verbose=True if metric in ['fmax', ] else False) 

    # how does it compare to BP? 
    docs = {'method': method}
    PerformanceMetrics.report(p_baseline=base_perf, p_target=perfAll, rule='max-all', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return perfAll



def toImplicitUserItem(fold, to_surprise_format=True, merge_=False, save_=False, fill=0): 
    """
    Convert level-1 training data to user-item dataframe format consisting of the following attributes (columns): 

    ['user_id', 'item_id', 'prediction', 'label'], 

    where user_id corresponds to classifiers 
          item_id corresponds to data points 


    Params
    ------
    fill: placeholder for missing values (e.g. used when masking FPs and FNs)

    """
    def to_label(y_hat, p_threshould=0.5): 
        labels = np.zeros(len(y_hat))
        for i, p in enumerate(y_hat): 
            if p >= p_threshould: 
               labels[i] = 1
        return list(labels)
    import pandas as pd
    import scipy.sparse as sparse
    import common

    inputdir = kargs.get('inputdir', None)
    if inputdir is None: inputdir = ProjectPath
    train_df, train_labels, test_df, test_labels = common.read_fold(inputdir, fold) # [todo] single out this part

    # get all data IDs 
    users_ref = train_df.columns.values
    
    splits = ['train', 'test', ] 
    dataset = [train_df, test_df, ]

    # treat classifiers as users, data points as items 
    # dataframe format: 
    #   user_id, item_id, rating
    header = ['user_id', 'item_id', 'prediction', 'label' ]
    pth = kargs.get('p_threshould', 0.5)

    D = []
    for split, ts in zip(splits, dataset): 
        users = ts.columns.values

        ts = ts.reset_index() # convert multilevel index to flat index
        
        idx = ts['id'].values
        assert len(idx) == len(set(idx)), "Data IDs are not unique!"
        labels = ts['label'].values

        assert set(users) == set(users_ref)

        nU = nUsers = len(users) # number of users/classifiers
        nI = nItems = len(idx)  # number of items/data points

        adict = {h: [] for h in header}
        for i, user in enumerate(users_ref): 
            y_score = ts[user].values
            # y_label = to_label(y_score, p_threshould=p_th) 

            adict['user_id'].extend([user] * nI) # the same user predicts nI items
            adict['item_id'].extend(idx)
            adict['prediction'].extend(y_score)
            adict['label'].extend(labels)
            # adict['preference'].extend(y_label)

        ## mask FP, FN    
        ts = DataFrame(adict, columns=header)   # level 1 training data
        nE0= ts.shape[0]
        # print('(toUserItem) sample set=%s | n_ids: %d, n_users: %d, dim(ts): %s' % (split, len(idx), nU, str(ts.shape)))
        # print('... ts(n=5):\n%s\n' % ts.head(5))

        # drop FP, FN
        # data.loc[data.plays != 0]
        cond_tp = (ts['prediction'] >= p_threshold) & (ts['label']==1)
        cond_tn = (ts['prediction'] < p_threshold) & (ts['label']==0)
        ts = ts.loc[cond_tp & cond_tn]
        nE = ts.shape[0]
        print('(test) TP+TN has %d entries, total %d => a reduction of: %d' % (nE, nE0, nE0-nE))
        
        # to sparse rating/preference matrix format 
        rows = ts.user_id.astype(int)
        cols = ts.item_id.astype(int)

        # if save_: 
        #     l1_data_path = os.path.join(src_path, 'level1')
        #     fpath = os.path.join(l1_data_path, 'cf-%s-f%d-b%d.csv' % (split, fold, bag_count))  # naming: test-b3-f1-s1.csv.gz
        #     print('(toUserItem) Saving level-1 CF %s set (dim=%s) to .csv: %s' % (split, str(ts.shape), fpath))
        #     ts.to_csv(fpath, sep=delimit, index=False, header=True)
        #     # data = Dataset.load_from_file(fpath, reader=reader, rating_scale=(0, 1))
        D.append(ts)
    ### end foreach split

    # [design] may be more convenient to return tuple consisting (combined data, training split, test split)
    #          because we don't want to get test split via train_test_split, instead we have a predefined test split
    return ret 