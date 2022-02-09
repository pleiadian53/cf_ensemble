



### toConfidenceMatrix with polarity modeling but this is too complex

def toConfidenceMatrix(X, L, **kargs):
    """
    Compute confidence matrix. First estimate an appropriate probability filter (M) for X, where 
    reliable probabilities are marked by 1 and unreliable probabilities are marked by 0

    Then, we ompute the confidence scores for each prediction in X given the confidence measure (conf_measure, e.g. 'brier') to form
    an initial confidence matrix C0

    The final confidencee matrix is an element-wise product between M and C0, which allows us to encourage the latent factor 
    to give higher weight on reliable probabilties while giving 0 or negative weights on unreliable probablities.

    Probabilty filter or mask is determined by at least 4 factors (see Memo). 

    **kargs
    -------
    policy: used to define the mask function; specfies the filtering dimension: 'user', 'item' 
            also see 'ratio_small_class', 'ratio_users', conf_measure

    fill: a marker for unreliable rating values, or for missing values; default 0

    pos_label

    p_threshold: a vector of proba thresholds of the same size as X.shape[0], representing the number of users/classifiers; 
                 used to to estimate confidence scores in mode = 'ratio'
    policy_threshold 
    ratio_small_class

    conf_measure: 'brier', 'ratio', 'corr', 
    scoring: scoring function for computing confidence scores

    ratio_users: used as a parameter for the mask function in item-centered mode (i.e. filtering along the axis of the items/data) 

    U
    L_true: ground truth labels (for testing only)

    sparse: 
    alpha

    Similar to toConfidenceMatrix() but returns both Cui and its complement

    Memo
    ----
    1. mask function is determined by: 
        a. filtering dimension: along the user/classifier axis (policy='user'), or along the item/data axis (policy='item')

           - [update] perhaps we should always consider 'item' dimension

        b. policy_threshold: how probability thresholds are determined in order to estimate the labeling, which then allows for 
                            the determination of masked entries representing unreliable conditional probability estimates (or in general any ratings in the entries of X)
        
        [update] c and d (unsupervised) are not favorable

        c. depending on the true labels (L) being given or not
           we'll need
             ratio_small_class, when L isn't given in order to estimate proba thresholds
        d. ratio_users: for each item/data point, the ratio of user ratings (classifier probabilities) to consider as sufficiently reliable rating esitmates
                        only used in item-centered mode (policy='item') i.e. filtering along the axis of data

    2. balancing skewed classes
       ref: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/


    """
    def to_label(R, labels, p_th=0.5): 
        P = np.zero((R.shape[0], R.shape[1]))
        P[np.where(R >= p_th)] = 1
        return P
    def mask_fp_fn(R, labels, p_th=0.5, pos_label=1, neg_label=0, labelize=True):  # rule: keep only TP, TN
        # convert to sparse repr while masking FP and FN
        
        # label predictions
        P = to_label(R, labels, p_th=p_th)
         
        cond_tp = (R >= p_th) & (labels == pos_label)
        cond_tn = (R < p_th) & (labels == neg_label)
        rows, cols = np.where(cond_tp | cond_tn)
        
        good_pred = P[rows, cols] if labelize else R[rows, cols]
        # good_probs = R[rows, cols]  # R[np.where(cond_tp | cond_tn)]
        
        nU, nI = R.shape[0], R.shape[1]
        S = sparse.csr_matrix((good_pred, (rows, cols)), shape=(nU, nI))
        return S 
    def mask(R, labels, p_th=0.5, pos_label=1, neg_label=0, rule='false_prediction'):

        # mask rule
        cond_tp = (R >= p_th) & (labels == pos_label)
        cond_tn = (R < p_th) & (labels == neg_label)

        # entries
        rows, cols = np.where(cond_tp | cond_tn)
        return (rows, cols)

    def show_thresholds(p_th, users=[]): # closure: p_threshold
        policy_threshold = kargs.get('policy_threshold', 'prior')
        sort_by = {0: 'classifer', 1: 'probability'}
        if len(users) == 0: 
            print('... probability thresholds:\n.....{0}\n'.format(p_threshold))
        else: 
            sid = 0 # by classifier or by probablity values
            div("(toConfidenceMatrix) List of proba thresholds given policy: {p} | sorted according to -- {v} -- ".format(p=policy_threshold, v=sort_by[sid]))
            pth_sorted = sorted(zip(users, p_th), key=lambda x: x[sid])   
            for i, (u, pth) in enumerate(pth_sorted): 
                print('... [%d] %s: p_th = %f' % (i+1, u, pth))
    def scan_and_update(C, L, min_class=None, multiple=None):
        if None in (min_class, multiple): # not known a priori 
            ret = classPrior(L, labels=[0, 1], ratio_ref=0.1, verbose=False)
            multiple = ret['n_max_class']/ret['n_min_class']
            min_class = ret['min_class']
        idx = np.where(L==min_class)[0]  # column-wise positional indices of miniroty class
        C[:, idx] = C[:, idx] * multiple  # only effect on subset of 'idx' that are unmasked (not zeros)
        print('(scan_and_update) weights of minority class multiplied by {m} | len(idx): {n} | class stats: {o}'.format(m=multiple, 
            n=len(idx), o=ret))
        # return C    # not necessary
    def unmask(C, L, U=[], test_cases=[], min_class=1, max_class=0):
        # for the test split, since labels are only estimated, maybe we do not want to be overly confident in masking the "wrong" predictions 
        n_users, n_items = C.shape

        global_min_weight = np.min(C[C>0])
        print('(unmask) global min confidence weight: {}'.format(global_min_weight))
        for i in range(n_users): 
            user_name = U[i] if len(U) > 0 else i
   
            idx_max_class_correct = np.where( (L==max_class) & (C[i] > 0) )[0]
            if len(idx_max_class_correct) == 0: 
                min_weight = global_min_weight 
            else:
                # row min 
                min_weight = np.min(C[i][idx_max_class_correct])

            idx_masked = np.where( C[i] == 0 )[0]
            C[i, idx_masked] = min_weight

            if i in test_cases: 
                print('(unmask) classifier=[{}] | min weight: {}'.format(user_name, min_weight))

        assert np.sum(C[C==0]) == 0
        return

    def identify_effective_users(Cr, L_train, fill=0): 
        n_users = Cr.shape[0]
        effective_states = np.zeros(n_users)
        for i in range(n_users):
            # non_masked_positive = (Cr[i] > fill) & (L_train == 1)
            n_non_masked = np.sum(Cr[i] > fill)
            if n_non_masked > 0: 
                effective_states[i] = 1
        return effective_states
    def regulate_weights(C, beta=10, U=[], suppress_max_class=False, 
            min_class=1, max_class=0, bag_count=10, test_cases=[]):
        if beta <= 1: 
            # no-op 
            return  # C would've been modified in place

        if len(test_cases) == 0: test_cases = np.random.choice(range(C.shape[0]), 5)

        adict = {0: 'Negative', 1: 'Positive', }
        sample_weights = {}
        sample_weights_max_class = {}
        tns, tps = {}, {}

        w_min = np.min(C[C > 0])   # note: np.min(C > 0) -> False

        for i in range(C.shape[0]): 
            uname = U[i] if len(U) > 0 else i
            # idx = np.where(L == min_class)[0]
            idx_min_class_correct = np.where((L == min_class) & (C[i] > 0))[0]
            idx_max_class_correct = np.where((L == max_class) & (C[i] > 0))[0]

            n_tmin, n_tmax = len(idx_min_class_correct), len(idx_max_class_correct)

            if n_tmin > 0: 
                C[i, idx_min_class_correct] = C[i, idx_min_class_correct] * beta

                # [test]
                if i in test_cases: 
                    sample_weights[uname] = np.random.choice(C[i, idx_min_class_correct], 10)
                    tps[uname] = n_tmin
                    sample_weights_max_class[uname] = np.random.choice(C[i, idx_max_class_correct], 10)
                    tns[uname] = n_tmax

            if suppress_max_class: 
                
                C[i, idx_max_class_correct] = w_min    # this may conflict with unmask in balance_class_weights()
                sample_weights_max_class[uname] = w_min

        # [test]
        # idx = []
        # n_max = 5
        # bag_start_indices = range(0, len(U), bag_count)
        # # names = [c.split(sep)[0] for c in U[bag_start_indices]]
        # for i in bag_start_indices:
        #     idx.extend( list(range(i, i+n_max)) )
        # idx = np.array(idx)
        div(message='(regulate_weights) minority class weight distribution after magnified by beta={}...'.format(beta))
        for u, wx in sample_weights.items(): 
            print("... [{}] {} (size: TP)".format(u, tps[u]))
            print("...      {} ... (sample weights: TP)".format(wx))
            print("...      {} (size: TN)".format(tns[u]))
            print("...      {} ... (sample weights: TN)".format(sample_weights_max_class[u]))

        return  # C is modified in place

    # from sklearn.metrics import brier_score_loss
    import scipy.sparse as sparse
    from evaluate import findOptimalCutoff   #  p_th <- f(y_true, y_score, metric='fmax', pos_label=1)
    import collections

    is_cascade = kargs.get('is_cascade', False) # [note] cascade mode seems more favorable (i.e. X=[R|T])

    # Probability filtering policy for the training set
    policy_filtering = kargs.get('policy', 'item') # [obsolete]

    # Probability filtering policy for the test set (only relevant in casade mode i.e. X = [R | T])
    policy_filtering_test = kargs.get('policy_test', 'polarity') if is_cascade else None # None as 'undefined'

    n_train = kargs.get('n_train', -1)
    # ... if not in cascade mode (where X = [R | T]), then policy_filtering_test is undefined and not meaningful

    null_marker = kargs.get('fill', 0) # [todo] marker for missing data
    topk_users, topk_items = 0, 0  # default, use all by setting to 0
    pos_label = kargs.get('pos_label', 1)
    n_users, n_items = X.shape[0], X.shape[1]
    fold = kargs.get('fold', -1)  # only used for debugging
    
    conf_measure = kargs.get('conf_measure', 'brier')
    U = users = kargs.get('U', []) # names of users/classifiers
    tSupervised = True # kargs.get('supervised', True) # Note: Always make use of the training set's labels if possible

    ### balance class weights in the confidence score 
    tBalanceClassWhileMasking = kargs.get('balance_class', False) 
    tBalanceClassWeights = kargs.get('balance_class_weights', False) # False if conf_measure == 'rank' else True
    tSuppressMaxClass = kargs.get('suppress_negative_examples', False)
    tEstimatedLabels = kargs.get('estimated_labels', False) # True if X is a test set
    print('(toConfidenceMatrix) balance while masking? {}, balance class weights? {}'.format(tBalanceClassWhileMasking, tBalanceClassWeights))

    # [design] Is it better to use cascade mode (i.e. X = [R | T]), so that no need to worry about message passing for T? 
    #####################################################################################
    # "Message passing": Pass training set information useful for determining T's probaility filter (assuming that X contains only the test set data i.e. T)
    tHasMessageFromTrainingSplit = False
    R = T = L_train = L_test_est = None
    Eu = [] # effective classifiers
    M = kargs.get('message', None)
    if M is not None: 
        R, L_train, *rest = M
        tHasMessageFromTrainingSplit = True
        print('(toConfidenceMatrix) Propagate from the training split (R) to the test split (T) | n(train): {}, n(test): {}'.format(
            R.shape[1], X.shape[1]))

        # use confidence matrix from training split (Cr) to find classifiers with support (correct predictions)
        if len(rest) > 0: 
            Cr = rest[0]

            # Eu? A classifier is "effective" if it's accumulated confidence score across all data points is greater than a baseline value (usu. 0)
            Eu = identify_effective_users(Cr, L_train=L_train, fill=null_marker)
            # ... Eu or effective user is a binary vector, where 1: effective 0: otherwise

            print('(toConfidenceMatrix) How many classifiers were effective in the training set? {n} | total: {N}'.format(n=np.sum(Eu), N=Cr.shape[0]))
    if is_cascade: # then X must be [R|T]
        if isinstance(X, (tuple, list)): 
            assert isinstance(L, (tuple, list))
            R, T = X
            L_train, L_test_est = L
        else: 
            assert n_train > 0, f"(toConfidenceMatrix) No way of knowing how to split X into R and T without knowing size of R (given n_train {n_train})"
            assert len(L) == X.shape[1]

            R, T = X[:,:n_train], X[:,n_train:]
            L_train, L_test_est = L[:n_train], L[n_train: ] # note that `L_test_est` is only a guess since we do not know test set's true labels
    
    #####################################################################################

    # conditions with p_threshold 
    #   1) p_threshold given externally 
    #   2) p_threshold is to be inferred from training data message (X_train, L_train); used for computing confidnece scores for the test split 
    #   3) p_threshold is to be esimated via 'prior'; L must be given
    #   
    #   4) p_threshold could be estimated in a unsupervised way even without L, but not recommended
    policy_threshold = kargs.get('policy_threshold', 'fmax')
    p_threshold = kargs.get('p_threshold', [])  # depends on policy_threshold: 'fmax', float, 'unsupervised' 

    # nickname 
    # X_train, X_test = R, T

    # [design] Always pre-compute probability thresholds so that this block can be skipped
    #####################################################################################
    if len(p_threshold) == 0: 
        if tHasMessageFromTrainingSplit: # estimate probabilty thresholdds
            # policy_threshold: prior, fmax, ...
            assert R is not None and L_train is not None

            print('(toConfidenceMatrix) message passing: use training set statistics to estimate proba thresholds | policy={p}'.format(p=kargs.get('policy_threshold', 'prior')))
            policy_threshold = kargs.get('policy_threshold', 'prior')
            p_threshold = estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=policy_threshold) # policy: any policy that utilizes L
        else: 
            assert len(L) > 0, "Both p_threshold and L are not given, and neither was training data message given!"
            policy_threshold = 'prior' # kargs.get('policy_threshold', 'prior') if len(L) > 0 else 'ratio' 
            p_threshold = estimateProbThresholds(X, L=L, pos_label=pos_label, policy=policy_threshold, ratio_small_class=kargs.get('ratio_small_class', 0.01)) # findOptimalCutoff(L_train, R, metric='fmax', beta=1.0, pos_label=1)
    ##################################################################################### 
    show_thresholds(p_threshold, users=kargs.get('U', []))
    
    # Estimtae labels if not given
    # policy: 
    #     a. majority vote 
    #     b. best lh match (need statistics from R); mask transfer
    # tEstimatedLabels = False

    # [design] Always pre-compute estimated labels before this call to make the logic clean
    ##################################################################################### 
    if len(L) == 0: 
        assert not is_cascade, "In cascade mode X=[R | T], we know at least the training data's labels, which must be provided through L"

        # then X must be just a test set (i.e. X = T)
        print("(toConfidenceMatrix) Estimating labels (Lh) using (1) input X and (2) proba thresholds given earlier | policy={p} ...".format(p=policy_threshold))

        # [note] joint model doesn't work well
        # if tHasMessageFromTrainingSplit: 
        #     # jointModel = estimateProbaByLabelMatrix(R, L_train, p_th=p_threshold, pos_label=1) # policy_threshold='prior' 
        #     L = lh = estimateLabels(X, L=[], p_th=p_threshold, pos_label=pos_label, M=(R, L_train) ) # joint_model=jointModel
        
        # Why passing Eu? Because we want majority vote to account for only effective classifiers
        L = lh = estimateLabels(X, L=[], Eu=Eu, p_th=p_threshold, pos_label=pos_label, ratio_small_class=kargs.get('ratio_small_class', 0.01)) # policy: 'ratio' by default when no labels are provided
        tEstimatedLabels = True
    else: 
        # use the proba threshold to compute the mask and keep track of their statistics to be used in estimating labels in T
        pass
    ##################################################################################### 
    # ... estimated labels ready L <- Lh
    # ... tEstimatedLabels is True when 'estimated_labels' passed externally is True or when L wasn't given
    
    # [design] L_test
    ##################################################################################### 
    # test accuracy (only useful when L is estimated)
    L_test = L_ext = kargs.get('L_test', [])
    if tEstimatedLabels and len(L_test) > 0: 
        accuracy = np.sum(lh == L_test) / (len(L_test)+0.0)
        div('(toConfidenceMatrix) accuracy of estimated labels: {} | n(L_ext): {}'.format(accuracy, len(L_test)), symbol='#', border=2)
    ##################################################################################### 

    if fold == 0: print('(toConfidenceMatrix) compute conficence scores | conf_measure: {0}, outer_product? {1}'.format(conf_measure, False))

    # Scoring would be ignored in mode: {'ratio', }
    C0 = confidence2D(X, L, mode=conf_measure, 
                scoring=kargs.get('scoring', brier_score_loss), 
                outer_product=False, 

                    # following params are used only when mode = 'ratio'
                    p_threshold=p_threshold,  
                    policy_threshold=kargs.get('policy_threshold', ''), 
                    ratio_small_class=kargs.get('ratio_small_class', 0.01))  # don't return outer(wu, Wi) 
    ################################################################# 
    # ... C0: raw confidence scores 
    Cui = np.zeros(C0.shape)+C0

    print('(toConfidenceMatrix) Introudce polarity: Mc(TP, TN) > 0, Mc(FP, FN) < 0 | has p_threshold? {tval} | message passing? {tm} | policy: {p} | supervised? {s}'.format(
        tval=len(p_threshold) > 0, p=policy_threshold, tm=tHasMessageFromTrainingSplit, s=tSupervised))

    #################################################################
    Mc = None
    tConservative = True
    isPolarityMatrix = True
    Pc, Lh = color_matrix(X, L, p_threshold) # TP=2, TN=1, FP=-2, FN=-1

    if tEstimatedLabels: # can only be true for a test split
        # Mc, Lh = color_matrix(X, L, p_threshold)  # Mc0 is a (reduced) color matrix 

        # estimate item support ratio (user_ratios)
        ratio_users = estimate_support_ratio(X, L, p_threshold, gamma=1.0) if kargs.get('ratio_users', -1) < 0 else kargs['ratio_users'] 

        if tConservative: # if true, use more conservative estimate of Mc
            # assert policy_filtering_test.startswith('po'), "filtering by polarity only applies to 'cascade' mode"
            if policy_filtering_test.startswith('po'): 
                Pc = mask_over(C=Cui, X=X, L=L, ratio_users=ratio_users,   # kargs.get('ratio_users', ratio_users)

                                # message passing
                                p_threshold = p_threshold, # may need to estimate Lh again
                                ratio_small_class=kargs.get('ratio_small_class', 0), factor_small_class=kargs.get('factor_small_class', 1.0),  # used for unsupervised mode
                                    kind='polarity',   # <<< (collaborative) filtering policy
                                    marker=null_marker, 
                                        supervised=tSupervised, 
                                            balance_class=tBalanceClassWhileMasking,
                                            estimated_labels=tEstimatedLabels,  # is L an estimated label (test data) or true labels? 
                                                labeling_model=kargs.get('labeling_model', 'simple'),   # used to determine/estimate the polarity matrix, which is used to select reliable 'ratings'
                                                constrained=kargs.get('constrained', True), stochastic=kargs.get('stochastic', True), 

                                                # used to test polarity matrix
                                                test_labels=L_test,  # labels of the test set

                                                fold=kargs.get('fold', -1))
                print('(toConfidenceMatrix) Altered mask | conservative n_ones(Mc): {} <? n_ones(Mc0): {}'.format(np.sum(Mc==1), np.sum(Mc0==1)))  
            # isPolarityMatrix = True 
            # Mc: still in 0, 1 representation

        # assert np.all( (Mc >= 0) | (Lh != L[None,:]) ), "If Mc_bar<i,j> is true, then it must be the case that the labeling is consistent." # ... ok. 
        # ... if Mc < 0 (if not preferred) => Lh != L (then it must be the case that labels are matched)
    else: 
        pass

    if is_cascade and tConservative: # and (policy_filtering != policy_filtering_test): 
        msg = ''
        Pcr0, Pct0 = Pc[:,:n_train], Pc[:, n_train:]  # Mct0
        # using correctnes matrix, 0: negative, 1: positive

        msg += '(toConfidenceMatrix) Altered mask? | policy_filtering_test: {ft} | policy_threshold: {p} \n'.format(ft=policy_filtering_test, p=policy_threshold)
        
        assert n_train > 0 and n_train < X.shape[1], "Invalid train-test split point: {}".format(n_train)
        # R, T = X[:,:n_train], X[:,n_train:]
        Cr0, Ct0 = Cui[:,:n_train], Cui[:,n_train:]
        Lr, Lt = L[:n_train], L[n_train:]

        nZ0 = np.sum(Ct0==0)  # number of neutral particles initially in the test split

        # Estimate item support ratio (user_ratios)
        ratio_users = estimate_support_ratio(R, Lr, p_threshold, gamma=1.0) if kargs.get('ratio_users', -1) < 0 else kargs['ratio_users']

        # Mc_bar
        if policy_filtering_test.startswith('po'): 
            # ... Cui is still the raw confidence score
            policy_polarity = kargs.get('policy_polarity', 'sequence')

            Mct = mask_over(C=Cui, X=X, L=L, ratio_users=ratio_users, 
                        U=U,  # user/classifer names
                        n_train=n_train, 
                        p_threshold = p_threshold, # may need to estimate Lh again
                            kind='polarity',   # <<< (collaborative) filtering policy
                            marker=null_marker,  
                                labeling_model=kargs.get('labeling_model', 'simple'),   # used to determine/estimate the polarity matrix, which is used to select reliable 'ratings'
                                    constrained=kargs.get('constrained', True), 
                                    stochastic=kargs.get('stochastic', True), 
                                        estimate_sample_type=kargs.get('estimate_sample_type', True),
                                        policy_polarity=policy_polarity,  # options: median  
                                        
                                        # used to test polarity matrix
                                        test_labels=L_test,  # labels of the test set
                                        
                                        fold=kargs.get('fold', -1))
            ### update 
            #   1. Mct returned from mask_over in polarity mode with policy_polarity 'classification' is now a "color matrix"

            # [test]
            colors = np.unique(Mct)  

            # ... if in multiclass mode
            # n_color_min = 4 if policy_polarity.startswith( ('class', 'seq',) ) else 3 
            # assert len(colors) >= n_color_min, "Expecting {}-color particles but got: {}".format(n_color_min, colors)   
        else: 
            Mct = mask_over(C=Ct0, X=T, L=Lt, ratio_users=ratio_users, 

                        # message passing
                        p_threshold = p_threshold, # may need to estimate Lh again
                        ratio_small_class=kargs.get('ratio_small_class', 0), factor_small_class=kargs.get('factor_small_class', 1.0),  # used for unsupervised mode
                            kind=policy_filtering_test,   # <<< (collaborative) filtering policy
                            marker=null_marker, 
                                supervised=tSupervised, 
                                    # balance_class=tBalanceClassWhileMasking,
                                    estimated_labels=tEstimatedLabels,  # is L an estimated label (test data) or true labels? 
                                       fold=kargs.get('fold', -1))

            assert len(np.unique(Mct)) >= 2, "n_polarity(Mct): {} < 3?".format(np.unique(Mct)) 
        msg += '(toConfidenceMatrix) original: n(pref)(Mct0) {} >? conservative: n(pref)(Mct)|T: {}\n'.format(np.sum(Pct0>0), np.sum(Mct>0))
        # ... expect Mct to be more conservative  ... ok
        
        # masking neutral entries
        Ct0[Mct == 0] = 0.0 # neutral particles do not enter into the optimization objective
        Ct = Ct0
        assert np.sum(Ct==0) >= nZ0, '(toConfidenceMatrix) n(zero)(Ct0): {} <? n(zero)(Ct): {}\n'.format(nZ0, np.sum(Ct==0))
        
        ##############################################
        # Mct = from_color_to_polarity(Mct) # other params: codes={}, verify=True
        Pc = np.hstack( (Pcr0, Mct) )   
        # Cui = np.hstack( (Cr0, Ct) )
        ##############################################

        # [test]
        Lht = estimateLabelMatrix(T, p_th=p_threshold)
        
        # whenever (Mct == 1), it must be true that (Lht == Lt[None,:])
        # [test]
        # for j in range(T.shape[1]): 
        #     assert all(Lht[:, j][Mct[:, j] == 1] == Lt[j]), "Wherever Mct is 1 (preferred), the corresponding entries in Lht must be consistent with L"
        # ... Mct[:, j] == 0 now would include both misaligned and uncertain cases
        # assert np.all( ~(Mct == 1) | (Lht == Lt[None,:]) ), "If Mct<i,j> is true, then it must be the case that the labeling is consistent." # ... ok. 
        n_pos = np.sum(Pc > 0)
        n_neg = np.sum(Pc < 0) # Cui.shape[0] * Cui.shape[1] - n_pos - n_neutral
        n_neutral = np.sum(Pc == 0)   # Mc == 0 at this point includes both negative and neutral
        assert np.sum(Mct==0) == np.sum(Pc==0) and n_neutral>=0, "(toConfidenceMatrix) Inconsistent number of neutral particles? np.sum(Ct==0): {} =?= {}".format(np.sum(Mct==0), np.sum(Pc==0))
        # ... only test split has neutral
        assert n_neg > 0
        msg += '(toConfidenceMatrix) Mc after intro neutral particles | n(pos): {np}, n(neg): {nn}, n(neutral): {nc}\n'.format(np=n_pos, nn=n_neg, nc=n_neutral)
        print(msg)   

        # nP = np.sum(Mc==1)

    #################################################################
    # ... colored polarity matrix Pc is determined

    # Mc = Mc.astype(float)
    # Mc[Mc == 0] = -1   # convert to (-1, 1)-matrix
    # Cui = Cui * Mc   
    # C(tp, tn) > 0, C(fp, fn) < 0

    # test_cases = np.random.choice(range(Cui.shape[0]), 5) 
    verify_confidence_matrix(Cui, X=X, L=L, p_threshold=p_threshold, Po=Pc, U=U, measure=conf_measure, message='(before) raw weights', test_cases=[])  # test_cases <- [] to use default
    # # side effect on W
    # if kargs.get('masked', True): 
    #     # mask FP, FN (and missing values if appilcable)

    # scaling C <- alpha * C 
    ####################################################
    alpha = kargs.get('alpha', 1.0)
    beta = kargs.get('beta', 1.0)  # increase the confidence weight further by this factor
    tDiscountTest = kargs.get('discount_test', False)
    # give higher weights to minority class 
    # if conf_measure != 'rank': 
    #     Cui = alpha * Cui      
        # ... it has precede balance_class_weights() because 
        #     the number of unmasked negative examples tend to be significantly larger
        #     than that of the unmasked positive/minority examples
    
    # ... Mc: (1: {TP, TN}, 0: {FP, FN})
    # if tBalanceClassWeights and not tBalanceClassWhileMasking: 
    #     # if test split: 
    #     #    a. unmask? 
    #     balance_class_weights(Cui, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, alpha=alpha, beta=beta, 
    #         suppress_max_class=tSuppressMaxClass, 
    #             discount_test=tDiscountTest, n_train=n_train, 
    #                 is_cascade=is_cascade, 
    #                     test_cases=test_cases, is_test_split=tEstimatedLabels)
    #     # ... alpha, is_test_split => eventually regulate_weights(), umask() will be merged in

    #     # up-regulate minority class weights and down-regulate majority class weights
    #     # regulate_weights(Cui, beta=beta, U=U, test_cases=test_cases, suppress_max_class=tSuppressMaxClass) 
    # ####################################################
    # Cui = alpha * Cui 

    ####################################################

    # verify_confidence_matrix(Cui, X=X, L=L, Po=Pc, p_threshold=p_threshold, U=U, measure=conf_measure,
    #     message='(after) balanced + magnified (alpha={}, beta={}) | dtype: {}'.format(alpha, beta, 'test set' if tEstimatedLabels else 'training set'), 
    #         test_cases=test_cases, plot=True if fold == 1 else False, 
    #         test_weight_constraints=True)

    n_incorrect = np.sum(Pc < 0)
    n_uncertain = np.sum(Pc == 0)
    n_correct = np.sum(Pc > 0)
    n_colors = len(np.unique(Pc))
    assert n_colors >= 4, "Expecting colored particles but got {}".format(colors)
    print('... Colored polarity matrix (Pc) | n_negative: {}, n_neutral: {} n_positive: {} | data: {}'.format(n_incorrect, 
        n_uncertain, n_correct, 'test set' if tEstimatedLabels else 'training set'))

    #     # supervised vs unsupervised will lead to different augmenented label set La
    #     # [note] offset is used to indicate where test split starts
    #     Cui = mask_over(Cui, Ra, La, p_th=p_threshold, offset=offset) # other params: mask_all_test
    n_zeros = n_nonzeros = 0
    if kargs.get('sparse', True): 
        Cui = sparse.csr_matrix(Cui)
        Pc = sparse.csr_matrix(Pc)

        # [test]
        n_nonzeros = sparse.csr_matrix.count_nonzero(Pc)
        n_zeros = Pc.shape[0] * Pc.shape[1] - n_nonzeros
        # if not tEstimatedLabels: 
        #     # for test split, consider unmask
        #     assert n_zeros > 0

        print('... (verify) Cui, Mc converted to sparse matrix.')
    else: 
        n_zeros = np.sum(Pc == 0)
        n_nonzeros = np.sum(Pc != 0)
    print('(toConfidenceMatrix) Cui (alpha: {}) |  dim(Cui): {}, n_zeros (uncertain): {} (>? 0) vs nonzeros: {} (masked ratio={})'.format(alpha, 
        Cui.shape, n_zeros, n_nonzeros, n_zeros/(Cui.shape[0] * Cui.shape[1] + 0.0)))
        
    # Cui = alpha * R 

    # C0: Raw confidence scores for all entries in X prior to filtering (unreliable entries)
    # Cui: masked confidence scores, where polarity neutral is 0
    # Po: polarity matrix (subsummed by Pc below)
    # Pc: colored polarity matrix
    return Cui, Pc, p_threshold   # where Pc represents colored polarity



### 

_BaseCF(_BaseStacking)
   - add data directory ... CF depends on off-line computation a lot


### 

evalConfidenceMatrix() 






### Stacking 

class _BaseStackingCF(_BaseStacking): 
    """
    Base class for stacking based on collaborative filtering methods. 
    """

    @abstractmethod
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="cf",
        n_jobs=None,
        verbose=0,
        passthrough=False,
    ):
        super().__init__(estimators=estimators)
        self.final_estimator = None # this is going to be a CF method
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

        self.X_meta = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the estimators in the same way as _BaseStacking.fit() does 
        but also do the following: 

        1. cache `X_meta` (predictions colleccted from cross_cross_val_predict() 
        2. re-train all the base estimators on the whole training data.

        Step 2 is done in preparation for predicting new (unseen) test data.  

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionchanged:: 0.23
               when not None, `sample_weight` is passed to all underlying
               estimators

        Returns
        -------
        self : object
        """
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()

        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(clone(est), X, y, sample_weight)
            for est in all_estimators
            if est != "drop"
        )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != "drop":
                current_estimator = self.estimators_[est_fitted_idx]
                self.named_estimators_[name_est] = current_estimator
                est_fitted_idx += 1
                if hasattr(current_estimator, "feature_names_in_"):
                    self.feature_names_in_ = current_estimator.feature_names_in_
            else:
                self.named_estimators_[name_est] = "drop"

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, "random_state") and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        fit_params = (
            {"sample_weight": sample_weight} if sample_weight is not None else None
        )
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(
                clone(est),
                X,
                y,
                cv=deepcopy(cv),
                method=meth,
                n_jobs=self.n_jobs,
                fit_params=fit_params,
                verbose=self.verbose,
            )
            for est, meth in zip(all_estimators, self.stack_method_)
            if est != "drop"
        )

        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth
            for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != "drop"
        ]

        self.X_meta = self._concatenate_predictions(X, predictions)
        # _fit_single_estimator(
        #     self.final_estimator_, X_meta, y, sample_weight=sample_weight
        # )

        return self






        from sklearn_crfsuite import scorers
        from sklearn_crfsuite import metrics


        polarity_labels = list(model.classes_)
        sorted_labels = sorted(
            polarity_labels, 
            key=lambda name: (name[1:], name[0])
        )

f1 = metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=polarity_labels)
        msg += "(polarity_modeling) flat F1 score on T: {}\n".format(f1)
        msg += metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3) + '\n'

        Wu = confidence_brier(X, L, mode='user', topk=topk_users)

        # now turn 1-D weight vector into a 2D column vector
        # np.repeat(W[:, np.newaxis], len(labels), axis=1)  # W[:, np.newaxis] => 2D col vec of W => repeats n_items of time
        
        # now compute item-label correlation
        if Wi is None: Wi = confidence_brier(X, L, mode='item', topk=topk_items)

        W = np.outer(Wu, Wi)  # n_users by n_items


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
    n_factors = kargs.get('n_factors', [10, 50, 100, 150, 200, 250])
    alpha = kargs.get('alpha', [100, ])

    # combine baseline and compute baseline scores 
    base_criteria = criteria = {'F': n_factors, 'A': alpha, 'S': Job.settings, } # ['F10_A100', 'F50_A100', 'F100_A100', 'F150_A100', 'F200_A100', ] 
    
    if Job.domain:  # --domain
        # individual domain
        domains = [Job.domain, ]
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
        n_factors = kargs.get('n_factors', [75, 100, 120, ])

        test_setting = kargs.get('test_setting', 6)
        Job.settings = [test_setting, ]
        base_criteria = criteira = {'F': n_factors, 'A': alpha, 'S': Job.settings, }
        # domains = ['pf1',]  # ['pf2', 'pf3', 'thaliana']  
        test_domain = kargs.get('test_domain', 'thaliana')  # <<< configure
        domains = [test_domain, ]  # kargs.get('domains', [test_domain, ])    
        domain_group = 'test'  # 'splice_site'

        n_cycles = 5
    ####################################
    # ... domains, domain_group, criteria 


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
        if len(criteria['S']) > 0: 
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
    
    print("(t_plot) target domains: {}".format(domains))

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


        # ... majority vote
        nbp = R.shape[0]
        r_maxvote = 0.0
        neg_label, pos_label = 0, 1
        
        lhv = np.zeros(nbp)
        idx_pos = np.where(R[:, j] >= p_th)[0]
        lhv[idx_pos] = 1

        counter = collections.Counter(lhv)
        # top_vote = counter.most_common(1)[0]
        n_neg, n_pos = counter[neg_label], counter[pos_label]
        # maxvote, n_maxvote = top_vote

        idx_activated = []
        if n_pos > n_neg: 
            maxvote = pos_label
            minvote = neg_label
            n_maxvote, n_minvote = counter[pos_label], counter[neg_label]
            idx_activated = idx_pos
        else: 
            maxvote = neg_label 
            minvote = pos_label
            n_maxvote, n_minvote = counter[neg_label], counter[pos_label]
            idx_activated = np.where(R[:, j] < p_th)[0]
        r_maxvote, r_minvote = n_maxvote/(nbp+0.0), n_minvote/(nbp+0.0) 
            
        # fv.extend( [r_minvote, r_maxvote] )
        vn = 'maxvote'   # maxvote and its proportional (degree of belief)
        fvn.append(vn)
        fv.append(maxvote)
        tags[i][vn] = maxvote

        vn = 'r_maxvote'
        fvn.append(vn)
        fv.append(r_maxvote)
        tags[i][vn] = r_maxvote

        # include the classifiers/users that are 'activated' (i.e. members of majority votes) for this column/datatum
        if U is not None: 
            for u in np.asarray(U)[idx_activated]: 
                vn, bnum = base_name(u)
                
                vn = decorate_var(vn, 'activated')
                fvn.append(vn)
                fv.append(1.0)   

def get_vars_hstats(R, i, j, p_th, Rm=None, C=None, Lh=None, p_model={}, r_min=0.1, name='', index=0, verbose=False, wsize=20, to_dict=False):  
    # get BP prediction vector statistics as variables
    from scipy.stats import kurtosis, skew, ks_2samp

    # sample_types = ['tp', 'tn'] + ['fp', 'fn']
    # codes = {'tp': 2, 'tn': 1, 'fp': -2, 'fn': -1, 
    #         'unk': 0, 't': 3, 'f': -3}
    sample_types = Polarity.sample_types
    codes = Polarity.codes

    msg = ""
    # query point 
    q = pt_q = R[i, j]   # q

    fv = []  # features 
    fvn = []  # feature names

    max_gap = 1.0

    N = R.shape[1]
    rk = -1   # rank of the query point
    
    wsize_min, wsize_max = N//100, 20 
    wsize = min(wsize_max, max(wsize_min, wsize))

    sv = []  # reset sv
    if Rm is not None: 
        # ... Rm[i, :] must be sorted
        rk = np.searchsorted(Rm[i, :], pt_q, side='left')  # rank, q's position in R[i, :]
        k = wsize # min(min_size, N//100)  # no less than 50 points
     
        # search k nearest neighbor
        bl, br = rk-k, rk+k   

        # boundary conditions
        if bl < 0: bl = 0
        if br >= N: br = N-1 

        pts_lower = Rm[i, bl:rk]
        pts_higher = Rm[i, rk:br]
        assert np.all(pts_lower <= pt_q) and np.all(pts_higher >= pt_q), "query: {} | max(low): {}, min(high): {} | sorted? {}".format(pt_q, np.max(pts_lower), np.min(pts_higher), scipy.stats.rankdata(Rm[i, :][:20]))
        
        # SE wrt q
        pts_knn = np.hstack([pts_lower, pts_higher])
        
        se_local = np.std(pts_knn) # mean_squared_error(pts_knn, np.full_like(pts_knn, pt_q))
        range_local = stats.iqr(pts_knn) # np.max(pts_knn)-np.min(pts_knn)
        # fv.append(se_local)  # <<< 

        if verbose: 
            n = 10
            pts_knn_subset = np.hstack([pts_lower[:n], pts_higher[-n:]])
            msg += "(get_vars_hstats) {} | local(SE): {} | R[i,j]: {}, rank: {} | pts_knn (subset:{}, total: {} vs N/100:{}): {}\n".format(name,
                se_local, pt_q, rk, len(pts_knn_subset), len(pts_knn), N//100, pts_knn_subset)

        # compare distribution with neighoring points of different flavors
        diffs = {stype: {} for stype in sample_types}
        for stype in sample_types:
            kss = np.max(pts_knn-0.0)   # assumed to be very large (as large as max abs distance to zero vectors)
            if i in scopes[stype]: 
                pts = scopes[stype][i]['sample']
                # ... assuming that pts has been pre-sorted
                n = len(pts)

                # rank wrt to this particular sample type
                rk = np.searchsorted(pts, pt_q, side='left')

                # search k nearest neighbor
                bl, br = rk-k, rk+k

                # boundary conditions
                if bl < 0: bl = 0
                if br >= N: br = n-1 

                pts_lower = pts[bl:rk]
                pts_higher = pts[rk:br]
                assert np.all(pts_lower <= pt_q) and np.all(pts_higher >= pt_q), "query: {} wrt flavor: {} | max(low): {}, min(high): {}".format(pt_q, stype, np.max(pts_lower), np.min(pts_higher))
                
                pts_knn_flavored = np.hstack([pts_lower, pts_higher])
                range_flavored = stats.iqr(pts_knn_flavored) # np.max(pts_knn_flavored)-np.min(pts_knn_flavored)
                se_flavored = np.std(pts_knn_flavored)

                # K-S tes
                ks_stats = ks_2samp(pts_knn, pts_knn_flavored)
                kss = -10*np.log(ks_stats.pvalue) # ks_stats.statistic, -10*np.log(ks_stats.pvalue) 
                msg += "(get_vars_hstats) Local structure: {} | flavor: {} | R[i,j]: {}, rank: {} | KS statistic: {}\n".format(name, stype, pt_q, rk, kss)

                diffs[stype]['ks.statistic'] = ks_stats.statistic
                diffs[stype]['ks.pvalue'] = -10*np.log(ks_stats.pvalue) # ks_stats.pvalue 
                diffs[stype]['median'] = np.median(pts_knn_flavored)/(np.median(pts_knn)+1e-4)
                diffs[stype]['skew'] = skew(pts_knn_flavored)/(skew(pts_knn)+1e-4)
                diffs[stype]['kurtosis'] = kurtosis(pts_knn_flavored)/(kurtosis(pts_knn)+1e-4)
                diffs[stype]['se'] = se_flavored/(se_local+1e-4)
                diffs[stype]['range'] = range_flavored/(range_local+1e-4)
            else: 
                msg += "... !!!{}-th point in R does not have sample type: {}\n".format(i, stype)

            # sv.append(kss) # ... use differential variables instead of the values themselves

        # horizontal differential variables 
        #  tp  fp
        #  tn  fn 
        dx = ['ks.pvalue',  'se', 'skew', 'kurtosis', ]  # 'se', 'kurtosis', 'skew', 'range', 
        terms = ['tpfp', 'tnfn'] # + ['tpfn', 'tnfp', ]
        for d in dx:  

            # if d in ['se', ]:  # ratio
            #     del_tpfp = diffs['tp'][d]/(diffs['fp'][d]+1e-4)
            #     del_tnfn = diffs['tn'][d]/(diffs['fn'][d]+1e-4)
            # else:  # difference

            # horizontal terms
            del_tpfp = diffs['tp'][d] - diffs['fp'][d]
            del_tnfn = diffs['tn'][d] - diffs['fn'][d]

            # cross terms
            del_tpfn = diffs['tp'][d] - diffs['fn'][d]
            del_tnfp = diffs['tn'][d] - diffs['fp'][d]

            # add as variables
            fv.extend([del_tpfp, del_tnfn]) # [del_tpfp, del_tnfn] + [del_tpfn, del_tnfp ]
            for term in terms: 
                fvn.append('%s-%s' % (d, term))

            msg += "... Differential KS on {}-example | metric: {} | R[i,j]: {}, rank: {} | KS(tp-fp, tn-fn): {}, {}\n".format(name, d, pt_q, rk, del_tpfp, del_tnfn) #  del_tpfn, del_tnfp

    vn = 'delta_pth'
    delta = pt_q - p_th[i]
    # fv.append( delta ); fvn.append(vn)
    # ... case q > p_th, likely TP if L = 1, or FP if L = 0
    # ... case q <= p_th, likely FN if L = 1, or TN if L = 0

    ### rank?  can also use q-
    if Rm is not None and Lh is not None:
        vn = 'rank'   # label-specific rank

        if delta >= 0: 
            pts = Rm[i, :][Lh[i, :] == pos_label]
            rk = np.searchsorted(pts, pt_q, side='left')
        else: 
            # negative rank
            pts = Rm[i, :][Lh[i, :] == neg_label]
            n_pts = len(pts)
            rkc = np.searchsorted(pts, pt_q, side='left')
            rk = -((n_pts+1)-rkc)

        # N = Rm.shape[1]
        # Rm: either a sorted array (or a rank array)
        # r = np.searchsorted(R[i, :], q, side='left')  
        fv.append( rk ); fvn.append(vn) 

    # --- (raw) confidence score ---
    vn = 'c-score'
    if C is not None: 
        fv.append(C[i, j])
        fvn.append(vn)

    # assert len(fv.shape) == 1
    assert len(fv) == len(fvn), "dim(fv): {} <> dim(fvn): {}".format(len(fv), len(fvn))
    if verbose: 
        # for vn, v in zip(fvn, fv): 
        msg += "(get_vars_hstats) vars name values ({}):\n... {}\n".format(name, list(zip(fvn, fv)))
        # print("... q: {}, topk_th: {} | r_min: {}".format(q, topk_th, r_min))   # ... ok
        print(msg)

    if to_dict: 
        return dict(zip(fvn, fv))

    return np.array(fv)



run_cluster_analysis(P, U=U, X=X, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=False, save_plot=True, file_type='train')

def to_preference(Po, beta_neutral=0.0):
    import scipy.sparse as sparse

    # polarity matrix to preference matrix 
    # assert beta_neutral < 1 and beta_neutral >= 0.0

    P = np.ones(Po.shape)  
    if sparse.issparse(Po): 
        Pa = Po.toarray()
        P[Pa==0] = beta_neutral      # masking neutral
        P[Pa < 0] = 0.0    # masking negative
        P = sparse.csr_matrix(P)
    else: 
        P[Po==0] = beta_neutral  # masking neutral
        P[Po < 0] = 0.0  # masking negative
    return P # {-1, 0, 1}
def preference_to_polarity(M):
    return to_polarity(M)
def to_polarity(M): 
    # from preference matrix to polarity matrix
    import scipy.sparse as sparse
    P = np.ones(M.shape)  
    if sparse.issparse(M):      
        P[M.toarray() == 0] = -1    # preference 0 ~ negative polarity 
        P = sparse.csr_matrix(P)
    else: 
        P[M == 0] = -1 
    return P
def to_polarity_matrix(M):
    # preference matrix {0, 1} to polarity matrix 
    return to_polarity(M)

def from_color_to_preference(M, codes={}, verify=True): 
    import scipy.sparse as sparse
    
    # [test]
    colors = set(np.unique(M))
    vmin, vmax = min(colors), max(colors)
    if vmin == 0 and vmax == 1 and len(colors) == 2: 
        print("(from_color_to_polarity) Input M is already a preference matrix! | colors: {}".format(colors))
        return M  

    # M: color matrix
    # use: for approximating ratings
    Po = np.zeros(M.shape)
    if sparse.issparse(M):
        mask = M.toarray()
    else:
        mask = np.copy(M)

    # only 3 possible values {1, -1, 0}
    # Po[(Po == codes['tp']) | (Po == codes['tn'])] = 1
    Po[mask > 0] = 1

    # Po[(Po == codes['fp']) | (Po == codes['fn'])] = -1
    Po[mask < 0] = 0

    # M[(M == codes['f']) ] = -1
    # Po[ mask  == codes['unk'] ] = 0
    return Po

def from_color_to_reduced_color(M, codes={}, verify=True):
    """
    convert from color (polarity) matrix to reduced color polarity matrix {-1, 0, 1, 2}
        used for rating approximation
    """
    import scipy.sparse as sparse
    
    # [test]
    colors = set(np.unique(M))
    vmin, vmax = min(colors), max(colors)
    if colors == set([-1, 0, 1, 2]): 
        print("(from_color_to_reduced_color) Input M is already a reduced-color matrix! | colors: {}".format(colors))
        return M
    
    # M: color matrix
    # use: for approximating ratings
    if sparse.issparse(M):
        Po = M.toarray()
    else:
        Po = np.copy(M)

    # only 3 possible values {1, -1, 0}
    # Po[(Po == codes['tp']) | (Po == codes['tn'])] = 1
    # Po[mask > 0] = 1

    # Po[(Po == codes['fp']) | (Po == codes['fn'])] = -1
    Po[Po < 0] = -1

    # M[(M == codes['f']) ] = -1
    # Po[ mask  == codes['unk'] ] = 0
    return Po

def from_color_to_polarity(M, codes={}, verify=True): 
    # convert from color (polarity) matrix to regular polarity matrix {-1, 0, 1}
    import scipy.sparse as sparse
    
    # [test]
    colors = set(np.unique(M))
    vmin, vmax = min(colors), max(colors)
    if vmin == -1 and vmax == 1: 
        print("(from_color_to_polarity) Input M is already a polarity matrix! | colors: {}".format(colors))
        return M
    
    # M: color matrix
    # use: for approximating ratings
    Po = np.zeros(M.shape)
    if sparse.issparse(M):
        mask = M.toarray()
    else:
        mask = np.copy(M)

    # only 3 possible values {1, -1, 0}
    # Po[(Po == codes['tp']) | (Po == codes['tn'])] = 1
    Po[mask > 0] = 1

    # Po[(Po == codes['fp']) | (Po == codes['fn'])] = -1
    Po[mask < 0] = -1

    # M[(M == codes['f']) ] = -1
    # Po[ mask  == codes['unk'] ] = 0
    return Po

def to_colored_preference_matrix(**kargs): 
    # use: for approximating ratings
    return to_colored_preference(**kargs)
def to_colored_preference(M, codes):
    import scipy.sparse as sparse
    # M: colored polarity
    # use: for approximating ratings
    if sparse.issparse(M):
        Po = M.toarray()
    else:
        Po = np.copy(M)

    # no-op for TP
    # Po[(Po == codes['tp'])]

    # no-op for TN
    # Po[Po == codes['tp'])]

    # FP, FN all considered negative
    Po[(Po == codes['fp']) | (Po == codes['fn'])] = -1
    Po[ Po == codes['unk'] ] = 0 

    return Po



# polarity classifier



        for j in pos_sample:  # foreach positive examples

            pos_i = np.where(Mc[:, j] == 1)[0] # TPs
            neg_i = np.where(Mc[:, j] == 0)[0]  # FNs

            # majority vote 
            counter = collections.Counter(Lhr[:, j])
            m_ratio = counter.most_common(1)[0][1]/(nbp+0.0)
            pn_ratio = counter['pos_label']/nbp    # should be relatively large
            max_vote = counter.most_common(1)[0][0]

            pos_i_sorted = np.argsort(-R[:, j])  # high to low
            # pv = np.sort(R[:, j])  # low to high
            # pv_low, pv_high = pv[:k], pv[-k:]   # lowest k and highest k
            # pv_range = np.hstack([pv_low, pv_high])  # features 
            pv_range = get_vars_vstats(j, p_th)
 
            # polarity positive: TP examples
            polarity = 1
            # ... it's possible that we do not have positive-polarity examples 
            npp = npn = 0
            if len(pos_i) > 0: # positive polarity

                # pos_i = np.random.choice(pos_i, min(k, len(pos_i)), replace=False)  # subsampling
                pos_i = [i for i in pos_i_sorted if i in pos_i][:k]  # top k (high to low)
        
                # BP dependent vars: 
                #    pv_st 
                #    query point 
                #    p_th[i]: p_threshold 
                #    m_ratio:  majority vote ratio 
                for i in pos_i:
                    # assert (Lh[i, j] == pos_label) and (Mc[i, j] == 1), "TP"   # ... ok

                    # distribution of proba estimates from the i-th BP according to sample types
                    pv_st = get_vars_hstats(i, p_th)

                    #########################
                    vars_query = [R[i, j], ] 
                    if tAdditionalVars:  
                        p_th_rel = R[i, j] - p_th[i]   # likely to be > 0
                        vars_query += [p_th_rel, max_vote, ]  # fraction of positive-polarity?
                        if Cr is not None: vars_query += [Cr[i, j], ]
                    #########################

                    # label = polarity, 
                    label = codes['tp'] if tMulticlass else polarity
                    yset.append(label) 
                    # class_label = Lr[j]

                    # define feature vector
                    #########################
                    fv = np.hstack([vars_query, pv_range, pv_st])
                    if nf == 0: nf = len(fv)
                    #########################

                    Xset.append( fv )  # size(Xset): r + 2 * k + 4 + 1
                    n_polarity_pos += 1  # number of positive polarity exmaples overall
                    npp += 1 # number of positive polarity exmaples for j-th data point
                    nTP += 1  # number of TPs
            ### end if
                
                # if there's no correct prediction for this positive example, then it becomes ambiguous
                # ... consider negative-polarity examples only when positive-polarity examples exist for this data point j

                # polarity negative: FN examples
                polarity = -1
                if len(neg_i) > 0:  # negative polarity
                    
                    npp_eff = int(npp * gamma)
                    # neg_i = np.random.choice(neg_i, min(nTP, len(neg_i)), replace=False)  # subsampling
                    neg_i = [i for i in pos_i_sorted if i in neg_i][:npp_eff]   # choose those that are higher (could have been positive)

                    for i in neg_i:
                        assert (Lh[i, j] == neg_label) and (Mc[i, j] == 0), "FN"

                        # features that characterize the distribution according to sample types
                        pv_st = []
                        for stype in sample_types: 
                            pv_st.extend(scopes[stype][i]['summary'])  # 5-number summary
                        # pv_st = [scopes[stype][i]['median'] for stype in sample_types] 

                        #########################
                        vars_query = [R[i, j], ] 
                        if tAdditionalVars:  
                            p_th_rel = R[i, j] - p_th[i]    # likely to be < 0 (too small); should have been higher
                            vars_query += [p_th_rel, max_vote, ]  # fraction of positive-polarity?
                            if Cr is not None: vars_query += [Cr[i, j], ]
                        #########################

                        # label = [polarity, ]
                        label = codes['fn'] if tMulticlass else polarity   # 'fn'
                        yset.append(label) 
                        # class_label = Lr[j]

                        fv = np.hstack([vars_query, pv_range, pv_st])
                        Xset.append( fv )  # size(Xset): r + 2 * k + 1
                        n_polarity_neg += 1  # number of positive polarity exmaples overall
                        npn += 1   # number of positive polarity exmaples for j-th data point
                        nFN += 1

            else: 
                # no positive polarity examples
                # ... then the entire column is ambiguous 
                pass

        # subsample the negative to match sample size of the positive
        neg_sample = np.where(Lr == neg_label)[0]
        n_neg = len(neg_sample)
        neg_sample = np.random.choice(neg_sample, min(n_pos, n_neg), replace=False)

        for j in neg_sample: 
            pos_i = np.where(Mc[:, j] == 1)[0] # TNs
            neg_i = np.where(Mc[:, j] == 0)[0]  # FPs

            # majority vote 
            counter = collections.Counter(Lhr[:, j])
            m_ratio = counter.most_common(1)[0][1]/(nbp+0.0)
            pn_ratio = counter['pos_label']/nbp    # should be relatively small
            max_vote = counter.most_common(1)[0][0]

            pos_i_sorted = np.argsort(R[:, j])  # low to high
            # pv = np.sort(R[:, j])  # low to high
            # pv_low, pv_high = pv[:k], pv[-k:]   # lowest k and highest k
            # pv_range = np.hstack([pv_low, pv_high])  # features 
            pv_range = common.five_number(R[:, j])

            # polarity positive: TN examples
            polarity = 1
            npp = npn = 0
            if len(pos_i) > 0:  
                
                # use n(TP) to control n(TN)
                # pos_i = np.random.choice(pos_i, min(nTP, len(pos_i)), replace=False)  # subsampling
                pos_i = [i for i in pos_i_sorted if i in pos_i][:k]  # top k lowest (low to high)

                # BP dependent vars: 
                #    pv_st 
                #    query point 
                #    p_th[i]: p_threshold 
                #    m_ratio:  majority vote ratio 
                for i in pos_i:  # foreach TN examples
                    assert (Lh[i, j] == neg_label) and (Mc[i, j] == 1), "TN"

                    # features that characterize the distribution according to sample types
                    pv_st = []
                    for stype in sample_types: 
                        pv_st.extend(scopes[stype][i]['summary'])  # 5-number summary
                    # pv_st = [scopes[stype][i]['median'] for stype in sample_types] 

                    #########################
                    vars_query = [R[i, j], ] 
                    if tAdditionalVars:  
                        p_th_rel = R[i, j] - p_th[i]    # likely to be < 0
                        vars_query += [p_th_rel, max_vote, ]  # fraction of positive-polarity?
                        if Cr is not None: vars_query += [Cr[i, j], ]
                    #########################

                    # label = [polarity, ]
                    label = codes['tn'] if tMulticlass else polarity
                    yset.append(label) 
                    # class_label = Lr[j]

                    fv = np.hstack([vars_query, pv_range, pv_st])
                    if nf == 0: nf = len(nf)

                    Xset.append( fv )  
                    n_polarity_pos += 1
                    npp += 1
                    nTN += 1
            ### end if 

                # polarity negative: FP examples
                # introudce negative polarity examples only when pp examples exist
                polarity = -1
                if len(neg_i) > 0: 

                    npp_eff = int(npp * gamma)
                    # neg_i = np.random.choice(neg_i, min(nTN, len(neg_i)), replace=False)  # subsampling
                    neg_i = [i for i in pos_i_sorted if i in neg_i][:npp_eff]   # choose those that are higher (could have been negative)

                    for i in neg_i:
                        assert (Lh[i, j] == pos_label) and (Mc[i, j] == 0), "FP"

                        # features that characterize the distribution according to sample types
                        pv_st = []
                        for stype in sample_types: 
                            pv_st.extend(scopes[stype][i]['summary'])  # 5-number summary
                        # pv_st = [scopes[stype][i]['median'] for stype in sample_types] 

                        #########################
                        vars_query = [R[i, j], ] 
                        if tAdditionalVars:  
                            p_th_rel = R[i, j] - p_th[i]    # likely to be > 0; too large
                            vars_query += [p_th_rel, max_vote, ]  # fraction of positive-polarity?
                            if Cr is not None: vars_query += [Cr[i, j], ]
                        #########################

                        # label = [polarity, ]
                        label = codes['fp'] if tMulticlass else polarity   # 'fp'
                        yset.append(label) 
                        # class_label = Lr[j]

                        fv = np.hstack([vars_query, pv_range, pv_st])
                        Xset.append( fv ) 
                        n_polarity_neg += 1
                        npn += 1 
                        nFP += 1
            else: 
                # no positive polarity examples
                # ... then the entire column is ambiguous 
                pass
        # ... training data generation complete

        msg = ""
        X, y = np.array(Xset), np.array(yset)
        n_features = X.shape[1]

        # 1a. feature transformation
        #############################################
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        #############################################

        assert X.shape[0] > len(pos_sample)+len(neg_sample), "sample size for polarity detection should be far greater than that of classification | n(polarity): {} >? n(cls): {}".format(X.shape[0], 
            len(pos_sample)+len(neg_sample))
        print("(polarity_modeling) number of features: {}".format(n_features))
        
        nX = X.shape[0]
        labels = np.unique(y) 

        msg += "(polarity_modeling) n(pos): {}, n(neg): {} | max: {}\n".format(n_pos, n_neg, max_size)
        msg += "...                 n_polarity(pos):{}, n_polarity(neg):{} | n(TP): {}, n(TN): {}\n".format(n_polarity_pos, n_polarity_neg, nTP, nTN)
        msg += "...                 n(tset): {} | dim(X): {}, dim(y): {}\n".format(nX, X.shape, y.shape)
        msg += "...                 n(label): {} | {}\n".format(len(labels), list(labels))
        # [log] pf2 
        # (polarity_modeling) number of features: 31
        # (polarity_modeling) n(pos): 265, n(neg): 2918 | max: 5000
        # ...                 n_polarity(pos):2635, n_polarity(neg):2511
        # ...                 n(tset): 5146 | dim(X): (5146, 31), dim(y): (5146,)
        print(msg)

        n_polarity_sample = nTP+nTN+nFP+nFN
        assert n_polarity_sample == n_polarity_pos + n_polarity_neg
        prior_polarity = np.array([nTP/(n_polarity_sample+0.0), nTN/(n_polarity_sample+0.0), nFP/(n_polarity_sample+0.0), nFN/(n_polarity_sample+0.0)])

        # now train a classifier
        if tMulticlass: 
            # note: the smaller the C, the stronger the regularization
            
            stacker = RandomForestClassifier(n_estimators=200, max_depth=10, bootstrap=True, random_state=0, class_weight='balanced')
            # LogisticRegression(penalty='l2', C=1, tol=1e-4, max_iter=200, solver='saga', multi_class='multinomial', class_weight='balanced')

            # candidates
            # LogisticRegression(penalty='l2', tol=1e-4, solver='sag', multi_class='multinomial', class_weight='balanced')
            # RandomForestClassifier(n_estimators=150, max_depth=8, bootstrap=True, random_state=0, class_weight='balanced')
            # QuadraticDiscriminantAnalysis(store_covariance=True)
            # ... variables are collinear
            # LinearDiscriminantAnalysis()
        else: 
            stacker = LogisticRegression(penalty='l2', C=0.1, tol=1e-4, solver='sag', class_weight='balanced') 
            # stacking.choose_classifier(p_classifier)  # e.g. log, enet, knn
        
        # 1b. model fitting
        #############################################
        model = stacker.fit(X, y) # remember to take transpose
        #############################################
        # ... how well does it fit the data?
        X = y = Xset = yset = None

        # now make test data 
        Xt, yt = [], []
        # M = np.zeros(T.shape)

        # [test]
        Mct, Lht = correctness_matrix(T, Lt, p_th)  # Mc is a (0, 1)-matrix
        # Lht = estimateLabelMatrix(T, p_th=p_th) 
        # ... Lht does not require true labels

        for j in range(T.shape[1]):

            Xset_j = []

            # majority vote 
            counter = collections.Counter(Lht[:, j])   # Lht does not require true labels
            m_ratio = counter.most_common(1)[0][1]/(nbp+0.0)
            pn_ratio = counter['pos_label']/nbp    # should be relatively large
            max_vote = counter.most_common(1)[0][0]
            
            # pv = np.sort(T[:, j])  # low to high
            # pv_low, pv_high = pv[:k], pv[-k:]   # lowest k and highest k
            # pv_range = np.hstack([pv_low, pv_high])  # features 
            pv_range = common.five_number(T[:, j])

            # BP dependent vars: 
            #    pv_st 
            #    query point 
            #    p_th[i]: p_threshold 
            #    m_ratio:  majority vote ratio 
            for i in range(T.shape[0]):   

                # features that characterize the distribution according to sample types
                pv_st = []
                for stype in sample_types: 
                    pv_st.extend(scopes[stype][i]['summary'])  # 5-number summary
                # pv_st = [scopes[stype][i]['median'] for stype in sample_types] 

                #########################
                vars_query = [T[i, j], ] 
                if tAdditionalVars:  
                    p_th_rel = T[i, j] - p_th[i]    
                    vars_query += [p_th_rel, max_vote, ]  # fraction of positive-polarity?
                    if Ct is not None: vars_query += [Ct[i, j], ]
                #########################

                # class_label = Lt[j]  # estimated class label in T

                Xset_j.append( np.hstack([vars_query, pv_range, pv_st]) )

                # [test]
                ################################################################################
                Xt.append(np.hstack([vars_query, pv_range, pv_st]))
                if Lht[i, j] == pos_label and Mct[i, j] ==1: 
                    yt.append(codes['tp'])
                elif Lht[i, j] == pos_label and Mct[i, j] ==0:
                    yt.append(codes['fp'])  # 'fp'
                elif Lht[i, j] == neg_label and Mct[i, j] == 1:
                    yt.append(codes['tn'])
                elif Lht[i, j] == neg_label and Mct[i, j] == 0:
                    yt.append(codes['fn']) # 'fn'
                ################################################################################
            
            fv = np.array(Xset_j)
            if j < 10: assert fv.shape[0] == T.shape[0] and fv.shape[1] == n_features

            # print("> fv <- T[:, j] | dim(fv): {}".format(fv.shape))
            # pvj = model.predict(fv) # model.predict_proba(fv)[:, 1]  # 1/foldCount worth of data
            pvj = model.predict(scaler.transform(fv))
            M[:, j] = pvj

        # [test]
        # Py1_x = model.predict_proba(scaler.transform(Xt))[:, 1]
        y_pred = model.predict(scaler.transform(Xt))
        atype = 'micro'  # 'macro'   
        # ... micro averaging is useful when classes different in sizes
        m_auc = common.multiclass_roc_auc_score(yt, y_pred, average=atype)
        print("(polarity_modeling) overall {} AUC on T: {}".format(atype, m_auc))

        # ... M determined 
        #     M_bar = None
        if tMulticlass: 
            M[(M == codes['tp']) | (M == codes['tn'])] = 1
            M[(M == codes['fp']) | (M == codes['fn'])] = -1
            # M[(M == codes['f']) ] = -1
            M[ M == codes['unk'] ] = 0

        npol = len(np.unique(M)); assert npol == 2, "Expecting 2 polarities but got {}".format(npol)

        # convert to -1 and 1 repr 

        # [test] evaluate preference score predictions
        if Lt is not None: 
            assert len(Lt) == T.shape[1] 
            # Mct, Lht = correctness_matrix(T, Lt, p_th)  # Mc is a (0, 1)-matrix
            # ... correctness and label matrix using true labels Lt
            ret = eval_polarity(M, Mct, Lht, verbose=True, name='T', neg_po=-1, title='(polarity_modeling) -- T given M --')

            # fmax metric for the estimated polarity matrix (M)
            msg = '(polarity_modeling) Compare estimated polarity matrix (M) with majority vote-induced preference matrix ...\n'
            pvt_max = predict_by_importance_weights(T, to_preference(M), aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
            fmax_t = common.fmax_score(Lt, pvt_max, beta = 1.0, pos_label = 1)
            msg += "... fmax(M): {}\n".format(fmax_t)
            # ... polarity performance of M 

            # how does it compare to majority votes? 
            # ... to_polarity(lh, Lh)
            ############################################################
            name = 'T'
            lh = estimateLabels(T, p_th=p_th, pos_label=pos_label) 
            Mct_max, Lht_max = correctness_matrix(T, lh, p_th)  # Mc is a (0, 1)-matrix
            
            # how many entries are different compared to True polarities? 
            n_agreed = np.sum(Mct_max == Mct)
            n_agreed_labeling = np.sum(Lt == lh)
            msg += '... {}max | n_agreed: {}, ratio: {} | lh vs Lt? n_agreed: {}, ratio: {}\n'.format(name, n_agreed, n_agreed/(Mct_max.shape[0] * Mct_max.shape[1]+0.0), 
                n_agreed_labeling, n_agreed_labeling/(len(lh)+0.0))
            pvt_max = predict_by_importance_weights(T, Mct_max, aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
            fmax_t = common.fmax_score(Lt, pvt_max, beta = 1.0, pos_label = 1)
            msg += "... fmax({}, majority): {}\n".format(name, fmax_t)

            # Mct_max = (Lht == lh).astype(int)  # use estimated labels and label matrix to compute polarity matrix
            ret = eval_polarity(to_polarity(Mct_max), Mct, Lht, verbose=True, name='Tmax', neg_po=-1, title='(polarity_modeling) -- {}: majority votes --'.format(name))
            # ... majority vote (T)   
            ############################################################

        # [test] pretend to predict the training data
        Xt, yt = [], []

        Mcr, Lhr = correctness_matrix(R, Lr, p_th)  # Mc is a (0, 1)-matrix
        Mr = np.zeros(R.shape)
        for j in range(R.shape[1]):

            Xset_j = []
            
            # majority vote 
            counter = collections.Counter(Lhr[:, j])   # Lht does not require true labels
            m_ratio = counter.most_common(1)[0][1]/(nbp+0.0)
            pn_ratio = counter['pos_label']/nbp    # should be relatively large
            max_vote = counter.most_common(1)[0][0]
            
            # pv = np.sort(R[:, j])  # low to high
            # pv_low, pv_high = pv[:k], pv[-k:]   # lowest k and highest k
            # pv_range = np.hstack([pv_low, pv_high])  # features 
            pv_range = common.five_number(R[:, j])

            for i in range(R.shape[0]):   

                pv_st = []
                for stype in sample_types: 
                    pv_st.extend(scopes[stype][i]['summary'])   
                # pv_st = [scopes[stype][i]['median'] for stype in sample_types] 
                
                #########################
                vars_query = [R[i, j], ] 
                if tAdditionalVars:  
                    p_th_rel = R[i, j] - p_th[i]    
                    vars_query += [p_th_rel, max_vote, ]  # fraction of positive-polarity?
                    if Cr is not None: vars_query += [Cr[i, j], ]
                #########################

                # class_label = Lt[j]  # estimated class label in T
                Xset_j.append( np.hstack([vars_query, pv_range, pv_st]) )

                # [test]
                ################################################################################
                Xt.append(np.hstack([vars_query, pv_range, pv_st]))
                if Lhr[i, j] == pos_label and Mcr[i, j] ==1: 
                    yt.append(codes['tp'])
                elif Lhr[i, j] == pos_label and Mcr[i, j] ==0:
                    yt.append(codes['fp'])  # fp
                elif Lhr[i, j] == neg_label and Mcr[i, j] == 1:
                    yt.append(codes['tn'])
                elif Lhr[i, j] == neg_label and Mcr[i, j] == 0:
                    yt.append(codes['fn'])  # fn
                ################################################################################
            
            fv = np.array(Xset_j)
            pvj = model.predict( scaler.transform(fv) ) # model.predict_proba(fv)[:, 1]  # 1/foldCount worth of data
            Mr[:, j] = pvj
        
        if tMulticlass: 
            Mr[(Mr == codes['tp']) | (Mr == codes['tn'])] = 1
            Mr[(Mr == codes['fp']) | (Mr == codes['fn'])] = -1
            # Mr[(Mr == codes['f']) ] = -1
            Mr[ Mr == codes['unk'] ] = 0

        npol = len(np.unique(Mr)); assert npol == 2, "Expecting 2 polarities but got {}".format(npol)
        
        ret = eval_polarity(Mr, Mcr, Lhr, verbose=True, name='R', neg_po=-1, title="(polarity_modeling) -- R given M --")
        # Py1_x = model.predict_proba(scaler.transform(Xt))[:, 1]
        y_pred = model.predict(scaler.transform(Xt))
        atype = 'micro'  # 'macro'   
        # ... micro averaging is useful when classes different in sizes
        m_auc = common.multiclass_roc_auc_score(yt, y_pred, average=atype)
        print("(polarity_modeling) overall {} AUC on R: {}".format(atype, m_auc))

        name = 'R'
        lh = estimateLabels(R, p_th=p_th, pos_label=pos_label) 
        Mcr_max, Lhr_max = correctness_matrix(R, lh, p_th)  # Mc is a (0, 1)-matrix
        # ... correctness and labeling matrix obtained via majority vote
        
        # how many entries are different compared to True polarities? 
        n_agreed = np.sum(Mcr_max == Mcr)
        n_agreed_labeling = np.sum(Lr == lh)
        msg += '... {}max | n_agreed: {}, ratio: {} | lh vs Lr? n_agreed: {}, ratio: {}\n'.format(name, n_agreed, n_agreed/(Mcr_max.shape[0] * Mcr_max.shape[1]+0.0), 
            n_agreed_labeling, n_agreed_labeling/(len(lh)+0.0))
        pvr_max = predict_by_importance_weights(R, Mcr_max, aggregate_func='mean') # np.average(A, weights=Mct, axis=0)
        fmax_r = common.fmax_score(Lr, pvr_max, beta = 1.0, pos_label = 1)
        msg += "... fmax({}, majority): {}\n".format(name, fmax_r)

        # if we were to use preference matrix as a polarity matrix
        ret = eval_polarity(to_polarity(Mcr_max), Mcr, Lhr, verbose=True, name='Rmax', neg_po=-1, title='(polarity_modeling) -- {}: majority votes --'.format(name))  
        # ... majority vote (R)   
        print(msg)    

    else:
        raise NotImplementedError
    # ... M encodes an estimate of sample types (e.g. TP, TN, negative, unknown)

    msg = ''
    msg += "(polarity_modeling) positive model M | n(pos): {}, n(neg): {}, n(neutral): {}\n".format(np.sum(M > 0), np.sum(M < 0), np.sum(M==0)) 
    if M_bar is not None: 
        msg += "...                 negative model M_bar | n(pos): {}, n(neg): {}, n(neutral): {}\n".format(np.sum(M_bar > 0), np.sum(M_bar < 0), np.sum(M_bar==0))
    print(msg)

    return Lh, M, M_bar


# majority vote 
            counter = collections.Counter(Lh[:, j])
            f_ratio = counter[pos_label]/(counter[neg_label]+1.)



n_pos = np.sum(Po == 1)
        n_neutral = np.sum(Po == 0)   # Mc == 0 at this point includes both negative and neutral
        assert np.sum(Mct==0) == np.sum(Po==0) and n_neutral>=0, "(toConfidenceMatrix) Inconsistent number of neutral particles? np.sum(Ct==0): {} =?= {}".format(np.sum(Mct==0), np.sum(Po==0))
        # ... only test split has neutral

        n_neg = np.sum(Po == -1) # Cui.shape[0] * Cui.shape[1] - n_pos - n_neutral
        assert n_neg > 0
        msg += '(toConfidenceMatrix) Mc after intro neutral particles | n(pos): {np}, n(neg): {nn}, n(neutral): {nc}\n'


def wmf_ensemble_preferred_ratings2(data, params, hyperparams={}, indices=[], vars=['wmfCV', 'wmfMetrics', ], kind='als', fold=-1, outer_fold=-1, null_marker=0, verbose=1, 
        project_path='?', piggyback=True, dev_ratio=0.2, max_dev=None, aggregation_methods=[], 
        post_hoc_analysis=False, save_data=False, enable_cluster_analysis=False):
    """
    
    Params
    ------
    fold: defined in a more generic sense than the 'fold' in wmf_ensemble_fold(). 
          specifically, 'fold' can refer to any of the following: 

           i) a CV fold ii) the index of iterations in random subsampling iii) other iteration index
    outer_fold: the iteration/fold number of the outer loop when wmf_ensemble_iter() is invoked for model selection (e.g. by model_select_core())

    indices: dataframe index and columns
    save: default set to False because this subroutine is typically used for model selection

    Memo
    ----
    1. use wmf_ensemble_fold() for CV iteration

    """
    def initmap(): # variable map is a nested dictionary in which each variable has it's own keys (kind) representing subcategories of the MF algorithm
        vmap = {} 
        for var in vars:
            vmap[var] = [] # {k: [] for k in kinds}   # var -> kind -> a list
        return vmap
    def get_class_stats(labels, pos_label=1, neg_label=0, ratio_imbalanced=0.1):
        # if fold == 0:  # fold is defined in a more generic sense: i) cv fold ii) the index of iterations in random subsampling iii) other iteration index
        
        # keys: n_pos, n_neg, r_pos, r_neg, 0/neg_label, 1/pos_label, is_balanced, r_minority, min_class
        stats = uc.classPrior(labels, labels=[neg_label, pos_label], ratio_ref=ratio_imbalanced)
        div('(wmf_ensemble_preferred_ratings2) class ratio | r(+): {rp}, r(-): {rn} | minority_class: {mc}'.format(rp=stats[pos_label], 
            rn=stats[neg_label], mc=stats['min_class']), symbol='#', border=1); print()

        return stats
    def verify_conditions():
        if params['setting'] in (1, 2, 7, 8): 
            assert params['supervised']
        if params['setting'] in (3, 4, 9, 10): 
            assert not params['supervised']
        if params['setting'] in ( 9, 10): 
            assert params['predict_probs'], "Setting 9 - 10 should attempt to re-estimate the entire T"
    def verify_confidence_matrix(C, X, L, p_threshold, U=[], Cbar=None, measure='rank', message='', test_cases=[], plot=False, index=0):  # closure: params
        # if params['setting'] in (11, 12): 
        #     assert Cbar is not None
        #     assert params['policy_opt'].startswith('trade')
        if plot: 
            # closure: alpha, beta
            analyze.plot_confidence_matrix(C, X, L, p_threshold, U=U, n_max=100, path=System.analysisPath, 
                measure=measure, target_label=None, alpha=params['alpha'], beta=2, index=index)

    def make_prediction_vector(X, L=[], M=None, policy=''):
        if not policy: policy=params['policy'] 
        if M is not None: assert len(L) == 0
        pv = uc.to_mean_vector(X, L=L, 
                M=M,  # message from training set when L is not accessible

                ratio_users=params['ratio_users'],  # filtering in the item direction 
                ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0),  # used for unsupervised mode 

                policy=policy,  # determining filtering dimension
                policy_threshold=params['policy_threshold'], # determining proba threshold

                    supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    fill=null_marker, fold=fold)
        return pv
    def name_params_setting(method_params=['F', 'A']):   # [todo]
        # MFEnsemble.abridge(meta, identifiers=method_params, replace=replacement, verify=True)

        # use MFEnsemble.params_to_ids
        return 'F{nf}A{a}'.format(nf=params['n_factors'], a=params['alpha'])

    from evaluate import Metrics, plot_roc, analyzePerf
    import analyze  # testing
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    from numpy import linalg as LA
    # from utils_als import implicit_als_cg, implicit_als
    import utils_cf as uc
    import utils_als as ua
    import utils_sys as us
    import stacking

    method = params.get('method', 'wmf')
    tMetaUsers = params.get('include_meta_users', False)  # if True, add extra meta classsifiers/users in the last rows of R and T
    vmap = initmap() # each call to this routine produces a new result

    # Note
    # if we don't re-configure cf_spec, then the information from the initial configuration (from sysConfig) 
    # will be lost => each thread sees its own version of cf_spec and its class definitions 
    ########################################################################################
    n_fold = FoldCount
    cf_spec.config(project_path=ProjectPath, domain=Domain, fold_count=FoldCount, bag_count=BagCount)
    ########################################################################################
    algorithm_setting = params.get('setting', '?') # see System.descriptions
    # verify_conditions()

    ### hyperparameters? 
    if len(hyperparams) > 0: params = {**params, **hyperparams}  

    ### Confidence Matrix
    # fold: iteration number (here, 'fold' is simply a repurposed variable borrowed from CV)
    # data: (R, T, L, Lt, U)-format
    if isinstance(data, DataFrame): # in this case, 'data' is a dataframe comprising a train-dev split 
        R, T, L_train, L_test, U =  uc.shuffle_split(data, ratio=dev_ratio, max_size=max_dev)
    else:
        # otherwise, assuming that shuffle-split operation had been invoked to produce the data input
        assert isinstance(data, (tuple, list)), "Invalid input data: %s" % data 
        print('(wmf_ensemble_preferred_ratings2) Input data is an n-tuple, whrere n={n}'.format(n=len(data)))
        R, T, L_train, L_test, U = data 
    
    n_train, n_test = len(L_train), len(L_test)
    n_samples = R.shape[1]+T.shape[1]; assert n_train+n_test == n_samples

    ### create extra prediction vectors (PVs) from R (say, mean vector) and attach these new PVs to R (piggyback)
    n_users, n_items = R.shape
    n_classifiers = int(n_users/BagCount)
    n_users0, n_items0 = n_users, n_items  # keep a copy of the original number of users and items
    n_users_test, n_items_test = T.shape
    n_meta_users = 0

    # preference parameters
    isPreferenceScore = True; assert params['policy_opt'].startswith('pref') # and not params['replace_subset'] 
    tPreferenceCalibration = params['binarize_pref'] # params.get('preference_calibration', True)
    pref_threshold = params.get('pref_threshold', -1)
    pref_threshold_test = params.get('pref_threshold_test', -1)

    tWeightedPrediction = params.get('weighted_output', False)
    tCalibrateTwoWay = params.get('two_way_calibration', False)
    tExplicit = params.get('explicit_mf', False) if not isPreferenceScore else False
    tApproximateRatingsViaPreference = params.get('approx_ratings_via_pref', False)
    ############################################################
    # piggy back extra meta-estimators
    meta_users = ['latent_mean', 'latent_mean_masked',]  # todo
    if tMetaUsers: 
        div(message='(wmf_ensemble_preferred_ratings2) Adding meta users (n={n})...'.format(n=len(meta_users)))
        
        # e.g. mean, and masked mean
        nU, nUT = n_users, n_users_test

        print('... augmenting T (by meta users)')
        mean_pv = make_prediction_vector(T, L=[], policy='none')   # policy: 'none' => no masking
        masked_mean_pv = make_prediction_vector(T, L=[], M=(R, L_train), policy=params['policy_test'])
        T = np.vstack((T, mean_pv, masked_mean_pv))
        n_users_test = T.shape[0]

        print('... augmenting R (by meta usrs)')
        mean_pv = make_prediction_vector(R, L_train, policy='none')
        masked_mean_pv = make_prediction_vector(R, L_train, policy=params['policy'])
        R = np.vstack((R, mean_pv, masked_mean_pv))
        n_users = R.shape[0]

        n_meta_users = n_users - nU
        assert n_meta_users == (n_users_test - nUT) == len(meta_users)
    ############################################################
    
    # estimate labels (lh) in the test split 
    pos_label = 1
    # Eu = identify_effective_users(Cr, L_train=L_train, fill=null_marker)
    Eu = []
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=params['policy_threshold']) 
    Lh = lh = uc.estimateLabels(T, L=[], Eu=Eu, p_th=p_threshold, pos_label=pos_label, ratio_small_class=params['ratio_small_class']) 

    X = np.hstack((R, T))
    L = np.hstack((L_train, Lh))

    ########################################################################################
    # A. confidence matrix for re-estimating proba values
    # CX = uc.evalConfidenceMatrix(X, L=L, U=U,  
    #             ratio_users=-1,  # params['ratio_users'], 
    #             # ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

    #             policy=params['policy'], # <<< filter axis for the training split in X
    #             policy_test=params['policy_test'],  # <<< filter axis for the test split in X

    #             policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
    #             policy_threshold=params['policy_threshold'],

    #                 supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
    #                     conf_measure=params['conf_measure'], 
    #                         alpha=params['alpha'], beta=params.get('beta', 1.0), 
    #                         # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
    #                         fill=null_marker, 
    #                             is_cascade=True, n_train=n_train,
    #                             suppress_negative_examples=params['suppress_negative_examples'], 

    #                             # polarity matrix parameters 
    #                             constrained=False,  # params.get('constrained', True),
    #                             stochastic=params.get('stochastic', True),
    #                             estimate_sample_type=params.get('estimate_sample_type', True),
    #                             labeling_model='simple', # params.get('labeling_model', 'simple'),
                                 
    #                             # for testing only 
    #                             fold=fold, path=System.analysisPath) # project_path=System.projectPath 
    # C0r, Cxr, Mcxr, p_threshold, *CX_res = CX    # Cx_bar is removed

    # B. confidence matrix for preference scores
    # compute confidence matrix for approximating preference
    CX = uc.evalConfidenceMatrix(X, L=L, U=U,  
            ratio_users=params['ratio_users'], 
            ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< filter axis for the training split in X
            policy_test=params['policy_test'],  # <<< filter axis for the test split in X

            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], 
                        alpha=params['alpha'], beta=params.get('beta', 1.0), 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, 
                            is_cascade=True, n_train=n_train,
                            suppress_negative_examples=params['suppress_negative_examples'], 

                            # polarity matrix parameters 
                            constrained=params.get('constrained', True),
                            stochastic=params.get('stochastic', True),
                            estimate_sample_type=params.get('estimate_sample_type', True),
                            labeling_model=params.get('labeling_model', 'simple'),
                             
                            # for testing only 
                            fold=fold, path=System.analysisPath) # project_path=System.projectPath 
    C0, Cx, Mcx, p_threshold, *CX_res = CX    # Cx_bar is removed
    ########################################################################################
    # ... Cx: confidence scores, zeros for neutral particles
    
    # verify_confidence_matrix(Cx, X, L, p_threshold, U=U, plot=False)
    
    div("(wmf_ensemble_preferred_ratings2) Completed C(X) | Cycle {cycle} |  dim(Cui): {dim}, filter_axis: {fdim} | conf_measure: {measure}, optimization: {opt} | predict ALL probabilities? {tval} | policy_threshold: {p_th}".format(
        cycle=(outer_fold, fold), 
            dim=str(Cx.shape), fdim=params['policy'], measure=params['conf_measure'], opt=params['policy_opt'], tval=params['predict_probs'], p_th=params['policy_threshold']), symbol='#', border=1)
    print('... Model | n_factors: {nf}, alpha: {a} | predict pref? {pref}, binarize? {bin}, supervised? {s}'.format(nf=params['n_factors'], a=params['alpha'], pref=isPreferenceScore, bin=params['binarize_pref'], s=params['supervised']))
    print('... Data | n_train: {ntr}, n_test: {nt}, ratio: {r}'.format(ntr=len(L_train), nt=len(L_test), r=(len(L_train)+len(L_test))/(len(L_test)+0.0)) )
    print('... Classifiers | n_users: {nu}, n_meta_users: {nmu} => n_original_users: {no}'.format(nu=n_users, nmu=n_meta_users, no=n_users-n_meta_users))
    ### Very ALS algorithms here ###
    #   Latent Factors
    #    >>> the input Cui must be in the form of 'alpha * f(R)' for this call (instead of 1 + alpha * f(R)) => minimal confidence at 0
    
    div('... (ALS 1) Going into the ALS loop on TRAINING data (R): {dim} | Cycle: ({fo}, {f})'.format(dim=R.shape, f=fold, fo=outer_fold))
    piggyback_msg = "+  Cycle: ({fo}, {f}) | setting: {setting}".format(fo=outer_fold, f=fold, setting=algorithm_setting).rjust(5)   # keep track of the thread from which the iteration 
    
    ########################################################################################
    # ... determining Cn (n: neutral)
    Cn = None
    Po = Mcx # or Mcxr?    # choose which polarity estimator and which confidence score
    tMaskNeutral = True
    if tExplicit:  # for tExplicit to be True, isPreferenceScore must be False
        print("(wmf_ensemble_preferred_ratings2) Using UNWEIGHTED Cw, non-weighted MF to approximate ratings ...")
        Cn = np.ones(Cx.shape)  
        if scipy.sparse.issparse(Cx): 
            if tMaskNeutral: Cn[Po.toarray()==0] = 0      # masking neutral
            Cn[Po.toarray()==-1] = 0     # masking negative
            Cn = sparse.csr_matrix(Cn)
        else: 
            if tMaskNeutral: Cn[Po==0] = 0  # masking neutral
            Cn[Po==-1] = 0  # masking negative
    else: 
        print("(wmf_ensemble_preferred_ratings2) Using WEIGHTED Cw, weighted MF to approximate ratings ...")
        # otherwise, we retain the weight but masking the neutral and negative examples so that they do not enter the cost function when approximating "ratings"
        Cn = np.zeros(Cx.shape)+C0 # toarray()  # copying
        # ... If Cx is sparse, then Cn+Cx is no longer sparse but of matrix type (if without .toarray())
        if scipy.sparse.issparse(Cx): 
            if tMaskNeutral: Cn[Po.toarray()==0] = 0  # masking neutral  
            Cn[Po.toarray()==-1] = 0  # masking negative
            Cn = sparse.csr_matrix(Cn)
        else: 
            if tMaskNeutral: Cn[Po==0] = 0   # masking neutral
            Cn[Po==-1] = 0  # masking negative
    n_neutral_cx, n_neutral_cn = np.sum(Cx == 0), np.sum(Cn == 0)
    # assert n_neutral_cn > n_neutral_cx, \
    #     "Masked entries of C when approximating 'ratings' must be more than those of C when approximating preference | n_masked(Cn): {}, n_masked(Cx): {}".format(n_neutral_cn, n_neutral_cx)
    # ... Cw: is a masked version of Cx, where both neutral and negative poloarity examples have zero weights
    print("... Cx vs Cn | n_neutral_cx: {}, n_neutral_cn: {}".format(n_neutral_cx, n_neutral_cn))
    P = Q = None
    if not tApproximateRatingsViaPreference: 
        print('... (1) approximating ratings (X) via Cn (n_masked: {}) | Cycle ({fo}, {fi})'.format(n_neutral_cn, fo=outer_fold, fi=fold))
        P, Q, *Xh_errs = ua.implicit_als(Cn, features=params['n_factors'], 

                                iterations=params['n_iter'],
                                lambda_val=System.lambda_val,  # 0.8 by default

                                # label_confidence=Cx_bar, 
                                # polarity=Mcx,   # polarity is not used when approximting ratings
                                ratings=X, labels=L,
                                policy='rating', message=piggyback_msg, ret_rmse=True)
        Xh_err, Xh_err_weighted = Xh_errs

        ne = 2
        e_pri, e_post = np.mean(Xh_err[:ne]), np.mean(Xh_err[-ne:])
        e_del = (e_pri-e_post)/e_pri * 100
        print('... (ALS 1) Complete | rmse: {e1} -> {e2} (n_errs: {n}, %reduction: %{e_del}) | wrmse: {ew1} -> {ew2}  ... (verify) #'.format(e1=e_pri, e2=e_post, 
                       e_del=e_del, ew1=np.mean(Xh_err_weighted[:ne]), ew2=np.mean(Xh_err_weighted[-ne:]), n=len(Xh_err) ))
    else: 
        print("... (1) defer ratings (X) approximation after preference scores are obtained ...")
    ########################################################################################

    print('... (2) estimating preferences (X) via Cx (n_masked: {}) | Cycle ({fo}, {fi})'.format(n_neutral_cx, fo=outer_fold, fi=fold))
    Pp, Qp, *Xp_errs = ua.implicit_als(Cx, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            # label_confidence=Cx_bar, 
                            polarity=Mcx, 
                            ratings=X, labels=L,
                            policy='preference', message=piggyback_msg, ret_rmse=True)

    ########################################################################################
    
    # e.g. 50 users, 3979 items with nf=100 > dim(P): (50, 100), dim(Q): (3979, 100)
    if P is not None and Q is not None: 
        assert P.shape[0] == X.shape[0] and P.shape[1] == params['n_factors']
        assert Q.shape[0] == X.shape[1] and Q.shape[1] == params['n_factors']
        assert Pp.shape == P.shape and Qp.shape == Q.shape

    ### ALS evaluation (RMS)
    print('(wmf_ensemble_preferred_ratings2) Prior to reconstruct_by_preferernce() | pref_threshold given? {}'.format(pref_threshold))
    # -- A. calibrate together
    factors = (P, Q) if not tApproximateRatingsViaPreference else None
    Xh, Xp, pref_threshold, *res = \
        reconstruct_by_preference(C0, X, factors=factors, prefs=(Pp, Qp), labels=L, 
                test_labels=np.hstack((L_train, L_test)),
                binarize=tPreferenceCalibration, 
                    p_threshold=p_threshold, 
                    pref_threshold=pref_threshold, # we don't know the preference threshold yet
                    policy_calibration=params['policy_calibration'],
                        is_cascade=True, n_train=n_train, 
                           replace_subset=True, replace_all=False, params=params, null_marker=0, 
                           name='X', verify=True, index=(outer_fold, fold), 

                               # only relevant when factors=(P, Q) has not been computed or not given
                               n_factors=params['n_factors'], n_iter=params['n_iter'],
                                   unweighted=tExplicit, message=piggyback_msg)
    if len(res) > 0: P, Q = res[0], res[1]
    # ... Now we have Xh, Xp, (Pp, Qp), and (P, Q)
    
    delta_X = LA.norm(Xh-X, 'fro')
    Rh, Th = Xh[:,:n_train], Xh[:,n_train:]
    Rp, Tp = Xp[:,:n_train], Xp[:,n_train:]
    pref_threshold_test = pref_threshold
    # ... Xh is a (re-estimated) proba matrix; Xp: preference matrix (binarized)
    
    # -- B. cailbrate separately
    if tCalibrateTwoWay: 
        Cr, Ct = Cx[:,:n_train], Cx[:,n_train:]
        Qr, Qt = Q[:n_train, :], Q[n_train:, :]  # row(Q) ~ items/data
        Qpr, Qpt = Qp[:n_train,:], Qp[n_train:, :]
        Lh_R, Lh_T = L_train, L[n_train:]

        Rh, Rp, pref_threshold = \
            reconstruct_by_preference(Cr, R, factors=(P, Qr), prefs=(Pp, Qpr), labels=Lh_R, 
                    # test_labels=np.hstack((L_train, L_test)),
                    binarize=tPreferenceCalibration, 
                        p_threshold=p_threshold, 
                        pref_threshold=pref_threshold, # we don't know the preference threshold yet
                        policy_calibration=params['policy_calibration'],
                            is_cascade=False, 
                               replace_subset=True, replace_all=False, params=params, null_marker=0, name='R', verify=True, index=(outer_fold, fold))
        # ... Th is a proba matrix; Xp: preference matrix (binarized)
        Th, Tp, pref_threshold_test = \
            reconstruct_by_preference(Ct, T, factors=(P, Qt), prefs=(Pp, Qpt), labels=Lh_T, 
                    test_labels=L_test,
                    binarize=tPreferenceCalibration, 
                        p_threshold=p_threshold_test, 
                        pref_threshold=-1, # we don't know the preference threshold yet
                        policy_calibration=params['policy_calibration'],
                            is_cascade=False, 
                               replace_subset=True, replace_all=False, params=params, null_marker=0, name='T', verify=True, index=(outer_fold, fold))
        print('(wmf_ensemble_preferred_ratings2) two-way calibration | th(R): {} <?> th(T): {}'.format(pref_threshold, pref_threshold_test))

    print('(wmf_ensemble_preferred_ratings2) preference thresholds | th(R): {} ~? th(T): {}'.format(pref_threshold, pref_threshold_test))
    assert Rh.shape == R.shape, "dim(R): {}, dim(Rh): {}".format(R.shape, Rh.shape)
    assert Th.shape == T.shape, "dim(T): {}, dim(Th): {}".format(T.shape, Th.shape)

    # estimate ratio_small_class using the training split
    class_stats = get_class_stats(L_train, pos_label=1, neg_label=0, ratio_imbalanced=0.1)

    # ... CF-transform T to get Th (using the classifier/user vectors learned from the training set (R))
    div("(wmf_ensemble_preferred_ratings2) Completed rating matrix reconstruction | Cycle: ({fo}, {fi}) | preference scores? {tval}, action='{act}'".format(
        fo=outer_fold, fi=fold, tval=isPreferenceScore, act='Replace Subset')) # predict => predict probabilities

    # classifier/user weights (usually just use confidence matrix)
    Cw = Cwt = None
    if tWeightedPrediction: 
        Cw, Cwt = C0[:,:n_train], C0[:,n_train:]
        print('(wmf_ensemble_preferred_ratings2) using weighetd output via confidence matrix | dim(Cwt): {}'.format(Cwt.shape))
    else: 
        Cw, Cwt = Xp[:,:n_train], Xp[:,n_train:]  # this will result in discarding new estimates wherever preference scores == 0
        print('(wmf_ensemble_preferred_ratings2) using only preference matrix (non-weigthed) for making final predictions ...')

    # X: (R, T, L_train, L_test, U)
    n_samples_reconstructed = Rh.shape[1]+Th.shape[1]
    ##############################################################################################################
    # ... prediction 
    if not aggregation_methods: aggregation_methods = System.aggregation_methods # e.g. ['mean', 'median', 'log', ]  
    # pv_mean = uc.combiner(Th, aggregate_func='mean')           

    ##############################################################################################################
    # ... input: 
    #     Rh: new ratings by replacing unreliable entries with new estimates (reliabilty is estimated by preference Rp)
    #     Rp: (estimated) preference scores in R
    #     Th: new ratings for T 
    #     Tp: (estimated) prefeerence scores in T
    file_types = ['prior', 'posterior', ]
    if post_hoc_analysis:   # only triggered after model selection cycle is complete and 'best params' has been determined
        
        # note that we shall save the data only after model selection loop is completed
        assert outer_fold >= 0 and fold == -1, "Intended action: only save the final model after model selection is complete (outer fold: {fo}, inner fold: {fi}".format(fo=outer_fold, fi=fold)
        the_params = name_params_setting(method_params=['F', 'A'])

        div('(wmf_ensemble_preferred_ratings2) Running posthoc analysis | algorithmic setting: {s}'.format(s=algorithm_setting), symbol='%') 
        # MFEnsemble.save_meta_tset((Rh, Th, L_train, L_test, U), fold=fold, method=method, params=params, verbose=verbose)

        pv_t = pv_th = pv_pref = None
        if tMetaUsers: 
            # drop meta users 
            R = R[:-n_meta_users]
            Rh = Rh[:-n_meta_users]
            Rp = Rp[:-n_meta_users]
            T, pv_t = T[:-n_meta_users], T[-n_meta_users:] 
            Th, pv_th = Th[:-n_meta_users], Th[-n_meta_users:]
            Tp, pv_pref = Tp[:-n_meta_users], Tp[-n_meta_users:]
            assert Th.shape[0] + pv_th.shape[0] == n_users_test
            assert pv_t.shape[1] == pv_th.shape[1] == len(L_test)

            P = P[:-n_meta_users]

            if Cwt is not None: 
                Cwt = Cwt[:-n_meta_users]
                assert Th.shape == Cwt.shape, "dim(Th): {}, dim(Cwt): {}".format(Th.shape, Cwt.shape)

        ### save predictions
        # Tp is a binary matrix
        n_uniq_pref = len(np.unique(Tp))
        assert n_uniq_pref == 2, "(wmf_ensemble_preferred_ratings2) Tp is not binary or is degenerated | n_uniq(Tp): {} | values(Tp): {}".format(n_uniq_pref, np.unique(Tp))
        # Th is a continuous rating matrix
        n_uniq = len(np.unique(Th))
        assert n_uniq > 2, "(wmf_ensemble_preferred_ratings2) Unique ratings should be >> 1 | u_uniq(T): {}".format(n_uniq)
        # ... although in this routine, Th is expected to be already a probability matrix (conditioned on the preference matrix).

        # the_params = name_params_setting(method_params=['F', 'A'])
        dataset = {'prior': [R, T], 'posterior': [Rh, Th]}
        for file_type in file_types: 
            X_train, X_test = dataset[file_type]
            for i, aggr in enumerate(aggregation_methods):    # defined earlier e.g. ['mean', 'median']
                pn = '{ftype}_{model}'.format(ftype=file_type, model=aggr)  # pn: predictor name
                
                # TODO: if aggr in ['logistic', ...]  # need to further train on X_train and test on X_test
                if aggr in ['mean', 'median', ]:   # System.simple_aggregation

                    # use Xh
                    if file_type.startswith('pri'): 
                        pv = uc.combiner(T, aggregate_func=aggr)  # T 
                    else: 
                        # 1. use Xh
                        # pv = uc.combiner(Th, weights=Cwt, aggregate_func=aggr)  # Th
                        
                        # 2. use Xp
                        pv = uc.predict_by_preference(T, Tp, W=Cwt, name='weighted preference', aggregate_func=aggr, fallback_on_low_weight=False, verify=True) 

                else:
                    ### put stacker code here! 

                    stacker = stacking.choose_classifier(aggr)  # e.g. log, enet, knn
                    model = stacker.fit(X_train.T, L_train) # remember to take transpose
                    pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data

                    # stacker training on Rh and testing on Th does not make sense
                    # if file_type.startswith('pri'): 
                    #     stacker = stacking.choose_classifier(aggr)  # e.g. log, enet, knn
                    #     model = stacker.fit(X_train.T, L_train) # remember to take transpose
                    #     pv = model.predict_proba(X_test.T)[:, 1]  # 1/foldCount worth of data
                    # else: 
                    #     pv = []

                if len(pv) == len(L_test): # if prediction vector is not null
                    y_pred, y_label = pv, L_test
                    vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

        ### save predictions from the meta users (and treat these data as if they came from an aggregation method)
        if tMetaUsers: 
            # vmap['prior_mean'], vmap['posterior_mean'] = {}, {}
            predictions = {'prior': pv_t, 'posterior': pv_th}  # each prediction dataframe consists of prediction vectors from n meta_users
            for file_type in file_types: 
                for i, meta_user in enumerate(meta_users): 
                    pn = '{ftype}_{model}'.format(ftype=file_type, model=meta_user) 
                    pv = predictions[file_type]

                    y_pred, y_label = pv[i], L_test
                    vmap[pn] = DataFrame({'prediction':y_pred,'label':y_label, 'method':pn, 'fold': outer_fold, 'params': the_params}, index=range(len(y_pred)))

            # for file_type, pv in predictions.items(): 
            #     pn = '{ftype}_{model}'.format(ftype=file_type, model='mean') # pn: predictor name
            #     vmap[pn] = {}
            #     for i, meta_user in enumerate(meta_users): # ['latent_mean', 'latent_mean_masked',]
            #         y_pred, y_label = pv[i], L_test
            #         vmap[pn][meta_user] = DataFrame({'prediction':y_pred,'label':y_label, 'method':meta_user, 'fold': fold}, index=range(len(y_pred)))
    
            print('... saved predictions for data type: {dtype} & meta users (n={n}) to dataframes'.format(dtype=file_type, n=len(meta_users)))
         
        #### optionally, run cluster analysis
        filter_axes = ['user', ]  # 'item'
        if enable_cluster_analysis: 
            # if meta users were included, consider R[:-n_meta_users] 
            for fdim in filter_axes: 
                
                # training split; if meta users were included, consider R[:-n_meta_users] 
                run_cluster_analysis(P, U=U, X=X, kind='user', n_clusters=n_users, index=outer_fold, params=params,
                    save=False, save_plot=True, file_type='train') # usually only at the training of the final model (i.e. after the best hyperparameters are determined; see wmf_ensemble_model_select())

        dset_id = MFEnsemble.get_dset_id(method='wmf', params=params)

        vmap['dset_id'] = dset_id 
        vmap['best_params_inner'] = the_params

    ##############################################################################################################
    if save_data: 
        div('(wmf_ensemble_preferred_ratings2) Output: saving transformed training and test sets (size: n(R): {nR}, n(T): {nT}), total size: {N}| delta(R): {dR}, delta(T): {dT} | algorithmic setting: {s}'.format(s=algorithm_setting, 
                    nR=Rh.shape[1], nT=Th.shape[1], N=n_samples_reconstructed, dR=delta_R, dT=delta_T), symbol='>')

        # test set
        MFEnsemble.save_data((T, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='prior', verbose=verbose, subsampling=True)  # prior data 
        MFEnsemble.save_data((Th, L_test, U), fold=outer_fold, indices=indices, base_method=method, dset_id=dset_id, dtype='posterior', verbose=verbose, subsampling=True)  # posterior data  (~ prediciton)

        # training set
        MFEnsemble.save_data((R, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-prior', verbose=verbose, subsampling=True)
        MFEnsemble.save_data((Rh, L_train, U), fold=outer_fold, indices=[], base_method=method, dset_id=dset_id, dtype='train-posterior', verbose=verbose, subsampling=True)
    

    ### kind: specific MF algorithm
    if not kind: kind = 'als'
    wmfMetrics, wmfCV = [], [] # vmap['wmfMetrics'], vmap['wmfCV']

    # naming convention: method_id = '{prefix}_{id}_{learner_type}' # e.g. rf_bp_stacker
    aggregate_mode = params.get('policy_ms_model', 'mean')
    aggregate_func = params.get('policy_aggregate_func', 'mean')
    div("(wmf_ensemble_preferred_ratings2) Comparison of model parameters | aggregate_func: {func}, mode: {mode}".format(func=aggregate_func, mode=aggregate_mode)) 
    # {stacker}.S-{dataset}-{suffix}
    # method_id = '{prefix}_{id}_{learner_type}'.format(prefix='wmf_%s' % kind, id=MFEnsemble.get_method_id(method, kind, params=params), learner_type=aggregate_func)
    method_id = "{prefix}.W-{id}-{suffix}".format(prefix=aggregate_func, id=MFEnsemble.get_method_id(method, kind, params=params), suffix=kind)
    ### model evaluation
    #   a. use mean proba in T vs labels as a summary score
    #   b. but we can compute a performance score for each base method (and then take the average)

    # header = MFEnsemble.header_model_evaluation # ['method', 'score', 'posterior_score', 'mean_score', ] ... used in evaluate.analyzePerf2()

    ############################################################################################################################
    # PerformanceMetrics object in wmfMetrics is the basis for selecting the best hyperparameter
    # 
    pv = uc.predict_by_preference(T, Tp, W=Cwt, name='weighted preference', aggregate_func='mean', fallback_on_low_weight=False, verify=True)
    prediction = pv # or Th
    performance_metrics, pv = analyzePerf(L_test, prediction,   # or (L_test, Th)
                                    method=method_id, aggregate_func=aggregate_func,
                                        weights=Cwt,  
                                        outer_fold=outer_fold, fold=fold,  # keep track of the iteration (debugging only when comparing Th and T)
                                        train_data=(Rh, L_train),  # only used in stacking mode
                                        mode=aggregate_mode)  # pass T=T to compare with T
    wmfMetrics.append( performance_metrics ) # optional params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
    wmfCV.append( (L_test, pv) )   
    vmap['wmfMetrics'], vmap['wmfCV'] = wmfMetrics, wmfCV

    ##########################################################
    # other variables that may be of interest 
    # >>> piggyback inputs
    if piggyback: 
        # args, posargs = us.arguments()
        vmap['hyperparams'] = hyperparams  # keep track of hyperparameters for convenience 
    # vmap['fold'] = fold
    ##########################################################

    print("(wmf_ensemble_preferred_ratings2) Ending cycle: ({fo}, {fi}) at setting {case} > returning vmaps: {keys} ... (verify) ".format(fo=outer_fold, fi=fold, 
        case=algorithm_setting, keys=vmap.keys()))

    # keys of vmap are the variables to return to caller: 
    #   i) saved in every cycle: wmfMetrics, wmfCV, hyperparams
    #  ii) saved only when training the final model after n cycles of model selection is completed: 
    #      dset_id, 
    #      best_params_nruns 
    return vmap


M = M_bar = None
    if estimate_sample_type: 
        Lh, M, M_bar = polarity_preprocessing(R, Lr, p_th, T, policy=policy, constrained=constrained)
    else:  
        Mc, Lh = correctness_matrix(R, Lr, p_th)  # Mc is a (0, 1)-matrix

### filter along item axis by polarity


def estimate_polarity(R, Lr, p_th, T, policy='median', 
        constrained=True, stochastic=False,
        k_upper=-1, k_lower=-1, k_max=-1, k_min=-1, verbose=True):
    def is_within(v, scope): 
        if not scope: return False
        return (v >= scope['min']) and (v <= scope['max'])
    def is_within_or_above(v, scope): 
        if not scope: return False
        return ((v >= scope['min']) and (v <= scope['max'])) or (v > scope['max'])
    def is_within_or_below(v, scope): 
        if not scope: return False
        return ((v >= scope['min']) and (v <= scope['max'])) or (v < scope['min'])
    def is_above(v, scope, k='mean'):
        if not scope: return False
        return v >= scope[k]
    def is_below(v, scope, k='mean'):
        if not scope: return False
        return v <= scope[k]  

    Mc, Lh = correctness_matrix(R, Lr, p_th)  # Mc is a (0, 1)-matrix
    
    predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
    predict_neg = (Lh == 0)

    cells_tp = (Mc == 1) & predict_pos   # estimated
    cells_tn = (Mc == 1) & predict_neg
    cells_fp = (Mc == 0) & predict_pos
    cells_fn = (Mc == 0) & predict_neg

    # estimate proba range for each classifier based on sample type (TP, TN, FP, FN)
    sample_types = ['tp', 'tn'] + ['fp', 'fn']
    codes = {'tp': 1, 'tn': 2, 'fp': 3, 'fn': 4 }
    scopes = {st: {} for st in sample_types}   # scope['tp'][0]: to be true positive, 0th classifier must have this proba range
    for i in range(R.shape[0]):  # foreach classifier
        scopes['tp'][i] = {}
        
        # TPs
        v = R[i, :][cells_tp[i, :]]
        if len(v) > 0: 
            scopes['tp'][i] = {'min': np.min(v), 'max': np.max(v), 'mean': np.mean(v), 'median': np.median(v)}   # min, max, mean, median

        # TNs 
        v2 = R[i, :][cells_tn[i, :]]
        if len(v2) > 0: 
            scopes['tn'][i] = {'min': np.min(v2), 'max': np.max(v2), 'mean': np.mean(v2), 'median': np.median(v2)}   
        
        # ... positive polarity candidates 
        assert scopes['tp'][i]['median'] != scopes['tn'][i]['median'] 

        # FPs ~ TPs
        v3 = R[i, :][cells_fp[i, :]]
        if len(v3) > 0: 
            scopes['fp'][i] = {'min': np.min(v3), 'max': np.max(v3), 'mean': np.mean(v3), 'median': np.median(v3)}

        # FNs ~ TNs
        v4 = R[i, :][cells_fn[i, :]]
        if len(v4) > 0: 
            scopes['fn'][i] = {'min': np.min(v4), 'max': np.max(v4), 'mean': np.mean(v4), 'median': np.median(v4)}   
        # ... negative polarity candidates
    
    # [test]
    msg = '(estimate_polarity) Policy: {}\n'.format(policy)
    for st in sample_types: 
        msg += '\n--- Sample Type: {} ---\n'.format(st.upper())
        for i in range(R.shape[0]):  # foreach classifier
            if i % 2 == 0: 
                if len(scopes[st][i]) > 0: 
                    msg += '... type: {} | min: {}, max: {}, mean: {}\n'.format(st.upper(), scopes[st][i]['min'], scopes[st][i]['max'], scopes[st][i]['mean'])
    print(msg)

    tConstrained = True if constrained and ((k_upper > 0) or (k_lower > 0)) else False
    # now scan through T while looking up table scope to determine if entries in T should be considered as positive or negative or neutral/unknown
    M = np.zeros(T.shape)
    if policy in ('mean', 'median', ): 

        for j in range(T.shape[1]):  # foreach item/datum
            for i in range(T.shape[0]):  # foreach user/classifier
                # positive? 
                is_tp = is_above(T[i, j], scopes['tp'][i], k=policy)    
                is_tn = is_below(T[i, j], scopes['tn'][i], k=policy)

                if is_tp != is_tn: # only one of them is true
                    M[i, j] = codes['tp'] if is_tp else codes['tn'] 

                elif not is_tp and not is_tn: 
                    M[i, j] = -1   # negative, either FP or FN depending on the label
                else: 
                    # both are true, then it's a neutral
                    # no-op 
                    # M[i, j] = 0
                    pass

    elif policy.startswith('interv'):  # interval 
        for j in range(T.shape[1]): 
            for i in range(T.shape[0]):

                # positive? 
                is_tp = is_within_or_above(T[i, j], scopes['tp'][i]) 
                is_tn = is_within_or_below(T[i, j], scopes['tn'][i]) 

                if is_tp != is_tn: # only one of them is true
                    M[i, j] = codes['tp'] if is_tp else codes['tn']

                elif not is_tp and not is_tn: 
                    M[i, j] = -1
                    

                else: 
                    # both are true, then it's a neutral
                    # no-op 
                    # M[i, j] = 0
                    pass
    else: 
        raise NotImplementedError
    # ... M encodes an estimate of sample types (e.g. TP, TN, negative, unknown)

    ### resolve sample types 
    # ... M eventually should only consists of 3 types of polarity: 0/neutral, 1/poistive, -1/negative
    #     positive => high confidence of being in {TP, TN}, negative: incorrect {FP, FN}, neutral: unknown 
    tStochastic = stochastic
    print('(estimate_polarity) Constrained? {}, Stochastic? {}'.format(constrained, stochastic))

    M2 = np.zeros(T.shape)
    if not tConstrained: 
        
        n_conflict_evidence = n_no_positive = 0
        n_tp_dominant = n_tn_dominant = 0

        for j in range(T.shape[1]):  # foreach item/datum
            tp_i = np.where(M[:, j] == codes['tp'])[0] 
            tn_i = np.where(M[:, j] == codes['tn'])[0]
            neg_i = np.where(M[:, j] == -1)[0]
            ntp_i, ntn_i = len(tp_i), len(tn_i)
            
            # k_upper_tp = k_upper_tn = 0
            if (ntp_i > 0) and (ntn_i > 0): 
                n_conflict_evidence += 1

            if ntp_i == 0 and ntn_i == 0: n_no_positive += 1  

            # T[:, j][M[:, j] == codes['tp']]
            ########################################################
            if tStochastic: 
                m_tp_dominant = ntp_i / (ntp_i + ntn_i+0.0)
                p = np.random.uniform(0, 1, 1)[0]
                is_tp_dominant = True if p <= m_tp_dominant else False
            else:
                # majority vote; deterministic 
                is_tp_dominant = True if ntp_i > ntn_i else False
            ########################################################

            if is_tp_dominant: 
                support_pos = tp_i 
                support_neg = neg_i

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1

                n_tp_dominant += 1
            else:  
                support_pos = tn_i
                support_neg = neg_i

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1
        
    else: # constrained
        # M2 = np.zeros(T.shape)

        n_conflict_evidence = n_no_positive = 0
        n_tp_dominant = n_tn_dominant = 0
        for j in range(T.shape[1]):  # foreach item/datum

            tp_i = np.where(M[:, j] == codes['tp'])[0] 
            tn_i = np.where(M[:, j] == codes['tn'])[0]
            neg_i = np.where(M[:, j] == -1)[0]
            ntp_i, ntn_i = len(tp_i), len(tn_i)
            
            # k_upper_tp = k_upper_tn = 0
            if (ntp_i > 0) and (ntn_i > 0): 
                n_conflict_evidence += 1
                # k_upper_tp = k_upper_tn = k_upper/2.0
                # k_lower_tp = k_lower_tn = k_lower/2.0
            # assert not ((ntp_i > 0) and (ntn_i) > 0), "Conflicting evidence in row {}, where n(tp): {}, n(tn): {}".format(i, ntp_i, ntn_i)
            # ... it's possible that some classfieris consider j as TPs while others consider it as TNs (simply because the value falls within the range)
            # ... which ones are more likely to be true?
            if ntp_i == 0 and ntn_i == 0: n_no_positive += 1  

            # is_tp_dominant? 
            ########################################################
            if tStochastic: 
                m_tp_dominant = ntp_i / (ntp_i + ntn_i+0.0)
                p = np.random.uniform(0, 1, 1)[0]
                is_tp_dominant = True if p <= m_tp_dominant else False
            else:
                # majority vote; deterministic 
                is_tp_dominant = True if ntp_i > ntn_i else False
            ########################################################

            # T[:, j][M[:, j] == codes['tp']]
            if is_tp_dominant: 
                assert len(tp_i) > 0  
                rows = np.argsort(-T[:, j])  # np.argsort(R[:, j])[:-k-1: -1] # choose indices of k highest probs
                # ... high to low
                
                # 1. positive examples: the larger the better
                support_pos = [i for i in rows if i in tp_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tp_i][:k_upper]
                # ... if k_upper_tp > 0, then we have conflicting evidence

                # 2. tn examples are demoted to neutral

                # 3. negative examples remain negative (not TP nor TN)
                support_neg = [i for i in rows[::-1] if i in neg_i][:k_lower]

                # [test]
                # if j % 100 == 0: assert np.min(T[support_pos,j]) >= np.max(T[support_neg,j]), "min(pos): {} <? max(neg): {}".format(np.min(T[support_pos,j]), np.max(T[support_neg,j]))

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1

                n_tp_dominant += 1
            else:  
                rows = np.argsort(T[:, j])
                # ... low to high

                if len(tn_i) > 0: 
                    
                    # ... low to high

                    # 1. positive examples: the smaller the better
                    support_pos = [i for i in rows if i in tn_i][:k_upper] # if k_upper_tp > 0 else [i for i in rows if i in tn_i][:k_upper]

                    # 2. tp examples are demoted to neutral

                    # 3. negative examples stay negative: the higher the better 
                    support_neg = [i for i in rows[::-1] if i in neg_i][:k_lower]
                    
                    n_tn_dominant += 1
                else: 
                    # tp_i == tn_i == 0
                    # then there's no way to tell just pick the smallest
                    support_pos = [i for i in rows if i in tn_i][:1]
                    support_neg = [i for i in rows[::-1] if i in neg_i][:k_lower]

                M2[support_pos, j] = 1
                M2[support_neg, j] = -1
        ### ... end foreach item

    if verbose: 
        msg = ''
        r = n_conflict_evidence/(T.shape[1]+0.0)
        msg += "(estimate_polarity) tContrained: True | Found n_conflict_evidence: {}, n_no_positive: {} | N={}, ratio_conflict_evidence: {}\n".format(n_conflict_evidence, n_no_positive, T.shape[1], r)
        msg += "... n_tp_dominant: {}, n_tn_dominant: {}".format(n_tp_dominant, n_tn_dominant)
        msg += "... n(pos): {}, n(neg): {}, n(neutral): {}".format(np.sum(M2>0), np.sum(M2<0), np.sum(M2==0))
        print(msg)

    return M2



 if policy == 'polarity': # need to separate R from T 
            assert 'n_train' in kargs, "Need n_train as a split point"
            n_train = kargs.get('n_train', -1)  # split point

            mask = maskEntriesItemAxis(X, L=L, p_th=thresholds, ratio_users=ratio_users, pos_label=1, policy='polarity')
        else: 

        # positive examples
        rows_aligned = np.where(Lh[:, j] == L[j])[0]  # Lh[:,j]=L[j] => boolean mask, np.where(...) gives indices where condition holds

        # negative examples
        rows_misaligned = np.where(Lh[:, j] != L[j])[0]

rows_sorted = np.argsort(-R[:, j])  # np.argsort(R[:, j])[:-k-1: -1] # choose indices of k highest probs

def mask_along_item_axis_given_labels(R, L, p_th, ratio_users=0.5, pos_label=1, verbose=True): 
    """

    Memo
    ---- 
    1. domain: diabetes

       ... n_candidates_per_item on average: 13.980456026058633 | min: 3, max: 15

    2. For test split, L is an estimated label vector (e.g. via applying majority vote)

    """
    # introducing colored particles
    # mask = np.ones(R.shape, dtype=bool) # all True 2D array (True: retain, False: zero out or set to a fill value)
    mask = np.zeros(R.shape, dtype=float)

    n_users = R.shape[0]
    # half_users = int(R.shape[0]/2)

    # k_ref = math.ceil(ratio_users * n_users) # select_k(n_users, ratio_users, r=1.0) # min(topk, n_users/2)
    assert R.shape[0] > 2
    ratio_users_upper, ratio_users_lower = ratio_users/2.0, ratio_users/2.0
    k_upper = k_max = max(1, math.ceil(ratio_users_upper * n_users)) #  # don't go over 50%
    k_lower = max(1, math.ceil(ratio_users_lower * n_users))
    k_min = 1 # max(1, math.ceil(n_users/10)) # each datum should at least have this many BP predictions

    n_under_repr = n_over_repr = 0
    n_candidates_per_item = []  # debug
    n_supports_per_item = []
    n_support = k_max

    # ratio_user_per_item = confidence_pointwise_ensemble_prediction(R, L, p_th, mode='item', suppress_majority=False)
    # print('(mask_along_item_axis_given_labels) ratio_user_per_item:\n... {}\n'.format(np.random.choice(ratio_user_per_item, 20)))
    
    # given proba thresholds, estimate label matrix
    Lh = estimateLabelMatrix(R, p_th=p_th, pos_label=pos_label)  # foreach class, use the topk probs in the horizontal direction as a proxy for positive labels

    ratios_label_aligned = []
    n_supports = []
    for j in range(R.shape[1]):  # foreach column/item

        rows_aligned = np.where(Lh[:, j] == L[j])[0]  # Lh[:,j]=L[j] => boolean mask, np.where(...) gives indices where condition holds
        k = len(rows_aligned)  # select_k(n_users, ru, r=1.0)      # [note] instead of using ratio_users/0.5, use ru, which differs by items
        ru = k/(n_users+0.0)
        ratios_label_aligned.append(ru)
        
        # if positive: choose indices of k highest probs
        # if negative: choose indices of k lowest probs  
        rows_sorted = np.argsort(R[:, j])[:-k-1: -1] if L[j] == pos_label else np.argsort(R[:, j])[:k] 
        # ... all candidates

        support = [r for r in rows_sorted if r in rows_aligned] # [:k]  # choice: keep all or just 'k'? 
        ns = len(support)
        # ... all candidates that are label-algined

        # clip
        if ns > k_max:  # too many (and all consistent)
            support = support[:k_max]  # indices aligned with (estimated) labels
            n_over_repr += 1

        # pad 
        if ns < k_min: # too few that are consistent
            # keep all in rows_good + pad extra BP predictions that are not correct but perhaps close enough

            # need at least k_min BP predictions
            residual = k_min - ns
            if residual > 0: 
                rows_extra = [r for r in rows_sorted if r not in rows_aligned][:residual]
                support = np.hstack( (rows_extra, support) ) # pad the top choices
            n_under_repr += 1

            # [condition] it's possible that the number of the label-aligned candidates are too few to even reach k_min

        # >>> see condition 1 
        # assert len(rows_candidates) >= k_min, "rows_aligned: %d, rows_candidates: %d, k_min: %d" % (len(rows_aligned), len(rows_candidates), k_min)
        if not (len(support) >= k_min): 
            msg = "Warning: Not enough row candidates | label-aligned: %d, support: %d, k_min: %d" % (len(rows_aligned), len(support), k_min)
            div(msg, symbol='%', border=2)

        # [test]
        n_support = len(support)
        assert n_support <= k_max and n_support >= k_min

        n_candidates_per_item.append(ns)  # raw support before clipping or padding
        n_supports_per_item.append(n_support)  # final number of support user/classifier for each item, from which we'll use to predict the labels
        mask[support, j] = 1.0   # polarity positive: 1.0
    
    if verbose: 
        msg = ''
        msg += '(mask_along_item_axis_given_labels) Found {} under-represented items and {} over-represented items (wrt k: [{}, {}]), ratio (under-represented): {}\n'.format(
                    n_under_repr, n_over_repr, k_min, k_max, n_under_repr/(R.shape[1]+0.0) )
        msg += '... n(raw candidates) | min={min}, max={max}, median={m}\n'.format( min=min(n_candidates_per_item), max=max(n_candidates_per_item), m=np.median(n_candidates_per_item))
        msg += '... n(support)        | min={min}, max={max}, median={m}, examples:\n... {ex}\n'.format( min=min(n_supports_per_item), max=max(n_supports_per_item), m=np.median(n_supports_per_item), ex=n_supports_per_item[:20])
        msg += '... ratio(Lh ~ L)     | min: {}, max: {}, median: {}, examples:\n... {}\n'.format(min(ratios_label_aligned), max(ratios_label_aligned), np.median(ratios_label_aligned), ratios_label_aligned[:20] )
        print(msg)

    return mask


# separate alignemnt 

        print('(wmf_ensemble_iter2) Quality of seed on (Rh) | th(Rh): {} => policy_calibration: {}    ... cycle: {}'.format( pref_threshold, policy_calibration, (outer_fold, fold) ))
        p_tp_preferred, p_fp_preferred, p_tn_preferred, p_fn_preferred, p_agreed, p_correct_agreed = \
            uc.ratio_of_alignment2(Rh, Mc_R, Lh_R, binarize=False, verbose=True)  

        print('(wmf_ensemble_iter2) Quality of seed on (Th) | th(Th): {} => policy_calibration: {}    ... cycle: {}'.format( pref_threshold_test, policy_calibration, (outer_fold, fold) ))
        p_tp_preferred, p_fp_preferred, p_tn_preferred, p_fn_preferred, p_agreed, p_correct_agreed = \
            uc.ratio_of_alignment2(Th, Mc_T, Lh_T, binarize=False, verbose=True)  

#  what is a better objective for preference calibration?  

        print('(wmf_ensemble_iter2) Quality of seed on (Rh) | th(Rh): {} => policy_calibration: {}    ... cycle: {}'.format( pref_threshold, policy_calibration, (outer_fold, fold) ))
        p_tp_preferred, p_fp_preferred, p_tn_preferred, p_fn_preferred, p_agreed, p_correct_agreed = \
            uc.ratio_of_alignment2(Rh, Mc_R, Lh_R, binarize=False, verbose=True)  

        print('(wmf_ensemble_iter2) Quality of seed on (Th) | th(Th): {} => policy_calibration: {}    ... cycle: {}'.format( pref_threshold_test, policy_calibration, (outer_fold, fold) ))
        p_tp_preferred, p_fp_preferred, p_tn_preferred, p_fn_preferred, p_agreed, p_correct_agreed = \
            uc.ratio_of_alignment2(Th, Mc_T, Lh_T, binarize=False, verbose=True)  

predict_pos = (Lh == 1)  # Given BP's prediction Lh, select entries ~ target label
# predict_neg = (Lh == 0)

cells_tp = (Mc == 1) & predict_pos
# cells_tn = (Mc == 1) & predict_neg
# cells_fp = (Mc == 0) & predict_pos
# cells_fn = (Mc == 0) & predict_neg

pref = (Xpf == 1)
n_tp_hit = np.sum( cells_preferred & cells_tp ) # correctly aligned TPs
# n_tn_hit = n_tn_pref = np.sum( cells_preferred & cells_tn )

not_pref = (Xpf == 0)
n_tp_missed = np.sum(not_pref & cells_tp)
# n_tn_missed = np.sum(not_pref & cells_tn)


p_tp_preferred, p_fp_preferred, p_tn_preferred, p_fn_preferred, p_agreed, p_correct_agreed = \
                uc.ratio_of_alignment2(Xp, correctness, Lh, binarize=False) 

    cells_preferred = (Xpf == 1)
    n_pref = np.sum(cells_preferred)

    cells_tp = (Cm == 1) & predict_pos
    cells_tn = (Cm == 1) & predict_neg
    cells_fp = (Cm == 0) & predict_pos

    n_agreed_tp = np.sum( (Xpf == Cm) & cells_tp )  # agreed & tp
    n_agreed_fp = np.sum( (Xpf == Cm) & cells_fp )  # agreed & tn

p_tp_preferred, p_tn_preferred, p_agreed, p_correct_agreed = uc.ratio_of_alignment2(Xp, correctness, Lh, target_label=1, binarize=False)  # target, overall, correct

            # assert abs(r - c_ratio) < 1e-3
            print('(reconstruct_by_preference) Quality of the seed | ratio_alignment(X<-R): ({}, {}) | test data? {}  ... Cycle: {}'.format(r, rc, is_test_set, index))
            print('... pref threshold: {} | P<R>(TP|pref): {}, P<R>(TN|pref):{} | P<R>(agreed): {}, P<R>(correct|agreed)'.format(pref_threshold, 
                p_tp_preferred, p_tn_preferred, p_agreed, p_correct_agreed))


# [test]
                scores[file_type][aggr] = common.fmax_score(y_label, y_pred, beta = 1.0, pos_label = 1)
                # 'ideal' baseline: if preference matrix is perfect, what happens? 
                if file_type.startswith('post') and isPreferenceScore: 
                    Mc, Lh_T = uc.correctness_matrix(T, L_test, p_threshold)  # Lh(T, p_threshold)
                    pv_perfect = uc.predict_by_preference(T, Mc, W=None, canonicalize=True, binarize=not isBinaryPrefMatrix, name='Th', aggregate_func='mean', fallback_on_low_weight=False, min_pref=0.1) 
                    scores[file_type]['perfect_{}'.format(aggr)] = common.fmax_score(y_label, pv_perfect, beta = 1.0, pos_label = 1)

                    Cwt = Cx[:,n_train:]
                    pv_perfect_weighted = uc.predict_by_preference(T, Mc, W=Cwt, canonicalize=True, binarize=not isBinaryPrefMatrix, name='Th', aggregate_func='mean', fallback_on_low_weight=False, min_pref=0.1) 
                    scores[file_type]['perfect_weighed_{}'.format(aggr)] = common.fmax_score(y_label, pv_perfect_weighted, beta = 1.0, pos_label = 1)
            ### ... end foreach aggregation method

        # [test]
        div('(wmf_ensemble_iter2) Performance | Cycle: {} | file type: {}, ID: {}'.format( (outer_fold, fold), file_type, MFEnsemble.get_dset_id(method=method, params=params)), symbol='#')
        msg = ''
        for file_type in file_types: 
            msg += '--- ({}) ---\n'.format(file_type)
            for i, (aggr, score) in enumerate(scores[file_type].items()): 
                msg += '... [{}] method: {} => score: {}\n'.format(i, aggr, scores[file_type][aggr] )


        isBinaryPrefMatrix = False
        if isPreferenceScore: 
            n_uniq, n_uniq_test = len(np.unique(Rh)), len(np.unique(Th))
            if tPreferenceCalibration: # then Xh must be a binary matrix
                assert n_uniq == 2 and n_uniq_test == 2, "Th is not binary or is degenerated | n_uniq: {}, n_uniq(T): {} | values(T): {}".format(n_uniq, n_uniq_test, np.unique(Th))
            else: 
                # Th is a continuous rating matrix
                assert n_uniq > 2 and n_uniq_test > 2, "Unique values should be >> 1 | n_uniq(R): {}, u_uniq(T): {}".format(n_uniq, n_uniq_test)
            if n_uniq_test == 2: isBinaryPrefMatrix = True
            print('(wmf_ensemble_iter2) Prior to making predictions | Th is a {}  ... (verify)'.format(
                'binary matrix' if isBinaryPrefMatrix else 'rating matrix'))

##############

        isBinaryPrefMatrix = False
        if isPreferenceScore: # True if policy_opt: preference
            n_uniq = len(np.unique(Th))
            assert n_uniq > 1, "Min number of unique values should be > 1 even when Th repr a preference matrix! n_uniq: {}".format(n_uniq)
            isBinaryPrefMatrix = True if n_uniq == 2 else False
            print('(wmf_ensemble_iter) Prior to making predictions | Th is a {}  ... (verify)'.format(
                'binary matrix' if isBinaryPrefMatrix else 'continous matrix'))

print('... approximating ratings (X) | Cycle ({fo}, {fi})'.format(fo=outer_fold, fi=fold))
    P, Q, *Xh_errs = ua.implicit_als(Cx, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            label_confidence=Cx_bar, ratings=X, labels=L,
                            policy='rating', message=piggyback_msg, ret_rmse=True)
    Xh_err, Xh_err_weighted = Xh_errs

    print('... estimating preferences (X) | Cycle ({fo}, {fi})'.format(fo=outer_fold, fi=fold))
    Pp, Qp, *Xp_errs = ua.implicit_als(Cx, features=params['n_factors'], 

                            iterations=params['n_iter'],
                            lambda_val=System.lambda_val,  # 0.8 by default

                            label_confidence=Cx_bar, ratings=X, labels=L,
                            policy='preference', message=piggyback_msg, ret_rmse=True)


    # classifier/user weights (usually just use confidence matrix)
    Cw = Cwt = None
    if tWeightedPrediction: 
        Cw, Cwt = Cx[:,n_train:], Cx[:,n_train:]
        print('(wmf_ensemble_iter2) using weighetd output via confidence matrix | dim(Cwt): {}'.format(Cwt.shape))
    else: 
        print('(wmf_ensemble_iter2) using only preference matrix (non-weigthed) for making final predictions ...')



        n_uniq = len(np.unique(Th))
        assert n_uniq > 1, "Min number of unique values should be > 1 even when Th repr a preference matrix! n_uniq: {}".format(n_uniq)
        isPreferenceScore = True if n_uniq == 2 else False
        print('(wmf_ensemble_iter2) Prior to making predictions | Th is a {}  ... (verify)'.format(
            'pref matrix' if isPreferenceScore else 'probabilities'))

# reweighting per user

        for i in range(n_users): 
            user_name = U[i] if len(U) > 0 else i

            # indices of majority classe & predictions were correct
            idx_max_class_correct = np.where( (L==max_class) & (C[i] > 0) )[0]
            idx_min_class_correct = np.where( (L==min_class) & (C[i] > 0) )[0]

            weights_max_class = np.sum(C[i, idx_max_class_correct])
            weights_min_class = np.sum(C[i, idx_min_class_correct])

            if weights_min_class > 0 and weights_max_class > 0: 

                # each classifier/user has a different weight distribution
                multiple_eff_i = weights_max_class/weights_min_class
                print('... user: {} | w(pos): {}, w(neg): {}, multiple: {} | N={} (> max(w))?'.format(user_name, weights_min_class, weights_max_class, multiple_eff_i, n_items))

                # >>> there are cases where the mask function masks all the majority class exmaples, leading to zero weights (werid but true)
                if multiple_eff_i <= 1: 
                    msg = "Warning: weights(class={max_class}) < weights(class={min_class}) ?? {w_min} > {w_max}".format(max_class=ret['max_class'], min_class=ret['min_class'],
                        w_min=weights_min_class, w_max=weights_max_class)
                    print(msg)
                    # assert multiple_eff_i > 1, "weights(class={max_class}) < weights(class={min_class}) ?? {w_min} > {w_max}".format(max_class=ret['max_class'], min_class=ret['min_class'],
                    #     w_min=weights_min_class, w_max=weights_max_class)
                    multiple_eff_i = multiple
            
                # [test]
                multiple_eff.append(multiple_eff_i)
                weights_min.append(weights_min_class)
                weights_max.append(weights_max_class)

                # update 
                prior_weights = C[i, idx_min_class_correct]
                ##########################
                C[i, idx_min_class_correct] = C[i, idx_min_class_correct] * multiple_eff_i  # magnify confidence scores of minority class
                ##########################
                post_weights = C[i, idx_min_class_correct]

                # [test]
                if i in test_cases: 
                    tidx = np.random.choice(range(len(prior_weights)), 5)
                    print('... sample weights (prior):\n... {}\n'.format(prior_weights[tidx]))
                    print('... sample weights (post): \n... {}\n'.format(post_weights[tidx]))
            else: 
                if weights_min_class == 0:
                    # ... not positive classes were selected => this classifier is probably not useful 
                    # reduce all to min weights 
                    

                    C[i, idx_max_class_correct] = min_weight

                    print('... suppress negative example weights for user {} to min weight {}'.format(user_name, min_weight))


    pos_label = 1
    # Eu = identify_effective_users(Cr, L_train=L_train, fill=null_marker)
    Eu = []
    p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=params['policy_threshold']) 
    Lh = lh = uc.estimateLabels(T, L=[], Eu=Eu, p_th=p_threshold, pos_label=pos_label, ratio_small_class=params['ratio_small_class']) 

    X = np.hstack((R, T))
    L = np.hstack((L_train, Lh))


n_zeros, n_ones = np.sum(Rh==0), np.sum(Rh==1)
        N = Rh.shape[0] * Rh.shape[1] + 0.0
        print('(wmf_ensemble_iter) n(zeros): {} (r={}), n(ones): {} (r={})'.format(n_zeros, n_zeros/N, n_ones, n_ones/N))

# wmf_ensemble_preferred_ratings2()


        CMt, Lt = uc.correctness_matrix(T, L_test, p_threshold)  # Lh(T, p_threshold)

        # vs using estimted labels 
        # CMt_est, Lt_est = uc.correctness_matrix(T, Lh, p_threshold)         

        # find ratio of alignment wrt true labels (not the estimted labels lh)
        rt, rtc = uc.ratio_of_alignment(Th, CMt, binarize=False)  # overall vs correct only

        # wrt Lt (not Lt_est) because we want to find how preference is aligned wrt to true labels
        rtc_pos = uc.ratio_of_alignment2(Th, CMt, Lt, target_label=1, binarize=False)
        print('(wmf_ensemble_preferred_ratings2) Quality of the seed (via L) | th(R+T): {}| r(T): {}, rc(T): {} | rc(T)<positive>: {}'.format(pref_threshold, 
            rt, rtc, rtc_pos))

# wmf_ensemble_iter2() T only 

        CMt, Lh_T = uc.correctness_matrix(T, L_test, p_threshold)  # Lh(T, p_threshold)
        rt, rtc = uc.ratio_of_alignment(Th, CMt, binarize=False)  # overall vs correct only
        rtc_pos = uc.ratio_of_alignment2(Th, CMt, Lh_T, target_label=1, binarize=False)
        print('(wmf_ensemble_iter2) Cycle: {} | Quality of the seed on test split (via L) | th(R+T): {}| r(T): {}, rc(T): {} | rc(T)<positive>: {}'.format((outer_fold, fold),
            pref_threshold, rt, rtc, rtc_pos))


# wmf_ensemble_iter2() 

    if tPreferenceCalibration: 
        Lt = np.hstack(L_train, L_test)
        CM, Lh_X = uc.correctness_matrix(X, Lt, p_threshold) # this should be wrt true labels all the way

        Xh, pref_threshold, c_ratio = uc.preference_calibration(Xh, Cm=CM, step=0.01, message='train+test split X=(R+T)')  

        r, rc = uc.ratio_of_alignment(Xh, CM, binarize=False)  # overall vs correct only
        assert abs(r - c_ratio) < 1e-3
        print('(wmf_ensemble_iter2) Quality of the seed on X:R+T | ratio_alignment(X): ({}, {})'.format(r, rc))
        
        rc_pos = uc.ratio_of_alignment2(Xh, CM, Lh_X, target_label=1, binarize=False)
        print('... pref threshold: {} | r(X): {}, rc(X): {} | rc(X)<positive>: {}'.format(pref_threshold, r, rc, rc_pos))



        # compute confidence matrix for R
    CX = uc.evalConfidenceMatrix(X, L=L, U=U,  
            ratio_users=params['ratio_users'], 
            ratio_small_class=params['ratio_small_class'], factor_small_class=params.get('factor_small_class', 1.0), 

            policy=params['policy'], # <<< determines the subroutine for computing Cui
            policy_opt=params['policy_opt'],  # <<< determine the optimization types (rating, tradeoff, preferenc, etc.) 
            policy_threshold=params['policy_threshold'],

                supervised=params['supervised'], # applicable to all policies (determines how p_threshold is computed if applicable)
                    conf_measure=params['conf_measure'], alpha=params['alpha'], 
                        # masked=params['masked'], mask_all_test=params['mask_all_test'],   # not used now
                        fill=null_marker, fold=fold) # project_path=System.projectPath 
    Cx, Cx_bar, p_threshold, *CX_res = CX


    Xh = reconstruct(Cx, X, P, Q, policy_opt=params['policy_opt'], policy_replace=params['policy_replace'], 
                replace_subset=params['replace_subset'], params=params, null_marker=null_marker, binarize=False, name='R+T')

    if tPreferenceCalibration: 
        CM, Lh_X = uc.correctness_matrix(X, L, p_threshold)
        Xh, pref_threshold, c_ratio = uc.preference_calibration(Xh, Cm=CM, step=0.01, message='train+test split X=(R+T)')  

        r, rc = uc.ratio_of_alignment(Xh, CM, binarize=False)  # overall vs correct only
        assert abs(r - c_ratio) < 1e-3
        print('(wmf_ensemble_iter2) Quality of the seed | ratio_alignment(X): ({}, {})'.format(r, rc))
        
        rc_pos = uc.ratio_of_alignment2(Xh, CM, Lh_X, target_label=1, binarize=False)
        print('... pref threshold: {} | r(X): {}, rc(X): {} | rc(X)<positive>: {}'.format(pref_threshold, r, rc, rc_pos))
    # ... Xh is binarized

    ### ALS evaluation (RMS)
    delta_X = LA.norm(Xh-X, 'fro')
    Rh, Th = Xh[:,:n_train], Xh[:,n_train:]
    assert Rh.shape == R.shape, "dim(R): {}, dim(Rh): {}".format(R.shape, Rh.shape)
    assert Th.shape == T.shape, "dim(T): {}, dim(Th): {}".format(T.shape, Th.shape)

    # [test]
    if tPreferenceCalibration: 
        # use L_test to get correctness matrix
        CMt, Lh_T = uc.correctness_matrix(T, L_test, p_threshold)  # Lh(T, p_threshold)
        rt, rtc = uc.ratio_of_alignment(Th, CMt, binarize=False)  # overall vs correct only
        rtc_pos = uc.ratio_of_alignment2(Th, CMt, Lh_T, target_label=1, binarize=False)
        print('(wmf_ensemble_iter2) Quality of the seed (via L) | th(R+T): {}| r(T): {}, rc(T): {} | rc(T)<positive>: {}'.format(pref_threshold, 
            rt, rtc, rtc_pos))


    if tPreferenceCalibration: 
        # binarize Th ... (b)
        # p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=params['policy_threshold']) 
        div('(wmf_ensemble_iter) 1. Use estimated label (by majority vote) to binarize T')
        Thb, pth_test, rc_test = uc.estimate_pref_threshold(Th, T, L=[], p_threshold=p_threshold, message='Tb using estimated labels')
        # ... now got the preference threshold

        # use L_test to get correctness matrix
        CMt, Lh_T = uc.correctness_matrix(T, L_test, p_threshold)  # Lh(T, p_threshold)

        rt, rtc = uc.ratio_of_alignment(Thb, CMt, binarize=False)  # overall vs correct only
        rtc_pos = uc.ratio_of_alignment2(Thb, CMt, Lh_T, target_label=1, binarize=False)
        print("... (1) Quality of the seed (via lh) | th(R): {} vs th(T): {}| r(T): {}, rc(T): {} | rc(T)<positive>: {}: {}".format(pref_threshold,
           pth_test, rt, rtc, rtc_pos))

        n_zeros, n_ones = np.sum(Thb==0), np.sum(Thb==1)
        N = Thb.shape[0] * Thb.shape[1] + 0.0
        print('... (1) Number of 0s and 1s | n<T,lh>(zeros): {} (r={}), n<T,lh>(ones): {} (r={})'.format(n_zeros, 
            n_zeros/N, n_ones, n_ones/N))
        ########################################
        # ... how does it compare to using true labels?
        div('(wmf_ensemble_iter) 2. How does it fare with using true labels?') 
     
        Thb2, pth_test, rc_test = uc.estimate_pref_threshold(Th, T, L=L_test, p_threshold=p_threshold, message='Tb using true test labels')
        rt, rtc = uc.ratio_of_alignment(Thb2, CMt, binarize=False)  # overall vs correct only
        rtc_pos = uc.ratio_of_alignment2(Thb2, CMt, Lh_T, target_label=1, binarize=False)
        print('... (2) Quality of the seed (via L) | th(R): {} vs th(T): {}| r(T): {}, rc(T): {} | rc(T)<positive>: {}'.format(pref_threshold, 
            pth_test, rt, rtc, rtc_pos))

        n_zeros, n_ones = np.sum(Thb2==0), np.sum(Thb2==1)
        N = Thb2.shape[0] * Thb2.shape[1] + 0.0
        print('... (2) Number of 0s and 1s | n<T,lh>(zeros): {} (r={}), n<T,lh>(ones): {} (r={})'.format(n_zeros, 
            n_zeros/N, n_ones, n_ones/N))

        div('(wmf_ensemble_iter) 3. How about just using the threshold from (R)?')
        pth_test = pref_threshold
        Thb3 = uc.binarize_pref(Th, p_th=pref_threshold)
        rt, rtc = uc.ratio_of_alignment(Thb3, CMt, binarize=False)  # overall vs correct only
        rtc_pos = uc.ratio_of_alignment2(Thb3, CMt, Lh_T, target_label=1, binarize=False)
        print("... (3) Quality of the seed (via R) | th(R): {} == th(T): {}| r(T): {}, rc(T): {} | rc(T)<positive>: {}".format(pref_threshold, 
            pth_test, rt, rtc, rtc_pos))

        n_zeros, n_ones = np.sum(Thb3==0), np.sum(Thb3==1)
        N = Thb3.shape[0] * Thb3.shape[1] + 0.0
        print('... (3) Number of 0s and 1s | n<T,lh>(zeros): {} (r={}), n<T,lh>(ones): {} (r={}) | values: {}'.format(n_zeros, 
            n_zeros/N, n_ones, n_ones/N, np.unique(Thb3)[:10]))

        Th = Thb
        assert len(np.unique(Th)) == 2, "Th was not binarized | tPreferenceCalibration? {}, binarize_pref? {} | unique values (n={}): {}".format(tPreferenceCalibration, 
            params['binarize_pref'], len(np.unique(Th)), np.unique(Th)[:10])

print('... Model | n_factors: {nf}, alpha: {a} | predict pref? {pref}, binarize? {bin}, supervised? {s}'.format(nf=params['n_factors'], a=params['alpha'], pref=isPreferenceScore, bin=params['binarize_pref'], s=params['supervised']))

    if tPreferenceCalibration: 
        # binarize Th ... (b)
        # p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=pos_label, policy=params['policy_threshold']) 
        lh = uc.estimateLabels(T, L=[], p_th=p_threshold, pos_label=1, ratio_small_class=params['ratio_small_class'])
        CMt, Lh_T = uc.correctness_matrix(T, lh, p_threshold)  # CM entries: 1, if correct predictions (TP, TN); 0 o.w. 
        Thb, pth_test, rc_test = uc.preference_calibration(Th, Cm=CMt, step=0.01, message='test split (T) via (lh)')

        n_zeros, n_ones = np.sum(Thb==0), np.sum(Thb==1)
        N = Thb.shape[0] * Thb.shape[1] + 0.0
        print('(wmf_ensemble_iter) n<T,lh>(zeros): {} (r={}), n<T,lh>(ones): {} (r={})'.format(n_zeros, 
            n_zeros/N, n_ones, n_ones/N))

        rt, rtc = uc.ratio_of_alignment(Thb, CMt, binarize=False)  # overall vs correct only
        print('... Quality of the seed | ratio_alignment(T): ({}, {}) | pref_th(T, lh): {}'.format(rt, rtc, pth_test))

        rtc_pos = uc.ratio_of_alignment2(Thb, CMt, Lh_T, target_label=1, binarize=False)
        print("... pref threshold(R): {}, r(T,lh): {}, rc(T,lh): {} | ratio_alignment(T| lh, positve): {}".format(pref_threshold, rt, rtc, rtc_pos))
        ########################################
        # ... how does it compare to using true labels?
        div('(wmf_ensemble_iter) How does it fare with using true labels?') 
     
        CMt, Lh_T = uc.correctness_matrix(T, L_test, p_threshold)  # Lh(T, p_threshold)
        Thb2, pth_test, rc_test = uc.preference_calibration(Th, Cm=CMt, step=0.01, message='test split (T) via (L_test)')
        rt, rtc = uc.ratio_of_alignment(Thb2, CMt, binarize=False)  # overall vs correct only
        print('... Quality of the seed | ratio_alignment(T, L_test): ({}, {}) | pref_th(T, L_test): {}'.format(rt, rtc, pth_test))
        
        rtc_pos = uc.ratio_of_alignment2(Thb2, CMt, Lh_T, target_label=1, binarize=False)
        print('... pref threshold(R): {}, r(T): {}, rc(T): {} | ratio_alignment(T | positve): {}'.format(pref_threshold, rt, rtc, rtc_pos))

        Th = Thb
        assert len(np.unique(Th)) == 2, "Th was not binarized | tPreferenceCalibration? {}, binarize_pref? {} | unique values (n={}): {}".format(tPreferenceCalibration, 
            params['binarize_pref'], len(np.unique(Th)), np.unique(Th)[:10])


policy_threshold = kargs.get('policy_threshold', 'prior')
p_threshold = estimateProbThresholds(X_train, L=L_train, pos_label=pos_label, policy=policy_threshold) 
Lh = lh = estimateLabels(X, L=[], Eu=Eu, p_th=p_threshold, pos_label=pos_label, ratio_small_class=kargs.get('ratio_small_class', 0.01)) 
    
def maskEntries(): 
    def mask_by_balancing_estimates(R, L, Lh, multiple=2, mask=None): # closure: min_label, max_label  
        # select majority class examples with sample size 'comparable' to that of the minority class 

        if mask is None: mask = np.ones(R.shape, dtype=bool)
        n_min_captured = 0
        for i in range(R.shape[0]):  # foreach row/user/classifier 
            idx_min_class = np.where( (Lh[i] == L) & (Lh[i] == min_label) )[0]
            idx_max_class = np.where( (Lh[i] == L) & (Lh[i] == max_label) )[0]

            # now select only a subset (n_min * multiple) of the matched majority-class examples
            ######################################
            n_min, n_max = len(idx_min_class), len(idx_max_class)  # let n: size(minority), then we will pick n * multiple examples from the majority class
            n_min_captured += n_min
            n_min_eff = int(n_min * multiple) # let n: size(minority), then we will pick 2n examples from the majority class
            if n_max > n_min_eff:  
                idx_max_class = np.random.choice(idx_max_class, n_min_eff, replace=False)
            else: 
                idx_max_class = np.random.choice(idx_max_class, n_min_eff, replace=True)
            ############################################################################

            idx_active = np.hstack( (idx_min_class, idx_max_class) ) # pad the top choices
            mask[i, idx_active] = False  # False: retained, True: to be zero out (e.g. X[mask] = fill, where 'fill' is usu 0)

        # [debug]
        idx_active0 = np.where(Lh == L)[0] # note that accuracy can be low (e.g. accuracy: 41.32% for pf2)
        print("... (verify) captured {n} minority class examples (avg: {ne}/classifier <? n_min_class: {n_min}) | n(Lh=L): {nr} (accuracy: {a}, total: {nL}) | policy: {policy} ... ".format(n=n_min_captured,
            ne=n_min_captured/(R.shape[0]+0.0), n_min=n_min_class, nr=len(idx_active0), a=len(idx_active0)/(nL+0.0), nL=nL, policy=policy))
        return mask


def toConfidenceMatrix():

    def balance_class_negative_sampling(C, L, min_class=None, max_class=None, multiple=10, verify_=True):
        ret = classPrior(L, labels=[0, 1], ratio_ref=0.1, verbose=False)
        
        if ret['n_min_class'] == 0: 
            # label dtpye 
            lt = np.random.choice(L, 1)[0]
            print('(balance_class_by_subsampling) Warning: No minority class found in this batch => No-op! | n_max: {n_max}, n_min: {n_min} | dtype(L): {dtype} (expected int), value: {val}'.format(
                n_max=ret['n_max_class'], n_min=ret['n_min_class'], dtype=type(lt), val=lt ))

        min_class, max_class = ret['min_class'], ret['max_class']
        n_users = C.shape[0]
        idx_max_class = np.where(L==max_class)[0]  # column-wise positional indices of majority class ... [0] because np.where returns a tuple
        idx_min_class = np.where(L==min_class)[0]

        # (verify)? for those that were not masked should be consistent with the labels
        # p_threshold

        n_min_class = ret['n_min_class']
        upperbound = multiple * n_min
        n_max_class = upperbound if upperbound < ret['n_max_class'] else ret['n_max_class']
        print('(negative_sampling) majority: (c={C}, n={N}->{Np}) | minority: (c={c}, n={n})'.format(C=max_class, N=len(idx_max_class), Np=n_max_class, c=min_class, n=len(idx_min_class)))
        
        # subsampling
        idx_max_class = np.random.choice(idx_max_class, n_max_class) # -- negative sampling --
        idx_max_class.sort()

        # for those not chosen, mask them 
        idx_to_mask = np.array(list(set(range(len(L)))-set(idx_min_class)-set(idx_max_class)))

        # Cp = np.zeros(C.shape)  # assuming the fill is zeros
        for i in range(n_users): 
            # want C[i, idx_max_class], C[i, idx_min_class]

            print('... user/classifier #{i} > masking {n} examples')

            C[i, idx_to_mask] = fill 

        return