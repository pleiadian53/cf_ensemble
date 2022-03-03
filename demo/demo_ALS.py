import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) # include parent directory to the module search path
import utils_sys
from utils_sys import div

from evaluate import Metrics, PerformanceMetrics
from sklearn.metrics import brier_score_loss

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import numpy as np
import scipy as sp
# import scipy.sparse
# import scipy.sparse.linalg

import random

"""

Reference
---------
1. ALS
    a. Daniel Nee   
        http://danielnee.com/2016/09/collaborative-filtering-using-alternating-least-squares/

    b. Victor 
        https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

    c. Ethan Rosenthal
        https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/

    d. Cython implementation of ALS from Ben Frederickson

       http://www.benfrederickson.com/fast-implicit-matrix-factorization/
       https://github.com/benfred/implicit

    e. Dimitriy Selivanov 
    
       http://dsnotes.com/post/2017-05-28-matrix-factorization-for-recommender-systems/


2. Nicolas Hug: SVD, Matrix Factorization 
   a. SGD
       http://nicolas-hug.com/blog/matrix_facto_3
       http://nicolas-hug.com/blog/matrix_facto_4

3. Matrix Factorization for Recommender Systems 

   http://dsnotes.com/post/2017-05-28-matrix-factorization-for-recommender-systems/

"""

### configurations
Domain = 'recommender'
ProjectPath = utils_sys.getProjectPath(domain=Domain, verify_=False)  # default
try: 
    ProjectPath = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    Domain = os.path.basename(ProjectPath)
    utils_cf.Domain = Domain
    utils_cf.ProjectPath = ProjectPath
except: 
    pass 
assert os.path.exists(ProjectPath)


def normaliseRow(x):
    return x / sum(x)

def initialiseMatrix(n, f):
    A = abs(np.random.randn(n, f))
    return sp.sparse.csr_matrix(np.apply_along_axis(normaliseRow, 1, A))

def ratingsPred(X, Y):
    return X.dot(Y.T)

def calculateMSE(X, Y, ratingsMatrix):
    ratingsPrediction = ratingsPred(X, Y)
    ratingsDiff = ratingsPrediction - ratingsMatrix
    return (ratingsDiff.multiply(ratingsDiff)).mean()

def calculateWeightedMSE(X, Y, P, ratingsMatrix, alpha):
    ratingsPrediction = ratingsPred(X, Y)
    C =  ratingsMatrix.multiply(1 + alpha)
    weightedDiff = C.multiply(P - ratingsPrediction)
    return weightedDiff.multiply(weightedDiff).mean()

# ALS.b: Victor
def load_data(inputdir=None): 
    """

    Reference
    ---------

    """
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    # from sklearn.preprocessing import MinMaxScaler

    #-------------------------
    # LOAD AND PREP THE DATA
    #-------------------------
    input_file = 'usersha1-artmbid-artname-plays.tsv'
    if inputdir is None: inputdir = ProjectPath
    input_path = os.path.join(inputdir, input_file) # 'data/usersha1-artmbid-artname-plays.tsv'

    # format: user-mboxsha1, musicbrainz-artist-id, artist-name, plays
    raw_data = pd.read_table(input_path)
    raw_data = raw_data.drop(raw_data.columns[1], axis=1)
    raw_data.columns = ['user', 'artist', 'plays']
 
    # Drop rows with missing values
    data = raw_data.dropna()
  
    # Convert artists names into numerical IDs
    data['user_id'] = data['user'].astype("category").cat.codes
    data['artist_id'] = data['artist'].astype("category").cat.codes
 
    # Create a lookup frame so we can get the artist names back in 
    # readable form later.
    item_lookup = data[['artist_id', 'artist']].drop_duplicates()
    item_lookup['artist_id'] = item_lookup.artist_id.astype(str)
 
    data = data.drop(['user', 'artist'], axis=1)

    # data: 
    # 
    #    plays  user_id  artist_id
    # 0   1099        0      90933
    # 1    897        0     185367
    #      ... 
    print('(test) data:\n%s\n' % data.head(10))
 
    # Drop any rows that have 0 plays
    data = data.loc[data.plays != 0]
 
    # Create lists of all users, artists and plays
    users = list(np.sort(data.user_id.unique()))
    artists = list(np.sort(data.artist_id.unique()))
    plays = list(data.plays)
    print('(test) users:\n%s\n' % users[:10])
    print('(test) artists:\n%s\n' % artists[:10])
    print('(test) plays:\n%s\n' % plays[:10])
    uid, aid = 0, 19356
    cond = (data['user_id']==uid) & (data['artist_id']==aid)  # top: () is necessary
    print('(query) (%d, %d)=%s' % (uid, aid, data.loc[cond]['plays']))
 
    # Get the rows and columns for our new matrix
    rows = data.user_id.astype(int)
    cols = data.artist_id.astype(int)
 
    # Contruct a sparse matrix for our users and items containing number of plays
    # R
    data_sparse = sparse.csr_matrix((plays, (rows, cols)), shape=(len(users), len(artists)))

    # adjacency repr:  (user_id, artist_id) -> play (n_times)
    print('(test) data_sparse:\n%s\n' % data_sparse[:10][:10]) 

    return data_sparse, item_lookup

def toConfidenceMatrix(fold, **kargs):
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

    def mask_over(W, R, labels, p_th=0.5, pos_label=1, neg_label=0):  
        # mask input W over the result of mask(R, labels)
        nU, nI = R.shape[0], R.shape[1]
        assert W.shape == R.shape

        rows, cols = mask(R, labels, p_th=p_th, pos_label=pos_label, neg_label=neg_label)
        W_nonzero = W[rows, cols]

        S = sparse.csr_matrix((W_nonzero, (rows, cols)), shape=(nU, nI))    
        return S   
    def estimate_labels(T, p_th=0.5, pos_label=1):
        # probs = np.mean(T, axis=0)  

        L_est = np.zeros(T.shape[1])
        L_est[np.where(np.mean(T, axis=0) > p_th)] = pos_label

        return L_est

    # from sklearn.metrics import brier_score_loss
    import scipy.sparse as sparse
    from cf import to_rating_matrix2
    import utils_cf as uc

    missing_value = kargs.get('fill', 0) # [todo] marker for missing data
    is_augmented = kargs.get('is_augmented', True)
    p_th = kargs.get('p_threshold', 0.5)
    topk_users, topk_items = 0, 0  # default, use all by setting to 0
    
    R, T, L_train, L_test, U = to_rating_matrix2(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
    n_users, n_items = R.shape[0], R.shape[1]
    print('[toConfidenceMatrix] dim(T): %s, n_test: %d' % (str(T.shape), len(L_test))) 

    # S = mask_fp_fn(R, labels, p_th=p_th, labelize=True)

    # now consider confidence levels
    # Wu = uc.confidence(R, L_train, T=None, mode='users', topk=topk, scoring=brier_score_loss)
    # Wi = uc.confidence(R, L_train, mode='items', topk=None, scoring=brier_score_loss)

    # userCounter = select(Wu, userCounter, topk=topk)
    
    # [tip] but we cannot do np.multiply(S, W)
    if is_augmented: 
    	Ra = np.hstack((R, T))  # augmented rating matrix by combining R (from train split) and T (from test split)
        L_test_est = estimate_labels(T, p_th=0.5, pos_label=1)

        La = np.hstack((L_train, L_test_est))
        W = uc.confidence2D(Ra, La, T=None, mode='label', topk=(topk_users, topk_items), scoring=brier_score_loss)
        
        # Cui = sparse.csr_matrix.multiply(S, W) # hadamard product
        Cui = mask_over(W, Ra, La, p_th=0.5, pos_label=1, neg_label=0)
    else: 
        W = uc.confidence2D(R, L_train, T=None, mode='label', topk=(topk_users, topk_items), scoring=brier_score_loss)
        # Cui = sparse.csr_matrix.multiply(S, W) # hadamard product
        Cui = mask_over(W, R, labels, p_th=0.5, pos_label=1, neg_label=0)
    # print('(test 1b) W:\n%s\n' % W[:tu][:ti])

    

    print('(test) Cui:\n%s\n' % Cui[:10][:10])

    return Cui, T, L_train, L_test, U

# utils_cf
def toImplicitRatingMatrix(fold, **kargs):
    """

    Memo
    ----
    1. analogous to toRatings()
    """
    def to_label(y_hat, p_threshould=0.5): 
        labels = np.zeros(len(y_hat))
        for i, p in enumerate(y_hat): 
            if p >= p_threshould: 
               labels[i] = 1
        return list(labels)

    import scipy.sparse as sparse
    import common

    inputdir = kargs.get('inputdir', None)
    if inputdir is None: inputdir = ProjectPath
    train_df, train_labels, test_df, test_labels = common.read_fold(inputdir, fold) # [todo] single out this part

    # get all data IDs 
    users = train_df.columns.values

    # [note]
    #   train: predictive scores (analogous to 'ratings') in the training split 
    #   test:  predictive scores in the test split
    cols = ['train', 'test', 'train_labels', 'test_labels', ]  
    data = {col: None for col in cols}
    data['users'] = data['classifiers'] = users  # add user/classifier names

    data['train_labels'] = train_labels; data['test_labels'] = test_labels
    
    p_th = kargs.get('p_threshould')
    pos_label, neg_label = 1, 0

    for split in ['train', 'test', ]: 
        ts = train_df if split.startswith('tr') else test_df

        ts = ts.reset_index() # convert multilevel index to flat index
        idx = ts['id'].values  # item/data IDS
        assert len(idx) == len(set(idx)), "Data IDs are not unique!"
        labels = ts['label'].values # ground truth labels

        # split = 'train'
        nU = nUsers = len(users) # number of users/classifiers
        nI = nItems = len(idx)  # number of items/data points

        R, P = [], []  # predicted label matrix (somewhat analogous to the preference matrix)

        # rating matrix for the training split
        for i, user in enumerate(users): 
            predictions = ts[user].values
            y_label = to_label(predictions, p_threshould=p_th) 

            # print('(toPredictiveScores) clf: %s, predictions: %s' % (user, predictions[:10]))
            # if i == 0: assert len(idx) == len(predictions)
            
            R.append(predictions)
            P.append(y_label)

        # convert to sparse repr while masking FP and FN
        R = np.array(R); P = np.array(P)

        cond_tp = (R >= p_th) & (labels == pos_label)
        cond_tn = (R < p_th) & (labels == neg_label)
        rows, cols = np.where(cond_tp | cond_tn)
        good_labels = P[rows, cols]  # R[np.where(cond_tp | cond_tn)]

        S = sparse.csr_matrix((good_probs, (rows, cols)), shape=(nU, nI))
        
        data[split] = S

    return data  # a dictionary of 5 entries: ['train', 'test', 'train_labels', 'test_labels', 'users', ]

# for concept demonstration only
def implicit_als_slow(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10, Cui=None, fill=0):
    """ Implementation of Alternating Least Squares with implicit data. We iteratively
    compute the user (x_u) and item (y_i) vectors using the following formulas:
 
    x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
    y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))
 
    Args:
        sparse_data (csr_matrix): Our sparse user-by-item matrix
 
        alpha_val (int): The rate in which we'll increase our confidence
        in a preference with more interactions.
 
        iterations (int): How many times we alternate between fixing and 
        updating our user and item vectors
 
        lambda_val (float): Regularization value
 
        features (int): How many latent features we want to compute.
    
    Returns:     
        X (csr_matrix): user vectors of size users-by-features
        
        Y (csr_matrix): item vectors of size items-by-features

    Memo
    ----
    1. references: 

        https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

    """
    # import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve

    # Calculate the concidence for each value in our data
    if Cui is None: 
        confidence = sparse_data * alpha_val
    else: 
    	confidence = Cui   # use the precomputed confidence weights
    
    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape
    
    # We create the user vectors X of size users-by-features, the item vectors
    # Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size = (user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size = (item_size, features)))
    
    #Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(item_size)
    
    I = sparse.eye(features)
    lI = lambda_val * I

    # Start main loop. For each iteration we first compute X and then Y
    for i in range(iterations):
        print 'iteration %d of %d' % (i+1, iterations)
        
        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in range(user_size):

            # Get the user row.
            u_row = confidence[u,:].toarray() 
            if i == 0 and u == 0: print('(test) u_row:\n%s\n' % u_row[:100])

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != fill] = 1.0  # missing values (or the incorrectly classified)

            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

    
        for i in range(item_size):

            # Get the item column and transpose it.
            i_row = confidence[:,i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
            p_i[p_i != fill] = 1.0  # missing values (or the incorrectly classified)

            # Calculate Ci and Ci - I
            CiI = sparse.diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y


def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]
      
def implicit_als_cg(Cui, features=20, iterations=20, lambda_val=0.1):
    user_size, item_size = Cui.shape

    # init random values
    X = np.random.rand(user_size, features) * 0.01
    Y = np.random.rand(item_size, features) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    # ALS
    for iteration in range(iterations):
        print 'iteration %d of %d' % (iteration+1, iterations)
        least_squares_cg(Cui, X, Y, lambda_val)  # updates X, Y in place
        least_squares_cg(Ciu, Y, X, lambda_val)
    
    return sparse.csr_matrix(X), sparse.csr_matrix(Y)
  
  
def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):
    users, features = X.shape
    
    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x
    # return X, Y


#------------------------------
# FIND SIMILAR ITEMS
#------------------------------
def findSimilarItems(item_vecs, item_lookup): 

    # Let's find similar artists to Jay-Z. 
    # Note that this ID might be different for you if you're using
    # the full dataset or if you've sliced it somehow. 
    item_id = 10277

    # Get the item row for Jay-Z
    item_vec = item_vecs[item_id].T

    # Calculate the similarity score between Mr Carter and other artists
    # and select the top 10 most similar.
    scores = item_vecs.dot(item_vec).toarray().reshape(1,-1)[0]
    top_10 = np.argsort(scores)[::-1][:10]

    artists = []
    artist_scores = []

    # Get and print the actual artists names and scores
    for idx in top_10:
        artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
        artist_scores.append(scores[idx])

    similar = pd.DataFrame({'artist': artists, 'score': artist_scores})

    print similar

    return

#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------
def recommend(user_id, data_sparse, user_vecs, item_vecs, item_lookup, num_items=10):
    """Recommend items for a given user given a trained model
    
    Args:
        user_id (int): The id of the user we want to create recommendations for.
        
        data_sparse (csr_matrix): Our original training data.
        
        user_vecs (csr_matrix): The trained user x features vectors
        
        item_vecs (csr_matrix): The trained item x features vectors
        
        item_lookup (pandas.DataFrame): Used to map artist ids to artist names
        
        num_items (int): How many recommendations we want to return:
        
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items artist names and scores
    
    """
    # Get all interactions by the user
    user_interactions = data_sparse[user_id,:].toarray()

    # We don't want to recommend items the user has consumed. So let's
    # set them all to 0 and the unknowns to 1.
    user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
    user_interactions[user_interactions > 1] = 0

    # This is where we calculate the recommendation by taking the 
    # dot-product of the user vectors with the item vectors.
    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    # Let's scale our scores between 0 and 1 to make it all easier to interpret.
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions*rec_vector_scaled
   
    # Get all the artist indices in order of recommendations (descending) and
    # select only the top "num_items" items. 
    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    artists = []
    scores = []

    # Loop through our recommended artist indicies and look up the actial artist name
    for idx in item_idx:
        artists.append(item_lookup.artist.loc[item_lookup.artist_id == str(idx)].iloc[0])
        scores.append(recommend_vector[idx])

    # Create a new dataframe with recommended artist names and scores
    recommendations = pd.DataFrame({'artist': artists, 'score': scores})
    
    return recommendations

def t_confidence_weights(test_=True): 
    def select(W, counter, topk=10):
        # for i in range(W.shape[0]):
        #     if not i in np.argsort(W)[::-1][:topk]:
        #        W[i] = 0
        counter.update(np.argsort(W)[::-1][:topk])
        assert all(W[np.argsort(W)[::-1][:topk]] != 0)
        print counter
        return counter

    import utils_cf as uc 
    from collections import Counter
    
    n_fold = 5
    missing_value = -1 # marker for missing data
    topk = 30

    ### Pearson correlation (pcorr) with the true labels
    topk = 30
    corrMetrics = Metrics()  # prediction-label correlation
    topKCorrMetrics = Metrics()
    
    topk = 10
    userCounter = Counter()  # the index of users/classifiers mostly selected (those that produce most correlated predictions across CV folds)
    for fold in range(n_fold): 
    	if test_ and fold > 2: break
        data = uc.toPredictiveScores(fold)
        R = data['train']
        T = data['test']
        L_train, L_test = data['train_labels'], data['test_labels']
        U = data['users']
        print('(test) n_train: %d, n_test: %d' % (len(L_test), len(L_test)))
        n_users, n_items = R.shape[0], R.shape[1]
	
        tu, ti = 5, 10
        Wu = uc.confidence(R, L_train, T=None, mode='users', topk=topk, scoring=brier_score_loss)
        Wi = uc.confidence(R, L_train, mode='items', topk=None, scoring=brier_score_loss)

        print('(test 1): Wu (n=%d):\n%s\n' % (len(Wu), str(Wu)))
        print('(test 1): Wi (n=%d):\n%s\n' % (len(Wi), str(Wi[:ti])) )
        userCounter = select(Wu, userCounter, topk=topk)
        # W = np.outer(Wu, Wi)

        W = uc.confidence2D(R, L_train, T=None, mode='label', topk=(0, 0), scoring=brier_score_loss)
        print('(test 1b) W:\n%s\n' % W[:tu][:ti])
 
    user_cnt = userCounter.most_common(10)
    print('(test) most selected classifiers:\n%s\n' % user_cnt)
    uids = [u for u, c in user_cnt]
    print('       they are:\n%s\n' % U[uids])

    return 

def base_predictors(): 
    import cf 
    return cf.base_predictors()

def t_neighborhood_ensemble():
    import utils_cf as uc
    import evaluate
    from evaluate import PerformanceMetrics, Metrics, plot_roc

    div(message='Running memory-based approach ...', symbol='#', border=1)
    
    n_fold = 5
    missing_value = -1 # marker for missing data
    topk = 30

    ### Pearson correlation (pcorr) with the true labels
    topk = 30
    corrMetrics = Metrics()  # prediction-label correlation
    topKCorrMetrics = Metrics()
    for fold in range(n_fold): 
        data = uc.toPredictiveScores(fold)
        R = data['train']
        T = data['test']
        L_train, L_test = data['train_labels'], data['test_labels']
        print('(test) n_train: %d, n_test: %d' % (len(L_test), len(L_test)))
        n_users, n_items = R.shape[0], R.shape[1]
    
        # Use pcorr as weights to predict T 
        L_pred = uc.predictNewItemsByCorr(T, R, L_train)
        metrics = evalTestSet2(L_test, L_pred, fold=fold)  # [log]
        corrMetrics.add(metrics)

        # consider only top k most correlated classifiers' predictions and take their weighted average
        topk_corr = topk = int(math.floor(n_users/2))
        L_pred = uc.predictNewItemsByCorr(T, R, L_train, topk=topk)
        metrics = evalTestSet2(L_test, L_pred, fold=fold)  # [log]
        topKCorrMetrics.add(metrics)

    corrMetrics.report(op=np.mean, message='Memory-based ensemble based on Pearson correlation as classifier weights.')
    topKCorrMetrics.report(op=np.mean, message='Memory-based ensemble based on top %d Pearson correlation as classifier weights.' % topk)

    return

def t_wmf_ensemble(base_perf=None):
    from evaluate import Metrics, plot_roc, analyzePerf
    from utils_als import implicit_als_cg
    import utils_cf as uc

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    n_factors = 10
    n_epochs = 300
    alpha_val = 100
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    if base_perf is None: 
        base_perf = base_predictors()['bp']  # consolidated PerformanceMetrics object across CV fold

    # bpMetrics, mfMetrics = Metrics(), Metrics() # matrix factorization metrics
    perfx = []
    for fold in [0, ]: # range(n_fold): 
     
        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        Cui, T, L_train, L_test, U = uc.toConfidenceMatrix(fold, p_threshold=p_th, fill=missing_value, verbose=True, is_augmented=True)
        n_users, n_items = Cui.shape[0], Cui.shape[1]

        # n_users, n_items = R.shape[0], R.shape[1]
        print('[wmf_ensemble] dim(Cui): %s, L_train: %d, n_test: %d' % (str(Cui.shape), len(L_train), len(L_test)))

        conf_data = (Cui * alpha_val).astype('double')
        n_nonzeros = sparse.csr_matrix.count_nonzero(conf_data)
        n_zeros = n_users * n_items - n_nonzeros
        print('(test) n_zeros: %d' % n_zeros)  # [log] 104902189710

        P, Q = implicit_als_cg(conf_data, iterations=20, features=20) 

        # P = P.todense()
        # Q = Q.todense()
        print('(test) dim(P): %s, dim(Q): %s | type: %s' % (str(P.shape), str(Q.shape), type(P)))
        
        # but we only need the Q from the test split 
        Qt = Q[len(L_train):, :]   # 30 * 10, (768-x) * 10
        print('... dim(P): %s, dim(Qt): %s' % (str(P.shape), str(Qt.shape)))
        Th = np.dot(P, Qt.T)
        Th = np.array(Th.todense())
        print('(test) dim(Th): %s | type: %s' % (str(Th.shape), type(Th)))
        
        # metrics = comparePerfMetrics0(T, L_test, Th=Th, R=None, L_train=None)
        df_perf = analyzePerf(L_test, Th, method='wmf', aggregate_func=np.mean, 
            T=T, fold=fold)  # optinal params: T, fold (used to comparing the utility between T and Th)  # Th: final prediction
        perfx.append(df_perf)

        # bpMetrics.add(metrics['bp'])
        # mfMetrics.add(metrics['cf'])

    # take the mean

    # op: a combiner function for performance scores across CV folds
    # bpMetrics.report(op=np.mean, message='Average of best BP performance.')
    # mfMetrics.report(op=np.mean, message='Model-based ensemble based on NMF only.')
    
    # Q1: does the reconstructed pro

    # ret: output dictionary
    #      key: {methods}, {metrics}
    ret['wmf'] = PerformanceMetrics.consolidate(perfx)  # foreach metric, take average over CV folds
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(ret['wmf'], metric=metric, verbose=True if metric in ['auc', ] else False) 

    # how does it compare to BP? 
    docs = {'method': 'WMF'}
    PerformanceMetrics.report(p_baseline=base_perf, p_target=ret['wmf'], rule='max', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked

    return ret

def wmf_similarity_ensemble(base_perf=None):
    """
    Use NMF factorized latent matrices to compute similarity
    """ 
    import math
    from utils_als import implicit_als_cg
    from evaluate import plot_roc, analyzePerf, Metrics, PerformanceMetrics
    import utils_cf as uc

    n_fold = 5
    missing_value = 0 # marker for missing data
    p_th = 0.5

    n_users = n_items = 0 
    n_factors = 10
    n_epochs = 300
    topk = 30

    # output
    ret = {}

    # find the performance of the baseline methods; this produces a PerformanceMetrics containing a table (cols: classifiers, rows indexed by metrics)
    if base_perf is None: 
        base_perf = base_predictors()  # consolidated PerformanceMetrics object across CV fold

    # bpMetrics, fullMetrics, topKMetrics = Metrics(), Metrics(), Metrics() # matrix factorization metrics
    fullMetrics, topKMetrics = [], []
    wmfCV, topKWMFCV = [], []
    for fold in range(n_fold): 
     
        # different than toPredictiveScores(), to_rating_matrix2() masks FP, FN and possible implement other 
        # strategies that utilize the ground truths 
        Cui, T, L_train, L_test, U = uc.toConfidenceMatrix(fold, p_threshold=p_th, missing_value=missing_value, verbose=True)
        n_users, n_items = Cui.shape[0], Cui.shape[1]
        print('[nmf_ensemble] dim(T): %s, n_test: %d' % (str(T.shape), len(L_test)))

        P, Q = implicit_als_cg(Cui, iterations=20, features=20) 
        P = np.array(P.todense())
        Q = np.array(Q.todense())

        # but we only need the Q from the test split 
        Qt = Q[len(L_train):, :]   # 30 * 10, (768-x) * 10
        print('... dim(P): %s, dim(Qt): %s' % (str(P.shape), str(Qt.shape)))
        # Th = np.dot(P, Qt.T)

        # use P (user latent features) to construct Su 
        Su = uc.eval_similarity_by_latent_factors(P, epsilon=1e-9)
        assert Su.shape[0] == Su.shape[1] == Cui.shape[0]
        print('(test) Sim (Su[i,j] in [0, 1]?):\n%s\n' % Su[:4, :4])
        Th0 = uc.predict(T, Su, kind='user')
        # Th0 = np.array(Th0.todense())

        # metrics = comparePerfMetrics0(T, L_test, Th=Th0, R=None, L_train=None)
        # bpMetrics.add(metrics['bp'])
        # fullMetrics.add(metrics['cf'])
        perf0 = analyzePerf(L_test, Th0, method='wmf_similarity', aggregate_func=np.mean)  # final prediction
        fullMetrics.append(perf0)
        wmfCV.append((L_test, uc.combiner(Th0, aggregate_func=np.mean)))  # plot K-Fold CV

        # use top K only 
        topk = int(math.floor(n_users/2))
        Th_topK = uc.predict_topk(T, Su, kind='user', k=topk)
        # Th_topK = np.array(Th_topK.todense())
        # metrics = comparePerfMetrics0(T, L_test, Th=Th_topK, R=None, L_train=None)
        # topKMetrics.add(metrics['cf'])
        perf1 = analyzePerf(L_test, Th_topK, method='wmf_similarity_top%d' % topk, aggregate_func=np.mean)  # final prediction
        topKMetrics.append(perf1)
        topKWMFCV.append((L_test, uc.combiner(Th_topK, aggregate_func=np.mean)))

    # op: a combiner function for performance scores across CV folds
    # bpMetrics.report(op=np.mean, message='Average of best BP performance.')
    # fullMetrics.report(op=np.mean, message='Model-based ensemble based on NMF with full user-user similarity')
    # topKMetrics.report(op=np.mean, message='Model-based ensemble based on NMF with topk=%d' % topk)

    plot_roc(wmfCV, file_name='wmf-user-roc-%s' % Domain)  # an import from evaluate
    plot_roc(topKWMFCV, file_name='wmf-top%d-user-roc-%s' % (topk, Domain))

    ## Compare with baseline methods
    ret['wmf_sim'] = PerformanceMetrics.consolidate(fullMetrics)  # foreach metric, take average over CV folds
    ret['wmf_sim_top%d' % topk] = PerformanceMetrics.consolidate(topKMetrics)
    for metric in PerformanceMetrics.tracked: 
        ret[metric] = PerformanceMetrics.sort2(ret['wmf_sim'], metric=metric, verbose=True if metric in ['auc', ] else False) 

    # how does it compare to BP? 
    docs = {'method': 'WMF_SIM'}
    PerformanceMetrics.report(p_baseline=base_perf['bp'], p_target=ret['wmf_sim'], rule='max', descriptions=docs, verbose=True) # metrics=[] => PerformanceMetrics.tracked
    
    return ret

def t_als(): 
    import scipy.sparse as sparse

    inputdir = '/Users/pleiades/Documents/work/data/recommender/lastfm-dataset-360K'
    data_sparse, item_lookup = load_data(inputdir)
    print('(test) dim(data): %s' % str(data_sparse.shape))
    n_users, n_items = data_sparse.shape[0], data_sparse.shape[1]
    ### ALS ### 

    ## very slow ALS 
    # user_vecs, item_vecs = implicit_als_slow(data_sparse, iterations=20, features=20, alpha_val=40)
    
    ## conjugate gradient in ALS 
    alpha_val = 15
    conf_data = (data_sparse * alpha_val).astype('double')
    print('(test) conf_data (dim:%s) :\n%s\n' % (str(conf_data.shape), conf_data[:10][:10]))
    assert conf_data.shape[0] == n_users and conf_data.shape[1] == n_items
    # has 0s? 
    
    # n_nonzeros = np.count_nonzero(conf_data.todense())  # this leads to memory overhead!!!
    n_nonzeros = sparse.csr_matrix.count_nonzero(conf_data)
    n_zeros = n_users * n_items - n_nonzeros
    print('(test) n_zeros: %d' % n_zeros)  # [log] 104902189710
    # user_vecs, item_vecs = implicit_als_cg(conf_data, iterations=20, features=20) 
    # print('(test) dim(U): %s, dim(V): %s' % (str(user_vecs.shape), str(item_vecs.shape)))
    
    
   

    ### Recommendataion 

    # Let's say we want to recommend artists for user with ID 2023
    user_id = 2023

    #------------------------------
    # GET ITEMS CONSUMED BY USER
    #------------------------------

    # Let's print out what the user has listened to
    # consumed_idx = data_sparse[user_id,:].nonzero()[1].astype(str)
    # consumed_items = item_lookup.loc[item_lookup.artist_id.isin(consumed_idx)]
    # print consumed_items

    # Let's generate and print our recommendations
    # recommendations = recommend(user_id, data_sparse, user_vecs, item_vecs, item_lookup)
    # print recommendations

    return

def test(): 

    ## experiment on confidence weights
    # t_confidence_weights()

    ## slow ALS 
    # t_als()

    ## ensemble learning application 
    t_wmf_ensemble(base_perf=None)

    wmf_similarity_ensemble(base_perf=None)

    return

if __name__ == "__main__": 
    test()



