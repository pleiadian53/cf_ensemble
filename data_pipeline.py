
import pandas as pd
from pandas import DataFrame, Series

import numpy as np
from zipfile import ZipFile

# Tensorflow
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from pathlib import Path
import matplotlib.pyplot as plt

from analyzer import is_sparse


def matrix_to_decoded_dataframe(X, U=None, items=None, **kargs):
    """

    Parameters
    ----------
    `X`:   A rating matrix where rows represent users (base classifiers) and columns represent items (data points)
    `U`:   A list of users consistent with the row order of X
    `items`: A list of items/data consistent with the column order of X

    """
    df = matrix_to_dataframe(X, **kargs)

    # If `U` is given, we will further create a recommender-stytle training data, where userId are 
    # converted to their canonical names (e.g. classifier names like SVM) and optionally, 
    # each item/datatum can be given a name as well

    # Encode users/classifiers
    user_ids = list(range(X.shape[0]))
    if U is not None: 
        assert X.shape[0] == len(U)
        user_ids = U

    u2u_encoded = {x: i for i, x in enumerate(user_ids)} # map name to index, honoring the order in U
    u_encoded2u = {i: x for i, x in enumerate(user_ids)} # index to name

    # print(f"> user2user_encoded:\n{user2user_encoded}\n")

    # Encode items/data
    item_ids = list(range(X.shape[1]))
    if items is not None: 
        assert X.shape[1] == len(items)
        item_ids = items

    i2i_encoded = {x: i for i, x in enumerate(items_ids)} # map item name to index, honoring the order in items
    i_encoded2i = {i: x for i, x in enumerate(items_ids)} # index to item name

    df["userId"] = df["user"].map(u_encoded2u) # map user index (numerical) to user names (string)
    # Note: In recommender's training data, we typically use `userId` to refer to named users

    df["itemId"] = df["item"].map(i_encoded2i) # map item index to item names

    num_users = len(u2u_encoded)
    num_items = len(i2i_encoded)
    df["rating"] = df["rating"].values.astype(np.float32)

    return df
#[alias]
def rating_matrix_to_decoded_dataframe(X, U=None, items=None, **kargs): 
    return matrix_to_decoded_dataframe(X, U=U, items=items, **kargs)

def reestimate(model, X, n_train, **kargs):
    """
    Use the (TF-trained) model to reconstruct the probaility rating matrix. 

    If `n_train` is not None, re-estiamte the whole matrix X with new probabilities
    using the learned latent factors given by `model` 

    If, however, `n_train` is a number, then re-estimate only the test split of X 
    i.e. X[n_train: ] (typically denoted by `T`). 
    """
    kargs['shuffle'] = False
    kargs['normalize'] = False

    # Convert the rating matrix into (X, y)-format that the model accepts
    R, T = X[:,:n_train], X[:,n_train:]
    X_train, X_test, y_train, y_test = matrix_to_train_test_split(X, n_train, **kargs)

    ####################################
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    ####################################

    # Put the prediction back into the rating-matrix format
    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')
    
    df_train = DataFrame(X_train, columns=[col_user, col_item])
    df_train[col_value] = y_train_pred
    Rh = df_train.pivot(col_user, col_item, col_value).values
    assert Rh.shape == R.shape

    df_test = DataFrame(X_test, columns=[col_user, col_item])
    df_test[col_value] = y_test_pred
    Th = df_test.pivot(col_user, col_item, col_value).values
    assert Th.shape == T.shape

    return (Rh, Th)
# [alias]
def reconstruct(model, X, n_train, **kargs): 
    return reestimate(model, X, n_train, **kargs)

def reestimate_unreliable_only(model, X, n_train, Pc, C=None, **kargs):
    L = kargs.pop('L', [])
    p_threshold = kargs.pop('p_threshold', [])
    use_confidence_weights = kargs.pop('use_confidence_weights', False) 
    verbose = kargs.pop('verbose', 0)

    Rh, Th = reestimate(model, X, n_train, **kargs)
    Xh = np.hstack((Rh, Th))
    XXh = interpolate(X, Xh, Pc, C=C, L=L, p_threshold=p_threshold, 
              use_confidence_weights=use_confidence_weights, verbose=verbose)

    return (XXh[:,:n_train], XXh[:,n_train:])

def interpolate(X, Xh, Pc=None, C=None, L=[], p_threshold=[], use_confidence_weights=False, verbose=0): 
    # from analyzer import is_sparse
    import utils_cf as uc

    W = Pc
    if use_confidence_weights: # Use confidence scores as weights 
        assert C is not None
        if is_sparse(C): C = C.toarray()
        W = uc.softmax(C, axis=0)
    else: 
        if is_sparse(W): W = W.A 
        # Note: Why converting to dense? Subtracting a sparse matrix from a nonzero scalar is not supported 
        #       E.g. can't do 1.0-W if W is sparse

        if W is None: 
            W, _ = uc.probability_filter(X, L, p_threshold)
        else: 
            if verbose > 1: 
                print('(reconstruct) Converting color matrix to a standard probability filter (aka preference matrix) ...')
            W = uc.to_preference(W) 

    wmin, wmax = np.min(W), np.max(W)
    assert wmin >= 0 and wmax <= 1, "W is not a probability filter | values: [{}, {}]".format(wmin, wmax)

    # Note: 
    # W as a weight matrix: the higher the W[i,j], the more weight on X[i,j]
    # W as a preference matrix: W[i,j] = 1 => use X[i,j] (original value), if W[i,j] = 0, use Xh[i,j] (re-est value)

    # Xh = uc.replace(P, Q, X=(W, X), canonicalize=True, 
    #         fill=null_marker, predict_func=ua.predict_by_factors, name=name)

    # uc.interpolate(X1, X2, W1, W2)
    #   if W1[i,j]=1, then use X1[i, j]
    #   if W1[i,j]=0 => W2[i,j]=1 => use X2[i,j]; effectively replacing X1[i,j] by X2[i, j]
    Xh_partial = uc.interpolate(X, Xh, W, 1.0-W) # replace X by Xs selectively according to W (and 1-W)

    return Xh_partial

def matrix_to_train_test_split(X, n_train, **kargs):
    assert X.shape[1] >= n_train, f"Error: Sample size {X.shape[1]} while n_train={n_train}"

    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')

    df = matrix_to_dataframe(X, **kargs)
    min_score = min(df[col_value])
    max_score = max(df[col_value])

    # min and max ratings will be used to normalize the ratings later (not necessary for probability scores)
    normalize = kargs.get('normalize', False)
    if normalize: 
        y = df[col_value].apply(lambda x: (x - min_score) / (max_score - min_score)).values
    else: 
        y = df[col_value].values

    # Train-test spilt 
    df_train = df.loc[df[col_item] < n_train]
    X_train = df_train[[col_user, col_item]] # `X_train` in user-item-index pair format 
    y_train = y[:n_train] # ratings in the training split

    df_test = df.loc[df[col_item] >= n_train] 
    X_test = df_test[[col_user, col_item]] # `X_test` in user-item-index pair format
    y_test = y[n_train:]

    assert X_train.shape[0] + X_test.shape[0] == X.size, \
            f"shape(X_train): {X_train.shape} and shape(X_test): {X_test.shape} not consistent with shape(X): {X.shape}"

    return (X_train, X_test, y_train, y_test)
#[alias]
def rating_matrix_to_train_test_split(X, n_train, **kargs): 
    return matrix_to_train_test_split(X, n_train, **kargs)

def make_sample_weights(R, C, Pc):
    import utils_cf as uc
    # from analyzer import is_sparse

    assert R.shape == C.shape
    assert R.shape == Pc.shape 

    if is_sparse(C): C = C.A
    if is_sparse(Pc): Pc = Pc.A

    Po = uc.to_preference(Pc) # to probablity filter where {TP, TN} => 1 and {FP, FN} => 0
    Cn = Po * C # from full confidence matrix to masked confidence matrix (where FPs, FNs are masked by 0)

    # Structure Cn in column coordiante format
    col_value = 'weight'
    df_conf =  matrix_to_dataframe(Cn, col_value=col_value, shuffle=False)

    return df_conf[col_value].values    

def matrix_to_augmented_training_data(R, C, Pc, **kargs):
    assert R.shape == C.shape
    assert R.shape == Pc.shape 

    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')

    if is_sparse(C): C = C.A
    if is_sparse(Pc): Pc = Pc.A

    # Structure R, C and Pc in column coordiante format
    col_value, col_conf, col_color = ('rating', 'weight', 'color')
    df_score = matrix_to_dataframe(R, col_value=col_value, shuffle=False)
    df_conf =  matrix_to_dataframe(C, col_value=col_conf, shuffle=False)
    df_color = matrix_to_dataframe(Pc, col_value=col_color, shuffle=False)

    X = df_score[[col_user, col_item]]

    # `rating` is the supervised signal
    # min and max ratings will be used to normalize the ratings later (not necessary for probability scores)
    normalize = kargs.get('normalize', False)
    if normalize: 
        min_score = min(df[col_value])
        max_score = max(df[col_value])
        y = df_score[col_value].apply(lambda x: (x - min_score) / (max_score - min_score)).values
    else: 
        y = df_score[col_value].values

    N = len(y)
    # Confidence scores (C) are used to weigh individual training instance (x, y)
    weights = df_conf[col_conf].values
    colors = df_color[col_color].values
    assert (len(weights) == N) and (len(colors) == N)

    # Shuffle the data
    shuffle = kargs.get('shuffle', False)
    random_state = kargs.get('random_state', 53)
    if shuffle:  
        # np.random.seed(random_state)
        shuffler = np.random.RandomState(seed=random_state).permutation(N)
        return (X[shuffler], y[shuffler], weights[shuffler], colors[shuffler])

    return (X, y, weights, colors)
#[alias]
def rating_matrix_to_augmented_training_data(R, C, Pc, **kargs): 
    return matrix_to_augmented_training_data(R, C, Pc, **kargs)

def unravel(R, **kargs): 
    
    if is_sparse(R): R = R.A
    y = np.ravel(R) # order='C'

    if kargs.get("verify", False): 
        # This operation should be the same as: 
        _, yp = matrix_to_training_data(R, normalize=False)
        assert np.all(y == yp)
    
    if kargs.get('normalize', False): 
        min_score = min(values)
        max_score = max(values)
        y = (y - min_score) / (max_score - min_score)
    
    return y

def matrix_to_training_data(R, **kargs): 
    """
    Convert a rating matrix (R) into a training data set in the user-item-pair format. 
    
    The user-item-pair format is a 2-tuple (X, y), where

    `X` is a dataframe with two columns: user, item. 
    Each user is represented by the positional index consistent with R's rows; similarly, 
    each item is represented by the positional index consistent with R's columns. 

    `y` is the labels (or any supervised signals), each of which corresponds to a user-item pair


    Parameters
    ----------
    `R`: A rating matrix where rows represent users (base classifiers) and columns represent items (data points)
    """
    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')

    if is_sparse(R): R = R.A

    df = matrix_to_dataframe(R, **kargs)
    X = df[[col_user, col_item]]

    # min and max ratings will be used to normalize the ratings later (not necessary for probability scores)
    normalize = kargs.get('normalize', False)
    if normalize: 
        min_score = min(df[col_value])
        max_score = max(df[col_value])
        y = df[col_value].apply(lambda x: (x - min_score) / (max_score - min_score)).values
    else: 
        y = df[col_value].values

    return (X, y)
#[alias]
def matrix_to_user_item_pair_format(X, **kargs):
    return matrix_to_training_data(X, **kargs)
def rating_matrix_to_training_data(X, **kargs):
    return matrix_to_training_data(X, **kargs)

def matrix_to_dataframe(X, **kargs): 
    """

    Parameters
    ----------
    `X`: A rating matrix where rows represent users (base classifiers) and columns represent items (data points)

    Memo
    ----
    1. df.stack(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html

    """
    user_ids = list(range(X.shape[0]))
    item_ids = list(range(X.shape[1])) # [todo] subsampling

    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')

    if is_sparse(X): X = X.A

    # Structure training data in "user-item-value format" in which each training instance x
    # is a user-item pair in their encoded indices, and the label y represents the rating 
    # given by x (the user rating on a given item), which corresponds to classifier predicting 
    # a data point in terms of P(y=1|x)
    df = DataFrame(X, index=user_ids, columns=item_ids)
    df = df.stack().reset_index().rename(columns={'level_0':col_user,'level_1':col_item, 0:col_value})
    df.astype({col_user: 'int32', col_item: 'int32', col_value:'float32'})

    # Shuffle the data
    shuffle = kargs.get('shuffle', False)
    random_state = kargs.get('random_state', 53)
    if shuffle: 
        df = df.sample(frac=1, random_state=random_state)

    return df
def rating_matrix_to_dataframe(X, **kargs): 
    # [todo] Check matrix type 
    return matrix_to_dataframe(X, **kargs)

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000, random_state=53):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=random_state)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

def demo_to_rating_matrix(): 
    # Download the actual data from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    # Use the ratings.csv file
    movielens_data_file_url = (
        "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    )
    movielens_zipped_file = keras.utils.get_file(
        "ml-latest-small.zip", movielens_data_file_url, extract=False
    )
    keras_datasets_path = Path(movielens_zipped_file).parents[0]
    movielens_dir = keras_datasets_path / "ml-latest-small"

    # Only extract the data the first time the script is run.
    if not movielens_dir.exists():
        with ZipFile(movielens_zipped_file, "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=keras_datasets_path)
            print("Done!")

    ratings_file = movielens_dir / "ratings.csv"
    df = pd.read_csv(ratings_file)


    return

def test(): 

    return

if __name__ == "__main__":
    test() 