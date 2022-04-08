
import pandas as pd
from pandas import DataFrame, Series

import numpy as np
import itertools
from zipfile import ZipFile

# Tensorflow
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from pathlib import Path

# Plot
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import seaborn as sns

from analyzer import is_sparse


def load_pretrained_level1_data(clf=None, fold_number=0, data_dir='./data', verbose=1):
    import utils_stacking as ustk   
    import utils_cf as uc 

    # Assuming that pre-trained level-1 datasets are available under ./data, we can simply instantiate a CFStacker and 
    # fetch the pre-trained data ...
    if clf is None: 
        meta_set = ustk.CFStacker.cf_fetch2(fold_number=fold_number, data_dir=data_dir)
    else: 
        # Load pre-trained level-1 data using an active CFStacker instance
        meta_set = clf.cf_fetch()
    assert len(meta_set) > 0, f"[I/O] No meta-data found at data_dir:\n{data_dir}\n"

    # Read training spilt
    X_train, y_train = meta_set['train']['X'], meta_set['train']['y'] 
    n_train = X_train.shape[0] # column vector format

    # Read test split
    X_test = meta_set['test']['X']
    y_test = None
    try: 
        y_test = meta_set['test']['y']
    except: 
        print("[I/O] test label is not available yet. Run the previous code block first.")

    # Names of the base classifiers (nicknamed "users")
    U = meta_set['train']['U']
    if verbose: 
        print(f"[info] list of base classifiers:\n{U}\n")

    ### Structure the rating/probability matrix

    # Rataing/probability matrix and labels for the TRAIN set
    R = X_train.T # transpose because we need users by items (or classifiers x data) for CF
    assert R.shape[1] == n_train, f"R should be a users-by-items rating matrix but has shape: {R.shape}, n_train={n_train}"
    L_train = y_train

    # Rating matrix and labels for the test set
    T = X_test.T
    L_test = y_test

    if verbose: 
        p_threshold = uc.estimateProbThresholds(R, L=L_train, pos_label=1, policy='fmax')
        print(f"[info] probability thresholds:\n{p_threshold}\n")

    # Use "estimated labels" for the test set; not the true label `L_test` that we are trying to predict
    # lh = uc.estimateLabels(T, p_th=p_threshold) # We cannot use L_test (cheating), but we have to guesstimate
    # L = np.hstack((L_train, lh)) 
    # X = np.hstack((R, T)) 

    return (R, T, U, L_train, L_test)

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


def reestimate(model, X, n_train=None, **kargs):
    """
    Use the (TF-trained) model to reconstruct the probaility rating matrix. 

    If `n_train` is not None, re-estiamte the whole matrix X with new probabilities
    using the learned latent factors given by `model` 

    If, however, `n_train` is a number, then re-estimate only the test split of X 
    i.e. X[n_train: ] (typically denoted by `T`). 
    """
    import cf_models as cm

    # return value (Rh, Th), where `Rh` is the reestimated R and `Th` is the reestimated T
    return cm.reestimate(model, X, n_train, **kargs)
# [alias]
def reconstruct(model, X, n_train=None, **kargs): 
    return reestimate(model, X, n_train, **kargs)

def reestimate_unreliable_only(model, X, *, Pc, C=None, n_train=None, **kargs):
    import cf_models as cm
    return cm.reestimate_unreliable_only(model, X, Pc=Pc, C=C, n_train=None, **kargs)


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
        df[col_value] = df[col_value].apply(lambda x: (x - min_score) / (max_score - min_score)).values

    # Train spilt 
    df_train = df[df[col_item] < n_train]
    X_train = df_train[[col_user, col_item]] # `X_train` in user-item-index pair format 
    y_train = df_train[col_value].values # ratings in the training split

    # Test split
    df_test = df[df[col_item] >= n_train] 
    X_test = df_test[[col_user, col_item]] # `X_test` in user-item-index pair format
    y_test = df_test[col_value].values

    assert X_train.shape[0] + X_test.shape[0] == X.size, \
            f"shape(X_train): {X_train.shape} and shape(X_test): {X_test.shape} not consistent with shape(X): {X.shape}"
    assert (y_train.size == X_train.shape[0]) and (y_test.size == X_test.shape[0])

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

def make_seq2seq_training_data(R, L, include_label=False, **kargs): 
    """

    Parameters
    ----------
    R: Rating matrix
    L: Label matrix (e.g. a probability filter where 1 represents reliable entries and 0 represents unreliable entries)

    Returns
    -------
    A 2-tuple (X, Y), where 
    X: 
    Y: 
    """
    import polarity_models as pmodel

    # return (X, Y)
    return pmodel.make_seq2seq_training_data(R, L, include_label, **kargs)

def matrix_to_augmented_training_data2(R, C, Pc, *, p_threshold=[], **kargs): 
    """
    Similar to matrix_to_augmented_training_data() but outputs the target ("y_true") by column-stacking 
    the label and other supporting information that CFNet's loss function depends on. 

    A loss function typically takes on the label as a vector (y_true), with which the model prediction
    (y_pred) is compared to define the loss. `y_true` can be continous values, such as the rating, that we wish for the
    model to approximate; `y_true` can also be a vector of class labels in classification tasks. 

    However, in a more general setting, we may wish to design a custom loss function that takes into account 
    not only the label (e.g. ratings) but also other useful information that could help us learn the latent factors 
    with desirable properties. These additional pieces of information could include, for instance, 
    condidence scores (unpacked from C) and colors (unpacked from Pc), in which case, the "y_true" would then comprise 
    three column vectors: 

    1. ratings (which corresponds to the label typically considered when we deal with a regression problem approximating the rating)
    2. confidence scores 
    3. colors

    The shape of `y_true` in this case is N x 3, where N is the sample size. 
    """
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

    # probability thresholds
    has_threshold = False
    if len(p_threshold) > 0: 
        thresholds = broadcast(R, p_threshold)
        has_threshold = True

    # Shuffle the data
    shuffle = kargs.get('shuffle', False)
    random_state = kargs.get('random_state', 53)
    if shuffle:  
        # np.random.seed(random_state)
        shuffler = np.random.RandomState(seed=random_state).permutation(N)

        if has_threshold: 
            return (X[shuffler], np.column_stack([ y[shuffler], weights[shuffler], colors[shuffler], thresholds[shuffler] ]))
        return (X[shuffler], np.column_stack( [y[shuffler], weights[shuffler], colors[shuffler]] ))

    if has_threshold: 
        return (X, np.column_stack([y, weights, colors, thresholds] ))
    return (X, np.column_stack([y, weights, colors])) # (X, y_augmented)-format

def matrix_to_augmented_training_data(R, C, Pc, **kargs):
    """
    Given the rating matrix (R) and its corresponding confidence matrix (C) and color matrix (Pc), 
    create training data in user-item-pair format for training CFNet model. 
    """

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

def broadcast(R, v, **kargs):
    # `v` is a vector of the same dimension as the row of R (R.shape[0])
    # An example: probability thresholds or `p_threshold`

    # As entries of R[i][j] get unraveled to a long vector of dimension R.size, 
    # sometimes it may be desirable to match `v` with this long vector, meaning that 
    # each vector component of v (vi) gets repeated R.shape[1] times

    # This can be useful in constructing loss functions (e.g. the loss function 
    # that takes into account probability thresholds)

    assert len(v) == R.shape[0]
    v_broadcast = np.hstack([np.repeat(v[i], R.shape[1]) for i in range(len(v))])
    assert len(v_broadcast) == R.size

    return v_broadcast

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

def zip_user_item_pairs(R, item_ids, user_ids=[], **kargs):

    n_users, _ = R.shape 

    if len(user_ids) == 0: user_ids = list(range(n_users))
    assert len(item_ids) == len(user_ids)

    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')

    return DataFrame(zip(user_ids, item_ids), columns=[col_user, col_item])

def make_user_item_pairs(R, *, user_ids=[], item_ids=[], **kargs):
    # import pandas as pd 

    n_users, n_items = R.shape 

    if len(user_ids) == 0: user_ids = list(range(n_users))
    if len(item_ids) == 0: item_ids = list(range(n_items))    

    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')

    # dfx = []
    # for uid in user_ids: 
    #     # IDs are structured in a way that reflects each user rating all the given items 
    #     adict = {col_user: np.repeat(uid, len(item_ids)), col_item: item_ids}
    #     dfi = DataFrame(adict, columns=[col_user, col_item])
    #     dfx.append(dfi)
    # df = pd.concat(dfx, ignore_index=True)
    return DataFrame(itertools.product(user_ids, item_ids), columns=[col_user, col_item])

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

    Output 
    ------
    A dataframe with three columns representing user, item and value; the value can be any numerical quantities 
    that are domain-specific. We generally refer to it as a "rating" or a score. In ensemble learning, the value 
    is the conditional probaiblity estimate P(y=1|x). 

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

    # Structure training data in "user-item-value format" as a dataframe with three columns representing 
    # user, item and value. Specifically, each training instance `x` (in X) is represented by the first two columns, 
    # user and item in their encoded indices; the 3rd column is the "value" associated with x, which corresponds 
    # to the rating given by the user to an item. This is analogous to the classifier's prediction/rating 
    # on a data point (as an item) in terms of the conditional probability P(y=1|x).
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

def generate_imbalanced_data(class_ratio=0.95, verbose=1):
    # from sklearn import datasets
    # import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from collections import Counter
    import utils_classifier as uclf

    # get the dataset
    c_ratio = class_ratio

    def get_dataset(n_samples=5000, noise=True):
        if noise: 
            X,y = make_classification(n_samples=n_samples, n_features=100, n_informative=30, 
                            n_redundant=6, n_repeated=3, n_classes=2, n_clusters_per_class=1,
                                class_sep=2,
                                flip_y=0.2, # <<< 
                                weights=[c_ratio, ], random_state=17)
        else: 
            X,y = make_classification(n_samples=n_samples, n_features=100, n_informative=30, 
                                n_redundant=6, n_repeated=3, n_classes=2, n_clusters_per_class=1,
                                    class_sep=2, 
                                    flip_y=0, weights=[c_ratio, ], random_state=17)
        return X, y

    X, y =  get_dataset(noise=True)

    uniq_labels = np.unique(y)
    n_classes = len(uniq_labels)

    # Turn into a binary classification problem 
    if n_classes > 2: 
        y0 = y
        y, y_map, le = uclf.to_binary_classification(y, target_class=2)
        
        if verbose > 1: 
            print('> y before:\n', y0)
            print('> y after:\n', y)

    print(f'> n_classes: {n_classes}\n{uniq_labels}\n')

    counter = Counter(y)
    if verbose: print(f'> counts:\n{counter}\n') 

    # Plot data
    f, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(20,8))
    sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1);
    ax1.set_title("With Noise");
    plt.show();
    
    return X, y

def demo_data_processing():

    X = np.random.choice(range(10), (5, 100))
    n_train = 50
    for n_train in [50, 100]: 
        X_train, X_test, y_train, y_test = matrix_to_train_test_split(X, n_train=n_train)
        # Expected: size(X_train) = 5 * 50 = 250, size(X_test) = 5 * 50 = 250
        print(f"> n_train={n_train}")
        print(f"> shape of X_train, y_train: {X_train.shape}, {y_train.shape}")
        print(f"> shape of X_test, y_test:   {X_test.shape},  {y_test.shape}")

    return

def test(): 
    demo_data_processing()

    return

if __name__ == "__main__":
    test() 