
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


def rating_matrix_to_decoded_dataframe(X, U=None, items=None, **kargs):
    """

    Parameters
    ----------
    `X`:   A rating matrix where rows represent users (base classifiers) and columns represent items (data points)
    `U`:   A list of users consistent with the row order of X
    `items`: A list of items/data consistent with the column order of X

    """
    df = rating_matrix_to_dataframe(X, **kargs)

    # If `U` is given, we will further create a recommender-stytle training data, where userId are 
    # converted to their canonical names (e.g. classifier names like SVM) and optionally, 
    # each item/datatum can be given a name as well

    # Encode users/classifiers
    user_ids = list(range(X.shape[0]))
    if U is not None: 
        assert X.shape[0] == len(U)
        user_ids = U

    u2u_encoded = {x: i for i, x in enumerate(user_ids)} # name to index
    u_encoded2u = {i: x for i, x in enumerate(user_ids)} # index to name

    # print(f"> user2user_encoded:\n{user2user_encoded}\n")

    # Encode items/data
    item_ids = list(range(X.shape[1]))
    if items is not None: 
        assert X.shape[1] == len(items)
        item_ids = items

    i2i_encoded = {x: i for i, x in enumerate(items_ids)} # item name to index
    i_encoded2i = {i: x for i, x in enumerate(items_ids)} # index to item name

    df["userId"] = df["user"].map(u_encoded2u) # map user index (numerical) to user names (string)
    # Note: In recommender's training data, we typically use `userId` to refer to named users

    df["itemId"] = df["item"].map(i_encoded2i) # map item index to item names

    num_users = len(u2u_encoded)
    num_items = len(i2i_encoded)
    df["rating"] = df["rating"].values.astype(np.float32)

    return df

def training_data_to_rating_matrix(X, y, **kargs):
    pass

def rating_matrix_to_training_data(X, **kargs): 
    """

    Parameters
    ----------
    `X`: A rating matrix where rows represent users (base classifiers) and columns represent items (data points)
    """
    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')

    df = rating_matrix_to_dataframe(X, **kargs)
    X = df[[col_user, col_item]]

    # min and max ratings will be used to normalize the ratings later (not necessary for probability scores)
    normalize = kargs.get('normalize', False)
    if normalize: 
        min_rating = min(df[col_value])
        max_rating = max(df[col_value])
        y = df[col_value].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else: 
        y = df[col_value].values

    return (X, y)

def rating_matrix_to_dataframe(X, **kargs): 
    """

    Parameters
    ----------
    `X`: A rating matrix where rows represent users (base classifiers) and columns represent items (data points)

    Memo
    ----
    1. df.stack(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.stack.html

    """
    user_ids = list(range(X.shape[0]))
    item_ids = list(range(X.shape[1]))

    col_user = kargs.get('col_user', 'user')
    col_item = kargs.get('col_item', 'item')
    col_value = kargs.get('col_value', 'rating')

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