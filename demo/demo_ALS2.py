import sys
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random

from sklearn.preprocessing import MinMaxScaler

import implicit # The Cython library, see https://github.com/benfred/implicit


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


2. Nicolas Hug: SVD, Matrix Factorization 
   a. SGD
       http://nicolas-hug.com/blog/matrix_facto_3
       http://nicolas-hug.com/blog/matrix_facto_4

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

# Load the data like we did before

def load_data(inputdir=None):
    # inputdir = '/Users/pleiades/Documents/work/data/recommender/lastfm-dataset-360K' 
    import scipy.sparse as sparse
    # from scipy.sparse.linalg import spsolve
    # from sklearn.preprocessing import MinMaxScaler

    #-------------------------
    # LOAD AND PREP THE DATA
    #-------------------------
    input_file = 'usersha1-artmbid-artname-plays.tsv'
    if inputdir is None: inputdir = ProjectPath
    input_path = os.path.join(inputdir, input_file) # 'data/usersha1-artmbid-artname-plays.tsv'

    raw_data = pd.read_table(input_path)
    raw_data = raw_data.drop(raw_data.columns[1], axis=1)
    raw_data.columns = ['user', 'artist', 'plays']

    # Drop NaN columns
    data = raw_data.dropna()
    data = data.copy()

    # Create a numeric user_id and artist_id column
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")
    data['user_id'] = data['user'].cat.codes
    data['artist_id'] = data['artist'].cat.codes

    # The implicit library expects data as a item-user matrix so we
    # create two matricies, one for fitting the model (item-user) 
    # and one for recommendations (user-item)
    sparse_item_user = sparse.csr_matrix((data['plays'].astype(float), (data['artist_id'], data['user_id'])))
    sparse_user_item = sparse.csr_matrix((data['plays'].astype(float), (data['user_id'], data['artist_id'])))

    return (sparse_item_user, sparse_user_item)

def fit(sparse_item_user): 
    # Initialize the als model and fit it using the sparse item-user matrix
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

    # Calculate the confidence by multiplying it by our alpha value.
    alpha_val = 15
    data_conf = (sparse_item_user * alpha_val).astype('double')

    print('(test) data_conf:\n%s\n' % (data_conf[:10][:10]))

    # Fit the model
    model.fit(data_conf)
   
    return model

#---------------------
# FIND SIMILAR ITEMS
#---------------------
def findSimilarItems(): 
    # Find the 10 most similar to Jay-Z
    item_id = 147068 #Jay-Z
    n_similar = 10

    # Get the user and item vectors from our trained model
    user_vecs = model.user_factors
    item_vecs = model.item_factors

    # Calculate the vector norms
    item_norms = np.sqrt((item_vecs * item_vecs).sum(axis=1))

    # Calculate the similarity score, grab the top N items and
    # create a list of item-score tuples of most similar artists
    scores = item_vecs.dot(item_vecs[item_id]) / item_norms
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    similar = sorted(zip(top_idx, scores[top_idx] / item_norms[item_id]), key=lambda x: -x[1])

    # Print the names of our most similar artists
    for item in similar:
        idx, score = item
        print data.artist.loc[data.artist_id == idx].iloc[0]

    return 


#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------

def recommend(user_id, sparse_user_item, user_vecs, item_vecs, num_items=10):
    """The same recommendation function we used before"""

    user_interactions = sparse_user_item[user_id,:].toarray()

    user_interactions = user_interactions.reshape(-1) + 1
    user_interactions[user_interactions > 1] = 0

    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions * rec_vector_scaled

    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    artists = []
    scores = []

    for idx in item_idx:
        artists.append(data.artist.loc[data.artist_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'artist': artists, 'score': scores})

    return recommendations

### Get the trained user and item vectors. We convert them to 
# csr matrices to work with our previous recommend function.
# user_vecs = sparse.csr_matrix(model.user_factors)
# item_vecs = sparse.csr_matrix(model.item_factors)

# # Create recommendations for user with id 2025
# user_id = 2025

# recommendations = recommend(user_id, sparse_user_item, user_vecs, item_vecs)

# print recommendations

def test(): 
    inputdir = '/Users/pleiades/Documents/work/data/recommender/lastfm-dataset-360K' 
    s_item_user, s_user_item = load_data(inputdir)

    model = fit(s_item_user)

    return 

if __name__ == "__main__":
    test()
