import numpy as np
from utilities import normalize

try: 
    import faiss
except: 
    import utils_sys as usys
    # pip install faiss
    usys.install('faiss')
    import faiss

class FaissKNN:
    def __init__(self, k=5, normalize=False):
        self.index = None
        self.y = None
        self.y_tag = None # other meta data for the label/target such as polarities, colors
        self.k = k
        self.normalize_input = normalize

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1]) # Each x in X is in row-vector format i.e. X has shape  (n_instances, n_dim)
        # Note: Rating matrix (X), however, is in column-vector format; therefore, we need to remember to take transpose before using it as an input
  
        if self.normalize_input: 
            X = normalize(X, axis=1) # X is in row-vector format

        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        # shape(distances): (n_instances, k)
        # shape(indices):   (n_instances, k)

        votes = self.y[indices] # note: shape(votes)=shape(indices)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        # np.bincount([1, 1, 1, 0, 1, 0, 0, 0, 1, 1]) 
        # ~> array([4, 6]) because index 0 occurs 4 times, and 1 occurs 6 times
        return predictions
    def search(self, X): 
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return distances, indices