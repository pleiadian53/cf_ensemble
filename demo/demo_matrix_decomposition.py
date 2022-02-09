


# >>> from scipy.sparse import csc_matrix
# >>> from scipy.sparse.linalg import svds
# >>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
# >>> u, s, vt = svds(A, k=2) # k is the number of factors
# >>> s
# array([ 2.75193379,  5.6059665 ])
def svd(X, n_factors=2):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import svds

    Xs = csc_matrix(X, dtype=float)
    u, s, vt = svds(A, k=n_factors) # k is the number of factors


    return

def pca(): 
    """

    References
    ----------
    1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html

    2. Data Science blogs: 
       https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
       
    """
    from sklearn.decomposition import PCA
    # from scipy.stats import zscore  # preprocessing the data
    

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)


    return

def test():


    return


if __name__ == "__main__":
    test() 



