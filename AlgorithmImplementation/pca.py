from numpy import *
# PCA analysis
def pca(X):
    """
    :param X: n-by-d data matrix X. Rows of X correspond to observations and columns correspond to dimensions.
    :return: Y - transformed new coordinate in new space, eig_vec - eigenvectors, eig_value
    """
    X = array(X);
    if (X.ndim != 2) or (X.shape[0]==1):
        raise ValueError('dimension of X should be 2 and X shoud have >1 samples')
    #subtract mean
    avg = mean(X,axis=0)
    avg = tile(avg,(X.shape[0],1))
    X -= avg;#WARNING: INPUT X WILL BE CHANGED
    #covariance matrix
    C = dot(X.transpose(),X)/(X.shape[0]-1)
    eig_values,eig_vecs = linalg.eig(C)
    idx = eig_values.argsort()
    idx = idx[ : :-1]
    eig_values = eig_values[idx]
    eig_vecs = eig_vecs[:,idx]
    #new coordinate in new space
    Y = dot(X,eig_vecs)
    
    return (Y, eig_vecs, eig_values
