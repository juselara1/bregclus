import numpy as np

def euclidean(X, Y):
    """
    Computes a pairwise Euclidean distance between two matrices: D_ij=||x_i-y_j||^2.

    Parameters
    ----------
        X: array-like, shape=(batch_size, n_features)
           Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
           Matrix in which each row represents the mean vector of each cluster.

    Returns
    -------
        D: array-like, shape=(batch_size, n_clusters)
           Matrix of paiwise dissimilarities between the batch and the cluster's parameters.

    """
    func = lambda x_i: np.sqrt(np.sum((x_i-Y)**2, axis=1))
    return np.array([func(x_i) for x_i in X])

def mahalanobis(X, Y, cov):
    """
    Computes a pairwise Mahalanobis distance between two matrices: (x_i-y_j)^T Cov_j (x_i-y_j).

    Parameters
    ----------
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        cov: array-like, shape=(n_clusters, n_features, n_features)
            Tensor with all the covariance matrices.

    Returns
    -------
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.

    """
    def func(x_i):
        diff = np.expand_dims(x_i-Y, axis=-1)
        return np.squeeze(np.sum((np.linalg.pinv(cov)@diff)*diff, axis=1))
    return np.array([func(x_i) for x_i in X])

def squared_manhattan(X, Y):
    """
    Computes a pairwise squared manhattan distance between two matrices: D_ij=sum(|x_i|-|y_j|)^2.

    Parameters
    ----------
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.

    Returns
    -------
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.

    """
    func = lambda x_i: np.sum(np.abs(x_i-Y), axis=1)**2
    return np.array([func(x_i) for x_i in X])

