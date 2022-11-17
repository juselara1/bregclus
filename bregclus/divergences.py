import numpy as np
import functools

def distance_function_vec(func):
    """
    This decorates any distance function that expects two vectors as input. 
    """
    @functools.wraps(func)
    def wraped_distance(X, Y):
        """
        Computes a pairwise distance between two matrices.

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
        
        # naive implementation (without vectorization idk if it possible to do it in a generic way... maybe with jax)
        # probably it can be faster if we directly use numpy instead of list of compreensions to numpy
        
        # builds the np.array (batch_size, n_cluster) by testing the func for all combinations of XxY.
        return np.array([[func(sample, cluster_center) for cluster_center in Y] for sample in X])

    return wraped_distance

def _euclidean_vectorized(X,Y):
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
    
    # same computation as the _old_euclidean function, but a new axis is added
    # to X so that Y can be directly broadcast, speeding up computations
    return np.sqrt(np.sum((np.expand_dims(X, axis=1)-Y)**2, axis=-1))

def _old_euclidean(X, Y):
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

# expose the vectorized version as the default one
euclidean=_euclidean_vectorized

def _mahalanobis_vectorized(X, Y, cov):
    
    diff = np.expand_dims(np.expand_dims(X, axis=1)-Y, axis=-1)
    return np.sum(np.squeeze(((np.linalg.pinv(cov)@diff)*diff)), axis=-1)

def _old_mahalanobis(X, Y, cov):
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

mahalanobis = _mahalanobis_vectorized

def _squared_manhattan_vectorized(X, Y):
    
    return np.sum(np.abs(np.expand_dims(X, axis=1)-Y), axis=-1)**2

def _old_squared_manhattan(X, Y):
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

squared_manhattan = _squared_manhattan_vectorized

