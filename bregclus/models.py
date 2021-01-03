import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class BregmanHard(BaseEstimator, ClusterMixin):
    """
    TODOC
    """

    def __init__(self, n_clusters=3, divergence=None, n_iters=1000, has_cov=False):
        self.__n_clusters = n_clusters
        self.__divergence = divergence
        self.__n_iters = n_iters
        self.__has_cov = has_cov

    def fit(self, X):
        self.__create_params(X)
        for _ in range(self.__n_iters):
            H = self.__assignments(X)
            self.__reestimate(X, H)

    def __create_params(self, X):
        self.params = self.__init_params(X)
        if self.__has_cov:
            self.cov = self.__init_cov(X)

    def __init_params(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx[:self.__n_clusters]]

    def __init_cov(self, X):
        return np.concatenate([np.expand_dims(np.eye(X.shape[1]), axis=0) for _ in range(self.__n_clusters)])

    def __assignments(self, X):
        if self.__has_cov:
            H = self.__divergence(X, self.params, self.cov)
        else:
            H = self.__divergence(X, self.params)
        return np.argmin(H, axis=1)

    def __reestimate(self, X, H):
        for k in range(self.__n_clusters):
            X_k = X[H==k]
            self.params[k] = np.mean(X_k, axis=0)
            if self.__has_cov:
                X_mk = X-self.params[k]
                self.cov[k] = np.einsum("ij,ik->jk", X_mk, X_mk)/X_k.shape[0]

    def predict(self, X):
        return self.__assignments(X)
