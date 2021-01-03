import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from bregclus.divergences import euclidean

class BregmanHard(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters, divergence=euclidean, n_iters=1000, has_cov=False,
                 initializer="rand", init_iters=100):
        """
        Bregman Hard Clustering Algorithm

        Parameters
        ----------
        n_clusters : INT
            Number of clusters.
        divergence : function
            Pairwise divergence function. The default is euclidean.
        n_iters : INT, optional
            Number of clustering iterations. The default is 1000.
        has_cov : BOOL, optional
            Specifies if the divergence requires a covariance matrix. The default is False.
        initializer : STR, optional
            Specifies if the centroids are initialized at random "rand" or using K-Means++ "kmeans++". The default is "rand".
        init_iters : INT, optional
            Number of iterations for K-Means++. The default is 100.

        Returns
        -------
        None.

        """
        self.__n_clusters = n_clusters
        self.__divergence = divergence
        self.__n_iters = n_iters
        self.__has_cov = has_cov
        self.__initializer = initializer
        self.__init_iters = init_iters

    def fit(self, X):
        """
        Training step.

        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.

        Returns
        -------
        TYPE
            Trained model.

        """
        self.__create_params(X)
        for _ in range(self.__n_iters):
            H = self.__assignments(X)
            self.__reestimate(X, H)
        return self

    def __create_params(self, X):
        if self.__initializer=="rand":
            self.params = self.__init_params(X)
        else:
            self.params = self.__kmeanspp(X)
        if self.__has_cov:
            self.cov = self.__init_cov(X)

    def __init_params(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx[:self.__n_clusters]]
    
    def __kmeanspp(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        selected = idx[:self.__n_clusters]
        init_vals = X[idx[:self.__n_clusters]]

        for i in range(self.__init_iters):
            clus_sim = euclidean(init_vals, init_vals)
            np.fill_diagonal(clus_sim, np.inf)

            candidate = X[np.random.randint(X.shape[0])].reshape(1, -1)
            candidate_sims = euclidean(candidate, init_vals).flatten()
            closest_sim = candidate_sims.min()
            closest = candidate_sims.argmin()
            if closest_sim>clus_sim.min():
                replace_candidates_idx = np.array(np.unravel_index(clus_sim.argmin(), clus_sim.shape))
                replace_candidates = init_vals[replace_candidates_idx, :]

                closest_sim = euclidean(candidate, replace_candidates).flatten()
                replace = np.argmin(closest_sim)
                init_vals[replace_candidates_idx[replace]] = candidate
            else:
                candidate_sims[candidate_sims.argmin()] = np.inf
                second_closest = candidate_sims.argmin()
                if candidate_sims[second_closest] > clus_sim[closest].min():
                    init_vals[closest] = candidate
        return init_vals


    def __init_cov(self, X):
        dists = euclidean(X, self.params)
        H = np.argmin(dists, axis=1)
        covs = []
        for k in range(self.__n_clusters):
            covs.append(np.expand_dims(np.cov(X[H==k].T), axis=0))
        return np.concatenate(covs, axis=0)

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
        """
        Prediction step.

        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.

        Returns
        -------
        y: Array
            Assigned cluster for each data point (n, )

        """
        return self.__assignments(X)
