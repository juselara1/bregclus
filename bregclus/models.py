import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from bregclus.divergences import euclidean

class BregmanHard(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters, divergence=euclidean, n_iters=1000, has_cov=False,
                 initializer="rand", init_iters=100, pretrainer=None):
        """
        Bregman Hard Clustering Algorithm

        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        divergence : function
            Pairwise divergence function. The default is euclidean.
        n_iters : INT, optional
            Number of clustering iterations. The default is 1000.
        has_cov : BOOL, optional
            Specifies if the divergence requires a covariance matrix. The default is False.
        initializer : STR, optional
            Specifies if the centroids are initialized at random "rand", K-Means++ "kmeans++", or a pretrained K-Means model "pretrained". The default is "rand".
        init_iters : INT, optional
            Number of iterations for K-Means++. The default is 100.
        pretrainer : MODEL, optional
            Pretrained K-Means model to use as pretrainer.

        Returns
        -------
        None.

        """
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.n_iters = n_iters
        self.has_cov = has_cov
        self.initializer = initializer
        self.init_iters = init_iters
        self.pretrainer = pretrainer

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
        self.create_params(X)
        for _ in range(self.n_iters):
            H = self.assignments(X)
            self.reestimate(X, H)
        return self

    def create_params(self, X):
        if self.initializer=="rand":
            self.params = self.init_params(X)
        elif self.initializer=="kmeans++":
            self.params = self.kmeanspp(X)
        else:
            self.params = self.use_pretrainer(X)
        if self.has_cov:
            self.cov = self.init_cov(X)

    def init_params(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx[:self.n_clusters]]
    
    def kmeanspp(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        selected = idx[:self.n_clusters]
        init_vals = X[idx[:self.n_clusters]]

        for i in range(self.init_iters):
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

    def use_pretrainer(self, X):
        self.params = self.pretrainer.cluster_centers_

    def init_cov(self, X):
        dists = euclidean(X, self.params)
        H = np.argmin(dists, axis=1)
        covs = []
        for k in range(self.n_clusters):
            #covs.append(np.expand_dims(np.cov(X[H==k].T), axis=0))
            covs.append(np.expand_dims(np.eye(X.shape[1]), axis=0))
        return np.concatenate(covs, axis=0)

    def assignments(self, X):
        if self.has_cov:
            H = self.divergence(X, self.params, self.cov)
        else:
            H = self.divergence(X, self.params)
        return np.argmin(H, axis=1)

    def reestimate(self, X, H):
        for k in range(self.n_clusters):
            X_k = X[H==k]
            self.params[k] = np.mean(X_k, axis=0)
            if self.has_cov:
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
        return self.assignments(X)
    
class BregmanSoft(BregmanHard):
    """
    Bregman Soft Clustering Algorithm

    Parameters
    ----------
    n_clusters : INT
        Number of clustes.
    divergence : function
        Pairwise divergence function. The default is euclidean.
    n_iters : INT, optional
        Number of clustering iterations. The default is 1000.
    has_cov : BOOL, optional
        Specifies if the divergence requires a covariance matrix. The default is False.
    initializer : STR, optional
        Specifies if the centroids are initialized at random "rand", K-Means++ "kmeans++", or a pretrained K-Means model "pretrained". The default is "rand".
    init_iters : INT, optional
        Number of iterations for K-Means++. The default is 100.
    pretrainer : MODEL, optional
        Pretrained K-Means model to use as pretrainer.

    Returns
    -------
    None.

    """
        
    def __init__(self, *args, **kwargs):
        super(BregmanSoft, self).__init__(*args, **kwargs)

    def init_mixing(self):
        logits = np.ones((self.n_clusters, ))
        return logits/logits.sum()

    def create_params(self, X):
        self.mixing = self.init_mixing()
        if self.initializer=="rand":
            self.params = self.init_params(X)
        else:
            self.params = self.kmeanspp(X)
        if self.has_cov:
            self.cov = self.init_cov(X)

    def assignments(self, X):
        if self.has_cov:
            H = self.divergence(X, self.params, self.cov)
        else:
            H = self.divergence(X, self.params)
        P = np.exp(-H)*self.mixing
        return P/P.sum(axis=1).reshape(-1, 1)

    def reestimate(self, X, P):
        self.mixing = np.mean(P, axis=0)
        for k in range(self.n_clusters):
            self.params[k] = (X*P[:, k].reshape(-1, 1)).sum(axis=0)/P[:, k].sum()
            if self.has_cov:
                X_mk = (X-self.params[k])*P[:, k].reshape(-1, 1)
                new_covs = np.einsum("ij,ik->jk", X_mk, X_mk)+np.eye(X.shape[1])*0.01
                self.cov[k] = new_covs/P[:, k].sum()

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
        return np.argmax(self.assignments(X), axis=1)

    def predict_proba(self, X):
        """
        Probabilities for each cluster.

        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.

        Returns
        -------
        Y: Array
            Probability of each cluster for each point (n, k)

        """
        return self.assignments(X)
