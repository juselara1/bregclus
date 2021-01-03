import numpy as np

def euclidean(X, Y):
    func = lambda x_i: np.sqrt(np.sum((x_i-Y)**2, axis=1))
    return np.array([func(x_i) for x_i in X])

def mahalanobis(X, Y, cov):
    def func(x_i):
        diff = np.expand_dims(x_i-Y, axis=-1)
        #cov = np.linalg.inv(cov)
        return np.squeeze(np.sum((np.linalg.pinv(cov)@diff)*diff, axis=1))
    return np.array([func(x_i) for x_i in X])

def squared_manhattan(X, Y):
    func = lambda x_i: np.sum(np.abs(x_i-Y), axis=1)**2
    return np.array([func(x_i) for x_i in X])

