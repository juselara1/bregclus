import numpy as np

def euclidean(X, Y):
    func = lambda x_i: np.sqrt(np.sum(x_i-Y, axis=1)**2)
    return np.array([func(x_i) for x_i in X])

def mahalanobis(X, Y, cov):
    def func(x_i):
        diff = np.expand_dims(x_i-Y, axis=-1)
        return np.squeeze(np.sum((cov@diff)*diff, axis=1))
    return np.array([func(x_i) for x_i in X])

class divergences():
    def __init__(self):
        self.euclidean = euclidean
