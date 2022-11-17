from bregclus.divergences import _old_squared_manhattan, _old_euclidean, _old_mahalanobis, _euclidean_vectorized, _mahalanobis_vectorized, _squared_manhattan_vectorized
import numpy as np

def test_euclidean_consistency():
    # verifies if _old_euclidean and _euclidean_vectorized produce the same output
    Y = np.random.uniform(size=(5, 1000))
    
    for i in range(1,4):
        X = np.random.uniform(size=(100*10**i, 1000))
        old_r = _old_euclidean(X, Y)
        vec_r = _euclidean_vectorized(X,Y)
        
        assert (old_r==vec_r).all()
        
        
def test_mahalanobis_consistency():
    # verifies if _old_mahalanobis and _mahalanobis_vectorized produce the same output
    Y = np.random.uniform(size=(5, 1000))
    cov = np.random.uniform(size=(5, 1000, 1000))
    
    # the _old is to slow to run a with a lot of iterations
    for i in range(1,2):
        X = np.random.uniform(size=(1*10**i, 1000))
        old_r = _old_mahalanobis(X, Y, cov)
        vec_r = _mahalanobis_vectorized(X,Y, cov)
        
        assert (old_r==vec_r).all()
        
def test_squared_manhattan_consistency():
    # verifies if _old_squared_manhattan and _squared_manhattan_vectorized produce the same output
    Y = np.random.uniform(size=(5, 1000))
    
    for i in range(1,4):
        X = np.random.uniform(size=(10*10**i, 1000))
        old_r = _old_squared_manhattan(X, Y)
        vec_r = _squared_manhattan_vectorized(X,Y)
        
        assert (old_r==vec_r).all()
