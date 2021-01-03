import models, divergences
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
np.random.seed(1)

X = make_blobs(1000, centers=5, cluster_std=0.5)[0]
#model = models.BregmanHard(5, divergences.euclidean, n_iters=5000)
model = models.BregmanHard(5, divergences.mahalanobis, n_iters=5000, has_cov=True)
model.fit(X)

y = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
