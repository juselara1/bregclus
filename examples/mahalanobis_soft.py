from bregclus.models import BregmanSoft
from bregclus.divergences import mahalanobis
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# Setting random seed for replication
np.random.seed(4)

# helper functions
def decision_region(clf, X):
    # Creating a grid in the data's range
    ran_x = X[:, 0].max()-X[:, 0].min()
    ran_y = X[:, 1].max()-X[:, 1].min()
    x = np.linspace(X[:, 0].min()-ran_x*0.05, X[:,0].max()+ran_x*0.05, 256)
    y = np.linspace(X[:, 1].min()-ran_y*0.05, X[:,1].max()+ran_y*0.05, 256)
    A, B = np.meshgrid(x, y)
    A_flat = A.reshape(-1, 1)
    B_flat = B.reshape(-1, 1)
    X_grid = np.hstack([A_flat, B_flat])

    # Defining a matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # compute predictions
    preds = clf.predict(X_grid)

    # Show the voronoi regions as an image
    ax.imshow(preds.reshape(A.shape), interpolation="bilinear", extent=(x.min(), x.max(), y.min(), y.max()), 
              cmap="rainbow", aspect="auto", origin="lower", alpha=0.5)
    ax.axis("off")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    # Scatter plot of the original points
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4, c="k")

    ax.set_xlim([X_grid[:, 0].min(), X_grid[:, 0].max()])
    ax.set_ylim([X_grid[:, 1].min(), X_grid[:, 1].max()])
    
    return fig, ax

def make_anisotropic():
    X, y = make_blobs(n_samples=1000, cluster_std=0.5, centers=3)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso

if __name__ == '__main__':
    # Creating samples
    X = make_anisotropic()
    # Pretrainer
    pretrain = KMeans(n_clusters=3).fit(X)
    # Model definition
    model = BregmanSoft(n_clusters=3, divergence=mahalanobis, has_cov=True, n_iters=100,
                        initializer="pretrainer", pretrainer=pretrain)
    # Model training
    model.fit(X)
    # Decision region
    fig, ax = decision_region(model, X)
    plt.show()
