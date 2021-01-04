# Clustering with Bregman Divergences

Python implementation of *Clustering with Bregman Divergences* as a `scikit-learn` module:

* Banerjee, A., Merugu, S., Dhillon, I. S., & Ghosh, J. (2005). Clustering with Bregman divergences. Journal of machine learning research, 6(Oct), 1705-1749. [pdf](https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf)

## Requirements

---

If you have [anaconda](https://www.anaconda.com/) installed, you can create an environment with the dependences as follows:

```sh
conda env create -f requirements.yml
```

Then, you must activate the environment:

```sh
source activate env
```

or

```sh
conda activate env
```

## Installation 

---

You can install this package via `setuptools`:

```sh
python setup.py install
```

### Usage

---

You can use the models as you usually do in sklearn:

```python
from bregclus.models import BregmanHard
from bregclus.divergences import euclidean
import numpy as np

X = np.random.uniform(size=(100, 2))

model = BregmanHard(n_clusters=5, divergence=euclidean)
model.fit(X)
y_pred = model.predict(X)
```

Feel free to check the example codes: `examples/`

```sh
python euclidean_hard.py
python mahalanobis_hard.py
```
