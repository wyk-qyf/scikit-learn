
"""
Testing for Isolation Forest algorithm (sklearn.ensemble.iforest).
"""

# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns

from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.ensemble import IsolationForest
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston, load_iris
from sklearn.utils import check_random_state

from scipy.sparse import csc_matrix, csr_matrix

rng = check_random_state(0)

# load the iris dataset
# and randomly permute it
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_iforest():
    """Check Isolation Forest for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid({"n_estimators": [3],
                          "max_samples": [0.5, 1.0],
                          "bootstrap": [True, False]})

    for params in grid:
        IsolationForest(random_state=rng,
                        **params).fit(X_train).predict(X_test)


def test_iforest_sparse():
    """Check IForest for various parameter settings on sparse input."""
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(boston.data[:50],
                                                        boston.target[:50],
                                                        random_state=rng)
    grid = ParameterGrid({"max_samples": [0.5, 1.0],
                          "bootstrap": [True, False]})

    for sparse_format in [csc_matrix, csr_matrix]:
        X_train_sparse = sparse_format(X_train)
        X_test_sparse = sparse_format(X_test)

        for params in grid:
            # Trained on sparse format
            sparse_classifier = IsolationForest(
                random_state=1,
                **params
            ).fit(X_train_sparse)
            sparse_results = sparse_classifier.predict(X_test_sparse)

            # Trained on dense format
            dense_results = IsolationForest(
                random_state=1,
                **params
            ).fit(X_train).predict(X_test)

            assert_array_equal(sparse_results, dense_results)
            assert_array_equal(sparse_results, dense_results)


def test_iforest_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data

    # Test max_samples
    assert_raises(ValueError,
                  IsolationForest(max_samples=-1).fit, X)
    assert_raises(ValueError,
                  IsolationForest(max_samples=0.0).fit, X)
    assert_raises(ValueError,
                  IsolationForest(max_samples=2.0).fit, X)
    assert_warns(UserWarning,
                  IsolationForest(max_samples=1000).fit, X)
    # cannot check for string values


def test_iforest_parallel_regression():
    """Check parallel regression."""
    rng = check_random_state(0)

    X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                        boston.target,
                                                        random_state=rng)

    ensemble = IsolationForest(n_jobs=3,
                               random_state=0).fit(X_train)

    ensemble.set_params(n_jobs=1)
    y1 = ensemble.predict(X_test)
    ensemble.set_params(n_jobs=2)
    y2 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y2)

    ensemble = IsolationForest(n_jobs=1,
                               random_state=0).fit(X_train)

    y3 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y3)


def test_iforest_gridsearch():
    """Check that Isolation Forest can be grid-searched."""
    # Transform iris into a binary classification task
    X, y = iris.data, iris.target
    y[y == 2] = 1

    # Grid search with scoring based on decision_function
    parameters = {'n_estimators': (1, 2, 100)}

    grid_search = GridSearchCV(IsolationForest(random_state=0,
                                               max_samples=0.1),
                               parameters,
                               scoring="roc_auc").fit(X, y)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    assert best_score > 0.7
    assert best_params['n_estimators'] == 100
