import numpy as np
from sklearn import neighbors

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_greater


def test_lof():
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [5, 3], [-4, 2]] 

    # Test LOF
    clf = neighbors.LOF()
    clf.fit(X)
    pred = clf.predict()
    assert_array_equal(clf._fit_X, X)

    # assert detect outliers:
    assert_greater(np.min(pred[-2:]), np.max(pred[:-2]))


def test_lof_performance():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)
    X_train = np.r_[X + 2, X - 2]
    X_train = X[:100]

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X[100:], X_outliers]
    y_test = np.array([0] * 20 + [1] * 20)

    # fit the model
    clf = neighbors.LOF().fit(X_train)

    # predict scores (the lower, the more normal)
    y_pred = clf.predict(X_test)

    # check that roc_auc is good
    assert_greater(roc_auc_score(y_test, y_pred), .99)
