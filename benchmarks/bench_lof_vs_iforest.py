"""
===============================================
LocalOutlierFactor vs IsolationForest benchmark
===============================================

A test of LocalOutlierFactor vs IsolationForest on classical anomaly detection
 datasets.

"""
print(__doc__)

from time import time
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.datasets import one_class_data
from sklearn.utils import shuffle as sh
from scipy.interpolate import interp1d


np.random.seed(1)

# training only on normal data?
novelty_detection = True
nb_exp = 2

# # datasets available:
# datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'internet_ads', 'adult']

datasets = ['shuttle', 'forestcover', 'ionosphere', 'spambase',
            'annthyroid', 'arrhythmia',
            'pendigits', 'pima', 'wilt', 'internet_ads', 'adult']
#datasets=['pima', 'wilt']
# # continuous datasets:
# datasets = ['http', 'smtp', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'adult']

plt.figure(figsize=(25, 17))

AUC_lof = []
AUPR_lof = []
AUC_iforest = []
AUPR_iforest = []
fit_time_lof = []
fit_time_iforest = []
predict_time_lof = []
predict_time_iforest = []

for dat in datasets:
    # loading and vectorization
    X, y = one_class_data(dat, scaling=False, continuous=False)

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    n_axis = 1000
    x_axis = np.linspace(0, 1, n_axis)
    tpr_lof = np.zeros(n_axis)
    tpr_iforest = np.zeros(n_axis)
    precision_lof = np.zeros(n_axis)
    precision_iforest = np.zeros(n_axis)
    fit_time_lof_ = 0
    fit_time_iforest_ = 0
    predict_time_lof_ = 0
    predict_time_iforest_ = 0

    for ne in range(nb_exp):
        print 'exp num:', ne
        X, y = sh(X, y)

        X_train = X[:n_samples_train, :]
        X_test = X[n_samples_train:, :]
        y_train = y[:n_samples_train]
        y_test = y[n_samples_train:]

        if novelty_detection:
            # training only on normal data:
            X_train = X_train[y_train == 0]
            y_train = y_train[y_train == 0]

        print('LocalOutlierFactor processing...')
        lof = LocalOutlierFactor(n_neighbors=20)
        tstart = time()
        lof.fit(X_train)
        fit_time_lof_ += time() - tstart
        tstart = time()

        scoring = -lof.decision_function(X_test)  # the lower,the more normal
        predict_time_lof_ += time() - tstart
        fpr_, tpr_, thresholds_ = roc_curve(y_test, scoring)

        f = interp1d(fpr_, tpr_)
        tpr_lof += f(x_axis)
        tpr_lof[0] = 0.

        precision_, recall_ = precision_recall_curve(y_test, scoring)[:2]

        # cluster: old version of scipy -> interpol1d needs sorted x_input
        arg_sorted = recall_.argsort()
        recall_ = recall_[arg_sorted]
        precision_ = precision_[arg_sorted]

        f = interp1d(recall_, precision_)
        precision_lof += f(x_axis)

        print('IsolationForest processing...')
        iforest = IsolationForest()
        tstart = time()
        iforest.fit(X_train)
        fit_time_iforest_ += time() - tstart
        tstart = time()

        scoring = -iforest.decision_function(X_test)  # low = normal
        predict_time_iforest_ += time() - tstart
        fpr_, tpr_, thresholds_ = roc_curve(y_test, scoring)

        f = interp1d(fpr_, tpr_)
        tpr_iforest += f(x_axis)
        tpr_iforest[0] = 0.

        precision_, recall_ = precision_recall_curve(y_test, scoring)[:2]

        # cluster: old version of scipy -> interpol1d needs sorted x_input
        arg_sorted = recall_.argsort()
        recall_ = recall_[arg_sorted]
        precision_ = precision_[arg_sorted]

        f = interp1d(recall_, precision_)
        precision_iforest += f(x_axis)

    tpr_lof /= float(nb_exp)
    fit_time_lof += [fit_time_lof_ / float(nb_exp)]
    predict_time_lof += [predict_time_lof_ / float(nb_exp)]
    AUC_lof += [auc(x_axis, tpr_lof)]
    precision_lof /= float(nb_exp)
    precision_lof[0] = 1.
    AUPR_lof += [auc(x_axis, precision_lof)]

    tpr_iforest /= float(nb_exp)
    fit_time_iforest += [fit_time_iforest_ / float(nb_exp)]
    predict_time_iforest += [predict_time_iforest_ / float(nb_exp)]
    AUC_iforest += [auc(x_axis, tpr_iforest)]
    precision_iforest /= float(nb_exp)
    precision_iforest[0] = 1.
    AUPR_iforest += [auc(x_axis, precision_iforest)]

plt.subplot(221)
plt.xlim([-0.1, len(datasets)])
plt.ylim([0., 1.])
print range(len(datasets))
plt.bar(range(len(datasets)), AUC_lof, width=0.3,
        color='blue', label='lof')
plt.bar(np.array(range(len(datasets))) + 0.3, AUC_iforest, width=0.3,
        color='red', label='iforest')

# plt.xlabel('False Positive Rate', fontsize=25)
# plt.ylabel('True Positive Rate', fontsize=25)
plt.title('ROC AUC', fontsize=25)
plt.legend(loc="lower right", prop={'size': 12})

plt.suptitle('datasets order: http, smtp, SA, SF, shuttle, forestcover '
             + 'ionosphere, spambase, annthyroid, arrhythmia, pendigits, pima '
             + 'wilt, internet_ads, adult')
plt.subplot(222)
plt.xlim([-0.1, len(datasets)])
plt.ylim([0., 1.])
plt.bar(range(len(datasets)), AUPR_lof, color='blue', width=0.3,
        label='lof')
plt.bar(np.array(range(len(datasets))) + 0.3, AUPR_iforest, color='red',
        width=0.3, label='iforest')
plt.title('PR AUC', fontsize=25)

plt.subplot(223)
plt.xlim([-0.1, len(datasets)])
plt.yscale('log')
plt.bar(range(len(datasets)), fit_time_lof, color='blue', width=0.3,
        label='lof')
plt.bar(np.array(range(len(datasets))) + 0.3, fit_time_iforest, color='red',
        width=0.3, label='iforest')
plt.title('training time', fontsize=25)

plt.subplot(224)
plt.xlim([-0.1, len(datasets)])
plt.yscale('log')
plt.bar(range(len(datasets)), predict_time_lof, color='blue', width=0.3,
        label='lof')
plt.bar(np.array(range(len(datasets))) + 0.3, predict_time_iforest,
        color='red', width=0.3, label='iforest')
plt.title('testing time', fontsize=25)

plt.show()
