# -*- coding: utf-8 -*-

import numpy as np

from .base import NeighborsBase
from .base import KNeighborsMixin
from .base import UnsupervisedMixin

from ..metrics import pairwise_distances

from ..utils import check_array
from ..utils import _get_n_jobs

__all__ = ["LOF"]



class LOF(NeighborsBase, KNeighborsMixin, UnsupervisedMixin):
    """Unsupervised Outlier Detection.

    Return an anomaly score of each sample: its Local Outlier Factor.
    It measures the local deviation of a given data point with respect to
    its neighbours.
    It is local in that the degree depends on how isolated the object is
    with respect to the surrounding neighborhood.
    More precisely, locality is given by k nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of an object to the local densities of
    its neighbors, one can identify points that have a substantially lower
    density than their neighbors. These are considered to be outliers.


    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p: integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only :meth:`k_neighbors` and :meth:`kneighbors_graph` methods.


    Attributes
    ----------

    """
    def __init__(self, n_neighbors=5, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None,
                 n_jobs=1):
        self._init_params(n_neighbors=n_neighbors,
                          algorithm=algorithm,
                          leaf_size=leaf_size, metric=metric, p=p,
                          metric_params=metric_params, n_jobs=n_jobs)

        self.get_n_jobs_ = _get_n_jobs(self.n_jobs)
 
    def predict(self, X):
        """Predict LOF score of X.

        The (local) outlier factor (LOF) of a instance p captures its supposed
        `degree of abnormality'.
        It is the average of the ratio of the local reachability density of X
        and those of X's self.n_neighbors-NN.

        Parameters
        ----------
        X : array-like, last dimension same as that of fit data, optional
            (default=None)
            The querry sample or samples to compute the LOF wrt to the training
            samples.

        Returns
        -------
        lof_scores : array of shape (n_samples,)
            The LOF score of each input samples. The lower, the more normal.
        """
        X = check_array(X, accept_sparse='csr')

        return self._local_outlier_factor(X)

    def fit_predict(self, X, y=None):
        """Predict LOF score of X.

        The (local) outlier factor (LOF) of a instance p captures its supposed
        `degree of abnormality'.
        It is the average of the ratio of the local reachability density of X
        and those of X's self.n_neighbors-NN.

        Parameters
        ----------
        X : array-like, last dimension same as that of fit data, optional
            (default=None)
            The querry sample or samples to compute the LOF. The query point
            is not considered its own neighbor.

        Returns
        -------
        lof_scores : array of shape (n_samples,)
            The LOF score of each input samples. The lower, the more normal.
        """
        X = check_array(X, accept_sparse='csr')
        self.fit(X)

        return self._local_outlier_factor()


    def decision_function(self, X):
        """Opposite of the LOF score of X (as bigger is better).

        The (local) outlier factor (LOF) of a instance p captures its supposed
        `degree of abnormality'.
        It is the average of the ratio of the local reachability density of p
        and those of p's min_pts-NN.

        Parameters
        ----------

        X : array-like, last dimension same as that of fit data, optional
            (default=None)
            The querry sample or samples to compute the LOF wrt to the training
            samples.

        Returns
        -------
        lof_scores : array of shape (n_samples,)
            The LOF score of each input samples. The lower, the more abnormal.
        """
        return -self.predict(X)

#######################################################################################
    def _k_distance(self, p=None):
        """
        Compute the k_distance and the neighborhood of query samples p wrt
        training samples self._fit_X.
        If p=None, neighbors of each sample self._fit_X are returned.
        In this case, the query point is not considered its own neighbor.
        """
        distances, neighbors_indices = self.kneighbors(
            X=p, n_neighbors=self.n_neighbors)
        k_dist = distances[:, self.n_neighbors-1]

        return k_dist, neighbors_indices

    def _local_reachability_density(self, p):
        """The local reachability density (LRD) of p is the inverse of the
        average reachability distance based on the self.n_neighbors-nearest
        neighbors of instance.

        Parameters
        ----------
        p : array-like of shape (n_samples, n_features)
        The samples to compute the LRD w.r.t. self._fit_X
        If None, compute the LRD of self._fit_X w.r.t. to itself.
        (In this case samples are not considered in their own neighbourhood)

        Returns
        -------
        local reachability density : float
        The LRD of p.
        """

        p_0 = self._fit_X if p is None else p

        neighbors_indices = self.neighbors_indices_fit_X_ if p is None else self._k_distance(p)[1]

        # for efficiency, use squared euclidean distances
        if self.effective_metric_ == 'euclidean':
            dist = pairwise_distances(p_0, self._fit_X, 'euclidean',
                                      n_jobs=self.get_n_jobs_, squared=True)
        else:
            dist = pairwise_distances(
                p_0, self._fit_X, self.effective_metric_, n_jobs=self.get_n_jobs_,
                **self.effective_metric_params_)

        reach_dist_array = np.zeros((p_0.shape[0], self.n_neighbors))

        for j in range(p_0.shape[0]):
            neighbors_number = -1
            for i in neighbors_indices[j, :]:
                neighbors_number += 1
                if self.effective_metric_ == 'euclidean':
                    reach_dist_array[j, neighbors_number] = np.max(
                        [self._k_distance_value_fit_X_[i],  np.sqrt(dist[j, i])])
                else:
                    reach_dist_array[j, neighbors_number] = np.max(
                        [self._k_distance_value_fit_X_[i],  dist[j, i]])

        return self.n_neighbors / np.sum(reach_dist_array, axis=1)

    def _local_outlier_factor(self, p=None):
        """ The (local) outlier factor (LOF) of instance captures its
        supposed `degree of abnormality'.
        It is the average of the ratio of the local reachability density of p
        and those of p's self.n_neighbors-NN.

        Parameters
        ----------
        p : array-like of shape (n_samples, n_features)
        The points to compute the LOF w.r.t. training samples X=self._fit_X.
        Note that samples p are not considered in the neighbourhood of X for
        computing local_reachability_density of samples X.
        If None, compute LOF of self._fit_X w.r.t. to itself. (In this case
        samples are not considered in their own neighbourhood)

        Returns
        -------
        Local Outlier Factor : array-like of shape (n_samples,)
        The LOF of p. The lower, the more normal.

        """
        p_0 = self._fit_X if p is None else p

        self._k_distance_value_fit_X_, self.neighbors_indices_fit_X_ = self._k_distance(p=None)
        # Compute it in fit()?
        # May be optimized (if p is not None) by only computing it for
        # X = neighbors or neighbors of p

        self.neighbors_indices_p_ = self.neighbors_indices_fit_X_ if p is None else self._k_distance(p)[1]

        # Compute the local_reachibility_density of samples p:
        p_lrd = self._local_reachability_density(p)
        lrd_ratios_array = np.zeros((p_0.shape[0], self.n_neighbors))

        # Avoid re-computing p_lrd if p is None:
        lrd = p_lrd if p is None else self._local_reachability_density(p=None)

        for j in range(p_0.shape[0]):
            neighbors_number = -1
            for i in self.neighbors_indices_p_[j, :]:
                neighbors_number += 1
                i_lrd = lrd[i]
                lrd_ratios_array[j, neighbors_number] = i_lrd / p_lrd[j]

        return np.sum(lrd_ratios_array, axis=1) / self.n_neighbors
