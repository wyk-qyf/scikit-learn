# -*- coding: utf-8 -*-

import numpy as np
from .unsupervised import NearestNeighbors
from ..utils import check_array
 

__all__ = ["LOF2"]

def k_distance(k, p, X):
    """ 
    Compute the k_distance and the neighbourhood of p 
    The samples X should not contain p, otherwise p is considered in its own neighbourhood.
    """
    # print '__k_distance querry:__'
    # print 'k=', k
    # print 'p=', p
    # print 'X.shape=', X.shape
    n_samples = X.shape[0]
    k = k if k < n_samples else n_samples
    # print 'k moved to ', k
    if True: #k == n_samples:
        neigh = NearestNeighbors(n_neighbors=k, algorithm='auto')
        neigh.fit(X) 
        distances, neighbours_indices =  neigh.kneighbors(p)
        # print 'neighbours_indices=', neighbours_indices
        neighbours_indices = list(neighbours_indices[0])
        k_dist = distances[0, k-1]
    # else:
    #     # compute one additionnal NN to see if the distance really increases
    #     neigh = NearestNeighbors(n_neighbors=k+1, algorithm='auto')  
    #     neigh.fit(X) 
    #     distances, neighbours_indices =  neigh.kneighbors(p)
    #     neighbours_indices = list(neighbours_indices[0])
    #     k_dist = distances[0, k-1]
    #     if k_dist == distances[0, k]:  # then the number of k-nn is larger than k
    #         print 'LOF2: k_dist=', k_dist
    #         print 'LOF2: distances[0,k]', distances[0,k]
    #         for i in range(n_samples):
    #             if i not in neighbours_indices:
    #                 if np.linalg.norm(X[i].reshape(1,-1) - p) == k_dist:
    #                     neighbours_indices.append(i)
#    print 'LOF2 k_distance returns', k_dist, neighbours_indices
    return k_dist, neighbours_indices



def reachability_distance(k, p, o, X):
    """Compute the reachability distance of p with respect to o.
    """
    # print '____reachability_distance querry with k=', k
    # print 'p=',p
    # print 'o=',o
    # print 'X.shape=',X.shape
    k_distance_value = k_distance(k, o, X)[0]
    return max([k_distance_value,  np.linalg.norm(p - o)])


def local_reachability_density(min_pts, p, X):
    """The local reachability density (LRD) of p is the inverse of the average reachability
    distance based on the min_pts-nearest neighbors of instance.

    Parameters
    ----------
    min_pts : int
    Number of instances to consider for computing LRD value
    
    p : array-like
    The point where to compute the LRD.

    X : array-like of shape (n_samples, n_features)
    The input samples.
    
    Returns
    -------
    local reachability density : float
    The LRD of p. 
    """
    # print '____________local_reach_dens querry with min_pts=', min_pts
    # print 'p=', p
    # print 'X.shape=', X.shape
    (k_distance_value, neighbours_indices) = k_distance(min_pts, p, X)
    nb_neighbours = len(neighbours_indices)
    reach_dist_array = np.zeros(nb_neighbours)

    cpt=-1
    for i in neighbours_indices:
        cpt += 1
        # without function reachability_distance:
        ind_without_i = np.ones(X.shape[0], dtype='bool')
        ind_without_i[i] = False
        k_distance_value = k_distance(min_pts, X[i], X[ind_without_i])[0]
        reach_dist_array[cpt] = max([k_distance_value,  np.linalg.norm(p - X[i])])

#    print ' LOF2: local_reachability_density returns:', nb_neighbours / np.sum(reach_dist_array)
    return nb_neighbours / np.sum(reach_dist_array)


def local_outlier_factor(min_pts, p, X):
    """ The (local) outlier factor (LOF) of instance captures its supposed `degree of abnormality'.
    It is the average of the ratio of the local reachability density of p and those of p's min_pts-NN.

    Parameters
    ----------
    min_pts : int
    Number of neighbours to consider for computing LOF value
    
    p : array-like
    The point to compute the LOF.

    X : array-like of shape (n_samples, n_features)
    The input samples. 
    Sould not contain p, otherwise p is considered in its own neighbourhood.
    
    Returns
    -------
    Local Outlier Factor : float
    The LOF of p. The lower, the more normal.
    
    """
    # print '__________________________local_outlier_factor querry with'
    # print 'min_pts=', min_pts
    # print 'p=', p
    # print 'X.shape=', X.shape
    (k_distance_value, neighbours_indices) = k_distance(min_pts, p, X)
    p_lrd = local_reachability_density(min_pts, p, X) # remove p from X?

    n_neighbours = len(neighbours_indices)  # may be larger than min_pts
    # print 'n_neighbours=', n_neighbours
    lrd_ratios_array = np.zeros(n_neighbours)

    cpt = -1
    for i in neighbours_indices:
        cpt += 1
        ind_without_i = np.ones(X.shape[0], dtype='bool')
        ind_without_i[i] = False
        i_lrd = local_reachability_density(min_pts, X[i].reshape(1,-1), X[ind_without_i])
        lrd_ratios_array[cpt] = i_lrd / p_lrd
#    print 'LOF2: local_outlier_factor return:', np.sum(lrd_ratios_array) / n_neighbours
    return np.sum(lrd_ratios_array) / n_neighbours


class LOF2():
    """Unsupervised Outlier Detection.

    Return an anomaly score of each sample: its Local Outlier Factor.


    Parameters
    ----------

    n_neighbors : int, optional (default=5)
        Number of neighbors to use for computing LOF (default is the value passed to
        the constructor).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    """
    def __init__(self, n_neighbors=5, random_state=None):

        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns self.

        """
        X = check_array(X)
        self.training_samples_ = X
        # print 'in fit class, self.n_neighbors=', self.n_neighbors
        return self

    def predict(self, X=None, n_neighbors=None):
        """Predict LOF score of X.
        The (local) outlier factor (LOF) of a instance p captures its supposed `degree of abnormality'.
        It is the average of the ratio of the local reachability density of p and those of p's min_pts-NN.

        Parameters
        ----------

        X : array-like, last dimension same as that of fit data, optional (default=None)
        The querry sample or samples to compute the LOF wrt to the training samples.
        If not provided, LOF of each training sample are returned. In this case, 
        the query point is not considered its own neighbor.

        n_neighbors : int, optional
        Number of neighbors to use for computing LOF (default is the value passed to
        the constructor).
    
        Returns
        -------
        lof_scores : array of shape (n_samples,)
        The LOF score of each input samples. The lower, the more normal.
        """
        if n_neighbors != None:
            self.n_neighbors = n_neighbors

        if X == None:
            X = self.training_samples_
            n_samples = X.shape[0]
            lof_scores = np.zeros(n_samples)
            for i in range(n_samples):
                ind_without_i = np.ones(n_samples, dtype='bool')
                ind_without_i[i] = False
                lof_scores[i] = local_outlier_factor(min_pts=self.n_neighbors,
                                                     p=X[i].reshape(1,-1),
                                                     X=X[ind_without_i]) 
        else:
            X = check_array(X)
            n_samples = X.shape[0]
            lof_scores = np.array([local_outlier_factor(min_pts=self.n_neighbors, p=X[i].reshape(1,-1), X=self.training_samples_) 
                                   for i in range(n_samples)])
        return lof_scores

    def decision_function(self, X=None, n_neighbors=None):
        """Opposite of the LOF score of X (as bigger is better).
        The (local) outlier factor (LOF) of a instance p captures its supposed `degree of abnormality'.
        It is the average of the ratio of the local reachability density of p and those of p's min_pts-NN.

        Parameters
        ----------

        X : array-like, last dimension same as that of fit data, optional (default=None)
        The querry sample or samples to compute the LOF wrt to the training samples.
        If not provided, LOF of each training sample are returned. In this case, 
        the query point is not considered its own neighbor.
        
        n_neighbors : int, optional
        Number of neighbors to use for computing LOF (default is the value passed to
        the constructor).
    
        Returns
        -------
        lof_scores : array of shape (n_samples,)
        The LOF score of each input samples. The lower, the more normal.
        """
        return -self.predict(X, n_neighbors)
