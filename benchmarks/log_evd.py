# -*- coding: utf-8 -*-
import numpy as np

# TODO : PB :  alog_evd(Dep=np.array([7]), size=1, d=7) pas vraiment dependantes les 3 premieres...


def PS(alpha):
    if alpha == 1:
        return 1
    else:
        U = np.random.uniform(0, np.pi)
        W = np.random.exponential()
        return np.power(np.sin((1-alpha) * U) / W, (1 - alpha) / alpha) * (
                                    np.sin(alpha*U)/np.power(np.sin(U), 1/alpha))


def log_evd(alpha, d):
    """
    alpha close to 0 -> dependence in extremes
    alpha = 1 -> independence
    """
    S = PS(alpha)
    W = np.random.exponential(size=d)
    return np.array([np.power(S/W[i], alpha) for i in range(d)])


def alog_evd(Alpha=None, Dep=None, d=2, size=1):
    """
    generate a multivariate asymetric logistic distributed vector
    (see stephenson 2003, algo 2.2 without assymetry parameters)

    Parameters
    ----------
    Alpha : numpy array
        The exponent parameters of each 2^d simplex face

        If len(Alpha) = 2^d-1, 'Dep' need not to be precised since it takes the
        natural order defined in base 2 to assign an alpha value to each face
        of the d-dim simplex (2^d-1 faces). However if 'Dep' is precised, this
        order is used to the assignation.

        If len(Alpha) < 2^d-1, take into account the 'Dep' parameter to associate
        each value of Alpha to a simplex face. Alpha is then completed by
        values '1' (independence) corresponding to face not present in 'Dep'

        If Alpha = None, automatically assign 0.1 to faces in 'Dep' parameter,
        and value 1 to the others.

    Dep : numpy array
        dependence parameters, Dep[m] is a number <= 2^d - 1 corresponding to
        the face base2(Dep[m]), whose exponent parameter is Alpha[m], or 0.1 if
        Alpha=None.

    d : integer
        dimension, namely n_features

    size : integer
        number of sample to generate

    Returns
    -------
    X : numpy array with shape d
        random vector whose law follows extreme value distrib of logistic type
    """
    n_sample = size
    D = pow(2, d) - 1

    if Alpha is None:
        if Dep is None:
            Alpha = np.ones(D+1)  # Alpha[i] correspond to face base2[i] -> i=0 useless
        else:
            Alpha = np.ones(D+1) + 1  # face with alpha=2 are not mentionned in Dep and are to not be in the sum_{b in B} in V(x) 
            if Dep is not None:
                for i in Dep:
                    Alpha[i] = 0.1

    X = np.zeros((n_sample, d))
    teta = np.ones((n_sample, d))

    for i in range(n_sample):
        # X[i,:] = log_evd(1, d)

        for m in range(1, D+1):
            M = base2(m, d)
            M_dim = sum(M)
            if Alpha[m] < 2:  # M_dim > 1 and Alpha[m] < 1:  # the case M_dim=1 is in fact treated in the previous step X[i,:] = log_evd(1, d)
                                            # and for Alpha[m]=1, all 1/teta (teta in the paper) have to be =0 
                if M_dim == 1:
                    Alpha[m] = 1
                Z = log_evd(Alpha[m], M_dim)
                cpt = -1
                for j in range(d):
                    if M[j] is True:
                        cpt += 1
                        X[i, j] = max(X[i, j], Z[cpt])
                        teta[i, j] += 1
    return X/teta


def base2str(n, d):
    """Compute the base2 of n (max lenght : d)
    return a str (ex: '00101' if d=5)
    """
    alpha = ''  # [False for a in range(d)]
    cpt = 0
    q = -1
    while q != 0:
        cpt += 1
        q = n // 2
        r = n % 2
        if r == 1:
            alpha += '1'  # [cpt-1] = True
        else:
            alpha += '0'
        n = q
    while len(alpha) < d:
        alpha += '0'
    return alpha


def base2(n, d):
    """Compute the base2 of n (max lenght : d) with True/false instead of 0/1
    """
    alpha = [False for a in range(d)]
    cpt = 0
    q = -1
    while q != 0:
        cpt += 1
        q = n // 2
        r = n % 2
        if r == 1:
            alpha[cpt-1] = True
        n = q
    return alpha


def nombre(alpha, d):
    """Inverse of function base2
    """
    alpha = [int(a) for a in alpha]
    n = 0
    for i in range(d):
        n += np.power(2, i) * alpha[i]
    return n
