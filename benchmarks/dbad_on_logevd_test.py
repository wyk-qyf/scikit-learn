import numpy as np
from log_evd import *
from sklearn.manifold import Damex
import pylab as pl

results = []
precision = 1  # 100
d = 10

size = 50000
with_rectangles = False
k_pow = 1./2
epsilon = 0.1  # 0.2 if with_rectangle = True

np.random.seed(1)
nb_faces = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # np.random.randint(2, 10, nb_exp)
nb_exp = len(nb_faces)
error = np.zeros(nb_exp)
with open('EVAresult' + str(precision) + '_' + str(size) + '_' + 'with_rectangles-' + str(with_rectangles) + '.txt','a') as result:
    result.write('experience with precision = ' + str(precision) +  ' and ' + ' size= ' + str(size) + ':')


for i in range(nb_exp):
    nb_faces_ = nb_faces[i]
    for j in range(precision):
        Dep = np.random.random_integers(1, pow(2, d)-1, nb_faces_)
        Dep = list(Dep)
        X_train = alog_evd(Dep=Dep, size=size, d=d)
        damex = Damex(epsilon=epsilon, k_pow=k_pow, with_rectangles=with_rectangles, pruning_faces_coef=0.1)
        damex.fit(X_train)

        mu = damex.mu
        mu_keys = mu.keys()
        mu_keys.sort()

        for k in range(len(Dep)):
            Dep[k] = base2str(Dep[k], d)

        error[i] += (len(list(set(Dep) - set(mu_keys))) + len(list(set(mu_keys) - set(Dep)))) #/ float(len(Dep))
        print 'set(Dep)', set(Dep)
        print 'set(mu_keys)', set(mu_keys)
        print 'diff:', 'list(set(mu_keys) - set(Dep))', list(set(mu_keys) - set(Dep))

    error[i] /= float(precision)

    with open('EVAresult' + str(precision) + '_' + str(size) + '_' + 'with_rectangles-' + str(with_rectangles) + '.txt','a') as result:
        result.write('\n \n'+ str(i) + ' ######## nb faces:  ' + str(nb_faces[i]) + ' #### averaged error ' + str(error[i]) + '\n ################################################# \n')
