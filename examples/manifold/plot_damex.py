"""
==========================================
DBAD example
==========================================

An example using DBAD for anomaly detection.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.manifold import Damex
from sklearn.preprocessing import scale

# xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
x_max = 100
y_max = 100
xx, yy = np.meshgrid(np.linspace(0, x_max, 300), np.linspace(0, y_max, 300))

# Generate train and test data
n_train = 2000
n_test = 100
a = np.random.pareto(1, 2*n_train)
a = a[a < x_max]
a_train = a[:n_train]
a_test = a[n_train: (n_train + n_test)]
b = np.random.pareto(1.5, 2*n_train)
b = b[b < y_max]
b_train = b[:n_train]
b_test = b[n_train: (n_train + n_test)]
X_train = np.c_[a_train, b_train]
X_test = np.c_[a_test, b_test]


# Generate some abnormal novel observations
n_outliers = 30
X_outliers = np.random.uniform(low=0, high=100, size=(n_outliers, 2))

# fit the model
clf = Damex(estimator=None, with_rectangles=True, epsilon=0.05)
clf.fit(X_train)

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = scale(Z)
Z = Z.reshape(xx.shape)
Z = np.exp(10*Z)
#Z = np.exp(np.exp(np.exp(np.exp(Z/1.1))/100)/50)
#Z = np.exp(np.exp(np.exp(np.exp(np.exp(25*Z)/1e24)))) #pour accentuer les variations de la Z quand elle est 
#grande (pour qu'elle tombe dans 2 couleurs different

plt.subplot(121)
axe_color = np.linspace(Z.min(), Z.max(), 1000)
plt.contourf(xx, yy, Z, axe_color, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
#b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
#c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')

plt.title("in the input space", fontsize=20)
plt.axis('tight')
plt.xlim((0, x_max))
plt.ylim((0, y_max))
#plt.legend([b1, b2, c],
#           ["training observations",
#            "new regular observations", "new abnormal observations"],
#           loc="upper right",
#           prop=matplotlib.font_manager.FontProperties(size=11))

plt.suptitle("levels set of DAMEX scoring function", fontsize=25)


################## in the transformed space : #############################
#R = order(X_train)
A = clf.transform(np.array([[100, 100]]))
max_ = np.max(A)
x_max = max_ + 10
y_max = max_ + 10
X_train3 = clf.transform(X_train) #/ max_ * 100
X_test3 = clf.transform(X_test) #/ max_ * 100
X_outliers3 = clf.transform(X_outliers) #/ max_ * 100
xx, yy = np.meshgrid(np.linspace(0, x_max, 300), np.linspace(0, y_max, 300))

clf2 = Damex(estimator=None, with_rectangles=True, with_transform=False, epsilon=0.05)
clf2.fit(X_train3)


# plot the line, the points, and the nearest vectors to the plane
Z = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = scale(Z)
Z = Z.reshape(xx.shape)
Z = np.exp(10*Z)
#Z = np.exp(np.exp(np.exp(np.exp(Z/1.1))/100)/70)
#Z = np.exp(np.exp(np.exp(np.exp(np.exp(25*Z)/1e24)))) #pour accentuer les variations de la Z quand elle est 
#grande (pour qu'elle tombe dans 2 couleurs differentes, et non dans la meme ie blanche)

plt.subplot(122)
axe_color = np.linspace(Z.min(), Z.max(), 100)
plt.contourf(xx, yy, Z, axe_color, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train3[:, 0], X_train3[:, 1], c='white')
#b2 = plt.scatter(X_test3[:, 0], X_test3[:, 1], c='green')
#c = plt.scatter(X_outliers3[:, 0], X_outliers3[:, 1], c='red')

plt.title("in the transformed space", fontsize=20)
plt.axis('tight')
plt.xlim((0, x_max))
plt.ylim((0, y_max))
#plt.legend([b1, b2, c],
#           ["training observations",
#            "new regular observations", "new abnormal observations"],
#           loc="upper right",
#           prop=matplotlib.font_manager.FontProperties(size=11))
           
plt.show()
           
################ in the transformed space (seen in a window): #################################
#R = rank(X_train)
#X_train2 = transform(R, X_train)
#X_test2 = transform(R, X_test)
#X_outliers2 = transform(R, X_outliers)
#
#x_max = 100
#y_max = 100
#
##X_train2 = X_train2[np.array([np.max(X_train2[i, :]) < x_max for i in range(n_train)])]
##X_test2 = X_test2[np.array([np.max(X_test2[i, :]) < x_max for i in range(n_test)])]
##X_outliers2 = X_test2[np.array([np.max(X_outliers2[i, :]) < x_max for i in range(n_outliers)])]
#
#
#
## fit the model
#epsilon=0.1
#k_pow=2/3.
#k = pow(n_train, k_pow)
#print 'n/k', n_train/k
#clf = Dbad(epsilon=epsilon, k_pow=k_pow, with_norm=True, with_transform=False)
#clf.fit(X_train2)
#
#
## plot the line, the points, and the nearest vectors to the plane
#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
##Z = Z - np.min(Z)
#Z = np.exp(np.exp(1.2*Z))#pow(Z, 1./10)
#
#plt.subplot(133)
#axe_color = np.linspace(np.min(np.min(Z)), 
#                        np.exp(np.exp(1.2*clf.decision_function(np.array([[100,100]])))), 20)
#plt.contourf(xx, yy, Z, axe_color, cmap=plt.cm.Blues_r)
#
#b1 = plt.scatter(X_train2[:, 0], X_train2[:, 1], c='white')
#b2 = plt.scatter(X_test2[:, 0], X_test2[:, 1], c='green')
#c = plt.scatter(X_outliers2[:, 0], X_outliers2[:, 1], c='red')
#
#plt.title("in the transformed space (seen in a window)")
#plt.axis('tight')
#plt.xlim((0, x_max))
#plt.ylim((0, y_max))
#plt.legend([b1, b2, c],
#           ["training observations",
#            "new regular observations", "new abnormal observations"],
#           loc="upper right",
#           prop=matplotlib.font_manager.FontProperties(size=11))


############### in the transformed space : #################################
#
#R = rank(X_train)
#X_train = transform(R, X_train)
#X_test = transform(R, X_test)
#X_outliers = transform(R, X_outliers)
#
#a = np.c_[np.linspace(0, x_max, 300), np.linspace(0, y_max, 300)]
#A = transform(R, a)
#xx, yy = np.meshgrid(A[:, 0], A[:, 1])
#x_max = np.max(xx)
#y_max = np.max(yy)
#
#plt.subplot(122)
##axe_color = np.linspace(np.min(np.min(Z)), np.max(np.max(Z)), 1000)
#plt.contourf(xx, yy, Z, axe_color, cmap=plt.cm.Blues_r)
#
#
#b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
#b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
#c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
#
#
#plt.title("in the transformed space")
#plt.axis('tight')
#plt.xlim((0, x_max))
#plt.ylim((0, y_max))
#plt.legend([b1, b2, c],
#           ["training observations",
#            "new regular observations", "new abnormal observations"],
#           loc="upper right",
#           prop=matplotlib.font_manager.FontProperties(size=11))
#           
#plt.show()
