# https://scikit-learn.org/stable/tutorial/basic/tutorial.html

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
print(digits.target)

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))
#predict() returns an array, because it could be fed with an array

# Model persistence
from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma = 'scale') # To avoid a future warning
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)


import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))

print(y[0])

from joblib import dump, load
dump(clf, 'filename.joblib') 
clf = load('filename.joblib') 

# Conventions
import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
print(X)
print(X.shape)
print(X.dtype)
X = np.array(X, dtype='float32')
print(X.dtype)

# Reduce dimensionality through Gaussian random projection
# https://en.wikipedia.org/wiki/Random_projection
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new)
print(X_new.shape)
print(X_new.dtype)


from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC(gamma='scale') # To avoid a future warning
clf.fit(iris.data, iris.target)
pr_list = list(clf.predict(iris.data[:3]))
print(pr_list)
clf.fit(iris.data, iris.target_names[iris.target])
pr_list = list(clf.predict(iris.data[:3]))
print(pr_list)

myset = set(iris.target)
print(myset)
print(len(myset))
print(len(iris.target))
print(iris.target_names)
print(len(iris.target_names))

# Updating parameters and refitting teh model
X, y = datasets.load_iris(return_X_y=True)
clf = SVC(gamma='scale')
clf.set_params(kernel='linear').fit(X, y)
print(clf.predict(X[:5]))
clf.set_params(kernel='rbf').fit(X, y)
print(clf.predict(X[:5]))

# Multiclass
from sklearn.multiclass import OneVsRestClassifier

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1], [3,3], [4,2]]
y = [0, 2, 1, 1, 2, 0, 2]

from sklearn.metrics import accuracy_score
classif = OneVsRestClassifier(estimator=SVC(random_state=0, gamma='scale'))
pred = classif.fit(X, y).predict(X)
print(pred)
print(accuracy_score(y, pred, normalize = True))

classif = SVC(random_state=0, gamma='scale')
pred = classif.fit(X, y).predict(X)
print(pred)
print(accuracy_score(y, pred, normalize = True))

from sklearn.svm import LinearSVC
classif = LinearSVC(random_state=0)
pred = classif.fit(X, y).predict(X)
print(pred)
print(accuracy_score(y, pred, normalize = True))

classif = OneVsRestClassifier(estimator=LinearSVC(random_state=0))
pred = classif.fit(X, y).predict(X)
print(pred)
print(accuracy_score(y, pred, normalize = True))