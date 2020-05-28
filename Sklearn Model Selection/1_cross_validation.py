import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))

######
# cross_val_score
######


from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print('CV scores')
print(scores) # The five accuracy values

import math
# The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))
# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule 
# http://onlinestatbook.com/2/calculators/normal_dist.html Normal distribution calculator


# By default, the score computed at each CV iteration is the score method of the estimator (accuracy for classification, R^2 for regression).
# It is possible to change this by using the scoring parameter:
scores = cross_val_score(
    clf, X_train, y_train, cv=5, scoring='f1_macro')
# StratifiedKFold is the cross validation iterator used by default for classification problems if y is binary/multiclass.
# Otherwise, KFold 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html?highlight=cross_val_score#sklearn.model_selection.cross_val_score

print(scores)
print("f1_score: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))
# See also scikit-learn.ppt for accuracy, precision, f1, etc.
clf.fit(X_train, y_train) # fit() is needed
print(clf.score(X_test, y_test))


#####
# Cross validate function
#####
# More flexible function

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)
print(sorted(scores.keys())) # Let's see what we get...
# The keys of the scores dictionary: fit_time and score_time are fixed, then you have the actual scores
# It's the fourth key above (test_ or train_ is automatically prefixed)

print("Test recall: {:.2f} (+/- {:.2f})".format(scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
print("Train precision: {:.2f} (+/- {:.2f})".format(scores['train_precision_macro'].mean(), scores['train_precision_macro'].std() * 2))

scores = cross_validate(clf, X_train, y_train,
                        scoring='precision_macro', cv=5,
                        return_estimator=True) # Returns also info on the estimator used in the 5 cases (They are different, because different training. See below)
print(sorted(scores.keys()))

for est in scores['estimator']:
    print(est.score(X_test, y_test)) # fit() is not needed

#################
# Cross-validation iterators
#################

#####
# KFold
#####

import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d", "e", "f"]
kf = KFold(n_splits=3)
print('KFold')
for train, test in kf.split(X):
    print(f"{train}, {test}")

kf = KFold(n_splits=3, shuffle=True)
print('KFold')
for train, test in kf.split(X):
    print(f"{train}, {test}")

#####
# Leave one out
#####

from sklearn.model_selection import LeaveOneOut

X = [1, 2, 3, 4]
loo = LeaveOneOut()
print('LeaveOneOut')
for train, test in loo.split(X):
    print(f"{train}, {test}")

#####
# Leave P out
#####

from sklearn.model_selection import LeavePOut

X = np.ones(4)
lpo = LeavePOut(2)
print('LeavePOut')
for train, test in lpo.split(X):
    print(f"{train}, {test}")

#####
# ShuffleSplit
#####

from sklearn.model_selection import ShuffleSplit
X = np.arange(10)*2
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
print('ShuffleSplit')
for train_index, test_index in ss.split(X):
    print(f"{train_index}, {test_index}")
    # You see, e.g., '8' appears twice, '9' never in the val set

#######
# Cross-validation iterators with stratification based on class labels.
#######

#####
# Stratified kFold
#####

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5)) # 50 samples, 1 feature, binary classification
skf = StratifiedKFold(n_splits=3)
print('StratifiedKFold')
for train, test in skf.split(X, y): # 33 train, 17 test
    print('train -  {}   |   test -  {}'.format(
        np.bincount(y[train]), np.bincount(y[test])))
        # Count number of occurrences of each value in array of non-negative ints.
        # https://numpy.org/doc/stable/reference/generated/numpy.bincount.html?highlight=bincount#numpy.bincount

kf = KFold(n_splits=3)
print('Compared to KFold')
for train, test in kf.split(X, y):
    print('train -  {}   |   test -  {}'.format(
        np.bincount(y[train]), np.bincount(y[test])))


#####
# Group K-Fold
#####

from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
print('GroupKFold')
for train, test in gkf.split(X, y, groups=groups):
    print(f"{train}, {test}")
# Each subject is in a different testing fold, and the same subject is never in both testing and training.
# Notice that the folds do not have exactly the same size due to the imbalance in the data.

#####
# Leave P groups out (leave from the training, as for the sample case)
#####
from sklearn.model_selection import LeavePGroupsOut

X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
print('LeavePGroupsOut')
for train, test in lpgo.split(X, y, groups=groups):
    print(f"{train}, {test}")

from sklearn.model_selection import GroupShuffleSplit

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
print('GroupShuffleSplit') # Shuffled per group at every iteration
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print(f"{train}, {test}") 

#####
#  Cross validation of time series data
#####

#####
# Time Series Split
#####

from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
print('TimeSeriesSplit')
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)

for train, test in tscv.split(X):
    print(f"{train}, {test}")