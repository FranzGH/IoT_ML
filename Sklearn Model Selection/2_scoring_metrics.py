# https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

######
# Scoring parameter
######

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
X, y = datasets.load_iris(return_X_y=True)
clf = svm.SVC(random_state=0, gamma='auto')
scores = cross_val_score(clf, X, y, cv=5, scoring='recall_macro')

print(scores)
# The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Recall: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))
# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule


clf = svm.SVC(random_state=0, gamma='auto')
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy') # Accuracy does not need '_macrp'
print(scores)
print("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))

X_bin, y_bin = datasets.make_classification(n_classes=2, random_state=0)
clf = svm.SVC(random_state=0, gamma='auto')
scores = cross_val_score(clf, X_bin, y_bin, cv=5, scoring='recall_macro')
print(scores)
print("Recall: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))

# ValueError: 'wrong_choice' is not a valid scoring value.
# model = svm.SVC(gamma='auto')
# cross_val_score(model, X, y, cv=5, scoring='wrong_choice')

#####
# Defining your scoring strategy from metric functions
#####

from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
                    scoring=ftwo_scorer, cv=5)
grid.fit(X_bin, y_bin)
print("Best parameters set found on development set:")
print(grid.best_params_)
print("Grid scores:")
means = grid.cv_results_['mean_test_score'] # dictionary. One mean_test_score for every parameter value (mean over cv iterations). One cv test ofr every parameter combination
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print(f"{mean:.3f} (+/-{std * 2:.03f}) for {params}")


######
# Completely custom scorer objects
######

import numpy as np
def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff) # log(1+x) # log shifted to the left by 1

# score will negate the return value of my_custom_loss_func,
# which will be np.log(2), 0.693, given the values for X
# and y defined below.
score = make_scorer(my_custom_loss_func, greater_is_better=False)
X = [[1], [1]]
y = [0, 1]
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='most_frequent', random_state=0) # always predicts the most frequent label in the training set.
clf = clf.fit(X, y)
print(my_custom_loss_func(y, clf.predict(X)))

print(score(clf, X, y)) # Here I use the scorer directly, not in CV
# N.B. score automatically negated, as it is a loss

from sklearn.metrics import accuracy_score
print(accuracy_score(clf.predict(X), y)) # Just to show a comparison

#####
# Using multiple metric evaluation
#####

# as an iterable of string metrics
scoring = ['accuracy', 'precision']

# or as a dict mapping the scorer name to the scoring function::
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
scoring = {'accuracy': make_scorer(accuracy_score),
           'prec': 'precision'}


from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
# A sample toy binary classification dataset
X, y = datasets.make_classification(n_classes=2, random_state=0)
svm = LinearSVC(random_state=0)
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}
cv_results = cross_validate(svm.fit(X, y), X, y, cv=5, scoring=scoring) # Before we saw cross_val_score
# fit() returns the classifier itself

# Getting the test set true positive scores
print(cv_results['test_tp'])

# Getting the test set false negative scores
print(cv_results['test_fn'])


#####
# Classifictaion metrics
#####

#####
# Accuracy score
#####

import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)

print('Accuracy score')
print(accuracy_score(y_true, y_pred, normalize=False)) # 2
print(accuracy_score(y_true, y_pred)) # .5
# In the multilabel case with binary label indicators:
print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))) # .5

# The balanced_accuracy_score function computes the balanced accuracy,
# which avoids inflated performance estimates on imbalanced datasets. 

####
# Confusion matrix
####

print('Confusion matrix')
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))

y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
#print(confusion_matrix(y_true, y_pred, normalize='true'))
# normalize does not work

# Binary problems only
y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(tn, fp, fn, tp)

####
# Classification report
####

from sklearn.metrics import classification_report
print('Classification report')
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

####
# Hamming loss
####

from sklearn.metrics import hamming_loss
y_pred = [1, 2, 0, 4, 2, 5]
y_true = [2, 2, 5, 4, 4, 5]
print(hamming_loss(y_true, y_pred)) # .5

# Multilable case. Hamming loss considers each label 
print(hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))) # .75

####
# Binary classification
####
print('Binary classification')
from sklearn import metrics
y_pred = [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
print(metrics.accuracy_score(y_true, y_pred))
print(metrics.precision_score(y_true, y_pred))
print(metrics.recall_score(y_true, y_pred))
print(metrics.f1_score(y_true, y_pred))
#print(metrics.fbeta_score(y_true, y_pred, beta=0.5))
#print(metrics.fbeta_score(y_true, y_pred, beta=1))
#print(metrics.fbeta_score(y_true, y_pred, beta=2))
#print(metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5))

'''
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, threshold = precision_recall_curve(y_true, y_scores)
print(precision)
print(recall)
print(threshold)
print(average_precision_score(y_true, y_scores))
'''

#####
# Multilabel confusion matrix
#####

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
y_true = np.array([[1, 0, 1],
                   [0, 1, 0]])
y_pred = np.array([[1, 0, 0],
                   [0, 1, 1]])
print('Multilabel confusion matrix')
print(multilabel_confusion_matrix(y_true, y_pred)) #N.B. Label by label (3 CMs)

# Or a confusion matrix can be constructed for each sample’s labels
print('Samplewise')
print(multilabel_confusion_matrix(y_true, y_pred, samplewise=True))

# Multiclass input
print('Multiclass input')
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
print(multilabel_confusion_matrix(y_true, y_pred,
                            labels=["ant", "bird", "cat"])) # Similar, but not the same as confusion matrix (see above)
                                                            # One CM per each class


####
# Receiver operating characteristic (ROC)
####

import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
y_true = np.array([1,   1,   2,    2])
scores = np.array([0.1, 0.4, 0.35, 0.8]) # probabilities, or confidence
fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=2)
## If I accept 0 FP out of 2 negatives, 0% FPR -> I get 1 TP out of 2
## If I accept 1 FP, 0.5% FPR -> I get 2 TP

# y_true= np.array ([0,   0,   0,   0,   1,   1,   1,   1])
# scores = np.array([0.7, 0.1, 0.2, 0.5, 0.4, 0.3, 0.6, 0.3])
# fpr, tpr, _ = roc_curve(y_true, scores)
## If I accept 0 FP out of 4 negatives (@0.7), it's FPR = 0 -> I get 0 TP out of 4
## If I accept 1 FP (0.5), it's FPR 0.25 -> I get 1 TP
## If I accept 2 FP (0.2) it's FPR 0.5 -> I get 4 TP

# y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# scores = np.array([0.7, 0.1, 0.2, 0.6, 0.4, 0.3, 0.7, 0.8, 0.5, 0.4])
# fpr, tpr, _ = roc_curve(y_true, scores)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

from sklearn.metrics import explained_variance_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print('explained_variance_score')
print(explained_variance_score(y_true, y_pred))

from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print('mean_absolute_error')
print(mean_absolute_error(y_true, y_pred))

from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print('mean_squared_error')
print(mean_squared_error(y_true, y_pred))

from sklearn.metrics import median_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print('median_absolute_error')
print(median_absolute_error(y_true, y_pred)) # Median is robust to outlier

from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print('r2_score')
print(r2_score(y_true, y_pred))