print(__doc__)

import numpy as np

from time import time
import scipy.stats as stats
# from sklearn.utils.fixes import loguniform -> stats.reciprocal


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier

# get some data
X, y = load_digits(return_X_y=True)

# build a classifier
clf = SGDClassifier(loss='hinge', penalty='elasticnet',
                    fit_intercept=True) # This estimator implements regularized linear models with stochastic gradient descent (SGD) 


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i) # rank considering the test_score, Return indices that are non-zero in the flattened version of a
        for candidate in candidates: # There could be ties
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {'average': [True, False], # The hyper-parameter ranges
              'l1_ratio': stats.uniform(0, 1), # l1 - l2 penalty
              #'alpha': loguniform(1e-4, 1e0)}
              'alpha': stats.reciprocal(1e-4, 1e0)}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, # param_distribution vs param_grid
                                   n_iter=n_iter_search, cv=5)
                  # For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
                  # StratifiedKFold is used.
                  # In all other cases, KFold is used.

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_) # Report on the results...

# use a full grid over all parameters
param_grid = {'average': [True, False],
              'l1_ratio': np.linspace(0, 1, num=10),
              'alpha': np.power(10, np.arange(-4, 1, dtype=float))}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_) # Report on the results...