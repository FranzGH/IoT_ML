# https://scikit-learn.org/stable/modules/cross_validation.html

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

X, y = datasets.load_iris(return_X_y=True)
print(X.shape, y.shape)

# Data transformation with held out data
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
print(clf.score(X_test_transformed, y_test))

# A Pipeline makes it easier to compose estimators, providing this behavior under cross-validation:
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1)) # A pipeline is an estimator, with its pre-processing
scores = cross_val_score(clf, X_train, y_train, cv=5)
print('Pipeline score')
print(scores) # The five accuracy values
print("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))
clf.fit(X_train, y_train) # N.B. Fit is necessary! ####
print("Accuracy on the test set: {:.2f}".format(clf.score(X_test, y_test))) # It's a pipe, so it does the transform


# https://stackoverflow.com/questions/49955951/standardscaler-with-make-pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import decomposition 
pipe = Pipeline([
        ('scale', preprocessing.StandardScaler()),
        ('reduce_dims', decomposition.PCA(n_components=4)),
        ('clf', svm.SVC(kernel = 'linear', C = 1))
])

# Use the pipe strightforwardly
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

# You could also apply model selection to the pipe
param_grid = dict(reduce_dims__n_components=[1,2,3],
                  clf__C=np.logspace(-4, 1, 6),
                  clf__kernel=['rbf','linear'])

grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
print(grid.best_estimator_.score(X_test, y_test)) # same result
print(grid.best_params_)


# https://iaml.it/blog/optimizing-sklearn-pipelines
from sklearn import feature_selection 
from sklearn.linear_model import Ridge
n_features_to_test = np.arange(1, 3)
alpha_to_test = 2.0**np.arange(-6, +6)
scalers_to_test = [preprocessing.StandardScaler(), preprocessing.RobustScaler(), preprocessing.QuantileTransformer()]

params = [
        {'scaler': scalers_to_test,
         'reduce_dim': [decomposition.PCA()], # Parameter of the parameter
         'reduce_dim__n_components': n_features_to_test,
         'regressor__alpha': alpha_to_test}, # Parameter of the parameter

        {'scaler': scalers_to_test,
         'reduce_dim': [feature_selection.SelectKBest(feature_selection.f_regression)],
         'reduce_dim__k': n_features_to_test,\
         'regressor__alpha': alpha_to_test}
        ]

pipe = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('reduce_dim', decomposition.PCA()),
        ('regressor', Ridge())
        ])

gridsearch = GridSearchCV(pipe, params, verbose=1).fit(X_train, y_train)
print('Final score is: ', gridsearch.score(X_test, y_test))
print('Same as: ', gridsearch.best_estimator_.score(X_test, y_test))
print(gridsearch.best_params_)