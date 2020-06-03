import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd

from sklearn import preprocessing, decomposition

df_heart = pd.read_csv('../datasets/heart.csv')
print(df_heart.describe())

target_name = df_heart.columns[-1]
features = df_heart.columns[:-1]

X = df_heart[features]
y = df_heart[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

scalers_to_test = [preprocessing.StandardScaler(), preprocessing.RobustScaler(), preprocessing.MinMaxScaler()]


pipe = Pipeline([
        ('scale', preprocessing.StandardScaler()),
        ('reduce_dims', decomposition.PCA(n_components=4)),
        ('clf', svm.SVC(kernel = 'linear', C = 1, gamma='scale'))
])

param_grid = dict(scale=scalers_to_test,
                    reduce_dims__n_components=[2,6,8, 0.9],
                    clf__C=np.logspace(-4, 1, 6),
                    clf__kernel=['rbf','linear'])

grid = GridSearchCV(pipe, param_grid=param_grid, cv=4)
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
print(grid.best_estimator_.score(X_test, y_test)) 

best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))




# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
pipe = Pipeline([
        ('scale', preprocessing.MinMaxScaler()), # in chi2 raise ValueError("Input X must be non-negative.
        ('reduce_dim', decomposition.PCA()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
])


param_grid = [{
                'scale': scalers_to_test,
                'reduce_dim__n_components' : [2,6,8,0.9, 'mle'],
                'clf__n_neighbors' : np.arange(5, 20, 2)
},
{
                'reduce_dim': [SelectKBest(chi2)],
                'reduce_dim__k': [2,8],#'reduce_dim__k': uniform(2,8),
                'clf__n_neighbors' : np.arange(5, 20, 2)
                }
]

grid = GridSearchCV(pipe, param_grid=param_grid, cv=4) 
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test)) # How is it possible that it needs the line below???????
print(grid.best_estimator_.score(X_test, y_test)) 

best_parameters = grid.best_estimator_.get_params()

params = []
for par in param_grid:
    for param_name in sorted(par.keys()):
        if (not param_name in params):
            params.append(param_name)

for param_name in params:
    if (param_name in best_parameters): # It might not be relevant (e.g., reduce_dim__k)
        print("\t%s: %r" % (param_name, best_parameters[param_name]))







    # https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
pipe = Pipeline([
        ('scale', preprocessing.MinMaxScaler()), # in chi2 raise ValueError("Input X must be non-negative.
        ('reduce_dim', decomposition.PCA()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
])


param_grid = { # With a list there seems to be a problem
                'scale': scalers_to_test,
                'reduce_dim__n_components' : [2,6,8,0.9, 'mle'],
                'clf__n_neighbors' : randint(5, 20) #uniform(5, 20), for float
}

grid = RandomizedSearchCV(pipe, param_grid, cv=4, n_iter=40, scoring = 'precision')
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
from sklearn.metrics import precision_score
print(precision_score(y_test, grid.best_estimator_.predict(X_test))) 

best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))