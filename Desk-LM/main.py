import argparse

import logger as lg

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset')
parser.add_argument('-p', '--preprocess')
parser.add_argument('-e', '--estimator')
parser.add_argument('-v', '--validation')
parser.add_argument('-o', '--output')
args = parser.parse_args()

import sys
print("Python version")
print (sys.version)
import numpy as np
print("Numpy version")
print(np.__version__) 
import sklearn as skl
print("Sklearn version")
print(skl.__version__) 

sys.path.insert(1, 'config')
import Dataset as cds
try:
    ds = cds.Dataset(args.dataset)
except ValueError as err:
    quit(err)

import Estimator as ce
try:
    esti = ce.Estimator.create(args.estimator, ds)
except ValueError as err:
    quit(err)

import Preprocess as pp
try:
    prep = pp.Preprocess(args.preprocess)
except ValueError as err:
    quit(err)

import Crossvalidation as cv
try:
    cv = cv.Crossvalidation(args.validation)
except ValueError as err:
    quit(err)

# lg.initLogger(args.dataset, args.estimator)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    ds.X, ds.y, test_size=ds.test_size, random_state=0)

from sklearn.pipeline import Pipeline
from sklearn import preprocessing, decomposition

steps = []
if len(prep.scalers) > 0:
    steps.append(('scale', preprocessing.StandardScaler()))
if len(prep.pca_values) > 0:
    steps.append(('reduce_dims', decomposition.PCA()))
steps.append(('clf', esti.estimator))
pipe = Pipeline(steps)

param_grid = {}
if len(prep.scalers) > 0:
    param_grid['scale'] = prep.scalers
if len(prep.pca_values) > 0:
    param_grid['reduce_dims__n_components'] = prep.pca_values
for p in esti.params:
    param_grid['clf__'+p] = esti.params[p]

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv.cv, scoring = cv.scoring)
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
print(grid.best_params_)

esti.output_manager.saveParams(grid.best_estimator_['clf'])

sys.path.insert(1, 'output')
import Preprocessing_OM
if 'scale' in grid.best_estimator_:
    best_scaler = grid.best_estimator_['scale']
else:
    best_scaler = None
if 'reduce_dims' in grid.best_estimator_:
    best_reduce_dims = grid.best_estimator_['reduce_dims']
else:
    best_reduce_dims = None
Preprocessing_OM.savePPParams(best_scaler, best_reduce_dims, esti)

if esti.nick == 'knn':
    import OutputMgr as omgr
    fu.saveTrainingSet(X_train, y_train)

if args.output != None:
    import Output
    try:
        op = Output.Output(args.output)
    except ValueError as err:
        quit(err)

    sys.path.insert(1, 'output')
    import OutputMgr as omgr

    if op.is_dataset_test:
        if op.dataset_test_size == 1:
            omgr.OutputMgr.saveTestingSet(X_test, y_test, esti)
        elif op.dataset_test_size < 1:
            n_tests = int(y_test * op.dataset_test_size.shape[0])
            omgr.OutputMgr.saveTestingSet(X_test[0:n_tests], y_test[0:n_tests], esti, full=False)
        elif op.dataset_test_size != None:
            omgr.OutputMgr.saveTestingSet(X_test[0:n_tests], y_test[0:n_tests], esti, full=False)
    
    if op.export_path != None:
        from distutils.dir_util import copy_tree
        omgr.OutputMgr.cleanSIDirs(f'{op.export_path}/')
        fromDirectory = f"./out/include"
        toDirectory = f"{op.export_path}/ds/include"
        copy_tree(fromDirectory, toDirectory)
        fromDirectory = f"./out/source"
        toDirectory = f"{op.export_path}/ds/source"
        copy_tree(fromDirectory, toDirectory)



print('The end')