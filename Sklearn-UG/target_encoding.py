# Target encoding
# See also: 
# https://medium.com/analytics-vidhya/target-encoding-vs-one-hot-encoding-with-simple-examples-276a7e7b3e64
# Valore medio del target per ogni categoria

import pandas as pd
import numpy as np

X = pd.DataFrame(
    np.array(['tall', 'large', 'light', 1,  
             'tall', 'narrow', 'heavy', 1,
              'low', 'narrow', 'heavy', 1,
              'tall', 'large', 'heavy', 1,
              'low', 'narrow', 'heavy', 0,
              'low', 'narrow', 'light', 0,
              'low', 'large', 'light', 0])
              .reshape((-1,4)))
X.columns = ['height', 'width', 'weight', 'score']
X['score']=X['score'].astype(int)
print(X)


def code_mean(data, cat_col, target): #real_feature is actually the target
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over target
    """

    codes = pd.Categorical(data[cat_col]).codes
    data[cat_col] = codes
    return dict(data.groupby(cat_col)[target].mean())
    #return dict(data.groupby(cat_col)[target].mean())

dicts = []
for col in X:
    target_encoded = code_mean(X, col, X.score.name)
    X[col] = X[col].astype(float)
    for k in target_encoded:
        # X.loc[X[col]==k][col]=target_encoded[k]
        #X[X[col]==k] [col]=target_encoded[k]
        X.loc[X[col]==k, col]=target_encoded[k]
        print(target_encoded[k])
        print(X[col])
    print(X[col])
    #dicts.append(target_encoded)

data = pd.DataFrame(dicts)
data.columns = X.columns
X = data
y = X['score']
X.drop('score', inplace=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)