# Target encoding
# See also: 
# https://medium.com/analytics-vidhya/target-encoding-vs-one-hot-encoding-with-simple-examples-276a7e7b3e64
# Valore medio del target per ogni categoria

import pandas as pd
import numpy as np

X = pd.DataFrame(
    np.array(['tall', 'large', 'light', 1,  
             'tall', 'narrow', 'heavy', 1,
              'medium', 'narrow', 'heavy', 1,
              'tall', 'large', 'heavy', 1,
              'low', 'narrow', 'heavy', 0,
              'low', 'medium', 'medium', 0,
              'medium', 'narrow', 'medium', 0,
              'low', 'medium', 'heavy', 1,
              'low', 'narrow', 'light', 0,
              'low', 'large', 'light', 0,
              'medium', 'medium', 'medium', 0,
              'medium', 'narrow', 'heavy', 1,
              'tall', 'narrow', 'large', 1,
              'low', 'narrow', 'light', 0,
              'tall', 'large', 'light', 1])
              .reshape((-1,4)))
X.columns = ['height', 'width', 'weight', 'score']
X['score']=X['score'].astype(int)
print(X)

X_orig = X.copy()


def code_mean(data, cat_col, target): #real_feature is actually the target
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over target
    """
    data[cat_col] = pd.Categorical(data[cat_col]).codes
    return dict(data.groupby(cat_col)[target].mean())

for col in X:
    target_encoded = code_mean(X, col, X.score.name)
    X[col] = X[col].astype(float)
    for k in target_encoded:
        X.loc[X[col]==k, col]=target_encoded[k]
# Attention! Target encoding should be learned on trai set only

y = X['score']
X.drop('score', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


X = pd.DataFrame()
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(dtype=np.int, sparse=True)
X = pd.DataFrame(
    onehot.fit_transform(X_orig[['height', 'width', 'weight']])\
    .toarray())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

X = pd.DataFrame()
#X.drop(df.iloc[:, :], inplace = True, axis = 1)
cat = pd.Categorical(X_orig['height'], 
                     categories=['low',  
                                 'medium', 'tall'], 
                     ordered=True)
codes, unique = pd.factorize(cat, sort=True) # factorize(), see below
X['height'] = codes

cat = pd.Categorical(X_orig['width'], 
                     categories=['narrow',  
                                 'medium', 'large'], 
                     ordered=True)
codes, unique = pd.factorize(cat, sort=True) # factorize(), see below
X['width'] = codes

cat = pd.Categorical(X_orig['weight'], 
                     categories=['light',  
                                 'medium', 'heavy'], 
                     ordered=True)
codes, unique = pd.factorize(cat, sort=True) # factorize(), see below
X['weight'] = codes

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)