# https://towardsdatascience.com/https-medium-com-lorrli-classification-and-regression-analysis-with-decision-trees-c43cdbc58054

import pandas as pd
from sklearn import datasets

boston = datasets.load_boston()            # Load Boston Dataset
df = pd.DataFrame(boston.data[:, 12])      # Create DataFrame using only the LSAT feature
df.columns = ['LSTAT']
df['MEDV'] = boston.target                 # Create new column with the target MEDV
df.head()

from sklearn.tree import DecisionTreeRegressor    # Import decision tree regression model

X = df[['LSTAT']].values                          # Assign matrix X # X taken as a DF, matrix shape:(506, 1), see commented below to take X as a vector shape:(506,1)
y = df['MEDV'].values                             # Assign vector y

sort_idx = X.flatten().argsort()                  # Sort X and y by ascending values of X. Returns the indices that would sort an array
X = X[sort_idx]
y = y[sort_idx]

'''
# X taken as a vector
X = df['LSTAT'].values                          # Assign matrix X
y = df['MEDV'].values                             # Assign vector y
# Without 'values' it would be a series

sort_idx = X.argsort()                  # Sort X and y by ascending values of X. Returns the indices that would sort an array
X = X[sort_idx]
y = y[sort_idx]
X = X.reshape(-1,1)
'''

tree = DecisionTreeRegressor(criterion='mse',     # Initialize and fit regressor
                             max_depth=3)         
tree.fit(X, y)

print(f'train_score = {tree.score(X, y)}')
# https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score 
# r2 score

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
plt.scatter(X, y, c='steelblue',                  # Plot actual target against features
            edgecolor='white', s=70)
plt.plot(X, tree.predict(X),                      # Plot predicted target against features
         color='black', lw=2)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()