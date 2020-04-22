# https://www.datacamp.com/community/tutorials/feature-selection-python

# importance of feature selection:
# It enables the machine learning algorithm to train faster.
# It reduces the complexity of a model and makes it easier to interpret.
# It improves the accuracy of a model if the right subset is chosen.
# It reduces Overfitting.

# difference between dimensionality reduction and feature selection

# Filter methods
# Wrapper methods
# Embedded methods

# Filter methods are generally used as a data preprocessing step.
# The selection of features is independent of any machine learning algorithm.
# Features give rank on the basis of statistical scores
# which tend to determine the features' correlation with the outcome variable.
 
import pandas as pd
import numpy as np 

# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = pd.read_csv(url, names=names)
dataframe = pd.read_csv('indian-diabetes-preprocessed.csv', names=names)

print(dataframe.head())

# Let's convert the DataFrame object to a NumPy array to achieve faster computation
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


#######
# Filter method: Chi-squared
#######

# First, you will implement a Chi-Squared statistical test
# for non-negative features to select 4 of the best features from the dataset
      #####################

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])
# print(features.head()) # Error because np.array, no pd.dataframe

# You can see the scores for each attribute and the 4 attributes chosen (those with the highest scores): plas, test, mass, and age.
# This scores will help you further in determining the best features for training your model.

# No care for the target ML method


#######
# Wrapper method: Recursive Feature Elimination
#######

# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
y_pred = rfe.predict(X)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, Y))

# Let's compare...
model.fit(X, Y)
y_pred = model.predict(X)
print(accuracy_score(y_pred, Y))


#######
# Embedded method: Ridge regularization
#######

# First things first
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X,Y)

# A helper method for pretty-printing the coefficients
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

print ("Ridge model:", pretty_print_coefs(ridge.coef_))


# Try also lasso regularization (L1), while Ridge is L2

# Why do these traditional feature selection methods still hold?
# Yes, this question is obvious. Because there are neural net architectures
# (for example CNNs) which are quite capable of extracting the most significant features
# from data but that too has a limitation. Using a CNN for a regular tabular dataset 
# which does not have specific properties (the properties that a typical image holds
# like transitional properties, edges, positional properties, contours etc.)
# is not the wisest decision to make. Moreover, when you have limited data and limited resources,
# training a CNN on regular tabular datasets might turn into a complete waste.
#So, in situations like that, the methods that you studied will definitely come handy.