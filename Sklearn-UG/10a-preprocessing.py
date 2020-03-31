# https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9

#####
# Missing values
#####

# Do I have missing values? How are they expressed in the data?
# Should I withhold samples with missing values?
# Or should I replace them? If so, which values should they be replaced with?

# If they are completely at random, they don’t give any extra information and can be omitted.
# On the other hand, if they’re not at random, the fact that a value is missing is itself information and
# can be expressed as an extra binary feature.

import numpy as np
import pandas as pd
X = pd.DataFrame(
    np.array([5,7,8, np.NaN, np.NaN, np.NaN, -5,
              0,25,999,1,-1, np.NaN, 0, np.NaN])\
              .reshape((5,3)))
X.columns = ['f1', 'f2', 'f3'] #feature 1, feature 2, feature 3

print(X.head())

# We update our dataset by deleting all the rows (axis=0) with only missing values
X.dropna(axis=0, thresh=1, inplace=True) # Drop rows with less than 1 sample not null
X.reset_index(inplace=True) # To update the index column
X.drop(['index'], axis=1, inplace=True) # To drop the old index

print(X.head())

from sklearn.impute import MissingIndicator
X.replace({999.0 : np.NaN}, inplace=True) # 999 => NaN, as multimple type of missing values are not supported
indicator = MissingIndicator(missing_values=np.NaN)
indicator = indicator.fit_transform(X)
print(indicator)
indicator = pd.DataFrame(indicator, columns=['m1', 'm3'])
print(indicator)

#####
# Inputation
#####
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
Y = imp.fit_transform(X)
# print(Y)
# Note that the values returned are put into an Numpy array and we lose all the meta-information. 

#Pandas alternative
X.fillna(X.mean(), inplace=True)
# print(X)

# Other popular ways to impute missing data are clustering the data with the k-nearest neighbor (KNN) algorithm or
# interpolating the values using a wide range of interpolation methods.
# Both techniques are not implemented in sklearn’s preprocessing library

######
# Categorical features
######

# Munging categorical data is another essential process during data preprocessing.
# Unfortunately, sklearn’s machine learning library does not support handling categorical data.
# Even for tree-based models, it is necessary to convert categorical features to a numerical representation.

X = pd.DataFrame(
    np.array(['M', 'O-', 'medium',
             'M', 'O-', 'high',
              'F', 'O+', 'high',
              'F', 'AB', 'low',
              'F', 'B+', np.NaN])
              .reshape((5,3)))
X.columns = ['sex', 'blood_type', 'edu_level']
print(X)

# Looking at the dataframe, you should notice education level is the only ordinal feature
# (it can be ordered and the distance between the categories is not known). 

'''
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X.edu_level = encoder.fit_transform(X.edu_level.values.reshape(-1, 1)) #-1 mens: compatible with the original shape
print(X)
'''

'''
# our missing value is encoded as a separate class (3.0)
# the order of our data is not respected
encoder = OrdinalEncoder(categories=[['nan', 'low', 'medium', 'high']])
X.edu_level = encoder.fit_transform(X.edu_level.values.reshape(-1, 1))
print(X)
'''

# Pandas
cat = pd.Categorical(X.edu_level, 
                     categories=['low',  
                                 'medium', 'high'], 
                     ordered=True)
#cat.fillna('low') # Missing values automatically addressed
labels, unique = pd.factorize(cat, sort=True)
X.edu_level = labels   

# Nominal features
# The most popular way to encode nominal features is one-hot-encoding.
# Essentially, each categorical feature with n categories is transformed into n binary features.
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(dtype=np.int, sparse=True)
nominals = pd.DataFrame(
    onehot.fit_transform(X[['sex', 'blood_type']])\
    .toarray(),
    columns=['F', 'M', 'AB', 'B+','O+', 'O-'])
nominals['edu_level'] = X.edu_level
print(nominals)


###########
###########
# Numericl features
###########
###########


#######
# Discretization
#######
# Example modified, as it was incorrect
X = nominals
from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=3, encode='ordinal', 
                        strategy='uniform')
Z = disc.fit_transform(X['edu_level'].values.reshape(-1, 1))
print(Z)


#######
# Binarization
#######
# Example modified, as it was incorrect
from sklearn.preprocessing import Binarizer
X = nominals
binarizer = Binarizer(threshold=0, copy=True)
Z = binarizer.fit_transform(X.values.reshape(-1, 1)).reshape(X.shape)
print(Z)


#######
#######
# Feature scaling
#######
#######

# Before applying any scaling transformations it is very important to split your data into a train set and a test set.
# If you start scaling before, your training (and test) data might end up scaled around a mean value
# (see below) that is not actually the mean of the train or test data, and go past the whole reason why you’re scaling in the first place.

# Standardization can drastically improve the performance of models.

X = pd.DataFrame(
    np.array([8,25,-1,10.6667])\
              .reshape((-1,1)))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ss = scaler.fit_transform(X.values.reshape(-1, 1))
print(ss)

# MinMax Scaler
# sensitive to outliers
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-3,3))
ss =scaler.fit_transform(X.values.reshape(-1, 1))
print(ss)
 
# This scaler is meant for data that is already centered at zero or sparse data.
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
ss = scaler.fit_transform(X.values.reshape(-1, 1))
print(ss)

# It removes the median and scales the data according to the quantile range
# Robust vs outlier
from sklearn.preprocessing import RobustScaler
robust = RobustScaler(quantile_range = (0.1,0.9))
robust.fit_transform(X.values.reshape(-1, 1))
print(ss)

# One of the key differences between scaling (e.g. standardizing) and normalizing is that
# normalizing is a row-wise operation, while scaling is a column-wise operation.

# Normalization is the process of scaling individual samples to have unit norm.
# In basic terms you need to normalize data when the algorithm predicts
# based on the weighted relationships formed between data points. (Ridge and Lasso regularized regression)
# Scaling inputs to unit norms is a common operation for text classification or clustering.