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
X.reset_index(inplace=True) # To update the index column (remove a hole)
# N.B. When we reset the index, the old index is added as a column, and a new sequential index is used
X.drop(['index'], axis=1, inplace=True) # To drop the old index

print(X.head())

# MissingIndicator - Indicator of missing values, per column
from sklearn.impute import MissingIndicator
X.replace({999.0 : np.NaN}, inplace=True) # 999 => NaN, as multimple type of missing values are not supported
indicator = MissingIndicator(missing_values=np.NaN)
indicator = indicator.fit_transform(X)
#print(indicator)
indicator = pd.DataFrame(indicator, columns=['m1', 'm3']) # The only two columns in which missing values are
print(indicator)


# MissingIndicator - more in depth
import numpy as np
from sklearn.impute import MissingIndicator
X1 = np.array([[np.nan, 1, 3],
               [4, 0, np.nan],
               [8, 1, 0]])
X2 = np.array([[5, 1, np.nan],
               [np.nan, 2, 3],
               [2, 4, 0]])
indicator = MissingIndicator()
indicator.fit(X1) # Creates the possible indicator columns (i.e., not all) 

X2_tr = indicator.transform(X2)
X1_tr = indicator.transform(X1)

print('X2_tr')
print(X2_tr)
print('X1_tr')
print(X1_tr)


#####
# Inputation
#####
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
Y = imp.fit_transform(X)
print(Y)
# Note that the values returned are put into an Numpy array and we lose all the meta-information. 

#Pandas alternative
X.fillna(X.mean(), inplace=True)
print(X.head())

# Other popular ways to impute missing data are clustering the data with the k-nearest neighbor (KNN) algorithm or
# interpolating the values using a wide range of interpolation methods.
# Both techniques are not implemented in sklearn’s preprocessing library

######
# Categorical features
######

# Munging categorical data is another essential process during data preprocessing.
# Unfortunately, sklearn’s machine learning library does not support handling categorical data.
# Even for tree-based models, it is necessary to convert categorical features to a numerical representation.

# Before you start transforming your data, it is important to figure out if the feature you’re working on is ordinal (as opposed to nominal).
# An ordinal feature is best described as a feature with natural, ordered categories and the distances between the categories is not known.
# Once you know what type of categorical data you’re working on, you can pick a suiting transformation tool.
# In sklearn that will be a OrdinalEncoder for ordinal data, and a OneHotEncoder for nominal data.

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
# our missing value is encoded as a separate class (3.0)
# the order of our data is not respected
''' # Use this or the two below
'''
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['nan', 'low', 'medium', 'high']])
X.edu_level = encoder.fit_transform(X.edu_level.values.reshape(-1, 1))
print(X)
'''

# Pandas
cat = pd.Categorical(X.edu_level, 
                     categories=['missing', 'low',  
                                 'medium', 'high'], 
                     ordered=True)
cat = cat.fillna('missing') # Missing values automatically addressed
labels, unique = pd.factorize(cat, sort=True) # factorize(), see below
# Encode the object as an enumerated type or categorical variable.
# This method is useful for obtaining a numeric representation of an array when all that matters is identifying distinct values.
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
print(nominals) # The final result


###########
###########
# Numerical features
###########
###########

 # Just like categorical data can be encoded, numerical features can be ‘decoded’ into categorical features.
 # The two most common ways to do this are discretization and binarization.

#######
# Discretization
#######

# Discretization, also known as quantization or binning,
# divides a continuous feature into a pre-specified number of categories (bins), and thus makes the data discrete.
# One of the main goals of a discretization is to significantly reduce the number of discrete intervals of a continuous attribute. 

# Example modified, as it was incorrect
X = nominals
from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=3, encode='ordinal', 
                        strategy='uniform') # all bins in each feature have identical widths.
                        # default strategy = 'quantile' -> all bins in each feature have the same number of points
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

# Standardization can drastically improve the performance of models.
# Standardization is a transformation that centers the data by removing the mean value of each feature
# and then scale it by dividing (non-constant) features by their standard deviation.
# After standardizing data the mean will be zero and the standard deviation one.
# Standardization can drastically improve the performance of models.
# For instance, many elements used in the objective function of a learning algorithm
# (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models)
# assume that all features are centered around zero and have variance in the same order.
# If a feature has a variance that is orders of magnitude larger than others,
# it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

X = pd.DataFrame(
    np.array([8, 25, -1, 10.637])\
              .reshape((-1,1)))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print('StandardScaler')
ss = scaler.fit_transform(X.values.reshape(-1, 1))
print(ss)

# MinMax Scaler
# sensitive to outliers
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-3,3)) # of the target feature (default is obviously 0-1)
print('MinMaxScaler')
ss =scaler.fit_transform(X.values.reshape(-1, 1))
print(ss)
 
# This scaler is meant for data that is already centered at zero or sparse data.
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
print('MaxAbsScaler')
ss = scaler.fit_transform(X.values.reshape(-1, 1))
print(ss)

# It removes the median and scales the data according to the quantile range
# Robust vs outlier
from sklearn.preprocessing import RobustScaler
robust = RobustScaler(quantile_range = (0.1,0.9))
print('RobustScaler')
robust.fit_transform(X.values.reshape(-1, 1))
print(ss)

# Normalization is the process of scaling individual samples to have unit norm.
# In basic terms you need to normalize data when the algorithm predicts based on the weighted relationships formed between data points.
# Scaling inputs to unit norms is a common operation for text classification or clustering.

# One of the key differences between scaling (e.g. standardizing) and normalizing is that
# normalizing is a row-wise operation, while scaling is a column-wise operation.

# Normalization based on the max of the components
# or on l1 (sum of the abs of the components), or l2 (sqrt of the sum of the squares of the components)