# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# PCA for data visualization

# PCA for ML speed-up

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

print(df.head())
print(df.describe())

from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values # Access column by name
# Standardizing the features
x = StandardScaler().fit_transform(x)

print(x[:5,])
print(x.min(axis=0))
print(x.max(axis=0))

#####
# PCA Projection to 2D
#####

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf.head())


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):  # a zip object, which is an iterator of tuples
    idx = finalDf['target'] == target
    ax.scatter(finalDf.loc[idx, 'principal component 1']
               , finalDf.loc[idx, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()


# Let's compare
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 8))
ax0.set_xlabel('Principal Component 1', fontsize = 15)
ax0.set_ylabel('Principal Component 2', fontsize = 15)
ax0.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    idx = finalDf['target'] == target
    ax0.scatter(finalDf.loc[idx, 'principal component 1']
               , finalDf.loc[idx, 'principal component 2']
               , c = color
               , s = 50)
ax0.legend(targets)
ax0.grid()

ax1.set_xlabel('Sepal length', fontsize = 15)
ax1.set_ylabel('Sepal width', fontsize = 15)
ax1.set_title('2 component original', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    idx = df['target'] == target
    ax1.scatter(df.loc[idx, 'sepal length']
               , df.loc[idx, 'sepal width']
               , c = color
               , s = 50)
ax1.legend(targets)
ax1.grid()

plt.show()

# Explained Variance
print(pca.explained_variance_ratio_)
# Explained variance (also called explained variation) is used to measure the discrepancy between a model and actual data. 
# For each component, how much of the target variance is explained by the component's variance

#####
# PCA to Speed-up Machine Learning Algorithms
#####

# Download and Load the Data (digits dataset)
from sklearn.datasets import fetch_openml
# mnist = fetch_openml('MNIST original')
# https://www.kaggle.com/avnishnish/mnist-original
import scipy.io
mnist = scipy.io.loadmat('datasets/MNIST-original.mat')


# Split Data into Training and Test Sets
from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
# train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)
mnist_data = mnist["data"].T # data contains 784 rows (the pixels), 7000 columns (the images)
mnist_label = mnist["label"][0] # 7,000 labels
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist_data, mnist_label, test_size=1/7.0, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.95) # choose the minimum number of principal components such that 95% of the variance is retained.

pca.fit(train_img)
print(pca.n_components_) # 330 out of 784

train_img = pca.transform(train_img) # 60,000 * 330
test_img = pca.transform(test_img) # 10,000 * 330

from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)

# Predict for One Observation (image)
res = logisticRegr.predict(test_img[0].reshape(1,-1))
print(res) # 1 digit predicted
# Predict for several Observations (image)
res = logisticRegr.predict(test_img[0:10])
print(res) # 10 digits predicted

print(logisticRegr.score(test_img, test_lbl)) #92%
