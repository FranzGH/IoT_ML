# https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python

# What is a Principal Component?
# Principal components are the key to PCA; they represent what's underneath the hood of your data.
# In a layman term, when the data is projected into a lower dimension (assume three dimensions) from a higher space,
# the three dimensions are nothing but the three Principal Components that captures (or holds) most of the variance (information) of your data.

#Principal components have both direction and magnitude.
# The direction represents across which principal axes the data is mostly spread out or has most variance and
# the magnitude signifies the amount of variance that Principal Component captures of the data when projected onto that axis.
# The principal components are a straight line, and the first principal component holds the most variance in the data.
# Each subsequent principal component is orthogonal to the last and has a lesser variance.
# In this way, given a set of x correlated variables over y samples you achieve a set of u uncorrelated principal components 
# over the same y samples.

# The reason you achieve uncorrelated principal components from the original features is that 
# the correlated features contribute to the same principal component, 
# thereby reducing the original data features into uncorrelated principal components; 
# each representing a different set of correlated features with different amounts of variation.

# Each principal component represents a percentage of total variation captured from the data.

######
# Breast Cancer Data Exploration
######

from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
breast_data = breast.data
print(breast_data.shape)
breast_labels = breast.target
print(breast_labels.shape)

import numpy as np
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
print(final_breast_data.shape)

import pandas as pd
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
print(features)

features_labels = np.append(features,'label')
breast_dataset.columns = features_labels #Dataset columns also includes the label
print(breast_dataset.head())

breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)
print(breast_dataset.tail())

#####
# CIFAR - 10 Data Exploration
#####

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('Traning data shape:', x_train.shape) # 5,000 images, 32*32 pixles, 3 colors per pixel
print('Testing data shape:', x_test.shape)
print(y_train.shape,y_test.shape) # The labels

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses) # 2
print('Output classes : ', classes) # [0, ..., 9]

import matplotlib.pyplot as plt
# %matplotlib inline

# The dictionary of our targets
label_dict = {
 0: 'airplane',
 1: 'automobile',
 2: 'bird',
 3: 'cat',
 4: 'deer',
 5: 'dog',
 6: 'frog',
 7: 'horse',
 8: 'ship',
 9: 'truck',
}

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(x_train[0], (32,32,3)) #RGB
plt.imshow(curr_img)
print(plt.title("(Label: " + str(label_dict[y_train[0][0]]) + ")")) # y_train[0] is a 1-element vector

# Display the first image in testing data
plt.subplot(122)
#curr_img = np.reshape(x_test[0],(32,32,3))
#plt.imshow(curr_img)
plt.imshow(x_test[0]) # works as above
print(plt.title("(Label: " + str(label_dict[y_test[0][0]]) + ")"))
plt.show()

# Text(0.5, 1.0, '(Label: frog)')  is the text of the print('Title')
# Text(0.5, 1.0, '(Label: cat)')

########
# Data Visualization using PCA
########

from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
print(x.shape)

print(f'{np.mean(x)}, {np.std(x)}') # all elements
print(f'{np.mean(x, axis = 0)[:10]}, {np.std(x, axis = 0)[:10]}') # per columns
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
print(normalised_breast.tail())

from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
print(principal_breast_Df.tail())
print(print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_)))


plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    idx = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[idx, 'principal component 1']
               , principal_breast_Df.loc[idx, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()

######
# Visualizing the CIFAR - 10 data (the ten animals)
######

print(np.min(x_train),np.max(x_train))
x_train = x_train/255.0
print(np.min(x_train),np.max(x_train))
print(x_train.shape)
x_train_flat = x_train.reshape(-1,3072) # 32*32*3 columns  (50,000 rows)
feat_cols = ['pixel'+str(i) for i in range(x_train_flat.shape[1])] # 3,072 features
df_cifar = pd.DataFrame(x_train_flat,columns=feat_cols)
df_cifar['label'] = y_train
print('Size of the dataframe: {}'.format(df_cifar.shape)) # 50,000 * 3,073
print(df_cifar.head())
pca_cifar = PCA(n_components=2)
principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:,:-1]) # Avoid last column, which is the labels

principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar
             , columns = ['principal component 1', 'principal component 2'])
principal_cifar_Df['y'] = y_train

print(principal_cifar_Df.head())
print('Explained variation per principal component: {}'.format(pca_cifar.explained_variance_ratio_))
import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=principal_cifar_Df,
    legend="full",
    alpha=0.3
)
plt.show()

######
# Speed Up Deep Learning Training using PCA with CIFAR - 10 Dataset (10 animals)
#######

x_test = x_test/255.0
x_test = x_test.reshape(-1,32,32,3)
x_test_flat = x_test.reshape(-1,3072)

pca = PCA(0.9)
pca.fit(x_train_flat)
print(pca.n_components_)
train_img_pca = pca.transform(x_train_flat)
test_img_pca = pca.transform(x_test_flat)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop
# Now, you will convert your training and testing labels to one-hot encoding vector.
y_train = np_utils.to_categorical(y_train) # 1 column, multiclass -> multiple (binary) columns
y_test = np_utils.to_categorical(y_test)

batch_size = 128
num_classes = 10
epochs = 20

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(99,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) # num_classes outputs

print(model.summary())
model.compile(loss='categorical_crossentropy', # Loss function to be minimized
              optimizer=RMSprop(),
              metrics=['accuracy']) # Performance metrics for an end user

history = model.fit(train_img_pca, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(test_img_pca, y_test))

## Original dataset
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3072,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train_flat, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(x_test_flat, y_test))

