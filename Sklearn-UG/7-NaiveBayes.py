# https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print(weather_encoded)

# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print(f"Temp: {temp_encoded}")
print(f"Play: {label}")

#Combinig weather and temp into single listof tuples
features=zip(weather_encoded,temp_encoded)
print(features)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(list(features),label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print(f"Predicted Value: {predicted}")


#######
# Naive Bayes with Multiple Labels
#######
#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
wine = datasets.load_wine()

# You would have to define feature_names and target_names, as they are not native pandas attributes
# print the names of the 13 features
print(f"Features: {wine.feature_names}")
# print the label type of wine(class_0, class_1, class_2)
print(f"Labels: {wine.target_names}")

# print data(feature)shape
print(wine.data.shape)
# print the wine data features (top 5 records)
print(wine.data[0:5])
# print the wine labels (0:Class_0, 1:class_2, 2:class_2)
print(wine.target)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print(f"\nAccuracy: {metrics.accuracy_score(y_test, y_pred)}")

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cnf_matrix)
print("\nClassification report:")
print(metrics.classification_report(y_test,y_pred))