import Training_testing_set as tts

#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0)
# the underlying implementation of LinearSVC is random
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(tts.data_train, tts.target_train).predict(tts.data_test)
#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(tts.target_test, pred, normalize = True))
print(classification_report(tts.target_test, pred))
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# Report for each class (We usually focus on the Positive class, but may not always be the case)

from sklearn.svm import SVC
svc_model = SVC(C=1, gamma='scale', random_state=0, max_iter=2000)
pred = svc_model.fit(tts.data_train, tts.target_train).predict(tts.data_test)
print("RBF SVC classification report")
print(classification_report(tts.target_test, pred))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_train = scaler.fit_transform(tts.data_train)
data_test = scaler.transform(tts.data_test)
svc_model = LinearSVC(random_state=0)
pred = svc_model.fit(data_train, tts.target_train).predict(data_test)
print("LinerSVC classification report - with transform")
print(classification_report(tts.target_test, pred))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_train = scaler.fit_transform(tts.data_train)
data_test = scaler.transform(tts.data_test)
svc_model = SVC(C=1, gamma='scale', random_state=0, max_iter=8000)
pred = svc_model.fit(data_train, tts.target_train).predict(data_test)
print("RBF SVC classification report - with transform")
print(classification_report(tts.target_test, pred))


''' To be verified
from yellowbrick.classifier import ClassificationReport
# https://pypi.org/project/yellowbrick/

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(svc_model, classes=['Won','Loss'])
visualizer.fit(tts.data_train, tts.target_train) # Fit the training data to the visualizer
visualizer.score(tts.data_test, tts.target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data
'''