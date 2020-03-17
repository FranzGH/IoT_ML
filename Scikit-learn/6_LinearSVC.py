import Training_testing_set as tts

#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0)
# the underlying implementation of LinearSVC is random
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(tts.data_train, tts.target_train).predict(tts.data_test)
#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(tts.target_test, pred, normalize = True))

from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(svc_model, classes=['Won','Loss'])
visualizer.fit(tts.data_train, tts.target_train) # Fit the training data to the visualizer
visualizer.score(tts.data_test, tts.target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data