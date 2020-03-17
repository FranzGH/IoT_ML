import Training_testing_set as tts

#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(tts.data_train, tts.target_train)
# predict the response
pred = neigh.predict(tts.data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(tts.target_test, pred))

from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(neigh, classes=['Won','Loss'])
visualizer.fit(tts.data_train, tts.target_train) # Fit the training data to the visualizer
visualizer.score(tts.data_test, tts.target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data