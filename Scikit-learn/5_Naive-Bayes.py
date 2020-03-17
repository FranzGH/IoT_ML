import Training_testing_set as tts

# import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#create an object of the type GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(tts.data_train, tts.target_train).predict(tts.data_test)
#print(pred.tolist())
#print the accuracy score of the model (normalize => percent)
print("Naive-Bayes accuracy : ",accuracy_score(tts.target_test, pred, normalize = True))

# https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(tts.target_test, pred)
lpp = cf[0,1] + cf[1,1]
lap = cf[1,0] + cf[1,1]
lp = cf[1,1]/lpp
lr = cf[1,1]/lap
lf1 = 2 * (lp*lr) / (lp + lr)
print(f'Loss precision: {lp}, recall: {lr}, f1 {lf1}')
wpp = cf[0,0] + cf[1,0]
wap = cf[0,0] + cf[0,1]
wp = cf[0,0]/wpp
wr = cf[0,0]/wap
wf1 = 2 * (wp*wr) / (wp + wr)
print(f'Win precision: {wp}, recall: {wr}, f1 {wf1}')

from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(gnb, classes=['Won','Loss'])
visualizer.fit(tts.data_train, tts.target_train) # Fit the training data to the visualizer
visualizer.score(tts.data_test, tts.target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
acc = accuracy_score(tts.target_test, pred, normalize = True)
prec = precision_score(tts.target_test, pred)
rec = recall_score(tts.target_test, pred)
f1 = f1_score(tts.target_test, pred)
print(f"Naive-Bayes performance. acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}",)