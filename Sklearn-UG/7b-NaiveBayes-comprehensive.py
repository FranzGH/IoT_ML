# https://medium.com/@awantikdas/a-comprehensive-naive-bayes-tutorial-using-scikit-learn-f6b71ae84431

#########
# Gaussian Naive Bayes (also for continuous variables)
#########

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
import seaborn as sns; sns.set(color_codes=True)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#%matplotlib inline

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(df,iris.target)

sco = gnb.score(df,iris.target)
print(sco)

##########
# Multinomial Naive Bayes
##########
review_data = pd.read_csv('Reviews.csv')
print(review_data.head())
review_data = review_data[['Text','Score']]
review_data = review_data[review_data.Score != 3] # Remove value 3, it's an intermediate value, not suited for binariztion
review_data['Sentiment'] = review_data.Score.map(lambda s:0 if s < 3 else 1)
review_data.drop('Score',axis=1,inplace=True)
print(review_data.head())
review_data.Sentiment.value_counts() # histogram
review_data = review_data.sample(100) # Sample 100000 only

##########
# Sentiment analysis example
##########


# Remove punchuations
from nltk.tokenize import RegexpTokenizer
#tokenizer = RegexpTokenizer(r'[A-Za-z]+')
tokenizer = RegexpTokenizer('[A-Za-z]+')
review_data['Text'] = review_data.Text.map(lambda x:tokenizer.tokenize(x))
print(review_data.Text)

# Stemming
from nltk.stem.snowball import SnowballStemmer # pip install snowballstemmer
stemmer = SnowballStemmer("english")
review_data['Text'] = review_data.Text.map(lambda l: [stemmer.stem(word) for word in l])
review_data.Text = review_data.Text.str.join(sep=' ') # str accessor only for pd string values. For each line, it merges all its words in a single line

# Preprocessing

# https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english') # Convert a collection of text documents to a matrix of token counts
review_data_tf = cv.fit_transform(review_data.Text)
# For each entry, for each word in the vocabulary, the number of times it appears
# Sparse vectors, because full of 0

# Splitting data into train_test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(review_data_tf,review_data.Sentiment)

# Create model
print(review_data.Sentiment.value_counts()) # Show what the values are
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(class_prior=[.25,.75])
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
mnb.fit(trainX,trainY)
print(mnb.class_prior)
y_pred = mnb.predict(testX)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_true=testY, y_pred=y_pred))
print(classification_report(testY,y_pred))

# Bernoulli’s Naive Bayes
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0)
plt.scatter(X[:,0],X[:,1],c=Y,s=10, cmap='viridis')
plt.show()

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y)
bnb = BernoulliNB(binarize=0.0) # Threshold for binarizing
mnb = MultinomialNB()
bnb.fit(trainX, trainY)
#mnb.fit(trainX, trainY)
print(bnb.score(testX,testY))
y_pred = bnb.predict(testX)
print(confusion_matrix(y_true=testY, y_pred=y_pred))
print(classification_report(testY,y_pred))

#mnb.score(testX,testY)
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = bnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=Y,s=10)
plt.show()


#######
# Out-of-core training - Partial fit
#######
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[A-Za-z]+')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                               alternate_sign=False)
# We use this class as the vectorizer for the vocabulary

review_data_chunks = pd.read_csv('Reviews.csv', chunksize=20000)
'''
test = pd.read_csv('Reviews.csv').sample(10000)
test = test[['Text','Score']]
test = test[test.Score != 3]
test['Sentiment'] = test.Score.map(lambda s:0 if s < 3 else 1)
test.Text = test.Text.map(lambda x:tokenizer.tokenize(x))
test.Text = test.Text.map(lambda l: [stemmer.stem(word) for word in l])
test.Text = test.Text.str.join(sep=' ')
test_tf = vectorizer.transform(test.Text)
'''
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(class_prior=[.22,.78]) 

for idx,review_data in enumerate(review_data_chunks):
    print ('iter : ',idx)
    review_data = review_data[['Text','Score']]
    review_data = review_data[review_data.Score != 3] # Remove value 3, it's an intermediate value, not suited for binariztion
    review_data['Sentiment'] = review_data.Score.map(lambda s:0 if s < 3 else 1)
    review_data.Text = review_data.Text.map(lambda x:tokenizer.tokenize(x))
    review_data.Text = review_data.Text.map(lambda l: [stemmer.stem(word) for word in l])
    review_data.Text = review_data.Text.str.join(sep=' ')
    text_tf = vectorizer.transform(review_data.Text)
    mnb.partial_fit(text_tf,review_data.Sentiment,classes=[0,1])
    y_pred = mnb.predict(text_tf)
    print (confusion_matrix(y_pred=y_pred, y_true=review_data.Sentiment))
    print(f"Accuracy: {accuracy_score(review_data.Sentiment, y_pred)}")