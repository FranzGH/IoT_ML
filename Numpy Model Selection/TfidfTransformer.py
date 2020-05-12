# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)
print('Vectorizaion array') # Count of the vocabulary terms in the array
print(pipe['count'].transform(corpus).toarray())
print('idf') # Frequency of the terms
print(pipe['tfid'].idf_)
print('tfidf') # It's a sparse matrix
tfidf = pipe.transform(corpus)
print(tfidf)
print('tfidf shape')
print(pipe.transform(corpus).shape)