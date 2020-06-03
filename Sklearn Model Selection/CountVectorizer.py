# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray()) # 2D, of course

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2)) # bigrms only (default, unigrams only)
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())
print(X2.toarray()) # 2D, of course

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2)) # unigrams or bigrms
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())
print(X2.toarray()) # 2D, of course