# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:48:24 2018

@author: kevin
"""
#NLP
# Get the 20 newsgroups Corpus (just the train set for now)
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='train',
                          categories=('rec.autos',
                             'rec.motorcycles',
                             'rec.sport.baseball',
                             'rec.sport.hockey'),
                          remove=('headers', 'footers', 'quotes'))
#print out the target names (y)
print(news.target_names)
#print out the 50th index
print(news.data[50])

#essentially what we want to do it transform raw text into features
#first step is boundary detection
import nltk
nltk.download('punkt')
# Let's play around with the first document for a while
doc = news.data[50]

# Starting with boundary detection,
#We are tokenizing the sentences
sents = nltk.sent_tokenize(doc)

for i, s in enumerate(sents):
    print (30*"-", "sentence %d" % i, 30*"-")
    print (s)
    
#Word Segmentation, Tokenization
sent = sents[4]
print(sent)
# Segment the words in sentence with a "tokenizer"
tokens = nltk.word_tokenize(sent)
tokens

# Normalize the tokens
normalized_tokens = [t.lower() for t in tokens]
print ('\nNormalized tokens:\n', normalized_tokens)

# Build the vocabulary
vocabulary = sorted(set(normalized_tokens))
print ('\nThe vocabulary:\n', vocabulary)

#Stemming
print ([nltk.PorterStemmer().stem(t) for t in tokens])

#For example we can show that the stemmer works:
example_tokens = ['lie', 'lied', 'lay', 'lies', 'lying']
stemmed_tokens = [nltk.PorterStemmer().stem(t) for t in example_tokens]
print(stemmed_tokens)
#Lets look at some statistics for the words
ltokens = [nltk.word_tokenize(doc) for doc in news.data[:500]]
# convert list of list of tokens (ltokens) into a list of tokens
import itertools
tokens_all = list(itertools.chain.from_iterable(ltokens))
# convert list of tokens to nltk text object
x = nltk.Text(t.lower() for t in tokens_all)

print ("The text comprises %d normalized tokens." % len(x))
print ("The first few are", x[:10])


from nltk import FreqDist
fdist = FreqDist(x)
print ('Number of times \"car\" occurs = ', fdist['car'])
print ('Number of times \"Car\" occurs = ', fdist['Car'])
#Why do you think one of them is 0?
#lets find the most common
fdist.most_common(10)
#lets plot the top 30 and see how frequent they are
fdist.plot(30)
#Are these words surprising?
# We are working with car, motorcycle, hockey, baseball, moon, and japan 
#shows the distribution of the words through the text
x.dispersion_plot(['car','motorcycle','goal','puck','base', 'moon', 'japan'])
#concordence
x.concordance('motorcycle',lines=5,width=80)

#Part of speech tagging
tokens = nltk.word_tokenize(u"Dr. J. Gubler studied dengue at the CDC.")
print(tokens)
nltk.download('averaged_perceptron_tagger')
post_tokens = nltk.pos_tag(tokens)
for item in post_tokens:
    print (item)

#we can even look up what they mean
nltk.download('tagsets')
print (nltk.help.upenn_tagset('NNP'))

nltk.download('maxent_ne_chunker')
nltk.download('words')
chunked_sentence = nltk.ne_chunk(post_tokens)
print (chunked_sentence)

#Lets  create a bag of words:
#documents
docs = [u'The patient was seen for bird flu.', 
        u'The patient was seen for chickenpox.', 
        u'The patient was seen for dengue.']
from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                max_features=100,
                                ngram_range=(1, 1),
                                stop_words=None)

tf = tf_vectorizer.fit_transform(docs)
import pandas as pd
pd.DataFrame(tf.todense(),columns=tf_vectorizer.get_feature_names())


#We can also classify our sentences
#We look at a sentence, and predict which category its from
#Get training data!
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='train',
                          categories=('rec.autos',
                             'rec.motorcycles',
                             'rec.sport.baseball',
                             'rec.sport.hockey'),
                          remove=('headers', 'footers', 'quotes'))
#Import Vectorizor
from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=1., min_df=10,
                                max_features=1000,
                                ngram_range=(1, 3),
                                stop_words='english')
tf = tf_vectorizer.fit_transform(news.data)#transform the data
#train on Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(tf, news.target)
#Get test set
from sklearn import metrics
news_test = fetch_20newsgroups(subset='test',
                               categories=('rec.autos',
                                 'rec.motorcycles',
                                 'rec.sport.baseball',
                                 'rec.sport.hockey'),
                               remove=('headers', 'footers', 'quotes'))
tf_test = tf_vectorizer.transform(news_test.data)
#predict
pred = clf.predict(tf_test)
print(metrics.classification_report(news_test.target, pred, target_names=news.target_names))
#simple perceptron
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
y = np.array([[0], [1], [1], [0]])

# create model
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])
# Fit the model
model.fit(X, y, epochs=2000)

#model.predict(np.array([[1,0]])).round()

#plt.scatter(X[:,0], X[:,1], c=y.flatten())


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape).round(),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap = ListedColormap(('red', 'green')))
