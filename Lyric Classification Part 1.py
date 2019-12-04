#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('billboard_lyrics_1964-2015.csv', encoding = 'latin-1')
df = df.dropna()
df.head()


# In[3]:


"""
determine top 51 most occuring words in the lyrics and their counts
"""
from collections import Counter 

seperator = ', '
words = seperator.join(df['Lyrics'])
  
# split() returns list of all the words in the string 
split_it = words.split() 
  
# Pass the split_it list to instance of Counter class. 
Counter = Counter(split_it) 
  
# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur = Counter.most_common(51) 
  
print(most_occur)


# In[4]:


df['Song']


# In[5]:


def return_words(words_and_occurances):
    words = np.array([tup[0] for tup in words_and_occurances])
    return words


# In[6]:


def return_occurances(words_and_occurances):
    occurances = np.array([tup[1] for tup in words_and_occurances])
    return occurances


# In[7]:


"""
Plot 20 words and their counts for comparison
"""
words = return_words(most_occur)
occurances = return_occurances(most_occur)
plt.figure(figsize=(10,4))
plt.bar(x = words[0:20], height = occurances[0:20]);


# In[8]:


"""
Cleaning up the words
"""
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


# In[9]:


df['Lyrics'] = df['Lyrics'].apply(clean_text)


# In[10]:


print( "Number of words total: ", df['Lyrics'].apply(lambda x: len(x.split(' '))).sum())


# In[11]:


X_0 = df.Lyrics
y_0 = df.Year
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.3, random_state = 42)


# In[32]:


"""
Train a Multinomial Naive Bayes classifier by creating pipeline with a count vectorizer and tfidf transformer 
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train_0, y_train_0)


from sklearn.metrics import classification_report
y_pred_0 = np.round_(nb.predict(X_test_0), decimals=0, out=None)
print('accuracy %s' % accuracy_score(y_pred_0, y_test_0))
print(classification_report(y_test_0, y_pred_0))


# In[33]:


"""
Train a SGD classifier by creating pipeline with a count vectorizer and tfidf transformer 
"""
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train_0, y_train_0)

y_pred_0 = sgd.predict(X_test_0)

print('accuracy %s' % accuracy_score(y_pred_0, y_test_0))
print(classification_report(y_test_0, y_pred_0))


# In[34]:


"""
Train a Logistic Regression by creating pipeline with a count vectorizer and tfidf transformer 
"""
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train_0, y_train_0)

y_pred_0 = logreg.predict(X_test_0)

print('accuracy %s' % accuracy_score(y_pred_0, y_test_0))
print(classification_report(y_test_0, y_pred_0))


# In[ ]:




