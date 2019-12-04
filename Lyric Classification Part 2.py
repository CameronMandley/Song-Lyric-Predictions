#!/usr/bin/env python
# coding: utf-8

# In[85]:


from sklearn.naive_bayes import BernoulliNB
import numpy as np
import itertools
import matplotlib.pyplot as plt
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
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('billboard_lyrics_1964-2015.csv', encoding = 'latin-1')
df = df.dropna()
df.head()


# In[5]:


"""
determine top 20 most occuring words in the lyrics and their counts
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


# In[6]:


df['Song']


# In[7]:


"""
Cleans up the words
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


# In[36]:


"""
Splits data into training and testing set
"""
def split_data(data):
    total = len(data)
    training_ratio = 0.75
    x = int(np.round(total*training_ratio, decimals=0, out=None))
    return data[0:x], data[x:]


# In[50]:


"""
Split data and clean words
"""
def preprocessing_step():
    data = df[['Lyrics', 'Year']]
    data['Lyrics'] = data['Lyrics'].apply(clean_text)
    return split_data(data)


# In[71]:


"""
train Bernoulli Naive Bayes classifier on vectorized text
"""
def training_step(data, vectorizer):
    training_text = data['Lyrics']
    training_result = data['Year']
    training_text = vectorizer.fit_transform(training_text)

    return BernoulliNB().fit(training_text, training_result)


# In[86]:


"""
performs steps
"""
training_data, evaluation_data = preprocessing_step()
vectorizer = CountVectorizer()
classifier = training_step(training_data, vectorizer)
result = classifier.predict(vectorizer.transform(evaluation_data['Lyrics']))

y_pred = np.round_(result, decimals=0, out=None)
print(y_pred)
print('accuracy %s' % accuracy_score(y_pred, evaluation_data['Year']))
print(classification_report(evaluation_data['Year'], y_pred))


# In[75]:





# In[ ]:




