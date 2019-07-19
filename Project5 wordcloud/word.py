import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt 

data = pd.read_csv('words.csv')
data = data[['text','sentiment']]
print(data)
train, test = train_test_split(data,test_size = 0.1)
#print(train,test)
train = train[train.sentiment != "Neutral"]
train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
#print(train_pos)
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']
#print(train_neg)
def wordcloud_draw(data, color = 'black'):
 words = ' '.join(data)
 cleaned_word = " ".join([word for word in words.split()
  if 'http' not in word
   and not word.startswith('@')
   and not word.startswith('#')
   and word != 'RT'])
 wordcloud = WordCloud(stopwords=STOPWORDS,
 background_color=color,
 width=2500,
  height=2000).generate(cleaned_word)
 plt.figure("wordcloud",figsize=(13, 13))
 plt.imshow(wordcloud)
 #plt.axis('off')
 plt.show()
wordcloud_draw(train_pos)
wordcloud_draw(train_neg)
