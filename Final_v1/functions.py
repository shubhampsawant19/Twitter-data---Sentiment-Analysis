import re
import string
import numpy as np
import nltk
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import pandas as pd

#Removing Punctuation
def percentage(part,whole):
  return 100 * float(part)/float(whole)

def remove_punct(text):
  text = "".join([char for char in text if char not in string.punctuation])
  text = re.sub('[0–9]+', "", text)
  return text

#Appliyng tokenization
def tokenization(text):
  text = re.split('\W+', text)
  return text

#Removing stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
  text = [word for word in text if word not in stopword]
  return text

#Appliyng Stemmer
ps = nltk.PorterStemmer()
def stemming(text):
  text = [ps.stem(word) for word in text]
  return text

#Cleaning Text
def clean_text(text):
  text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
  text_rc = re.sub('[0-9]+', '', text_lc)
  tokens = re.split('\W+', text_rc)    # tokenization
  text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
  return text

#Function to Create Wordcloud
def create_wordcloud(text):
  mask = np.array(Image.open('/content/ukraine-flag-png-19360.png'))
  stopwords = set(STOPWORDS)
  wc = WordCloud(background_color='black',
  mask = mask,
  max_words=1000,
  stopwords=stopwords,
  repeat=True)
  wc.generate(str(text))
  wc.to_file('wc.png')
  print('Word Cloud Saved Successfully')

def count_values_in_column(data,feature):
  total=data.loc[:,feature].value_counts(dropna=False)
  percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
  return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

