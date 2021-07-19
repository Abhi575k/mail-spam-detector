import sys
import joblib
import pickle
import numpy as np
import Algorithmia
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_preprocessing():
  with open('./preprocessing/count_vectorizer.pkl', 'rb') as f:
    object = joblib.load(f)
    return object    

def load_model():
  with open('./models/spam-detection-model.pkl', 'rb') as f:
    model = joblib.load(f)
    return model

model = load_model()
vectorizer = load_preprocessing()

#set stopwords
stop_words = stopwords.words('english')

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
  text = re.sub(r"[^A-Za-z0-9]", " ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"I'm", "I am", text)
  text = re.sub(r"ur", " your ", text)
  text = re.sub(r" nd ", " and ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r" tkts ", " tickets ", text)
  text = re.sub(r" c ", " can ", text)
  text = re.sub(r" e g ", " eg ", text)
  text = re.sub(r'http\S+', ' link ', text)
  text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
  text = re.sub(r" u ", " you ", text)
  text = text.lower()

  text = ''.join([c for c in text if c not in punctuation])

  if remove_stop_words:
    text = text.split()
    text = [w for w in text if not w in stop_words]
    text = " ".join(text)

  if lemmatize_words:
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)

  return (text)

def process_input(inpt):
#  message = str(inpt)
  clean_message = text_cleaning(inpt)
  vect_message = vectorizer.transform([clean_message])
#  print(vect_message)
  return vect_message

def apply(inpt):
  message = process_input(inpt)
  prediction = model.predict(message)
  if prediction[0] == 0:
    return "Normal message"
  else:
    return "Spam message"
