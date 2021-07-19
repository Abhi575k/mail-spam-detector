import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

from collectWords import collect_words
from visualize import show_frequent_words
from cleanText import text_cleaning

data = pd.read_csv("spam_ham_dataset.csv")
data_new = data.drop(columns=['Unnamed: 0', 'label'])
data_new = data_new.rename(columns={'label_num':'label'})

data_new["text"] = data_new["text"].apply(text_cleaning)

x = data_new['text']
y = data_new['label']

#ham_words = collect_words(data_new, 1)
#show_frequent_words(ham_words)
#spam_words = collect_words(data_new, 0)
#show_frequent_words(spam_words)

vectorizer = CountVectorizer(lowercase=False)

best = 0.0000
for _ in range(20):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1, shuffle=True)

    vectorizer.fit(x_train)

    x_train_trans = vectorizer.transform(x_train)

    x_test_trans = vectorizer.transform(x_test)

    model = MultinomialNB()

    model.fit(x_train_trans, y_train)

    y_pred = model.predict(x_test_trans)

    acc = accuracy_score(y_test, y_pred)
    print(acc)

    if (acc > best):
        joblib.dump(vectorizer,'../preprocessing/count_vectorizer.pkl')
        joblib.dump(model, '../models/spam-detection-model.pkl')
        best = acc