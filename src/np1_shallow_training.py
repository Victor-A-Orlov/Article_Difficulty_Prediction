# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# with open('article_texts.txt', 'r', encoding='utf-8') as f:
#     texts = f.readlines()
# with open('targets.txt', 'r', encoding='utf-8') as f:
#     targets = [float(i) for i in f.readlines()]
    
# sns.distplot([len(i) for i in texts], bins=1000)

# texts, targets = zip(*((text, target) for text, target in zip(texts, targets) if len(text)>1000))

# X_train, X_test, y_train, y_test = train_test_split(
# ...     texts, targets, test_size=0.2, random_state=42)

# del texts, targets

# with open('X_train.txt', 'w', encoding='utf-8') as f:
#     for text in tqdm(X_train):
#         f.write(text + '\n')
        
# with open('X_test.txt', 'w', encoding='utf-8') as f:
#     for text in tqdm(X_test):
#         f.write(text + '\n')
        
# with open('y_train.txt', 'w', encoding='utf-8') as f:
#     for target in tqdm(y_train):
#         f.write("{:2.1f}".format(target) + '\n')
        
# with open('y_test.txt', 'w', encoding='utf-8') as f:
#     for target in tqdm(y_test):
#         f.write("{:2.1f}".format(target) + '\n')
        

# https://www.kaggle.com/alxmamaev/how-to-easy-preprocess-russian-text
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
import pickle
import time

#Create lemmatizer and stopwords list
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")
tokenizer = nltk.RegexpTokenizer(r"\w+")

def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
          and token != '\n' \
          and token.strip() not in punctuation
          ]
    text = " ".join(tokens)
    return text

def preprosses_corpus(corpus):
    corpus_clean = []
    for text in tqdm(corpus):
        corpus_clean.append(preprocess_text(text))
    return corpus_clean

with open('X_train_clean.txt', 'r', encoding='utf-8') as f:
    X_train_clean = f.readlines()
with open('X_train.txt', 'r', encoding='utf-8') as f:
    X_train = f.readlines()
with open('y_train.txt', 'r', encoding='utf-8') as f:
    y_train = [float(i) for i in f.readlines()]
with open('X_test_clean.txt', 'r', encoding='utf-8') as f:
    X_test_clean = f.readlines()
with open('X_test.txt', 'r', encoding='utf-8') as f:
    X_test = f.readlines()
with open('y_test.txt', 'r', encoding='utf-8') as f:
    y_test = [float(i) for i in f.readlines()]


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features=10000)
X_train_counts = count_vect.fit_transform(X_train_clean)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=True)#.fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)



model = SVR(verbose=1)
# model = RandomForestRegressor(verbose=1)
# model = LinearRegression()
# model = CatBoostRegressor(verbose=1,task_type="GPU",devices='0:1')

# model = ensemble.GradientBoostingRegressor(verbose=1)
# model.fit(X_train_tf[:n_samples], y_train[:n_samples])

v
text_regression = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('model', model)])



start_time = time.time()
text_regression.fit(X_train_clean, y_train)
print((time.time() - start_time))

preds = text_regression.predict(X_test_clean)
mse = mean_squared_error(y_test, preds)
sqrtmse = np.sqrt(mse)
print(sqrtmse)


plt.scatter(y_test, preds, alpha=0.1)
plt.plot([0,10],[0,10], c='r', alpha=0.1)


pickle.dump(text_regression, open('text_regression_10000_idf_on', 'wb'))
text_regression = pickle.load(open('text_regression_10000', 'rb'))
# # pickle.dump(text_regression, open('pipeline', 'wb'))

# plt.figure(figsize=(8,8))
# plt.scatter(y_test, preds, alpha=0.1)
# plt.plot([0,10],[0,10], c='r', alpha=0.2)

# with open('X_train_clean.txt', 'w', encoding='utf-8') as f:
#     for text in tqdm(X_train_clean):
#         f.write(text)
        
# with open('X_test_clean.txt', 'w', encoding='utf-8') as f:
#     for text in tqdm(X_test_clean):
#         f.write(text)
        
        
seen = set()
uniq = [x for x in X_train_clean if x in seen or seen.add(x)]
        
i = np.random.randint(0,len(X_train))
print(X_train_clean[i][:20])
print(X_train[i][:20])