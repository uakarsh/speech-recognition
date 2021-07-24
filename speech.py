# Basic libraries

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
import random
import itertools
from collections import defaultdict

# Preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from itertools import combinations
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from imblearn.under_sampling import NearMiss, RandomUnderSampler
# from imblearn.over_sampling import SMOTE, ADASYN


import warnings
warnings.filterwarnings("ignore")
# Models

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV

import _pickle as cPickle
with open('logistic_regression.pkl', 'rb') as fid:
    model = cPickle.load(fid)

word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = get_average_word2vec(clean_questions, vectors, generate_missing=generate_missing)
    return list(embeddings)


# Text preparation

def text_prepare(text):

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub('', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    words = text.split()
    i = 0
    while i < len(words):
        if words[i] in STOPWORDS:
            words.pop(i)
        else:
            i += 1
    text = ' '.join(map(str, words))# delete stopwords from text
    
    return text

def basic_preprocessing(df):
    
    # df_temp = df.copy(deep = True)
    
    # df_temp = df_temp.rename(index = str, columns = {'transcription': 'text'})
    
    text = text_prepare(df) 
    
    # le = LabelEncoder()
    # le.fit(df_temp['medical_specialty'])
    # df_temp.loc[:, 'class_label'] = le.transform(df_temp['medical_specialty'])
    
    tokenizer = RegexpTokenizer(r'\w+')

    tokens = tokenizer.tokenize(text)
    
    return tokens


def w2v(data):

    embeddings = get_word2vec_embeddings(word2vec, data)
    #list_labels = df_temp["class_label"].tolist()
    
    return embeddings


mappings = ['Cardiovascular / Pulmonary','Consult - History and Phy.','Discharge Summary',
'Gastroenterology','General Medicine','Neurology', 'Obstetrics / Gynecology',
 'Orthopedic', 'Radiology', 'SOAP / Chart / Progress Notes', 'Surgery','Urology']
def prediction(x):
    with open(x) as f:
        lines = f.readlines()
    tokenized = w2v(lines)
    tokenized = np.expand_dims(tokenized,axis = 0)
    return mappings[model.predict(tokenized)[0]]

# file = 'audio/yes.wav'
# sample,samples_rate = librosa.load(file)
# samples = librosa.resample(sample,samples_rate,8000)
# # samples = np.expand_dims(samples,axis=0).reshape(-1,1)
# #print("The shape of the sample is:",samples.shape)
# print("The keyword is:",prediction(file))
#prediction('temp.txt')