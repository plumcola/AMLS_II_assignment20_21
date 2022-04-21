# -*- coding: utf-8 -*-
"""Arabic.ipynb
"""

import os
import re
import random
import nltk
import itertools
import string 
import pandas as pd
import numpy as np
nltk.download('stopwords')
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.metrics import precision_score, recall_score,f1_score
from keras.preprocessing import text
import warnings
warnings.filterwarnings("ignore")

# mount your google drive to load the dataset uploaded on it 

from google.colab import drive
drive.mount('/content/drive')

class A_Arabic:
  def process_tweets(self, text, stemmer):
    #Replace @username with empty string
    text = re.sub(r'@[^\s]+', ' ', text)
    text = re.sub(r'_', ' ',  text)
    text = re.sub(r'\n', ' ',  text)
    text = re.sub(r'[a-z,A-Z]', '',  text)
    text = re.sub(r'\d', '',  text)

    #Convert www.* or https?://* to " "
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    #Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)

    # remove punctunation
    punctuations_list = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
    text = text.translate(str.maketrans('', '', punctuations_list))

    # normalize arabic words
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    # remove repeating characters
    text = re.sub(r'(.)\1+', r'\1', text)

    #stemming
    text = stemmer.stem(text)

    return text

  def preprocess(self, df):
    # get data
    df = pd.read_table(df , usecols=[1,2], encoding='utf-8', names=['sentiment', 'tweet'])
    # process data using ISRIStemmer (stemmer for arabic)
    stemmer = ISRIStemmer()
    df["tweet"] = df['tweet'].apply(lambda x: self.process_tweets(x, stemmer))
    df["sentiment"].replace({"negative":0,"positive":1,"neutral":2}, inplace=True)
    # create a keras tokenizar
    tokenizer =text.Tokenizer(num_words=20000)
    # fit the tokenaizar to the content columns
    tokenizer.fit_on_texts(df['tweet'])
    # convert sentences to tokens and create training data (X)
    X = tokenizer.texts_to_sequences(df['tweet'])
    # ensure that all sequences in a list have the same length by padding 0 in the beginning of each sequence until each sequence has the same length
    X = preprocessing.sequence.pad_sequences(X, maxlen=200, padding='post', truncating='post')
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
    return  X_train,   X_test, y_train,  y_test

  def build_model(self):
    
    model = models.Sequential()
    # add an embedding layer that takes the tokenized data
    model.add(layers.Embedding(20000, 200))
    # add SpatialDropout1D layer
    model.add(layers.SpatialDropout1D(0.2))
    # add Bi-Directional LSTM Layer
    model.add(layers.Bidirectional(layers.LSTM(32)))
    # add dropout layer
    model.add(layers.Dropout(0.1))
    # add flatten layer
    model.add(layers.Flatten())
    # add 2 dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    
    
    return model


  def Train_Model(self, model, train_X, train_Y):
      
    model.compile(optimizer= optimizers.Adam(learning_rate= 1e-4), loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(np.array(train_X), np.array(train_Y), epochs= 10, validation_split= 0.2, shuffle=True)
    train_loss, train_acc = model.evaluate(np.array(train_X), np.array(train_Y), verbose=0)

    return  history , train_acc

  def get_accuracy_metrics(self, X_test, y_test, model):
    pred = model.predict(X_test)
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Test Acc: ", acc)
    print("\nprecision score: ",precision_score(np.argmax(pred, axis = 1), y_test, average='micro'))
    print("\nrecall score: ",recall_score(np.argmax(pred, axis = 1), y_test, average='micro'))
    print("\nF1 Score: ",f1_score(np.argmax(pred, axis = 1), y_test, average='micro'))
    return acc

  def plot_learning_curves(self, history):  
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']

    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

task_A_Arabic = A_Arabic()
df = '/content/drive/MyDrive/AMLS2/SemEval2017-task4-train.subtask-A.arabic.txt'
X_train, X_test, y_train, y_test = task_A_Arabic.preprocess(df)

print(f'Number of  train examples is: {len(X_train)}')
print(f'X_train shape is: {X_train.shape}')
print(f'y_train shape is: {y_train.shape}')
print('---------------------------------------')
print(f'Number of  test examples is: {len(X_test)}')
print(f'X_test shape is: {X_test.shape}')
print(f'y_test shape is: {y_test.shape}')

tf.keras.backend.clear_session()
model = task_A_Arabic.build_model()
model.summary()
history_A, acc_A_train = task_A_Arabic.Train_Model(model, X_train, y_train)

acc_A_test =  task_A_Arabic.get_accuracy_metrics(X_test, np.array(y_test), model)

task_A_Arabic.plot_learning_curves(history_A.history)
