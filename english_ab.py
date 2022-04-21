# -*- coding: utf-8 -*-
"""English-AB.ipynb
"""

!pip install ekphrasis
!pip install keras-self-attention
!pip install keras-tuner --upgrade

#Invoke the previous function

# mount google drive to load the dataset uploaded on it 
from google.colab import drive
drive.mount('/content/drive')

"""
Step 1 : Import the libraries we’ll need during our model building phase.
"""
#!pip install ekphrasis
#!pip install keras-self-attention
#!pip install keras-tuner --upgrade

import os
import pandas as pd
import numpy as np
import keras_tuner as kt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import preprocessing
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.utils import plot_model

import warnings
warnings.filterwarnings("ignore")

class A:

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


  def Train_Model(self,model, train_X, train_Y):
     
    model.compile(optimizer= optimizers.Adam(learning_rate= 1e-4), loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(np.array(train_X), np.array(train_Y), epochs= 10, validation_split= 0.2, shuffle=True)
    train_loss, train_acc = model.evaluate(np.array(train_X), np.array(train_Y), verbose=0)

    return  history , train_acc

  def process_tweets(self,df):
    # use text processor from ekphrasis (https://github.com/cbaziotis/ekphrasis#readme) to process the tweets 
    text_processor = TextPreProcessor(normalize=['url','email','percent','money','phone','user','time','url','date','number'],
                                      annotate={"hashtag","allcaps","elongated","repeated",'emphasis','censored'},
                                      fix_html=True,segmenter="twitter",corrector="twitter",unpack_hashtags=True,unpack_contractions=True,
                                      spell_correct_elong=False,tokenizer=SocialTokenizer(lowercase=True).tokenize,dicts=[emoticons])
    vocabulary_set = set()
    sentenses = []
    # iterate in df and process each tweet of df and save the processed tweet
    for i in range(len(df)):
        tweet = df["tweet"][i]
        sentenses.append(" ".join(text_processor.pre_process_doc(tweet)))
        vocabulary_set.update(text_processor.pre_process_doc(tweet))
    return vocabulary_set, pd.DataFrame(sentenses, columns=['content'])

  def get_accuracy_metrics_A(self, X_test, y_test, model):
    pred = model.predict(X_test)
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Test Acc: ", acc)
    print("\nprecision score: ",precision_score(np.argmax(pred, axis = 1), y_test, average='micro'))
    print("\nrecall score: ",recall_score(np.argmax(pred, axis = 1), y_test, average='micro'))
    print("\nF1 Score: ",f1_score(np.argmax(pred, axis = 1), y_test, average='micro'))
    return acc

  def preprocess(self, df):
    # read data
    df = pd.read_csv(df, sep='\t', header=None)
    # remove unwanted columns
    if df.shape[1] ==3:
      df = df.drop(columns=[0])
    else:
      df = df.drop(columns=[0,3])
    # get desired columns 
    df.columns = ['label','tweet']
    # process label (convert it from categorical to numerical)
    df['label'].replace({"negative":0,"positive":1,"neutral":2}, inplace=True)
    # create a dataframe contains sentences after being process by the textprocessor
    # create a set contain all unique words
    vocab_set, sentenses_df =  self.process_tweets(df)
    # create a new column called content to contains tweets after processing(sentences)
    df[["content"]]= sentenses_df[["content"]]

    """
    Keras offers an Embedding layer that can be used for neural networks on text data.
    It requires that the input data be integer encoded, so that each word is represented by a unique integer.
    This data preparation step can be performed using the Tokenizer API also provided with Keras.
    """
    # create a keras tokenizar
    tokenizer = preprocessing.text.Tokenizer(num_words=20000)
    # fit the tokenaizar to the content columns
    tokenizer.fit_on_texts(df['content'])
    # convert sentences to tokens and create training data (X)
    X = tokenizer.texts_to_sequences(df['content'])
    # ensure that all sequences in a list have the same length by padding 0 in the beginning of each sequence until each sequence has the same length
    X = preprocessing.sequence.pad_sequences(X, maxlen=200, padding='post', truncating='post')
    
    # get labels from df
    Y = df['label']

    # split the dataset into test and train parts
    X_train , X_test, y_train, y_test = train_test_split(X , Y , test_size = 0.2, random_state = 42)
    return  X_train,  y_train, X_test,  y_test

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

train_A = '/content/drive/MyDrive/AMLS2/SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
task_A = A()
train_X_A, train_Y_A, test_X_A, test_Y_A =  task_A.preprocess(train_A)

print(f'Number of  train examples is: {len(train_X_A)}')
print(f'X_train shape is: {train_X_A.shape}')
print(f'y_train shape is: {train_Y_A.shape}')
print('---------------------------------------')
print(f'Number of  test examples is: {len(test_X_A)}')
print(f'X_test shape is: {test_X_A.shape}')
print(f'y_test shape is: {test_Y_A.shape}')

tf.keras.backend.clear_session()
bi_lstm_model_A = task_A.build_model()
plot_model(bi_lstm_model_A,to_file='model_A_E.png',show_shapes=True, show_layer_names=True)

pip install visualkeras

import visualkeras
from PIL import ImageFont
#legend=True,draw_volume=False, scale_xy=2,scale_z=0.00001
visualkeras.layered_view(bi_lstm_model_A )

tf.keras.backend.clear_session()
bi_lstm_model_A = task_A.build_model()
bi_lstm_model_A.summary()
history_A,acc_A_train = task_A.Train_Model(bi_lstm_model_A ,train_X_A, train_Y_A)

acc_A_test = task_A.get_accuracy_metrics_A(test_X_A, test_Y_A, bi_lstm_model_A)

task_A.plot_learning_curves(history_A.history)

def hyper_tune_optimizer(hp):
  """
  this function is used to build model architecture including the tunning process for the learning rate
  """
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
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  optimizer = hp.Choice('optimizer', values= ['sgd', 'rmsprop', 'adagrad', 'adam'])
  model.compile(optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

tf.keras.backend.clear_session()
# create an instance from the tunner and perform hypertuning
tuner = kt.Hyperband(hyper_tune_optimizer, objective='val_accuracy',max_epochs=10 )

tuner.search(train_X_A, train_Y_A, epochs=10, validation_split=0.2)
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps.get('optimizer')

def hyper_tune_learning_rate(hp):
  """
  this function is used to build model architecture including the tunning process for the learning rate
  """
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
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4])
  model.compile(optimizer= optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

tf.keras.backend.clear_session()
# create an instance from the tunner and perform hypertuning
tuner = kt.Hyperband(hyper_tune_learning_rate, objective='val_accuracy',max_epochs=10, overwrite=True)

tuner.search(train_X_A, train_Y_A, epochs=10, validation_split=0.2)
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps.get('learning_rate')

#B

class B:

  def build_model(self):
    
    model = models.Sequential()
    # add an embedding layer that takes the tokenized data
    model.add(layers.Embedding(20000, 100, input_length= 200))
    # add Bi-Directional LSTM Layer
    model.add(layers.Bidirectional(layers.LSTM(units=200, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    # add self_attention layer
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    # add a flatten layer
    model.add(layers.Flatten())
    # add dense layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
  

  def Train_Model(self, model, train_X, train_Y):
    model.compile(optimizer=optimizers.Adam(1e-4),loss=losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
    history = model.fit(np.array(train_X), np.array(train_Y), epochs= 10, validation_split= 0.2, shuffle=True)
    train_loss, train_acc = model.evaluate(np.array(train_X), np.array(train_Y), verbose=0)
    return history, train_acc

  def get_accuracy_metrics(self, X_test, y_test, model):
    pred = model.predict(X_test)
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Test Acc: ", acc)
    print("precision score: ",precision_score(np.round(pred), y_test, average='micro'))
    print("recall score: ",recall_score(np.round(pred), y_test, average='micro'))
    print("F1 Score: ",f1_score(np.round(pred), y_test, average='micro'))
    return acc

  def process_tweets(self, df):

      # use text processor from ekphrasis (https://github.com/cbaziotis/ekphrasis#readme) to process the tweets 
      text_processor = TextPreProcessor(normalize=['url','email','percent','money','phone','user','time','url','date','number'],
                                        annotate={"hashtag","allcaps","elongated","repeated",'emphasis','censored'},
                                        fix_html=True,segmenter="twitter",corrector="twitter",unpack_hashtags=True,unpack_contractions=True,
                                        spell_correct_elong=False,tokenizer=SocialTokenizer(lowercase=True).tokenize,dicts=[emoticons])
      
      vocabulary_set, sentenses = set(), []
      for i in range(len(df)):
          # get tweet
          tweet = df["tweet"][i]
          # get topic and convert it to lower case
          topic = df["topic"][i].lower()
          # make sure that that topic exists
          if tweet.find(topic) == -1:
              tweet = topic + ' ' + tweet
          # add preprocessed tweets to sentences list      
          sentenses.append(" ".join(text_processor.pre_process_doc(tweet)))
          # add preprocessed tweets to vocabulary set
          vocabulary_set.update(text_processor.pre_process_doc(tweet))
      return vocabulary_set, pd.DataFrame(sentenses, columns=['content'])

  def preprocess(self, df):
      # read data
      df = pd.read_csv(df, sep='\t', header=None)
      # drop unwannted columns
      df = df.drop(columns=[0,4])
      # get desired columns 
      df.columns = ['topic','label','tweet']
      # process label (convert it from categorical to numerical)
      df['label'].replace({"negative":0, "positive":1}, inplace=True)
        
      df = shuffle(df)
      df.reset_index(inplace=True, drop=True)
      # create a dataframe contains sentences after being process by the textprocessor
      # create a set contain all unique words
      vocab_set, sentenses_df = self.process_tweets(df)
      # create a new column called content to contains tweets after processing(sentences)
      df[["content"]]= sentenses_df[["content"]]

      # create a keras tokenizar
      tokenizer = preprocessing.text.Tokenizer(num_words=20000)
      # fit the tokenaizar to the content columns
      tokenizer.fit_on_texts(df['content'])
      # convert sentences to tokens and create training data (X)
      X = tokenizer.texts_to_sequences(df['content'])
      # ensure that all sequences in a list have the same length by padding 0 in the beginning of each sequence until each sequence has the same length
      X = preprocessing.sequence.pad_sequences(X, maxlen=200, padding='post', truncating='post')
      # get labels from df
      Y = df['label']

      # split the dataset into test and train parts
      X_train , X_test, y_train, y_test = train_test_split(X , Y , test_size = 0.2, random_state = 42)
      return  X_train,  y_train, X_test,  y_test


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

train_B = '/content/drive/MyDrive/AMLS2/SemEval2017-task4-dev.subtask-BD.english.INPUT.txt'
task_B = B()
train_X_B, train_Y_B, test_X_B, test_Y_B  = task_B.preprocess(train_B)

print(f'Number of  train examples is: {len(train_X_B)}')
print(f'X_train shape is: {train_X_B.shape}')
print(f'y_train shape is: {train_Y_B.shape}')
print('---------------------------------------')
print(f'Number of  test examples is: {len(test_X_B)}')
print(f'X_test shape is: {test_X_B.shape}')
print(f'y_test shape is: {test_Y_B.shape}')

tf.keras.backend.clear_session()
bi_lstm_model_B = task_B.build_model()
plot_model(bi_lstm_model_B,to_file='model_B_E.png',show_shapes=True, show_layer_names=False)

#tf.keras.backend.clear_session()
bi_lstm_model_B = task_B.build_model()
bi_lstm_model_B.summary()
history_B,acc_B_train =task_B.Train_Model(bi_lstm_model_B ,train_X_B, train_Y_B)

acc_B_test =  task_B.get_accuracy_metrics(test_X_B, test_Y_B, bi_lstm_model_B)

task_B.plot_learning_curves(history_B.history)
