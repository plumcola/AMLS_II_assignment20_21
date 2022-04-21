




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