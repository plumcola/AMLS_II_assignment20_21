import A.a_A as A_A
import A.a_E as A_E
import B.b as B_E

"""
Step 1 : Import the libraries we’ll need during our model building phase.
"""
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

import re
import random
import nltk
import itertools
import string 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn import preprocessing
from keras.preprocessing import text

# ======================================================================================================================
# hyper parameter tuning

print('Hyper parameter tuning:')

train_A = './Datasets/SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
task_A = A_E.A()
train_X_A, train_Y_A, test_X_A, test_Y_A =  task_A.preprocess(train_A)

# to tune optimizer

# create an instance from the tunner and perform hypertuning
tuner = kt.Hyperband(A_E.hyper_tune_optimizer, objective='val_accuracy',max_epochs=10 )

tuner.search(train_X_A, train_Y_A, epochs=10, validation_split=0.2)
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps.get('optimizer')

# to tune learning rate 

# create an instance from the tunner and perform hypertuning
tuner = kt.Hyperband(A_E.hyper_tune_learning_rate, objective='val_accuracy',max_epochs=10, overwrite=True)

tuner.search(train_X_A, train_Y_A, epochs=10, validation_split=0.2)
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps.get('learning_rate')


# ======================================================================================================================
# Task A (Arabic)
print('Task A (Arabic):')

task_A_Arabic = A_A.A_Arabic()
df = './Datasets/SemEval2017-task4-train.subtask-A.arabic.txt'
X_train, X_test, y_train, y_test = task_A_Arabic.preprocess(df)

model = task_A_Arabic.build_model()
history_A, acc_A_A_train = task_A_Arabic.Train_Model(model, X_train, y_train)
acc_A_A_test =  task_A_Arabic.get_accuracy_metrics(X_test, np.array(y_test), model)
task_A_Arabic.plot_learning_curves(history_A.history)


# ======================================================================================================================
# Task A (English)
print('Task A (English):')

train_A = './Datasets/SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
task_A = A_E.A()
train_X_A, train_Y_A, test_X_A, test_Y_A =  task_A.preprocess(train_A)

bi_lstm_model_A = task_A.build_model()
plot_model(bi_lstm_model_A,to_file='model_A_E.png',show_shapes=True, show_layer_names=True)
history_A,acc_A_E_train = task_A.Train_Model(bi_lstm_model_A ,train_X_A, train_Y_A)
acc_A_E_test = task_A.get_accuracy_metrics_A(test_X_A, test_Y_A, bi_lstm_model_A)
task_A.plot_learning_curves(history_A.history)

# ======================================================================================================================
# Task B (English)
print('Task B (English):')

train_B = './Datasets/SemEval2017-task4-dev.subtask-BD.english.INPUT.txt'
task_B = B_E.B()
train_X_B, train_Y_B, test_X_B, test_Y_B  = task_B.preprocess(train_B)

bi_lstm_model_B = task_B.build_model()
plot_model(bi_lstm_model_B,to_file='model_B_E.png',show_shapes=True, show_layer_names=False)
history_B,acc_B_E_train = task_B.Train_Model(bi_lstm_model_B ,train_X_B, train_Y_B)
acc_B_E_test =  task_B.get_accuracy_metrics(test_X_B, test_Y_B, bi_lstm_model_B)
task_B.plot_learning_curves(history_B.history)

# ======================================================================================================================
## Print out your results with following format:
print('All acc result is: TA_A:{},{};TA_E:{},{};TB_E:{},{};'.format(acc_A_A_train, acc_A_A_test,
                                                        acc_A_E_train, acc_A_E_test,acc_B_E_train, acc_B_E_test ))

