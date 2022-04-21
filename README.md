# AMLS_II_assignment20_21

## Introduction of the tasks
Hello! Here are two main tasks: 

Task A. Message Polarity Classification: Given a message, classify whether the message is of positive, negative, or neutral sentiment.
This task is divided into 2 parts: English and Arabic.

Task B. Given a message and a topic, classify the message on two-point scale: positive or negative sentiment towards that topic.
This task is implemented only in English.

## File organization

<<README.md

<<A (a_A.py, a_E.py)

<<B (b.py)

<<Datasets (SemEval2017-task4-dev.subtask-A.english.INPUT.txt, SemEval2017-task4-dev.subtask-BD.english.INPUT.txt, SemEval2017-task4-train.subtask-A.arabic.txt)

<<main.py

## How to run the code
To run the code, please do:
1. Use Google Colab Pro with configuration: high RAM and GPU
2. Create a folder in your Google Drive named: AMLS2
3. Upload file main.py, Run.ipynb and folder A, B and Datasets this AMLS2 folder
4. Run Run.ipynb in Google Colab

## Necessary packages
Need to pip install ekphrasis, keras-self-attention, keras-tuner --upgrade
