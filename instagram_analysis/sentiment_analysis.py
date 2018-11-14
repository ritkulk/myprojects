#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:10:54 2018
Implementation of a sentiment analysis model. The model used here has been trained
on IMDB sentiment data and the best performing model on the test set is saved
and used here.
@author: rtwik
"""

import pandas as pd
import _pickle
import os
from keras.models import load_model
from keras.preprocessing import sequence
from insta_word_freq import preprocess
import numpy
from sklearn.metrics import precision_recall_fscore_support


def get_mertrics(targets, predictions, upper_bound, lower_bound):
    '''calculates precision, recall, fscore. labels = [], defines the desired labels
        and their order to retuern'''
    predicitons[predicitons > upper_bound] = 1
    predicitons[predicitons < lower_bound] = 0
    predicitons[(predicitons >= lower_bound) & (predicitons <= upper_bound)] = 2

    precision, recall, fscore, support = precision_recall_fscore_support(
                                            targets, predicitons, labels=[1,0,2])

    return numpy.hstack([precision.reshape([3, 1]),
                         recall.reshape([3, 1]),
                         fscore.reshape([3, 1])])


CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data/'
MODELS_DIR = CURRENT_DIR + '/models/'
MODEL_FILENAME = 'sentiment_basic.h5'  # saved model trained on IMDB data
TAGGED_FILENAME = 'NLPtask_manual_classification1.csv'
DATA_FILENAME = 'NLPTask_Instagram_dataset.csv'
WORD_INDEX_FILENAME = 'word_index' # _pickle file containing map of word to id for trained IMDB model
col_label = 'text'


def vectorise_text(sent, PROC):
    '''transform text into vectors for the model'''
    words = PROC.alphanumeric_and_split_text(sent)
    vec = numpy.array([word_index[word] if word in word_index else 0 for word in words])
    vec = sequence.pad_sequences([vec], maxlen=500)

    return vec


data_tagged = pd.read_csv(DATA_DIR + TAGGED_FILENAME)
data_tagged = data_tagged[data_tagged['Sentiment'] != 'Other']

with open(DATA_DIR + WORD_INDEX_FILENAME, 'rb') as f:
    word_index = _pickle.load(f)

model = load_model(MODELS_DIR + MODEL_FILENAME)

PROC = preprocess()

# calculate the overlap between model vocab and data vocab
freq = PROC.calculate_word_freq(data_tagged, col_label)
vocab = set([w for w in freq])
vocab_train = set([w for w in word_index])
common_vocab = len(vocab.intersection(vocab_train))/len(vocab)
print('{} % of vocab in common'.format(common_vocab*100))

inputs = []
for sent in data_tagged[col_label]:
    inputs.append(vectorise_text(sent, PROC))

predicitons = model.predict(numpy.vstack(inputs)) # gets predictions for inputs

print(numpy.max(predicitons), numpy.min(predicitons), numpy.mean(predicitons), numpy.std(predicitons))

senti_labels = set(data_tagged['Sentiment'])
senti_to_id = {'Positive': 1, 'Neutral': 2, 'Negative': 0}

targets = list(data_tagged['Sentiment'].apply(lambda x: senti_to_id[x]))


results = get_mertrics(targets,predicitons,0.7, 0.1)

data_insta = pd.read_csv(DATA_DIR + DATA_FILENAME, error_bad_lines=False)
data_insta = data_insta.dropna()

sent_class = []
for sent in data_insta[col_label][:500]:
    vec = vectorise_text(sent, PROC)
    p = model.predict(vec)

    if p > 0.7:
        sentiment = 1
    elif p < 0.1:
        sentiment = 0
    else:
        sentiment = 2

    sent_class.append([sent, sentiment])

sent_class = pd.DataFrame(sent_class)

