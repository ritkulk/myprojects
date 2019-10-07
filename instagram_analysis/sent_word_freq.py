#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 04:07:45 2019

@author: rtwik
"""

import pandas as pd
import os
import re
import gensim
from insta_word_freq import preprocess

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data/'
FILENAME = 'NRC_Emotions_classified_senti.csv'
W2V_MODEL_PATH = '/home/rtwik/Downloads/GoogleNews-vectors-negative300.bin'


data_insta = pd.read_csv(DATA_DIR+FILENAME, error_bad_lines=False)

loc_col_label = 'Park'
sent_col_label = 'Senti_class'
sent_val = 'negative'
text_col_label = 'text'

#data_insta = data_insta.dropna()
#data_insta[loc_col_label] = data_insta[loc_col_label].apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))

locations = ['AENP', 'GRNP', 'KNP', 'TMNP']

data_filtered = data_insta[data_insta[sent_col_label] == sent_val]

PROC = preprocess()

print('Loading W2V model') # gensim model
model = gensim.models.KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)

gram_n = 2
top_n_words = 10

top_words = {}
words_corr = {}
freqs = {}
for loc in locations:
    data = data_filtered[data_filtered[loc_col_label] == loc]

    f = PROC.calculate_word_freq(data, text_col_label, gram_n)
    freqs[loc] = f

    PROC.dict_to_csv(f, DATA_DIR, sent_val + '_' + loc, gram_n)

    top_words[loc] = PROC.get_top_words(data, text_col_label, top_n_words, gram_n)

    words_corr[loc] = PROC.get_correlated_words(top_words[loc], f, model)
    print(loc, len(data), len(f))