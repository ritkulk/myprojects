#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:55:06 2018

@author: rtwik
"""

import os
import pandas as pd
from nltk.corpus import stopwords
import re
from collections import Counter
import spacy

class preprocess(object):

    def __init__(self):
        self.stops =  set(stopwords.words('english'))
        stops_extra = ["south", "vscocam",
                       "africa", "instagood", "point", "hope", "cape", "capeofgoodhope",
                       "instagram", "thisissouthafrica", "cape", "town",
                       "just", "nofilter", "photooftheday","one",
                       "africa", "african", "park", "national", "South", "southafrica", "capetown",
                       "westerncape", "picoftheday", "vsco", "south", "vscocam",
                       "instagood", "instalike", "likesforlikes", "gardenroute",
                       "krugernationalpark", "krugerpark",
                       "kruger", "knp", "addo", "addoelephant", 'addonationalpark', 'addoelephantpark']
        self.stops = self.stops | set(stops_extra)

    def alphanumeric_and_split_text(self, text, keep_stops=False, lower_text=True):
            # filter text to keep only alpha-numeric characters and split sentences
            # into words
            if not keep_stops:
                stops = self.stops
            else:
                stops = []

            if lower_text:
                text = re.sub("[^a-zA-Z0-9]", " ", str(text.lower()))
            else:
                text = re.sub("[^a-zA-Z0-9]", " ", str(text))

            words = text.split()
            words = [w for w in words if w not in stops]

            return words

    def join_all_text_in_col(self, data_frame, col_label):
        return ' '.join(data_frame[col_label])

    def count_words(self, word_list, normalise=True):
        if normalise:
            counts = Counter(word_list)
            total_words = sum(counts.values())
            return {k: v/total_words for k,v in counts.items()}
        else:
            return Counter(word_list)

    def calculate_word_freq(self, data_frame, col_label):
        string = self.join_all_text_in_col(data_frame, col_label)
        string_split = self.alphanumeric_and_split_text(string)

        return self.count_words(string_split)

    def get_top_words(self, data, col_label, n_words):
        freq = self.calculate_word_freq(data, col_label)
        sorted_freq = sorted((value, key) for (key,value) in freq.items())

        return sorted_freq[-n_words:]


CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data/'
TAGGED_FILENAME = 'NLPtask_manual_classification1.csv'
DATA_FILENAME = 'NLPTask_Instagram_dataset.csv'

data_insta = pd.read_csv(DATA_DIR + DATA_FILENAME, error_bad_lines=False)
data_insta = data_insta.dropna()
data_insta['Park;;;'] = data_insta['Park;;;'].apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))

col_label = 'text'
locations = set(data_insta['Park;;;'])
PROC = preprocess()

top_words = {}
freqs = []
for loc in locations:
    data = data_insta[data_insta['Park;;;'] == loc]
    freqs.append(PROC.calculate_word_freq(data, col_label))
    top_words[loc] = PROC.get_top_words(data, col_label, 10)
    print(loc, len(data))

top_words['all'] = PROC.get_top_words(data_insta, col_label, 10)
count = sum((Counter(y) for y in freqs), Counter())
sorted_freq1 = sorted((value, key) for (key,value) in count.items())
top_words['all_norm'] = sorted_freq1[-10:]
#data_tagged = pd.read_csv(DATA_DIR + TAGGED_FILENAME)
