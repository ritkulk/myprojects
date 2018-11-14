#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:55:06 2018
This code implements a word frequency calculation for Instagram data on
4 national parks in South Africa
@author: rtwik
"""

import os
import pandas as pd
from nltk.corpus import stopwords
import re
from collections import Counter
import gensim

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
        data_frame[col_label] = data_frame[col_label].apply(lambda x: str(x))
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

    def get_correlated_words(self, top_words, freqs, model):
        words_t = [i[1] for i in top_words]
        top_corr = {}
        for w in freqs.keys():
            w_corr = []
            for w_t in words_t:
                if w in model and w_t in model:
                    if model.similarity(w, w_t) > 0.5:
                        if w_t in top_corr:
                            w_corr = top_corr[w_t]
                            w_corr.append(w)
                            top_corr[w_t] = w_corr
                        else:
                            top_corr[w_t] = w_corr
        return top_corr


if __name__ == '__main__':
    CURRENT_DIR = os.getcwd()
    DATA_DIR = CURRENT_DIR + '/data/'
    TAGGED_FILENAME = 'NLPtask_manual_classification1.csv'
    DATA_FILENAME = 'NLPTask_Instagram_dataset.csv'
    W2V_MODEL_PATH = '/home/rtwik/Downloads/GoogleNews-vectors-negative300.bin'

    data_insta = pd.read_csv(DATA_DIR + DATA_FILENAME, error_bad_lines=False)
    data_insta = data_insta.dropna()
    data_insta['Park;;;'] = data_insta['Park;;;'].apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))

    col_label = 'text'
    locations = set(data_insta['Park;;;'])
    PROC = preprocess()

    print('Loading W2V model') # gensim model
    model = gensim.models.KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)

    top_words = {}
    words_corr = {}
    freqs = []
    for loc in locations:
        data = data_insta[data_insta['Park;;;'] == loc]

        f = PROC.calculate_word_freq(data, col_label)
        freqs.append(f)

        top_words[loc] = PROC.get_top_words(data, col_label, 10)

        words_corr[loc] = PROC.get_correlated_words(top_words[loc], f, model)
        print(loc, len(data), len(f))

    top_words['all'] = PROC.get_top_words(data_insta, col_label, 10)
    count = sum((Counter(y) for y in freqs), Counter()) # combines dictionaries and sums common keys
    sorted_freq1 = sorted((value, key) for (key, value) in count.items())
    top_words['all_norm'] = sorted_freq1[-10:]

