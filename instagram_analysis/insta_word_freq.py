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
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer

class preprocess(object):

    def __init__(self):
        self.lemmatiser = WordNetLemmatizer()
        self.mwe_tokeniser = MWETokenizer([('indian','ocean'), ('atlantic', 'ocean'), ('wild', 'dog'), ('game', 'drive'),
                              ('lion','head'), ('watering','hole'), ('garden', 'route'), ('suspension', 'bridge'),
                              ('wild','elephant'), ('baby','elephant')])

        self.stops =  set(stopwords.words('english'))

        stops_extra = ["south", "vscocam",
                       "africa", "instagood", "point", "hope", "cape", "capeofgoodhope",
                       "instagram", "thisissouthafrica", "cape", "town",
                       "just", "nofilter", "photooftheday","one",
                       "africa", "african", "park", "national", "South", "southafrica", "capetown",
                       "westerncape", "picoftheday", "vsco", "south", "vscocam",
                       "instagood", "instalike", "likesforlikes", "gardenroute",
                       "krugernationalpark", "krugerpark",
                       "kruger", "knp", "addo", "addoelephant", 'addonationalpark', 'addoelephantpark',
                       'addoelephantnationalpark', "nationalpark", "sa", "easterncape", "meetsouthafrica",
                       'got', "tbt", "go",
                       "en", "us", 'ig', "thisissouthafrica", "igers", "latergram", "za", "nofilter",
                       "one", "park", "national", "South", "southafrica", "capetown", "westerncape",
                       "vsco", "pe", "vscocam", "instagood", "instalike", "likesforlikes", "gardenroute",
                       "krugernationalpark", "krugerpark", 'tagsforlikes', 'nstatravel', "kruger", "knp",
                       "addo", "addoelephant", 'tsitsikamma','tsitsikammanationalpark' ]

        self.stops = self.stops | set(stops_extra)

    def alphanumeric_and_split_text(self, text, keep_stops=False, lower_text=True):
            # filter text to keep only alpha-numeric characters and split sentences
            # into words
            if not keep_stops:
                stops = self.stops
            else:
                stops = []

            if lower_text:
                text = re.sub("[^a-zA-Z]", " ", str(text.lower()))
            else:
                text = re.sub("[^a-zA-Z]", " ", str(text))

            text_split = text.split()
            #words = [self.lemmatiser.lemmatize(w) for w in text_split if w not in stops]

#            self.mwe_tokeniser.add_mwe([('indian','ocean'), ('atlantic', 'ocean'), ('wild', 'dog'), ('game', 'drive'),
#                              ('lion','head'), ('watering','hole'), ('garden', 'route'), ('suspension', 'bridge'),
#                              ('wild','elephant'), ('baby','elephant')])

            words = [self.lemmatiser.lemmatize(w) for w in self.mwe_tokeniser.tokenize(text_split) if w not in stops]

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

    def make_ngrams(self, word_list, gram_num):
        if gram_num == 1:
            return ['-'.join(sorted(i)) for i in zip(*[word_list[i:] for i in range(gram_num)])]
        elif gram_num == 2:
            return ['-'.join(sorted(i)) for i in zip(*[word_list[i:] for i in range(gram_num)]) if i[0]!=i[1]]
        elif gram_num > 2:
            return ['-'.join(sorted(i)) for i in zip(*[word_list[j:] for j in range(gram_num)]) \
                    if i[0] != i[1] and i[1] != i[2]]

    def calculate_word_freq(self, data_frame, col_label, gram_num):
        string = self.join_all_text_in_col(data_frame, col_label)
        tokens = self.alphanumeric_and_split_text(string)

        if gram_num > 1:
            token_grams = self.make_ngrams(tokens, gram_num)
            return self.count_words(token_grams)
        else:
            return self.count_words(tokens)

    def get_top_words(self, data, col_label, n_words, gram_num):
        freq = self.calculate_word_freq(data, col_label, gram_num)
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

    def dict_to_csv(self, word_dict, DATA_DIR, loc, gram_n):
        f_data_frame = pd.DataFrame([word_dict])
        f_data_frame = f_data_frame.transpose()
        f_data_frame =f_data_frame.reset_index(level=0)
        f_data_frame.columns = ['word', 'frequency']
        f_data_frame.sort_values(by=['frequency'], ascending=False, inplace=True)

        f_data_frame.to_csv(DATA_DIR + loc+'_gram_'+str(gram_n)+'_freq.csv')
        print('csv saved to ' + loc + '_gram_'+str(gram_n)+'_freq.csv')


if __name__ == '__main__':
    CURRENT_DIR = os.getcwd()
    DATA_DIR = CURRENT_DIR + '/data/'
    TAGGED_FILENAME = 'NLPtask_manual_classification1.csv'
    DATA_FILENAME = 'NLPTask_Instagram_dataset.csv'
    W2V_MODEL_PATH = '/home/rtwik/Downloads/GoogleNews-vectors-negative300.bin'

    loc_col_label = 'Park;;;'
    data_insta = pd.read_csv(DATA_DIR + DATA_FILENAME, error_bad_lines=False)
    data_insta = data_insta.dropna()
    data_insta[loc_col_label] = data_insta[loc_col_label].apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))

    col_label = 'text'
    locations = set(data_insta[loc_col_label])
    PROC = preprocess()

    print('Loading W2V model') # gensim model
    model = gensim.models.KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)

    gram_n = 1
    top_n_words = 10

    top_words = {}
    words_corr = {}
    freqs = {}
    for loc in locations:
        data = data_insta[data_insta[loc_col_label] == loc]

        f = PROC.calculate_word_freq(data, col_label, gram_n)
        freqs[loc] = f

        #PROC.dict_to_csv(f, DATA_DIR ,loc, gram_n)

        top_words[loc] = PROC.get_top_words(data, col_label, top_n_words, gram_n)

        words_corr[loc] = PROC.get_correlated_words(top_words[loc], f, model)
        print(loc, len(data), len(f))

    # calculating for all the parks
    top_words['all'] = PROC.get_top_words(data_insta, col_label, top_n_words, gram_n)
    count = sum((Counter(y) for y in freqs), Counter()) # combines dictionaries and sums common keys
    sorted_freq1 = sorted(((value, key) for (key, value) in count.items()), reverse=False)
    top_words['all_norm'] = sorted_freq1[-10:]

