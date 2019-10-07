#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:44:01 2018
Implements an LDA model for topic modelling. Preplexity is used to determine
optimal number of topics.
@author: rtwik
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
import pandas as pd
import os
from nltk.corpus import stopwords
import numpy
from insta_word_freq import preprocess


def get_topics(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[topic_idx] = [" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])]

    return topics

def get_ngram_doc(proc_object, data, gram_num, stops):

    docs = []
    for t in data['text']:
        words = [w.lower() for w in t.split() if w.lower() not in stops]
        docs.append(' '.join(proc_object.make_ngrams(words, gram_num)))

    return docs


CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data/'
TAGGED_FILENAME = 'NLPtask_manual_classification1.csv'
DATA_FILENAME = 'NLPTask_Instagram_dataset.csv'


# define stopwords for exclusion
stops =  set(stopwords.words('english'))
stops_extra = ["south", "vscocam",
               "africa", "instagood", "point", "hope", "cape", "capeofgoodhope",
               "instagram", "thisissouthafrica", "cape", "town",
               "just", "nofilter", "photooftheday","one",
               "africa", "african", "park", "national", "South", "southafrica", "capetown",
               "westerncape", "picoftheday", "vsco", "south", "vscocam",
               "instagood", "instalike", "likesforlikes", "gardenroute",
               "krugernationalpark", "krugerpark",
               "kruger", "knp", "addo", "addoelephant", 'addonationalpark', 'addoelephantpark']
stops = stops | set(stops_extra)

PROC = preprocess()

# prepare data
data_insta = pd.read_csv(DATA_DIR + DATA_FILENAME, error_bad_lines=False)
data_insta = data_insta.dropna()
data_insta['text'] = data_insta['text'].apply(lambda x: x.encode(encoding='ascii', errors='ignore').decode())
data_insta['text'] = data_insta['text'].apply(lambda x: re.sub("[^a-zA-Z0-9]", " ", x))
data_insta['Park;;;'] = data_insta['Park;;;'].apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))
#data_insta['text'] = data_insta['text'].apply(lambda x: re.sub("[^a-zA-Z0-9]", " ", x))
#data_insta['text'] = data_insta['text'].applyI was a PhD candidate at SISSA in the Cognitive Neuroscience Department and completed my PhD in May 2014.(lambda x: ' '.join(x.split()))

locations = set(data_insta['Park;;;'])

gram_num = 1
n_features = 50  # feature dimension for vectoriser
n_top_words = 5  # number of top n words to extract
topic_results = {}
perp_results = {}

for loc in locations:
    data = data_insta[data_insta['Park;;;'] == loc]

    if gram_num > 1:
        documents = get_ngram_doc(PROC, data, gram_num, stops)
        vocab = set()
        for d in documents:
            vocab = vocab | set(d.split())

        word_to_id = {w:i for i,w in enumerate(vocab)}
        tf_vectorizer = CountVectorizer(max_df=0.5, min_df=1, max_features=n_features, stop_words=stops, vocabulary=word_to_id)

    else:
        documents = data['text']
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=0.01, max_features=n_features, stop_words=stops)

    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    n_topics = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40] # number of topics

    perp = numpy.zeros((len(n_topics), 3))
    topics = {}
    c = 0
    for n_t in n_topics:

        lda = LatentDirichletAllocation(n_components=n_t, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

        topics[n_t] = get_topics(lda, tf_feature_names, n_top_words)
        perp[c, 0] = n_t
        perp[c, 1] = lda.perplexity(tf)
        perp[c, 2] = lda.score(tf)

        c += 1

    topic_results[loc] = topics
    perp_results[loc] = perp
    min_id = numpy.where(perp[:, 1] == min(perp[:, 1]))[0]
    print(loc)
    print(topics[perp[min_id[0],0]])
