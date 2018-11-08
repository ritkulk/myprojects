#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:44:01 2018

@author: rtwik
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
import pandas as pd
import os
from nltk.corpus import stopwords
import numpy

def get_topics(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[topic_idx] = [" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])]

    return topics

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data/'
TAGGED_FILENAME = 'NLPtask_manual_classification1.csv'
DATA_FILENAME = 'NLPTask_Instagram_dataset.csv'

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


data_insta = pd.read_csv(DATA_DIR + DATA_FILENAME, error_bad_lines=False)
data_insta = data_insta.dropna()
data_insta['Park;;;'] = data_insta['Park;;;'].apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))
#data_insta['text'] = data_insta['text'].apply(lambda x: re.sub("[^a-zA-Z0-9]", " ", x))
#data_insta['text'] = data_insta['text'].apply(lambda x: ' '.join(x.split()))

locations = set(data_insta['Park;;;'])

n_features = 50
n_top_words = 10
for loc in locations:
    data = data_insta[data_insta['Park;;;'] == loc]

    documents = data['text']

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=n_features, stop_words=stops)
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    n_topics = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]

    perp = numpy.zeros((len(n_topics), 2))
    topics = {}
    c = 0
    for n_t in n_topics:

        lda = LatentDirichletAllocation(n_components=n_t, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

        topics[n_t] = get_topics(lda, tf_feature_names, n_top_words)
        perp[c, 0] = n_t
        perp[c, 1] = lda.perplexity(tf)

        c += 1

    min_id = numpy.where(perp[:, 1] == min(perp[:, 1]))[0]
    print(topics[perp[min_id[0],0]])
