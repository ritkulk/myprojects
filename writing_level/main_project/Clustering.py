#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code performs unsupervised clustering using DBSCAN algorithm on the
features of the writing data. Clusters extracted are stored as texts clusters,
level clusters and unit clusters into a pickle file.
@author: rtwik
"""

import os
import numpy
import pandas as pd
from data_processing import data_preprocess, data_gen
from sklearn.cluster import DBSCAN
import _pickle

# set folder and file paths
current_dir = os.getcwd()
data_dir = current_dir+'/data/'
data_train_filename = 'data_train.csv'
word_frequency_file = 'wiki-100k.txt'
train_filepath = data_dir + data_train_filename


# instantiates data preprocessing object
dp = data_preprocess()

# gets word occurrence frequency in wiki corpus from table
word_frequency = dp.get_word_frequency_from_file(data_dir + word_frequency_file)

groupby = 1  # refer run_project for detials
params = {'batch_size': 100,
          'output_classes': dp.get_output_classes(train_filepath, groupby),
          'output_map': dp.get_output_map(dp.get_output_classes(train_filepath,
                                                                1), groupby),
          'n_features': 4,
          'n_chunks': 100
          }

# create data generator object
train_data_generator = data_gen(train_filepath, params, params['n_chunks'],
                                word_frequency)

print('collecting data features')
c = 0
data = []
for i in train_data_generator:
    data.append(i[0])

    c += 1
    if c >= params['n_chunks']:
        break

data = numpy.vstack(data)
data = data/numpy.max(data, axis=0)  # normalising data
numpy.random.shuffle(data)

print('performing DBSCAN')
db = DBSCAN(eps=0.1, min_samples=25).fit(data)  # requires more exploration
core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = numpy.array(db.labels_)

print('obtaining visible clusters from source data')
df = pd.read_csv(train_filepath)
df = df.iloc[:10000]

texts = []
levels = []
units = []
for i in range(numpy.max(labels)):
    inds = numpy.where(labels == i)[0]

    df_cluster = df.iloc[inds]
    texts.append([t for t in df_cluster['text']])
    levels.append(df_cluster['@level'])
    units.append(df_cluster['@unit'])

print('saving visible clusters to pickle file')
f = open(data_dir + 'cluster_results', 'wb')
_pickle.dump({'text': texts, 'levels': levels, 'units': units}, f)
f.close()
