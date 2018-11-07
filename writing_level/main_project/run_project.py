#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs the entire experiment for training and evaluating the model
@author: rtwik
"""
import os
from data_processing import data_preprocess, data_gen
from model_functions import model_functions

# sets all the required files given the folder structure of the project
current_dir = os.getcwd()
data_dir = current_dir+'/data/'
data_train_filename = 'data_train.csv'
data_test_filename = 'data_test.csv'
data_dev_filename = 'data_dev.csv'
model_filename = 'wr_level.h5'
field_filename = 'field_names.txt'
word_frequency_file = 'wiki-100k.txt'
train_filepath = data_dir + data_train_filename
test_filepath = data_dir + data_test_filename
dev_filepath = data_dir + data_dev_filename
model_filepath = data_dir + model_filename

# instantiates data preprocessing object
dp = data_preprocess()

# gets word occurrence frequency in wiki corpus from table
word_frequency = dp.get_word_frequency_from_file(data_dir + word_frequency_file)

# groupby, map from level number to target value, if groupby=1 then produces
# a regular map, if groupby=n then maps groups of n consecutive levels
# to one target value. groupby=3 for task 2, 1 otherwise for task 1, refer
# report
groupby = 1

# various parameter values required by the model and data processor,
# not exhaustive but sufficient for the task
output_class_list = dp.get_output_classes(train_filepath, 1)

params = {'batch_size': 100,
          'train_length': dp.get_data_length(train_filepath),
          'test_length': dp.get_data_length(test_filepath),
          'output_classes': dp.get_output_classes(train_filepath, groupby),
          'output_map': dp.get_output_map(output_class_list, groupby),
          'n_features': 4,  # number of input features
          'n_chunks_train': 3000,  # number of batches to process
          'n_chunks_test': 1000,
          'n_chunks_dev': 100,
          'epochs': 50,  # training epochs
          'model_filepath': data_dir + model_filename  # path to save model
          }

# create data generator objects for model
train_data_generator = data_gen(train_filepath, params,
                                params['n_chunks_train'],
                                word_frequency)
test_data_generator = data_gen(test_filepath, params,
                               params['n_chunks_test'],
                               word_frequency)
dev_data_generator = data_gen(dev_filepath, params,
                              params['n_chunks_dev'],
                              word_frequency)

# instantiate model functions object
mf = model_functions()

print('building model')
model = mf.compile_model(params)
print('training model')
model = mf.train_model(model, train_data_generator, dev_data_generator, params)

del(model)

print('loading best saved model')
model = mf.load_saved_model(params['model_filepath'])
print('evaluating model')
score = mf.evaluate_model(model, test_data_generator, params)
print('getting predictions')
predictions = mf.get_predictions(model, test_data_generator, params)

targets = dp.get_targets_for_eval(test_data_generator, params['n_chunks_test'])

# claculate metrics
precision, recall, fscore, support = mf.get_F_metrics(targets, predictions)

print('accuracy:', score[1])
print('precision: {}, recall: {}, fscore: {}'.format(precision, recall,
                                                     fscore))
