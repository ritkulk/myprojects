#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code trains an image recognition model to detect pangolin images scraped
from tweets. Inputs are extracted from VGG16 model.
@author: rtwik
"""

import os
from data_processing import preprocess, binary_labelled_data, data_gen, metrics
from model_functions import model_functions
import numpy
import pandas as pd
import ast

def train_and_save_model(TRAIN_FILEPATH, TEST_FILEPATH, data_gen, model_func):
    ''' train, evaluate and save best model'''

    dg_train = data_gen(TRAIN_FILEPATH, 'train', params)
    dg_test = data_gen(TEST_FILEPATH, 'test', params)

    model = model_func.compile_model(params)

    model = model_func.train_model(model, dg_train, dg_test, params)

    del(model)

    model = model_func.load_saved_model(params['model_filepath'])

    score = model_func.evaluate_model(model, dg_test, params)

    return score



# setting operational folders
CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data/'
MODELS_DIR = CURRENT_DIR + '/models/'
TRAIN_FILEPATH = DATA_DIR + 'data_train.csv'
TEST_FILEPATH = DATA_DIR + 'data_test.csv'
PREDICT_FILEPATH = DATA_DIR + 'data_img_features.csv'

# filenames for raw downloaded data
binary_filenames = ['pang_features.csv', 'rand_features.csv']

# instantiating objects for pre-processing, labelling data and eval metrics
# and model functions
preprocessor = preprocess()
bd = binary_labelled_data()
met = metrics()
model_func = model_functions()
batch_size = 10
train_steps = int(len(pd.read_csv(TRAIN_FILEPATH))/batch_size)
test_steps = int(len(pd.read_csv(TEST_FILEPATH))/batch_size)

params = {'batch_size': batch_size,
          'data_dir': DATA_DIR,
          'train_split_ratio': 0.9,
          'n_features': 25088,
          'n_chunks_train': train_steps,
          'n_chunks_test': test_steps,
          'model_filepath': MODELS_DIR + 'pangolin_classifier.h5',
          'model_type': 'cnn_lstm',  # 'cnn_lstm' or 'multi_cnn'
          'epochs': 50
          }

#----------to prepare data when running the first time----------
# creates a data frame with binary labels
#data_binary = bd.make_binary_labelled_data(DATA_DIR, binary_filenames)
#
#preprocessor.prepare_and_save_train_test_data(data_binary, params)
#--------------------------------------------------------------


#--------train,test,save model---------
#score = train_and_save_model(TRAIN_FILEPATH, TEST_FILEPATH, data_gen, model_func)

model = model_func.load_saved_model(params['model_filepath'])

data_predict = pd.read_csv(PREDICT_FILEPATH)

c = 0
predictions = {}
for _, rows in data_predict.iterrows():
    x = numpy.array(ast.literal_eval(rows['features']))
    x = x.reshape(1, params['n_features'])

    predictions[rows['name']] = model.predict(x)

    if c % 200 == 0:
        print('done prediction', c/len(data_predict))
    c += 1

