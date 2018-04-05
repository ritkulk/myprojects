#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:21:48 2018
This is the main script that runs the entire pipeline from data loading,
processing, traing and evaluating a cross validation experiment for the 
subjectivity classifier
@author: rtwik
"""
import os
from data_processing import preprocess, binary_labelled_data, data_gen, metrics
from model_functions import Models
import numpy

#setting operational folders
current_dir = os.getcwd()
data_dir = current_dir + '/data/'
models_dir = current_dir + '/models/'

# filenames for raw downloaded data
filenames = {'subj':'plot.tok.gt9.5000','obj':'quote.tok.gt9.5000'}

# instantiating objects for pre-processing, labelling data and eval metrics
# and model functions
preprocessor = preprocess()
bd = binary_labelled_data()
met = metrics()
model_func = Models()

# creates a data frame where subjective data is labelled as 1 and the other 0
data_frame = bd.make_binary_labelled_data(data_dir,filenames)

# build vocabulary and index over all data
vocab = preprocessor.get_vocab(data_frame['sent'])
word_to_id = preprocessor.build_word_to_id(vocab)

data_tokenised = preprocessor.get_tokenised_data(data_frame['sent'])
sent_maxlen = preprocessor.get_max_sent_len(data_tokenised)
vocab_size = len(vocab)

# parameters required for processing and running model (not exhaustive but 
# sufficient for this example)
params = {'batch_size': 100, 'sent_maxlen': sent_maxlen, 
          'vocab_size':vocab_size, 'word_to_id': word_to_id,
          'sent_maxlen':sent_maxlen, 'data_dir':data_dir,
          'models_dir':models_dir, 'model_name':'subj_classifier.h5',
          'model_type':'cnn_lstm', # 'cnn_lstm' or 'multi_cnn' 
          'epochs': 20
          }
 
n_folds = 10  # number of folds for cross validation
n_scores = 4 # number of result metrics used (fixed)

results = numpy.zeros((n_folds,n_scores))

# main loop to run the cross validation
for n_loop in range(n_folds):
    split_ratio = 0.8 # fraction of data set aside for train, remaining for test
    data_train, data_test = preprocessor.make_train_test_split(data_frame,
                                                               split_ratio)
    
    #data iterator objects     
    dg_train = data_gen(data_train,params)
    dg_test = data_gen(data_test,params)
    
    params['train_steps'] = int(len(data_train)/params['batch_size'])
    params['eval_steps'] = int(len(data_test)/params['batch_size'])
    
    # builds the model, cnn+lstm or multi channel cnn
    if params['model_type'] == 'cnn_lstm':
        model = model_func.compile_conv_lstm_model(params)
    if params['model_type'] == 'multi_cnn':
        model = model_func.compile_multi_cnn(params)
    
    #train model
    model = model_func.train_model(model,dg_train, dg_test, params)
    
    del(model)
    
    # load best model
    model = model_func.load_model(params['model_name'],params)
        
    # evaluate model
    scores = model_func.evaluate_model(model, dg_test, params)
    
    # get output predictions 
    prediction = model.predict_generator(dg_test,
                                         steps=params['eval_steps'])
    
    # get targets to evaluate metrics over predictions
    targets = preprocessor.get_targets_for_eval(dg_test,len(data_test))
    
    # get evaluation metrics
    precision,recall,fscore,support = met.get_prec_rec_fscore(
                                                targets, prediction)

    # store results
    results[n_loop,0] = scores[1]
    results[n_loop,1] = precision
    results[n_loop,2] = recall
    results[n_loop,3] = fscore
    
print('Mean Accuracy:{}, Mean Precision:{}, Mean Recall:{}, Mean Fscore:{}'.format(
         numpy.mean(results[:,0]),numpy.mean(results[:,1]),numpy.mean(results[:,2]),
         numpy.mean(results[:,3]),))