#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:21:48 2018

@author: rtwik
"""
import os
from data_processing import preprocess, binary_labelled_data, data_gen, metrics
from model_functions import Models
import numpy

current_dir = os.getcwd()
data_dir = current_dir + '/data/'
models_dir = current_dir + '/models/'

filenames = {'subj':'plot.tok.gt9.5000','obj':'quote.tok.gt9.5000'}

preprocessor = preprocess()
bd = binary_labelled_data()

data_frame = bd.make_binary_labelled_data(data_dir,filenames)

vocab = preprocessor.get_vocab(data_frame['sent'])
word_to_id = preprocessor.build_word_to_id(vocab)

data_tokenised = preprocessor.get_tokenised_data(data_frame['sent'])
sent_maxlen = preprocessor.get_max_sent_len(data_tokenised)
vocab_size = len(vocab)

n_folds = 1
n_scores = 4

results = numpy.zeros((n_folds,n_scores))

for n_loop in range(n_folds):
    split_ratio = 0.8
    data_train, data_test = preprocessor.make_train_test_split(data_frame,split_ratio)
        
    
    params = {'batch_size': 100, 'sent_maxlen': sent_maxlen, 
              'vocab_size':vocab_size, 'word_to_id': word_to_id,
              'sent_maxlen':sent_maxlen, 'data_dir':data_dir,
              'models_dir':models_dir, 'model_name':'subj_classifier.h5'}
    
    dg_train = data_gen(data_train,params)
    dg_test = data_gen(data_test,params)
    params['train_steps'] = int(len(data_train)/params['batch_size'])
    params['eval_steps'] = int(len(data_test)/params['batch_size'])
    
    
    model_func = Models()
    
#    model = model_func.compile_conv_bilstm_model(params)
    model = model_func.compile_multi_cnn(params)
    
    model = model_func.train_model(model,dg_train, dg_test, params)
    
    del(model)
    
    model = model_func.load_model(params['model_name'],params)
        
    scores = model_func.evaluate_model(model, dg_test, params)
    
   
    prediction = model.predict_generator(dg_test,
                                         steps=params['eval_steps'])
    
    met = metrics()
    
    targets = []
    for i in dg_test:
        targets = targets + list(i[1][0])
        if len(targets)==len(data_test):
            break
    
    precision,recall,fscore,support = met.get_prec_rec_fscore(
                                                targets, prediction)

    results[n_loop,0] = scores[1]
    results[n_loop,1] = precision
    results[n_loop,2] = recall
    results[n_loop,3] = fscore
    
    print("Accuracy: %.2f%%" % (scores[1]))
    print(precision,recall,fscore,support)

print('Mean Accuracy:{}, Mean Precision:{}, Mean Recall:{}, Mean Fscore:{}'.format(
         numpy.mean(results[:,0]),numpy.mean(results[:,1]),numpy.mean(results[:,2]),
         numpy.mean(results[:,3]),))