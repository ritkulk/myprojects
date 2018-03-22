#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu- Mar 22 00:38:28 2018

@author: rtwik
"""

import os
import re
import pandas as pd
import numpy
from keras.preprocessing.sequence import pad_sequences


class preprocess(object):
    def __init__(self):
        pass
        
    def get_raw_data(self,data_dir,filenames,category):
        with open(data_dir + filenames[category],'rb') as datafile:
            data_raw = datafile.readlines()
            
        return data_raw        
    
    def alphanumeric_and_split_text(self,text,stops=[]):                
        text = re.sub("[^a-zA-Z0-9]"," ", str(text))
        words = text.split()
        words = [w for w in words if w not in stops]
    
        return words
    
    def get_vocab(self,data):
        vocab = set()
        for sent in data:
            vocab = vocab | set(self.alphanumeric_and_split_text(sent.lower()))
    
        return vocab
    
    def build_word_to_id(self,vocab):
        # index 0 is reserved for padding
        word_to_id = dict((c, i + 1) for i, c in enumerate(vocab))
        
        return word_to_id
    
    def get_max_sent_len(self,data):
        sent_maxlen = max([len(sent) for sent in data])
        return sent_maxlen
    
    
    def get_tokenised_data(self,data):
        data_tokenised = [self.alphanumeric_and_split_text(sent.lower())
                            for sent in data]
        
        return data_tokenised


    def vectorize_data(self,data_X,data_Y,params):
        X = []
        
        sent_maxlen = params['sent_maxlen']
        
        word_to_id = params['word_to_id']
        
        for sent in data_X:
            x = [word_to_id[w] for w in sent]
            X.append(x)
        
        return (pad_sequences(X, maxlen=sent_maxlen), numpy.array(data_Y,
                                                                    dtype=int))


class data_gen(preprocess):
    def __init__(self,data,params):
        preprocess.__init__(self)
        self.data = data
        self.params = params
        self.batch_size = params['batch_size']
        
        
    def __iter__(self):
        while True:
            data_steps = int(len(self.data)/self.batch_size)+1
            for i in range(data_steps):
                start_ind = i*self.batch_size
                end_ind = (i+1)*self.batch_size
                
                if end_ind > len(self.data):
                    end_ind = len(self.data)
                
                data_batch = self.data.iloc[start_ind: end_ind]
                
                data_tokenised = self.get_tokenised_data(data_batch['sent'])
                
                yield (self.vectorize_data(data_tokenised, data_batch['label'],
                                                 self.params))



class binary_labelled_data(preprocess):
    def __init__(self):
        preprocess.__init__(self)

           
    def read_raw_data_from_file(self,data_dir,filenames):
        
        data_subj = [string.decode('utf-8') 
                        for string in self.get_raw_data(data_dir, filenames,
                                                        'subj')]
        
        data_obj = [string.decode('latin-1') 
                        for string in self.get_raw_data(data_dir, filenames,
                                                        'obj')]

        return data_subj, data_obj
    
    def make_binary_labels(self, data_subj, data_obj):
        labels_subj = [1]*len(data_subj)
        labels_obj = [0]*len(data_obj)
        
        return labels_subj, labels_obj
    
    def make_binary_labelled_data(self,data_dir,filenames):
        data_subj, data_obj = self.read_raw_data_from_file(data_dir,filenames)
        labels_subj, labels_obj = self.make_binary_labels(data_subj, data_obj)
        
        data_frame = pd.DataFrame([data_subj+data_obj, labels_subj+labels_obj])
        data_frame = data_frame.transpose()
        data_frame.columns = ['sent','label']

        return data_frame

current_dir = os.getcwd()
data_dir = current_dir + '/data/'

filenames = {'subj':'plot.tok.gt9.5000','obj':'quote.tok.gt9.5000'}

preprocessor = preprocess()
bd = binary_labelled_data()

data_frame = bd.make_binary_labelled_data(data_dir,filenames)

vocab = preprocessor.get_vocab(data_frame['sent'])
word_to_id = preprocessor.build_word_to_id(vocab)

data_tokenised = preprocessor.get_tokenised_data(data_frame['sent'])
sent_maxlen = preprocessor.get_max_sent_len(data_tokenised)
vocab_size = len(vocab)


                
params = {'batch_size': 10, 'sent_maxlen': sent_maxlen, 
          'vocab_size':vocab_size, 'word_to_id': word_to_id }

db = data_gen(data_frame,params)

c=0
for d in db:
    x = d[0]
    y = d[1]
    c+=1
    if c>0:break
