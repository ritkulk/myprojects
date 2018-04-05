#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu- Mar 22 00:38:28 2018
modules necessary to pre-process data and prepare to be fed into the model
Also contains module for the evaluation metrics
@author: rtwik
"""

import re
import pandas as pd
import numpy
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support

class preprocess(object):
    def __init__(self):
        pass
        
    def get_raw_data(self,data_dir,filenames,category):
        # get raw data from file
        with open(data_dir + filenames[category],'rb') as datafile:
            data_raw = datafile.readlines()
            
        return data_raw        
    
    def get_stopwords(self):
        # make a list of stopwords
        stops = set(stopwords.words('english'))
        
        return stops
        
    def alphanumeric_and_split_text(self,text,keep_stops=False):  
        # filter text to keep only alpha-numeric characters and split sentences
        # into words
        if not keep_stops:
            stops = self.get_stopwords()
        else:
            stops = []
              
        text = re.sub("[^a-zA-Z0-9]"," ", str(text))
        words = text.split()
        words = [w for w in words if w not in stops]
    
        return words
    
    def get_vocab(self,data):
        # makes a set of all the words in the data
        vocab = set()
        for sent in data:
            vocab = vocab | set(self.alphanumeric_and_split_text(sent.lower()))
    
        return vocab
    
    def build_word_to_id(self,vocab):
        # builds index for the vocab. index 0 is reserved for padding
        word_to_id = dict((c, i + 1) for i, c in enumerate(vocab))
        
        return word_to_id
    
    def get_max_sent_len(self,data):
        # claculates the maximum words in a sentence
        sent_maxlen = max([len(sent) for sent in data])
        return sent_maxlen
    
    
    def get_tokenised_data(self,data):
        # tokenises the data
        data_tokenised = [self.alphanumeric_and_split_text(sent.lower())
                            for sent in data]
        
        return data_tokenised


    def vectorize_data(self,data_X,data_Y,params):
        # transforms sentences into vectors 
        X = []
        
        sent_maxlen = params['sent_maxlen']
        
        word_to_id = params['word_to_id']
        
        for sent in data_X:
            x = [word_to_id[w] for w in sent]
            X.append(x)
        
        return (pad_sequences(X, maxlen=sent_maxlen), numpy.array(data_Y,
                                                                    dtype=int))

    def make_train_test_split(self,data_frame,ratio):
        # splits the data into train and test
        data_frame = data_frame.sample(frac=1)
        data_train = data_frame.iloc[:int(len(data_frame)*ratio)]
        data_test = data_frame.iloc[int(len(data_frame)*ratio):]
        
        return data_train, data_test
    
    def get_targets_for_eval(self, data_iter, stop_len):
        # gets targets to evaluate metrics over predictions
        targets = []
        for i in data_iter:
            targets = targets + [i[1][0]]
            if len(targets)==stop_len:
                break
        return targets

class data_gen(preprocess):
    # iterator object to stream data to the model
    def __init__(self,data,params):
        preprocess.__init__(self)
        self.data = data
        self.params = params
        self.batch_size = params['batch_size']
        
        
    def __iter__(self):
        return self
        
    def __next__(self):
        data_steps = int(len(self.data)/self.batch_size)+1
        for i in range(data_steps):
            start_ind = i*self.batch_size
            end_ind = (i+1)*self.batch_size
            
            if end_ind > len(self.data):
                end_ind = len(self.data)
            
            data_batch = self.data.iloc[start_ind: end_ind]
            
            data_tokenised = self.get_tokenised_data(data_batch['sent'])
            
            X,Y=self.vectorize_data(data_tokenised, data_batch['label'],
                                             self.params)
            if self.params['model_type'] == 'multi_cnn':
                return ([X,X,X],[Y])
            if self.params['model_type'] == 'cnn_lstm':
                return (X,Y)



class binary_labelled_data(preprocess):
    # creates binary labels for subjective and objective data
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

class metrics(object):
    # module to get precision, recall and fscore over predictions
    def __init__(self):
        pass
    
    def get_prec_rec_fscore(self,y_true, y_pred, threshold=0.5):
        '''
        claculates the precision, recall, fscore and support
        of a binary output classifier
        '''
        binary_labels = [0, 1]
        y_pred = numpy.piecewise(y_pred, [y_pred <= threshold, y_pred > threshold],
                                 binary_labels)
    
        precision, recall, fscore, support = precision_recall_fscore_support(
                                                y_true, y_pred, average='binary')
    
        return precision, recall, fscore, support
