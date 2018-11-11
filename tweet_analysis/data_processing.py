#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:57:40 2018

@author: rtwik
"""

import ast
import pandas as pd
import numpy
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support


class preprocess(object):
    def __init__(self):
        pass

    def vectorize_data(self, data_X, data_Y, params):
        # transforms sentences into vectors

        return (numpy.vstack(data_X), numpy.array(data_Y, dtype=int))

    def make_train_test_split(self, data_frame, ratio):
        # splits the data into train and test
        data_frame = data_frame.sample(frac=1)
        data_train = data_frame.iloc[:int(len(data_frame)*ratio)]
        data_test = data_frame.iloc[int(len(data_frame)*ratio):]

        return data_train, data_test

    def get_targets_for_eval(self, data_iter, stop_len):
        # gets targets to evaluate metrics over predictions
        targets = []
        for i in data_iter:
            targets = targets + list(i[1])
            if len(targets) == stop_len:
                break
        return targets

    def prepare_and_save_train_test_data(self, data_frame, params):
        ratio = params['train_split_ratio']
        DATA_DIR = params['data_dir']

        data_train, data_test = self.make_train_test_split(data_frame, ratio)

        data_train.to_csv(DATA_DIR + 'data_train.csv')
        data_test.to_csv(DATA_DIR + 'data_test.csv')

        print('train data saved to {}'.format(DATA_DIR + 'data_train.csv'))
        print('test data saved to {}'.format(DATA_DIR + 'data_test.csv'))


class data_gen(preprocess):
    # iterator object to stream data to the model
    def __init__(self, data_filepath, mode, params):
        preprocess.__init__(self)
        self.data_filepath = data_filepath
        self.params = params
        self.batch_size = params['batch_size']
        self.mode = mode

    def __iter__(self):
        return self

    def __next__(self):

        for chunk in pd.read_csv(self.data_filepath,
                                 chunksize=self.batch_size):


            chunk['features'] = chunk['features'].apply(lambda x: ast.literal_eval(x))

            if self.mode == 'train' or self.mode == 'test':
                X, Y = self.vectorize_data(chunk['features'], chunk['label'], self.params)

                return (X, Y)

            if self.mode == 'predicit':
                return numpy.vstack(chunk['features'])


class binary_labelled_data(preprocess):
    # creates binary labels for subjective and objective data
    def __init__(self):
        preprocess.__init__(self)

    def read_raw_data_from_file(self, data_dir, filenames):

        data_one = pd.read_csv(data_dir + filenames[1])
        data_zero = pd.read_csv(data_dir + filenames[0])
        return data_one, data_zero

    def make_binary_labels(self, data_subj, data_obj):
        labels_subj = [1]*len(data_subj)
        labels_obj = [0]*len(data_obj)

        return labels_subj, labels_obj

    def make_binary_labelled_data(self, data_dir, filenames):
        data_one, data_zero = self.read_raw_data_from_file(data_dir, filenames)
        labels_one, labels_zero = self.make_binary_labels(data_one, data_zero)

        data_one.insert(0, 'label', labels_one)
        data_zero.insert(0, 'label', labels_zero)

        data_frame = pd.concat([data_one , data_zero])

        for i in range(3):
            data_frame = data_frame.sample(frac=1)

        return data_frame


class metrics(object):
    # module to get precision, recall and fscore over predictions
    def __init__(self):
        pass

    def get_prec_rec_fscore(self, y_true, y_pred, threshold=0.5):
        '''
        claculates the precision, recall, fscore and support
        of a binary output classifier
        '''
        binary_labels = [0, 1]
        y_pred = numpy.piecewise(y_pred, [y_pred <= threshold,
                                          y_pred > threshold],
                                 binary_labels)

        precision, recall, fscore, support = precision_recall_fscore_support(
                                                y_true, y_pred,
                                                average='binary')

        return precision, recall, fscore, support
