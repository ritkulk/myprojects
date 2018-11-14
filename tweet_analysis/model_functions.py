#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains methods for all model related functions, compiling, training, predictind
and testing
@author: rtwik
"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support


class model_functions(object):
    def __init__(self):
        pass

    def compile_model(self, params):
        ''' builds a simple 3 layer feedforward model'''
        n_features = params['n_features']
        n_outputs = 1

        model = Sequential()
        model.add(Dense(2000, input_dim=n_features, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_outputs, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['binary_accuracy', 'mse'])

        return model

    def train_model(self, model, train_data_gen, dev_data_gen, params):
        ''' trains the model with data generator and saves best model based on
            validation accuracy '''
        n_chunks_train = params['n_chunks_train']
        n_chunks_test = params['n_chunks_test']
        epochs = params['epochs']
        model_filepath = params['model_filepath']

        CP = ModelCheckpoint(model_filepath,
                             monitor='val_binary_accuracy',
                             verbose=1, save_best_only=True, mode='auto')

        model.fit_generator(train_data_gen, steps_per_epoch=n_chunks_train,
                            epochs=epochs,
                            validation_data=dev_data_gen,
                            validation_steps=n_chunks_test,
                            callbacks=[CP])

        return model

    def load_saved_model(self, model_filepath):
        print('loading model {}'.format(model_filepath))

        model = load_model(model_filepath)
        return model

    def evaluate_model(self, model, test_data_gen, params):
        n_chunks_test = params['n_chunks_test']

        score = model.evaluate_generator(test_data_gen, steps=n_chunks_test)

        return score

    def get_predictions(self, model, test_data_gen, params):
        n_chunks_test = params['n_chunks_test']
        predictions = model.predict_generator(test_data_gen,
                                              steps=n_chunks_test)

        return predictions

