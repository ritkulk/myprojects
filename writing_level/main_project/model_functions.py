#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code has all the necessary functions to execute all model related
functions from build, train to predict and more
@author: rtwk
"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support


class model_functions(object):
    def __init__(self):
        pass

    def compile_model(self, params):
        # builds a simple 2 layer feedforward model
        n_features = params['n_features']
        n_outputs = len(params['output_classes'])

        model = Sequential()
        model.add(Dense(20, input_dim=n_features, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_outputs, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        return model

    def train_model(self, model, train_data_gen, dev_data_gen, params):
        # trains the model with data generator and saves best model based on
        # validation accuracy
        n_chunks_train = params['n_chunks_train']
        n_chunks_dev = params['n_chunks_dev']
        epochs = params['epochs']
        model_filepath = params['model_filepath']

        CP = ModelCheckpoint(model_filepath,
                             monitor='val_categorical_accuracy',
                             verbose=1, save_best_only=True, mode='auto')

        model.fit_generator(train_data_gen, steps_per_epoch=n_chunks_train,
                            epochs=epochs,
                            validation_data=dev_data_gen,
                            validation_steps=n_chunks_dev,
                            callbacks=[CP])

        return model

    def load_saved_model(self, model_filepath):
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

    def get_F_metrics(self, targets, predictions):
        # calculates precision, recall and F score over predictions
        # returns micro averaged values of multi-class labels
        # equivalent to global accuracy
        y_pred = (predictions == predictions.max(axis=1)[:, None]).astype(int)
        y_true = targets.astype(int)

        precision, recall, fscore, sup = precision_recall_fscore_support(y_true, 
                                                             y_pred,
                                                             average='macro')

        return precision, recall, fscore, sup
