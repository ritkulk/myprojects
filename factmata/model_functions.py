#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 04:45:33 2018

@author: rtwik
"""

import numpy
from keras.models import Sequential, load_model, Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout, Bidirectional, Input, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate

# fix random seed for reproducibility
numpy.random.seed(7)


class Models(object):
    def __init__(self):
        pass
    
    def compile_conv_bilstm_model(self,params):
        sent_maxlen = params['sent_maxlen']
        vocab_size = params['vocab_size']
        
        model = None
        model = Sequential()
        model.add(Embedding(input_dim = vocab_size +1, output_dim = 128, 
                            input_length=sent_maxlen))
        model.add(Conv1D(filters=1000, kernel_size=1, padding='same',
                         activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
#        model.add(Bidirectional(LSTM(64,dropout=0.5,recurrent_dropout=0.5)))
        model.add(LSTM(128,dropout=0.5,recurrent_dropout=0.5))

        model.add(Dropout(0.5))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', 
                      metrics=['accuracy'])
    
        return model
    
    
    def compile_multi_cnn(self, params):
        sent_maxlen = params['sent_maxlen']
        vocab_size = params['vocab_size']
        # channel 1
        inputs1 = Input(shape=(sent_maxlen,))
        embedding1 = Embedding(vocab_size, 100)(inputs1)
        conv1 = Conv1D(filters=640, kernel_size=1, activation='relu',
                       padding='same')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        inputs2 = Input(shape=(sent_maxlen,))
        embedding2 = Embedding(vocab_size, 100)(inputs2)
        conv2 = Conv1D(filters=640, kernel_size=3, activation='relu',
                       padding='same')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # channel 3
        inputs3 = Input(shape=(sent_maxlen,))
        embedding3 = Embedding(vocab_size, 100)(inputs3)
        conv3 = Conv1D(filters=640, kernel_size=5, activation='relu',
                       padding='same')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(100, activation='relu')(merged)
        dense1 = Dropout(0.5)(dense1)
        outputs = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # compile
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
    
    def train_model(self, model, data_gen_train, data_gen_test,params):
    
        filepath = params['models_dir']+params['model_name']
        CP=ModelCheckpoint(filepath, monitor='val_acc', 
                               verbose=1, save_best_only=True, mode='auto')
        
        
        train_steps = params['train_steps']
        eval_steps = params['eval_steps']
        
        model.fit_generator(data_gen_train,steps_per_epoch=train_steps,
                            validation_data = data_gen_test,
                            validation_steps=eval_steps,
                            epochs=20,callbacks=[CP])
        
        return model

    
    def evaluate_model(self,model,data_gen,params):
        print('Evaluating model')
        eval_steps = params['eval_steps']

        scores = model.evaluate_generator(data_gen, steps=eval_steps)
        return scores

    def load_model(self,model_name,params):
        print('Loading model')
        filepath = params['models_dir']+model_name
        model = load_model(filepath)
        
        return model
