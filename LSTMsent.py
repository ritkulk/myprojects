# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 03:43:31 2017

@author: rtwik
"""

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
import os

CURRENT_DIR = os.getcwd()
MODELS_DIR = CURRENT_DIR + '/instagram_analysis/models/'
FILEPATH = MODELS_DIR+'sentiment_basic.h5'
# fix random seed for reproducibility
numpy.random.seed(1)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)

word_index = imdb.get_word_index()

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 64
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=64, kernel_size=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

CP = ModelCheckpoint(FILEPATH, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='auto')

model.fit(X_train, y_train,
          validation_data=(X_test, Y_test),
          epochs=3, batch_size=50, callbacks=[CP])

predictions = model.predict(X_test)

#model.save(FILEPATH)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
