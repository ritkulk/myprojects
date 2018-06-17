#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:44:11 2018

@author: rtwik
"""
import os
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import pandas as pd

# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = pd.DataFrame()
    image_count = 0
    for name in os.listdir(directory + 'Flicker8k_Dataset'):
        # load an image from file
        filename = directory + '/Flicker8k_Dataset/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1],
                               image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        f_dict = {image_id: feature.reshape(4096)}

        print('>%s' % name)
        image_count += 1

        data = pd.DataFrame.from_dict(f_dict, orient='index')
        with open(directory + 'vgg_features/vgg_features.csv', 'a') as f:
            data.to_csv(f, header=None)

    print('Images Processed: ', image_count)


current_dir = os.getcwd()
img_data_dir = current_dir + '/data/images/'
img_features = extract_features(img_data_dir)
