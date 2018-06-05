#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code converts xml data to a dataframe and splits it into three groups
data_test, data_dev and data_train
@author: rtwik
"""

import xmltodict
import os
import pandas as pd


def xml_to_dataframe(xml_string):
    entry = xmltodict.parse(xml_string)
    dataframe = pd.DataFrame(entry)
    dataframe = dataframe.transpose()
    dataframe.reset_index(inplace=True)
    dataframe = dataframe.drop(columns=['index'])
    dataframe.set_index('@id', inplace=True)

    return dataframe


current_dir = os.getcwd()
data_dir = current_dir+'/data/'
filename = 'EFWritingData.xml'

# the following loop to extract data from xml. a brute force method wasbyby
# implemented as there were several invalid parses causing traditional
# approaches to fail
s = ''
append = False
c = 0
line_count = 0
with open(data_dir+filename, encoding='utf-8') as datafile:
    for i in datafile:
        if '<writing id' in i:
            append = True
        if append:
            s = s + i
        if '</writing>' in i:
            s = s.replace('<br/>', '')  # to parse xml into desired form
            try:
                dataframe = xml_to_dataframe(s)
                cols = list(dataframe.columns)
                with open(data_dir+'writing_data.csv', 'a') as f:
                    dataframe.to_csv(f, header=False)
                s = ''
                append = False
                c += 1

                if c % 10000 == 0:
                    print('done ', c)
                    print('line_count', line_count)

            except Exception:
                print('passed', line_count)
                pass
        # this number is decided from experiments. the xml file does not have
        # valid parses after that and the data acquired upto this point is
        # sufficient
        if line_count >= 4400000:
            print('DONE!! datapoints saved :', c)
            break
        line_count += 1

# saves header information in separate file.
with open(data_dir+'field_names.txt', 'w') as f:
    for i in cols:
        f.write(i+' ')

# read saved csv as a single dataframe
data = pd.read_csv(data_dir+'writing_data.csv', header=None)
data = data.drop(columns=[0])

# split the data into ~70% train and and ~10% dev and 20% test
ratio = 0.8
data_train_dev = data.iloc[:int(len(data) * ratio)]
data_test = data.iloc[int(len(data)*ratio):]

data_train = data_train_dev.iloc[:int(len(data_train_dev) * 0.9)]
data_dev = data_train_dev.iloc[int(len(data_train_dev) * 0.9):]

# save the splits as csv
data_train.to_csv(open(data_dir+'data_train.csv', 'w'), header=cols)
data_dev.to_csv(open(data_dir+'data_dev.csv', 'w'), header=cols)
data_test.to_csv(open(data_dir+'data_test.csv', 'w'), header=cols)
