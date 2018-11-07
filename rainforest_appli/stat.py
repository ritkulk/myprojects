#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:05:17 2018

@author: rtwik
"""

import os
import numpy
from collections import Counter
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

fpath = os.getcwd()

f = open(fpath+'/rfapp.txt', 'r')
data = f.readlines()
f.close()

lens = []
words_all = []
for i in data:
    words = i.split()
    lens = lens + [len(w) for w in words]
    words_all = words_all + [w for w in words if w not in stops]

print(numpy.median(lens), numpy.mean(lens), numpy.std(lens))

counts = Counter(words_all)

print(counts.most_common(3))
