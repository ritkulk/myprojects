# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:00:39 2017

@author: rtwik
"""

import nltk
from nltk.corpus import brown



tag_sent = brown.tagged_sents()

verbtags = ['VB','VBD','VBG','VBN','VBP','VBZ', 'CD',]
        
#markers=['a','an','numerl','every','each','many','few','much','little','a lot of','numeral +pl','sing+clas']

markers=['a','an','every','each','many','few','much','little']

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


f = open(dir_path + '/data/POStags.csv','r')
tags=[i.strip('\n') for i in f.readlines()]
f.close()

clean_sents =[] 
for i in tag_sent:
    sents = []
    nouns = 0
    record = False
    for j in i:
        if j[1] in ['NN','NNS']:
            nouns = nouns + 1     
        if j[0] in markers:            
           record = True
    if record == True and nouns < 2:
       clean_sents.append(i)
        