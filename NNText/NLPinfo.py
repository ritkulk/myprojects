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

markers=['a','an','every','each','many','few','much','lot', 'little']

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


f = open(dir_path + '/data/POStags.csv','r')
tags=[i.strip('\n') for i in f.readlines()]
f.close()

pairs = []
clean_sents =[] 
bare =[]
mass = []
c=0
for i in tag_sent:
    nouns = 0
    record = False
    pos=0
    nn = []
    mr=[]
    for j in i:
        if 'NN' in j[1]:
            nouns = nouns + 1     
            nn.append(j[0])
            pos_n = pos
        if j[0] in markers:            
           record = True
           mr.append(j[0])
           pos_m = pos
        if pos ==0 and j[1]=='NN':
            bare.append(j[0])
            
        if j[0]=='little' and i[pos-1][0]=='a' and i[pos+1][0]=='of':
            posf = pos+1
            stop = False
            while stop ==False:
                if posf+1 < len(i): posf = posf+1
                else: stop=True
                if i[posf][1] == 'NN':
                    c=c+1
                    stop = True
                    mass.append(i[posf][0])
            
        pos = pos +1
    if record == True and nouns < 2 and nn != None:
       clean_sents.append(i)
       pairs.append([mr,nn])
        