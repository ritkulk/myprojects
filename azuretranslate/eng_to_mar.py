#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:09:02 2019

@author: rtwik
"""

import os
from googletrans import Translator
import os, requests, uuid, json

translator = Translator()

with open(os.getcwd() + '/book1.txt', 'r') as f:
    lines = f.readlines()

n_chars = 0
for l in lines:
    n_chars += len(l)
'''
print('loaded text')
translated = []
c = 0
for l in lines[10:11]:
    if len(l) > 0:
        try:
            translated.append(translator.translate(l, dest='mr').text)
            print(translator.translate(l, dest='mr').text)
            if c % 100 == 0 :
                print('{} % done'.format(c*100/len(lines)))
            c += 1

        except Exception:
            print('error')
    else:
        translated.append()
'''


subscription_key = '37e212e7c14c45968181480216a3cf23'
endpoint = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
params = '&to=hi'
constructed_url = endpoint + params
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}
body = [{
    'text': 'Physics is a great subject'
}]


endpoint2 = 'https://api.cognitive.microsofttranslator.com/languages?api-version=3.0'
constructed_url2 = endpoint2
headers2 = {
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}


request = requests.post(constructed_url, headers=headers, json=body)
request2 = requests.get(constructed_url2, headers=headers2)

response = request.json()

print(response[0]['translations'])


