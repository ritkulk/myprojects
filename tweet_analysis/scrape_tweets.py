#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:02:48 2018

@author: rtwik
"""

import tweepy
import pandas as pd
import os
import wget
from urllib.request import urlopen
from os.path import basename
from urllib.parse import urlsplit
import re

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data/'
DATA_FILENAME = 'tweets.csv'
IMG_DIR = '/mnt/mydisk/wdat/data_images/tweet_images/'


####input your credentials here
consumer_key = 'V1XlezzNHQ1yyNFVxMET8nqzy'
consumer_secret = 'phSCZsXLyaH4PauYDI8mAYpV2780VBxMnQJSxWCVXllhQ5TF2c'
access_token = '1059141396697878529-aOtVjOuv3fB0o3rBXuJD4rnVKqUPo4'
access_token_secret = 'ENpzCzJ4DScRxRRyOxtCiy32ohGH9lrkPhYzAdO4ungNU'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

media_files = []

print('fetching tweets')
tweet_id = 0
tweet_data = []
for tweet in tweepy.Cursor(api.search,q="pangolin",count=1000,
                           lang="en",
                           since="2017-01-01").items():

    media = tweet.entities.get('media', [])
    img_name = 'none'
    if(len(media) > 0):
        if media[0]['type'] == 'photo':
            media_files.append([tweet_id, media[0]['media_url']])
            img_name = media[0]['media_url'].split('/')[-1]

    print (tweet.created_at, tweet.text, img_name)
    tweet_data.append([tweet_id, tweet.created_at, tweet.text, img_name])

    tweet_id += 1

tweet_data = pd.DataFrame(tweet_data)
tweet_data.columns = ['id', 'time', 'text', 'image_name']

print('downloading attached images')
seen_urls = set()
for media_file in media_files:
    if media_file[1] not in seen_urls:
        wget.download(media_file[1], out=IMG_DIR)
        seen_urls.add(media_file[1])


urls = []
for _, row in tweet_data.iterrows():
    if 'https' in row['text']:
        t = row['text'].split()
        urls = urls + [[row['id'], i] for i in t if 'https' in i[:5]]


print('downloading ref url images')
c = 0
seen_files = set()
for u in urls:
    u_id = u[0]
    url = u[1]
    try:
        urlContent = str(urlopen(url).read())
        # HTML image tag: <img src="url" alt="some_text"/>
        imgUrls = re.findall('img .*?src="(.*?)"', urlContent)

        # download all images
        for imgUrl in imgUrls:
            try:
                imgData = urlopen(imgUrl).read()
                fileName = basename(urlsplit(imgUrl)[2])
                ext = fileName.split('.')[-1]
                if 'jpg' in ext or 'png' in ext:
                    if 'icon' not in fileName and 'logo' not in fileName and fileName not in seen_files:
                        output = open(IMG_DIR + fileName,'wb')
                        output.write(imgData)
                        output.close()
                        seen_files.add(fileName)

                        tweet_index = tweet_data.index[tweet_data['id'] == u_id].tolist()[0]
                        tweet_data.at[tweet_index, 'image_name'] = tweet_data.at[tweet_index, 'image_name'] \
                                                                   + '---' + fileName


            except Exception:
                pass
    except Exception:
        pass
    c += 1
    if c % 10 == 0:
        print('done', c/len(urls))

tweet_data.to_csv(DATA_DIR + DATA_FILENAME, index=False)
