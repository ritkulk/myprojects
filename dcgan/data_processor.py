#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:10:16 2021

@author: rtwik
"""

import boto3
import io
from PIL import Image
import time
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
import pandas as pd
import os

class ImageDataset(Dataset):
    def __init__(self, source, dataframe, transform_dict,  s3_resource=None, mode = ''):
        self.data = dataframe
        self.source = source
        
        if self.source == 's3':
            self.s3_resource = s3_resource
            
        self.transform_dict = transform_dict
        self.mode = mode
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        sample = self.data.iloc[index]
        # Read an image with PIL
        
        if self.source == 's3':
            bucket = self.s3_resource.Bucket(sample['bucket_name'])
            file_stream = io.BytesIO()
            bucket.download_fileobj(sample['key'], file_stream)
            image = Image.open(file_stream)

        elif self.source == 'local':
            image =  Image.open(sample['filename'])
            
        
        if len(image.getbands()) != 3:
            image = image.convert('RGB')
        
        if sample['transform_method'] and self.mode not in ['val', 'predict', 'view', 'gan']:
            image = self.transform_dict[sample['transform_method']](image)
        elif sample['transform_method'] and self.mode in ['val']:
            image = self.transform_dict['val_transform'](image)
        elif sample['transform_method'] and self.mode in ['view']:
            image = self.transform_dict['view_transform'](image)
        elif sample['transform_method'] and self.mode in ['gan']:
            image = self.transform_dict['gan_transform'](image)
               
        return image
    
    
class DataProcessor(object):
    def __init__(self):
        pass

    def make_dataframe_s3(self, bucket_list, shuffle=True):
      dataframes = []
      for info in bucket_list:
          dataframes.append(pd.DataFrame([info['keys'], 
                              [info['bucket_name']]*len(info['keys']), 
                              [info['transforms']]*len(info['keys']),
                              [info['label']]*len(info['keys'])]).transpose())
          
      data = pd.concat(dataframes, ignore_index=True)
      data.columns = ['key', 'bucket_name', 'transform_method', 'label']   
      
      if shuffle:
          data = data.sample(frac=1, random_state=123)
          
      return data
  
    def make_dataframe_local(self, data_info, shuffle=True):
        
        filenames = [data_info['data_path'] + '/' +f for f in os.listdir(data_info['data_path']) \
                     if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]

        data = pd.DataFrame([filenames, list([data_info['transforms']]*len(filenames))])
        data = data.transpose()        
        data.columns = ['filename', 'transform_method']
        
        if shuffle:
            data = data.sample(frac=1, random_state=123)
            
        return data

    def get_keys(self, s3_resource, bucket_name='', ext_filter = []):
        bucket = s3_resource.Bucket(bucket_name)
        files_in_bucket = list(bucket.objects.all())
        
        return [f.key for f in files_in_bucket if f.key.split('.')[-1].lower() in ext_filter]
    
    def make_train_test_split(self, data, train_ratio):
        data_train = data[:int(len(data)*train_ratio)]
        data_val = data[int(len(data)*train_ratio):]
        
        return data_train, data_val
    
        
    def make_transform_pipeline(self, input_size, mode='view'):
        
        if mode == 'exotic':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.CenterCrop(size=input_size[0]*0.85),
                transforms.RandomCrop(input_size[0]*0.85),
                transforms.RandomApply(transforms=[transforms.RandomCrop(size=(input_size[0]*0.78))], p=0.3),
                transforms.RandomApply(transforms=[transforms.transforms.RandomHorizontalFlip()], p=0.5),
                transforms.RandomApply(transforms=[transforms.transforms.RandomRotation(degrees=(0, 180))], p=0.5),
                transforms.Resize(input_size), 
        
        ])
        
        elif mode == 'wild':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomCrop(input_size[0]*0.85),
                transforms.RandomApply(transforms=[transforms.RandomCrop(size=(input_size[0]*0.80))], p=0.3),
                transforms.RandomApply(transforms=[transforms.transforms.RandomHorizontalFlip()], p=0.5),
                transforms.RandomApply(transforms=[transforms.transforms.RandomRotation(degrees=(0, 180))], p=0.5),
                transforms.Resize(input_size), 
            ])
            
        elif mode == 'background':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomApply(transforms=[transforms.RandomCrop(size=(input_size[0]*0.80))], p=0.3),
                transforms.RandomApply(transforms=[transforms.transforms.RandomHorizontalFlip()], p=0.5),
                transforms.RandomApply(transforms=[transforms.transforms.RandomRotation(degrees=(0, 180))], p=0.5),
                transforms.Resize(input_size), 
            ])
        
        elif mode == 'gan':
            transform=transforms.Compose([
                               transforms.Resize(input_size),
                               transforms.CenterCrop(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    
        elif mode == 'val':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Resize(input_size)
            ])
            
        elif mode == 'view':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size)
            ])
        else:
            print('mode must be in [exotic, wild, val, view, gan]')
            raise(SystemExit(0))
            
        return transform



