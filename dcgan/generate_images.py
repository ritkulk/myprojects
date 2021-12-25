#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:06:17 2021

@author: rtwik
"""
import json
import os
import argparse
from torch.utils.data import DataLoader
import torch
import pylab as plt
import numpy as np
import torchvision.utils as vutils
from model_functions import ModelFunctions
from model_functions import Generator
from data_processor import DataProcessor, ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='json for required parameters', type=str, required=True)
parser.add_argument('--n_images', help='number of images to generate', type=int, required=True)
parser.add_argument('--model_path', help='path to saved model', type=str, required=False)
parser.add_argument('--save_path', help='path to saved result image', type=str, required=False)
parser.add_argument('--save_flag', help='json for required parameters', type=str, required=True)
args = parser.parse_args()


with open(args.config) as f:
    config = json.load(f)
    
    
MF = ModelFunctions()

device = torch.device("cuda:0" if (torch.cuda.is_available() and config['ngpu'] > 0) else "cpu")
# device = 'cpu'

generator = Generator(config).to(device)
checkpoint = torch.load(args.model_path)
generator.load_state_dict(checkpoint['gen_state_dict'])

# MF.generate_images(generator, args.n_images, args.save_flag, config, device)


fixed_noise = torch.randn(args.n_images, config['nz'], 1, 1, device=device)

output = generator(fixed_noise).detach().cpu()
result = vutils.make_grid(output, padding=2, normalize=True)

f1 = plt.figure(1, figsize=[10,10], dpi=300)
ax1 = f1.add_subplot(111)
ax1.axis("off")
ax1.set_title('Generator Output')
ax1.imshow(np.transpose(result,(1,2,0)))

save_filename = os.getcwd() + '/result_' + str(args.save_flag) + '.png'
f1.savefig(save_filename)
print('Image result saved to {}'.format(save_filename))