#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:22:17 2022

@author: rtwik
"""
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
from torchvision import transforms
import torch
import pylab as plt
import numpy as np
import torchvision.utils as vutils
from model_functions import ModelFunctions
from model_functions import Generator128
from data_processor import DataProcessor, ImageDataset
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='json for required parameters', type=str, required=True)
parser.add_argument('--grid_size', help='n for an n x n image grid', type=int, required=True)
parser.add_argument('--n_tiles', help='number of tiles', type=int, required=True)
parser.add_argument('--model_path', help='path to saved model', type=str, required=False)
parser.add_argument('--save_path', help='path to saved result image', type=str, required=False)
parser.add_argument('--save_flag', help='json for required parameters', type=str, required=True)
args = parser.parse_args()


if args.grid_size % 2 != 0:
    print('grid size must be a multiple of 2')
    SystemExit(0)

with open(args.config) as f:
    config = json.load(f)
    
device = torch.device("cuda:0" if (torch.cuda.is_available() and config['ngpu'] > 0) else "cpu")
# device = 'cpu'

generator = Generator128(config).to(device)
checkpoint = torch.load(args.model_path)
generator.load_state_dict(checkpoint['gen_state_dict'])

transImg = transforms.ToPILImage()
transPIL = transforms.PILToTensor()

n_images = args.grid_size * args.grid_size * args.n_tiles
batch_size = 128
if int(n_images/batch_size) < 1:
    n_batch = 1
else:
    n_batch = int(n_images/batch_size)

n_fig = 1
for n in range(n_batch):
    
    fixed_noise = torch.randn(batch_size, config['nz'], 1, 1, device=device)
    results = generator(fixed_noise).detach().cpu()
    
    inds = [i for i in range(batch_size)]
    np.random.shuffle(inds)
    
    pairs = [i for i in zip(inds[0:int(len(inds)/2)], inds[int(len(inds)/2):])]
    
    binary  = [1]*int(len(pairs)*0.9) + [0]*(len(pairs) - int(len(pairs)*0.9))
    np.random.shuffle(binary)
    
    for nt in range(int(args.n_tiles/n_batch)):
        np.random.shuffle(pairs)
        
      
        blended_images = []
        for p in pairs[:args.grid_size * args.grid_size]:
        
            img1 = results[p[0]]
            img2 = results[p[1]]
            img3 = results[np.random.randint(0, len(inds))]
            
            a = np.random.rand()
            img_b = img1 * a + img2 * (1-a) + 0.3 * img3
            
            img_b -= img_b.min()
            img_b = img_b/img_b.max()
            
            blended_images.append(transImg(img_b))
        
        f1 = plt.figure(n_fig, figsize=[3,3], dpi=300, facecolor='black',)
        c = 1
        for i in range(args.grid_size * args.grid_size):
            ax1 = f1.add_subplot(args.grid_size, args.grid_size, c )
            ax1.axis("off")
            ax1.imshow(blended_images[c-1])
            c +=1
        f1.subplots_adjust(hspace=0.05, wspace=0.05)
        n_fig += 1
      
        
        # ax1.set_title('Generator Output')
        
        # f1 = plt.figure(1, figsize=[10,10], dpi=300)
        # ax1 = f1.add_subplot(111)
        # ax1.axis("off")
        # ax1.set_title('Generator Output')
        # ax1.imshow(np.transpose(result,(1,2,0)))
        
    
    # save_filename = os.getcwd() + '/result_' + str(args.save_flag) + '.png'
    # f1.savefig(save_filename)
    # print('Image result saved to {}'.format(save_filename))
