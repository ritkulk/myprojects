#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 20:24:36 2021

@author: rtwik
"""
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


class ModelFunctions(object):
    
    def __init__(self):
        pass
    
    # custom weights initialization called on netG and discriminator
    def weights_init(self, layer):
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)
            
            
    def train_discriminator(self, discriminator, generator, loss_function, optimiser_disc, optimiser_gen,
                            data, device, config):
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), config['real_label'], dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = discriminator(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = loss_function(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        disc_out_real = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, config['nz'], 1, 1, device=device)
        # Generate fake image batch with G
        fake_output = generator(noise)
        label.fill_(config['fake_label'])
        # Classify all fake batch with D
        output = discriminator(fake_output.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = loss_function(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        disc_out_fake = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        err_disc = errD_real + errD_fake
        # Update D
        optimiser_disc.step()
        
        return disc_out_real, disc_out_fake , err_disc, label, fake_output
        
        

    def train_generator(self, generator, discriminator, optimiser_gen, loss_function, data,
                        fake_output, device, config):
        
        generator.zero_grad()
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), config['real_label'], dtype=torch.float, device=device)

        # TODO: check label fill
        # label.fill_(config['real_label'])  # fake labels are real for generator cost 
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake_output).view(-1)
        # Calculate G's loss based on this output
        errG = loss_function(output, label)
        # Calculate gradients for G
        errG.backward()
        disc_out_fake2 = output.mean().item()
        # Update G
        optimiser_gen.step()
        
        return disc_out_fake2, errG

class Generator(nn.Module):
    
    def __init__(self, config):
        super(Generator, self).__init__()
        self.ngpu = config['ngpu']
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( config['nz'], config['ngf'] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config['ngf'] * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config['ngf'] * 8, config['ngf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['ngf'] * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( config['ngf'] * 4, config['ngf'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['ngf'] * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( config['ngf'] * 2, config['ngf'], 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['ngf']),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( config['ngf'], config['nc'], 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64  x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.ngpu = config['ngpu']
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(config['nc'], config['ndf'], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(config['ndf'], config['ndf'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(config['ndf'] * 2, config['ndf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(config['ndf'] * 4, config['ndf'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config['ndf'] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(config['ndf'] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class ImageModels(ModelFunctions):
    
    def __init__(self, model_path):
        os.environ['TORCH_HOME'] = model_path

