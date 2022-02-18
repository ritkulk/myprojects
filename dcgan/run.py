#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:01:36 2021

@author: rtwik
"""

import json
import os
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import pylab as plt
import numpy as np
import torchvision.utils as vutils
from torchvision import transforms
from model_functions import ModelFunctions
from model_functions import Generator128
from model_functions import Discriminator128
from data_processor import DataProcessor, ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='json for required parameters', type=str, required=True)
parser.add_argument('--data_path', help='folder with training images', type=str, required=True)
parser.add_argument('--resume_trained', help='folder with training images', type=str, default='n', required=False)
parser.add_argument('--model_path', help='path to saved model', type=str, required=False)
parser.add_argument('--save_flag', help='flag to save model filename', type=str, required=False)
args = parser.parse_args()


with open(args.config) as f:
    config = json.load(f)
    
    
MF = ModelFunctions()
DP = DataProcessor()

device = torch.device("cuda:0" if (torch.cuda.is_available() and config['ngpu'] > 0) else "cpu")
# device = 'cpu'

generator = Generator128(config).to(device)
generator.apply(MF.weights_init)

discriminator = Discriminator128(config).to(device)
discriminator.apply(MF.weights_init)

loss_function = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, config['nz'], 1, 1, device=device)

optimiser_disc = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
optimiser_gen = optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

start_epoch = 0

if args.resume_trained == 'y':
    checkpoint = torch.load(args.model_path)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    discriminator.load_state_dict(checkpoint['disc_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    optimiser_gen.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    optimiser_disc.load_state_dict(checkpoint['disc_optimizer_state_dict'])
    
    print('chekpoint loaded.. resuming training from epoch {}'.format(start_epoch))


data_info = {'data_path': args.data_path, 'transforms': 'view'}
data = DP.make_dataframe_local(data_info)

input_size = config['image_size']
view_transform = DP.make_transform_pipeline(input_size=(input_size, input_size))
gan_transform = DP.make_transform_pipeline(input_size=(input_size, input_size), mode='gan')
transforms_dict = {'view_transform': view_transform, 'gan_transform': gan_transform}

dataset = ImageDataset(source='local', dataframe=data, transform_dict=transforms_dict, mode='gan')
data_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(start_epoch, start_epoch + config['num_epochs']):
    # For each batch in the dataloader
    for i, data in enumerate(data_loader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        disc_out_real, disc_out_fake , err_disc, label, fake_output = MF.train_discriminator(discriminator, generator, loss_function, optimiser_disc, optimiser_gen, data, device, config)
        disc_out_fake2, err_gen = MF.train_generator(generator, discriminator, optimiser_gen, loss_function, data, fake_output, device, config)
        
        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, start_epoch + config['num_epochs'], i, len(data_loader),
                      err_disc.item(), err_gen.item(), disc_out_real, disc_out_fake, disc_out_fake2))

        # Save Losses for plotting later
        G_losses.append(err_gen.item())
        D_losses.append(err_disc.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == config['num_epochs']-1) and (i == len(data_loader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    if epoch % 250 == 0:
        torch.save({
                    'epoch': epoch,
                    'gen_state_dict': generator.state_dict(),
                    'disc_state_dict': discriminator.state_dict(),
                    'gen_optimizer_state_dict': optimiser_gen.state_dict(),
                    'disc_optimizer_state_dict': optimiser_disc.state_dict(),
                    'gen_err': G_losses[-1],
                    'disc_err': D_losses[-1],
                    }, os.getcwd() + '/' + 'dcgan_ep' + str(epoch) + '_' + args.save_flag + '.pt')

torch.save({
            'epoch': epoch,
            'gen_state_dict': generator.state_dict(),
            'disc_state_dict': discriminator.state_dict(),
            'gen_optimizer_state_dict': optimiser_gen.state_dict(),
            'disc_optimizer_state_dict': optimiser_disc.state_dict(),
            'gen_err': G_losses[-1],
            'disc_err': D_losses[-1],
            }, os.getcwd() + '/' + 'dcgan_ep' + str(epoch) + '_' + args.save_flag + '.pt')

        
# Grab a batch of real images from the dataloader
real_batch = next(iter(data_loader))

# Plot the real images
f = plt.figure(figsize=(15,15))
ax1 = f.add_subplot(1,2,1)
ax1.axis("off")
ax1.set_title("Real Images")
ax1.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
ax2 = f.add_subplot(1,2,2)
ax2.axis("off")
ax2.set_title("Fake Images")
ax2.imshow(np.transpose(img_list[-1],(1,2,0)))

f.savefig(os.getcwd() + '/result_ep' + str(epoch) + '_' + args.save_flag + '.png')
# 
