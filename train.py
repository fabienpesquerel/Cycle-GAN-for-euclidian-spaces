#!/usr/bin/python3

import os
import glob

import csv
import numpy as np

import argparse
import itertools

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=400,
                    help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=100, help='size of the\
 batches')
parser.add_argument('--dataroot', type=str,
                    default='datasets/gaussian2gaussian/',
                    help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.004, help='initial\
 learning rate')
parser.add_argument('--decay_epoch', type=int, default=290,
                    help='epoch to start linearly decaying the\
 learning rate to 0')
parser.add_argument('--hidden_layers', type=int, default=1,
                    help='number of hidden layers, default=1')
parser.add_argument('--in_layer', type=int, default=2,
                    help='number of features of input data')
parser.add_argument('--out_layer', type=int, default=2,
                    help='number of features of output data')
parser.add_argument('--cuda', action='store_true',
                    help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4,
                    help='number of cpu threads to use during\
 batch generation')
parser.add_argument('--output', type=str, default='output/gaussian2gaussian/',
                    help='directory in which to store outputs')

opt = parser.parse_args()

print(opt)

"""
   _____ _       _           _  __      __        _       _     _
  / ____| |     | |         | | \ \    / /       (_)     | |   | |
 | |  __| | ___ | |__   __ _| |  \ \  / /_ _ _ __ _  __ _| |__ | | ___  ___
 | | |_ | |/ _ \| '_ \ / _` | |   \ \/ / _` | '__| |/ _` | '_ \| |/ _ \/ __|
 | |__| | | (_) | |_) | (_| | |    \  / (_| | |  | | (_| | |_) | |  __/\__ \
  \_____|_|\___/|_.__/ \__,_|_|     \/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/
"""

# NETWORKS

netG_A2B = Generator(opt.in_layer, opt.out_layer, opt.hidden_layers)
netG_B2A = Generator(opt.out_layer, opt.in_layer, opt.hidden_layers)
netD_A = Discriminator(opt.in_layer, opt.hidden_layers)
netD_B = Discriminator(opt.out_layer, opt.hidden_layers)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Losses

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.MSELoss()

# Optimizers and LR scheduler/policy

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),
                                               netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))

optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr,
                                 betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr,
                                 betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G,
    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A,
    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B,
    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs and targets memory allocation

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.in_layer)
input_B = Tensor(opt.batchSize, opt.out_layer)
print(input_A.size())
target_real = [[1.0, 0.] for _ in range(opt.batchSize)]
target_real = Tensor(target_real)
print('target_real.size():', target_real.size())
target_real.requires_grad = False
target_fake = [[0., 1.0] for _ in range(opt.batchSize)]
target_fake = Tensor(target_fake)
target_fake.requires_grad = False

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader


def load_csv(path):
    data = []
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            data.append(np.array(row))
        data = np.array(data).astype(float)
        data = torch.from_numpy(data)
        data.float()
        return data


class SetDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.files_A = sorted(glob.glob(os.path.join(root,
                                                     '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root,
                                                     '%s/B' % mode) + '/*.*'))
        self.item_A = load_csv(self.files_A[0])
        self.item_B = load_csv(self.files_B[0])

    def __getitem__(self, index):
        item_A = self.item_A[index % len(self.item_A)]
        item_B = self.item_B[index % len(self.item_B)]
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.item_A), len(self.item_B))


dataloader = DataLoader(SetDataset(opt.dataroot),
                        batch_size=opt.batchSize,
                        shuffle=True, num_workers=opt.n_cpu)
print("len(dataloader):", len(dataloader))
# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))

"""
  _______        _       _
 |__   __|      (_)     (_)
    | |_ __ __ _ _ _ __  _ _ __   __ _
    | | '__/ _` | | '_ \| | '_ \ / _` |
    | | | | (_| | | | | | | | | | (_| |
    |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
                                  __/ |
                                 |___/
"""
lambda1 = 60.
lambda2 = 6.
lambda3 = 55.
l1 = 7.
l2 = 0.1
l3 = 5.
r1 = (l1 - lambda1) / (opt.n_epochs - opt.epoch)
r2 = (l2 - lambda2) / (opt.n_epochs - opt.epoch)
r3 = (l3 - lambda3) / (opt.n_epochs - opt.epoch)

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        print(batch['A'].size())
        g = input_A.copy_(batch['A'])
        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])

        ######################
        ##    GENERATORS    ##
        ######################

        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # Only for images
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*0.0

        # G_B2A(A) should equal A if real A is fed
        # Only for images
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*0.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)*lambda1

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)*lambda1

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*lambda2

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*lambda2

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B
        loss_G += loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        loss_G.backward()
        optimizer_G.step()

        ######################
        ##  DISCRIMINATORS  ##
        ######################

        # Discriminator A
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*lambda3
        loss_D_A.backward()

        optimizer_D_A.step()

        # Discriminator B
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*lambda3
        loss_D_B.backward()

        optimizer_D_B.step()

        # Progress report on http://localhost:8097
        logger.log({'loss_G': loss_G,
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D': (loss_D_A + loss_D_B)})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), '%snetG_A2B.pth' % opt.output)
    torch.save(netG_B2A.state_dict(), '%snetG_B2A.pth' % opt.output)
    torch.save(netD_A.state_dict(), '%snetD_A.pth' % opt.output)
    torch.save(netD_B.state_dict(), '%snetD_B.pth' % opt.output)

    # update lambdas
    lambda1 += r1
    lambda2 += r2
    lambda3 += r3
