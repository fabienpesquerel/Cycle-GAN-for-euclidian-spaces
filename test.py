#!/usr/bin/python3

import argparse
import sys
import os
import csv
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import visdom

from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int,
                    default=100, help='size of the batches')
parser.add_argument('--dataroot', type=str,
                    default='datasets/gaussian2gaussian/',
                    help='root directory of the dataset')
parser.add_argument('--in_layer', type=int, default=2,
                    help='number of input features')
parser.add_argument('--out_layer', type=int, default=2,
                    help='number of output features')
parser.add_argument('--hidden_layers', type=int, default=1,
                    help='number of hidden layers in the network, default=1')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4,
                    help='number of cpu threads to use during \
batch generation')
parser.add_argument('--generator_A2B', type=str,
                    default='output/gaussian2gaussian/netG_A2B.pth',
                    help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str,
                    default='output/gaussian2gaussian/netG_B2A.pth',
                    help='B2A generator checkpoint file')
parser.add_argument('--plot', action='store_true',
                    help='plot after generation')
parser.add_argument('--output', type=str, default='output/gaussian2gaussian/',
                    help='output directory of generated samples')
opt = parser.parse_args()
print(opt)

"""
   _____ _       _           _  __      __        _       _     _
  / ____| |     | |         | | \ \    / /       (_)     | |   | |
 | |  __| | ___ | |__   __ _| |  \ \  / __ _ _ __ _  __ _| |__ | | ___ ___
 | | |_ | |/ _ \| '_ \ / _` | |   \ \/ / _` | '__| |/ _` | '_ \| |/ _ / __|
 | |__| | | (_) | |_) | (_| | |    \  | (_| | |  | | (_| | |_) | |  __\__ \
  \_____|_|\___/|_.__/ \__,_|_|     \/ \__,_|_|  |_|\__,_|_.__/|_|\___|___/
"""

# Networks
netG_A2B = Generator(opt.in_layer, opt.out_layer, opt.hidden_layers)
netG_B2A = Generator(opt.out_layer, opt.in_layer, opt.hidden_layers)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state from the last checkpoint
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set networks to test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs and targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.in_layer)
input_B = Tensor(opt.batchSize, opt.out_layer)

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
    def __init__(self, root, mode='test'):
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

"""
  _______       _   _
 |__   __|     | | (_)
    | | ___ ___| |_ _ _ __   __ _
    | |/ _ / __| __| | '_ \ / _` |
    | |  __\__ | |_| | | | | (_| |
    |_|\___|___/\__|_|_| |_|\__, |
                             __/ |
                            |___/
"""

# Create output dirs if they don't exist
if not os.path.exists('%sA' % opt.output):
    os.makedirs('%sA' % opt.output)
if not os.path.exists('%sB' % opt.output):
    os.makedirs('%sB' % opt.output)

Bs = []
As = []
RAs = []
RBs = []
rebuild_A = []
rebuild_B = []

for i, batch in enumerate(dataloader):
    real_A = input_A.copy_(batch['A'])
    real_B = input_B.copy_(batch['B'])
    real_A.requires_grad = False
    real_B.requires_grad = False
    fake_B = netG_A2B(real_A)
    fake_A = netG_B2A(real_B)
    reconstruction_A = netG_B2A(fake_B).detach().numpy()
    reconstruction_B = netG_A2B(fake_A).detach().numpy()
    fake_A = fake_A.detach().numpy()
    fake_B = fake_B.detach().numpy()

    Bs.append(fake_B)
    As.append(fake_A)
    RAs.append(real_A.detach().numpy())
    RBs.append(real_B.detach().numpy())
    rebuild_A.append(reconstruction_A)
    rebuild_B.append(reconstruction_B)

    sys.stdout.write('\rGenerated batch number %04d' % (i+1))

Bs = np.array(Bs).reshape(-1, 2)
As = np.array(As).reshape(-1, 2)
RAs = np.array(RAs).reshape(-1, 2)
RBs = np.array(RBs).reshape(-1, 2)
rebuild_A = np.array(rebuild_A).reshape(-1, 2)
rebuild_B = np.array(rebuild_B).reshape(-1, 2)

# Storing


def write_csv(path, data):
    np.savetxt(path, data, delimiter=",")


write_csv('%sA/fake.csv' % opt.output, As)
write_csv('%sB/fake.csv' % opt.output, Bs)

# Plot

if opt.plot:
    vis = visdom.Visdom()
    # A
    vis.scatter(As[:200],
                opts={'title': 'Generated A ' + str(As.shape)})
    vis.scatter(RAs[:200],
                opts={'title': 'Real A ' + str(RAs[:200].shape)})
    vis.scatter(rebuild_A[:200],
                opts={'title': 'Reconstruct A ' + str(rebuild_A[:200].shape)})
    # B
    vis.scatter(Bs[:200], opts={'title': 'Generated B'})
    vis.scatter(RBs[:200], opts={'title': 'Real B'})
    vis.scatter(rebuild_B[:200], opts={'title': 'Reconstruct B'})

    plt.scatter(Bs[:300, 0], Bs[:300, 1]) #, marker='+')
    plt.scatter(RBs[:300, 0], RBs[:300, 1]) #, marker='+')
    plt.scatter(rebuild_B[:300, 0], rebuild_B[:300, 1]) #, marker='+')
    # plt.show()
    plt.scatter(As[:300, 0], As[:300, 1], marker='+')
    plt.scatter(RAs[:300, 0], RAs[:300, 1], marker='+')
    plt.scatter(rebuild_A[:300, 0], rebuild_A[:300, 1], marker='+')
    plt.show()
