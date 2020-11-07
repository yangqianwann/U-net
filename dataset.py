#!/usr/bin/env deeplearning
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:54:42 2020

@author: yangqianwan
"""

import numpy as np
from torch.utils import data
import torch.nn

class MyDataset(data.Dataset):
    def __init__(self, transform=None):
        #self.input_images = np.load('/home/amax/data/yqw/segmentation/dataset/data.npy')
        self.input_images = np.load('/home/yqw/neuron/outfile.npy')
        #self.input_images = self.input_images[0:66,:,:]
        #self.mask = np.load('/home/amax/data/yqw/segmentation/dataset/masks.npy')
        self.mask =np.load('/home/amax/data/yqw/neuron/labels.npy')
        #self.mask = self.mask[0:66, :, :]
        self.mask=self.mask.astype(int)
        #self.labels =np.load('/home/amax/data/yqw/segmentation/dataset/labels.npy')
        self.transform = transform
        self.n_class   = 2
        self.image=torch.from_numpy(np.expand_dims(np.array(self.input_images),axis=0)).float()
        self.mask = torch.from_numpy(np.array(self.mask)).long()
    def __getitem__(self,index):
        image, mask = self.image[:,index,:,:], self.mask[index]
        #print('image',image.shape)
        # create one-hot encoding
        h, w = mask.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][mask == c] = 1
        sample = {'X': image, 'Y': target, 'l': mask}
        #print('target',target.shape)
        return sample
    def __len__(self):
        return len(self.input_images)

