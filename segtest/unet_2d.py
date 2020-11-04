#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:03:48 2020

@author: yangqianwan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#def initialize_weights(*models):
#    for model in models:
#        for module in model.modules():
#            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                nn.init.kaiming_normal(module.weight)
#                if module.bias is not None:
#                    module.bias.data.zero_()
#            elif isinstance(module, nn.BatchNorm2d):
#                module.weight.data.fill_(1)
#                module.bias.data.zero_()    

def max_pooling_2d():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def conv_block_out(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation)

def conv_block(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim))
    
def conv_trans_block(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        activation)

class unet_2d(nn.Module):
    def __init__(self):
        super(unet_2d, self).__init__()
        self.filters_list = [8, 16, 32, 64, 128, 256]
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block(1,self.filters_list[0],activation)
        self.pool_1 = max_pooling_2d()
        self.down_2 = conv_block(self.filters_list[0], self.filters_list[1], activation)
        self.pool_2 = max_pooling_2d()
        self.down_3 = conv_block(self.filters_list[1], self.filters_list[2], activation)
        self.pool_3 = max_pooling_2d()
        self.down_4 = conv_block(self.filters_list[2], self.filters_list[3], activation)
        self.pool_4 = max_pooling_2d()
        self.down_5 = conv_block(self.filters_list[3], self.filters_list[4], activation)
        self.pool_5 = max_pooling_2d()
        
        # Bridge
        self.bridge = conv_block(self.filters_list[4], self.filters_list[5], activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block(self.filters_list[5], self.filters_list[5], activation)
        self.up_1 = conv_block(self.filters_list[5]+self.filters_list[4], self.filters_list[4], activation)
        self.trans_2 = conv_trans_block(self.filters_list[4], self.filters_list[4], activation)
        self.up_2 = conv_block(self.filters_list[4]+self.filters_list[3], self.filters_list[3], activation)
        self.trans_3 = conv_trans_block(self.filters_list[3], self.filters_list[3], activation)
        self.up_3 = conv_block(self.filters_list[3]+self.filters_list[2], self.filters_list[2], activation)
        self.trans_4 = conv_trans_block(self.filters_list[2],self.filters_list[2], activation)
        self.up_4 = conv_block(self.filters_list[2]+self.filters_list[1], self.filters_list[1], activation)
        self.trans_5 = conv_trans_block(self.filters_list[1], self.filters_list[1], activation)
        self.up_5 = conv_block(self.filters_list[1]+self.filters_list[0], self.filters_list[0], activation)
        
        # Output
        self.out = conv_block_out(self.filters_list[0], 2, activation)
        self.final_activation = nn.Softmax(dim=1)

#        initialize_weights(self)
    
    def forward(self, x):
        # Down sampling
        #print('input',x.shape)
        down_1 = self.down_1(x) 
        pool_1 = self.pool_1(down_1)
        
        down_2 = self.down_2(pool_1) 
        pool_2 = self.pool_2(down_2)
        
        down_3 = self.down_3(pool_2) 
        pool_3 = self.pool_3(down_3) 
        
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4) 
        
        down_5 = self.down_5(pool_4) 
        pool_5 = self.pool_5(down_5) 
        
        # Bridge
        bridge = self.bridge(pool_5)
        #print('d1',down_1.shape,pool_1.shape)
        #print('d2',down_2.shape,pool_2.shape)
        #print('d3',down_3.shape,pool_3.shape)
        #print('d4',down_4.shape,pool_4.shape)
        #print('d5',down_5.shape,pool_5.shape)
        #print(bridge.shape)

        # Up sampling
        trans_1 = self.trans_1(bridge) 
        #print(trans_1.shape,down_5.shape)
        concat_1 = torch.cat([trans_1, down_5], dim=1) 
        up_1 = self.up_1(concat_1)
        
        trans_2 = self.trans_2(up_1) 
        #print(trans_2.shape,down_4.shape)
        concat_2 = torch.cat([trans_2, down_4], dim=1) 
        up_2 = self.up_2(concat_2) 
        
        trans_3 = self.trans_3(up_2)
        #print(trans_3.shape,down_3.shape)
        concat_3 = torch.cat([trans_3, down_3], dim=1) 
        up_3 = self.up_3(concat_3) 
        
        trans_4 = self.trans_4(up_3) 
        concat_4 = torch.cat([trans_4, down_2], dim=1) 
        up_4 = self.up_4(concat_4) 
        
        trans_5 = self.trans_5(up_4) 
        concat_5 = torch.cat([trans_5, down_1], dim=1) 
        up_5 = self.up_5(concat_5) 
        
        # Output
        out = self.out(up_5)
        #out = self.final_activation(out)
        #print('u1',up_1.shape,pool_1.shape)
        #print('u2',up_2.shape,pool_2.shape)
        #print('u2',up_3.shape,pool_3.shape)
        #print('u4',up_4.shape,pool_4.shape)
        #print('u5',up_5.shape,pool_5.shape)
        #print('out1',out.shape)
            #(1,2,512,512)
        out = self.final_activation(out)
        return out

