#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:09:18 2020

@author: yangqianwan
"""

#import sys
#sys.path.append(r'/home/yqw/seg/data')
from dataset import MyDataset
from unet_2d import unet_2d
#import sys
#sys.path.append(r'/Users/yangqianwan/Desktop/lab/NET')
#from config import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import time
import copy
from torch.utils import data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from losses import BCEDiceLoss

use_gpu = torch.cuda.is_available()
#print(use_gpu)
num_gpu = list(range(torch.cuda.device_count()))
#print(num_gpu)
pixel_acc_list = []
mIOU_list = []
dice_coef_list=[]
#batch_size = 4
#dataloaders = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
## parameters for Solver-Adam in this example
num_class=2
lr         = 1e-4    # achieved besty results
#step_size  = 4 # Won't work when epochs <=100
gamma      = 0.5 #
epochs=50
global_index=0
torch.cuda.set_device(1)
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
whole_set=MyDataset()
length=len(whole_set)
train_size=2600
train_size,validate_size=train_size,len(whole_set)-train_size
train_set,validate_set=data.random_split(whole_set,[train_size,validate_size])
batch_size = 32
train_loader = data.DataLoader(train_set, batch_size=batch_size,  num_workers=0, shuffle=True)
val_loader = data.DataLoader(validate_set, batch_size=4, num_workers=0, shuffle=False)
#criterion = nn.BCEWithLogitsLoss().cuda()
criterion=BCEDiceLoss().cuda()
model = unet_2d()
optimizer = optim.Adam(model.parameters(), lr=lr)
#model = nn.DataParallel(model, device_ids=[0])
model = model.cuda()
#scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#loss_meter=AverageMeter()
print('end')

# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def dice_coef(pred, target):
    smooth = 1e-5
    dice_coefs = []
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        dice_coefa=(2 * intersection + smooth) / (pred_inds.sum() + target_inds.sum() + smooth)
        dice_coefs.append(float(dice_coefa))
    return dice_coefs

def save_result_comparison(input_np, input_mask, output_np):

    global global_index

    original_im = np.zeros((512,512))
    #plt.figure()
    #plt.imshow(input_np[0,0,:,:],cmap='gray')
    #print(input_np.shape)
    original_im[:,:]=input_np[0,0,:,:]
    im_seg = np.zeros((512, 512))
    im_mask = np.zeros((512, 512))

    # the following version is designed for 11-class version and could still work if the number of classes is fewer.
    for i in range(512):
        for j in range(512):
            if output_np[i, j] == 0:
                im_seg[i, j] = 255
            elif output_np[i, j] == 1:
                im_seg[i, j] = 0
            if input_mask[0,i,j] == 0:
                im_mask[i, j] = 255
            elif input_mask[0,i,j] == 1:
                im_mask[i, j] = 0

    # horizontally stack original image and its corresponding segmentation results
    hstack_image = np.hstack((original_im, im_seg, im_mask))
    new_im = Image.fromarray(np.uint8(hstack_image))
    file_name = '/home/yqw/seg/check2/' + str(global_index) + '.jpg'
    #plt.imshow(hstack_image, cmap='gray')
    #plt.savefig(file_name)
    #plt.show()
    new_im.save(file_name)
    global_index = global_index + 1

def train(epoch):
    #scheduler.step()
    ts = time.time()
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            target = Variable(batch['Y'].cuda())
            #model = model.cuda()
        else:
            inputs, target = Variable(batch['X']), Variable(batch['Y'])
        #print('input', inputs.shape)
        outputs = model(inputs)
        #print('outputs', outputs.shape)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if iter % 10 == 0:
            print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data.item()))
    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

    #val(epoch)
    #if 1 > 0:
    #    prefix='/home/yqw/seg/check1/'
    #    name=time.strftime(prefix+ '%m%d_%H:%M:%S.pth')
    #    torch.save(model.state_dict(),name)

def val(epoch):
    model.eval()
    total_ious = []
    pixel_accs = []
    total_dice_coef=[]

    for iter, batch in enumerate(val_loader): ## batch is 1 in this case
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            #print('gpu')
        else:
            inputs = Variable(batch['X'])
            #print('nogpu')

        output = model(inputs)                                
        
        # only save the 1st image for comparison
        if iter == 0:
            print('---------iter={}'.format(iter))
            # generate images
            images = output.data.max(1)[1].cpu().numpy()[:,:,:]
            image = images[0,:,:]
            #print(batch['l'].shape,'input',batch['X'].shape)
            save_result_comparison(batch['X'],batch['l'],image)
            #print('batch',batch['X'].shape)
            #save_result_comparison(batch['X'], image)
                            
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, num_class).argmax(axis=1).reshape(N, h, w)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
#        print('pred',pred.shape)
#        print('target',target.shape)
#        plt.figure()
#        plt.imshow(pred[0,:,:],cmap='gray')
#        plt.title('predict')
#        plt.figure()
#        plt.imshow(target[0,:,:],cmap='gray')
#        plt.title('original')
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))
            total_dice_coef.append(dice_coef(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    total_dice_coef=np.array(total_dice_coef).T
    dice_coefs=np.mean(total_dice_coef,axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}, meanDice: {}, Dice: {}".format(epoch, pixel_accs, np.nanmean(ious), ious ,np.mean(dice_coefs), dice_coefs))
    
    global pixel_acc_list
    global mIOU_list
    global dice_coef_list
    
    pixel_acc_list.append(pixel_accs)
    mIOU_list.append(np.nanmean(ious))
    dice_coef_list.append(np.mean(dice_coefs))

def main():
    val(0)
    for epoch in range(epochs):
        train(epoch)
        val(epoch)
        prefix='/home/yqw/seg/check2/'
        name=time.strftime(prefix+ '%m%d_%H:%M:%S.pth')
        torch.save(model.state_dict(),name)
    highest_pixel_acc = max(pixel_acc_list)
    highest_mIOU = max(mIOU_list)
    highest_dice_coef = max(dice_coef_list)
    highest_pixel_acc_epoch = pixel_acc_list.index(highest_pixel_acc)
    highest_mIOU_epoch = mIOU_list.index(highest_mIOU)
    highest_dice_coef_epoch = dice_coef_list.index(highest_dice_coef)

    print("The highest mIOU is {} and is achieved at epoch-{}".format(highest_mIOU, highest_mIOU_epoch))
    print("The highest pixel accuracy  is {} and is achieved at epoch-{}".format(highest_pixel_acc, highest_pixel_acc_epoch))
    print("The highest dice is {} and is achieved at epoch-{}".format(highest_dice_coef, highest_dice_coef_epoch))

if __name__ == "__main__":
    main()
