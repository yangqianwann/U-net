# -*- coding: utf-8 -*-


from metrics import dice_coefs, pixel_accs, ious, save_result_comparison
import numpy as np
import torch
from dataset import MyDataset
from unet_2d import unet_2d
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm
import random

def main():
    #test gpu
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    num_gpu = list(range(torch.cuda.device_count()))
    print(num_gpu)
    # create model
    model = unet_2d()
    model = model.cuda()
    model.load_state_dict(torch.load('/home/yqw/seg/20201030/0.95,0.99/1030_12:54:09.pth'))
    model.eval()
    #load data
    # Data loading code
    whole_set = MyDataset()
    length = len(whole_set)
    train_size = 3000
    train_size, test_size = train_size, length - train_size
    train_set, test_set = data.random_split(whole_set, [train_size, test_size])
    val_loader = data.DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

    #if val_args.mode == "GetPicture":
    if 1>0:
        """
        Generate result pictures
        """
        with torch.no_grad():
            for iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                if use_gpu:
                    inputs = Variable(batch['X'].cuda())
                else:
                    inputs = Variable(batch['X'])
                # compute output
                output = model(inputs)
                outputs = output.data.cpu().numpy()
                N, _, h, w = outputs.shape
                pred = outputs.transpose(0, 2, 3, 1).reshape(-1, 2).argmax(axis=1).reshape(N, h, w)
                mask = batch['l'].cpu().numpy().reshape(N, h, w)
                image = pred[0, :, :]
                save_result_comparison(iter, batch['X'], mask, image)
        print("Done!")


if __name__ == '__main__':
    main()
