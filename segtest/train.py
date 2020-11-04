import argparse
from dataset import MyDataset
from unet_2d import unet_2d
import pandas as pd
from tqdm import tqdm
import losses
import torch
from metrics import dice_coefs, pixel_accs, ious, save_result_comparison
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision import transforms
from torch.optim import lr_scheduler
import time
from torch.utils import data
from utils import str2bool
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image


loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', default=2600, type=int,
                        help='number of train set')
    parser.add_argument('--num_class', default=2, type=int,
                        help='class_number')
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, use_gpu, train_loader, model, criterion, optimizer, scheduler=None):
    losses = AverageMeter()
    dice_coef = AverageMeter()
    pixel_acc = AverageMeter()
    iou = AverageMeter()
    model.train()
    #ts = time.time()
    for iter, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            targets = Variable(batch['Y'].cuda())
        else:
            inputs, targets = Variable(batch['X']), Variable(batch['Y'])
        # compute output
        output = model(inputs)
        #print('output',output.shape,'targets',targets.shape)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        output = output.data.cpu().numpy()
        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, args.num_class).argmax(axis=1).reshape(N, h, w)
        mask = batch['l'].cpu().numpy().reshape(N, h, w)
        ioum = ious(pred, mask, args.num_class)
        dice_coefm=dice_coefs(pred, mask, args.num_class)
        pixel_accm=pixel_accs(pred, mask)
        losses.update(loss.item(), inputs.size(0))
        iou.update(ioum, inputs.size(0))
        dice_coef.update(dice_coefm, inputs.size(0))
        pixel_acc.update(pixel_accm, inputs.size(0))


    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', iou.avg),
        ('dice_coef', dice_coef.avg),
        ('pixel_acc', pixel_acc.avg),
    ])

    return log

def validate(epoch, args, use_gpu, val_loader, model, criterion):
    losses = AverageMeter()
    dice_coef = AverageMeter()
    pixel_acc = AverageMeter()
    iou = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                targets = Variable(batch['Y'].cuda())
            else:
                inputs, targets = Variable(batch['X']), Variable(batch['Y'])
            # compute output
            output = model(inputs)
            loss = criterion(output, targets)
            outputs = output.data.cpu().numpy()
            N, _, h, w = outputs.shape
            pred = outputs.transpose(0, 2, 3, 1).reshape(-1,args.num_class).argmax(axis=1).reshape(N, h, w)
            mask = batch['l'].cpu().numpy().reshape(N, h, w)
            ioum = ious(pred, mask, args.num_class)
            dice_coefm = dice_coefs(pred, mask, args.num_class)
            pixel_accm = pixel_accs(pred, mask)
            losses.update(loss.item(), inputs.size(0))
            iou.update(ioum, inputs.size(0))
            dice_coef.update(dice_coefm, inputs.size(0))
            pixel_acc.update(pixel_accm, inputs.size(0))
            # only save the 1st image for comparison
            if iter == 0:
                # only save the 1st image for comparison
                image = pred[0, :, :]
                save_result_comparison(epoch, batch['X'], mask, image)

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', iou.avg),
        ('dice_coef', dice_coef.avg),
        ('pixel_acc', pixel_acc.avg),
    ])

    return log

def main():
    args = parse_args()

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    # connect to cuda
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    num_gpu = list(range(torch.cuda.device_count()))
    print(num_gpu)
    torch.cuda.set_device(1)

    # Data loading code
    whole_set = MyDataset()
    length = len(whole_set)
    train_size = args.train_size
    train_size, validate_size = train_size, length - train_size
    train_set, validate_set = data.random_split(whole_set, [train_size, validate_size])
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=0, shuffle=True)
    val_loader = data.DataLoader(validate_set, batch_size=8, num_workers=0, shuffle=False)

    # create model
    model = unet_2d()
    model = model.cuda()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'dice_coef', 'pixel_acc', 'val_loss', 'val_iou', 'val_dice_coef', 'val_pixel_acc'
    ])

    best_iou = 0
    trigger = 0
    print('save result without train')
    validate('original', args, use_gpu, val_loader, model, criterion)
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args,use_gpu, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(epoch,args,use_gpu, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - dice_coef %.4f - pixel_acc %.4f - val_loss %.4f - val_iou %.4f - val_dice_coef %.4f - val_pixel_acc %.4f'
            %(train_log['loss'], train_log['iou'], train_log['dice_coef'], train_log['pixel_acc'], val_log['loss'], val_log['iou'], val_log['dice_coef'], val_log['pixel_acc']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            train_log['dice_coef'],
            train_log['pixel_acc'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice_coef'],
            val_log['pixel_acc']
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_coef', 'pixel_acc', 'val_loss', 'val_iou','val_dice_coef', 'val_pixel_acc'])

        log = log.append(tmp, ignore_index=True)
        prefix='/home/yqw/seg/check2/'
        name=time.strftime(prefix+ '%m%d_%H')
        log.to_csv(name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), name + '.pth')
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
