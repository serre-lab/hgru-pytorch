#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:04:57 2019

@author: alekhka
"""

import os
import time
import torch
from torchvision.transforms import Compose as transcompose
import torch.nn.parallel
import torch.optim
import numpy as np

from dataset import DataSetPol
from hgru import hConvGRU, FFConvNet
from transforms import GroupScale, Augmentation, Stack, ToTorchFormatTensor
from misc_functions import AverageMeter, accuracy

from opts import parser

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.ion()
plt.show()

global best_prec1
best_prec1 = 0
args = parser.parse_args()

transform_list = transcompose([GroupScale((150,150)), Augmentation(), Stack(), ToTorchFormatTensor(div=True)])

print("Loading training dataset")
train_loader = torch.utils.data.DataLoader(DataSetPol("/media/data_cifs/curvy_2snakes_300/", args.train_list, transform = transform_list ), 
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

print("Loading validation dataset")
val_loader = torch.utils.data.DataLoader(DataSetPol("/media/data_cifs/curvy_2snakes_300/", args.val_list, transform = transform_list),
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

def validate(val_loader, model, iter, criterion, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, target) in enumerate(val_loader):
            target = target.cuda()
            imgs = imgs.cuda()
            output = model.forward(imgs)
            
            loss = criterion(output, target)
            losses.update(loss.data, imgs.size(0))
            
            [prec1] = accuracy(output.data, target, topk=(1,))
            top1.update(prec1, imgs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t Time: {batch_time.avg:.3f}\t Loss: {loss.val:.4f} ({loss.avg: .4f})\t'
                       'Prec: {top1.val:.3f} ({top1.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
            
    print('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1, top5=top5, loss=losses))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.name, 'accuracy', str(state['best_prec1'].item()), 'epoch', str(state['epoch']), filename))
    torch.save(state, filename)


if __name__ == '__main__':
    
    #global best_prec1

    print("Init model")
    if args.parallel == True:
        model = hConvGRU(timesteps=8)
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        print("Loading parallel finished")
    else:
        model = hConvGRU(timesteps=8).cuda()
        print("Loading finished")

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_init = args.lr
    print("Starting training: ")
    f_val= []
    f_training = []
    train_loss_history = []
    for epoch in range(args.start_epoch, args.epochs):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
    
        model.train()
        end = time.perf_counter()
        for i, (imgs, target) in enumerate(train_loader):
            data_time.update(time.perf_counter() - end)
            
            imgs = imgs.cuda()
            target = target.cuda()
            
            output  = model.forward(imgs)
            #import pdb; pdb.set_trace()
            
            loss = criterion(output, target)
            [prec1] = accuracy(output.data, target, topk=(1,))
            
            losses.update(loss.data, imgs.size(0))
            top1.update(prec1, imgs.size(0))
            
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            #plot_grad_flow(model.named_parameters())
            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.perf_counter() - end)
            
            end = time.perf_counter()
            if i % (args.print_freq) == 0:
                print('Epoch: [{0}][{1}/{2}]\t lr: {lr:g}\t Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Prec: {top1.val:.3f} ({top1.avg:.3f})\t Loss: {loss.val:.6f} ({loss.avg:.6f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, lr=args.lr, top1=top1))
            
        f_training.append(top1.avg)
        train_loss_history += losses.history
        if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
            prec = validate(val_loader, model, (epoch + 1) * len(train_loader), criterion)
            f_val.append(prec)
            is_best = prec > best_prec1
            if is_best:
                best_prec1 = max(prec, best_prec1)
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                }, is_best)

    np.array(f_training).dump(open("{}.npy".format(args.name),'w'))
    np.array(f_val).dump(open("{}.npy".format(args.name),'w'))




