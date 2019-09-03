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
import rbp_optim
import numpy as np

from dataset import DataSetPol
from hgru_rbp import hConvGRU, FFConvNet
from transforms import GroupScale, Augmentation, Stack, ToTorchFormatTensor
from misc_functions import AverageMeter, accuracy, plot_grad_flow
# from statistics import mean

from opts import parser
from rbp import RBP
from utils.model_helper import detach_param_with_grad

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from torchviz import make_dot
except Exception as e:
    print('Failed to import torchviz: %s' % e)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        model = hConvGRU(timesteps=8, filt_size = 15)
        model = torch.nn.DataParallel(model).to(device)
        print("Loading parallel finished on GPU count:", torch.cuda.device_count())
    else:
        model = hConvGRU(timesteps=8, filt_size = 15).to(device)
        print("Loading finished")

    grad_method = 'RBP'
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = rbp_optim.Adam(model.parameters(), lr=args.lr)

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
        end = time.time()  # perf_counter()
        for i, (imgs, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            
            imgs = imgs.to(device)
            target = target.to(device)
            
            output, states = model.forward(imgs)
            state_last = states[-1]
            state_2nd_last = states[-2]

            loss = criterion(output, target)
            
            #import ipdb; ipdb.set_trace()
            #test = make_dot(loss, params=dict(model.named_parameters()))
            #test.render(filename='testnew.dot')
            #import ipdb; ipdb.set_trace()
            tuple_params = [(nn, pp) for nn, pp in model.named_parameters()]
            names, params = zip(*tuple_params)

            # Hardcode this for now
            output_idx = 8
            param_output = params[-output_idx:]
            name_output = names[-output_idx:]
            param_update = params[:-output_idx]
            name_update = names[:-output_idx]

            if grad_method == 'RBP':
                grad_dict = {}
                grad_output = torch.autograd.grad(loss, param_output, retain_graph=True)
                for nn, gg in zip(name_output, grad_output):
                   grad_dict[nn] = gg
                #import ipdb; ipdb.set_trace()
                grad_state_last = torch.autograd.grad(loss, state_last, retain_graph=True)
                grad_update = RBP(param_update,
                   [state_last],
                   [state_2nd_last],
                   grad_state_last,
                   # update_forward_diff=None,
                   # eta=1.0e-5,
                   truncate_iter=40,
                   rbp_method='Neumann_RBP')
                #import ipdb; ipdb.set_trace()
                for nn, gg in zip(name_update, grad_update):
                   grad_dict[nn] = gg
                grad = [grad_dict[nn] for nn, pp in model.named_parameters()]
            else:
                grad = torch.autograd.grad(loss, params)
            
            # loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            #plot_grad_flow(model.named_parameters())
            # assign gradient
            for pp, ww in zip(params, grad):
                pp.grad = ww

            optimizer.step()
            #import ipdb; ipdb.set_trace()
            #[print(nn, pp.shape) for nn, pp in model.named_parameters()]
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            
            losses.update(loss.data.item(), imgs.size(0))
            
            [prec1] = accuracy(output.data, target, topk=(1,))
            top1.update(prec1.data.item(), imgs.size(0))
            
            end = time.time()
            if i % (args.print_freq) == 0:
                #plot_grad_flow(model.named_parameters())
                print('Epoch: [{0}][{1}/{2}]\t lr: {lr:g}\t Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Prec: {top1.val:.3f} ({precprint:.3f}) ({top1.avg:.3f})\t Loss: {loss.val:.6f} ({lossprint:.6f}) ({loss.avg:.6f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, lossprint= np.mean(losses.history[-args.print_freq:]), lr=args.lr, top1=top1, precprint= np.mean(top1.history[-args.print_freq:])))
            
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

