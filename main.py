import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.optim
import torch.nn as nn
import numpy as np
from tempfile import TemporaryFile
from dataset import DataSetPol
from hgru import hConvGRU, FFConvNet
from transforms import *
from opts import parser

best_prec = 0

def main():
    global args, best_prec
    args = parser.parse_args()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    print("Loading training dataset")
    train_loader = torch.utils.data.DataLoader(
        DataSetPol("/home/josueortc/", args.train_list,
                   modality=args.modality,
                       transform=torchvision.transforms.Compose([
                       GroupScale((150,150)),Augmentation(), Stack(),
                       ToTorchFormatTensor(div=True)])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print("Loading validation dataset")
    val_loader = torch.utils.data.DataLoader(
        DataSetPol("/home/josueortc/", args.val_list,
                   modality=args.modality,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale((150,150)), Augmentation(),Stack(),
                       ToTorchFormatTensor(div=True)])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    features = [[16,16],[32,32], [64, 64], [64, 64]]
    print("Input Shape: ", in_shape)
    print("Loading Model")
    if args.parallel == True:
        model = hConvGRU(timesteps=8)
        print("Loading Finished")
        model = torch.nn.DataParallel(model, device_ids=[0]]).cuda()
    else:
        model = hConvGRU(timesteps=8).cuda()
        print("Loading Finished")


    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_init = args.lr
    print("Starting training: ")
    f_val= []
    f_training = []

    for epoch in range(args.start_epoch, args.epochs):
        train_accuracy = train(train_loader, model, optimizer, epoch, criterion)
        f_training.append(train_accuracy)
        if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, (epoch + 1) * len(train_loader), criterion)
            prec2 = validate(val_loader, model, (epoch + 1) * len(train_loader), criterion)
            prec3 = validate(val_loader, model, (epoch + 1) * len(train_loader), criterion)
            prec4 = validate(val_loader, model, (epoch + 1) * len(train_loader), criterion)
            prec = (prec1+prec2+prec3+prec4)/4.0
            f_val.append(prec)
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                }, is_best)

    np.array(f_training).dump(open("{}.npy".format(args.name),'w'))
    np.array(f_val).dump(open("{}.npy".format(args.name),'w'))


def train(train_loader, model, optimizer, epoch, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    opt_readout = readout
    end = time.time()
    model.train()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)


        # compute output for the number of timesteps selected by train loader
        output  = model.forward(x=input_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], input_var.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if i % args.print_freq == 0:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                   'Loss {loss.val:.6f} ({loss.avg:.6f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lr=args.lr, top1=top1, top5=top5)))

    return top1.avg

def validate(val_loader, model, iter, criterion, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #Change if using pytorch > 0.3
        target = target.cuda(async=True).cuda()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model.forward(x=input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 2))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.avg:.3f}\t'
                   'Loss {loss.val:.4f} ({loss.avg: .4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), '{}.pth.tar'.format(args.name)))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
