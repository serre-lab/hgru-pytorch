 #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="3" python main.py Polygons Flow train_images.txt \
 test_images.txt  --modules gru --arch inception \
--print-freq 1 --num_segments 10 --gd 20 --lr 1e-04 --lr_steps 30 60 --gpus 1 \
 --epochs 8 -b 32 -j 8 --dropout 0.0 --snapshot_pref ucf101_bniception_

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of hGRU")

parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)


# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')


# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('-parallel', '--parallel', default= False, action='store_true',
                    help='Wanna parallelize the training')
