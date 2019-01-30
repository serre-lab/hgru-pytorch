#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="0,1,2" python main.py Polygons Flow /mnt/group1/PathFinder/train_pathfinder_14.txt \
 /mnt/group1/PathFinder/test_pathfinder_14.txt  --modules gru --arch inception \
--print-freq 1 --num_segments 10 --gd 20 --lr 1e-03 --lr_steps 30 60 --gpus 4 \
 --epochs 8 -b 32 -j 8 --dropout 0.0 --snapshot_pref ucf101_bniception_
