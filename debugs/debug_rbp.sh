#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
python -m torch.utils.bottleneck main_rbp_20_debug.py train_images_9.txt test_images_9.txt --print-freq 200 --lr 1e-03 --epochs 1 -b 48  # 64 --parallel
