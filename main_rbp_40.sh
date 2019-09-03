#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main_rbp_40.py train_images_9.txt test_images_9.txt --print-freq 200 --lr 1e-03 --epochs 80 -b 48  # 64 --parallel
