 #!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,3,5
python main.py train_images_9.txt test_images_9.txt --print-freq 200 --lr 1e-03 --epochs 80 -b 64 --parallel
