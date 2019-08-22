 #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="3" 
python main.py train_images_9.txt test_images_9.txt --print-freq 200 --lr 1e-03 --epochs 8 -b 32