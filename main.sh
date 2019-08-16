 #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="3" python main.py Polygons Flow train_images.txt \
 test_images.txt  --modules gru --arch inception \
--print-freq 1 --num_segments 10 --gd 20 --lr 1e-04 --lr_steps 30 60 --gpus 1 \
 --epochs 8 -b 32 -j 8 --dropout 0.0 --snapshot_pref ucf101_bniception_

