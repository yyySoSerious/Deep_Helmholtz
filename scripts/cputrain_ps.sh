#!/usr/bin/env bash

mode='ps'
train_script="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/train.py"
root_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src'
save_dir="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/$mode"
train_list='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/test/train.txt'
val_list='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/test/val.txt'
ckpt="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/$mode/checkpoints/model_000003.ckpt"
save_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Test_2'
ckpt='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Test_2/checkpoint.txt'
train_list='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/Dataset/train.txt'

batch=8
epochs=60
lr=0.0010
num_workers=$2
num_gpus=$1

torchrun  --standalone \
          --nnodes=1  \
          --nproc_per_node=$num_gpus  \
          $train_script --root_dir $root_dir --save_dir $save_dir --train_list $train_list \
          --val_list $val_list --net_type $mode --batch $batch --epochs $epochs --lr $lr \
          --lr_idx '2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60:0.5' --num_reciprocals 2 --log_freq 50 --num_workers $num_workers  --sync_bn\
          --use_bn --lr_decay 0  --base_channels 32  --num_gpus $num_gpus   --rcpcl_fuse_type 'max' --fuse_type 'mean'