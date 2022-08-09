#!/usr/bin/env bash

#train_script="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/train.py"
#root_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src'
#save_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/ps'
#train_list='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/Dataset/train.txt'
#val_list='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/Dataset/val.txt'
#ckpt='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/mvs/checkpoints/model_000001.ckpt'

mode='ps'
train_script='/vol/research/iview-data/Ope/Deep_Helmholtz/train.py'
root_dir='/vol/research/iview-data/Ope/'
save_dir="/vol/research/iview-data/Ope/Helmholtz_save/$mode"
train_list='/vol/research/iview-data/Ope/Deep_Helmholtz/Dataset/train.txt'
val_list='/vol/research/iview-data/Ope/Deep_Helmholtz/Dataset/val.txt'
ckpt='/vol/research/iview-data/Ope/Helmholtz_save/ps/checkpoints/model_000010.ckpt'


batch=2
epochs=60
lr=0.0016
num_gpus=$1
num_workers=$2

torchrun  --standalone \
          --nnodes=1  \
          --nproc_per_node=$num_gpus  \
          $train_script --root_dir $root_dir --save_dir $save_dir --train_list $train_list \
          --val_list $val_list --net_type $mode --batch $batch --epochs 10 --lr $lr \
          --lr_idx '20, 30, 40, 50:0.625' --loss_weights '0.5, 1.0, 2.0' --num_sel_views 1 --conf_lambda 1.5 \
          --planes_in_stages '64, 32, 8' --sync_bn --log_freq 50 --num_workers $num_workers

