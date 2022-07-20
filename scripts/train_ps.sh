#!/usr/bin/env bash

system_info="$(uname -s)"
case "${system_info}" in
    Linux*)     system=Linux;;
    Darwin*)    system=Mac;;
    CYGWIN*)    system=Cygwin;;
    MINGW*)     system=MinGw;;
    *)          system="UNKNOWN:${system_info}"
esac

#train_script="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/train.py"
#root_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src'
#save_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/ps'
#train_list='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/Dataset/train.txt'
#val_list='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/Dataset/val.txt'
#ckpt='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/mvs/checkpoints/model_000001.ckpt'

train_script='/vol/research/iview-data/Ope/Deep_Helmholtz/train.py'
root_dir='/vol/research/iview-data/Ope/'
save_dir='/vol/research/iview-data/Ope/Helmholtz_save/ps0'
train_list='/vol/research/iview-data/Ope/Deep_Helmholtz/Dataset/train.txt'
val_list='/vol/research/iview-data/Ope/Deep_Helmholtz/Dataset/val.txt'


batch=2
epochs=60
lr=0.0016

#torchrun params
NUM_GPUS=$1
JOB_ID=0
#HOST_NODE_ADDR=''

torchrun  --standalone \
          --nnodes=1  \
          --nproc_per_node=$NUM_GPUS  \
          $train_script --system ${system} --root_dir $root_dir --save_dir $save_dir --train_list $train_list \
          --val_list $val_list --net_type 'ps' --batch $batch --epochs 10 --lr $lr \
          --lr_idx '20, 30, 40, 50:0.625' --loss_weights '0.5, 1.0, 2.0' --num_sel_views 1 --conf_lambda 1.5 \
          --planes_in_stages '64, 32, 8' --sync_bn --log_freq 50

