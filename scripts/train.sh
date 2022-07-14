#!/usr/bin/env bash

#train_script="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/train.py"
train_script='Deep_Helmholtz/train.py'

system_info="$(uname -s)"
case "${system_info}" in
    Linux*)     system=Linux;;
    Darwin*)    system=Mac;;
    CYGWIN*)    system=Cygwin;;
    MINGW*)     system=MinGw;;
    *)          system="UNKNOWN:${system_info}"
esac

#root_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src' #shared file system
#save_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save' #shared file system

root_dir=''
save_dir='Helmholtz_save'

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
          --max_restarts=3  \
          $train_script --system ${system} --root_dir $root_dir --save_dir $save_dir --batch $batch --epochs 10 --lr 0.0016 \
          --lr_idx '20, 30, 40, 50:0.625' --loss_weights '0.5, 1.0, 2.0' --num_sel_views 1 --conf_lambda 1.5 \
          --planes_in_stages '64, 32, 8' --sync_bn --log_freq 50 | tee -a $save_dir/log.txt

