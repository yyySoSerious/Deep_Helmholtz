#!/usr/bin/env bash

system_info="$(uname -s)"
case "${system_info}" in
    Linux*)     system=Linux;;
    Darwin*)    system=Mac;;
    CYGWIN*)    system=Cygwin;;
    MINGW*)     system=MinGw;;
    *)          system="UNKNOWN:${system_info}"
esac

train_script="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/train.py"
root_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src'
save_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/mvs2'
ckpt='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/mvs/checkpoints/model_000001.ckpt'

#train_script='Deep_Helmholtz/train.py'
#root_dir=''
#save_dir='Helmholtz_save/mvs'


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
          $train_script --system ${system} --root_dir $root_dir --save_dir $save_dir --batch $batch --epochs 10 --lr 0.0016 \
          --lr_idx '20, 30, 40, 50:0.625' --loss_weights '0.5, 1.0, 2.0' --num_sel_views 1 --conf_lambda 1.5 \
          --planes_in_stages '64, 32, 8' --sync_bn --log_freq 50 --ckpt_to_continue $ckpt

