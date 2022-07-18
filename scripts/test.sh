#!/usr/bin/env bash

test_script="/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Deep_Helmholtz/test.py"
root_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src'
save_dir='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/test_results'
ckpt='/Users/culsu/Documents/UNI_stuff/Surrey/Courses/MSC_Project/src/Helmholtz_save/mvs/checkpoints/model_000001.ckpt'

#train_script='Deep_Helmholtz/test.py'
#root_dir=''
#save_dir='Helmholtz_save/mvs'
#ckpt='Helmholtz_save/mvs/checkpoints/model_000001.ckpt'

python $test_script --root_dir $root_dir --save_dir $save_dir --net_type 'mvs' --num_sel_views 1 --conf_lambda 1.5 \
          --planes_in_stages '64, 32, 8' --ckpt $ckpt