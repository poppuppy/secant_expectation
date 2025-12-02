#!/usr/bin/env sh

train_data_root='your_data'

loss_type='sdei'
model=DiT-XL/2
batch_size=256
lr=1e-5
online_cfg='1-2.5'
num_steps=1

exp_name=${model//\//-}_bs${batch_size}_lr${lr}_cfgembed${online_cfg}_${loss_type}_${num_steps}step

accelerate launch --multi_gpu --main_process_port=12069 --num_processes 1 --mixed_precision fp16 \
    train.py \
    --loss-type ${loss_type} \
    --feature-path ${train_data_root} \
    --model ${model} \
    --online-cfg ${online_cfg} \
    --num-steps ${num_steps} \
    --lr ${lr} \
    --global-batch-size ${batch_size} \
    --results-dir results \
    --exp-name ${exp_name} \
    --init-from 'pretrained/Init-SiT-XL-2-256.pt' \
    --teacher-ckpt 'pretrained/SiT-XL-2-256.pt'