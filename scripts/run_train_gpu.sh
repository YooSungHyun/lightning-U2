#!/bin/bash
GPU_IDS="0"
HF_DATA_DIRS=""
PL_DATA_DIR="
"

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 train.py \
    --pl_data_dir $PL_DATA_DIR \
    --cache_main_dir= \
    --num_shards=20 \
    --model_config="./config/conformer_u2++.yaml" \
    --vocab_path="./config/vocab.json" \
    --output_dir="../model_outputs" \
    --seed=42 \
    --num_proc=12 \
    --per_device_train_batch_size=16 \
    --train_batch_drop_last=false \
    --eval_batch_drop_last=false \
    --val_check_interval=0.25 \
    --accumulate_grad_batches=1 \
    --max_epochs=120 \
    --log_every_n_steps=3200 \
    --accelerator=gpu \
    --devices=1 \
    --learning_rate=0.001 \
    --precision=32 \
    --weight_decay=0.01 \
    --warmup_ratio=0.01 \
    --final_div_factor=10 \
    --div_factor=20 \
    --label_name=labels \
    --group_by_length=true \
    --input_name=input_values \
    --length_column_name=length \
    --encoder_type=conformer \
    --detect_anomaly=true
