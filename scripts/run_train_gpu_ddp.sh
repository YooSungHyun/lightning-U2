#!/bin/bash
GPU_IDS=""
HF_DATA_DIRS=""
PL_DATA_DIR=""

OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
    --hf_data_dirs=$HF_DATA_DIRS \
    --pl_data_dir=$PL_DATA_DIR \
    --num_shards=20 \
    --model_config="./config/config.json" \
    --vocab_path="./config/vocab.json" \
    --output_dir="../model_outputs" \
    --seed=42 \
    --num_proc=12 \
    --per_device_train_batch_size=32 \
    --train_batch_drop_last=false \
    --per_device_eval_batch_size=32 \
    --eval_batch_drop_last=false \
    --val_check_interval=0.25 \
    --accumulate_grad_batches=1 \
    --max_epochs=40 \
    --log_every_n_steps=400 \
    --accelerator=gpu \
    --devices=1 \
    --learning_rate=0.001 \
    --precision=16 \
    --weight_decay=0.01 \
    --warmup_ratio=0.01 \
    --final_div_factor=10 \
    --div_factor=20 \
    --label_name=grapheme_input_ids
