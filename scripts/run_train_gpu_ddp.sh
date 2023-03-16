#!/bin/bash
GPU_IDS="0,1"
PL_DATA_DIR="/ext_disk/stt/datasets/fine-tuning/42maru/data-KsponSpeech-42maru-not-normal-20"

OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
    --pl_data_dir=$PL_DATA_DIR \
    --num_shards=20 \
    --model_config="./config/conformer_u2++.yaml" \
    --vocab_path="./config/grapheme_vocab.json" \
    --output_dir="../model_outputs" \
    --seed=42 \
    --num_proc=12 \
    --per_device_train_batch_size=8 \
    --train_batch_drop_last=false \
    --per_device_eval_batch_size=8 \
    --eval_batch_drop_last=false \
    --val_check_interval=0.25 \
    --accumulate_grad_batches=4 \
    --max_epochs=40 \
    --log_every_n_steps=400 \
    --accelerator=gpu \
    --strategy=ddp \
    --devices=2 \
    --learning_rate=0.001 \
    --gradient_clip_val=5.0 \
    --precision=32 \
    --weight_decay=0.01 \
    --warmup_ratio=0.01 \
    --final_div_factor=10 \
    --div_factor=20 \
    --label_name=grapheme_labels \
    --group_by_length=true \
    --input_name=input_values \
    --length_column_name=length \
    --encoder_type=conformer \
    --detect_anomaly=true
