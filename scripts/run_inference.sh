#!/bin/bash
MODEL_DIR=""

python3 inference.py \
    --model_path="${MODEL_DIR}" \
    --model_config="${MODEL_DIR}/conformer_u2++.yaml" \
    --vocab_path="${MODEL_DIR}/vocab.json" \
    --beam_size=2 \
    --decoding_chunk_size=16 \
    --num_decoding_left_chunks=1 \
    --simulate_streaming=true \
    --reverse_weight=0.5 \
    --accelerator=cpu \
    --encoder_type=conformer \
    --mode=attention_rescoring