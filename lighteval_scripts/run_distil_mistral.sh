#!/usr/bin/env bash

MODEL_ID="sanchit-gandhi/distil-mistral-1.5B-v0.1-fineweb-checkpoint-15000"
BATCH_SIZE=16

accelerate launch --num_processes=1 --gpu_ids="2" run_evals_accelerate.py \
    --model_args "pretrained=${MODEL_ID}" \
    --tasks ./all_cosmo_tasks.txt \
    --override_batch_size ${BATCH_SIZE} \
    --use_chat_template \
    --output_dir "./cosmo"