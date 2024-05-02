#!/usr/bin/env bash

MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
BATCH_SIZE=16

accelerate launch --num_processes=1 --gpu_ids="1" run_evals_accelerate.py \
    --model_args "pretrained=${MODEL_ID}" \
    --tasks ./leaderboard_tasks.txt \
    --override_batch_size ${BATCH_SIZE} \
    --use_chat_template \
    --output_dir "./leaderboard"

