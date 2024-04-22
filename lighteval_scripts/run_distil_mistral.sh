#!/usr/bin/env bash

MODEL_ID="sanchit-gandhi/distil-mistral-1.5B-Instruct-v0.2"

accelerate launch run_evals_accelerate.py \
    --model_args "pretrained=${MODEL_ID}" \
    --tasks ./subset_cosmo_tasks.txt \
    --override_batch_size 8 \
    --output_dir "./intermediate-results"

accelerate launch run_evals_accelerate.py \
    --model_args "pretrained=${MODEL_ID}" \
    --tasks ./all_cosmo_tasks.txt \
    --override_batch_size 8 \
    --output_dir "./"
