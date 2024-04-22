#!/usr/bin/env bash

accelerate launch run_evals_accelerate.py \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.2" \
    --tasks ./all_cosmo_tasks.txt \
    --override_batch_size 8 \
    --output_dir "./"
