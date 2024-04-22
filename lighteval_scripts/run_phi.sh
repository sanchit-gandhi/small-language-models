#!/usr/bin/env bash

accelerate launch run_evals_accelerate.py \
    --model_args "pretrained=microsoft/phi-1_5" \
    --tasks ./all_cosmo_tasks.txt \
    --override_batch_size 8 \
    --output_dir "./phi-rerun"
