#!/usr/bin/env bash

python run_initialization.py \
  --model_name_or_path "mistralai/Mixtral-8x7B-v0.1" \
  --num_hidden_layers "6" \
  --output_dir "./" \
  --hub_model_id "sanchit-gandhi/mixtral-8x1.5B" \
  --push_to_hub
