#!/usr/bin/env bash

python run_initialization.py \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --num_hidden_layers "6" \
  --output_dir "./" \
  --hub_model_id "sanchit-gandhi/Mistral-1.5B-Instruct-v0.2" \
  --push_to_hub
