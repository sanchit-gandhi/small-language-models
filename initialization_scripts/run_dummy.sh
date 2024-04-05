#!/usr/bin/env bash

python run_initialization.py \
  --model_name_or_path "HuggingFaceM4/tiny-random-MistralForCausalLM" \
  --num_hidden_layers "1" \
  --output_dir "./" \
  --push_to_hub \
  --hub_model_id "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer"

