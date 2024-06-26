#!/usr/bin/env bash

python run_evaluation.py \
  --model_name_or_path "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer" \
  --output_dir "./" \
  --dataset_name "HuggingFaceTB/cosmopedia-100k" \
  --dataset_config_name "default" \
	--split_name "train" \
	--max_samples 8 \
  --per_device_eval_batch_size 4 \
  --overwrite_output_dir \
  --output_router_logits True \
  --report_to "wandb" \
  --load_in_4bit \
  --wandb_project "distil-mixtral-dummy" \
  --streaming