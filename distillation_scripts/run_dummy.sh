#!/usr/bin/env bash

accelerate launch --multi_gpu --num_processes=8 --gpu_ids="all" run_distillation.py \
  --model_name_or_path "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer" \
  --teacher_model_name_or_path "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer" \
  --output_dir "./" \
  --train_dataset_name "HuggingFaceTB/cosmopedia-100k" \
  --train_dataset_config_name "default" \
	--train_split_name "train" \
	--eval_split_name "train" \
	--do_train \
  --do_eval \
	--max_train_samples 1000 \
  --num_train_epochs 2 \
  --max_eval_samples 1000 \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 8 \
  --save_strategy "no" \
	--evaluation_strategy "epoch" \
	--logging_steps 1 \
  --overwrite_output_dir \
  --output_router_logits True \
  --report_to "wandb" \
  --load_teacher_in_4bit \
  --dtype "bfloat16" \
  --wandb_project "distil-mixtral-dummy"
