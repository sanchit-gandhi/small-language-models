#!/usr/bin/env bash

accelerate launch --multi_gpu --num_processes=8 --gpu_ids="all" run_distillation.py \
  --model_name_or_path "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer" \
  --teacher_model_name_or_path "sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer" \
  --output_dir "./" \
  --train_dataset_name "HuggingFaceFW/fineweb" \
  --train_dataset_config_name "CC-MAIN-2024-10" \
	--train_split_name "train[1000:]" \
	--eval_split_name "train[:1000]" \
	--preprocessing_num_workers "32" \
	--do_train \
  --do_eval \
	--max_train_samples 1000 \
  --num_train_epochs 2000 \
  --max_eval_samples 1000 \
  --learning_rate 1e-3 \
  --warmup_steps 0 \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 8 \
  --save_strategy "no" \
	--evaluation_strategy "epoch" \
	--logging_steps 1 \
  --overwrite_output_dir \
  --output_router_logits True \
  --report_to "wandb" \
  --load_teacher_in_4bit \
  --gradient_checkpointing \
  --dtype "bfloat16" \
  --wandb_project "distil-mixtral-dummy"
