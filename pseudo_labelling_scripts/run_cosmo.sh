#!/usr/bin/env bash

source ~/.bashrc

CONFIGS=('auto_math_text' 'khanacademy' 'openstax' 'stanford' 'stories' 'web_samples_v1' 'web_samples_v2' 'wikihow')

for config in "${CONFIGS[@]}"; do
  if [[ $config == "stanford" ]]; then
    max_eval_samples=1000000
  else
    max_eval_samples=500000
  fi
  accelerate launch --config_file ./accelerate_config.yaml run_pseudo_labelling.py \
    --model_name_or_path "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    --per_device_eval_batch_size 16 \
    --dataset_name "HuggingFaceTB/cosmopedia" \
    --dataset_config_name "${config}" \
    --dataset_split_name "train" \
    --preprocessing_num_workers 32 \
    --output_dir "./cosmopedia-${config}" \
    --hub_dataset_id "cosmopedia-logprobs" \
    --push_to_hub \
    --report_to wandb \
    --wandb_project "distil-mixtral-logprobs" \
    --load_in_4bit \
    --max_eval_samples $max_eval_samples
done
