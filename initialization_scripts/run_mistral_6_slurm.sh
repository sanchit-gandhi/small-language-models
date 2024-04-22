#!/bin/bash

#SBATCH --partition=hopper-cpu
#SBATCH --name=mistral-init
#SBATCH --mem=1g
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1
#SBATCH -o /fsx/sanchit/logs/init-%j-%x.out

echo "Starting job"
srun python3 run_initialization.py \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --num_hidden_layers "6" \
  --output_dir "./" \
  --hub_model_id "sanchit-gandhi/Mistral-1.5B-Instruct-v0.2" \
  --push_to_hub
wait
