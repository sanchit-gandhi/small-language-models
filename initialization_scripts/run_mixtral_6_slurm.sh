#!/bin/bash

#SBATCH --partition=hopper-cpu
#SBATCH --name=mixtral-init
#SBATCH --mem=1g
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1
#SBATCH -o /fsx/sanchit/logs/init-%j-%x.out

echo "Starting job"
srun python3 run_initialization.py \
  --model_name_or_path "mistralai/Mixtral-8x7B-v0.1" \
  --num_hidden_layers "6" \
  --output_dir "./" \
  --hub_model_id "sanchit-gandhi/mixtral-8x1.5B" \
  --push_to_hub
wait
