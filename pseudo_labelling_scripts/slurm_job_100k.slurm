#!/bin/bash
#SBATCH --job-name=distil-mistral
#SBATCH --nodes=1
# set 24h for job wall time limit
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/sanchit/logs/%x-%j.out

set -x -e

# START EDIT
source ~/.bashrc
source /fsx/sanchit/miniconda3/bin/activate venv

LOG_PATH="/fsx/sanchit/logs/main_log.txt"
SAVE_DIR="/fsx/sanchit"
# END EDIT

echo "START TIME: $(date)"

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`

# From https://i.hsfzxjy.site/2021-03-10-obtain-a-random-unused-tcp-port-with-bash/
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
MASTER_PORT=$(unused_port)

# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export LAUNCHER="python -u -m accelerate.commands.launch --config_file ./accelerate_config.yaml"

export PROGRAM="./run_pseudo_labelling.py ./config_cosmo_websamples.yaml"
export CMD="$LAUNCHER $PROGRAM"
echo $CMD

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$CMD" 2>&1 | tee -a $SAVE_DIR/logs/main_log.txt


# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code

# SRUN_ARGS=" \
#     --wait=60 \
#     --kill-on-bad-exit=1 \
#     "
#
# # py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
# clear; srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $SAVE_DIR/logs/main_log.txt

echo "END TIME: $(date)"
