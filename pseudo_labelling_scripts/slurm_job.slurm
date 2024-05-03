#!/bin/bash
#SBATCH --job-name=distil-mixtral
#SBATCH --nodes=1
# set 24h for job wall time limit
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/sanchit/pseudo-label-logs/%x-%j.out

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

export PROGRAM="./run_cosmo.sh"
export CMD="$PROGRAM"
echo $CMD

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$CMD" 2>&1 | tee -a $SAVE_DIR/logs/main_log.txt


echo "END TIME: $(date)"
