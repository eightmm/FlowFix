#!/bin/bash
#SBATCH --job-name=train-flow-full
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1  # torchrun handles launching processes
#SBATCH --cpus-per-task=64   # 8 GPUs * 8 CPUs
#SBATCH --mem=256G           # Increased memory for 8 processes
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.out

source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine

CONFIG_PATH="configs/train_rectified_flow_full.yaml"
# Extract run name from config
RUN_NAME=$(grep "name:" $CONFIG_PATH | grep -v "#" | head -n 1 | awk -F'"' '{print $2}')
CKPT_DIR="save/${RUN_NAME}/checkpoints"
LATEST_CKPT="${CKPT_DIR}/latest.pt"

# NCCL / DDP Settings for stability
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Fix potential P2P hangs on some clusters
export NCCL_IB_DISABLE=1   # Fix potential InfiniBand hangs
export OMP_NUM_THREADS=1

# Use torchrun for Multi-GPU (DDP) launch
# --standalone enables single-node multi-gpu without needing rdzv endpoint manually
CMD="torchrun --standalone --nproc_per_node=8 train.py --config $CONFIG_PATH"

if [ -f "$LATEST_CKPT" ]; then
    echo "🔄 Found existing checkpoint at $LATEST_CKPT. Resuming..."
    CMD="$CMD --resume $LATEST_CKPT"
else
    echo "🆕 No checkpoint found. Starting fresh."
fi

echo "🚀 Running on 8 GPUs: $CMD"
PYTHONPATH=. $CMD
