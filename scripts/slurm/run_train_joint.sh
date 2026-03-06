#!/bin/bash
#SBATCH --job-name=flowfix_joint
#SBATCH --output=logs/slurm_%j.out
#SBATCH --partition=6000ada                            # GPU partition
#SBATCH --nodes=1                                      # Single node
#SBATCH --ntasks-per-node=1                            # One task per node
#SBATCH --cpus-per-task=32                             # CPUs for data loading
#SBATCH --gres=gpu:8                                   # Request 8 GPUs
#SBATCH --mem=0                                        # Request all available memory
#SBATCH --time=30-00:00:00                              # 30 days time limit
#SBATCH --exclude=gpu3                                  # Exclude gpu3 (driver mismatch)

# Python path
PYTHON=/home/jaemin/miniforge3/envs/torch-2.8/bin/python

# Project directory
PROJECT_DIR=/home/jaemin/project/protein-ligand/pose-refine

# Resume checkpoint (set this to resume training, leave empty for fresh start)
CHECKPOINT_PATH=""

# Print job info
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $PROJECT_DIR"
echo "Config: configs/train_joint.yaml"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="

# Print GPU info
nvidia-smi

# Check Python and CUDA
echo ""
echo "Python: $PYTHON"
echo "Python version: $($PYTHON --version)"
echo "PyTorch version: $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $($PYTHON -c 'import torch; print(torch.version.cuda)')"
echo "Number of CUDA devices: $($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Change to project directory
cd "$PROJECT_DIR"

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Set NCCL environment variables
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Set number of GPUs
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_GPUS_PER_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_PER_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    NUM_GPUS=$(echo $SLURM_JOB_GPUS | awk -F',' '{print NF}')
else
    NUM_GPUS=8
    echo "Warning: SLURM GPU variables not found, using default NUM_GPUS=8"
fi

# Build resume argument if checkpoint path is set
RESUME_ARG=""
if [ -n "$CHECKPOINT_PATH" ]; then
    RESUME_ARG="--resume $CHECKPOINT_PATH"
    echo "Resuming from checkpoint: $CHECKPOINT_PATH"
fi

# Run distributed training using torch.distributed.run
echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
echo "Command: python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train.py --config configs/train_joint.yaml $RESUME_ARG"
echo ""

$PYTHON -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    --config configs/train_joint.yaml \
    $RESUME_ARG

# Print end time
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
