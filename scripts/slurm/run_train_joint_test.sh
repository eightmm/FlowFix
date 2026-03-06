#!/bin/bash
#SBATCH --job-name=flowfix_test
#SBATCH --output=logs/slurm_%j.out
#SBATCH --partition=test                               # Test partition (gpu2, 2h limit)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a5000:1                              # 1 A5000 GPU
#SBATCH --mem=32G

# Python path
PYTHON=/home/jaemin/miniforge3/envs/torch-2.8/bin/python

# Project directory
PROJECT_DIR=/home/jaemin/project/protein-ligand/pose-refine

# Print job info
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Config: configs/train_joint_test.yaml"
echo "=========================================="

nvidia-smi

echo ""
echo "Python: $PYTHON"
$PYTHON --version
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA avail: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo ""

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

# Single-GPU test run (no distributed)
echo "Starting single-GPU test run..."
$PYTHON train.py --config configs/train_joint_test.yaml

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
