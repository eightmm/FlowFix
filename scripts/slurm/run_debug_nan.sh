#!/bin/bash
#SBATCH --job-name=debug-nan
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_nan_%j.out
#SBATCH --error=logs/debug_nan_%j.out
#SBATCH --exclude=gpu3

# Debug NaN issue in validation
# Tests the BatchNorm fix by running validation on checkpoint

source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine

# Enable TF32
export TORCH_ALLOW_TF32_CUBLAS=1

# Run debug script
python scripts/debug_nan.py
