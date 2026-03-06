#!/bin/bash
#SBATCH --job-name=overfit-quick
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/overfit_quick_%j.out
#SBATCH --error=logs/overfit_quick_%j.out
#SBATCH --exclude=gpu3

# Quick overfit test to verify BatchNorm fix
# Runs shorter training with more frequent validation to test NaN fix

source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine

# Enable TF32
export TORCH_ALLOW_TF32_CUBLAS=1

# Run with quick validation (validate every 50 epochs instead of 100)
python train.py \
    --config configs/train_overfit_test.yaml \
    --name overfit-quick-batchnorm-fix \
    --training.num_epochs 200 \
    --training.validation.frequency 50
