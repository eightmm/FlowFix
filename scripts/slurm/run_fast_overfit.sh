#!/bin/bash
#SBATCH --job-name=overfit-fast-sink
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/overfit_fast_sink_%j.out
#SBATCH --error=logs/overfit_fast_sink_%j.out
#SBATCH --exclude=gpu3

source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine

# Enable TF32
export TORCH_ALLOW_TF32_CUBLAS=1

# Run training
python train.py --config configs/train_fast_overfit.yaml
