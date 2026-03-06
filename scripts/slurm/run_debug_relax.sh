#!/bin/bash
#SBATCH --job-name=debug-relax
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_relax_%j.out
#SBATCH --error=logs/debug_relax_%j.out

source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine
PYTHONPATH=/home/jaemin/project/protein-ligand/pose-refine python scripts/debug_ckpt.py --ckpt save/overfit-fast-sink/checkpoints/latest.pt --config configs/train_rectified_flow.yaml --split train --pdb 10gs --relax
