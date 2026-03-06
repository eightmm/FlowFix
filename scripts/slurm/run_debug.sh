#!/bin/bash
#SBATCH --job-name=debug-velocity
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.out
#SBATCH --exclude=gpu3

source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine
PYTHONPATH=/home/jaemin/project/protein-ligand/pose-refine python scripts/debug_velocity.py
