#!/bin/bash
#SBATCH --job-name=debug-ode
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/debug_ode_%j.out
#SBATCH --error=logs/debug_ode_%j.out

# Detailed ODE analysis script
source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8

cd /home/jaemin/project/protein-ligand/pose-refine

python scripts/debug_nan.py
