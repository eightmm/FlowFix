#!/bin/bash
#SBATCH --job-name=flowfix_viz
#SBATCH --output=logs/viz_%j.out
#SBATCH --error=logs/viz_%j.err
#SBATCH --partition=g4090_short,6000ada_short,h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "=============================================="
echo "FlowFix Trajectory Visualization"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

cd /home/sim/project/flowfix
mkdir -p logs

PYTHON=/home/sim/miniconda3/envs/protein-ligand/bin/python

echo ""
echo "Running trajectory visualization..."
$PYTHON -u scripts/analysis/visualize_trajectory.py

echo ""
echo "=============================================="
echo "Visualization completed!"
echo "End time: $(date)"
echo "=============================================="
