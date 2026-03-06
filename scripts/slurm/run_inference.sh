#!/bin/bash
#SBATCH --job-name=flowfix_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=g4090_short,6000ada_short,h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# FlowFix Inference Script (Direct Python path, no conda activate)

echo "=============================================="
echo "FlowFix Inference"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Change to project directory
cd /home/sim/project/flowfix

# Create logs directory if it doesn't exist
mkdir -p logs

# Direct Python path (no conda activate)
PYTHON=/home/sim/miniconda3/envs/protein-ligand/bin/python

# Configuration
CONFIG=configs/inference.yaml

# Checkpoint path (use argument or default)
if [ -n "$1" ]; then
    CHECKPOINT=$1
else
    CHECKPOINT=save/flowfix_20251209_141538/checkpoints/epoch_0800.pt
fi

# Output directory
OUTPUT_DIR=inference_results/$(date +%Y%m%d_%H%M%S)

echo ""
echo "Configuration:"
echo "  Python: $PYTHON"
echo "  Config file: $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Run inference with unbuffered output
echo "Starting inference..."
$PYTHON -u inference.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --device cuda

echo ""
echo "=============================================="
echo "Inference completed!"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
