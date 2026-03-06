#!/bin/bash
#SBATCH --job-name=full_val_infer
#SBATCH --output=logs/full_val_infer_%j.out
#SBATCH --error=logs/full_val_infer_%j.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

echo "=============================================="
echo "FlowFix Full Validation Inference (All Poses)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

cd /home/jaemin/project/protein-ligand/pose-refine
export PYTHONPATH=/home/jaemin/project/protein-ligand/pose-refine:$PYTHONPATH

mkdir -p logs inference_results

PYTHON=/home/jaemin/miniforge3/envs/torch-2.8/bin/python

# Checkpoint: use argument or default to latest
CHECKPOINT=${1:-save/rectified-flow-full-v4/checkpoints/latest.pt}
OUTPUT=inference_results/full_validation_results.json

echo ""
echo "Configuration:"
echo "  Python: $PYTHON"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT"
echo ""

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Starting full validation inference (200 val PDBs, all poses)..."
$PYTHON -u scripts/analysis/infer_full_validation.py \
    --config configs/train_joint.yaml \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT"

echo ""
echo "=============================================="
echo "Inference completed!"
echo "End time: $(date)"
echo "=============================================="
