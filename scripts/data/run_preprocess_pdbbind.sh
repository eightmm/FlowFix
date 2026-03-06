#!/bin/bash
#SBATCH --job-name=preprocess_pdbbind
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --partition=mix_short,a5000_short,g3090_short,g4090_short,6000ada_short,h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00

# PDBbind Data Preprocessing Script
# Converts protein PDB and ligand mol2 to graph format

echo "=============================================="
echo "PDBbind Data Preprocessing"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Change to project directory
cd /home/sim/project/flowfix

# Create logs directory
mkdir -p logs

# Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate protein-ligand

# Python path
PYTHON=/home/sim/miniconda3/envs/protein-ligand/bin/python

# Input/Output directories
PDBBIND_DIR=/appl/DB/PLI/PDBbind/refined-set
OUTPUT_DIR=/home/sim/project/flowfix/data/pdbbind_refined

# Parameters
PROTEIN_CUTOFF=6.0
LIGAND_CUTOFF=10.0
NUM_WORKERS=16

echo ""
echo "Configuration:"
echo "  Input: $PDBBIND_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Protein cutoff: $PROTEIN_CUTOFF A"
echo "  Ligand cutoff: $LIGAND_CUTOFF A"
echo "  Workers: $NUM_WORKERS"
echo ""

# Run preprocessing
$PYTHON -u scripts/preprocess_pdbbind.py \
    --data_dir $PDBBIND_DIR \
    --output_dir $OUTPUT_DIR \
    --protein_cutoff $PROTEIN_CUTOFF \
    --ligand_cutoff $LIGAND_CUTOFF \
    --num_workers $NUM_WORKERS \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42

echo ""
echo "=============================================="
echo "Preprocessing completed!"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=============================================="
