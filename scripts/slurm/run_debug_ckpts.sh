#!/bin/bash
#SBATCH --job-name=debug-train
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_train_%j.out
#SBATCH --error=logs/debug_train_%j.out

source /home/jaemin/miniforge3/etc/profile.d/conda.sh
conda activate torch-2.8
export PYTHONPATH=$PYTHONPATH:.

echo "============================================================"
echo "Analyzing Job 203 (32-Sample Fixed) - Training Sample 10gs"
echo "============================================================"
python scripts/debug_ckpt.py --ckpt save/overfit-test-32-fixed/checkpoints/latest.pt --config configs/train_overfit_32.yaml --split train --pdb 10gs

echo -e "\n\n============================================================"
echo "Analyzing Job 226 (Rectified Flow) - Training Sample 10gs (Epoch 20)"
echo "============================================================"
python scripts/debug_ckpt.py --ckpt save/overfit-fast-sink/checkpoints/epoch_0020.pt --config configs/train_rectified_flow.yaml --split train --pdb 10gs
