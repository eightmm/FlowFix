# FlowFix: Protein-Ligand Binding Pose Refinement

## Project Overview
FlowFix is a flow matching-based system for refining protein-ligand binding poses to achieve crystal-like accuracy using SE(3)-equivariant neural networks with cuEquivariance library.

## Key Components

### 1. Data Processing (`data/`)
- `protein_features.py`: Protein feature extraction at residue level
- `ligand_features.py`: Ligand atom-level feature extraction  
- `dataset.py`: PyTorch dataset with pose perturbation for training

### 2. Model (`models/`)
- `flowfix_equivariant.py`: Main SE(3)-equivariant model with cuEquivariance

### 3. Training (`train_equivariant.py`)
- Flow matching training with physics-based guidance
- Gradient accumulation for effective batch size

### 4. Scripts (`scripts/`)
- `prepare_data.py`: Process PDBbind dataset into graph format

## Commands

### Environment Setup
```bash
# Python environment path
PYTHON=/home/jaemin/miniforge3/envs/flowfix/bin/python
```

### Data Preparation
```bash
# Process PDBbind data
$PYTHON scripts/prepare_data.py \
    --data_dir /mnt/data/PLI/PDBbind/general-set \
    --output_dir data/graph \
    --max_samples 100  # For testing
```

### Training
```bash
# Train model
$PYTHON train.py

# With custom config
$PYTHON train.py --config configs/train_equivariant_config.yaml

# Resume training
$PYTHON train.py --resume checkpoints/equivariant/latest.pt
```

### Model Architecture
- **Input**: Protein residue features (72-dim) + Ligand atom features (14-dim)
- **Processing**: SE(3)-equivariant message passing with cuEquivariance
- **Output**: Vector field for ligand refinement + physics forces

### Configuration (`configs/train_equivariant_config.yaml`)
- Batch size: 1 (due to varying molecular sizes)
- Gradient accumulation: 8 steps
- Learning rate: 5e-4 with OneCycle schedule
- Cutoff distance: 10Å for protein-ligand interactions

## Key Features
1. **SE(3) Equivariance**: Preserves rotational and translational symmetry
2. **Physics Guidance**: Incorporates force field calculations
3. **Flow Matching**: Learns transformation from perturbed to crystal poses
4. **Efficient Processing**: Handles full protein-ligand complexes

## Dataset Structure
```
data/graph/
├── [pdb_id].pt  # Processed protein-ligand complexes
└── splits.json  # Train/val/test splits
```

## Checkpointing
Models are saved in `checkpoints/equivariant/` with:
- `best_model.pt`: Best validation loss
- `latest.pt`: Most recent checkpoint
- `checkpoint_epoch_N.pt`: Periodic saves