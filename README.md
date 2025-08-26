# FlowFix: Flow Matching for Protein-Ligand Binding Pose Refinement

## Overview
FlowFix implements SE(3)-equivariant flow matching models for refining protein-ligand binding poses to achieve crystal-like accuracy.

## Models

### 1. FlowFixEquivariantModel (`models/flowfix_equivariant.py`)
- Original SE(3)-equivariant model using cuEquivariance
- Stable and tested implementation
- Recommended for initial experiments

### 2. ConditionalFlowMatching (`models/flowfix_cfm.py`)
- Advanced conditional flow matching with optimal transport
- Enhanced features: learnable Fourier time encoding, gating, multi-scale processing
- Higher capacity model for better performance

## Quick Start

### Installation
```bash
# Ensure cuEquivariance is installed
pip install cuequivariance-torch
```

### Data Preparation
```bash
# Process PDBbind dataset
python scripts/prepare_data.py \
    --data_dir /path/to/PDBbind \
    --output_dir data/graph \
    --max_samples 100  # For testing
```

### Training

#### Option 1: Train Original Model
```bash
python train.py --config configs/train.yaml
```

#### Option 2: Train CFM Model (Advanced)
```bash
python train_cfm.py --config configs/train_cfm.yaml
```

### Testing
```bash
# Test both models
python test_models.py

# Test training functionality
python test_train.py
```

## Configuration

### Key Parameters (`configs/train.yaml` or `configs/train_cfm.yaml`)

```yaml
model:
  num_layers: 6-8        # Number of equivariant layers
  cutoff: 10.0           # Distance cutoff in Angstroms
  hidden_scalars: 48-128 # Scalar feature channels
  hidden_vectors: 16-32  # Vector feature channels

training:
  batch_size: 4-8
  learning_rate: 0.0001-0.0005
  num_epochs: 200-300

data:
  perturbation:
    translation_std: 5.0  # Translation noise (Å)
    rotation_std: 1.5     # Rotation noise (radians)
    min_rmsd: 5.0         # Minimum perturbation
    max_rmsd: 15.0        # Maximum perturbation
```

## Model Features

### SE(3) Equivariance
- Rotation and translation equivariant processing
- Preserves geometric symmetries
- Tested with equivariance checks

### Flow Matching
- Learns transformation from perturbed to crystal poses
- Time-conditioned vector field prediction
- Smooth trajectory generation

### Architecture
- **Input**: Protein residue features + Ligand atom features
- **Processing**: SE(3)-equivariant message passing
- **Output**: Vector field for ligand refinement

## File Structure
```
FlowFix/
├── models/
│   ├── flowfix_equivariant.py  # Original model
│   └── flowfix_cfm.py           # CFM model
├── train.py                     # Training script (original)
├── train_cfm.py                 # Training script (CFM)
├── configs/
│   ├── train.yaml               # Config for original model
│   └── train_cfm.yaml           # Config for CFM model
├── data/
│   ├── dataset.py               # PyTorch dataset
│   └── batch_dataset.py        # Batched collation
└── utils/
    ├── ema.py                   # Exponential moving average
    └── initialization.py        # Weight initialization
```

## Performance Tips

1. **Start with smaller models**: Use fewer layers (2-4) for initial experiments
2. **Gradient accumulation**: Increase effective batch size without memory issues
3. **EMA**: Use exponential moving average for stable inference
4. **Mixed precision**: Enable for faster training (if supported)

## Common Issues

### Out of Memory
- Reduce `batch_size` in config
- Reduce `num_layers` or `hidden_scalars/vectors`
- Enable gradient accumulation

### No Edges Found
- Increase `cutoff` distance
- Check data preprocessing
- Ensure protein and ligand are close enough

### Zero Gradients
- Model is initialized and working
- Check learning rate (not too small)
- Verify data is loaded correctly

## Citation
This implementation uses the cuEquivariance library for SE(3)-equivariant operations.

## License
MIT