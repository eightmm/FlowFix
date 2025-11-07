# FlowFix: SE(3)-Equivariant Flow Matching for Protein-Ligand Pose Refinement

FlowFix refines perturbed protein-ligand binding poses back to crystal structures using SE(3)-equivariant flow matching.

## Overview

- **Goal**: Refine docked protein-ligand poses to high-accuracy crystal structures
- **Method**: SE(3)-equivariant flow matching with cuequivariance
- **Architecture**: Protein-Ligand interaction network with atomwise velocity prediction

## Installation

### Environment Setup

```bash
# Create conda environment
conda create -n protein-ligand python=3.11
conda activate protein-ligand

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia

# Install PyTorch Geometric
pip install torch-geometric torch-cluster torch-scatter torch-sparse

# Install cuequivariance
pip install cuequivariance cuequivariance-torch

# Install other dependencies
pip install -r requirements.txt
```

## Project Structure

```
flowfix/
├── src/
│   ├── data/
│   │   ├── dataset.py          # FlowFixDataset with new ligand format
│   │   └── feat.py             # Feature extraction utilities
│   ├── models/
│   │   ├── flowmatching.py     # Main flow matching model
│   │   ├── network.py          # Equivariant networks
│   │   ├── cue_layers.py       # Cuequivariance layers
│   │   └── torch_layers.py     # Standard PyTorch layers
│   └── utils/
│       ├── training_utils.py   # Training helpers
│       ├── experiment.py       # Experiment management
│       └── ...                 # Other utilities
├── configs/
│   └── train.yaml              # Training configuration
├── tests/
│   └── train_overfit.py        # Overfitting test
├── training.py                 # Main training script
└── requirements.txt            # Python dependencies
```

## Data Format

### Ligand Data (ligands.pt)
List of 60 docked poses per PDB, each containing:
- `edges`: [2, E] edge indices
- `node_feats`: [N, 122] atom features
- `edge_feats`: [E, 44] edge features
- `coord`: [N, 3] docked pose coordinates (x₀)
- `crystal_coord`: [N, 3] crystal structure coordinates (x₁)
- `distance_lower_bounds`: [N, N] distance constraints
- `distance_upper_bounds`: [N, N] distance constraints

### Protein Data (protein.pt)
- `node['coord']`: Residue coordinates
- `node['node_scalar_features']`: Residue scalar features
- `node['node_vector_features']`: Residue vector features (31 x 3D vectors)
- `edge['edges']`: Edge indices
- `edge['edge_scalar_features']`: Edge scalar features
- `edge['edge_vector_features']`: Edge vector features (8 x 3D vectors)

## Training

```bash
# Train model
python training.py --config configs/train.yaml

# Overfit test
python tests/train_overfit.py
```

## Key Features

- **Memory-efficient**: Samples one pose per PDB per epoch
- **SE(3) Equivariant**: Preserves rotation and translation symmetries
- **Pocket Extraction**: 12Å cutoff for binding site residues
- **Vector Features**: Utilizes directional protein features (31 node + 8 edge vectors)
- **Flow Matching**: Linear interpolation from docked to crystal pose

## Model Architecture

1. **Protein Network**: Encodes fixed protein structure with vector features
2. **Ligand Network**: Encodes ligand at current time t
3. **Interaction Network**: Protein-ligand cross-attention
4. **Velocity Predictor**: Atomwise velocity field for pose refinement

## License

MIT License
