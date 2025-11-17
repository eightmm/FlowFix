# FlowFix: SE(3)-Equivariant Flow Matching for Protein-Ligand Pose Refinement

## Project Overview
FlowFix refines perturbed protein-ligand binding poses back to crystal structures using SE(3)-equivariant flow matching. The model learns velocity fields in a decomposed SE(3) + Torsion space (3D translation + 3D rotation + M torsion angles) instead of 3N Cartesian coordinates, achieving 3-10x dimension reduction while preserving physical constraints.

## 🎯 NEW: SE(3) + Torsion Decomposition (November 2024)

**Status**: ✅ Implemented and verified - Ready for training

### Key Innovation
Instead of learning in 3N-dimensional Cartesian space, we decompose molecular poses into:
- **Translation [3D]**: Move molecule's center of mass
- **Rotation [3D, SO(3)]**: Rotate molecule's orientation
- **Torsion [M]**: Rotate M rotatable bonds

**Benefits**:
- 3-10x dimension reduction (e.g., 132D → 28D for typical drug molecules)
- Physical constraints automatically preserved (bond lengths/angles)
- Interpretable degrees of freedom
- Follows DiffDock's successful approach for molecular docking

### Data Format
Each ligand now includes SE(3) + Torsion decomposition:
```python
{
    'coord': [N, 3],                    # Docked pose
    'crystal_coord': [N, 3],            # Crystal structure

    # Torsion decomposition
    'mask_rotate': [E, N],              # Edge-wise rotation mask
    'rotatable_edges': [M],             # Rotatable edge indices
    'torsion_angles_x0': [M],           # Docked torsions (rad)
    'torsion_angles_x1': [M],           # Crystal torsions (rad)
    'torsion_changes': [M],             # Target changes (rad)

    # Rigid body decomposition
    'translation': [3],                 # Translation vector (Å)
    'rotation': [3],                    # Rotation axis-angle (rad)
}
```

### Documentation
- **`claudedocs/implementation_summary.md`**: Quick reference guide
- **`claudedocs/torsion_decomposition_analysis.md`**: Detailed analysis and limitations
- **`tests/verify_decomposition.py`**: Verification script

### Transformation Order
✅ Correct order: **Torsion → Translation → Rotation**

Expected residual error after decomposition: ~1.5-2Å RMSD (due to torsion coupling - this is normal and acceptable for learning)

## 🧬 NEW: Chain-wise ESM Embeddings (November 2024)

**Status**: ✅ Implemented and tested - Ready for use

### Overview
FlowFix now supports chain-wise sequence extraction and ESM embedding generation using Featurizer's `get_sequence_by_chain()` method. Successfully tested on 10 PDB samples with 100% success rate.

### Key Features
- **Chain-wise Sequence Extraction**: Extract 1-letter amino acid sequences for each chain
- **ESM Integration**: Generate ESMC or ESM3 embeddings for protein sequences
- **GPU Acceleration**: Fast processing (~0.5s per PDB file with ESMC 300M)
- **Multi-chain Support**: Correctly handles proteins with multiple chains
- **Easy API**: Simple Python functions and command-line scripts

### Quick Usage

```python
from data.protein_feat import extract_chain_embeddings

# Extract sequences + generate ESM embeddings
embeddings = extract_chain_embeddings(
    pdb_path="protein.pdb",
    model_name="esmc_300m",  # 960-dim embeddings
    device="cuda"
)

# Access results
for chain_id, emb_dict in embeddings.items():
    seq = emb_dict['sequence']           # "MKLVFF..."
    per_res = emb_dict['per_residue']    # [L, 960]
    mean_emb = emb_dict['mean']          # [960]
```

### Command Line

```bash
# Batch process multiple PDB files
python scripts/batch_test_chain_embeddings.py \
    --data_dir /mnt/data/PROJECT/pdbbind_redocked_pose \
    --num_samples 100 \
    --model_name esmc_300m \
    --device cuda \
    --output_dir ./embeddings
```

### Available Models
- **ESMC 300M**: 960-dim embeddings (recommended, fastest)
- **ESMC 600M**: 1152-dim embeddings
- **ESM3**: 1536-dim embeddings (slower)

### Documentation
- **`claudedocs/chain_embeddings_quickstart.md`**: Quick start guide with test results
- **`claudedocs/chain_embeddings_guide.md`**: Comprehensive API documentation
- **`scripts/batch_test_chain_embeddings.py`**: Batch processing script
- **`scripts/example_chain_embeddings.py`**: Complete usage examples

### Test Results
✓ 10/10 PDB files processed successfully (100%)
✓ Processing time: ~0.5s per file (ESMC 300M on GPU)
✓ Handles multi-chain proteins (tested with dimers)

## Architecture Design

### Core Approach
- **Atomwise Flow Matching**: Learn per-atom velocity fields v(x_t, t) that move each ligand atom independently
- **SE(3)-Equivariant EGNN**: Preserves rotation and translation symmetries through equivariant message passing
- **Linear Interpolation Path**: Simple trajectory x_t = (1-t)·x_0 + t·x_1 from perturbed (x_0) to crystal (x_1)
- **Protein-Ligand Cross-Attention**: Context-aware refinement using transformer-based cross-attention

## Key Components

### 1. Data Processing (`data/`)
- `protein/`: Pre-processed protein features (residue-level, 57-dim)
- `ligand/`: Pre-processed ligand features (atom-level, 57-dim)
- `dataset.py`: DGL-based dataset with dynamic perturbation generation
- `fixed_perturbations.py`: Consistent validation perturbations for reproducible evaluation

### 2. Models (`models/`)
- `model.py`: Main SE(3)-equivariant flow matching model
  - `SE3EquivariantFlowModel`: Top-level model coordinating all components
  - `GeometricGraphEmbedding`: EGNN-based encoder for ligand/protein features
  - `AtomwiseFlowDynamics`: Predicts per-atom velocity fields
- `layer.py`: Core building blocks
  - `EGNNNetwork`: Multi-layer EGNN with adaptive conditioning
  - `ContextTransformer`: Cross-attention for protein-ligand interactions
  - `ConditionedTransitionBlock`: Adaptive layer normalization with gating

### 3. Training (`train_direct.py`)
- Direct atomwise flow matching in coordinate space
- Fixed perturbations for validation consistency
- Progressive curriculum with increasing perturbation difficulty

### 4. Scripts (`scripts/`)
- `prepare_pdbbind.py`: Process PDBbind dataset into graph format
- `evaluate_model.py`: Comprehensive evaluation metrics

## Commands

### Environment Setup
```bash
# Python environment path
PYTHON=/home/jaemin/miniforge3/envs/protein-ligand/bin/python
```

### Data Preparation
```bash
# Process PDBbind data with perturbations
$PYTHON scripts/prepare_data.py \
    --data_dir /mnt/data/PLI/PDBbind/general-set \
    --output_dir data/graph \
    --perturbation_scale 2.0  # RMSD perturbation in Angstroms
    --max_samples 1000

# Add SE(3) + Torsion decomposition to existing ligand data
$PYTHON tests/add_torsion_data.py \
    --data_dir train_data \
    --source_dir /mnt/data/PROJECT/pdbbind_redocked_pose \
    --max_pdbs 100

# Verify decomposition implementation
$PYTHON tests/verify_decomposition.py \
    --data_file train_data/10gs/ligands.pt \
    --num_samples 10
```

### Training
```bash
# Train direct flow matching model
$PYTHON train_direct.py --config configs/train_direct.yaml

# Quick test run
$PYTHON train_direct.py --config configs/train_direct.yaml --device cpu

# Resume training
$PYTHON train_direct.py --resume checkpoints/direct_flow/latest.pt
```

### Model Architecture

#### SE3EquivariantFlowModel (Top-level)
- **Purpose**: Coordinates all components for end-to-end flow matching
- **Components**:
  - Separate protein/ligand encoders
  - Atomwise flow dynamics predictor
  - Cross-attention for protein-ligand interactions

#### GeometricGraphEmbedding (Feature Encoder)
- **Input**: 
  - Node features: (N, input_dim) 
  - Node coordinates: (N, 3)
  - Edge features: (E, edge_dim)
- **Processing**: 
  - Feature encoding via MLP
  - 4-layer EGNN with optional coordinate updates
  - Preserves SE(3) equivariance
- **Output**: 
  - Encoded node features: (N, hidden_dim)
  - Original/updated coordinates: (N, 3)

#### AtomwiseFlowDynamics (Velocity Predictor)
- **Input**: 
  - Encoded ligand features: (N, hidden_dim)
  - Current coordinates: (N, 3) at time t
  - Time embedding: Sinusoidal encoding of t
  - Protein context: (B, P, hidden_dim) via cross-attention
  - Target coordinates: (N, 3) for training guidance (optional)
- **Architecture**:
  - Dual EGNN networks:
    - Primary: 6-layer EGNN with coordinate updates for geometric patterns
    - Velocity: 3-layer EGNN specifically for velocity prediction
  - Coordinate difference encoder for explicit displacement modeling
  - ContextTransformer for protein-ligand cross-attention
  - Velocity combiner network merging features and coordinate information
  - Zero-initialized final projection with learnable scales
- **Key Features**:
  - Coordinate updates enabled in EGNN for spatial awareness
  - Explicit encoding of coordinate displacements
  - Weighted combination of feature-based and coordinate-based velocities
  - Separate scale factors for features (1.0) and coordinates (0.5)
- **Output**: 
  - Velocity field: v(x_t, t) ∈ R^(N×3)
  - Updated features: (N, hidden_dim)

### Loss Functions

1. **Weighted Flow Matching Loss**
   ```
   L_flow = mean((v_θ(x_t, t) - v_true) * w(t))²
   ```
   where:
   - x_t = (1-t)·x_0 + t·x_1 (linear interpolation)
   - v_true = x_1 - x_0 (constant velocity field)
   - w(t) = 1 + 2|t - 0.5| (emphasis on endpoints)
   - v_θ: predicted velocity at (x_t, t)

2. **Velocity Regularization**
   ```
   L_reg = 0.01 * mean(||v_θ||²)
   ```
   Encourages smooth velocity fields

3. **Time Sampling Strategy**
   - Mixed sampling: 50% uniform, 25% near t=0, 25% near t=1
   - Better coverage of critical regions
   - Improves convergence at trajectory endpoints

4. **Monitoring Metrics**
   - RMSD at t=1: Final refinement quality
   - Intermediate RMSD: Mid-trajectory accuracy
   - Velocity magnitude: Smoothness indicator

### Training Configuration
- **Batch size**: 64 (DGL handles batching efficiently)
- **Learning rate**: 1e-4 with cosine annealing (min_lr: 1e-6)
- **Optimizer**: Adam with eps=1e-8 for numerical stability
- **Gradient clipping**: 1.0 for stable training
- **Dropout**: 0.1 throughout the network
- **Hidden dimension**: 256 for all transformations

### Validation Strategy
- **Fixed Perturbations**: 5 consistent perturbations per molecule for reproducibility
- **Frequency**: Every 10 epochs
- **Sampling Method**: Euler ODE integration (50 steps)
- **Metrics**:
  - RMSD to crystal structure (primary)
  - Success rates: <2Å, <1Å, <0.5Å
  - Per-perturbation tracking for variance analysis
- **Early Stopping**: Patience of 20 epochs on validation RMSD

## Key Features
1. **Atomwise Velocity Fields**: Per-atom displacements for flexible refinement
2. **SE(3) Equivariance**: EGNN preserves physical symmetries
3. **Protein Context Integration**: Cross-attention for binding site awareness
4. **Stable Training**: Zero-initialized outputs with learnable scaling
5. **Reproducible Validation**: Fixed perturbations for consistent evaluation
6. **Adaptive Conditioning**: Layer normalization with learned gating

## Dataset Structure
```
data/
├── ligand/
│   └── [pdb_id].pt    # Ligand graphs with:
│       ├── x          # Atom features (57-dim)
│       ├── pos        # Crystal coordinates
│       └── edge_attr  # Edge features
├── protein/
│   └── [pdb_id].pt    # Protein graphs with:
│       ├── x          # Residue features (57-dim)
│       ├── pos        # CA coordinates
│       └── edge_attr  # Edge features
├── fixed_perturbations/  # Validation perturbations
│   └── [pdb_id]_pert_[N].pt
└── splits.json        # Train/val/test splits
```

## Checkpointing
Models are saved in `checkpoints/se3_flow/` with:
- `best.pt`: Best validation RMSD
- `latest.pt`: Most recent checkpoint  
- `epoch_N.pt`: Periodic checkpoints

## Monitoring & Metrics
- **Training Metrics**: 
  - Flow matching loss (MSE of velocity fields)
  - RMSD to crystal (monitoring only)
  - Gradient norms for stability tracking
- **Validation Metrics**: 
  - Mean RMSD across all perturbations
  - Success rates: <2Å, <1Å, <0.5Å
  - Per-molecule RMSD distribution
  - Convergence trajectory visualization

## Implementation Notes

### EGNN Coordinate Handling
- **Training**: Coordinates ARE updated in EGNN layers for spatial awareness
- **Velocity Prediction**: Dual approach combining:
  - Feature-based velocity from learned representations
  - Coordinate-based velocity from EGNN updates
  - Weighted combination: v_final = v_features + 0.5 * v_coordinates
- **Inference**: ODE integration of combined velocity fields

### Cross-Attention Mechanism
- Query: Ligand atom features
- Key/Value: Protein residue features  
- Masking: Handles variable protein sizes
- Output: Context-aware ligand features

### Numerical Stability
- Zero initialization for final velocity projection
- Learnable scale factor (initialized to 1.0)
- Gradient clipping at 1.0
- Adam eps=1e-8 for small gradients