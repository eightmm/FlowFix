#!/usr/bin/env python
"""
Quick training test to verify the models can train properly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.flowfix_equivariant import FlowFixEquivariantModel
from models.flowfix_cfm import ConditionalFlowMatching
from train_cfm import CFMLoss


def create_dummy_batch(device='cuda', batch_size=2):
    """Create a dummy batch for testing."""
    n_protein = 50
    n_ligand = 20
    
    # Create batched data
    protein_coords = torch.randn(n_protein * batch_size, 2, 3, device=device)
    protein_features = torch.randn(n_protein * batch_size, 72, device=device)
    
    # Create perturbed and target ligand coordinates
    ligand_coords_0 = torch.randn(n_ligand * batch_size, 3, device=device)  # Crystal/target
    noise = torch.randn_like(ligand_coords_0) * 2.0  # Perturbation
    ligand_coords_t = ligand_coords_0 + noise  # Perturbed
    
    ligand_features = torch.randn(n_ligand * batch_size, 14, device=device)
    
    # Batch indices
    protein_batch = torch.cat([torch.full((n_protein,), i, dtype=torch.long) for i in range(batch_size)]).to(device)
    ligand_batch = torch.cat([torch.full((n_ligand,), i, dtype=torch.long) for i in range(batch_size)]).to(device)
    
    # Time values
    t = torch.rand(batch_size, device=device)
    
    # Vector field (target)
    vector_field = ligand_coords_0 - ligand_coords_t
    
    return {
        'protein_coords': protein_coords,
        'protein_features': protein_features,
        'ligand_coords_t': ligand_coords_t,
        'ligand_coords_0': ligand_coords_0,
        'ligand_features': ligand_features,
        'vector_field': vector_field,
        't': t,
        'protein_batch': protein_batch,
        'ligand_batch': ligand_batch,
        'batch_size': batch_size
    }


def test_training_original():
    """Test training with the original model."""
    print("\n" + "="*60)
    print("Testing Original Model Training")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = FlowFixEquivariantModel(
        protein_feat_dim=72,
        ligand_feat_dim=14,
        edge_dim=32,
        hidden_scalars=48,
        hidden_vectors=16,
        hidden_dim=128,
        out_dim=256,
        num_layers=2,
        max_ell=2,
        cutoff=10.0,
        time_embedding_dim=64,
        dropout=0.0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()
    
    # Training loop
    model.train()
    losses = []
    
    for step in range(5):
        batch = create_dummy_batch(device)
        
        # Forward pass
        outputs = model(
            protein_coords=batch['protein_coords'],
            protein_features=batch['protein_features'],
            ligand_coords=batch['ligand_coords_t'],
            ligand_features=batch['ligand_features'],
            t=batch['t'],
            protein_batch=batch['protein_batch'],
            ligand_batch=batch['ligand_batch']
        )
        
        # Compute loss
        loss = criterion(outputs['vector_field'], batch['vector_field'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step+1}: Loss = {loss.item():.6f}")
    
    # Check if loss decreases
    if losses[-1] < losses[0]:
        print("✓ Training successful - loss decreased")
    else:
        print("✗ Training may have issues - loss did not decrease")


def test_training_cfm():
    """Test training with the CFM model."""
    print("\n" + "="*60)
    print("Testing CFM Model Training")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = ConditionalFlowMatching(
        protein_feat_dim=72,
        ligand_feat_dim=14,
        edge_dim=64,
        hidden_scalars=128,
        hidden_vectors=32,
        hidden_dim=256,
        out_dim=256,
        num_layers=2,
        max_ell=2,
        cutoff=10.0,
        time_embedding_dim=128,
        dropout=0.0,
        num_heads=4,
        use_layer_norm=True,
        use_gate=True
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CFMLoss(sigma_min=0.001, use_ot=True, lambda_reg=0.01)
    
    # Training loop
    model.train()
    losses = []
    
    for step in range(5):
        batch = create_dummy_batch(device)
        
        # Forward pass
        outputs = model(
            protein_coords=batch['protein_coords'],
            protein_features=batch['protein_features'],
            ligand_coords=batch['ligand_coords_t'],
            ligand_features=batch['ligand_features'],
            t=batch['t'],
            protein_batch=batch['protein_batch'],
            ligand_batch=batch['ligand_batch']
        )
        
        # Compute loss
        loss_dict = criterion(
            pred_v=outputs['vector_field'],
            x0=batch['ligand_coords_t'],
            x1=batch['ligand_coords_0'],
            t=batch['t']
        )
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step+1}: Loss = {loss.item():.6f}, RMSD = {loss_dict['rmsd'].item():.4f}")
    
    # Check if loss decreases
    if losses[-1] < losses[0]:
        print("✓ Training successful - loss decreased")
    else:
        print("✗ Training may have issues - loss did not decrease")


def test_config_loading():
    """Test loading and using configuration files."""
    print("\n" + "="*60)
    print("Testing Configuration Loading")
    print("="*60)
    
    config_paths = [
        'configs/train.yaml',
        'configs/train_cfm.yaml'
    ]
    
    for config_path in config_paths:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Loaded {config_path}")
            print(f"  Model layers: {config['model']['num_layers']}")
            print(f"  Batch size: {config['training']['batch_size']}")
            print(f"  Learning rate: {config['training']['learning_rate']}")
        else:
            print(f"✗ Config not found: {config_path}")


def main():
    """Run all training tests."""
    print("\n" + "="*60)
    print("FlowFix Training Test Suite")
    print("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    # Run tests
    test_training_original()
    test_training_cfm()
    test_config_loading()
    
    print("\n" + "="*60)
    print("Training tests completed!")
    print("="*60)
    print("\nTo start actual training, run:")
    print("  python train.py --config configs/train.yaml")
    print("  python train_cfm.py --config configs/train_cfm.yaml")


if __name__ == "__main__":
    main()