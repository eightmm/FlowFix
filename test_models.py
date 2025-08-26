#!/usr/bin/env python
"""
Test script to verify both FlowFix models work correctly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.flowfix_equivariant import FlowFixEquivariantModel
from models.flowfix_cfm import ConditionalFlowMatching


def test_equivariant_model():
    """Test the original equivariant model."""
    print("\n" + "="*60)
    print("Testing FlowFixEquivariantModel")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = FlowFixEquivariantModel(
        protein_feat_dim=72,
        ligand_feat_dim=14,
        edge_dim=32,
        hidden_scalars=48,
        hidden_vectors=16,
        hidden_dim=128,
        out_dim=256,
        num_layers=2,  # Reduced for testing
        max_ell=2,
        cutoff=10.0,
        time_embedding_dim=64,
        dropout=0.0
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create test data
    batch_size = 2
    n_protein = 50
    n_ligand = 20
    
    protein_coords = torch.randn(n_protein * batch_size, 2, 3, device=device)
    protein_features = torch.randn(n_protein * batch_size, 72, device=device)
    ligand_coords = torch.randn(n_ligand * batch_size, 3, device=device)
    ligand_features = torch.randn(n_ligand * batch_size, 14, device=device)
    
    # Batch indices
    protein_batch = torch.cat([torch.full((n_protein,), i, dtype=torch.long) for i in range(batch_size)]).to(device)
    ligand_batch = torch.cat([torch.full((n_ligand,), i, dtype=torch.long) for i in range(batch_size)]).to(device)
    
    t = torch.rand(batch_size, device=device)
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(
                protein_coords=protein_coords,
                protein_features=protein_features,
                ligand_coords=ligand_coords,
                ligand_features=ligand_features,
                t=t,
                protein_batch=protein_batch,
                ligand_batch=ligand_batch
            )
        
        print(f"✓ Forward pass successful")
        print(f"  Vector field shape: {outputs['vector_field'].shape}")
        print(f"  Vector field mean: {outputs['vector_field'].mean().item():.6f}")
        print(f"  Vector field std: {outputs['vector_field'].std().item():.6f}")
        
        # Test equivariance
        rotation = torch.tensor([[0.8, -0.6, 0.0],
                                 [0.6, 0.8, 0.0],
                                 [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        
        protein_coords_rot = torch.matmul(protein_coords, rotation.T)
        ligand_coords_rot = torch.matmul(ligand_coords, rotation.T)
        
        with torch.no_grad():
            outputs_rot = model(
                protein_coords=protein_coords_rot,
                protein_features=protein_features,
                ligand_coords=ligand_coords_rot,
                ligand_features=ligand_features,
                t=t,
                protein_batch=protein_batch,
                ligand_batch=ligand_batch
            )
        
        vector_field_rot_expected = torch.matmul(outputs['vector_field'], rotation.T)
        error = torch.mean(torch.abs(outputs_rot['vector_field'] - vector_field_rot_expected))
        
        print(f"  Rotation equivariance error: {error.item():.6e}")
        if error < 1e-5:
            print("✓ Equivariance test passed")
        else:
            print("✗ Equivariance test failed")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_cfm_model():
    """Test the Conditional Flow Matching model."""
    print("\n" + "="*60)
    print("Testing ConditionalFlowMatching Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = ConditionalFlowMatching(
        protein_feat_dim=72,
        ligand_feat_dim=14,
        edge_dim=64,
        hidden_scalars=128,
        hidden_vectors=32,
        hidden_dim=256,
        out_dim=256,
        num_layers=2,  # Reduced for testing
        max_ell=2,
        cutoff=10.0,
        time_embedding_dim=128,
        dropout=0.0,
        num_heads=4,
        use_layer_norm=True,
        use_gate=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create test data
    batch_size = 2
    n_protein = 50
    n_ligand = 20
    
    protein_coords = torch.randn(n_protein * batch_size, 2, 3, device=device)
    protein_features = torch.randn(n_protein * batch_size, 72, device=device)
    ligand_coords = torch.randn(n_ligand * batch_size, 3, device=device)
    ligand_features = torch.randn(n_ligand * batch_size, 14, device=device)
    
    # Batch indices
    protein_batch = torch.cat([torch.full((n_protein,), i, dtype=torch.long) for i in range(batch_size)]).to(device)
    ligand_batch = torch.cat([torch.full((n_ligand,), i, dtype=torch.long) for i in range(batch_size)]).to(device)
    
    t = torch.rand(batch_size, device=device)
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(
                protein_coords=protein_coords,
                protein_features=protein_features,
                ligand_coords=ligand_coords,
                ligand_features=ligand_features,
                t=t,
                protein_batch=protein_batch,
                ligand_batch=ligand_batch
            )
        
        print(f"✓ Forward pass successful")
        print(f"  Vector field shape: {outputs['vector_field'].shape}")
        print(f"  Vector field mean: {outputs['vector_field'].mean().item():.6f}")
        print(f"  Vector field std: {outputs['vector_field'].std().item():.6f}")
        print(f"  Confidence shape: {outputs['confidence'].shape}")
        print(f"  Confidence mean: {outputs['confidence'].mean().item():.6f}")
        
        # Test equivariance
        rotation = torch.tensor([[0.8, -0.6, 0.0],
                                 [0.6, 0.8, 0.0],
                                 [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        
        protein_coords_rot = torch.matmul(protein_coords, rotation.T)
        ligand_coords_rot = torch.matmul(ligand_coords, rotation.T)
        
        with torch.no_grad():
            outputs_rot = model(
                protein_coords=protein_coords_rot,
                protein_features=protein_features,
                ligand_coords=ligand_coords_rot,
                ligand_features=ligand_features,
                t=t,
                protein_batch=protein_batch,
                ligand_batch=ligand_batch
            )
        
        vector_field_rot_expected = torch.matmul(outputs['vector_field'], rotation.T)
        error = torch.mean(torch.abs(outputs_rot['vector_field'] - vector_field_rot_expected))
        
        print(f"  Rotation equivariance error: {error.item():.6e}")
        if error < 1e-5:
            print("✓ Equivariance test passed")
        else:
            print("✗ Equivariance test failed")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("Testing Training Step")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use smaller model for testing
    model = FlowFixEquivariantModel(
        protein_feat_dim=72,
        ligand_feat_dim=14,
        edge_dim=32,
        hidden_scalars=32,
        hidden_vectors=8,
        hidden_dim=64,
        out_dim=128,
        num_layers=2,
        max_ell=1,
        cutoff=8.0,
        time_embedding_dim=32,
        dropout=0.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create test batch
    n_protein = 30
    n_ligand = 15
    
    protein_coords = torch.randn(n_protein, 2, 3, device=device)
    protein_features = torch.randn(n_protein, 72, device=device)
    ligand_coords = torch.randn(n_ligand, 3, device=device)
    ligand_features = torch.randn(n_ligand, 14, device=device)
    t = torch.tensor([0.5], device=device)
    
    # Target vector field
    target_vector_field = torch.randn(n_ligand, 3, device=device) * 0.1
    
    try:
        # Forward pass
        outputs = model(
            protein_coords=protein_coords,
            protein_features=protein_features,
            ligand_coords=ligand_coords,
            ligand_features=ligand_features,
            t=t
        )
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(outputs['vector_field'], target_vector_field)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Gradient norm: {grad_norm:.6f}")
        
    except Exception as e:
        print(f"✗ Error in training step: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FlowFix Model Testing Suite")
    print("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available, using CPU")
    
    # Run tests
    test_equivariant_model()
    test_cfm_model()
    test_training_step()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()