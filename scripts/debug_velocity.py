#!/usr/bin/env python
"""
Debug script to check model velocity predictions.
"""
import os
import sys
import torch
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from src.utils.model_builder import build_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint from overfit-test-8
    checkpoint_path = 'save/overfit-test-8/checkpoints/latest.pt'
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config from checkpoint if available, otherwise from file
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Using config from checkpoint")
    else:
        config_path = 'configs/train_joint.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"Using config from {config_path}")
    
    # Build model
    model = build_model(config['model'], device)
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded, epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create dataset
    dataset = FlowFixDataset(
        data_dir='train_data',
        split_file='train_data/splits_overfit_tiny.json',
        split='train',  # Use train split to get fixed pose
        seed=42,
        fix_pose=True,  # Use pose 1 (easier)
    )
    
    # Get one sample
    sample = dataset[0]
    batch = collate_flowfix_batch([sample])
    
    pdb_id = batch['pdb_ids'][0]
    protein_batch = batch['protein_graph'].to(device)
    ligand_batch = batch['ligand_graph'].to(device)
    ligand_coords_x0 = batch['ligand_coords_x0'].to(device)
    ligand_coords_x1 = batch['ligand_coords_x1'].to(device)
    t = batch['t'].to(device)
    
    print(f"\nPDB: {pdb_id}")
    print(f"Num ligand atoms: {ligand_coords_x0.shape[0]}")
    print(f"t: {t.item():.4f}")
    
    # True velocity
    true_velocity = ligand_coords_x1 - ligand_coords_x0
    true_velocity_norm = torch.norm(true_velocity, dim=-1).mean()
    
    print(f"\n=== True Velocity ===")
    print(f"  Mean norm: {true_velocity_norm.item():.4f} Å")
    print(f"  First 5 atoms:\n{true_velocity[:5]}")
    
    # Predicted velocity at t=0
    with torch.no_grad():
        ligand_batch.pos = ligand_coords_x0.clone()
        t_zero = torch.zeros(1, device=device)
        pred_velocity_t0 = model(protein_batch, ligand_batch, t_zero)
        
    pred_velocity_t0_norm = torch.norm(pred_velocity_t0, dim=-1).mean()
    
    print(f"\n=== Predicted Velocity (t=0) ===")
    print(f"  Mean norm: {pred_velocity_t0_norm.item():.4f} Å")
    print(f"  First 5 atoms:\n{pred_velocity_t0[:5]}")
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        pred_velocity_t0.flatten(), 
        true_velocity.flatten(), 
        dim=0
    )
    print(f"\n=== Comparison ===")
    print(f"  Cosine similarity: {cos_sim.item():.4f}")
    print(f"  Pred/True norm ratio: {pred_velocity_t0_norm.item() / true_velocity_norm.item():.4f}")
    
    # Check if velocity is near zero
    if pred_velocity_t0_norm.item() < 0.1:
        print(f"\n⚠️  WARNING: Predicted velocity is near zero!")
    
    # Predicted velocity at various t
    print(f"\n=== Velocity at different timesteps ===")
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t_test = torch.tensor([t_val], device=device)
        x_t = (1 - t_val) * ligand_coords_x0 + t_val * ligand_coords_x1
        ligand_batch.pos = x_t.clone()
        
        with torch.no_grad():
            pred_v = model(protein_batch, ligand_batch, t_test)
        
        pred_norm = torch.norm(pred_v, dim=-1).mean()
        cos = torch.nn.functional.cosine_similarity(
            pred_v.flatten(), 
            true_velocity.flatten(), 
            dim=0
        )
        print(f"  t={t_val:.2f}: ||v||={pred_norm.item():.4f}, cos_sim={cos.item():.4f}")
    
    # Initial and final RMSD
    initial_rmsd = torch.sqrt(torch.mean((ligand_coords_x0 - ligand_coords_x1)**2))
    print(f"\n=== RMSD ===")
    print(f"  Initial (docked vs crystal): {initial_rmsd.item():.4f} Å")
    
    # Simulate one step
    dt = 0.05
    new_coords = ligand_coords_x0 + dt * pred_velocity_t0
    new_rmsd = torch.sqrt(torch.mean((new_coords - ligand_coords_x1)**2))
    print(f"  After one step (dt={dt}): {new_rmsd.item():.4f} Å")
    
    # Check for NaN/Inf in velocity
    print(f"\n=== NaN/Inf Check ===")
    print(f"  Velocity has NaN: {torch.isnan(pred_velocity_t0).any().item()}")
    print(f"  Velocity has Inf: {torch.isinf(pred_velocity_t0).any().item()}")
    print(f"  Velocity max: {pred_velocity_t0.abs().max().item():.4f}")
    
    # Full ODE simulation
    print(f"\n=== Full ODE Simulation (10 steps) ===")
    current = ligand_coords_x0.clone()
    for step in range(10):
        t_val = step / 10
        t_test = torch.tensor([t_val], device=device)
        ligand_batch.pos = current.clone()
        with torch.no_grad():
            v = model(protein_batch, ligand_batch, t_test)
        dt = 0.1
        current = current + dt * v
        rmsd = torch.sqrt(torch.mean((current - ligand_coords_x1)**2)).item()
        v_norm = torch.norm(v, dim=-1).mean().item()
        v_max = torch.abs(v).max().item()
        has_nan = torch.isnan(current).any().item()
        print(f"  Step {step}: t={t_val:.2f}, RMSD={rmsd:.4f}A, ||v||={v_norm:.4f}, max|v|={v_max:.4f}, NaN={has_nan}")
        if has_nan:
            print("  NaN detected, stopping")
            break

    # Test at t=0 vs t=0.5 to see if model generalizes
    print(f"\n=== Model behavior at different timesteps (same position) ===")
    ligand_batch.pos = ligand_coords_x0.clone()
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t_test = torch.tensor([t_val], device=device)
        with torch.no_grad():
            v = model(protein_batch, ligand_batch, t_test)
        v_norm = torch.norm(v, dim=-1).mean().item()
        v_max = torch.abs(v).max().item()
        cos_sim = torch.nn.functional.cosine_similarity(v.flatten(), true_velocity.flatten(), dim=0).item()
        print(f"  t={t_val:.2f}: ||v||={v_norm:.4f}, max|v|={v_max:.4f}, cos_sim={cos_sim:.4f}")
    
    # Test at EXACT training position x_t
    print(f"\n=== Test at EXACT training position x_t (should match training) ===")
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Compute x_t = (1-t)*x0 + t*x1
        x_t = (1 - t_val) * ligand_coords_x0 + t_val * ligand_coords_x1
        ligand_batch.pos = x_t.clone()
        t_test = torch.tensor([t_val], device=device)
        with torch.no_grad():
            v = model(protein_batch, ligand_batch, t_test)
        v_norm = torch.norm(v, dim=-1).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(v.flatten(), true_velocity.flatten(), dim=0).item()
        mse = torch.mean((v - true_velocity)**2).item()
        print(f"  t={t_val:.2f}: ||v||={v_norm:.4f}, cos_sim={cos_sim:.4f}, MSE={mse:.4f}")


if __name__ == '__main__':
    main()
