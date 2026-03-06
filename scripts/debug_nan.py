#!/usr/bin/env python
"""
Debug script to identify NaN source in validation.
Tests model forward pass step by step.
"""

import torch
import yaml
import sys
sys.path.insert(0, '/home/jaemin/project/protein-ligand/pose-refine')

from src.utils.model_builder import build_model
from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from torch.utils.data import DataLoader

def check_tensor(name, t):
    """Check tensor for NaN/Inf and print stats."""
    if t is None:
        print(f"  {name}: None")
        return
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    print(f"  {name}: shape={tuple(t.shape)}, nan={nan_count}, inf={inf_count}, "
          f"min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}")

def main():
    # Load config
    config_path = '/home/jaemin/project/protein-ligand/pose-refine/save/overfit-test-32/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda:0')
    
    # Load dataset (validation mode - uses x0 positions, not x_t)
    val_dataset = FlowFixDataset(
        data_dir=config['data']['data_dir'],
        split_file=config['data']['split_file'],
        split='val',
        fix_pose=config['data'].get('fix_pose', True),
        fix_pose_high_rmsd=config['data'].get('fix_pose_high_rmsd', False),
        position_noise_scale=0.0,  # No noise for validation
        loading_mode='lazy',
        max_samples=1,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_flowfix_batch,
    )
    
    # Build model
    model = build_model(config['model'], device)
    
    # Load checkpoint
    checkpoint_path = '/home/jaemin/project/protein-ligand/pose-refine/save/overfit-test-32/checkpoints/latest.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Test both eval modes: standard eval and eval with BatchNorm in train mode
    for bn_mode in ['eval', 'train']:
        print(f"\n{'='*60}")
        print(f"TESTING WITH BATCHNORM IN {bn_mode.upper()} MODE")
        print(f"{'='*60}")
        
        model.eval()
        
        # Optionally keep BatchNorm in train mode
        if bn_mode == 'train':
            for module in model.modules():
                cls_name = type(module).__name__
                if 'BatchNorm' in cls_name:
                    module.train()
        
        test_model(model, val_loader, device)


def test_model(model, val_loader, device):
    """Test model forward pass and ODE integration with detailed metrics."""
    
    # Get one batch
    batch = next(iter(val_loader))
    
    # Move to device
    ligand_batch = batch['ligand_graph'].to(device)
    protein_batch = batch['protein_graph'].to(device)
    ligand_coords_x0 = batch['ligand_coords_x0'].to(device)
    ligand_coords_x1 = batch['ligand_coords_x1'].to(device)
    
    print(f"\n=== Input Data Check ===")
    print(f"PDB: {batch['pdb_ids'][0]}")
    check_tensor("ligand_coords_x0", ligand_coords_x0)
    check_tensor("ligand_coords_x1", ligand_coords_x1)
    
    # Initial RMSD
    initial_rmsd = torch.sqrt(torch.mean((ligand_coords_x0 - ligand_coords_x1) ** 2)).item()
    print(f"Initial RMSD: {initial_rmsd:.4f} Å")
    
    # Simulate ODE integration with detailed debugging
    print(f"\n=== Detailed ODE Integration Debug (Euler, 20 steps) ===")
    print(f"{'Step':>4} | {'t':>5} | {'RMSD':>7} | {'|v|':>7} | {'|v_tgt|':>7} | {'Sim':>7}")
    print("-" * 60)
    
    num_steps = 20
    current_coords = ligand_coords_x0.clone()
    
    for step in range(num_steps):
        t_current = step / num_steps
        t_next = (step + 1) / num_steps
        dt = t_next - t_current
        
        t = torch.tensor([t_current], device=device)
        ligand_batch.pos = current_coords.clone()
        
        # Self-conditioning (if enabled in model, usually x1_self_cond)
        # Note: In actual validation, we might need to handle self_conditioning properly 
        # but for simple velocity direction check, t=0 is most critical.
        
        with torch.no_grad():
            # In validation, we might use x_t directly for self_cond
            velocity = model(protein_batch, ligand_batch, t)
        
        # Ideal velocity at this point to reach x1
        # In CFM with linear interpolation, the constant velocity is (x1 - x0)
        # However, our target direction at current point is (x1 - xt)
        v_target = ligand_coords_x1 - current_coords
        
        # Metrics
        rmsd = torch.sqrt(torch.mean((current_coords - ligand_coords_x1) ** 2)).item()
        v_norm = torch.norm(velocity, dim=-1).mean().item()
        v_target_norm = torch.norm(v_target, dim=-1).mean().item()
        
        # Cosine similarity between predicted velocity and target direction
        v_flat = velocity.view(-1)
        target_flat = v_target.view(-1)
        sim = torch.nn.functional.cosine_similarity(v_flat, target_flat, dim=0).item()
        
        print(f"{step:4d} | {t_current:5.2f} | {rmsd:7.4f} | {v_norm:7.4f} | {v_target_norm:7.4f} | {sim:7.4f}")
        
        # Clip velocity (same as in validation)
        velocity_norm = torch.norm(velocity, dim=-1, keepdim=True)
        max_velocity = 5.0
        scale = torch.clamp(max_velocity / (velocity_norm + 1e-8), max=1.0)
        velocity = velocity * scale
        
        # Update coords
        current_coords = current_coords + dt * velocity
        
        if torch.isnan(current_coords).any():
            print(f"!!! Error: NaN detected at step {step}")
            break
            
    final_rmsd = torch.sqrt(torch.mean((current_coords - ligand_coords_x1) ** 2)).item()
    print("-" * 45)
    print(f"Final RMSD: {final_rmsd:.4f} Å")
    print(f"Refinement: {initial_rmsd - final_rmsd:+.4f} Å")

if __name__ == '__main__':
    main()
