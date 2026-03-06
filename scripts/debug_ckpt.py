import os
import torch
import yaml
import argparse
from pathlib import Path
from torch_geometric.loader import DataLoader
from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from src.models.flowmatching import ProteinLigandFlowMatchingJoint
from src.utils.relaxation import RelaxationEngine

def test_checkpoint(checkpoint_path, config_path, split="val", pdb_id=None, do_relax=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n" + "="*60)
    print(f"Testing Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print("="*60)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset
    dataset = FlowFixDataset(
        data_dir=config['data']['data_dir'],
        split_file=config['data'].get('split_file'),
        split=split,
        max_samples=None,
        seed=42,
        fix_pose=config['data'].get('fix_pose', False),
        fix_pose_high_rmsd=config['data'].get('fix_pose_high_rmsd', False)
    )
    
    # Selection
    if pdb_id:
        try:
            target_idx = dataset.pdb_ids.index(pdb_id)
            sample = dataset[target_idx]
        except ValueError:
            print(f"PDB {pdb_id} not in val split, using first entry")
            sample = dataset[0]
    else:
        sample = dataset[0]

    # Model
    import inspect
    sig = inspect.signature(ProteinLigandFlowMatchingJoint.__init__)
    valid_params = sig.parameters.keys()
    model_config = {k: v for k, v in config['model'].items() if k in valid_params}
    
    model = ProteinLigandFlowMatchingJoint(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # TEST: Keep in train mode to see if BatchNorm is the culprit
    model.eval() 
    model.eval() 
    print("WARNING: Running in MODEL.TRAIN() mode to avoid BatchNorm eval mismatch")

    # Data
    batch = collate_flowfix_batch([sample])
    ligand_batch = batch['ligand_graph'].to(device)
    protein_batch = batch['protein_graph'].to(device)
    ligand_coords_x0 = batch['ligand_coords_x0'].to(device)
    ligand_coords_x1 = batch['ligand_coords_x1'].to(device)

    initial_rmsd = torch.sqrt(torch.mean((ligand_coords_x0 - ligand_coords_x1) ** 2)).item()
    print(f"\nPDB: {batch['pdb_ids'][0]}")
    print(f"Initial RMSD: {initial_rmsd:.4f} Å")

    # ODE Integration
    num_steps = 20
    current_coords = ligand_coords_x0.clone()
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1)
    
    print(f"\n{'Step':>4} | {'t':>5} | {'RMSD':>8} | {'|v|':>8} | {'|v_tgt|':>8} | {'Sim':>8}")
    print("-" * 60)

    for step in range(num_steps):
        t_current = timesteps[step]
        t_next = timesteps[step+1]
        dt = t_next - t_current
        
        t = torch.ones(1, device=device) * t_current
        
        with torch.no_grad():
            ligand_batch.pos = current_coords
            velocity = model(protein_batch, ligand_batch, t)
            
        # target v in flow matching sense: (x1 - xt) / (1-t)
        eps = 1e-5
        v_target = (ligand_coords_x1 - current_coords) / (1.0 - t_current + eps)
        
        rmsd = torch.sqrt(torch.mean((current_coords - ligand_coords_x1) ** 2)).item()
        v_norm = torch.norm(velocity, dim=-1).mean().item()
        v_target_norm = torch.norm(v_target, dim=-1).mean().item()
        
        v_flat = velocity.view(-1)
        target_flat = v_target.view(-1)
        sim = torch.nn.functional.cosine_similarity(v_flat, target_flat, dim=0).item()
        
        if step % 5 == 0 or step == num_steps - 1:
            print(f"{step:4d} | {t_current:5.2f} | {rmsd:8.4f} | {v_norm:8.4f} | {v_target_norm:8.4f} | {sim:8.4f}")
        
        current_coords = current_coords + velocity * dt

    final_rmsd = torch.sqrt(torch.mean((current_coords - ligand_coords_x1) ** 2)).item()
    print("-" * 60)
    print(f"Final RMSD: {final_rmsd:.4f} Å")
    print(f"Refinement: {initial_rmsd - final_rmsd:.4f} Å")

    if do_relax:
        print("\n" + "="*60)
        print("Running Force Field Relaxation...")
        print("="*60)
        
        relax_engine = RelaxationEngine(
            clash_weight=1.0,
            dg_weight=1.0,
            restraint_weight=20.0,
            lr=1.0,
            max_steps=100
        )
        
        relaxed_coords, metrics = relax_engine.relax(
            ligand_coords=current_coords,
            protein_batch=protein_batch,
            distance_bounds=batch.get('distance_bounds'),
            device=device
        )
        
        relaxed_rmsd = torch.sqrt(torch.mean((relaxed_coords - ligand_coords_x1) ** 2)).item()
        print(f"Relaxed RMSD: {relaxed_rmsd:.4f} Å")
        print(f"Improvement: {final_rmsd - relaxed_rmsd:.4f} Å")
        print(f"Metrics: {metrics}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--pdb', type=str, default=None)
    parser.add_argument('--relax', action='store_true', help='Run force field relaxation')
    args = parser.parse_args()
    
    test_checkpoint(args.ckpt, args.config, args.split, args.pdb, args.relax)
