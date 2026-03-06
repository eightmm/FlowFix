import os
import torch
import json
from pathlib import Path
from tqdm import tqdm

def calculate_rmsd(coords1, coords2):
    return torch.sqrt(torch.mean(torch.sum((coords1 - coords2)**2, dim=1))).item()

def main():
    data_dir = Path('/home/jaemin/project/protein-ligand/pose-refine/train_data')
    split_file = data_dir / 'splits_overfit_32.json'
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    pdb_ids = split_data['train']
    high_rmsd_mapping = {}
    
    print(f"Finding highest RMSD poses for {len(pdb_ids)} PDBs...")
    
    for pdb_id in tqdm(pdb_ids):
        ligands_path = data_dir / pdb_id / 'ligands.pt'
        if not ligands_path.exists():
            print(f"Warning: {ligands_path} not found")
            continue
            
        try:
            ligands_list = torch.load(ligands_path, weights_only=False)
        except Exception as e:
            print(f"Error loading {ligands_path}: {e}")
            continue
            
        max_rmsd = -1
        best_pose_idx = 1
        
        # Skip pose 0 (crystal structure)
        for i in range(1, len(ligands_list)):
            pose = ligands_list[i]
            rmsd = calculate_rmsd(pose['coord'], pose['crystal_coord'])
            if rmsd > max_rmsd:
                max_rmsd = rmsd
                best_pose_idx = i
        
        high_rmsd_mapping[pdb_id] = best_pose_idx
        print(f"  {pdb_id}: Max RMSD = {max_rmsd:.3f} Å at Pose {best_pose_idx}")

    output_path = data_dir / 'high_rmsd_poses.json'
    with open(output_path, 'w') as f:
        json.dump(high_rmsd_mapping, f, indent=2)
    
    print(f"\nSaved high RMSD mapping to {output_path}")

if __name__ == "__main__":
    main()
