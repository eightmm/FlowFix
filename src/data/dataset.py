"""
FlowFix Dataset for Flow Matching

Loads data from train_data directory
Flow matching: docked pose (x0, t=0) → crystal structure (x1, t=1)

Data structure:
- train_data/<pdb_id>/ligands.pt: List of 60 poses, each with 'coord' (docked) and 'crystal_coord'
- train_data/<pdb_id>/protein.pt: Protein structure

Memory-efficient dynamic dataset that samples a new docked pose per PDB each epoch.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm


class FlowFixDataset(Dataset):
    """
    Memory-efficient FlowFix dataset that samples a new docked pose per PDB each epoch.

    Unlike static datasets which load all pose pairs, this dataset:
    - Has length = number of PDB IDs (not total pairs)
    - Randomly samples one docked pose per PDB on each access
    - Reduces memory usage and epoch time
    - Provides natural data augmentation through pose sampling
    """

    def __init__(
        self,
        data_dir: str = "train_data",
        split_file: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seed = seed
        self.epoch = 0  # Track current epoch for reproducible sampling

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get all PDB directories
        all_pdbs = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        # Apply split if split_file provided
        if split_file and os.path.exists(split_file):
            self.pdb_ids = self._load_split_from_tsv(all_pdbs, split_file)
        else:
            # Simple 80/10/10 split
            np.random.shuffle(all_pdbs)
            n_train = int(0.8 * len(all_pdbs))
            n_val = int(0.1 * len(all_pdbs))

            if split == "train":
                self.pdb_ids = all_pdbs[:n_train]
            elif split == "valid":
                self.pdb_ids = all_pdbs[n_train:n_train + n_val]
            else:  # test
                self.pdb_ids = all_pdbs[n_train + n_val:]

        # Limit samples if specified
        if max_samples is not None:
            self.pdb_ids = self.pdb_ids[:max_samples]

        # Validate PDB directories: check for ligands.pt and protein.pt
        valid_pdbs = []
        print(f"Validating {split} dataset from {len(self.pdb_ids)} PDB IDs...")

        for pdb_id in tqdm(self.pdb_ids, desc=f"Validating {split} PDBs"):
            pdb_dir = self.data_dir / pdb_id
            ligands_file = pdb_dir / "ligands.pt"
            protein_file = pdb_dir / "protein.pt"

            if ligands_file.exists() and protein_file.exists():
                valid_pdbs.append(pdb_id)

        self.pdb_ids = valid_pdbs

        if len(self.pdb_ids) == 0:
            raise ValueError(
                f"No valid PDBs found in {self.data_dir}!\n"
                f"Expected structure: {self.data_dir}/<pdb_id>/ligands.pt and protein.pt"
            )

        print(f"Loaded {len(self.pdb_ids)} valid PDBs for {split} split")

    def _load_split_from_tsv(self, all_pdbs: List[str], split_file: str) -> List[str]:
        """Load train/val/test split from TSV file."""
        import pandas as pd

        df = pd.read_csv(split_file, sep='\t')

        # Map split names
        split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
        split_name = split_map.get(self.split, self.split)

        # Filter by split
        split_pdbs = df[df['split'] == split_name]['pdb_id'].tolist()

        # Return intersection with available PDBs
        return [pdb for pdb in all_pdbs if pdb in split_pdbs]

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible random sampling across workers."""
        self.epoch = epoch
        # Reset seed based on epoch for consistent sampling
        np.random.seed(self.seed + epoch)
        torch.manual_seed(self.seed + epoch)

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pdb_id = self.pdb_ids[idx]
        pdb_dir = self.data_dir / pdb_id

        # Load ligands and protein
        ligands_path = pdb_dir / "ligands.pt"
        protein_path = pdb_dir / "protein.pt"

        ligands_list = torch.load(ligands_path, weights_only=False)
        protein_data = torch.load(protein_path, weights_only=False)

        # Randomly sample one docked pose for this PDB
        # Use (epoch, idx) for deterministic sampling within each epoch
        rng = np.random.RandomState(self.seed + self.epoch * 10000 + idx)
        pose_idx = rng.randint(0, len(ligands_list))
        ligand_data = ligands_list[pose_idx]

        # Extract coordinates
        ligand_coords_x0 = ligand_data['coord']  # Docked (source, t=0) - [N_ligand, 3]
        ligand_coords_x1 = ligand_data['crystal_coord']  # Crystal (target, t=1) - [N_ligand, 3]
        num_ligand_atoms = ligand_coords_x0.shape[0]

        # Validate atom count consistency
        if ligand_coords_x0.shape[0] != ligand_coords_x1.shape[0]:
            # Skip samples with inconsistent atom counts
            return None

        # Extract pocket from protein based on crystal ligand position
        protein_coords = self._extract_coords(protein_data)
        protein_graph = self._extract_pocket(
            protein_data,
            protein_coords,
            ligand_coords_x1,  # Use crystal ligand as reference
            cutoff=12.0
        )

        # Create PyG Data object for ligand
        # Important: ligand_data['node']['coords'] may have more nodes than actual ligand atoms
        # (e.g., 61 nodes vs 33 ligand atoms). We need to use only the actual ligand atoms.
        ligand_graph = self._create_pyg_graph(ligand_data, ligand_coords_x0, num_ligand_atoms)

        # Extract distance bounds separately (can't be batched with PyG)
        distance_bounds = {}
        if 'distance_lower_bounds' in ligand_data:
            distance_bounds['lower'] = ligand_data['distance_lower_bounds']
        if 'distance_upper_bounds' in ligand_data:
            distance_bounds['upper'] = ligand_data['distance_upper_bounds']

        return {
            'pdb_id': pdb_id,
            'protein_graph': protein_graph,
            'ligand_graph': ligand_graph,
            'ligand_coords_x0': ligand_coords_x0,
            'ligand_coords_x1': ligand_coords_x1,
            'distance_bounds': distance_bounds if distance_bounds else None,
        }

    def _extract_pocket(self, protein_data: Any, protein_coords: torch.Tensor,
                        ligand_coords: torch.Tensor, cutoff: float = 12.0) -> Data:
        """
        Extract pocket residues within cutoff distance from ligand using PyG subgraph.

        Args:
            protein_data: Raw protein data (dict or Data object)
            protein_coords: Protein residue coordinates [N_residues, 3]
            ligand_coords: Ligand atom coordinates [N_atoms, 3]
            cutoff: Distance cutoff in Angstroms

        Returns:
            Filtered PyG Data object with only pocket residues
        """
        from torch_geometric.utils import subgraph

        # 1. Convert to PyG graph first
        protein_graph = self._create_pyg_graph(protein_data, protein_coords)

        # 2. Compute pocket mask (distance-based)
        dist = torch.cdist(protein_coords, ligand_coords)  # [N_res, N_lig]
        min_dist = dist.min(dim=1)[0]  # [N_residues]
        pocket_mask = min_dist <= cutoff
        pocket_indices = torch.where(pocket_mask)[0]

        if pocket_indices.numel() == 0:
            # Fallback: keep all residues if pocket is empty (shouldn't happen)
            return protein_graph

        # 3. Extract subgraph using PyG (C++ optimized, much faster)
        edge_index, edge_attr = subgraph(
            pocket_indices,
            protein_graph.edge_index,
            protein_graph.edge_attr,
            relabel_nodes=True,
            num_nodes=protein_graph.num_nodes
        )

        # 4. Handle edge_vector_features separately (subgraph doesn't support this)
        edge_vector_features = None
        if hasattr(protein_graph, 'edge_vector_features') and protein_graph.edge_vector_features is not None:
            # Find which edges are kept
            src, dst = protein_graph.edge_index
            edge_mask = pocket_mask[src] & pocket_mask[dst]
            edge_vector_features = protein_graph.edge_vector_features[edge_mask]

        # 5. Create filtered graph
        filtered_graph = Data(
            x=protein_graph.x[pocket_mask],
            pos=protein_coords[pocket_mask],
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # Add optional features
        if hasattr(protein_graph, 'node_vector_features') and protein_graph.node_vector_features is not None:
            filtered_graph.node_vector_features = protein_graph.node_vector_features[pocket_mask]

        if edge_vector_features is not None:
            filtered_graph.edge_vector_features = edge_vector_features

        return filtered_graph

    def _extract_coords(self, data: Any) -> torch.Tensor:
        """Extract 3D coordinates from data."""
        if hasattr(data, 'pos'):
            coords = data.pos
        elif hasattr(data, 'coord'):
            coords = data.coord
        elif isinstance(data, dict):
            if 'node' in data:
                coords = data['node']['coord']
            elif 'coord' in data:
                coords = data['coord']
            elif 'pos' in data:
                coords = data['pos']
            else:
                raise ValueError(f"Could not find coordinates in data with keys: {data.keys()}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Handle shape (N, 2, 3) -> (N, 3)
        if coords.dim() == 3:
            coords = coords[:, 0, :]

        return coords

    def _create_pyg_graph(self, data: Any, coords: torch.Tensor, num_atoms: Optional[int] = None) -> Data:
        """Convert raw data to PyG Data object with vector features."""
        if isinstance(data, Data):
            return data

        # Handle new flat dictionary format (ligand data)
        if isinstance(data, dict) and 'edges' in data and 'node_feats' in data:
            # New ligand data format with flat keys
            x = data['node_feats'].float()  # [N, feature_dim]
            edge_index = data['edges'].long()  # [2, E]
            edge_attr = data.get('edge_feats', None)

            if edge_attr is not None:
                edge_attr = edge_attr.float()  # [E, edge_feature_dim]

            # Create PyG Data object
            # Note: distance bounds are handled separately in __getitem__
            # because they can't be batched with PyG (different sizes per molecule)
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=coords
            )

            return graph

        # Handle old nested dictionary format
        if isinstance(data, dict) and 'node' in data:
            node_data = data['node']
            edge_data = data['edge']

            # Protein structure: node_scalar_features, node_vector_features (tuples)
            if 'node_scalar_features' in node_data:
                # Protein data
                node_feat_tuple = node_data['node_scalar_features']
                if isinstance(node_feat_tuple, tuple):
                    feat_list = [feat.reshape(feat.shape[0], -1) if feat.dim() > 2 else feat
                                for feat in node_feat_tuple if isinstance(feat, torch.Tensor)]
                    x = torch.cat(feat_list, dim=1).float()
                else:
                    x = node_feat_tuple.float()

                # Concatenate node vector features
                node_vector_features = None
                if 'node_vector_features' in node_data:
                    node_vec_tuple = node_data['node_vector_features']
                    if isinstance(node_vec_tuple, tuple):
                        vec_list = [vec.reshape(vec.shape[0], -1, 3) if vec.dim() > 3 else vec
                                   for vec in node_vec_tuple if isinstance(vec, torch.Tensor)]
                        node_vector_features = torch.cat(vec_list, dim=1).float()  # [N, num_vectors, 3]
                    else:
                        node_vector_features = node_vec_tuple.float()

                # Edge index: [2, num_edges]
                src, dst = edge_data['edges']
                edge_index = torch.stack([src, dst], dim=0).long()

                # Concatenate edge scalar features
                edge_attr = None
                if 'edge_scalar_features' in edge_data:
                    edge_feat_tuple = edge_data['edge_scalar_features']
                    if isinstance(edge_feat_tuple, tuple):
                        edge_feat_list = [feat.reshape(feat.shape[0], -1) if feat.dim() > 2 else feat
                                         for feat in edge_feat_tuple if isinstance(feat, torch.Tensor)]
                        edge_attr = torch.cat(edge_feat_list, dim=1).float()
                    else:
                        edge_attr = edge_feat_tuple.float()

                # Concatenate edge vector features
                edge_vector_features = None
                if 'edge_vector_features' in edge_data:
                    edge_vec_tuple = edge_data['edge_vector_features']
                    if isinstance(edge_vec_tuple, tuple):
                        edge_vec_list = [vec.reshape(vec.shape[0], -1, 3) if vec.dim() > 3 else vec
                                        for vec in edge_vec_tuple if isinstance(vec, torch.Tensor)]
                        edge_vector_features = torch.cat(edge_vec_list, dim=1).float()  # [E, num_vectors, 3]
                    else:
                        edge_vector_features = edge_vec_tuple.float()

                return Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=coords,
                    node_vector_features=node_vector_features,
                    edge_vector_features=edge_vector_features
                )

            # Ligand structure: node_feats (single tensor) in nested format
            elif 'node_feats' in node_data:
                # Old ligand data format
                # Note: node_feats may have more nodes than actual ligand atoms
                # (e.g., includes virtual nodes or hydrogen atoms)
                # We use num_atoms to slice only the actual ligand atoms
                if num_atoms is not None:
                    x = node_data['node_feats'][:num_atoms].float()  # [num_atoms, 122]
                else:
                    x = node_data['node_feats'].float()  # [N, 122]

                # Edge index: [2, num_edges] (already in PyG format)
                edge_index = edge_data['edges'].long()

                # Filter edges: only keep edges where both src and dst < num_atoms
                if num_atoms is not None:
                    valid_edge_mask = (edge_index[0] < num_atoms) & (edge_index[1] < num_atoms)
                    edge_index = edge_index[:, valid_edge_mask]

                    # Also filter edge features
                    edge_feats = edge_data.get('edge_feats', None)
                    if edge_feats is not None:
                        edge_attr = edge_feats[valid_edge_mask].float()
                    else:
                        edge_attr = None
                else:
                    edge_attr = edge_data.get('edge_feats', None)
                    if edge_attr is not None:
                        edge_attr = edge_attr.float()

                # Ligand has no vector features
                return Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=coords
                )

        raise ValueError(f"Unsupported data format: {type(data)}")


def collate_flowfix_batch(samples: List[Dict]) -> Dict[str, Any]:
    """Batch FlowFix samples using PyG Batch."""
    # Filter out None samples (inconsistent atom counts)
    samples = [s for s in samples if s is not None]

    if len(samples) == 0:
        raise ValueError("All samples in batch have inconsistent atom counts!")

    pdb_ids = [s['pdb_id'] for s in samples]

    ligand_batch = Batch.from_data_list([s['ligand_graph'] for s in samples])
    protein_batch = Batch.from_data_list([s['protein_graph'] for s in samples])
    ligand_coords_x0 = torch.cat([s['ligand_coords_x0'] for s in samples], dim=0)
    ligand_coords_x1 = torch.cat([s['ligand_coords_x1'] for s in samples], dim=0)

    # Collect distance bounds as a list (can't be batched due to different sizes)
    distance_bounds = [s.get('distance_bounds', None) for s in samples]

    return {
        'pdb_ids': pdb_ids,
        'protein_graph': protein_batch,
        'ligand_graph': ligand_batch,
        'ligand_coords_x0': ligand_coords_x0,
        'ligand_coords_x1': ligand_coords_x1,
        'ligand_batch': ligand_batch.batch,  # PyG batch indices
        'distance_bounds': distance_bounds,  # List of per-molecule distance bounds
    }


def test_flowfix_dataset():
    """Test FlowFix dataset with epoch-based pose sampling."""
    print("="*60)
    print("Testing FlowFix Dataset")
    print("="*60)

    # Create dynamic dataset
    dataset = FlowFixDataset(
        data_dir="train_data",
        split="train",
        max_samples=3,
        seed=42
    )

    print(f"\nDataset size: {len(dataset)} PDB IDs")

    if len(dataset) == 0:
        print("⚠️  No samples found!")
        return

    # Test epoch-based sampling
    print("\n" + "="*60)
    print("Testing Epoch-based Pose Sampling")
    print("="*60)

    pdb_id = dataset.pdb_ids[0]
    print(f"\nPDB ID: {pdb_id}")

    # Sample same index across 3 epochs
    poses_sampled = []
    for epoch in range(3):
        dataset.set_epoch(epoch)
        sample = dataset[0]

        if sample is None:
            print(f"⚠️  Epoch {epoch}: Sample is None (inconsistent atom counts)")
            continue

        # Get docked pose filename by checking RMSD
        x0 = sample['ligand_coords_x0']
        x1 = sample['ligand_coords_x1']
        rmsd = torch.sqrt(torch.mean(torch.sum((x0 - x1)**2, dim=1)))

        print(f"\nEpoch {epoch}:")
        print(f"  Ligand atoms: {x0.shape[0]}")
        print(f"  RMSD: {rmsd:.3f} Å")
        poses_sampled.append(rmsd.item())

    print(f"\n📊 RMSD Variation Across Epochs:")
    print(f"  Different poses sampled: {len(set(poses_sampled)) > 1}")
    print(f"  RMSDs: {[f'{r:.3f}' for r in poses_sampled]}")

    # Test batching
    print("\n" + "="*60)
    print("Testing Batch Collation")
    print("="*60)

    from torch.utils.data import DataLoader

    dataset.set_epoch(0)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_flowfix_batch
    )

    batch = next(iter(dataloader))
    print(f"\nBatch with {len(batch['pdb_ids'])} samples:")
    print(f"  PDB IDs: {batch['pdb_ids']}")
    print(f"  Total ligand atoms: {batch['ligand_coords_x0'].shape[0]}")
    print(f"  Total protein residues: {batch['protein_graph'].num_nodes}")

    print("\n" + "="*60)
    print("✅ FlowFix Dataset Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_flowfix_dataset()
