import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from tqdm import tqdm
import argparse


class PoseFlowDataset(Dataset):
    """
    Dataset for protein-ligand pose refinement using flow matching.
    Loads preprocessed graph data and applies perturbations for training.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        perturbation_config: Optional[Dict] = None,
        cache_data: bool = True,
        seed: int = 42,
        perturbation_mode: str = 'random',  # 'random' or 'fixed'
        resample_every_epoch: bool = True,
        val_fixed_t: Optional[float] = 0.5
    ):
        """
        Args:
            data_dir: Directory containing processed .pt files
            split: Data split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            perturbation_config: Configuration for pose perturbation
            cache_data: Whether to cache loaded data in memory
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_data = cache_data
        self.seed = seed
        self.perturbation_mode = perturbation_mode
        self.resample_every_epoch = resample_every_epoch
        self.val_fixed_t = val_fixed_t
        self.current_epoch = 0
        
        # Default perturbation configuration
        self.perturbation_config = perturbation_config or {
            'translation_std': 2.0,  # Angstroms
            'rotation_std': 0.5,  # Radians
            'torsion_std': 0.3,  # Radians for ligand torsions
            'min_rmsd': 2.0,  # Minimum RMSD after perturbation
            'max_rmsd': 8.0,  # Maximum RMSD after perturbation
        }
        
        # Find all data files
        self.data_files = sorted(list(self.data_dir.glob('*.pt')))
        if max_samples:
            self.data_files = self.data_files[:max_samples]
        
        # Split data (simple split for now, can be improved with predefined splits)
        self._split_data()
        
        # Cache for loaded data
        self.cache = {} if cache_data else None
        # Cache for fixed perturbations in validation/test
        self._fixed_perturbation_cache: Dict[int, Dict] = {}
        
        # Set initial random seed (will be optionally updated per-epoch for training)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Loaded {len(self.data_files)} samples for {split} split")
    
    def _split_data(self):
        """Split data into train/val/test sets."""
        n_total = len(self.data_files)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if self.split == 'train':
            self.data_files = self.data_files[:n_train]
        elif self.split == 'val':
            self.data_files = self.data_files[n_train:n_train + n_val]
        elif self.split == 'test':
            self.data_files = self.data_files[n_train + n_val:]
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single sample.
        
        Returns:
            Dictionary containing:
            - protein features and coordinates
            - ligand features and coordinates (crystal and perturbed)
            - time step for flow matching
            - vector field target
        """
        # Load data from cache or disk
        if self.cache_data and idx in self.cache:
            data = self.cache[idx]
        else:
            data = torch.load(self.data_files[idx])
            if self.cache_data:
                self.cache[idx] = data
        
        # Clone to avoid modifying cached data
        sample = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
        
        # Apply perturbation policy by split/mode
        if self.split == 'train':
            # Random augmentation each access; may be re-seeded per epoch by set_epoch()
            if self.perturbation_mode == 'random':
                sample = self._apply_perturbation(sample)
            else:
                # Deterministic fixed perturbation for debugging if requested
                sample = self._apply_fixed_perturbation(sample, idx)
        else:
            # Validation/test: prefer fixed perturbation for stable metrics
            if self.perturbation_mode == 'fixed':
                sample = self._apply_fixed_perturbation(sample, idx)
        
        # Add flow matching components
        sample = self._add_flow_components(sample)
        
        return sample
    
    def _apply_perturbation(self, sample: Dict) -> Dict:
        """
        Apply perturbation to ligand coordinates to create training pairs.
        
        Args:
            sample: Data dictionary
            
        Returns:
            Updated sample with perturbed ligand coordinates
        """
        # Store crystal (target) coordinates
        sample['ligand_coords_crystal'] = sample['ligand_coords'].clone()
        
        # Get ligand center
        ligand_center = sample['ligand_coords'].mean(dim=0)
        
        # Apply random rotation
        rotation_matrix = self._random_rotation_matrix(
            std=self.perturbation_config['rotation_std']
        )
        centered_coords = sample['ligand_coords'] - ligand_center
        rotated_coords = centered_coords @ rotation_matrix.T
        
        # Apply random translation
        translation = torch.randn(3) * self.perturbation_config['translation_std']
        perturbed_coords = rotated_coords + ligand_center + translation
        
        # Apply torsion perturbations (simplified - can be improved with RDKit)
        if 'ligand_edge_index' in sample and sample['ligand_edge_index'].numel() > 0:
            perturbed_coords = self._perturb_torsions(
                perturbed_coords,
                sample['ligand_edge_index'],
                std=self.perturbation_config['torsion_std']
            )
        
        # Ensure perturbation is within desired RMSD range
        rmsd = self._calculate_rmsd(perturbed_coords, sample['ligand_coords_crystal'])
        
        # Scale perturbation if needed
        if rmsd < self.perturbation_config['min_rmsd']:
            scale = self.perturbation_config['min_rmsd'] / (rmsd + 1e-6)
            diff = perturbed_coords - sample['ligand_coords_crystal']
            perturbed_coords = sample['ligand_coords_crystal'] + diff * scale
        elif rmsd > self.perturbation_config['max_rmsd']:
            scale = self.perturbation_config['max_rmsd'] / rmsd
            diff = perturbed_coords - sample['ligand_coords_crystal']
            perturbed_coords = sample['ligand_coords_crystal'] + diff * scale
        
        sample['ligand_coords_perturbed'] = perturbed_coords
        sample['perturbation_rmsd'] = torch.tensor(
            self._calculate_rmsd(perturbed_coords, sample['ligand_coords_crystal'])
        )
        
        return sample

    def _apply_fixed_perturbation(self, sample: Dict, idx: int) -> Dict:
        """
        Apply deterministic, cached perturbation for validation/test or fixed-mode train.
        The same index will yield the same perturbed pose across epochs/runs.
        """
        # Return cached if available
        if idx in self._fixed_perturbation_cache:
            fixed = self._fixed_perturbation_cache[idx]
            sample['ligand_coords_crystal'] = fixed['ligand_coords_crystal'].clone()
            sample['ligand_coords_perturbed'] = fixed['ligand_coords_perturbed'].clone()
            sample['perturbation_rmsd'] = fixed['perturbation_rmsd'].clone()
            return sample

        # Clone base coordinates
        ligand_coords = sample['ligand_coords'].clone()
        sample['ligand_coords_crystal'] = ligand_coords.clone()

        # Build per-index deterministic RNGs
        base_seed = (self.seed * 1000003 + idx) & 0x7FFFFFFF
        gen = torch.Generator()
        gen.manual_seed(base_seed)
        np_rng = np.random.RandomState(base_seed)

        # Deterministic rotation
        rotation_matrix = self._random_rotation_matrix_with_generator(
            std=self.perturbation_config['rotation_std'], gen=gen, np_rng=np_rng
        )
        ligand_center = ligand_coords.mean(dim=0)
        centered = ligand_coords - ligand_center
        rotated = centered @ rotation_matrix.T

        # Deterministic translation
        translation = torch.randn(3, generator=gen) * self.perturbation_config['translation_std']
        perturbed = rotated + ligand_center + translation

        # Deterministic torsion-like noise
        if 'ligand_edge_index' in sample and sample['ligand_edge_index'].numel() > 0:
            perturbed = self._perturb_torsions_with_generator(
                perturbed, sample['ligand_edge_index'], std=self.perturbation_config['torsion_std'], gen=gen
            )

        # Enforce RMSD range deterministically
        rmsd = self._calculate_rmsd(perturbed, ligand_coords)
        if rmsd < self.perturbation_config['min_rmsd']:
            scale = self.perturbation_config['min_rmsd'] / (rmsd + 1e-6)
            diff = perturbed - ligand_coords
            perturbed = ligand_coords + diff * scale
        elif rmsd > self.perturbation_config['max_rmsd']:
            scale = self.perturbation_config['max_rmsd'] / rmsd
            diff = perturbed - ligand_coords
            perturbed = ligand_coords + diff * scale

        # Save to sample and cache
        sample['ligand_coords_perturbed'] = perturbed
        sample['perturbation_rmsd'] = torch.tensor(self._calculate_rmsd(perturbed, ligand_coords))

        self._fixed_perturbation_cache[idx] = {
            'ligand_coords_crystal': sample['ligand_coords_crystal'].clone(),
            'ligand_coords_perturbed': sample['ligand_coords_perturbed'].clone(),
            'perturbation_rmsd': sample['perturbation_rmsd'].clone(),
        }
        return sample
    
    def _random_rotation_matrix(self, std: float = 0.5) -> torch.Tensor:
        """
        Generate a random rotation matrix using axis-angle representation.
        
        Args:
            std: Standard deviation for rotation angle
            
        Returns:
            3x3 rotation matrix
        """
        # Random axis (normalized)
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)
        
        # Random angle
        angle = torch.randn(1) * std
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Cross product matrix
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rotation matrix
        R = torch.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
        
        return R

    def _random_rotation_matrix_with_generator(self, std: float, gen: torch.Generator, np_rng: np.random.RandomState) -> torch.Tensor:
        """Deterministic rotation using provided RNGs."""
        axis = torch.tensor(np_rng.randn(3), dtype=torch.float32)
        axis = axis / torch.norm(axis)
        angle = torch.randn(1, generator=gen) * std
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=torch.float32)
        R = torch.eye(3, dtype=torch.float32) + sin_angle * K + (1 - cos_angle) * (K @ K)
        return R
    
    def _perturb_torsions(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        std: float = 0.3
    ) -> torch.Tensor:
        """
        Apply small torsion angle perturbations to ligand.
        Simplified version - proper implementation would use RDKit conformer generation.
        
        Args:
            coords: Ligand coordinates
            edge_index: Bond connectivity
            std: Standard deviation for torsion perturbation
            
        Returns:
            Perturbed coordinates
        """
        # This is a simplified placeholder
        # In practice, you'd want to:
        # 1. Identify rotatable bonds
        # 2. Apply torsion rotations around those bonds
        # 3. Maintain bond lengths and angles
        
        # For now, just add small random noise
        noise = torch.randn_like(coords) * std * 0.1
        return coords + noise

    def _perturb_torsions_with_generator(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        std: float,
        gen: torch.Generator
    ) -> torch.Tensor:
        """Deterministic torsion-like noise using provided generator."""
        noise = torch.randn(coords.shape[0], coords.shape[1], generator=gen, dtype=coords.dtype) * std * 0.1
        return coords + noise
    
    def _calculate_rmsd(self, coords1: torch.Tensor, coords2: torch.Tensor) -> float:
        """
        Calculate RMSD between two sets of coordinates.
        
        Args:
            coords1: First set of coordinates (N, 3)
            coords2: Second set of coordinates (N, 3)
            
        Returns:
            RMSD value
        """
        diff = coords1 - coords2
        rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))
        return rmsd.item()
    
    def _add_flow_components(self, sample: Dict) -> Dict:
        """
        Add flow matching components to the sample.
        
        Args:
            sample: Data dictionary
            
        Returns:
            Updated sample with flow matching components
        """
        # For both train and val/test, if perturbed/crystal are available, build flow targets
        if 'ligand_coords_crystal' in sample and 'ligand_coords_perturbed' in sample:
            if self.split == 'train':
                t = torch.rand(1)
            else:
                # Fixed t for validation/test if provided
                fixed_t = 1.0 if self.val_fixed_t is None else float(self.val_fixed_t)
                t = torch.tensor([fixed_t], dtype=torch.float32)
            sample['t'] = t

            x0 = sample['ligand_coords_perturbed']
            x1 = sample['ligand_coords_crystal']
            xt = (1 - t) * x0 + t * x1
            sample['ligand_coords_t'] = xt
            
            # Keep original coordinates for RMSD calculation
            sample['ligand_coords_0'] = x1  # Crystal/target coordinates

            v = x1 - x0
            sample['vector_field'] = v
        else:
            # Fallback: keep previous behavior (rare if preprocessing included crystal coords)
            sample['ligand_coords_t'] = sample.get('ligand_coords', sample.get('ligand_coords_t'))
            if 't' not in sample:
                sample['t'] = torch.ones(1)
        
        return sample

    def set_epoch(self, epoch: int):
        """Optionally reseed randomness per epoch for training variability."""
        self.current_epoch = int(epoch)
        if self.split == 'train' and self.resample_every_epoch and self.perturbation_mode == 'random':
            seed = (self.seed + epoch * 1009) & 0x7FFFFFFF
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    # Removed physics-based additions; dataset now only provides geometric flow targets


def _print_coords(title: str, coords: torch.Tensor, max_rows: int = 10):
    print(f"\n{title} (shape={tuple(coords.shape)}):")
    rows = coords.shape[0]
    show = min(rows, max_rows)
    for i in range(show):
        x, y, z = coords[i].tolist()
        print(f"  [{i:3d}]  {x: .4f}  {y: .4f}  {z: .4f}")
    if rows > show:
        print(f"  ... ({rows - show} more atoms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview perturbed coordinates from PoseFlowDataset")
    parser.add_argument("--data_dir", type=str, default="data/graph", help="Directory with processed .pt files")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--index", type=int, default=0, help="Sample index to preview")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "fixed"], help="Perturbation mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_fixed_t", type=float, default=0.5, help="Fixed t for validation preview")
    parser.add_argument("--max_rows", type=int, default=10, help="Max number of atoms to print")
    parser.add_argument("--translation_std", type=float, default=2.0)
    parser.add_argument("--rotation_std", type=float, default=0.5)
    parser.add_argument("--torsion_std", type=float, default=0.3)
    parser.add_argument("--min_rmsd", type=float, default=2.0)
    parser.add_argument("--max_rmsd", type=float, default=8.0)
    args = parser.parse_args()

    perturb_cfg = {
        'translation_std': args.translation_std,
        'rotation_std': args.rotation_std,
        'torsion_std': args.torsion_std,
        'min_rmsd': args.min_rmsd,
        'max_rmsd': args.max_rmsd,
    }

    dataset = PoseFlowDataset(
        data_dir=args.data_dir,
        split=args.split,
        perturbation_config=perturb_cfg,
        cache_data=False,
        seed=args.seed,
        perturbation_mode=args.mode,
        resample_every_epoch=False,
        val_fixed_t=args.val_fixed_t,
    )

    if args.split == 'train' and args.mode == 'random':
        # Ensure deterministic single run for preview
        dataset.set_epoch(0)

    if len(dataset) == 0:
        raise SystemExit("No samples found in data_dir.")

    sample = dataset[args.index % len(dataset)]

    # Print summary
    pdb_id = sample.get('pdb_id', 'unknown')
    print(f"\nSample index={args.index} pdb_id={pdb_id}")
    print(f"Split={args.split} mode={args.mode} seed={args.seed}")
    print(f"translation_std={args.translation_std} rotation_std={args.rotation_std} torsion_std={args.torsion_std}")
    print(f"min_rmsd={args.min_rmsd} max_rmsd={args.max_rmsd}")

    # Original and perturbed coordinates
    if 'ligand_coords' in sample:
        _print_coords("Original ligand_coords (from file)", sample['ligand_coords'], args.max_rows)
    if 'ligand_coords_crystal' in sample:
        _print_coords("Crystal ligand_coords (x1)", sample['ligand_coords_crystal'], args.max_rows)
    if 'ligand_coords_perturbed' in sample:
        _print_coords("Perturbed ligand_coords (x0)", sample['ligand_coords_perturbed'], args.max_rows)
        print(f"RMSD(x0, x1) = {float(sample.get('perturbation_rmsd', torch.tensor(-1.0))):.4f} Å")

    # Interpolated coordinates at t
    if 'ligand_coords_t' in sample and 't' in sample:
        t_val = float(sample['t'].view(-1)[0].item())
        _print_coords(f"Interpolated ligand_coords_t at t={t_val:.2f}", sample['ligand_coords_t'], args.max_rows)

    # Training target vector field
    if 'vector_field' in sample and torch.is_tensor(sample['vector_field']):
        _print_coords("Target vector_field (x1 - x0)", sample['vector_field'], args.max_rows)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching graph data.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data dictionary
    """
    # Simple batching - can be improved with proper graph batching
    collated = {}
    
    # Keys that should be stacked (scalars)
    stack_keys = ['t', 'perturbation_rmsd', 'ligand_num_atoms', 'num_protein_residues', 
                  'num_ligand_atoms']
    
    # Keys that should be padded and batched
    pad_keys = [
        'ligand_coords', 'ligand_coords_crystal', 'ligand_coords_perturbed',
        'ligand_coords_t', 'ligand_coords_0', 'vector_field', 'ligand_x', 'ligand_center'
    ]
    
    # Keys that should be passed as lists (protein data, edge indices, etc.)
    list_keys = [
        'protein_coord_CA', 'protein_coord_SC', 'protein_x', 
        'protein_edge_index', 'protein_edge_distance', 'protein_edge_relative_pos',
        'protein_edge_vectors', 'protein_node_vectors', 'binding_site_label',
        'ligand_edge_index', 'pdb_id'
    ]
    
    for key in batch[0].keys():
        if key in stack_keys:
            # Ensure all elements are tensors before stacking
            values = []
            for sample in batch:
                val = sample[key]
                if not torch.is_tensor(val):
                    val = torch.tensor(val)
                values.append(val)
            collated[key] = torch.stack(values)
        elif key in pad_keys:
            # Find max size for padding
            max_size = max(sample[key].shape[0] for sample in batch if key in sample)
            padded = []
            masks = []
            
            for sample in batch:
                if key in sample:
                    data = sample[key]
                    if data.shape[0] < max_size:
                        # Pad with zeros
                        pad_size = max_size - data.shape[0]
                        if data.dim() == 1:
                            padded_data = torch.cat([
                                data,
                                torch.zeros(pad_size, dtype=data.dtype)
                            ])
                        else:
                            padded_data = torch.cat([
                                data,
                                torch.zeros(pad_size, *data.shape[1:], dtype=data.dtype)
                            ])
                    else:
                        padded_data = data
                    
                    # Create mask
                    mask = torch.ones(max_size, dtype=torch.bool)
                    mask[data.shape[0]:] = False
                    
                    padded.append(padded_data)
                    masks.append(mask)
            
            if padded:
                collated[key] = torch.stack(padded)
                collated[f'{key}_mask'] = torch.stack(masks)
        elif key in list_keys:
            # For protein data and other graph data, keep as list
            collated[key] = [sample.get(key) for sample in batch if key in sample]
        else:
            # For any other keys not specified, try to handle intelligently
            values = [sample.get(key) for sample in batch if key in sample]
            if values and torch.is_tensor(values[0]):
                # If it's a tensor, try to stack
                try:
                    collated[key] = torch.stack(values)
                except:
                    # If stacking fails, keep as list
                    collated[key] = values
            else:
                # Keep as list for non-tensor data
                collated[key] = values
    
    return collated