"""
Torsion-aware dataset and collation for SE(3) + Torsion decomposition training.

FlowFixTorsionDataset extends FlowFixDataset to compute and return
torsion decomposition data (translation, rotation, torsion_changes, mask_rotate).
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any

from .dataset import FlowFixDataset, collate_flowfix_batch


class FlowFixTorsionDataset(FlowFixDataset):
    """
    FlowFixDataset with SE(3) + Torsion decomposition.

    Returns additional torsion_data dict with:
    - translation [3], rotation [3]
    - torsion_changes [M], rotatable_edges [M, 2], mask_rotate [M, N]
    """

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = super().__getitem__(idx)
        if sample is None:
            return None

        # Compute torsion decomposition
        pdb_id = self.pdb_ids[idx]
        pdb_dir = self.data_dir / pdb_id

        # Reload ligand_data to access edges (parent only returns coords)
        if self.loading_mode == "preload":
            ligands_list = self.preloaded_data[pdb_id]['ligands']
        elif self.loading_mode == "hybrid":
            ligands_list = torch.load(pdb_dir / "ligands.pt", weights_only=False)
        else:
            ligands_list = torch.load(pdb_dir / "ligands.pt", weights_only=False)

        # Use same pose index as parent (deterministic via epoch + idx)
        rng = np.random.RandomState(self.seed + self.epoch * 10000 + idx)
        pose_idx = rng.randint(0, len(ligands_list))
        ligand_data = ligands_list[pose_idx]

        torsion_data = _compute_torsion_data(
            ligand_data,
            sample['ligand_coords_x0'],
            sample['ligand_coords_x1'],
        )
        sample['torsion_data'] = torsion_data
        return sample


def _compute_torsion_data(
    ligand_data: dict,
    coords_x0: torch.Tensor,
    coords_x1: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Compute SE(3) + Torsion decomposition from ligand data.

    Returns dict with translation, rotation, torsion_changes, rotatable_edges, mask_rotate.
    """
    # Use pre-computed if available
    if 'torsion_changes' in ligand_data and 'mask_rotate' in ligand_data:
        return {
            'translation': ligand_data.get('translation', torch.zeros(3)),
            'rotation': ligand_data.get('rotation', torch.zeros(3)),
            'torsion_changes': ligand_data['torsion_changes'],
            'rotatable_edges': ligand_data.get('rotatable_edges', torch.zeros(0, 2, dtype=torch.long)),
            'mask_rotate': ligand_data['mask_rotate'],
        }

    # Compute on-the-fly
    try:
        from src.data.ligand_feat import compute_rigid_transform, get_transformation_mask

        translation, rotation = compute_rigid_transform(coords_x0, coords_x1)

        # Get edges
        edges = None
        if 'edges' in ligand_data:
            edges = ligand_data['edges']
        elif 'edge' in ligand_data and 'edges' in ligand_data['edge']:
            edges = ligand_data['edge']['edges']

        n_atoms = coords_x0.shape[0]
        empty_result = {
            'translation': translation,
            'rotation': rotation,
            'torsion_changes': torch.zeros(0),
            'rotatable_edges': torch.zeros(0, 2, dtype=torch.long),
            'mask_rotate': torch.zeros(0, n_atoms, dtype=torch.bool),
        }

        if edges is None:
            return empty_result

        mask_rotate, rotatable_edge_indices = get_transformation_mask(edges, n_atoms)

        if len(rotatable_edge_indices) == 0:
            return empty_result

        edges_np = edges.numpy() if torch.is_tensor(edges) else edges
        rot_edges = torch.tensor(
            [[int(edges_np[0, i]), int(edges_np[1, i])] for i in rotatable_edge_indices],
            dtype=torch.long,
        )
        mask_rot_filtered = mask_rotate[rotatable_edge_indices]

        return {
            'translation': translation,
            'rotation': rotation,
            'torsion_changes': torch.zeros(len(rotatable_edge_indices)),
            'rotatable_edges': rot_edges,
            'mask_rotate': mask_rot_filtered,
        }

    except Exception:
        return None


def collate_torsion_batch(samples: List[Dict]) -> Dict[str, Any]:
    """
    Collate batch with torsion data.

    Wraps collate_flowfix_batch and adds torsion_data collation.
    """
    # Filter None
    samples = [s for s in samples if s is not None]
    if not samples:
        raise ValueError("All samples in batch are None!")

    # Base collation (protein, ligand, coords, distance bounds)
    batch = collate_flowfix_batch(samples)

    # Collate torsion data
    torsion_list = [s.get('torsion_data', None) for s in samples]
    batch['torsion_data'] = _collate_torsion_data(torsion_list, batch['ligand_graph'])

    return batch


def _collate_torsion_data(
    torsion_data_list: List[Optional[Dict[str, torch.Tensor]]],
    ligand_batch,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate torsion data across batch samples.

    Concatenates variable-length rotatable bonds, adjusts atom indices
    with batch offsets, and expands mask_rotate to full batch size.
    """
    valid = [td for td in torsion_data_list if td is not None]
    if not valid:
        return None

    total_atoms = ligand_batch.num_nodes
    atom_counts = torch.bincount(ligand_batch.batch)
    atom_offsets = torch.cat([torch.zeros(1, dtype=torch.long), atom_counts.cumsum(0)[:-1]])

    translations = []
    rotations = []
    torsion_changes = []
    rotatable_edges = []
    mask_rotate_list = []

    for i, td in enumerate(torsion_data_list):
        if td is None:
            translations.append(torch.zeros(3))
            rotations.append(torch.zeros(3))
            continue

        translations.append(td['translation'])
        rotations.append(td['rotation'])

        if td['torsion_changes'].numel() > 0:
            torsion_changes.append(td['torsion_changes'])

            offset = atom_offsets[i].item()
            edges = td['rotatable_edges'].clone() + offset
            rotatable_edges.append(edges)

            # Expand mask to full batch atom count
            m_i = td['mask_rotate'].shape[0]
            n_i = td['mask_rotate'].shape[1]
            full_mask = torch.zeros(m_i, total_atoms, dtype=torch.bool)
            full_mask[:, offset:offset + n_i] = td['mask_rotate']
            mask_rotate_list.append(full_mask)

    result = {
        'translation': torch.stack(translations),
        'rotation': torch.stack(rotations),
    }

    if torsion_changes:
        result['torsion_changes'] = torch.cat(torsion_changes)
        result['rotatable_edges'] = torch.cat(rotatable_edges)
        result['mask_rotate'] = torch.cat(mask_rotate_list)
    else:
        result['torsion_changes'] = torch.zeros(0)
        result['rotatable_edges'] = torch.zeros(0, 2, dtype=torch.long)
        result['mask_rotate'] = torch.zeros(0, total_atoms, dtype=torch.bool)

    return result
