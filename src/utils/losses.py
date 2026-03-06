"""
Loss functions for FlowFix training.

Contains auxiliary losses that supplement the main velocity loss:
- Distance geometry constraint loss
- Protein-ligand clash loss
"""

import torch


def compute_distance_geometry_loss(
    x_pred: torch.Tensor,
    batch_indices: torch.Tensor,
    distance_bounds: dict,
    t: torch.Tensor,
    device: torch.device,
    weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute distance geometry constraint loss.

    Penalizes predicted coordinates that violate lower/upper distance bounds
    between atom pairs within a molecule.

    Args:
        x_pred: Predicted coordinates [N_atoms, 3]
        batch_indices: Batch assignment for each atom [N_atoms]
        distance_bounds: Dict with 'lower', 'upper' [B, max_atoms, max_atoms], 'num_atoms' [B]
        t: Timesteps [B] - used for time-aware weighting
        device: Torch device
        weight: Maximum weight for DG loss

    Returns:
        Scalar loss tensor
    """
    distance_lower_bounds = distance_bounds["lower"].to(device)
    distance_upper_bounds = distance_bounds["upper"].to(device)
    num_atoms = distance_bounds["num_atoms"].to(device)

    batch_size = distance_lower_bounds.shape[0]
    max_atoms = distance_lower_bounds.shape[1]

    # Create padded tensor for predicted coordinates
    x_pred_padded = torch.zeros(batch_size, max_atoms, 3, device=device)

    atom_counts = torch.bincount(batch_indices, minlength=batch_size)
    atom_offsets = torch.cat(
        [torch.tensor([0], device=device), atom_counts.cumsum(0)[:-1]]
    )
    atom_indices_within_mol = (
        torch.arange(len(batch_indices), device=device) - atom_offsets[batch_indices]
    )

    x_pred_padded[batch_indices, atom_indices_within_mol] = x_pred

    # Compute pairwise distances
    dists = torch.cdist(x_pred_padded, x_pred_padded)

    # Compute violations
    lower_violation = torch.relu(distance_lower_bounds - dists)
    upper_violation = torch.relu(dists - distance_upper_bounds)

    # Valid mask for actual atoms (not padding)
    atom_range = torch.arange(max_atoms, device=device).unsqueeze(0)
    valid_atom_mask = atom_range < num_atoms.unsqueeze(1)
    valid_mask = valid_atom_mask.unsqueeze(2) & valid_atom_mask.unsqueeze(1)

    # Time-aware weighting (stronger constraint near t=1)
    time_weight = t.unsqueeze(-1).unsqueeze(-1) * weight
    time_weight = time_weight.expand(-1, max_atoms, max_atoms)

    # Compute masked losses
    masked_lower_violation = lower_violation * valid_mask.float() * time_weight
    masked_upper_violation = upper_violation * valid_mask.float() * time_weight
    dg_loss = (masked_lower_violation.sum() + masked_upper_violation.sum()) / batch_size

    return dg_loss


def compute_clash_loss(
    x_pred: torch.Tensor,
    ligand_batch_indices: torch.Tensor,
    protein_batch: "torch_geometric.data.Batch",
    t: torch.Tensor,
    device: torch.device,
    ca_threshold: float = 3.0,
    sc_threshold: float = 2.5,
    margin: float = 1.0,
    weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute protein-ligand clash loss.

    Penalizes predicted ligand coordinates that are too close to protein atoms.
    Separate thresholds for CA (backbone) and SC (sidechain) atoms.

    Args:
        x_pred: Predicted ligand coordinates [N_ligand, 3]
        ligand_batch_indices: Batch assignment for each ligand atom [N_ligand]
        protein_batch: Protein PyG batch with .pos (CA) and .pos_sc (SC) coordinates
        t: Timesteps [B] - used for time-aware weighting
        device: Torch device
        ca_threshold: Minimum allowed distance to CA atoms
        sc_threshold: Minimum allowed distance to SC atoms
        margin: Soft margin for clash penalty
        weight: Maximum weight for clash loss

    Returns:
        Tuple of (ca_clash_loss, sc_clash_loss)
    """
    protein_batch_indices = protein_batch.batch
    batch_size = ligand_batch_indices.max().item() + 1

    # Count atoms per molecule for padding
    ligand_atom_counts = torch.bincount(ligand_batch_indices, minlength=batch_size)
    protein_atom_counts = torch.bincount(protein_batch_indices, minlength=batch_size)

    max_ligand_atoms = ligand_atom_counts.max().item()
    max_protein_atoms = protein_atom_counts.max().item()

    # Pad ligand coordinates
    x_pred_padded = torch.zeros(batch_size, max_ligand_atoms, 3, device=device)
    ligand_atom_offsets = torch.cat(
        [torch.tensor([0], device=device), ligand_atom_counts.cumsum(0)[:-1]]
    )
    ligand_atom_indices = (
        torch.arange(len(ligand_batch_indices), device=device)
        - ligand_atom_offsets[ligand_batch_indices]
    )
    x_pred_padded[ligand_batch_indices, ligand_atom_indices] = x_pred

    # Protein atom offsets and indices
    protein_atom_offsets = torch.cat(
        [torch.tensor([0], device=device), protein_atom_counts.cumsum(0)[:-1]]
    )
    protein_atom_indices = (
        torch.arange(len(protein_batch_indices), device=device)
        - protein_atom_offsets[protein_batch_indices]
    )

    # Valid mask for actual atoms (not padding)
    ligand_valid_mask = (
        torch.arange(max_ligand_atoms, device=device).unsqueeze(0)
        < ligand_atom_counts.unsqueeze(1)
    )
    protein_valid_mask = (
        torch.arange(max_protein_atoms, device=device).unsqueeze(0)
        < protein_atom_counts.unsqueeze(1)
    )
    valid_mask = ligand_valid_mask.unsqueeze(2) & protein_valid_mask.unsqueeze(1)

    # Time-aware weighting
    time_weight = t.unsqueeze(-1).unsqueeze(-1) * weight
    time_weight = time_weight.expand(-1, max_ligand_atoms, max_protein_atoms)

    # CA Clash Loss
    protein_ca_padded = torch.zeros(batch_size, max_protein_atoms, 3, device=device)
    protein_ca_padded[protein_batch_indices, protein_atom_indices] = protein_batch.pos
    ca_dists = torch.cdist(x_pred_padded, protein_ca_padded, p=2)

    ca_violations = torch.relu(ca_threshold + margin - ca_dists)
    ca_clash_penalty = ca_violations**2
    masked_ca_clash = ca_clash_penalty * valid_mask.float() * time_weight
    clash_loss_ca = masked_ca_clash.sum() / batch_size

    # SC Clash Loss
    protein_sc_padded = torch.zeros(batch_size, max_protein_atoms, 3, device=device)
    protein_sc_padded[protein_batch_indices, protein_atom_indices] = protein_batch.pos_sc
    sc_dists = torch.cdist(x_pred_padded, protein_sc_padded, p=2)

    sc_violations = torch.relu(sc_threshold + margin - sc_dists)
    sc_clash_penalty = sc_violations**2
    sc_valid_mask = valid_mask & torch.isfinite(sc_dists)
    masked_sc_clash = sc_clash_penalty * sc_valid_mask.float() * time_weight
    clash_loss_sc = masked_sc_clash.sum() / batch_size

    return clash_loss_ca, clash_loss_sc


def compute_se3_torsion_loss(
    pred: dict,
    target: dict,
    coords_x0: torch.Tensor,
    coords_x1: torch.Tensor,
    mask_rotate: torch.Tensor,
    rotatable_edges: torch.Tensor,
    batch_indices: torch.Tensor,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
    w_tor: float = 1.0,
    w_coord: float = 0.5,
) -> dict:
    """
    Compute SE(3) + Torsion decomposition loss.

    Computes weighted MSE for each component (translation, rotation, torsion)
    plus an optional coordinate reconstruction loss.

    Args:
        pred: Model output dict with 'translation' [B,3], 'rotation' [B,3], 'torsion' [M]
        target: Target dict with 'translation' [B,3], 'rotation' [B,3], 'torsion_changes' [M]
        coords_x0: Docked coordinates [N, 3]
        coords_x1: Crystal coordinates [N, 3]
        mask_rotate: [M, N] boolean mask for torsion application
        rotatable_edges: [M, 2] atom indices
        batch_indices: [N] batch assignment
        w_trans, w_rot, w_tor, w_coord: Loss weights

    Returns:
        Dict with 'total', 'translation', 'rotation', 'torsion', 'coord_recon' losses
    """
    device = pred['translation'].device
    batch_size = pred['translation'].shape[0]

    # Translation loss
    loss_trans = torch.nn.functional.mse_loss(
        pred['translation'], target['translation'].to(device)
    )

    # Rotation loss (axis-angle MSE)
    loss_rot = torch.nn.functional.mse_loss(
        pred['rotation'], target['rotation'].to(device)
    )

    # Torsion loss
    if pred['torsion'].numel() > 0 and target['torsion_changes'].numel() > 0:
        target_torsion = target['torsion_changes'].to(device)
        # Wrap to [-pi, pi] for circular loss
        diff = pred['torsion'] - target_torsion
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        loss_tor = (diff ** 2).mean()
    else:
        loss_tor = torch.zeros(1, device=device).squeeze()

    # Coordinate reconstruction loss (optional end-to-end supervision)
    loss_coord = torch.zeros(1, device=device).squeeze()
    if w_coord > 0:
        try:
            from src.data.ligand_feat import apply_torsion_updates
            reconstructed = _reconstruct_coords(
                coords_x0, pred, mask_rotate, rotatable_edges, batch_indices
            )
            loss_coord = torch.nn.functional.mse_loss(reconstructed, coords_x1.to(device))
        except Exception:
            pass  # Skip coord loss if reconstruction fails

    total = (
        w_trans * loss_trans
        + w_rot * loss_rot
        + w_tor * loss_tor
        + w_coord * loss_coord
    )

    return {
        'total': total,
        'translation': loss_trans.detach(),
        'rotation': loss_rot.detach(),
        'torsion': loss_tor.detach(),
        'coord_recon': loss_coord.detach(),
    }


def _reconstruct_coords(
    coords_x0: torch.Tensor,
    pred: dict,
    mask_rotate: torch.Tensor,
    rotatable_edges: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct coordinates from SE(3) + Torsion prediction.

    Apply order: Torsion → Translation → Rotation (same as DiffDock).

    Args:
        coords_x0: [N, 3] docked coordinates
        pred: Dict with 'translation' [B, 3], 'rotation' [B, 3], 'torsion' [M]
        mask_rotate: [M, N] boolean mask
        rotatable_edges: [M, 2] atom indices
        batch_indices: [N] batch assignment

    Returns:
        [N, 3] reconstructed coordinates
    """
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    device = coords_x0.device
    coords = coords_x0.clone()
    batch_size = pred['translation'].shape[0]

    for b in range(batch_size):
        mol_mask = (batch_indices == b)
        mol_coords = coords[mol_mask]  # [N_b, 3]

        # 1. Apply torsion updates
        if pred['torsion'].numel() > 0 and mask_rotate.shape[0] > 0:
            # Filter mask_rotate for this molecule's atoms
            mol_indices = torch.where(mol_mask)[0]
            n_atoms = mol_indices.shape[0]
            offset = mol_indices[0].item()

            for m in range(mask_rotate.shape[0]):
                angle = pred['torsion'][m].item()
                if abs(angle) < 1e-6:
                    continue

                mask = mask_rotate[m, offset:offset + n_atoms]
                if not mask.any():
                    continue

                # Rotation axis from rotatable bond
                src, dst = rotatable_edges[m]
                src_local = src.item() - offset
                dst_local = dst.item() - offset

                if src_local < 0 or src_local >= n_atoms or dst_local < 0 or dst_local >= n_atoms:
                    continue

                axis = mol_coords[dst_local] - mol_coords[src_local]
                axis_norm = axis.norm()
                if axis_norm < 1e-6:
                    continue
                axis = axis / axis_norm

                # Rodrigues rotation
                pivot = mol_coords[dst_local]
                relative = mol_coords[mask] - pivot
                cos_a = torch.cos(torch.tensor(angle, device=device))
                sin_a = torch.sin(torch.tensor(angle, device=device))
                dot = (relative * axis).sum(dim=-1, keepdim=True)
                cross = torch.cross(axis.unsqueeze(0).expand_as(relative), relative, dim=-1)
                rotated = relative * cos_a + cross * sin_a + axis * dot * (1 - cos_a)
                mol_coords[mask] = rotated + pivot

        # 2. Apply translation
        trans = pred['translation'][b]  # [3]
        mol_coords = mol_coords + trans

        # 3. Apply rotation around CoM
        rot_vec = pred['rotation'][b]  # [3] axis-angle
        rot_angle = rot_vec.norm()
        if rot_angle > 1e-6:
            com = mol_coords.mean(dim=0)
            relative = mol_coords - com

            axis = rot_vec / rot_angle
            cos_a = torch.cos(rot_angle)
            sin_a = torch.sin(rot_angle)
            dot = (relative * axis).sum(dim=-1, keepdim=True)
            cross = torch.cross(axis.unsqueeze(0).expand_as(relative), relative, dim=-1)
            rotated = relative * cos_a + cross * sin_a + axis * dot * (1 - cos_a)
            mol_coords = rotated + com

        coords[mol_mask] = mol_coords

    return coords
