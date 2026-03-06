"""
Loss functions for SE(3) + Torsion decomposition training.
"""

import torch
import torch.nn.functional as F


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

    Args:
        pred: Model output with 'translation' [B,3], 'rotation' [B,3], 'torsion' [M]
        target: Target with 'translation' [B,3], 'rotation' [B,3], 'torsion_changes' [M]
        coords_x0: Docked coordinates [N, 3]
        coords_x1: Crystal coordinates [N, 3]
        mask_rotate: [M, N] boolean mask
        rotatable_edges: [M, 2] atom indices
        batch_indices: [N] batch assignment
        w_trans, w_rot, w_tor, w_coord: Component weights

    Returns:
        Dict with 'total', 'translation', 'rotation', 'torsion', 'coord_recon' losses
    """
    device = pred['translation'].device

    # Translation MSE
    loss_trans = F.mse_loss(pred['translation'], target['translation'].to(device))

    # Rotation MSE (axis-angle)
    loss_rot = F.mse_loss(pred['rotation'], target['rotation'].to(device))

    # Torsion circular MSE
    if pred['torsion'].numel() > 0 and target['torsion_changes'].numel() > 0:
        target_tor = target['torsion_changes'].to(device)
        diff = pred['torsion'] - target_tor
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))  # wrap to [-pi, pi]
        loss_tor = (diff ** 2).mean()
    else:
        loss_tor = torch.zeros(1, device=device).squeeze()

    # Coordinate reconstruction loss
    loss_coord = torch.zeros(1, device=device).squeeze()
    if w_coord > 0:
        reconstructed = reconstruct_coords(
            coords_x0, pred, mask_rotate, rotatable_edges, batch_indices
        )
        loss_coord = F.mse_loss(reconstructed, coords_x1.to(device))

    total = w_trans * loss_trans + w_rot * loss_rot + w_tor * loss_tor + w_coord * loss_coord

    return {
        'total': total,
        'translation': loss_trans.detach(),
        'rotation': loss_rot.detach(),
        'torsion': loss_tor.detach(),
        'coord_recon': loss_coord.detach(),
    }


def reconstruct_coords(
    coords_x0: torch.Tensor,
    pred: dict,
    mask_rotate: torch.Tensor,
    rotatable_edges: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct coordinates from SE(3) + Torsion prediction.

    Apply order: Torsion → Translation → Rotation.
    """
    device = coords_x0.device
    coords = coords_x0.clone()
    batch_size = pred['translation'].shape[0]

    for b in range(batch_size):
        mol_mask = (batch_indices == b)
        mol_coords = coords[mol_mask]
        mol_indices = torch.where(mol_mask)[0]
        n_atoms = mol_indices.shape[0]
        offset = mol_indices[0].item()

        # 1. Torsion
        if pred['torsion'].numel() > 0 and mask_rotate.shape[0] > 0:
            mol_coords = _apply_torsions(
                mol_coords, pred['torsion'], mask_rotate,
                rotatable_edges, offset, n_atoms, device
            )

        # 2. Translation
        mol_coords = mol_coords + pred['translation'][b]

        # 3. Rotation around CoM
        mol_coords = _apply_rotation(mol_coords, pred['rotation'][b])

        coords[mol_mask] = mol_coords

    return coords


def _apply_torsions(mol_coords, torsion_values, mask_rotate, rotatable_edges, offset, n_atoms, device):
    """Apply torsion angle changes to molecule coordinates."""
    for m in range(mask_rotate.shape[0]):
        angle = torsion_values[m]
        if angle.abs() < 1e-6:
            continue

        mask = mask_rotate[m, offset:offset + n_atoms]
        if not mask.any():
            continue

        src, dst = rotatable_edges[m]
        src_l = src.item() - offset
        dst_l = dst.item() - offset
        if not (0 <= src_l < n_atoms and 0 <= dst_l < n_atoms):
            continue

        axis = mol_coords[dst_l] - mol_coords[src_l]
        axis_norm = axis.norm()
        if axis_norm < 1e-6:
            continue
        axis = axis / axis_norm

        mol_coords = _rodrigues_rotate(mol_coords, mask, axis, mol_coords[dst_l], angle)

    return mol_coords


def _rodrigues_rotate(coords, mask, axis, pivot, angle):
    """Rodrigues rotation of masked atoms around axis through pivot."""
    relative = coords[mask] - pivot
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    dot = (relative * axis).sum(dim=-1, keepdim=True)
    cross = torch.cross(axis.unsqueeze(0).expand_as(relative), relative, dim=-1)
    rotated = relative * cos_a + cross * sin_a + axis * dot * (1 - cos_a)
    coords = coords.clone()
    coords[mask] = rotated + pivot
    return coords


def _apply_rotation(coords, rot_vec):
    """Apply axis-angle rotation around center of mass."""
    angle = rot_vec.norm()
    if angle < 1e-6:
        return coords
    com = coords.mean(dim=0)
    relative = coords - com
    axis = rot_vec / angle
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    dot = (relative * axis).sum(dim=-1, keepdim=True)
    cross = torch.cross(axis.unsqueeze(0).expand_as(relative), relative, dim=-1)
    rotated = relative * cos_a + cross * sin_a + axis * dot * (1 - cos_a)
    return rotated + com
