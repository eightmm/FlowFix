"""
ODE sampling for SE(3) + Torsion decomposition.

Applies Torsion → Translation → Rotation at each Euler step.
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .losses_torsion import _rodrigues_rotate, _apply_rotation


@torch.no_grad()
def sample_trajectory_torsion(
    model: torch.nn.Module,
    protein_batch,
    ligand_batch,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    rotatable_edges: torch.Tensor,
    mask_rotate: torch.Tensor,
    return_trajectory: bool = False,
) -> dict:
    """
    Sample trajectory using SE(3) + Torsion decomposition.

    Args:
        model: Flow matching model (ProteinLigandFlowMatchingTorsion)
        protein_batch: Protein PyG batch
        ligand_batch: Ligand PyG batch
        x0: Initial docked coordinates [N_atoms, 3]
        timesteps: Integration timesteps [num_steps + 1]
        rotatable_edges: [M, 2] rotatable bond atom indices
        mask_rotate: [M, N] boolean mask for torsion
        return_trajectory: Whether to return full trajectory

    Returns:
        Dict with 'final_coords', optionally 'trajectory'
    """
    device = x0.device
    batch_size = ligand_batch.batch.max().item() + 1
    num_steps = len(timesteps) - 1

    current_coords = x0.clone()
    trajectory = [current_coords.clone()] if return_trajectory else []

    for step in range(num_steps):
        t_current = timesteps[step]
        t_next = timesteps[step + 1]
        dt = (t_next - t_current).item()

        t = torch.ones(batch_size, device=device) * t_current

        ligand_batch_t = ligand_batch.clone()
        ligand_batch_t.pos = current_coords.clone()

        # Unwrap DDP if needed
        model_fn = model.module if isinstance(model, DDP) else model
        output = model_fn(protein_batch, ligand_batch_t, t, rotatable_edges=rotatable_edges)

        # Apply per molecule: Torsion → Translation → Rotation
        for b in range(batch_size):
            mol_mask = (ligand_batch.batch == b)
            mol_coords = current_coords[mol_mask].clone()
            mol_indices = torch.where(mol_mask)[0]
            n_atoms = mol_indices.shape[0]
            offset = mol_indices[0].item()

            # 1. Torsion
            if output['torsion'].numel() > 0 and mask_rotate.shape[0] > 0:
                for m in range(mask_rotate.shape[0]):
                    angle = dt * output['torsion'][m].item()
                    if abs(angle) < 1e-6:
                        continue

                    mask_local = mask_rotate[m, offset:offset + n_atoms]
                    if not mask_local.any():
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

                    mol_coords = _rodrigues_rotate(
                        mol_coords, mask_local, axis, mol_coords[dst_l],
                        torch.tensor(angle, device=device)
                    )

            # 2. Translation
            mol_coords = mol_coords + dt * output['translation'][b]

            # 3. Rotation
            mol_coords = _apply_rotation(mol_coords, dt * output['rotation'][b])

            current_coords[mol_mask] = mol_coords

        if return_trajectory:
            trajectory.append(current_coords.clone())

    result = {"final_coords": current_coords}
    if return_trajectory:
        result["trajectory"] = trajectory
    return result
