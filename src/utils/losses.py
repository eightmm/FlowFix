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
