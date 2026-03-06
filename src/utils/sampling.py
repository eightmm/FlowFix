"""
ODE sampling utilities for FlowFix.

Contains functions for:
- Timestep schedule generation
- Euler/RK4 ODE integration
- Trajectory sampling
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def generate_timestep_schedule(
    num_steps: int,
    schedule: str,
    device: torch.device,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """
    Generate timestep schedule for ODE integration.

    Args:
        num_steps: Number of integration steps
        schedule: Schedule type ('uniform', 'quadratic', 'root', 'sigmoid')
        device: Torch device
        t_start: Start time (default 0.0)
        t_end: End time (default 1.0, can be < 1.0 for partial integration)

    Returns:
        Tensor of timesteps [num_steps + 1] from t_start to t_end
    """
    if schedule == "uniform":
        timesteps = torch.linspace(t_start, t_end, num_steps + 1, device=device)
    elif schedule == "quadratic":
        raw = torch.linspace(0, 1, num_steps + 1, device=device)
        scaled = 1 - (1 - raw) ** 1.5
        timesteps = t_start + scaled * (t_end - t_start)
    elif schedule == "root":
        raw = torch.linspace(0, 1, num_steps + 1, device=device) ** (2 / 3)
        timesteps = t_start + raw * (t_end - t_start)
    elif schedule == "sigmoid":
        raw = torch.linspace(-6, 6, num_steps + 1, device=device)
        scaled = torch.sigmoid(raw)
        timesteps = t_start + scaled * (t_end - t_start)
    else:
        timesteps = torch.linspace(t_start, t_end, num_steps + 1, device=device)

    return timesteps


@torch.no_grad()
def sample_trajectory(
    model: torch.nn.Module,
    protein_batch,
    ligand_batch,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    method: str = "euler",
    use_self_cond: bool = False,
    return_trajectory: bool = False,
    trajectory_mask: torch.Tensor = None,
) -> dict:
    """
    Sample trajectory from x0 (docked) to x1 (crystal) using ODE integration.

    Args:
        model: Flow matching model
        protein_batch: Protein PyG batch
        ligand_batch: Ligand PyG batch (will be cloned, .pos modified during sampling)
        x0: Initial coordinates (docked pose) [N_atoms, 3]
        timesteps: Integration timesteps [num_steps + 1]
        method: Integration method ('euler' or 'rk4')
        use_self_cond: Whether to use self-conditioning
        return_trajectory: Whether to return full trajectory
        trajectory_mask: Optional mask for which atoms to track in trajectory

    Returns:
        Dict with:
            - 'final_coords': Final refined coordinates [N_atoms, 3]
            - 'trajectory': List of coordinates at each step (if return_trajectory=True)
            - 'velocities': List of velocities at each step (if return_trajectory=True)
    """
    device = x0.device
    batch_size = ligand_batch.batch.max().item() + 1
    num_steps = len(timesteps) - 1

    current_coords = x0.clone()
    x1_self_cond = None

    trajectory = []
    velocities = []

    if return_trajectory and trajectory_mask is not None:
        trajectory.append(current_coords[trajectory_mask].clone())

    for step in range(num_steps):
        t_current = timesteps[step]
        t_next = timesteps[step + 1]
        dt = t_next - t_current

        t = torch.ones(batch_size, device=device) * t_current

        # Clone ligand batch and update position
        ligand_batch_t = ligand_batch.clone()
        ligand_batch_t.pos = current_coords.clone()

        # Forward pass
        velocity = model(protein_batch, ligand_batch_t, t, x1_self_cond=x1_self_cond)

        # Update self-conditioning for next step
        if use_self_cond:
            t_expanded = t[ligand_batch_t.batch].unsqueeze(-1)
            x1_self_cond = current_coords + (1 - t_expanded) * velocity

        # Track trajectory
        if return_trajectory and trajectory_mask is not None:
            velocities.append(velocity[trajectory_mask].clone())

        # Integration step
        if method == "euler":
            current_coords = current_coords + dt * velocity
        elif method == "rk4":
            k1 = velocity

            t_mid = t_current + 0.5 * dt
            ligand_batch_t.pos = (current_coords + 0.5 * dt * k1).clone()
            k2 = model(
                protein_batch,
                ligand_batch_t,
                torch.ones(batch_size, device=device) * t_mid,
                x1_self_cond=x1_self_cond,
            )

            ligand_batch_t.pos = (current_coords + 0.5 * dt * k2).clone()
            k3 = model(
                protein_batch,
                ligand_batch_t,
                torch.ones(batch_size, device=device) * t_mid,
                x1_self_cond=x1_self_cond,
            )

            ligand_batch_t.pos = (current_coords + dt * k3).clone()
            k4 = model(
                protein_batch,
                ligand_batch_t,
                torch.ones(batch_size, device=device) * t_next,
                x1_self_cond=x1_self_cond,
            )

            current_coords = current_coords + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Track trajectory after step
        if return_trajectory and trajectory_mask is not None:
            trajectory.append(current_coords[trajectory_mask].clone())

    result = {"final_coords": current_coords}
    if return_trajectory:
        result["trajectory"] = trajectory
        result["velocities"] = velocities

    return result


def get_model_self_cond_flag(model: torch.nn.Module) -> bool:
    """Get self_conditioning flag from model, handling DDP wrapping."""
    model_module = model.module if isinstance(model, DDP) else model
    return getattr(model_module, "self_conditioning", False)
