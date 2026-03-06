"""
Metrics utilities for FlowFix training.

Contains functions for:
- RMSD calculation
- Success rate calculation
- DDP metric gathering
"""

from typing import List, Dict, Any
import numpy as np
import torch
import torch.distributed as dist


def compute_success_rates(rmsds: np.ndarray) -> Dict[str, float]:
    """
    Compute success rates at various RMSD thresholds.

    Args:
        rmsds: Array of RMSD values

    Returns:
        Dict with success rates: 'success_2A', 'success_1A', 'success_05A'
    """
    return {
        "success_2A": float(np.mean(rmsds < 2.0) * 100),
        "success_1A": float(np.mean(rmsds < 1.0) * 100),
        "success_05A": float(np.mean(rmsds < 0.5) * 100),
    }


def gather_metrics_ddp(
    local_metrics: Dict[str, List[float]],
    world_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Gather metrics from all DDP processes.

    Handles variable-length lists across processes by padding before all_gather.

    Args:
        local_metrics: Dict of metric_name -> list of values
        world_size: Number of DDP processes
        device: Torch device

    Returns:
        Dict of metric_name -> numpy array of all gathered values
    """
    gathered = {}

    for metric_name, values in local_metrics.items():
        values_tensor = torch.tensor(values, device=device)

        # Gather sizes from all ranks
        local_size = torch.tensor([len(values)], device=device, dtype=torch.long)
        all_sizes = [
            torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)
        ]
        dist.all_gather(all_sizes, local_size)

        # Find max size for padding
        max_size = max(s.item() for s in all_sizes)

        # Pad tensor to max size
        padded = torch.zeros(max_size, device=device)
        padded[: len(values)] = values_tensor

        # Gather padded tensors
        gathered_tensors = [
            torch.zeros(max_size, device=device) for _ in range(world_size)
        ]
        dist.all_gather(gathered_tensors, padded)

        # Unpad and concatenate
        unpadded = [
            gathered_tensors[i][: all_sizes[i].item()] for i in range(world_size)
        ]
        gathered[metric_name] = torch.cat(unpadded).cpu().numpy()

    return gathered


def average_metrics_ddp(
    local_metrics: List[float],
    world_size: int,
    device: torch.device,
) -> float:
    """
    Average a list of metrics across all DDP processes.

    Args:
        local_metrics: List of metric values from this process
        world_size: Number of DDP processes
        device: Torch device

    Returns:
        Averaged metric value
    """
    # Convert to tensor and compute local stats
    tensor = torch.tensor(local_metrics, device=device)

    # All-reduce sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Average across GPUs and samples
    return tensor.mean().item() / world_size
