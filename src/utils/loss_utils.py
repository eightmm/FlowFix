"""
Simplified loss utilities for FlowFix training.
Only velocity matching loss with logistic-normal time sampling.
"""

import torch
import torch.distributions as dist


def compute_flow_matching_loss(pred_velocity, true_velocity, batch_indices):
    """
    Simple uniform flow matching loss: pure MSE on velocity field.

    For linear interpolation, true velocity is constant across all t,
    so all timesteps should be equally weighted.

    Args:
        pred_velocity: Predicted velocity, shape [N, 3]
        true_velocity: True velocity (x1 - x0), shape [N, 3]
        batch_indices: Which batch each atom belongs to, shape [N] (unused but kept for compatibility)

    Returns:
        loss: Uniform MSE loss
        loss_dict: Dictionary with loss components
    """
    # Simple uniform MSE loss (no per-sample normalization, no weighting)
    total_loss = torch.nn.functional.mse_loss(pred_velocity, true_velocity)

    loss_dict = {
        'velocity_loss': total_loss.item(),
        'total_loss': total_loss.item()
    }

    return total_loss, loss_dict


def sample_timesteps_logistic_normal(batch_size, device, mu=0.0, sigma=1.0, mix_ratio=0.5):
    """
    Sample timesteps using logistic-normal distribution.

    Mixture of:
    - Logistic-normal: sigma(Normal(mu, sigma)) where sigma is sigmoid
    - Uniform: for diversity

    Args:
        batch_size: Number of timesteps to sample
        device: torch device
        mu: Mean of underlying normal (logit space)
        sigma: Std of underlying normal (logit space)
        mix_ratio: Fraction to sample from logistic-normal (rest is uniform)

    Returns:
        t: Sampled timesteps in [0, 1], shape [batch_size]
    """
    n_logistic = int(batch_size * mix_ratio)
    n_uniform = batch_size - n_logistic

    # Logistic-normal samples
    normal_dist = dist.Normal(mu, sigma)
    logit_samples = normal_dist.sample((n_logistic,)).to(device)
    logistic_samples = torch.sigmoid(logit_samples)

    # Uniform samples
    uniform_samples = torch.rand(n_uniform, device=device)

    # Concatenate and shuffle
    t = torch.cat([logistic_samples, uniform_samples], dim=0)
    t = t[torch.randperm(len(t), device=device)]

    return t
