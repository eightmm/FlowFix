"""
Training utilities for FlowFix
"""

import torch
import torch.optim as optim
from src.utils.scheduler import CosineAnnealingWarmUpRestarts


def sample_timesteps_logistic_normal(batch_size, device, mu=0.8, sigma=1.7, mix_ratio=0.9):
    """
    Sample timesteps with logistic-normal resampling:
    p(t) = mix_ratio * LN(mu, sigma) + (1-mix_ratio) * U(0, 1)

    This biases sampling towards t=1 (crystal structure), helping the model
    learn refined atomic positions in ligand binding poses.

    Inspired by timestep resampling in Boltz-1 (Wohlwend et al., 2024):
    - 90% logistic-normal: concentrates near clean data (t=1)
    - 10% uniform: maintains coverage of full trajectory

    Args:
        batch_size: Number of timesteps to sample
        device: torch device
        mu: Mean of the normal distribution (before sigmoid), default 0.8
        sigma: Std of the normal distribution (before sigmoid), default 1.7
        mix_ratio: Probability of using logistic-normal vs uniform, default 0.9

    Returns:
        t: Sampled timesteps in [0, 1], shape [batch_size]
    """
    # Sample mixture indicator
    use_logistic_normal = torch.rand(batch_size, device=device) < mix_ratio

    # Logistic-normal: sigmoid(Normal(mu, sigma))
    # This creates a distribution that concentrates near t=1
    normal_samples = torch.randn(batch_size, device=device) * sigma + mu
    logistic_normal_samples = torch.sigmoid(normal_samples)

    # Uniform samples: fallback for diversity
    uniform_samples = torch.rand(batch_size, device=device)

    # Mix the two distributions
    t = torch.where(use_logistic_normal, logistic_normal_samples, uniform_samples)

    # Clamp to ensure [0, 1] (numerical stability)
    t = torch.clamp(t, min=0.0, max=1.0)

    return t


def build_optimizer_and_scheduler(model, training_config):
    """
    Build optimizer and learning rate scheduler.

    Args:
        model: PyTorch model
        training_config: Training configuration dict

    Returns:
        Tuple of (optimizer, scheduler)
    """
    opt_config = training_config.get('optimizer', {})

    lr = training_config.get('learning_rate', opt_config.get('lr', 1e-4))
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))

    # Build optimizer
    if opt_config.get('type', 'adam').lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=opt_config.get('weight_decay', 0.0),
            betas=betas,
            eps=opt_config.get('eps', 1e-8)
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=opt_config.get('weight_decay', 0.0),
            betas=betas,
            eps=opt_config.get('eps', 1e-8)
        )

    # Build scheduler
    scheduler_config = training_config.get('scheduler', {})
    eta_max = scheduler_config.get('eta_max', lr)  # Default to initial LR if not specified
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=scheduler_config.get('T_0', 20),
        T_mult=scheduler_config.get('T_mult', 1),
        eta_max=eta_max,
        T_up=scheduler_config.get('T_up', 5),
        gamma=scheduler_config.get('gamma', 0.9)
    )

    return optimizer, scheduler
