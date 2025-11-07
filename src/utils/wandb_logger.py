"""
Centralized WandB logging utilities for FlowFix training.

Provides a clean, hierarchical structure for all logging:
- meta/         : Training metadata (LR, epoch, step)
- train/        : Training metrics (loss, RMSD)
- validation/   : Validation metrics (loss, RMSD, success rates)
- gradients/    : Key gradient statistics (total norm, module norms)
- parameters/   : Key parameter statistics (module-level only)
"""

import wandb
import numpy as np
import torch


class WandBLogger:
    """Centralized WandB logging with clean hierarchy."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def log(self, metrics_dict):
        """Log metrics if enabled."""
        if self.enabled:
            wandb.log(metrics_dict)

    # ==================== Training Metrics ====================

    def log_training_step(self, loss, rmsd, lr, epoch, step):
        """Log training step metrics."""
        self.log({
            # Training metrics
            'train/loss': loss,
            'train/rmsd': rmsd,

            # Metadata
            'meta/learning_rate': lr,
            'meta/epoch': epoch,
            'meta/step': step
        })

    # ==================== Validation Metrics ====================

    def log_validation_epoch(self, val_loss, val_rmsd, val_rmsd_initial, val_rmsd_final,
                            success_rate_2a, success_rate_1a, success_rate_05a, epoch):
        """Log validation epoch metrics."""
        self.log({
            # Core validation metrics
            'validation/loss': val_loss,
            'validation/rmsd': val_rmsd,
            'validation/rmsd_initial': val_rmsd_initial,
            'validation/rmsd_final': val_rmsd_final,

            # Success rates
            'validation/success_rate_2A': success_rate_2a,
            'validation/success_rate_1A': success_rate_1a,
            'validation/success_rate_0.5A': success_rate_05a,

            # Metadata
            'meta/epoch': epoch
        })

    # ==================== Gradient Logging ====================

    def log_gradient_norms(self, total_norm, module_norms, step):
        """Log gradient norms (total and key modules only)."""
        metrics = {
            'gradients/total_norm': total_norm,
            'meta/step': step
        }

        # Key module norms only
        for module_name, (avg_norm, max_norm) in module_norms.items():
            metrics[f'gradients/{module_name}_avg'] = avg_norm
            metrics[f'gradients/{module_name}_max'] = max_norm

        self.log(metrics)

    # ==================== Parameter Logging ====================

    def log_parameter_stats(self, module_stats):
        """Log parameter statistics (module-level only).

        Args:
            module_stats: Dict[module_name, Dict[stat_name, value]]
        """
        metrics = {}
        for module_name, stats in module_stats.items():
            for stat_name, value in stats.items():
                metrics[f'parameters/{module_name}_{stat_name}'] = value

        if metrics:
            self.log(metrics)

    # ==================== Visualization ====================

    def log_animation(self, animation_path, epoch, pdb_id=None):
        """Log validation animation GIF."""
        try:
            caption = f"Epoch {epoch}"
            if pdb_id:
                caption += f" - {pdb_id}"

            self.log({
                f'validation/trajectory_epoch_{epoch}': wandb.Video(animation_path, caption=caption),
                'meta/epoch': epoch
            })
        except Exception as e:
            print(f"Warning: Could not log animation: {e}")

    # ==================== System Info ====================

    def log_early_stopping(self, epoch):
        """Log early stopping event."""
        self.log({
            'meta/early_stopped': True,
            'meta/epoch': epoch
        })


def extract_module_gradient_norms(model):
    """Extract gradient norms grouped by module.

    Returns:
        total_norm: float
        module_norms: Dict[module_name, (avg_norm, max_norm)]
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

    module_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Extract module type (first part of name)
            module_type = name.split('.')[0]

            grad_norm = param.grad.data.norm(2).item()

            if module_type not in module_norms:
                module_norms[module_type] = []
            module_norms[module_type].append(grad_norm)

    # Calculate average and max for each module
    module_stats = {}
    for module_type, norms in module_norms.items():
        module_stats[module_type] = (np.mean(norms), np.max(norms))

    return total_norm.item(), module_stats


def extract_parameter_stats(model):
    """Extract parameter statistics by module (simplified).

    Returns:
        module_stats: Dict[module_name, Dict[stat_name, value]]
    """
    module_data = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Extract module type (first part of name)
        module_type = name.split('.')[0]

        param_flat = param.detach().cpu().numpy().flatten()

        if len(param_flat) == 0:
            continue

        # Collect module data
        if module_type not in module_data:
            module_data[module_type] = []
        module_data[module_type].extend(param_flat.tolist())

    # Calculate statistics
    module_stats = {}
    for module_type, data in module_data.items():
        data_array = np.array(data)
        module_stats[module_type] = {
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'norm': float(np.linalg.norm(data_array)),
            'max_abs': float(np.max(np.abs(data_array)))
        }

    return module_stats
