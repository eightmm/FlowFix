"""
Centralized WandB logging utilities for FlowFix training.

Provides a clean, hierarchical structure for all logging:
- metrics/train/     : Training metrics (loss, RMSD, etc.)
- metrics/val/       : Validation metrics
- gradients/         : Gradient statistics and histograms
- parameters/        : Parameter statistics and histograms
- system/            : System info (LR, epoch, step)
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
            # Core metrics
            'metrics/train/loss': loss,
            'metrics/train/rmsd': rmsd,

            # System
            'system/learning_rate': lr,
            'system/epoch': epoch,
            'system/step': step
        })

    # ==================== Validation Metrics ====================

    def log_validation_epoch(self, val_loss, val_rmsd, val_rmsd_initial, val_rmsd_final,
                            success_rate_2a, success_rate_1a, success_rate_05a, epoch):
        """Log validation epoch metrics."""
        self.log({
            # Core validation metrics
            'metrics/val/loss': val_loss,
            'metrics/val/rmsd': val_rmsd,
            'metrics/val/rmsd_initial': val_rmsd_initial,
            'metrics/val/rmsd_final': val_rmsd_final,

            # Success rates
            'metrics/val/success_rate_2A': success_rate_2a,
            'metrics/val/success_rate_1A': success_rate_1a,
            'metrics/val/success_rate_0.5A': success_rate_05a,

            # System
            'system/epoch': epoch
        })

    # ==================== Gradient Logging ====================

    def log_gradient_norms(self, total_norm, module_norms, step):
        """Log gradient norms by module."""
        metrics = {
            'gradients/norms/total': total_norm,
            'system/step': step
        }

        # Module-level norms
        for module_name, (avg_norm, max_norm) in module_norms.items():
            metrics[f'gradients/norms/{module_name}/avg'] = avg_norm
            metrics[f'gradients/norms/{module_name}/max'] = max_norm

        self.log(metrics)

    def log_gradient_stats_by_layer(self, layer_stats):
        """Log detailed per-layer gradient statistics.

        Args:
            layer_stats: Dict[layer_name, Dict[stat_name, value]]
        """
        metrics = {}
        for layer_name, stats in layer_stats.items():
            for stat_name, value in stats.items():
                metrics[f'gradients/layers/{layer_name}/{stat_name}'] = value

        if metrics:
            self.log(metrics)

    def log_gradient_histograms(self, layer_histograms):
        """Log gradient histograms by layer.

        Args:
            layer_histograms: Dict[layer_name, List[float]]
        """
        metrics = {}
        for layer_name, grad_data in layer_histograms.items():
            if len(grad_data) > 0:
                metrics[f'gradients/histograms/{layer_name}'] = wandb.Histogram(grad_data)

        if metrics:
            self.log(metrics)

    # ==================== Parameter Logging ====================

    def log_parameter_histograms(self, module_hists, layer_hists, type_hists):
        """Log parameter histograms at different granularities.

        Args:
            module_hists: Dict[module_name, (data, stats)]
            layer_hists: Dict[layer_name, (data, stats)]
            type_hists: Dict[type_name, (data, stats)]
        """
        metrics = {}

        # Module-level histograms
        for module_name, (data, stats) in module_hists.items():
            metrics[f'parameters/histograms/module/{module_name}'] = wandb.Histogram(data)
            for stat_name, value in stats.items():
                metrics[f'parameters/stats/module/{module_name}/{stat_name}'] = value

        # Layer-level histograms (limited)
        for layer_name, (data, stats) in layer_hists.items():
            metrics[f'parameters/histograms/layer/{layer_name}'] = wandb.Histogram(data)
            for stat_name, value in stats.items():
                metrics[f'parameters/stats/layer/{layer_name}/{stat_name}'] = value

        # Type-level histograms (weight vs bias)
        for type_name, (data, stats) in type_hists.items():
            metrics[f'parameters/histograms/type/{type_name}'] = wandb.Histogram(data)
            for stat_name, value in stats.items():
                metrics[f'parameters/stats/type/{type_name}/{stat_name}'] = value

        if metrics:
            self.log(metrics)

    def log_parameter_stats_by_layer(self, layer_stats):
        """Log parameter statistics and changes by layer.

        Args:
            layer_stats: Dict[layer_name, Dict[stat_name, value]]
        """
        metrics = {}
        for layer_name, stats in layer_stats.items():
            for stat_name, value in stats.items():
                metrics[f'parameters/layers/{layer_name}/{stat_name}'] = value

        if metrics:
            self.log(metrics)

    def log_parameter_evolution(self, param_values):
        """Log evolution of specific key parameters.

        Args:
            param_values: Dict[param_name, value]
        """
        metrics = {}
        for param_name, value in param_values.items():
            metrics[f'parameters/evolution/{param_name}'] = value

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
                f'visualizations/trajectory_epoch_{epoch}': wandb.Video(animation_path, caption=caption),
                'system/epoch': epoch
            })
        except Exception as e:
            print(f"Warning: Could not log animation: {e}")

    # ==================== System Info ====================

    def log_early_stopping(self, epoch):
        """Log early stopping event."""
        self.log({
            'system/early_stopped': True,
            'system/epoch': epoch
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


def extract_gradient_stats_by_layer(model, max_layers=8):
    """Extract gradient statistics by layer.

    Returns:
        layer_stats: Dict[layer_name, Dict[stat_name, value]]
        layer_histograms: Dict[layer_name, List[float]]
    """
    layer_stats = {}
    layer_histograms = {}

    logged_count = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Extract layer name
        parts = name.split('.')
        if len(parts) >= 3 and 'layers' in name:
            layer_name = '.'.join(parts[:3])
        else:
            layer_name = parts[0]

        grad_flat = param.grad.detach().cpu().numpy().flatten()

        # Calculate stats
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {
                'mean': float(np.mean(grad_flat)),
                'std': float(np.std(grad_flat)),
                'norm': float(np.linalg.norm(grad_flat)),
                'max': float(np.max(np.abs(grad_flat)))
            }

        # Collect histogram data (limit to first N layers)
        if logged_count < max_layers:
            if layer_name not in layer_histograms:
                layer_histograms[layer_name] = []
                logged_count += 1
            layer_histograms[layer_name].extend(grad_flat.tolist())

    return layer_stats, layer_histograms


def extract_parameter_histograms(model, max_layers=8):
    """Extract parameter histograms at different granularities.

    Returns:
        module_hists: Dict[module_name, (data, stats)]
        layer_hists: Dict[layer_name, (data, stats)]
        type_hists: Dict[type_name, (data, stats)]
    """
    module_data = {}
    layer_data = {}
    type_data = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_flat = param.detach().cpu().numpy().flatten()

        if len(param_flat) == 0:
            continue

        parts = name.split('.')
        module_type = parts[0]
        param_type = parts[-1]

        # Module-level
        if module_type not in module_data:
            module_data[module_type] = []
        module_data[module_type].extend(param_flat.tolist())

        # Layer-level (only for layers)
        if 'layers' in name and len(parts) >= 3:
            layer_name = '.'.join(parts[:3])
            if layer_name not in layer_data:
                layer_data[layer_name] = []
            layer_data[layer_name].extend(param_flat.tolist())

        # Type-level (weight vs bias)
        type_key = f"{module_type}_{param_type}"
        if type_key not in type_data:
            type_data[type_key] = []
        type_data[type_key].extend(param_flat.tolist())

    # Convert to (data, stats) format
    def make_stats(data):
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }

    module_hists = {k: (v, make_stats(v)) for k, v in module_data.items()}

    # Limit layers
    layer_hists = {}
    for i, (k, v) in enumerate(sorted(layer_data.items())):
        if i >= max_layers:
            break
        layer_hists[k] = (v, make_stats(v))

    type_hists = {k: (v, make_stats(v)) for k, v in type_data.items()}

    return module_hists, layer_hists, type_hists


def extract_parameter_stats_by_layer(model, param_history):
    """Extract parameter statistics and changes by layer.

    Args:
        model: PyTorch model
        param_history: Dict[param_name, previous_values]

    Returns:
        layer_stats: Dict[layer_name, Dict[stat_name, value]]
        updated_history: Dict[param_name, current_values]
    """
    layer_stats = {}
    updated_history = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Extract layer name
        parts = name.split('.')
        if len(parts) >= 3 and 'layers' in name:
            layer_name = '.'.join(parts[:3])
        else:
            layer_name = parts[0]

        param_flat = param.detach().cpu().numpy().flatten()

        # Initialize layer stats
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {
                'mean': float(np.mean(param_flat)),
                'std': float(np.std(param_flat)),
                'norm': float(np.linalg.norm(param_flat))
            }

        # Calculate change from previous
        if name in param_history:
            prev_param = param_history[name]
            param_change = np.linalg.norm(param_flat - prev_param)
            layer_stats[layer_name]['change'] = float(param_change)
            layer_stats[layer_name]['change_pct'] = float(
                param_change / (np.linalg.norm(prev_param) + 1e-8) * 100
            )

        # Update history
        updated_history[name] = param_flat.copy()

    return layer_stats, updated_history
