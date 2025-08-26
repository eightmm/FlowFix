#!/usr/bin/env python
"""
Training script for FlowFix Equivariant model with improved flow matching.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import yaml
import numpy as np

from data.dataset import PoseFlowDataset, collate_fn
from data.batch_dataset import collate_fn_batched
from models.flowfix_equivariant import FlowFixEquivariantModel
from utils.ema import EMA
from utils.initialization import init_model
from validation_diffdock import DiffDockValidator, print_validation_summary


class FlowMatchingLoss(nn.Module):
    """
    Improved flow matching loss with trajectory regularization.
    """
    
    def __init__(
        self,
        smooth_l1_beta: float = 1.0,
        lambda_traj: float = 0.1
    ):
        super().__init__()
        self.smooth_l1_beta = smooth_l1_beta
        self.lambda_traj = lambda_traj
        self.smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta)
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        pred_vector_field: torch.Tensor,
        target_vector_field: torch.Tensor,
        ligand_coords_pred: Optional[torch.Tensor] = None,
        ligand_coords_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate loss with proper masking for batched data.
        """
        # Handle case where target_vector_field is None (validation without ground truth)
        if target_vector_field is None:
            target_vector_field = torch.zeros_like(pred_vector_field)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has correct shape
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)  # Add dimension for coordinates
            
            # Apply mask to all tensors
            pred_vector_field = pred_vector_field * mask
            target_vector_field = target_vector_field * mask
            
            # Count valid elements for proper averaging
            n_valid = mask.sum()
        else:
            n_valid = pred_vector_field.numel()
        
        # Vector field prediction loss (Smooth L1 for stability)
        if n_valid > 0:
            vector_loss = self.smooth_l1(pred_vector_field, target_vector_field)
            
            # Trajectory consistency loss
            if self.lambda_traj > 0 and ligand_coords_pred is not None and ligand_coords_target is not None:
                traj_loss = self.mse(ligand_coords_pred, ligand_coords_target) * self.lambda_traj
                vector_loss = vector_loss + traj_loss
        else:
            vector_loss = torch.tensor(0.0, device=pred_vector_field.device)
        
        # Calculate RMSD for monitoring only (not part of loss)
        rmsd_value = torch.tensor(0.0, device=pred_vector_field.device)
        if ligand_coords_pred is not None and ligand_coords_target is not None:
            coord_diff = ligand_coords_pred - ligand_coords_target
            if mask is not None:
                coord_diff = coord_diff * mask
            if n_valid > 0:
                rmsd_value = torch.sqrt(torch.mean(torch.sum(coord_diff ** 2, dim=-1)))
        
        return {
            'total': vector_loss,  # Only vector loss for training
            'vector': vector_loss,
            'rmsd': rmsd_value  # RMSD for monitoring only
        }


class EquivariantTrainer:
    """
    Trainer class for FlowFix Equivariant model with mixed precision support.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, training will be slow!")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        
        # Initialize model
        self.model = FlowFixEquivariantModel(
            protein_feat_dim=config['model']['protein_feat_dim'],
            ligand_feat_dim=config['model']['ligand_feat_dim'],
            edge_dim=config['model']['edge_dim'],
            hidden_scalars=config['model']['hidden_scalars'],
            hidden_vectors=config['model']['hidden_vectors'],
            hidden_dim=config['model']['hidden_dim'],
            out_dim=config['model']['out_dim'],
            num_layers=config['model']['num_layers'],
            max_ell=config['model']['max_ell'],
            cutoff=config['model']['cutoff'],
            time_embedding_dim=config['model']['time_embedding_dim'],
            dropout=config['model']['dropout']
        )
        
        # Initialize weights
        self.model = init_model(self.model)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize EMA if enabled
        self.use_ema = config.get('ema', {}).get('enabled', True)
        if self.use_ema:
            ema_decay = config.get('ema', {}).get('decay', 0.999)
            self.ema = EMA(self.model, decay=ema_decay, device=self.device)
            print(f"EMA enabled with decay={ema_decay}")
        else:
            self.ema = None
            
        # Initialize DiffDock-style validator
        self.diffdock_validator = DiffDockValidator(
            model=self.model,
            ema_model=self.ema.shadow_model if self.ema else None,
            device=self.device,
            num_inference_steps=config.get('inference', {}).get('num_steps', 20),
            early_stop_step=config.get('inference', {}).get('early_stop_step', 18)
        )
        
        # Validation settings
        self.diffdock_val_every = config.get('validation', {}).get('diffdock_every', 5)
        self.diffdock_num_samples = config.get('validation', {}).get('num_samples', 500)
        self.best_success_rate = 0.0
        
        # Initialize datasets
        self.train_dataset = PoseFlowDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            max_samples=config['data'].get('max_samples'),
            perturbation_config=config['data']['perturbation'],
            cache_data=config['data']['cache_data'],
            seed=config['data'].get('seed', 42),
            perturbation_mode=config['data'].get('train_perturbation_mode', 'random'),
            resample_every_epoch=config['data'].get('resample_every_epoch', True),
            val_fixed_t=None
        )
        
        self.val_dataset = PoseFlowDataset(
            data_dir=config['data']['data_dir'],
            split='val',
            max_samples=config['data'].get('max_samples_val'),
            cache_data=config['data']['cache_data'],
            seed=config['data'].get('seed', 42),
            perturbation_mode=config['data'].get('val_perturbation_mode', 'fixed'),
            resample_every_epoch=False,
            val_fixed_t=config['data'].get('val_fixed_t', 0.5)
        )
        
        # Initialize data loaders with batched collate function
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Disable multiprocessing for debugging
            collate_fn=collate_fn_batched,  # Use batched collate
            pin_memory=False,  # Disable pin_memory to avoid CUDA errors
            drop_last=False  # Keep all samples
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Disable multiprocessing for debugging
            collate_fn=collate_fn_batched,  # Use batched collate
            pin_memory=False  # Disable pin_memory to avoid CUDA errors
        )
        
        # Initialize optimizer
        # Ensure learning rate is float
        lr = float(config['training']['learning_rate'])
        weight_decay = float(config['training']['weight_decay'])
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Initialize scheduler
        total_steps = len(self.train_loader) * config['training']['num_epochs']
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,  # Use the same lr variable we converted to float
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos'
        )
        
        # Initialize loss function with trajectory regularization
        self.criterion = FlowMatchingLoss(
            smooth_l1_beta=config['loss'].get('smooth_l1_beta', 1.0),
            lambda_traj=config['loss'].get('lambda_traj', 0.1)
        )
        
        # Mixed precision disabled - not using GradScaler
        
        # Initialize wandb if enabled
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project'],
                name=config['logging']['run_name'],
                config=config
            )
        
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with mixed precision and gradient accumulation.
        """
        self.model.train()
        total_loss = 0
        total_vector_loss = 0
        total_rmsd_loss = 0
        num_batches = 0
        
        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Skip if batch is invalid
            if batch is None:
                continue
            
            # Forward pass without autocast
            try:
                # Forward pass with batch indices
                outputs = self.model(
                    protein_coords=batch['protein_coords'],
                    protein_features=batch['protein_features'],
                    ligand_coords=batch['ligand_coords_t'],
                    ligand_features=batch['ligand_features'],
                    t=batch['t'],
                    protein_batch=batch.get('protein_batch'),
                    ligand_batch=batch.get('ligand_batch')
                )
                
                # Calculate loss
                # For RMSD: apply predicted vector field to current coordinates
                t_expanded = batch['t'].view(-1, 1, 1) if batch['t'].dim() == 1 else batch['t'].unsqueeze(-1).unsqueeze(-1)
                ligand_coords_pred = batch['ligand_coords_t'] + outputs['vector_field'] * (1 - t_expanded)
                
                losses = self.criterion(
                    pred_vector_field=outputs['vector_field'],
                    target_vector_field=batch['vector_field'],  # Use vector_field from dataset
                    ligand_coords_pred=ligand_coords_pred,
                    ligand_coords_target=batch.get('ligand_coords_0'),  # Target (crystal) coordinates
                    mask=batch.get('ligand_coords_mask')
                )
                
                # Scale loss for gradient accumulation
                loss = losses['total'] / grad_accum_steps
                
            except RuntimeError as e:
                print(f"Error in forward pass: {e}")
                continue
            
            # Backward pass without gradient scaling
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if self.ema is not None:
                    self.ema.update()
                
                # Scheduler step
                self.scheduler.step()
                self.global_step += 1
            
            # Update metrics
            total_loss += losses['total'].item()
            total_vector_loss += losses['vector'].item()
            total_rmsd_loss += losses['rmsd'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'vector': f"{losses['vector'].item():.4f}",
                'rmsd': f"{losses['rmsd'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.config['logging']['use_wandb'] and self.global_step % 10 == 0:
                wandb.log({
                    'train/loss': losses['total'].item(),
                    'train/vector_loss': losses['vector'].item(),
                    'train/rmsd_loss': losses['rmsd'].item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step
                })
        
        # Calculate average metrics
        if num_batches > 0:
            metrics = {
                'train/loss': total_loss / num_batches,
                'train/vector_loss': total_vector_loss / num_batches,
                'train/rmsd_loss': total_rmsd_loss / num_batches
            }
        else:
            metrics = {
                'train/loss': 0.0,
                'train/vector_loss': 0.0,
                'train/rmsd_loss': 0.0
            }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Improved validation for flow matching-based pose refinement.
        Tests model at multiple time points and evaluates trajectory integration.
        """
        self.model.eval()
        
        # Metrics for different time regions
        metrics_by_time = {
            'early': [],  # t ∈ [0.1, 0.3]
            'mid': [],    # t ∈ [0.4, 0.6]  
            'late': []    # t ∈ [0.7, 0.9]
        }
        total_rmsd_loss = 0
        total_trajectory_rmsd = 0  # RMSD after full integration
        total_flow_consistency = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Skip if batch is invalid
                if batch is None:
                    continue
                
                try:
                    # Test at multiple time points for comprehensive evaluation
                    t_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # Sample different t
                    
                    batch_losses = []
                    for t_val in t_values:
                        t_tensor = torch.tensor(t_val, device=self.device).expand(batch['batch_size'])
                        
                        # Interpolate coordinates for this t
                        ligand_coords_t = (1 - t_val) * batch['ligand_coords_t'] + t_val * batch['ligand_coords_0']
                        
                        # Forward pass
                        outputs = self.model(
                            protein_coords=batch['protein_coords'],
                            protein_features=batch['protein_features'],
                            ligand_coords=ligand_coords_t,
                            ligand_features=batch['ligand_features'],
                            t=t_tensor,
                            protein_batch=batch.get('protein_batch'),
                            ligand_batch=batch.get('ligand_batch')
                        )
                        
                        # Calculate loss
                        losses = self.criterion(
                            pred_vector_field=outputs['vector_field'],
                            target_vector_field=batch.get('vector_field', 
                                                         torch.zeros_like(outputs['vector_field'])),
                            ligand_coords_pred=ligand_coords_t + outputs['vector_field'] * (1 - t_val),
                            ligand_coords_target=batch.get('ligand_coords_0'),
                            mask=batch.get('ligand_coords_mask')
                        )
                        
                        # Categorize by time region
                        if t_val <= 0.3:
                            metrics_by_time['early'].append(losses['vector'].item())
                        elif t_val <= 0.6:
                            metrics_by_time['mid'].append(losses['vector'].item())
                        else:
                            metrics_by_time['late'].append(losses['vector'].item())
                        
                        batch_losses.append(losses['total'].item())
                    
                    # Test trajectory integration (simulate actual inference)
                    trajectory_rmsd = self._integrate_and_evaluate(
                        batch, num_steps=50
                    )
                    total_trajectory_rmsd += trajectory_rmsd
                    
                    # Flow consistency at t=0.5
                    if batch.get('vector_field') is not None:
                        t_mid = torch.tensor(0.5, device=self.device).expand(batch['batch_size'])
                        ligand_coords_mid = 0.5 * batch['ligand_coords_t'] + 0.5 * batch['ligand_coords_0']
                        
                        outputs_mid = self.model(
                            protein_coords=batch['protein_coords'],
                            protein_features=batch['protein_features'],
                            ligand_coords=ligand_coords_mid,
                            ligand_features=batch['ligand_features'],
                            t=t_mid,
                            protein_batch=batch.get('protein_batch'),
                            ligand_batch=batch.get('ligand_batch')
                        )
                        
                        pred_norm = torch.nn.functional.normalize(outputs_mid['vector_field'], dim=-1)
                        target_norm = torch.nn.functional.normalize(batch['vector_field'], dim=-1)
                        flow_consistency = (pred_norm * target_norm).sum(dim=-1).mean()
                        total_flow_consistency += flow_consistency.item()
                    
                    # Average loss across time points
                    total_rmsd_loss += losses['rmsd'].item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    print(f"Error in validation: {e}")
                    continue
        
        # Calculate average metrics
        if num_batches > 0:
            metrics = {
                'val/loss_early': np.mean(metrics_by_time['early']) if metrics_by_time['early'] else float('inf'),
                'val/loss_mid': np.mean(metrics_by_time['mid']) if metrics_by_time['mid'] else float('inf'),
                'val/loss_late': np.mean(metrics_by_time['late']) if metrics_by_time['late'] else float('inf'),
                'val/rmsd_loss': total_rmsd_loss / num_batches,
                'val/trajectory_rmsd': total_trajectory_rmsd / num_batches,
                'val/flow_consistency': total_flow_consistency / num_batches
            }
        else:
            metrics = {
                'val/loss_early': float('inf'),
                'val/loss_mid': float('inf'),
                'val/loss_late': float('inf'),
                'val/rmsd_loss': float('inf'),
                'val/trajectory_rmsd': float('inf'),
                'val/flow_consistency': 0.0
            }
        
        return metrics
    
    def _integrate_and_evaluate(self, batch: Dict, num_steps: int = 50) -> float:
        """
        Integrate the flow from perturbed to refined pose and evaluate RMSD.
        This simulates actual inference.
        """
        # Start from perturbed coordinates
        x_t = batch['ligand_coords_t'].clone()
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.tensor(step * dt, device=self.device).expand(batch['batch_size'])
            
            outputs = self.model(
                protein_coords=batch['protein_coords'],
                protein_features=batch['protein_features'],
                ligand_coords=x_t,
                ligand_features=batch['ligand_features'],
                t=t,
                protein_batch=batch.get('protein_batch'),
                ligand_batch=batch.get('ligand_batch')
            )
            
            # Euler integration step
            x_t = x_t + outputs['vector_field'] * dt
        
        # Calculate RMSD with crystal structure
        diff = x_t - batch['ligand_coords_0']
        rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))
        
        return rmsd.item()
    
    def _prepare_batch(self, batch: Dict) -> Optional[Dict]:
        """
        Prepare batch for model input. Handles batched samples.
        """
        try:
            prepared = {}
            
            # Handle batched protein data
            if 'protein_coords' in batch and torch.is_tensor(batch['protein_coords']):
                prepared['protein_coords'] = batch['protein_coords'].to(self.device)
                prepared['protein_features'] = batch['protein_features'].to(self.device)
                prepared['protein_batch'] = batch.get('protein_batch', torch.zeros(
                    batch['protein_features'].shape[0], dtype=torch.long)).to(self.device)
            else:
                print(f"Warning: No protein data in batch")
                return None
            
            # Handle batched ligand data
            if 'ligand_coords_t' in batch and torch.is_tensor(batch['ligand_coords_t']):
                prepared['ligand_coords_t'] = batch['ligand_coords_t'].to(self.device)
                prepared['ligand_features'] = batch['ligand_features'].to(self.device)
                prepared['ligand_batch'] = batch.get('ligand_batch', torch.zeros(
                    batch['ligand_features'].shape[0], dtype=torch.long)).to(self.device)
            else:
                print(f"Warning: ligand_coords_t is not a tensor")
                return None
                
            # Handle batched vector field
            if 'vector_field' in batch and torch.is_tensor(batch['vector_field']):
                prepared['vector_field'] = batch['vector_field'].to(self.device)
            else:
                prepared['vector_field'] = None
            
            # Handle batched target coordinates for RMSD calculation
            if 'ligand_coords_0' in batch and torch.is_tensor(batch['ligand_coords_0']):
                prepared['ligand_coords_0'] = batch['ligand_coords_0'].to(self.device)
            else:
                prepared['ligand_coords_0'] = None
            
            # Handle time - should be (batch_size,) tensor
            if torch.is_tensor(batch['t']):
                prepared['t'] = batch['t'].to(self.device)
            else:
                prepared['t'] = torch.tensor(batch['t'], device=self.device)
            
            # Get batch size
            prepared['batch_size'] = batch.get('batch_size', 1)
            
            # Handle mask if available
            if 'ligand_coords_mask' in batch and torch.is_tensor(batch['ligand_coords_mask']):
                if batch['ligand_coords_mask'].dim() == 2:  # (batch, N)
                    prepared['ligand_coords_mask'] = batch['ligand_coords_mask'][0].to(self.device)
                else:
                    prepared['ligand_coords_mask'] = batch['ligand_coords_mask'].to(self.device)
            
            return prepared
            
        except Exception as e:
            print(f"Error preparing batch: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        val_loss = metrics.get('val/loss', float('inf'))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            # Ensure per-epoch reseeding for training dataset if enabled
            if hasattr(self.train_dataset, 'set_epoch'):
                self.train_dataset.set_epoch(epoch)
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config['training'].get('val_every', 1) == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            if self.config['logging']['use_wandb']:
                wandb.log(metrics, step=epoch)
            
            # Print metrics with improved formatting for new validation
            print(f"\nEpoch {epoch}/{self.config['training']['num_epochs']}")
            if 'val/loss_early' in metrics:
                print(f"  Validation Loss (early/mid/late): {metrics.get('val/loss_early', float('inf')):.4f} / "
                      f"{metrics.get('val/loss_mid', float('inf')):.4f} / {metrics.get('val/loss_late', float('inf')):.4f}")
                print(f"  Trajectory RMSD: {metrics.get('val/trajectory_rmsd', float('inf')):.4f}")
            for key, value in metrics.items():
                if 'early' not in key and 'mid' not in key and 'late' not in key:
                    print(f"  {key}: {value:.4f}")
            
            # Save checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch, metrics)
        
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint(self.config['training']['num_epochs'], metrics)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train FlowFix Equivariant model')
    parser.add_argument('--config', type=str, default='configs/train_equivariant_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = EquivariantTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()