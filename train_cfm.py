#!/usr/bin/env python
"""
Training script for Conditional Flow Matching (CFM) model.
Implements optimal transport flow matching for protein-ligand binding pose refinement.
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
import math

from data.dataset import PoseFlowDataset
from data.batch_dataset import collate_fn_batched
from models.flowfix_cfm import ConditionalFlowMatching
from utils.ema import EMA
from utils.initialization import init_model


class CFMLoss(nn.Module):
    """
    Conditional Flow Matching loss with optimal transport.
    """
    
    def __init__(
        self,
        sigma_min: float = 0.001,
        use_ot: bool = True,
        lambda_reg: float = 0.01
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.use_ot = use_ot
        self.lambda_reg = lambda_reg
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)
    
    def compute_ot_plan(self, x0, x1, batch0=None, batch1=None):
        """
        Compute optimal transport plan between x0 and x1.
        For batched data, only compute within same batch.
        """
        if batch0 is None:
            # Single sample - simple matching
            return torch.arange(x0.shape[0], device=x0.device)
        
        # Batched - ensure matching within batch (already handled by dataset)
        return torch.arange(x0.shape[0], device=x0.device)
    
    def forward(
        self,
        pred_v: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CFM loss.
        
        Args:
            pred_v: Predicted velocity field
            x0: Source coordinates (perturbed)
            x1: Target coordinates (crystal)
            t: Time values
            mask: Optional mask for valid atoms
        """
        # Compute target velocity (conditional OT)
        if self.use_ot:
            pi = self.compute_ot_plan(x0, x1)
            x1_matched = x1[pi] if pi is not None else x1
        else:
            x1_matched = x1
        
        target_v = x1_matched - x0
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            pred_v = pred_v * mask
            target_v = target_v * mask
            n_valid = mask.sum()
        else:
            n_valid = pred_v.numel()
        
        # Main CFM loss
        if n_valid > 0:
            # Use smooth L1 for robustness
            cfm_loss = self.smooth_l1(pred_v, target_v)
        else:
            cfm_loss = torch.tensor(0.0, device=pred_v.device)
        
        # Regularization: encourage smooth trajectories
        reg_loss = torch.tensor(0.0, device=pred_v.device)
        if self.lambda_reg > 0 and n_valid > 0:
            # Penalize large velocities at t close to 0 or 1
            t_expanded = t.view(-1, 1, 1) if t.dim() == 1 else t.unsqueeze(-1).unsqueeze(-1)
            weight = torch.minimum(t_expanded, 1 - t_expanded) + self.sigma_min
            reg_loss = torch.mean(torch.norm(pred_v, dim=-1) / weight.squeeze(-1))
        
        # Compute trajectory RMSD for monitoring
        xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1_matched
        x_pred = xt + pred_v * (1 - t.view(-1, 1, 1))
        
        rmsd = torch.sqrt(torch.mean(torch.sum((x_pred - x1_matched) ** 2, dim=-1)))
        
        total_loss = cfm_loss + self.lambda_reg * reg_loss
        
        return {
            'total': total_loss,
            'cfm': cfm_loss,
            'reg': reg_loss,
            'rmsd': rmsd
        }


class CFMTrainer:
    """
    Trainer for Conditional Flow Matching model.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize model
        self.model = ConditionalFlowMatching(
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
            dropout=config['model']['dropout'],
            num_heads=config['model'].get('num_heads', 8),
            use_layer_norm=config['model'].get('use_layer_norm', True),
            use_gate=config['model'].get('use_gate', True)
        )
        
        # Initialize weights
        self.model = init_model(self.model)
        self.model = self.model.to(self.device)
        
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # EMA
        self.use_ema = config.get('ema', {}).get('enabled', True)
        if self.use_ema:
            self.ema = EMA(self.model, decay=config['ema'].get('decay', 0.999), device=self.device)
        else:
            self.ema = None
        
        # Datasets
        self.train_dataset = PoseFlowDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            max_samples=config['data'].get('max_samples'),
            perturbation_config=config['data']['perturbation'],
            cache_data=config['data'].get('cache_data', False),
            seed=config['data'].get('seed', 42),
            perturbation_mode='random',
            resample_every_epoch=True
        )
        
        self.val_dataset = PoseFlowDataset(
            data_dir=config['data']['data_dir'],
            split='val',
            max_samples=config['data'].get('max_samples_val'),
            cache_data=config['data'].get('cache_data', False),
            seed=config['data'].get('seed', 42),
            perturbation_mode='fixed',
            resample_every_epoch=False,
            val_fixed_t=None  # Sample different t values for validation
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_batched,
            pin_memory=False,
            drop_last=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_batched,
            pin_memory=False
        )
        
        # Optimizer
        lr = float(config['training']['learning_rate'])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(config['training']['weight_decay']),
            betas=(0.95, 0.999)
        )
        
        # Scheduler - Cosine with warmup
        total_steps = len(self.train_loader) * config['training']['num_epochs']
        warmup_steps = int(0.05 * total_steps)  # 5% warmup
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss function
        self.criterion = CFMLoss(
            sigma_min=config['loss'].get('sigma_min', 0.001),
            use_ot=config['loss'].get('use_ot', True),
            lambda_reg=config['loss'].get('lambda_reg', 0.01)
        )
        
        # Logging
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project'],
                name=config['logging']['run_name'],
                config=config
            )
        
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_cfm_loss = 0
        total_reg_loss = 0
        total_rmsd = 0
        num_batches = 0
        
        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            batch = self._prepare_batch(batch)
            if batch is None:
                continue
            
            try:
                # Forward pass
                outputs = self.model(
                    protein_coords=batch['protein_coords'],
                    protein_features=batch['protein_features'],
                    ligand_coords=batch['ligand_coords_t'],
                    ligand_features=batch['ligand_features'],
                    t=batch['t'],
                    protein_batch=batch.get('protein_batch'),
                    ligand_batch=batch.get('ligand_batch')
                )
                
                # Compute loss
                losses = self.criterion(
                    pred_v=outputs['vector_field'],
                    x0=batch['ligand_coords_t'],
                    x1=batch['ligand_coords_0'],
                    t=batch['t'],
                    mask=batch.get('ligand_coords_mask')
                )
                
                loss = losses['total'] / grad_accum_steps
                
            except RuntimeError as e:
                print(f"Error in forward pass: {e}")
                continue
            
            # Backward
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.ema is not None:
                    self.ema.update()
                
                self.scheduler.step()
                self.global_step += 1
            
            # Update metrics
            total_loss += losses['total'].item()
            total_cfm_loss += losses['cfm'].item()
            total_reg_loss += losses['reg'].item()
            total_rmsd += losses['rmsd'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'cfm': f"{losses['cfm'].item():.4f}",
                'rmsd': f"{losses['rmsd'].item():.3f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.config['logging']['use_wandb'] and self.global_step % 10 == 0:
                wandb.log({
                    'train/loss': losses['total'].item(),
                    'train/cfm_loss': losses['cfm'].item(),
                    'train/reg_loss': losses['reg'].item(),
                    'train/rmsd': losses['rmsd'].item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step
                })
        
        if num_batches > 0:
            return {
                'train/loss': total_loss / num_batches,
                'train/cfm_loss': total_cfm_loss / num_batches,
                'train/reg_loss': total_reg_loss / num_batches,
                'train/rmsd': total_rmsd / num_batches
            }
        else:
            return {'train/loss': 0.0, 'train/cfm_loss': 0.0, 'train/reg_loss': 0.0, 'train/rmsd': 0.0}
    
    def validate(self) -> Dict[str, float]:
        """Validation with trajectory integration."""
        self.model.eval()
        
        total_loss = 0
        total_rmsd_2A = 0
        total_rmsd_5A = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = self._prepare_batch(batch)
                if batch is None:
                    continue
                
                try:
                    # Test at multiple time points
                    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
                    batch_losses = []
                    
                    for t_val in t_values:
                        t_tensor = torch.tensor([t_val], device=self.device).expand(batch['batch_size'])
                        
                        # Interpolate coordinates
                        ligand_coords_t = (1 - t_val) * batch['ligand_coords_t'] + t_val * batch['ligand_coords_0']
                        
                        outputs = self.model(
                            protein_coords=batch['protein_coords'],
                            protein_features=batch['protein_features'],
                            ligand_coords=ligand_coords_t,
                            ligand_features=batch['ligand_features'],
                            t=t_tensor,
                            protein_batch=batch.get('protein_batch'),
                            ligand_batch=batch.get('ligand_batch')
                        )
                        
                        losses = self.criterion(
                            pred_v=outputs['vector_field'],
                            x0=ligand_coords_t,
                            x1=batch['ligand_coords_0'],
                            t=t_tensor,
                            mask=batch.get('ligand_coords_mask')
                        )
                        
                        batch_losses.append(losses['total'].item())
                    
                    # Integrate trajectory
                    final_coords = self._integrate_trajectory(batch, num_steps=20)
                    
                    # Calculate RMSD
                    diff = final_coords - batch['ligand_coords_0']
                    rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))
                    
                    # Success rates
                    if rmsd < 2.0:
                        total_rmsd_2A += 1
                    if rmsd < 5.0:
                        total_rmsd_5A += 1
                    
                    total_loss += np.mean(batch_losses)
                    num_batches += 1
                    
                except RuntimeError as e:
                    print(f"Error in validation: {e}")
                    continue
        
        if num_batches > 0:
            return {
                'val/loss': total_loss / num_batches,
                'val/success_2A': total_rmsd_2A / num_batches,
                'val/success_5A': total_rmsd_5A / num_batches
            }
        else:
            return {'val/loss': float('inf'), 'val/success_2A': 0.0, 'val/success_5A': 0.0}
    
    def _integrate_trajectory(self, batch: Dict, num_steps: int = 20) -> torch.Tensor:
        """Integrate flow from perturbed to refined pose."""
        x = batch['ligand_coords_t'].clone()
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.tensor([step * dt], device=self.device).expand(batch['batch_size'])
            
            outputs = self.model(
                protein_coords=batch['protein_coords'],
                protein_features=batch['protein_features'],
                ligand_coords=x,
                ligand_features=batch['ligand_features'],
                t=t,
                protein_batch=batch.get('protein_batch'),
                ligand_batch=batch.get('ligand_batch')
            )
            
            x = x + outputs['vector_field'] * dt
        
        return x
    
    def _prepare_batch(self, batch: Dict) -> Optional[Dict]:
        """Prepare batch for model input."""
        try:
            prepared = {}
            
            # Move tensors to device
            for key in ['protein_coords', 'protein_features', 'ligand_coords_t', 
                       'ligand_features', 'ligand_coords_0', 't']:
                if key in batch and torch.is_tensor(batch[key]):
                    prepared[key] = batch[key].to(self.device)
            
            # Optional tensors
            for key in ['protein_batch', 'ligand_batch', 'ligand_coords_mask', 'vector_field']:
                if key in batch and torch.is_tensor(batch[key]):
                    prepared[key] = batch[key].to(self.device)
            
            prepared['batch_size'] = batch.get('batch_size', 1)
            
            return prepared
            
        except Exception as e:
            print(f"Error preparing batch: {e}")
            return None
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint."""
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
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        val_loss = metrics.get('val/loss', float('inf'))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.ema and checkpoint.get('ema_state_dict'):
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        print(f"Resumed from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            
            if hasattr(self.train_dataset, 'set_epoch'):
                self.train_dataset.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config['training'].get('val_every', 1) == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log
            if self.config['logging']['use_wandb']:
                wandb.log(metrics, step=epoch)
            
            # Print
            print(f"\nEpoch {epoch}/{self.config['training']['num_epochs']}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Save
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch, metrics)
        
        print("Training completed!")
        self.save_checkpoint(self.config['training']['num_epochs'], metrics)


def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train CFM model')
    parser.add_argument('--config', type=str, default='configs/train_cfm.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    trainer = CFMTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == '__main__':
    main()