#!/usr/bin/env python
"""
Training script for SE(3) + Torsion Decomposition Flow Matching.

Predicts translation [3] + rotation [3] + torsion [M] instead of per-atom velocity [N, 3].
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import wandb

from src.data.dataset_torsion import FlowFixTorsionDataset, collate_torsion_batch
from src.models.flowmatching_torsion import ProteinLigandFlowMatchingTorsion
from src.utils.losses_torsion import compute_se3_torsion_loss
from src.utils.sampling_torsion import sample_trajectory_torsion
from src.utils.sampling import generate_timestep_schedule
from src.utils.training_utils import build_optimizer_and_scheduler
from src.utils.early_stop import EarlyStopping
from src.utils.utils import set_random_seed
from src.utils.experiment import ExperimentManager
from src.utils.wandb_logger import (
    WandBLogger,
    extract_module_gradient_norms,
    extract_parameter_stats,
)


def build_torsion_model(model_config, device):
    """Build ProteinLigandFlowMatchingTorsion from config."""
    model = ProteinLigandFlowMatchingTorsion(
        protein_input_scalar_dim=model_config.get('protein_input_scalar_dim', 76),
        protein_input_vector_dim=model_config.get('protein_input_vector_dim', 31),
        protein_input_edge_scalar_dim=model_config.get('protein_input_edge_scalar_dim', 39),
        protein_input_edge_vector_dim=model_config.get('protein_input_edge_vector_dim', 8),
        protein_hidden_scalar_dim=model_config.get('protein_hidden_scalar_dim', 128),
        protein_hidden_vector_dim=model_config.get('protein_hidden_vector_dim', 32),
        protein_output_scalar_dim=model_config.get('protein_output_scalar_dim', 128),
        protein_output_vector_dim=model_config.get('protein_output_vector_dim', 32),
        protein_num_layers=model_config.get('protein_num_layers', 3),
        ligand_input_scalar_dim=model_config.get('ligand_input_scalar_dim', 121),
        ligand_input_edge_scalar_dim=model_config.get('ligand_input_edge_scalar_dim', 44),
        ligand_hidden_scalar_dim=model_config.get('ligand_hidden_scalar_dim', 128),
        ligand_hidden_vector_dim=model_config.get('ligand_hidden_vector_dim', 16),
        ligand_output_scalar_dim=model_config.get('ligand_output_scalar_dim', 128),
        ligand_output_vector_dim=model_config.get('ligand_output_vector_dim', 16),
        ligand_num_layers=model_config.get('ligand_num_layers', 3),
        interaction_num_heads=model_config.get('interaction_num_heads', 8),
        interaction_num_layers=model_config.get('interaction_num_layers', 2),
        interaction_num_rbf=model_config.get('interaction_num_rbf', 32),
        interaction_pair_dim=model_config.get('interaction_pair_dim', 64),
        velocity_hidden_scalar_dim=model_config.get('velocity_hidden_scalar_dim', 128),
        velocity_hidden_vector_dim=model_config.get('velocity_hidden_vector_dim', 16),
        velocity_num_layers=model_config.get('velocity_num_layers', 4),
        hidden_dim=model_config.get('hidden_dim', 256),
        dropout=model_config.get('dropout', 0.1),
        use_esm_embeddings=model_config.get('use_esm_embeddings', True),
        esmc_dim=model_config.get('esmc_dim', 1152),
        esm3_dim=model_config.get('esm3_dim', 1536),
    ).to(device)
    return model


class FlowFixTorsionTrainer:
    """Trainer for SE(3) + Torsion decomposition flow matching."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        set_random_seed(config.get('seed', 42))

        self.setup_experiment()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_early_stopping()

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_success = 0.0

        self.setup_wandb()
        self.wandb_logger = WandBLogger(enabled=self.wandb_enabled)

    def setup_experiment(self):
        """Setup experiment manager."""
        wandb_config = self.config.get('wandb', {})
        run_name = wandb_config.get('name')
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"torsion_{timestamp}"

        base_dir = self.config.get('experiment', {}).get('base_dir', 'save')
        self.exp_manager = ExperimentManager(base_dir=base_dir, run_name=run_name, config=self.config)
        self.checkpoint_dir = self.exp_manager.checkpoints_dir
        self.exp_manager.logger.info(f"Experiment: {run_name} | Device: {self.device}")

    def setup_data(self):
        """Setup torsion-aware datasets."""
        data_config = self.config['data']
        training_config = self.config['training']

        self.train_dataset = FlowFixTorsionDataset(
            data_dir=data_config.get('data_dir', 'train_data'),
            split_file=data_config.get('split_file'),
            split='train',
            max_samples=data_config.get('max_train_samples'),
            seed=self.config.get('seed', 42),
            loading_mode=data_config.get('loading_mode', 'lazy'),
        )
        self.val_dataset = FlowFixTorsionDataset(
            data_dir=data_config.get('data_dir', 'train_data'),
            split_file=data_config.get('split_file'),
            split='valid',
            max_samples=data_config.get('max_val_samples'),
            seed=self.config.get('seed', 42),
            loading_mode=data_config.get('loading_mode', 'lazy'),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            collate_fn=collate_torsion_batch,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=training_config.get('val_batch_size', 4),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            collate_fn=collate_torsion_batch,
        )

        self.exp_manager.logger.info(
            f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} PDBs"
        )

    def setup_model(self):
        """Initialize torsion model."""
        self.model = build_torsion_model(self.config['model'], self.device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.exp_manager.logger.info(f"Model: {total_params:,} trainable params (SE(3)+Torsion)")

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            self.model, self.config['training']
        )

    def setup_early_stopping(self):
        """Setup early stopping."""
        val_config = self.config['training'].get('validation', {})
        self.early_stopper = EarlyStopping(
            mode='min',
            patience=val_config.get('early_stopping_patience', 50),
            restore_best_weights=True,
            save_dir=str(self.checkpoint_dir),
        )

    def setup_wandb(self):
        """Setup WandB."""
        wandb_config = self.config.get('wandb', {})
        if not wandb_config.get('enabled', False):
            self.wandb_enabled = False
            return

        self.wandb_enabled = True
        wandb.init(
            project=wandb_config.get('project', 'protein-ligand-flowfix'),
            entity=wandb_config.get('entity'),
            name=self.exp_manager.run_name,
            tags=wandb_config.get('tags', []),
            dir=self.exp_manager.get_wandb_dir(),
            config={
                'model': self.config['model'],
                'training': self.config['training'],
                'output_mode': 'torsion',
            },
        )

    def train_step(self, batch):
        """Single training step for SE(3) + Torsion."""
        ligand_batch = batch['ligand_graph'].to(self.device)
        protein_batch = batch['protein_graph'].to(self.device)
        coords_x0 = batch['ligand_coords_x0'].to(self.device)
        coords_x1 = batch['ligand_coords_x1'].to(self.device)
        batch_size = len(batch['pdb_ids'])

        # Sample timestep
        t = torch.rand(batch_size, device=self.device)

        # Interpolate
        t_expanded = t[ligand_batch.batch].unsqueeze(-1)
        x_t = (1 - t_expanded) * coords_x0 + t_expanded * coords_x1

        ligand_batch_t = ligand_batch.clone()
        ligand_batch_t.pos = x_t

        # Torsion data
        torsion_data = batch.get('torsion_data')
        if torsion_data is not None:
            target = {
                'translation': torsion_data['translation'].to(self.device),
                'rotation': torsion_data['rotation'].to(self.device),
                'torsion_changes': torsion_data['torsion_changes'].to(self.device),
            }
            rotatable_edges = torsion_data['rotatable_edges'].to(self.device)
            mask_rotate = torsion_data['mask_rotate'].to(self.device)
        else:
            # Fallback: rigid body only
            from src.data.ligand_feat import compute_rigid_transform
            translations, rotations = [], []
            for b in range(batch_size):
                mol_mask = (ligand_batch.batch == b)
                trans, rot = compute_rigid_transform(coords_x0[mol_mask].cpu(), coords_x1[mol_mask].cpu())
                translations.append(trans)
                rotations.append(rot)
            target = {
                'translation': torch.stack(translations).to(self.device),
                'rotation': torch.stack(rotations).to(self.device),
                'torsion_changes': torch.zeros(0, device=self.device),
            }
            rotatable_edges = torch.zeros(0, 2, dtype=torch.long, device=self.device)
            mask_rotate = torch.zeros(0, coords_x0.shape[0], dtype=torch.bool, device=self.device)

        # Forward
        pred = self.model(protein_batch, ligand_batch_t, t, rotatable_edges=rotatable_edges)

        # Loss
        loss_config = self.config['training'].get('torsion_loss', {})
        losses = compute_se3_torsion_loss(
            pred=pred, target=target,
            coords_x0=coords_x0, coords_x1=coords_x1,
            mask_rotate=mask_rotate, rotatable_edges=rotatable_edges,
            batch_indices=ligand_batch.batch,
            w_trans=loss_config.get('w_trans', 1.0),
            w_rot=loss_config.get('w_rot', 1.0),
            w_tor=loss_config.get('w_tor', 1.0),
            w_coord=loss_config.get('w_coord', 0.5),
        )

        loss = losses['total']

        # Backward
        grad_accum = self.config['training'].get('gradient_accumulation_steps', 1)
        (loss / grad_accum).backward()

        if (self.global_step + 1) % grad_accum == 0:
            clip_val = self.config['training'].get('gradient_clip')
            if clip_val:
                nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            rmsd = torch.sqrt(torch.mean((x_t - coords_x1) ** 2))

        return {
            'loss': loss.item(),
            'rmsd': rmsd.item(),
            'loss_trans': losses['translation'].item(),
            'loss_rot': losses['rotation'].item(),
            'loss_tor': losses['torsion'].item(),
            'loss_coord': losses['coord_recon'].item(),
        }

    @torch.no_grad()
    def validate(self):
        """Validation with ODE sampling in torsion space."""
        self.model.eval()

        all_rmsds = []
        all_initial_rmsds = []

        num_steps = self.config['sampling'].get('num_steps', 20)
        schedule = self.config['sampling'].get('schedule', 'uniform')
        timesteps = generate_timestep_schedule(num_steps, schedule, self.device)

        for batch in tqdm(self.val_loader, desc="Validation"):
            ligand_batch = batch['ligand_graph'].to(self.device)
            protein_batch = batch['protein_graph'].to(self.device)
            coords_x0 = batch['ligand_coords_x0'].to(self.device)
            coords_x1 = batch['ligand_coords_x1'].to(self.device)

            # Initial RMSD
            init_rmsd = torch.sqrt(torch.mean((coords_x0 - coords_x1) ** 2, dim=-1))
            all_initial_rmsds.extend(init_rmsd.cpu().numpy())

            # Torsion data
            torsion_data = batch.get('torsion_data')
            if torsion_data is not None:
                rot_edges = torsion_data['rotatable_edges'].to(self.device)
                mask_rot = torsion_data['mask_rotate'].to(self.device)
            else:
                rot_edges = torch.zeros(0, 2, dtype=torch.long, device=self.device)
                mask_rot = torch.zeros(0, coords_x0.shape[0], dtype=torch.bool, device=self.device)

            result = sample_trajectory_torsion(
                model=self.model,
                protein_batch=protein_batch,
                ligand_batch=ligand_batch,
                x0=coords_x0,
                timesteps=timesteps,
                rotatable_edges=rot_edges,
                mask_rotate=mask_rot,
            )

            refined = result['final_coords']
            per_sample_rmsd = torch.sqrt(torch.mean((refined - coords_x1) ** 2, dim=-1))
            all_rmsds.extend(per_sample_rmsd.cpu().numpy())

        # Metrics
        rmsds = np.array(all_rmsds)
        avg_rmsd = rmsds.mean()
        avg_init_rmsd = np.mean(all_initial_rmsds)
        success_2a = (rmsds < 2.0).mean() * 100
        success_1a = (rmsds < 1.0).mean() * 100
        success_05a = (rmsds < 0.5).mean() * 100

        print(f"\n  Validation Results:")
        print(f"   Initial RMSD: {avg_init_rmsd:.4f} A")
        print(f"   Final RMSD: {avg_rmsd:.4f} A")
        print(f"   Success <2A: {success_2a:.1f}%  <1A: {success_1a:.1f}%  <0.5A: {success_05a:.1f}%")

        if self.wandb_enabled:
            self.wandb_logger.log_validation_epoch(
                val_loss=avg_rmsd, val_rmsd=avg_rmsd,
                val_rmsd_initial=avg_init_rmsd, val_rmsd_final=avg_rmsd,
                success_rate_2a=success_2a, success_rate_1a=success_1a,
                success_rate_05a=success_05a, epoch=self.current_epoch,
            )

        # Early stopping
        val_metrics = {
            'rmsd': avg_rmsd, 'success_2A': success_2a,
            'success_1A': success_1a, 'success_05A': success_05a,
        }
        early_stop = self.early_stopper.step(
            score=-success_2a, model=self.model,
            optimizer=self.optimizer, scheduler=self.scheduler,
            epoch=self.current_epoch, valid_metrics=val_metrics,
        )

        if success_2a > self.best_val_success:
            self.best_val_success = success_2a

        self.model.train()
        return avg_rmsd, early_stop

    def save_checkpoint(self, filename):
        """Save checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_success': self.best_val_success,
            'config': self.config,
        }, self.checkpoint_dir / filename)

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        val_freq = self.config['training'].get('validation', {}).get('frequency', 20)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            if hasattr(self.train_dataset, 'set_epoch'):
                self.train_dataset.set_epoch(epoch)

            self.model.train()
            epoch_losses = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                self.global_step += 1

                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'tr': f"{metrics['loss_trans']:.3f}",
                    'rot': f"{metrics['loss_rot']:.3f}",
                    'tor': f"{metrics['loss_tor']:.3f}",
                })

                if self.wandb_enabled and self.global_step % 10 == 0:
                    self.wandb_logger.log({
                        'train/loss': metrics['loss'],
                        'train/loss_trans': metrics['loss_trans'],
                        'train/loss_rot': metrics['loss_rot'],
                        'train/loss_tor': metrics['loss_tor'],
                        'train/loss_coord': metrics['loss_coord'],
                        'train/rmsd': metrics['rmsd'],
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'meta/epoch': epoch,
                        'meta/step': self.global_step,
                    })

            # Validation
            early_stop = False
            if epoch > 0 and epoch % val_freq == 0:
                _, early_stop = self.validate()

            if early_stop:
                print("\n  Training stopped early")
                break

            # Checkpoints
            save_freq = self.config['checkpoint'].get('save_freq', 10)
            if epoch % save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')
            if self.config['checkpoint'].get('save_latest', True):
                self.save_checkpoint('latest.pt')

            # Scheduler
            if self.scheduler:
                self.scheduler.step()

            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch}: loss={avg_loss:.4f} lr={lr:.6f} "
                  f"early_stop={self.early_stopper.counter}/{self.early_stopper.patience}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.device:
        config['device'] = args.device

    trainer = FlowFixTorsionTrainer(config)

    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.global_step = ckpt['global_step']
        print(f"Resumed from step {trainer.global_step}")

    try:
        trainer.train()
    finally:
        if trainer.wandb_enabled:
            wandb.finish()


if __name__ == '__main__':
    main()
