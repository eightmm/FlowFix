#!/usr/bin/env python
"""
Simplified Training script for FlowFix: Flow Matching for Protein-Ligand Pose Refinement
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml
import argparse
from datetime import datetime
import wandb

from src.utils.data_utils import create_datasets, create_dataloaders
from src.utils.model_builder import build_model
from src.utils.training_utils import build_optimizer_and_scheduler, sample_timesteps_logistic_normal
from src.utils.early_stop import EarlyStopping
from src.utils.utils import set_random_seed
from src.utils.visualization import MolecularVisualizer
from src.utils.experiment import ExperimentManager
from src.utils.wandb_logger import (
    WandBLogger,
    extract_module_gradient_norms,
    extract_parameter_stats
)


class FlowFixTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Set random seed
        set_random_seed(config.get('seed', 42))

        # Setup experiment manager (unified directory structure)
        self.setup_experiment_manager()

        # Setup directories
        self.setup_directories()

        # Load data
        self.setup_data()

        # Setup model
        self.setup_model()

        # Setup optimizer and scheduler
        self.setup_optimizer()

        # Setup early stopping
        self.setup_early_stopping()

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_rmsd = float('inf')

        # Initialize parameter history for tracking changes
        self.param_history = {}

        # Setup WandB logging
        self.setup_wandb()

        # Setup centralized WandB logger
        self.wandb_logger = WandBLogger(enabled=self.wandb_enabled)

    def setup_experiment_manager(self):
        """Setup unified experiment manager."""
        # Get run name from config or auto-generate
        wandb_config = self.config.get('wandb', {})
        run_name = wandb_config.get('name')

        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"flowfix_{timestamp}"

        # Get base directory from config (default: "save")
        base_dir = self.config.get('experiment', {}).get('base_dir', 'save')

        # Create experiment manager
        self.exp_manager = ExperimentManager(
            base_dir=base_dir,
            run_name=run_name,
            config=self.config
        )

        # Log experiment info
        self.exp_manager.logger.info(f"✓ Experiment: {run_name}")
        self.exp_manager.logger.info(f"  Device: {self.device}")
        self.exp_manager.logger.info(f"  Seed: {self.config.get('seed', 42)}")

    def setup_directories(self):
        """Setup directories using experiment manager."""
        # Use experiment manager paths
        self.checkpoint_dir = self.exp_manager.checkpoints_dir
        self.weights_dir = self.checkpoint_dir  # Checkpoints go directly in checkpoints/

        # Animation directory for validation visualizations
        if self.config.get('visualization', {}).get('enabled', False):
            self.animation_dir = self.exp_manager.visualizations_dir
            self.visualizer = MolecularVisualizer(str(self.animation_dir))

    def setup_data(self):
        """Setup datasets and dataloaders."""
        data_config = self.config['data']
        training_config = self.config['training']

        # Create datasets
        self.train_dataset, self.val_dataset, dataset_type = create_datasets(
            data_config,
            seed=self.config.get('seed', 42)
        )

        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            training_config,
            data_config
        )

        self.exp_manager.logger.info(f"Training on {self.device} | {dataset_type} Dataset | Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} samples")

    def setup_model(self):
        """Initialize ProteinLigandFlowMatching model."""
        model_config = self.config['model']

        # Build model
        self.model = build_model(model_config, self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.exp_manager.logger.info(f"✓ Model initialized with {total_params:,} trainable parameters")

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        training_config = self.config['training']

        # Build optimizer and scheduler
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            self.model,
            training_config
        )

    def setup_early_stopping(self):
        """Setup early stopping."""
        val_config = self.config['training'].get('validation', {})

        self.early_stopper = EarlyStopping(
            mode='min',
            patience=val_config.get('early_stopping_patience', 50),
            min_delta=val_config.get('min_delta', 0.0),
            restore_best_weights=True,
            save_dir=str(self.weights_dir)
        )

    def setup_wandb(self):
        """Setup WandB logging with experiment manager directory."""
        wandb_config = self.config.get('wandb', {})
        if not wandb_config.get('enabled', False):
            self.wandb_enabled = False
            return

        self.wandb_enabled = True

        # Use experiment manager's run name
        run_name = self.exp_manager.run_name

        # Initialize WandB with custom directory
        wandb.init(
            project=wandb_config.get('project', 'protein-ligand-flowfix'),
            entity=wandb_config.get('entity'),
            name=run_name,
            tags=wandb_config.get('tags', []),
            dir=self.exp_manager.get_wandb_dir(),  # Use experiment manager's wandb dir
            config={
                'model': self.config['model'],
                'training': self.config['training'],
                'data': self.config['data'],
                'device': str(self.device),
                'total_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        )

        # Log model architecture
        if wandb_config.get('log_model_weights', True):
            wandb.watch(self.model, log='all', log_freq=100)

        self.exp_manager.logger.info(f"✓ WandB initialized: {run_name}")
        self.exp_manager.logger.info(f"  Project: {wandb_config.get('project', 'protein-ligand-flowfix')}")
        self.exp_manager.logger.info(f"  WandB dir: {self.exp_manager.get_wandb_dir()}")

    def train_step(self, batch):
        """Single training step with flow matching.

        Samples multiple timesteps per PDB system for more efficient training.
        For each PDB system in the batch, we sample num_timesteps_per_sample different
        timesteps and compute the loss across all of them.
        """
        # Get number of timesteps to sample per system
        num_timesteps = self.config['training'].get('num_timesteps_per_sample', 4)

        # Move to device
        ligand_batch = batch['ligand_graph'].to(self.device)
        protein_batch = batch['protein_graph'].to(self.device)
        ligand_coords_x0 = batch['ligand_coords_x0'].to(self.device)  # Docked (t=0)
        ligand_coords_x1 = batch['ligand_coords_x1'].to(self.device)  # Crystal (t=1)

        original_batch_size = len(batch['pdb_ids'])

        # Sample multiple timesteps for each PDB system
        # Shape: [original_batch_size, num_timesteps]
        t_samples = []
        for _ in range(original_batch_size):
            t_for_system = sample_timesteps_logistic_normal(
                num_timesteps,
                device=self.device,
                mu=0.6,
                sigma=1.4,
                mix_ratio=0.7
            )
            t_samples.append(t_for_system)
        t_all = torch.cat(t_samples, dim=0)  # [batch_size * num_timesteps]

        # Replicate each PDB system num_timesteps times
        # We need to replicate ALL data (nodes, edges, features)

        # Helper function to replicate a batch
        def replicate_batch(batch_data, num_times):
            """Replicate PyG batch data num_times"""
            from torch_geometric.data import Batch

            # Get individual graphs
            data_list = batch_data.to_data_list()

            # Replicate each graph num_times
            replicated_list = []
            for data in data_list:
                for _ in range(num_times):
                    replicated_list.append(data.clone())

            # Create new batch
            return Batch.from_data_list(replicated_list)

        # Replicate protein and ligand batches
        protein_batch_expanded = replicate_batch(protein_batch, num_timesteps)
        ligand_batch_expanded = replicate_batch(ligand_batch, num_timesteps)

        # Replicate coordinates (system-wise replication)
        replicated_ligand_coords_x0 = []
        replicated_ligand_coords_x1 = []
        for i in range(original_batch_size):
            mask = (ligand_batch.batch == i)
            x0_system = ligand_coords_x0[mask]
            x1_system = ligand_coords_x1[mask]
            for _ in range(num_timesteps):
                replicated_ligand_coords_x0.append(x0_system)
                replicated_ligand_coords_x1.append(x1_system)

        replicated_ligand_coords_x0 = torch.cat(replicated_ligand_coords_x0, dim=0)
        replicated_ligand_coords_x1 = torch.cat(replicated_ligand_coords_x1, dim=0)

        # Linear interpolation: x_t = (1-t)·x0 + t·x1 for each replicated system
        t_expanded = t_all[ligand_batch_expanded.batch].unsqueeze(-1)  # [N_ligand_total, 1]
        x_t = (1 - t_expanded) * replicated_ligand_coords_x0 + t_expanded * replicated_ligand_coords_x1

        # Update ligand positions
        ligand_batch_t = ligand_batch_expanded.clone()
        ligand_batch_t.pos = x_t.clone()

        # Predict velocity field for all timesteps
        predicted_velocity = self.model(protein_batch_expanded, ligand_batch_t, t_all)

        # True velocity: v = dx/dt = x1 - x0 (constant for linear path)
        true_velocity = replicated_ligand_coords_x1 - replicated_ligand_coords_x0

        # Simple uniform flow matching loss
        # Pure MSE loss on velocity field (no weighting, no coordinate loss)
        # For linear interpolation, true velocity is constant across all t
        loss = torch.nn.functional.mse_loss(predicted_velocity, true_velocity)

        # Scale loss for gradient accumulation
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        scaled_loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaled_loss.backward()

        # Only step optimizer after accumulating gradients
        if (self.global_step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Calculate RMSD for monitoring (using replicated coordinates)
        with torch.no_grad():
            rmsd = torch.sqrt(torch.mean((x_t - replicated_ligand_coords_x1) ** 2))

        # Log gradients and parameters if WandB is enabled
        if self.wandb_enabled:
            # Log gradient norms (every step)
            if self.config.get('wandb', {}).get('log_gradients', True):
                total_norm, module_norms = extract_module_gradient_norms(self.model)
                self.wandb_logger.log_gradient_norms(total_norm, module_norms, self.global_step)

            # Log parameter stats every 50 steps
            if self.config.get('wandb', {}).get('log_model_weights', True):
                if (self.global_step + 1) % 50 == 0:
                    module_stats = extract_parameter_stats(self.model)
                    self.wandb_logger.log_parameter_stats(module_stats)

        return {
            'loss': loss.item(),
            'rmsd': rmsd.item()
        }


    @torch.no_grad()
    def validate(self):
        """Validation step with ODE sampling and success rate calculation."""
        self.model.eval()

        all_losses = []
        all_rmsds = []
        all_initial_rmsds = []
        
        # For animation: track first sample
        viz_config = self.config.get('visualization', {})
        create_animation = viz_config.get('enabled', False) and viz_config.get('save_animation', True)
        animation_saved = False
        trajectory_coords = []
        trajectory_rmsds = []
        
        # Randomly select which batch and sample to visualize (changes each epoch)
        if create_animation:
            num_val_batches = len(self.val_loader)
            rng = np.random.RandomState(self.current_epoch)  # Use epoch as seed for reproducibility
            target_batch_idx = rng.randint(0, num_val_batches)
            print(f"\n🎬 Will visualize batch {target_batch_idx} (randomly selected for epoch {self.current_epoch})")

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            # Move to device
            ligand_batch = batch['ligand_graph'].to(self.device)
            protein_batch = batch['protein_graph'].to(self.device)
            ligand_coords_x0 = batch['ligand_coords_x0'].to(self.device)
            ligand_coords_x1 = batch['ligand_coords_x1'].to(self.device)

            # Calculate initial RMSD (docked pose)
            initial_rmsd = torch.sqrt(torch.mean((ligand_coords_x0 - ligand_coords_x1) ** 2, dim=-1))
            all_initial_rmsds.extend(initial_rmsd.cpu().numpy())

            # Sample from docked to crystal using Euler/RK4 integration
            batch_size = len(batch['pdb_ids'])
            num_steps = self.config['sampling'].get('num_steps', 50)
            method = self.config['sampling'].get('method', 'euler')
            dt = 1.0 / num_steps

            current_coords = ligand_coords_x0.clone()
            
            # For randomly selected batch and sample: save trajectory for animation
            save_trajectory = (create_animation and not animation_saved and batch_idx == target_batch_idx)
            trajectory_velocities = []  # Store velocities for visualization

            if save_trajectory:
                # Randomly select a sample from this batch
                num_samples_in_batch = len(batch['pdb_ids'])
                rng = np.random.RandomState(self.current_epoch + 1000)  # Different seed for sample selection
                target_sample_idx = rng.randint(0, num_samples_in_batch)

                print(f"   📍 Selected sample {target_sample_idx}/{num_samples_in_batch-1} (PDB: {batch['pdb_ids'][target_sample_idx]})")

                # Get selected sample's ligand mask
                sample_mask = (ligand_batch.batch == target_sample_idx)
                trajectory_coords.append(current_coords[sample_mask].clone())
                # Calculate initial RMSD
                initial_rmsd = torch.sqrt(torch.mean(
                    (current_coords[sample_mask] - ligand_coords_x1[sample_mask]) ** 2
                ))
                trajectory_rmsds.append(initial_rmsd.item())

            for step in range(num_steps):
                t = torch.ones(batch_size, device=self.device) * (step * dt)

                # Create batch with current coordinates
                ligand_batch_t = ligand_batch.clone()
                ligand_batch_t.pos = current_coords.clone()

                # Predict velocity
                velocity = self.model(protein_batch, ligand_batch_t, t)

                # Save velocity for visualization (before integration step)
                if save_trajectory:
                    trajectory_velocities.append(velocity[sample_mask].clone())

                if method == 'euler':
                    # Euler step
                    current_coords = current_coords + dt * velocity
                elif method == 'rk4':
                    # RK4 integration (more accurate)
                    k1 = velocity

                    ligand_batch_t.pos = (current_coords + 0.5 * dt * k1).clone()
                    k2 = self.model(protein_batch, ligand_batch_t, t + 0.5 * dt)

                    ligand_batch_t.pos = (current_coords + 0.5 * dt * k2).clone()
                    k3 = self.model(protein_batch, ligand_batch_t, t + 0.5 * dt)

                    ligand_batch_t.pos = (current_coords + dt * k3).clone()
                    k4 = self.model(protein_batch, ligand_batch_t, t + dt)

                    current_coords = current_coords + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                # Save trajectory for animation
                if save_trajectory:
                    trajectory_coords.append(current_coords[sample_mask].clone())
                    # Calculate RMSD for this step
                    step_rmsd = torch.sqrt(torch.mean(
                        (current_coords[sample_mask] - ligand_coords_x1[sample_mask]) ** 2
                    ))
                    trajectory_rmsds.append(step_rmsd.item())

            refined_coords = current_coords

            # Calculate per-sample RMSD
            per_sample_rmsd = torch.sqrt(torch.mean((refined_coords - ligand_coords_x1) ** 2, dim=-1))

            # Calculate loss
            loss = torch.mean((refined_coords - ligand_coords_x1) ** 2)

            all_losses.append(loss.item())
            all_rmsds.extend(per_sample_rmsd.cpu().numpy())
            
            # Create animation for randomly selected sample
            if save_trajectory and not animation_saved:
                try:
                    # Get selected sample data
                    sample_crystal_coords = ligand_coords_x1[sample_mask]
                    sample_edge_index = ligand_batch.edge_index[:,
                        (ligand_batch.batch[ligand_batch.edge_index[0]] == target_sample_idx)]
                    # Reindex edges to start from 0
                    sample_edge_index = sample_edge_index - sample_edge_index.min()
                    sample_pdb_id = batch['pdb_ids'][target_sample_idx]

                    # Convert to numpy for visualization
                    trajectory_np = [coords.cpu().numpy() for coords in trajectory_coords]
                    velocities_np = [vel.cpu().numpy() for vel in trajectory_velocities] if trajectory_velocities else None
                    crystal_np = sample_crystal_coords.cpu().numpy()
                    edges_tuple = (sample_edge_index[0], sample_edge_index[1])

                    # Use visualization utility (GIF only)
                    print(f"\n🎬 Creating multi-view animation for {sample_pdb_id}...")
                    gif_path = self.visualizer.create_sampling_gif(
                        trajectory=trajectory_np,
                        crystal_coords=crystal_np,
                        edges=edges_tuple,
                        epoch=self.current_epoch,
                        pdb_id=sample_pdb_id,
                        multi_view=True,  # Enable multi-view animation
                        velocities=velocities_np  # Add velocity field visualization
                    )
                    
                    # Log animation to WandB
                    if self.wandb_enabled and self.config.get('wandb', {}).get('log_animations', True):
                        self.wandb_logger.log_animation(
                            animation_path=gif_path,
                            epoch=self.current_epoch,
                            pdb_id=sample_pdb_id
                        )
                    
                    animation_saved = True
                except Exception as e:
                    print(f"⚠️  Failed to create animation: {e}")
                    import traceback
                    traceback.print_exc()

        # Calculate metrics
        avg_loss = np.mean(all_losses)
        avg_rmsd = np.mean(all_rmsds)
        avg_initial_rmsd = np.mean(all_initial_rmsds)

        # Calculate success rates at different thresholds
        rmsds = np.array(all_rmsds)
        success_2A = np.mean(rmsds < 2.0) * 100
        success_1A = np.mean(rmsds < 1.0) * 100
        success_05A = np.mean(rmsds < 0.5) * 100

        print(f"\n📊 Validation Results:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Initial RMSD: {avg_initial_rmsd:.4f} Å")
        print(f"   Final RMSD: {avg_rmsd:.4f} Å")
        print(f"   Success Rate (<2.0Å): {success_2A:.1f}%")
        print(f"   Success Rate (<1.0Å): {success_1A:.1f}%")
        print(f"   Success Rate (<0.5Å): {success_05A:.1f}%")
        
        # Log validation metrics to WandB
        if self.wandb_enabled:
            self.wandb_logger.log_validation_epoch(
                val_loss=avg_loss,
                val_rmsd=avg_rmsd,
                val_rmsd_initial=avg_initial_rmsd,
                val_rmsd_final=avg_rmsd,
                success_rate_2a=success_2A,
                success_rate_1a=success_1A,
                success_rate_05a=success_05A,
                epoch=self.current_epoch
            )

        # Early stopping based on success rate (use <2Å as primary metric)
        val_metrics = {
            'loss': avg_loss,
            'rmsd': avg_rmsd,
            'initial_rmsd': avg_initial_rmsd,
            'success_2A': success_2A,
            'success_1A': success_1A,
            'success_05A': success_05A
        }

        # Use negative success rate for early stopping (higher is better)
        early_stop = self.early_stopper.step(
            score=-success_2A,  # Negative because early stopper minimizes
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            valid_metrics=val_metrics
        )

        if success_2A > self.best_val_rmsd:  # Reuse as best success rate
            self.best_val_rmsd = success_2A

        if early_stop:
            print(f"\n🛑 Early stopping triggered!")
            print(f"   Best Success Rate (<2Å): {-self.early_stopper.get_best_score():.1f}%")

        self.model.train()
        return avg_loss, avg_rmsd, early_stop

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_rmsd': self.best_val_rmsd,
            'config': self.config
        }
        torch.save(checkpoint, self.weights_dir / filename)

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']

        val_config = self.config['training'].get('validation', {})
        validation_freq = val_config.get('frequency', 20)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            print(f"\n📊 Epoch {epoch}/{num_epochs}...")

            # Set epoch for dynamic dataset (for reproducible pose sampling)
            if hasattr(self.train_dataset, 'set_epoch'):
                self.train_dataset.set_epoch(epoch)

            # Training
            self.model.train()
            epoch_losses = []
            epoch_rmsds = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                epoch_rmsds.append(metrics['rmsd'])
                self.global_step += 1

                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'rmsd': f"{metrics['rmsd']:.3f}"
                })
                
                # Log to WandB every 10 steps
                if self.wandb_enabled and self.global_step % 10 == 0:
                    self.wandb_logger.log_training_step(
                        loss=metrics['loss'],
                        rmsd=metrics['rmsd'],
                        lr=self.optimizer.param_groups[0]['lr'],
                        epoch=epoch,
                        step=self.global_step
                    )

            # Validation (skip epoch 0)
            early_stop = False
            if epoch > 0 and epoch % validation_freq == 0:
                val_loss, val_rmsd, early_stop = self.validate()

            if early_stop:
                print("\n🛑 Training stopped early")
                if self.wandb_enabled:
                    self.wandb_logger.log_early_stopping(epoch=epoch)
                break

            # Save checkpoints
            save_freq = self.config['checkpoint'].get('save_freq', 10)
            if epoch % save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')

            if self.config['checkpoint'].get('save_latest', True):
                self.save_checkpoint('latest.pt')

            # Cleanup old checkpoints
            keep_last_n = self.config['checkpoint'].get('keep_last_n', -1)
            if keep_last_n > 0 and epoch % save_freq == 0:
                self.exp_manager.cleanup_old_checkpoints(
                    keep_last_n=keep_last_n,
                    keep_best=True
                )

            # Print summary
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_epoch_loss = np.mean(epoch_losses)
            avg_epoch_rmsd = np.mean(epoch_rmsds)
            print(f"\nEpoch {epoch} Summary:")
            print(f"  📊 Train Loss: {avg_epoch_loss:.4f}")
            print(f"  📏 Train RMSD: {avg_epoch_rmsd:.3f} Å")
            print(f"  📈 Learning Rate: {current_lr:.6f}")
            print(f"  ⏰ Early stopping: {self.early_stopper.counter}/{self.early_stopper.patience}")

            # Log epoch summary to WandB
            if self.wandb_enabled:
                self.wandb_logger.log({
                    # Training epoch averages
                    'train/epoch_loss': avg_epoch_loss,
                    'train/epoch_rmsd': avg_epoch_rmsd,
                    # System info
                    'meta/early_stopping_counter': self.early_stopper.counter,
                    'meta/early_stopping_patience': self.early_stopper.patience,
                    'meta/epoch': epoch,
                    'meta/step': self.global_step
                })

            # Scheduler step (epoch-based)
            if self.scheduler:
                self.scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override device if specified
    if args.device:
        config['device'] = args.device

    # Create trainer
    trainer = FlowFixTrainer(config)

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        print(f"Resumed from step {trainer.global_step}")

    # Train
    try:
        trainer.train()
    finally:
        # Finish WandB run
        if trainer.wandb_enabled:
            wandb.finish()


if __name__ == '__main__':
    main()