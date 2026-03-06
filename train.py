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
from src.utils.losses import compute_se3_torsion_loss
from src.utils.sampling import sample_trajectory_torsion, generate_timestep_schedule
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

        # Track output mode
        self.output_mode = model_config.get('output_mode', 'cartesian')

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.exp_manager.logger.info(f"✓ Model initialized with {total_params:,} trainable parameters")
        self.exp_manager.logger.info(f"  Output mode: {self.output_mode}")

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

        Supports both 'cartesian' (per-atom velocity) and 'torsion' (SE(3)+torsion) modes.
        """
        if self.output_mode == 'torsion':
            return self._train_step_torsion(batch)
        return self._train_step_cartesian(batch)

    def _train_step_torsion(self, batch):
        """Training step for SE(3) + Torsion decomposition mode."""
        # Move to device
        ligand_batch = batch['ligand_graph'].to(self.device)
        protein_batch = batch['protein_graph'].to(self.device)
        ligand_coords_x0 = batch['ligand_coords_x0'].to(self.device)
        ligand_coords_x1 = batch['ligand_coords_x1'].to(self.device)
        batch_size = len(batch['pdb_ids'])

        # Sample timestep
        t = torch.rand(batch_size, device=self.device)

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        t_expanded = t[ligand_batch.batch].unsqueeze(-1)
        x_t = (1 - t_expanded) * ligand_coords_x0 + t_expanded * ligand_coords_x1

        # Update ligand positions to x_t
        ligand_batch_t = ligand_batch.clone()
        ligand_batch_t.pos = x_t.clone()

        # Collect torsion data from batch
        torsion_data = batch.get('torsion_data')
        if torsion_data is None:
            # Fallback: no torsion data, compute rigid body only
            from src.data.ligand_feat import compute_rigid_transform
            translations, rotations = [], []
            for b in range(batch_size):
                mol_mask = (ligand_batch.batch == b)
                x0_mol = ligand_coords_x0[mol_mask]
                x1_mol = ligand_coords_x1[mol_mask]
                trans, rot = compute_rigid_transform(x0_mol.cpu(), x1_mol.cpu())
                translations.append(trans)
                rotations.append(rot)
            target = {
                'translation': torch.stack(translations).to(self.device),
                'rotation': torch.stack(rotations).to(self.device),
                'torsion_changes': torch.zeros(0, device=self.device),
            }
            rotatable_edges = torch.zeros(0, 2, dtype=torch.long, device=self.device)
            mask_rotate = torch.zeros(0, ligand_coords_x0.shape[0], dtype=torch.bool, device=self.device)
        else:
            target = {
                'translation': torsion_data['translation'].to(self.device),
                'rotation': torsion_data['rotation'].to(self.device),
                'torsion_changes': torsion_data['torsion_changes'].to(self.device),
            }
            rotatable_edges = torsion_data['rotatable_edges'].to(self.device)
            mask_rotate = torsion_data['mask_rotate'].to(self.device)

        # Forward pass
        pred = self.model(protein_batch, ligand_batch_t, t, rotatable_edges=rotatable_edges)

        # Compute loss
        loss_config = self.config['training'].get('torsion_loss', {})
        losses = compute_se3_torsion_loss(
            pred=pred,
            target=target,
            coords_x0=ligand_coords_x0,
            coords_x1=ligand_coords_x1,
            mask_rotate=mask_rotate,
            rotatable_edges=rotatable_edges,
            batch_indices=ligand_batch.batch,
            w_trans=loss_config.get('w_trans', 1.0),
            w_rot=loss_config.get('w_rot', 1.0),
            w_tor=loss_config.get('w_tor', 1.0),
            w_coord=loss_config.get('w_coord', 0.5),
        )

        loss = losses['total']

        # Scale for gradient accumulation
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        if (self.global_step + 1) % gradient_accumulation_steps == 0:
            if self.config['training'].get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        # RMSD for monitoring
        with torch.no_grad():
            rmsd = torch.sqrt(torch.mean((x_t - ligand_coords_x1) ** 2))

        # Log gradient norms
        if self.wandb_enabled:
            if self.config.get('wandb', {}).get('log_gradients', True):
                total_norm, module_norms = extract_module_gradient_norms(self.model)
                self.wandb_logger.log_gradient_norms(total_norm, module_norms, self.global_step)

        return {
            'loss': loss.item(),
            'rmsd': rmsd.item(),
            'dg_loss': 0.0,
            'loss_trans': losses['translation'].item(),
            'loss_rot': losses['rotation'].item(),
            'loss_tor': losses['torsion'].item(),
            'loss_coord': losses['coord_recon'].item(),
        }

    def _train_step_cartesian(self, batch):
        """Original training step with per-atom Cartesian velocity prediction."""
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
        # Get sampling parameters from config
        timestep_config = self.config.get('timestep_sampling', {})
        mu = timestep_config.get('mu', 0.8)
        sigma = timestep_config.get('sigma', 1.7)
        mix_ratio = timestep_config.get('mix_ratio', 0.98)

        t_samples = []
        for _ in range(original_batch_size):
            t_for_system = sample_timesteps_logistic_normal(
                num_timesteps,
                device=self.device,
                mu=mu,
                sigma=sigma,
                mix_ratio=mix_ratio
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

        # Add distance geometry constraint loss (vectorized)
        dg_loss_value = 0.0
        if 'distance_bounds' in batch and batch['distance_bounds'] is not None:
            bounds = batch['distance_bounds']
            distance_lower_bounds = bounds['lower'].to(self.device)  # [original_batch_size, max_atoms, max_atoms]
            distance_upper_bounds = bounds['upper'].to(self.device)  # [original_batch_size, max_atoms, max_atoms]
            num_atoms = bounds['num_atoms'].to(self.device)  # [original_batch_size]
            max_atoms = distance_lower_bounds.shape[1]

            # One-step Euler integration to predict final coordinates
            # dt = 1 - t (remaining time to reach crystal structure at t=1)
            dt = (1 - t_all)[ligand_batch_expanded.batch].unsqueeze(-1)  # [N_ligand_total, 1]
            x_pred = x_t + dt * predicted_velocity  # [N_ligand_total, 3]

            # Reshape x_pred to [B*num_timesteps, max_atoms, 3] with padding (vectorized)
            batch_size_expanded = original_batch_size * num_timesteps

            # Use scatter to efficiently create padded tensor
            x_pred_padded = torch.zeros(batch_size_expanded, max_atoms, 3, device=self.device)
            batch_indices = ligand_batch_expanded.batch  # [N_total]

            # Create atom indices for each molecule
            atom_counts = torch.bincount(batch_indices, minlength=batch_size_expanded)  # [B*num_timesteps]
            atom_offsets = torch.cat([torch.tensor([0], device=self.device), atom_counts.cumsum(0)[:-1]])  # [B*num_timesteps]

            # Compute within-molecule atom indices
            atom_indices_within_mol = torch.arange(len(batch_indices), device=self.device) - atom_offsets[batch_indices]

            # Scatter x_pred into padded tensor
            x_pred_padded[batch_indices, atom_indices_within_mol] = x_pred

            # Replicate distance bounds for all timesteps: [B*num_timesteps, max_atoms, max_atoms]
            lower_bounds_expanded = distance_lower_bounds.repeat_interleave(num_timesteps, dim=0)
            upper_bounds_expanded = distance_upper_bounds.repeat_interleave(num_timesteps, dim=0)
            num_atoms_expanded = num_atoms.repeat_interleave(num_timesteps)  # [B*num_timesteps]

            # Compute pairwise distances for all molecules at once: [B*num_timesteps, max_atoms, max_atoms]
            dists = torch.cdist(x_pred_padded, x_pred_padded)

            # Compute violations
            lower_violation = torch.relu(lower_bounds_expanded - dists)
            upper_violation = torch.relu(dists - upper_bounds_expanded)

            # Create mask for valid atoms (vectorized): [B*num_timesteps, max_atoms, max_atoms]
            atom_range = torch.arange(max_atoms, device=self.device).unsqueeze(0)  # [1, max_atoms]
            valid_atom_mask = atom_range < num_atoms_expanded.unsqueeze(1)  # [B*num_timesteps, max_atoms]
            valid_mask = valid_atom_mask.unsqueeze(2) & valid_atom_mask.unsqueeze(1)  # [B*num_timesteps, max_atoms, max_atoms]

            # Time-aware weighting: t=0 → weight=0, t=1 → weight=max_weight
            # [B*num_timesteps, 1, 1] for broadcasting
            max_dg_weight = self.config['training'].get('distance_geometry_weight', 0.1)
            time_weight = t_all.unsqueeze(-1).unsqueeze(-1) * max_dg_weight  # [B*num_timesteps, 1, 1]
            time_weight = time_weight.expand(-1, max_atoms, max_atoms)  # [B*num_timesteps, max_atoms, max_atoms]

            # Apply mask and time-aware weight
            masked_lower_violation = lower_violation * valid_mask.float() * time_weight
            masked_upper_violation = upper_violation * valid_mask.float() * time_weight
            dg_loss = (masked_lower_violation.sum() + masked_upper_violation.sum())

            # Normalize by total number of samples
            dg_loss = dg_loss / batch_size_expanded
            dg_loss_value = dg_loss.item()

            # Add to total loss (already weighted by time)
            loss = loss + dg_loss

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
            'rmsd': rmsd.item(),
            'dg_loss': dg_loss_value
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
            rng = np.random.RandomState(self.current_epoch)
            target_batch_idx = rng.randint(0, num_val_batches)
            print(f"\n Will visualize batch {target_batch_idx} (randomly selected for epoch {self.current_epoch})")

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            # Move to device
            ligand_batch = batch['ligand_graph'].to(self.device)
            protein_batch = batch['protein_graph'].to(self.device)
            ligand_coords_x0 = batch['ligand_coords_x0'].to(self.device)
            ligand_coords_x1 = batch['ligand_coords_x1'].to(self.device)

            # Calculate initial RMSD (docked pose)
            initial_rmsd = torch.sqrt(torch.mean((ligand_coords_x0 - ligand_coords_x1) ** 2, dim=-1))
            all_initial_rmsds.extend(initial_rmsd.cpu().numpy())

            batch_size = len(batch['pdb_ids'])
            num_steps = self.config['sampling'].get('num_steps', 50)
            schedule = self.config['sampling'].get('schedule', 'uniform')

            timesteps = generate_timestep_schedule(num_steps, schedule, self.device)

            # For animation tracking
            save_trajectory = (create_animation and not animation_saved and batch_idx == target_batch_idx)
            trajectory_velocities = []

            if save_trajectory:
                num_samples_in_batch = len(batch['pdb_ids'])
                rng = np.random.RandomState(self.current_epoch + 1000)
                target_sample_idx = rng.randint(0, num_samples_in_batch)
                print(f"   Selected sample {target_sample_idx}/{num_samples_in_batch-1} (PDB: {batch['pdb_ids'][target_sample_idx]})")
                sample_mask = (ligand_batch.batch == target_sample_idx)
                trajectory_coords.append(ligand_coords_x0[sample_mask].clone())
                init_rmsd_val = torch.sqrt(torch.mean(
                    (ligand_coords_x0[sample_mask] - ligand_coords_x1[sample_mask]) ** 2
                ))
                trajectory_rmsds.append(init_rmsd_val.item())

            # --- SE(3) + Torsion sampling ---
            if self.output_mode == 'torsion':
                torsion_data = batch.get('torsion_data')
                if torsion_data is not None:
                    rotatable_edges = torsion_data['rotatable_edges'].to(self.device)
                    mask_rotate = torsion_data['mask_rotate'].to(self.device)
                else:
                    rotatable_edges = torch.zeros(0, 2, dtype=torch.long, device=self.device)
                    mask_rotate = torch.zeros(0, ligand_coords_x0.shape[0], dtype=torch.bool, device=self.device)

                result = sample_trajectory_torsion(
                    model=self.model,
                    protein_batch=protein_batch,
                    ligand_batch=ligand_batch,
                    x0=ligand_coords_x0,
                    timesteps=timesteps,
                    rotatable_edges=rotatable_edges,
                    mask_rotate=mask_rotate,
                    return_trajectory=save_trajectory,
                )
                refined_coords = result['final_coords']

                if save_trajectory and 'trajectory' in result:
                    for coords in result['trajectory']:
                        if sample_mask is not None:
                            trajectory_coords.append(coords[sample_mask].clone())
                            step_rmsd = torch.sqrt(torch.mean(
                                (coords[sample_mask] - ligand_coords_x1[sample_mask]) ** 2
                            ))
                            trajectory_rmsds.append(step_rmsd.item())

            # --- Cartesian sampling ---
            else:
                method = self.config['sampling'].get('method', 'euler')
                current_coords = ligand_coords_x0.clone()

                for step in range(num_steps):
                    t_current = timesteps[step]
                    t_next = timesteps[step + 1]
                    dt = t_next - t_current

                    t = torch.ones(batch_size, device=self.device) * t_current

                    ligand_batch_t = ligand_batch.clone()
                    ligand_batch_t.pos = current_coords.clone()

                    velocity = self.model(protein_batch, ligand_batch_t, t)

                    if save_trajectory:
                        trajectory_velocities.append(velocity[sample_mask].clone())

                    if method == 'euler':
                        current_coords = current_coords + dt * velocity
                    elif method == 'rk4':
                        k1 = velocity
                        t_mid = t_current + 0.5 * dt
                        ligand_batch_t.pos = (current_coords + 0.5 * dt * k1).clone()
                        k2 = self.model(protein_batch, ligand_batch_t, torch.ones(batch_size, device=self.device) * t_mid)
                        ligand_batch_t.pos = (current_coords + 0.5 * dt * k2).clone()
                        k3 = self.model(protein_batch, ligand_batch_t, torch.ones(batch_size, device=self.device) * t_mid)
                        ligand_batch_t.pos = (current_coords + dt * k3).clone()
                        k4 = self.model(protein_batch, ligand_batch_t, torch.ones(batch_size, device=self.device) * t_next)
                        current_coords = current_coords + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                    if save_trajectory:
                        trajectory_coords.append(current_coords[sample_mask].clone())
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
            epoch_dg_losses = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                epoch_rmsds.append(metrics['rmsd'])
                epoch_dg_losses.append(metrics['dg_loss'])
                self.global_step += 1

                postfix = {
                    'loss': f"{metrics['loss']:.4f}",
                    'rmsd': f"{metrics['rmsd']:.3f}",
                }
                if self.output_mode == 'torsion':
                    postfix['tr'] = f"{metrics.get('loss_trans', 0):.3f}"
                    postfix['rot'] = f"{metrics.get('loss_rot', 0):.3f}"
                    postfix['tor'] = f"{metrics.get('loss_tor', 0):.3f}"
                else:
                    postfix['dg'] = f"{metrics['dg_loss']:.4f}"
                pbar.set_postfix(postfix)

                # Log to WandB every 10 steps
                if self.wandb_enabled and self.global_step % 10 == 0:
                    log_dict = {
                        'train/step_loss': metrics['loss'],
                        'train/step_rmsd': metrics['rmsd'],
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'meta/epoch': epoch,
                        'meta/step': self.global_step
                    }
                    if self.output_mode == 'torsion':
                        log_dict.update({
                            'train/step_loss_trans': metrics.get('loss_trans', 0),
                            'train/step_loss_rot': metrics.get('loss_rot', 0),
                            'train/step_loss_tor': metrics.get('loss_tor', 0),
                            'train/step_loss_coord': metrics.get('loss_coord', 0),
                        })
                    else:
                        log_dict['train/step_dg_loss'] = metrics['dg_loss']
                    self.wandb_logger.log(log_dict)

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
            avg_epoch_dg_loss = np.mean(epoch_dg_losses)
            print(f"\nEpoch {epoch} Summary:")
            print(f"  📊 Train Loss: {avg_epoch_loss:.4f}")
            print(f"  📏 Train RMSD: {avg_epoch_rmsd:.3f} Å")
            print(f"  🔗 Distance Geometry Loss: {avg_epoch_dg_loss:.4f}")
            print(f"  📈 Learning Rate: {current_lr:.6f}")
            print(f"  ⏰ Early stopping: {self.early_stopper.counter}/{self.early_stopper.patience}")

            # Log epoch summary to WandB
            if self.wandb_enabled:
                self.wandb_logger.log({
                    # Training epoch averages
                    'train/epoch_loss': avg_epoch_loss,
                    'train/epoch_rmsd': avg_epoch_rmsd,
                    'train/epoch_dg_loss': avg_epoch_dg_loss,
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