#!/usr/bin/env python
"""
Inference script for FlowFix: Flow Matching for Protein-Ligand Pose Refinement

Loads a trained model and evaluates on validation data.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import json

# Note: create_datasets not needed - we create validation dataset directly to skip train validation
from src.utils.model_builder import build_model
from src.utils.utils import set_random_seed


class FlowFixInference:
    def __init__(self, config, checkpoint_path, use_ema=True):
        self.config = config
        self.device = torch.device(config['device'])
        self.checkpoint_path = checkpoint_path
        self.use_ema = use_ema

        # Set random seed
        set_random_seed(config.get('seed', 42))

        # Setup output directory
        self.setup_output_dir()

        # Load data
        self.setup_data()

        # Setup model and load checkpoint
        self.setup_model()
        self.load_checkpoint()

    def setup_output_dir(self):
        """Setup output directory for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = self.config.get('output_dir', 'inference_results')
        self.output_dir = Path(output_base) / f"inference_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def setup_data(self):
        """Setup datasets and dataloaders (validation only for inference)."""
        data_config = self.config['data']
        training_config = self.config.get('training', {'batch_size': 16, 'val_batch_size': 16})

        # Import dataset class directly to create only validation dataset
        from torch.utils.data import DataLoader
        from src.data.dataset import FlowFixDataset, collate_flowfix_batch

        # Create ONLY validation dataset (skip train dataset creation)
        loading_mode = data_config.get('loading_mode', 'lazy')
        self.val_dataset = FlowFixDataset(
            data_dir=data_config.get('data_dir', 'train_data'),
            split='valid',
            split_file=data_config.get('split_file', None),
            max_samples=data_config.get('max_val_samples', None),
            seed=self.config.get('seed', 42),
            loading_mode=loading_mode,
            rank=0
        )

        # Create validation dataloader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=training_config.get('val_batch_size', 16),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            collate_fn=collate_flowfix_batch
        )

        print(f"Loaded {len(self.val_dataset)} validation samples")

    def setup_model(self):
        """Initialize model."""
        model_config = self.config['model']
        self.model = build_model(model_config, self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {total_params:,} parameters")

    def load_checkpoint(self):
        """Load model weights from checkpoint. Uses EMA weights if available."""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # If checkpoint contains config, use it for model building
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            ckpt_model_config = checkpoint['config']['model']
            print(f"Using model config from checkpoint")
            # Rebuild model with checkpoint's config
            self.model = build_model(ckpt_model_config, self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Try strict loading first, fall back to non-strict if architecture mismatch
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"Strict loading failed: {e}")
                print("Trying non-strict loading...")
                missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing:
                    print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
            epoch = checkpoint.get('current_epoch', 'unknown')
            step = checkpoint.get('global_step', 'unknown')
            print(f"Loaded model from epoch {epoch}, step {step}")
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            print("Loaded model (early stopping format)")
        else:
            # Assume checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")

        # Apply EMA weights if available (EMA weights are better for inference)
        if self.use_ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            ema_shadow = checkpoint['ema_state_dict']['shadow']
            for name, param in self.model.named_parameters():
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name])
            print(f"Applied EMA weights (decay={checkpoint['ema_state_dict']['decay']})")
        elif self.use_ema:
            print("Warning: --use_ema specified but no EMA weights found in checkpoint")

        self.model.eval()

    @torch.no_grad()
    def run_inference(self):
        """Run inference on validation data."""
        self.model.eval()

        all_results = []
        all_rmsds = []
        all_initial_rmsds = []

        # Get sampling parameters
        num_steps = self.config['sampling'].get('num_steps', 50)
        method = self.config['sampling'].get('method', 'euler')
        schedule = self.config['sampling'].get('schedule', 'uniform')
        use_amp = self.config.get('use_amp', False)

        print(f"\nSampling parameters:")
        print(f"  Method: {method}")
        print(f"  Steps: {num_steps}")
        print(f"  Schedule: {schedule}")
        print(f"  Mixed Precision (AMP): {use_amp}")

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Inference")):
            # Move to device
            ligand_batch = batch['ligand_graph'].to(self.device)
            protein_batch = batch['protein_graph'].to(self.device)
            ligand_coords_x0 = batch['ligand_coords_x0'].to(self.device)
            ligand_coords_x1 = batch['ligand_coords_x1'].to(self.device)

            # Calculate initial RMSD per atom (will aggregate per sample later)
            batch_size = len(batch['pdb_ids'])

            # Generate timestep schedule
            timesteps = self._get_timesteps(num_steps, schedule)

            # ODE integration
            current_coords = ligand_coords_x0.clone()

            for step in range(num_steps):
                t_current = timesteps[step]
                t_next = timesteps[step + 1]
                dt = t_next - t_current

                # Broadcast t to batch size
                t = torch.ones(batch_size, device=self.device) * t_current

                # Create batch with current coordinates
                ligand_batch_t = ligand_batch.clone()
                ligand_batch_t.pos = current_coords.clone()

                # Predict velocity with optional mixed precision
                with torch.cuda.amp.autocast(enabled=use_amp):
                    velocity = self.model(protein_batch, ligand_batch_t, t)
                velocity = velocity.float()  # Ensure float32 for ODE integration

                if method == 'euler':
                    current_coords = current_coords + dt * velocity
                elif method == 'rk4':
                    k1 = velocity

                    t_mid = t_current + 0.5 * dt
                    ligand_batch_t.pos = (current_coords + 0.5 * dt * k1).clone()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        k2 = self.model(protein_batch, ligand_batch_t,
                                       torch.ones(batch_size, device=self.device) * t_mid)
                    k2 = k2.float()

                    ligand_batch_t.pos = (current_coords + 0.5 * dt * k2).clone()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        k3 = self.model(protein_batch, ligand_batch_t,
                                       torch.ones(batch_size, device=self.device) * t_mid)
                    k3 = k3.float()

                    ligand_batch_t.pos = (current_coords + dt * k3).clone()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        k4 = self.model(protein_batch, ligand_batch_t,
                                       torch.ones(batch_size, device=self.device) * t_next)
                    k4 = k4.float()

                    current_coords = current_coords + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            refined_coords = current_coords

            # Calculate RMSD per sample
            for i in range(batch_size):
                mask = (ligand_batch.batch == i)

                x0_sample = ligand_coords_x0[mask]
                x1_sample = ligand_coords_x1[mask]
                refined_sample = refined_coords[mask]

                # Initial RMSD
                initial_rmsd = torch.sqrt(torch.mean((x0_sample - x1_sample) ** 2)).item()

                # Final RMSD
                final_rmsd = torch.sqrt(torch.mean((refined_sample - x1_sample) ** 2)).item()

                all_initial_rmsds.append(initial_rmsd)
                all_rmsds.append(final_rmsd)

                result = {
                    'pdb_id': batch['pdb_ids'][i],
                    'initial_rmsd': initial_rmsd,
                    'final_rmsd': final_rmsd,
                    'improvement': initial_rmsd - final_rmsd,
                    'num_atoms': mask.sum().item()
                }
                all_results.append(result)

        # Calculate statistics
        self.print_statistics(all_results, all_rmsds, all_initial_rmsds)

        # Save results
        self.save_results(all_results)

        return all_results

    def _get_timesteps(self, num_steps, schedule):
        """Generate timestep schedule."""
        if schedule == 'uniform':
            return torch.linspace(0, 1, num_steps + 1, device=self.device)
        elif schedule == 'quadratic':
            return 1 - (1 - torch.linspace(0, 1, num_steps + 1, device=self.device)) ** 1.5
        elif schedule == 'root':
            return torch.linspace(0, 1, num_steps + 1, device=self.device) ** (2/3)
        elif schedule == 'sigmoid':
            raw = torch.linspace(-6, 6, num_steps + 1, device=self.device)
            return torch.sigmoid(raw)
        else:
            return torch.linspace(0, 1, num_steps + 1, device=self.device)

    def print_statistics(self, results, rmsds, initial_rmsds):
        """Print inference statistics."""
        rmsds = np.array(rmsds)
        initial_rmsds = np.array(initial_rmsds)

        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)

        print(f"\nTotal samples: {len(results)}")

        print(f"\nInitial RMSD (Docked):")
        print(f"  Mean: {np.mean(initial_rmsds):.4f} A")
        print(f"  Std:  {np.std(initial_rmsds):.4f} A")
        print(f"  Min:  {np.min(initial_rmsds):.4f} A")
        print(f"  Max:  {np.max(initial_rmsds):.4f} A")

        print(f"\nFinal RMSD (Refined):")
        print(f"  Mean: {np.mean(rmsds):.4f} A")
        print(f"  Std:  {np.std(rmsds):.4f} A")
        print(f"  Min:  {np.min(rmsds):.4f} A")
        print(f"  Max:  {np.max(rmsds):.4f} A")

        print(f"\nSuccess Rates:")
        print(f"  < 2.0 A: {np.mean(rmsds < 2.0) * 100:.1f}%")
        print(f"  < 1.0 A: {np.mean(rmsds < 1.0) * 100:.1f}%")
        print(f"  < 0.5 A: {np.mean(rmsds < 0.5) * 100:.1f}%")

        print(f"\nImprovement:")
        improvements = initial_rmsds - rmsds
        print(f"  Mean improvement: {np.mean(improvements):.4f} A")
        print(f"  Improved samples: {np.sum(improvements > 0)} / {len(improvements)} ({np.mean(improvements > 0) * 100:.1f}%)")

        print("="*60)

    def save_results(self, results):
        """Save results to JSON."""
        output_file = self.output_dir / "results.json"

        # Calculate summary statistics
        rmsds = [r['final_rmsd'] for r in results]
        initial_rmsds = [r['initial_rmsd'] for r in results]

        summary = {
            'checkpoint': str(self.checkpoint_path),
            'num_samples': len(results),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'sampling': self.config.get('sampling', {}),
                'data': {k: v for k, v in self.config.get('data', {}).items() if k != 'data_dir'}
            },
            'statistics': {
                'initial_rmsd_mean': float(np.mean(initial_rmsds)),
                'initial_rmsd_std': float(np.std(initial_rmsds)),
                'final_rmsd_mean': float(np.mean(rmsds)),
                'final_rmsd_std': float(np.std(rmsds)),
                'success_rate_2A': float(np.mean(np.array(rmsds) < 2.0)),
                'success_rate_1A': float(np.mean(np.array(rmsds) < 1.0)),
                'success_rate_05A': float(np.mean(np.array(rmsds) < 0.5)),
            },
            'per_sample_results': results
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Also save a CSV for easy analysis
        csv_file = self.output_dir / "results.csv"
        with open(csv_file, 'w') as f:
            f.write("pdb_id,initial_rmsd,final_rmsd,improvement,num_atoms\n")
            for r in results:
                f.write(f"{r['pdb_id']},{r['initial_rmsd']:.4f},{r['final_rmsd']:.4f},{r['improvement']:.4f},{r['num_atoms']}\n")

        print(f"CSV saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="FlowFix Inference")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--device', type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory")
    parser.add_argument('--no_ema', action='store_true', help="Disable EMA weights (use training weights)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override device if specified
    if args.device:
        config['device'] = args.device

    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Create inference runner
    inference = FlowFixInference(config, args.checkpoint, use_ema=not args.no_ema)

    # Run inference
    results = inference.run_inference()

    print("\nInference complete!")


if __name__ == '__main__':
    main()
