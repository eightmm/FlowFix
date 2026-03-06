#!/usr/bin/env python
"""Visualize ODE trajectory for selected samples - Best vs Latest epoch comparison."""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '/home/sim/project/flowfix')

from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from src.utils.model_builder import build_model
from src.utils.visualization import MolecularVisualizer
from torch.utils.data import DataLoader

# Vivid color palette (more visible than pastel)
COLORS = {
    'initial': '#E63946',      # Vivid red
    'final': '#2A9D8F',        # Teal green
    'target': '#F4A261',       # Orange
    'trajectory': '#264653',   # Dark blue-gray
    'success_2A': '#1D3557',   # Navy blue
    'success_1A': '#457B9D',   # Steel blue
    'improvement': '#E9C46A',  # Gold
    'threshold_2A': '#E63946', # Red
    'threshold_1A': '#F4A261', # Orange
}


def run_inference_with_trajectory(model, batch, num_steps=50, device='cuda'):
    """Run inference and record trajectory at each step."""
    model.eval()

    # Create time steps (uniform)
    t_steps = torch.linspace(0, 1, num_steps + 1, device=device)

    # Move batch to device
    ligand_batch = batch['ligand_graph'].to(device)
    protein_batch = batch['protein_graph'].to(device)
    ligand_coords_x0 = batch['ligand_coords_x0'].to(device)
    ligand_coords_x1 = batch['ligand_coords_x1'].to(device)

    batch_size = len(batch['pdb_ids'])

    # Store trajectories for each sample
    trajectories = []
    for i in range(batch_size):
        mask = (ligand_batch.batch == i)
        x0_sample = ligand_coords_x0[mask]
        x1_sample = ligand_coords_x1[mask]
        init_rmsd = torch.sqrt(torch.mean((x0_sample - x1_sample) ** 2)).item()

        # Get edges for this sample
        edge_index = ligand_batch.edge_index
        node_offset = (ligand_batch.batch < i).sum().item()
        node_count = mask.sum().item()

        # Filter edges belonging to this sample
        edge_mask = (edge_index[0] >= node_offset) & (edge_index[0] < node_offset + node_count)
        sample_edges = edge_index[:, edge_mask] - node_offset

        trajectories.append({
            'pdb_id': batch['pdb_ids'][i],
            'coords': [x0_sample.cpu().numpy()],
            'times': [0.0],
            'rmsds': [init_rmsd],
            'initial_rmsd': init_rmsd,
            'target_coords': x1_sample.cpu().numpy(),
            'edges': (sample_edges[0].cpu(), sample_edges[1].cpu())
        })

    # ODE integration
    current_coords = ligand_coords_x0.clone()

    with torch.no_grad():
        for step_idx in tqdm(range(num_steps), desc='ODE steps', leave=False):
            t_current = t_steps[step_idx]
            t_next = t_steps[step_idx + 1]
            dt = t_next - t_current

            t = torch.ones(batch_size, device=device) * t_current

            ligand_batch_t = ligand_batch.clone()
            ligand_batch_t.pos = current_coords.clone()

            velocity = model(protein_batch, ligand_batch_t, t)
            current_coords = current_coords + velocity * dt

            for i in range(batch_size):
                mask = (ligand_batch.batch == i)
                trajectories[i]['coords'].append(current_coords[mask].cpu().numpy())
                trajectories[i]['times'].append(t_next.item())
                x1_sample = ligand_coords_x1[mask]
                rmsd = torch.sqrt(torch.mean((current_coords[mask] - x1_sample) ** 2)).item()
                trajectories[i]['rmsds'].append(rmsd)

        for i in range(batch_size):
            trajectories[i]['final_rmsd'] = trajectories[i]['rmsds'][-1]
            trajectories[i]['rmsd_change'] = trajectories[i]['initial_rmsd'] - trajectories[i]['final_rmsd']

    return trajectories


def visualize_trajectory_png(traj, title, output_path):
    """Visualize a single trajectory as static PNG with vivid colors."""
    fig = plt.figure(figsize=(15, 5))

    times = traj['times']
    rmsds = traj['rmsds']

    # Plot 1: RMSD over time
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(times, rmsds, color=COLORS['trajectory'], linewidth=2.5, marker='o', markersize=4)
    ax1.axhline(y=2.0, color=COLORS['threshold_2A'], linestyle='--', linewidth=2, alpha=0.8, label='2Å threshold')
    ax1.axhline(y=1.0, color=COLORS['threshold_1A'], linestyle='--', linewidth=2, alpha=0.8, label='1Å threshold')
    ax1.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
    ax1.set_title('RMSD vs Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Plot 2: 3D trajectory of center of mass
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    coords = traj['coords']
    coms = np.array([c.mean(axis=0) for c in coords])

    # Color gradient from initial to final
    colors = plt.cm.plasma(np.linspace(0, 1, len(coms)))
    for i in range(len(coms) - 1):
        ax2.plot3D(coms[i:i+2, 0], coms[i:i+2, 1], coms[i:i+2, 2],
                   color=colors[i], linewidth=2.5)

    ax2.scatter(*coms[0], c=COLORS['initial'], s=150, marker='o', label='Start (t=0)', edgecolors='black', linewidth=1)
    ax2.scatter(*coms[-1], c=COLORS['final'], s=150, marker='s', label='End (t=1)', edgecolors='black', linewidth=1)
    ax2.scatter(*traj['target_coords'].mean(axis=0), c=COLORS['target'], s=150, marker='*', label='Target', edgecolors='black', linewidth=1)

    ax2.set_xlabel('X (Å)', fontweight='bold')
    ax2.set_ylabel('Y (Å)', fontweight='bold')
    ax2.set_zlabel('Z (Å)', fontweight='bold')
    ax2.set_title('Center of Mass Trajectory', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)

    # Plot 3: RMSD improvement bar
    ax3 = fig.add_subplot(1, 3, 3)
    bars = ax3.bar(['Initial', 'Final'], [traj['initial_rmsd'], traj['final_rmsd']],
                   color=[COLORS['initial'], COLORS['final']], edgecolor='black', linewidth=1.5)
    ax3.axhline(y=2.0, color=COLORS['threshold_2A'], linestyle='--', linewidth=2, alpha=0.8)
    ax3.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
    ax3.set_title(f'RMSD Change: {traj["rmsd_change"]:.2f}Å improvement', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, [traj['initial_rmsd'], traj['final_rmsd']]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}Å', ha='center', fontsize=12, fontweight='bold')

    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {output_path}")


def visualize_comparison_summary(all_results, output_path):
    """Create summary comparison plot for best vs latest."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (epoch_name, results) in enumerate(all_results.items()):
        rmsds_init = [t['initial_rmsd'] for t in results]
        rmsds_final = [t['final_rmsd'] for t in results]
        improvements = [t['rmsd_change'] for t in results]

        # Scatter plot: Initial vs Final RMSD
        ax = axes[0, idx]
        scatter = ax.scatter(rmsds_init, rmsds_final, c=improvements, cmap='RdYlGn',
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.plot([0, max(rmsds_init)], [0, max(rmsds_init)], 'k--', alpha=0.5, label='No change')
        ax.axhline(y=2.0, color=COLORS['threshold_2A'], linestyle='--', alpha=0.7, label='2Å')
        ax.axhline(y=1.0, color=COLORS['threshold_1A'], linestyle='--', alpha=0.7, label='1Å')
        ax.set_xlabel('Initial RMSD (Å)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Final RMSD (Å)', fontsize=11, fontweight='bold')
        ax.set_title(f'{epoch_name}: Initial vs Final RMSD', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Improvement (Å)')

        # Histogram: RMSD distribution
        ax2 = axes[1, idx]
        ax2.hist(rmsds_init, bins=30, alpha=0.7, label='Initial', color=COLORS['initial'], edgecolor='black')
        ax2.hist(rmsds_final, bins=30, alpha=0.7, label='Final', color=COLORS['final'], edgecolor='black')
        ax2.axvline(x=2.0, color=COLORS['threshold_2A'], linestyle='--', linewidth=2, label='2Å')
        ax2.axvline(x=1.0, color=COLORS['threshold_1A'], linestyle='--', linewidth=2, label='1Å')
        ax2.set_xlabel('RMSD (Å)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax2.set_title(f'{epoch_name}: RMSD Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Add statistics
        success_2A = sum(1 for r in rmsds_final if r < 2.0) / len(rmsds_final) * 100
        success_1A = sum(1 for r in rmsds_final if r < 1.0) / len(rmsds_final) * 100
        ax2.text(0.98, 0.98, f'<2Å: {success_2A:.1f}%\n<1Å: {success_1A:.1f}%',
                transform=ax2.transAxes, ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary: {output_path}")


def create_trajectory_gif(traj, output_dir, name_prefix, epoch=660):
    """Create GIF animation using MolecularVisualizer."""
    visualizer = MolecularVisualizer(output_dir)

    trajectory = traj['coords']
    crystal_coords = traj['target_coords']
    edges = traj['edges']
    pdb_id = traj['pdb_id']

    gif_path = visualizer.create_sampling_gif(
        trajectory=trajectory,
        crystal_coords=crystal_coords,
        edges=edges,
        epoch=epoch,
        pdb_id=f"{name_prefix}_{pdb_id}",
        multi_view=True
    )

    if gif_path:
        print(f"Saved GIF: {gif_path}")
    return gif_path


def run_inference_for_checkpoint(checkpoint_path, val_loader, config, device, num_steps=50):
    """Run inference for a single checkpoint and return all trajectories."""
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print('='*60)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = checkpoint['config']['model']
        print("Using model config from checkpoint")
    else:
        model_config = config['model']

    model = build_model(model_config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded model from epoch {epoch}")

    all_trajectories = []
    sample_idx = 0

    for batch in tqdm(val_loader, desc=f'Inference (epoch {epoch})'):
        trajectories = run_inference_with_trajectory(model, batch, num_steps, device)
        for traj in trajectories:
            traj['sample_idx'] = sample_idx
            all_trajectories.append(traj)
            sample_idx += 1

    print(f"Processed {len(all_trajectories)} samples")

    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()

    return all_trajectories, epoch


def select_top_samples(all_trajectories, n_samples=3):
    """Select top N samples for each criterion."""
    # 1. Largest RMSD improvement (top N)
    sorted_by_improvement = sorted(all_trajectories, key=lambda x: x['rmsd_change'], reverse=True)
    best_improvements = sorted_by_improvement[:n_samples]

    # 2. Among those with >2Å movement, best final RMSD (top N)
    moved_over_2A = [t for t in all_trajectories if t['rmsd_change'] > 2.0]
    if len(moved_over_2A) >= n_samples:
        sorted_by_final = sorted(moved_over_2A, key=lambda x: x['final_rmsd'])
        best_finals = sorted_by_final[:n_samples]
    else:
        # Fallback: just find the ones with smallest final RMSD
        sorted_by_final = sorted(all_trajectories, key=lambda x: x['final_rmsd'])
        best_finals = sorted_by_final[:n_samples]
        print(f"Warning: Only {len(moved_over_2A)} samples with >2Å movement, using smallest final RMSD")

    return best_improvements, best_finals


def main():
    # Configuration
    config_path = 'configs/inference.yaml'
    checkpoints = {
        'best': 'save/flowfix_20251209_141538/checkpoints/early_stop_2025-12-09_14-17-14.pth',
        'latest': 'save/flowfix_20251209_141538/checkpoints/latest.pt'
    }
    output_dir = 'inference_results/trajectory_visualization'
    os.makedirs(output_dir, exist_ok=True)

    num_steps = 50
    n_samples = 3  # Top N samples per criterion

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"ODE steps: {num_steps}")
    print(f"Top N samples per criterion: {n_samples}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = FlowFixDataset(
        data_dir=config['data'].get('data_dir', 'train_data'),
        split='valid',
        split_file=config['data'].get('split_file', None),
        max_samples=None,
        seed=config.get('seed', 42),
        loading_mode='lazy',
        rank=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_flowfix_batch
    )
    print(f"Loaded {len(val_dataset)} validation samples")

    # Run inference for each checkpoint
    all_results = {}
    all_epochs = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        trajectories, epoch = run_inference_for_checkpoint(
            ckpt_path, val_loader, config, device, num_steps
        )
        all_results[ckpt_name] = trajectories
        all_epochs[ckpt_name] = epoch

    # Create summary comparison plot
    print("\n" + "="*60)
    print("GENERATING SUMMARY COMPARISON")
    print("="*60)
    visualize_comparison_summary(all_results, os.path.join(output_dir, 'comparison_summary.png'))

    # Process each checkpoint
    for ckpt_name, trajectories in all_results.items():
        epoch = all_epochs[ckpt_name]
        ckpt_dir = os.path.join(output_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"PROCESSING {ckpt_name.upper()} (Epoch {epoch})")
        print("="*60)

        # Select top samples
        best_improvements, best_finals = select_top_samples(trajectories, n_samples)

        # Print statistics
        print(f"\nTop {n_samples} Largest RMSD Improvements:")
        for i, traj in enumerate(best_improvements):
            print(f"  {i+1}. PDB: {traj['pdb_id']}, "
                  f"Init: {traj['initial_rmsd']:.3f}Å → Final: {traj['final_rmsd']:.3f}Å, "
                  f"Improvement: {traj['rmsd_change']:.3f}Å")

        print(f"\nTop {n_samples} Best Final RMSD (with >2Å movement):")
        for i, traj in enumerate(best_finals):
            print(f"  {i+1}. PDB: {traj['pdb_id']}, "
                  f"Init: {traj['initial_rmsd']:.3f}Å → Final: {traj['final_rmsd']:.3f}Å, "
                  f"Improvement: {traj['rmsd_change']:.3f}Å")

        # Generate visualizations
        print(f"\nGenerating visualizations for {ckpt_name}...")

        # Best improvements
        for i, traj in enumerate(best_improvements):
            prefix = f"improvement_top{i+1}"

            visualize_trajectory_png(
                traj,
                f"{ckpt_name.upper()} (Epoch {epoch}) - Improvement Rank #{i+1}\n"
                f"PDB: {traj['pdb_id']} | {traj['initial_rmsd']:.2f}Å → {traj['final_rmsd']:.2f}Å",
                os.path.join(ckpt_dir, f'{prefix}_{traj["pdb_id"]}.png')
            )

            create_trajectory_gif(traj, ckpt_dir, prefix, epoch)

        # Best final RMSD
        for i, traj in enumerate(best_finals):
            prefix = f"final_rmsd_top{i+1}"

            visualize_trajectory_png(
                traj,
                f"{ckpt_name.upper()} (Epoch {epoch}) - Best Final RMSD #{i+1}\n"
                f"PDB: {traj['pdb_id']} | {traj['initial_rmsd']:.2f}Å → {traj['final_rmsd']:.2f}Å",
                os.path.join(ckpt_dir, f'{prefix}_{traj["pdb_id"]}.png')
            )

            create_trajectory_gif(traj, ckpt_dir, prefix, epoch)

        # Save trajectory data
        print(f"\nSaving trajectory data for {ckpt_name}...")
        save_data = {}
        for i, traj in enumerate(best_improvements):
            save_data[f'improvement_top{i+1}_pdb_id'] = traj['pdb_id']
            save_data[f'improvement_top{i+1}_times'] = traj['times']
            save_data[f'improvement_top{i+1}_rmsds'] = traj['rmsds']
            save_data[f'improvement_top{i+1}_coords'] = np.array(traj['coords'])
            save_data[f'improvement_top{i+1}_target'] = traj['target_coords']
            save_data[f'improvement_top{i+1}_edges_src'] = traj['edges'][0].numpy()
            save_data[f'improvement_top{i+1}_edges_dst'] = traj['edges'][1].numpy()

        for i, traj in enumerate(best_finals):
            save_data[f'final_top{i+1}_pdb_id'] = traj['pdb_id']
            save_data[f'final_top{i+1}_times'] = traj['times']
            save_data[f'final_top{i+1}_rmsds'] = traj['rmsds']
            save_data[f'final_top{i+1}_coords'] = np.array(traj['coords'])
            save_data[f'final_top{i+1}_target'] = traj['target_coords']
            save_data[f'final_top{i+1}_edges_src'] = traj['edges'][0].numpy()
            save_data[f'final_top{i+1}_edges_dst'] = traj['edges'][1].numpy()

        np.savez(os.path.join(ckpt_dir, 'trajectory_data.npz'), **save_data)
        print(f"Saved: {os.path.join(ckpt_dir, 'trajectory_data.npz')}")

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
