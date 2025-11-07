#!/usr/bin/env python3
"""
Visualization utilities for molecular pose refinement
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import torch


class MolecularVisualizer:
    """Molecular visualization utilities for pose refinement analysis"""
    
    def __init__(self, vis_dir: str):
        """
        Initialize visualizer
        
        Args:
            vis_dir: Directory to save visualization files
        """
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sampling_gif(self, trajectory, crystal_coords, edges, epoch, pdb_id=None, multi_view=True, velocities=None):
        """
        Create GIF animation showing sampling trajectory with multiple viewing angles

        Args:
            trajectory: List of coordinate arrays for each step [N_steps, N_atoms, 3]
            crystal_coords: Crystal structure coordinates [N_atoms, 3]
            edges: Tuple of (edge_src, edge_dst) for bonds
            epoch: Current epoch number
            pdb_id: PDB ID for naming (optional)
            multi_view: If True, create multi-view animation with rotating camera
            velocities: Optional list of velocity field arrays for each step [N_steps, N_atoms, 3]
        """
        try:
            if multi_view:
                return self._create_multi_view_gif(trajectory, crystal_coords, edges, epoch, pdb_id, velocities)
            else:
                return self._create_single_view_gif(trajectory, crystal_coords, edges, epoch, pdb_id, velocities)
        except Exception as e:
            print(f"Warning: Could not create sampling GIF: {e}")
            try:
                plt.close('all')
            except:
                pass

    def _create_single_view_gif(self, trajectory, crystal_coords, edges, epoch, pdb_id=None, velocities=None):
        """Create single view GIF animation"""
        # Create figure - single 3D view
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Get edges for bonds - handle empty edges case
        if len(edges[0]) > 0:
            edge_list = [(edges[0][i].item(), edges[1][i].item()) for i in range(len(edges[0]))]
        else:
            edge_list = []
            print(f"   ⚠️  No bonds found for visualization")

        # Calculate bounds
        all_coords = np.vstack([trajectory[0], trajectory[-1], crystal_coords])
        margin = 2
        x_range = [all_coords[:,0].min()-margin, all_coords[:,0].max()+margin]
        y_range = [all_coords[:,1].min()-margin, all_coords[:,1].max()+margin]
        z_range = [all_coords[:,2].min()-margin, all_coords[:,2].max()+margin]

        # Calculate RMSD progression
        rmsd_values = []
        for coords in trajectory:
            rmsd = np.sqrt(np.mean((coords - crystal_coords) ** 2))
            rmsd_values.append(rmsd)

        def animate(frame):
            # Remove existing artists to avoid accumulation
            for artist in ax.collections[:] + ax.lines[:] + ax.texts[:]:
                artist.remove()

            current_coords = trajectory[frame]
            current_rmsd = rmsd_values[frame]
            initial_coords = trajectory[0]  # First frame (docked position)
            current_velocity = velocities[frame] if velocities is not None else None

            self._plot_molecular_structure(ax, current_coords, crystal_coords, edge_list,
                                         x_range, y_range, z_range, current_rmsd, frame, epoch, pdb_id,
                                         initial_coords, current_velocity)

            # Force set the same axis limits for consistency (prevent auto-scaling)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_autoscale_on(False)  # Disable auto-scaling
        
        # Create animation with faster playback
        anim = FuncAnimation(fig, animate, frames=len(trajectory), 
                           interval=75, repeat=True, blit=False)  # 75ms for fast playback
        
        # Save as GIF with memory management
        if pdb_id:
            gif_path = self.vis_dir / f'epoch_{epoch:04d}_{pdb_id}.gif'
        else:
            gif_path = self.vis_dir / f'epoch_{epoch:04d}.gif'
        anim.save(gif_path, writer='pillow', fps=10)  # 10fps for smooth playback
        print(f"💫 Saved sampling GIF: {gif_path}")
        
        # Clean up matplotlib objects
        plt.close(fig)
        del anim
        del fig
        
        return str(gif_path)

    def _create_multi_view_gif(self, trajectory, crystal_coords, edges, epoch, pdb_id=None, velocities=None):
        """Create multi-view GIF animation with rotating camera"""
        # Create figure with 4 subplots for different views
        fig = plt.figure(figsize=(16, 12))
        
        # Define viewing angles (elevation, azimuth) - consistent axis directions
        view_angles = [
            (20, 45),   # Front-right view
            (20, 135),  # Back-right view  
            (20, 225),  # Back-left view
            (20, 315),  # Front-left view
        ]
        
        axes = []
        for i, (elev, azim) in enumerate(view_angles):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            axes.append(ax)
        
        # Get edges for bonds
        if len(edges[0]) > 0:
            edge_list = [(edges[0][i].item(), edges[1][i].item()) for i in range(len(edges[0]))]
        else:
            edge_list = []
            print(f"   ⚠️  No bonds found for visualization")
        
        # Calculate bounds
        all_coords = np.vstack([trajectory[0], trajectory[-1], crystal_coords])
        margin = 2
        x_range = [all_coords[:,0].min()-margin, all_coords[:,0].max()+margin]
        y_range = [all_coords[:,1].min()-margin, all_coords[:,1].max()+margin]
        z_range = [all_coords[:,2].min()-margin, all_coords[:,2].max()+margin]
        
        # Calculate RMSD progression
        rmsd_values = []
        for coords in trajectory:
            rmsd = np.sqrt(np.mean((coords - crystal_coords) ** 2))
            rmsd_values.append(rmsd)
        
        def animate(frame):
            current_coords = trajectory[frame]
            current_rmsd = rmsd_values[frame]
            initial_coords = trajectory[0]  # First frame (docked position)
            current_velocity = velocities[frame] if velocities is not None else None

            for i, ax in enumerate(axes):
                # Remove existing artists to avoid accumulation
                for artist in ax.collections[:] + ax.lines[:] + ax.texts[:]:
                    artist.remove()

                elev, azim = view_angles[i]

                self._plot_molecular_structure(ax, current_coords, crystal_coords, edge_list,
                                             x_range, y_range, z_range, current_rmsd, frame, epoch, pdb_id,
                                             initial_coords, current_velocity)
                ax.view_init(elev=elev, azim=azim)

                # Force set the same axis limits for consistency (prevent auto-scaling)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_zlim(z_range)
                ax.set_autoscale_on(False)  # Disable auto-scaling

        # Create animation with fixed camera angles (no rotation)
        def animate_fixed(frame):
            current_coords = trajectory[frame]
            current_rmsd = rmsd_values[frame]
            initial_coords = trajectory[0]  # First frame (docked position)
            current_velocity = velocities[frame] if velocities is not None else None

            for i, ax in enumerate(axes):
                # Remove existing artists to avoid accumulation
                for artist in ax.collections[:] + ax.lines[:] + ax.texts[:]:
                    artist.remove()

                # Use fixed camera angles - no rotation
                elev, azim = view_angles[i]

                self._plot_molecular_structure(ax, current_coords, crystal_coords, edge_list,
                                             x_range, y_range, z_range, current_rmsd, frame, epoch, pdb_id,
                                             initial_coords, current_velocity)
                ax.view_init(elev=elev, azim=azim)

                # Force set the same axis limits for all subplots (prevent auto-scaling)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_zlim(z_range)
                ax.set_autoscale_on(False)  # Disable auto-scaling
        
        # Create animation with fixed views
        anim = FuncAnimation(fig, animate_fixed, frames=len(trajectory), 
                           interval=100, repeat=True, blit=False)  # 100ms for multi-view
        
        # Save as GIF
        if pdb_id:
            gif_path = self.vis_dir / f'epoch_{epoch:04d}_{pdb_id}_multiview.gif'
        else:
            gif_path = self.vis_dir / f'epoch_{epoch:04d}_multiview.gif'
        anim.save(gif_path, writer='pillow', fps=8)  # 8fps for multi-view
        print(f"💫 Saved multi-view sampling GIF: {gif_path}")
        
        # Clean up matplotlib objects
        plt.close(fig)
        del anim
        del fig
        
        return str(gif_path)

    def _plot_molecular_structure(self, ax, current_coords, crystal_coords, edge_list,
                                x_range, y_range, z_range, current_rmsd, frame, epoch, pdb_id,
                                initial_coords=None, velocity=None):
        """Plot molecular structure on given axis with optional velocity field arrows"""
        # Plot crystal structure (target) in green with smaller atoms
        ax.scatter(crystal_coords[:, 0], crystal_coords[:, 1], crystal_coords[:, 2],
                   c='green', s=30, alpha=0.7, label='Crystal (Target)')

        # Plot current structure in red with smaller atoms
        ax.scatter(current_coords[:, 0], current_coords[:, 1], current_coords[:, 2],
                   c='red', s=40, alpha=0.9, label=f'Current (Step {frame})')

        # Plot initial structure (docked) in blue if provided
        if initial_coords is not None:
            ax.scatter(initial_coords[:, 0], initial_coords[:, 1], initial_coords[:, 2],
                       c='blue', s=25, alpha=0.6, label='Docked (Initial)')

        # Plot velocity field as arrows (if provided)
        if velocity is not None and len(velocity) > 0:
            # Normalize velocities for better visualization
            velocity_magnitudes = np.linalg.norm(velocity, axis=1)
            max_magnitude = velocity_magnitudes.max() if velocity_magnitudes.max() > 0 else 1.0

            # Scale arrows for visibility (adjust scale factor for better visualization)
            arrow_scale = 0.5  # Adjust this to make arrows longer/shorter

            # Draw arrows for each atom
            # Use quiver for efficient 3D arrows
            ax.quiver(
                current_coords[:, 0], current_coords[:, 1], current_coords[:, 2],
                velocity[:, 0], velocity[:, 1], velocity[:, 2],
                color='orange', alpha=0.8, arrow_length_ratio=0.3,
                linewidth=1.5, length=arrow_scale, normalize=False,
                label='Velocity Field'
            )
        
        # Draw bonds for current structure
        for bond in edge_list:
            if bond[0] < len(current_coords) and bond[1] < len(current_coords):
                ax.plot3D([current_coords[bond[0], 0], current_coords[bond[1], 0]],
                          [current_coords[bond[0], 1], current_coords[bond[1], 1]],
                          [current_coords[bond[0], 2], current_coords[bond[1], 2]], 
                          'r-', alpha=0.6, linewidth=1.5)
        
        # Draw bonds for crystal structure
        for bond in edge_list:
            if bond[0] < len(crystal_coords) and bond[1] < len(crystal_coords):
                ax.plot3D([crystal_coords[bond[0], 0], crystal_coords[bond[1], 0]],
                          [crystal_coords[bond[0], 1], crystal_coords[bond[1], 1]],
                          [crystal_coords[bond[0], 2], crystal_coords[bond[1], 2]], 
                          'g-', alpha=0.4, linewidth=1)
        
        # Draw bonds for initial structure if provided
        if initial_coords is not None:
            for bond in edge_list:
                if bond[0] < len(initial_coords) and bond[1] < len(initial_coords):
                    ax.plot3D([initial_coords[bond[0], 0], initial_coords[bond[1], 0]],
                              [initial_coords[bond[0], 1], initial_coords[bond[1], 1]],
                              [initial_coords[bond[0], 2], initial_coords[bond[1], 2]], 
                              'b-', alpha=0.3, linewidth=0.8)
        
        # Draw dashed lines connecting initial to target positions (trajectory arrows)
        if initial_coords is not None and len(initial_coords) == len(crystal_coords):
            for i in range(len(initial_coords)):
                # Connection from initial (docked) to target (crystal)
                ax.plot3D([initial_coords[i, 0], crystal_coords[i, 0]],
                          [initial_coords[i, 1], crystal_coords[i, 1]],
                          [initial_coords[i, 2], crystal_coords[i, 2]], 
                          'k--', alpha=0.4, linewidth=0.8)  # Black dashed lines
        
        # Set labels (limits are set by caller for consistency)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)') 
        ax.set_zlabel('Z (Å)')
        
        # Add title with RMSD info
        title = f'Epoch {epoch} - Step {frame}\nRMSD: {current_rmsd:.3f} Å'
        if pdb_id:
            title += f' - PDB: {pdb_id}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Add text box with additional info (only on first subplot)
        if hasattr(self, '_first_subplot') and not self._first_subplot:
            self._first_subplot = True
            initial_rmsd = current_rmsd  # Will be updated in calling function
            improvement = 0  # Will be calculated properly
            ax.text2D(0.02, 0.98, f'RMSD: {current_rmsd:.3f} Å', 
                     transform=ax.transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=8)


# Convenience function for easy import
def create_visualizations(vis_dir, trajectory, crystal_coords, edges, epoch, pdb_id=None):
    """
    Convenience function to create GIF visualization
    
    Args:
        vis_dir: Directory to save visualizations
        trajectory: List of coordinate arrays for each step
        crystal_coords: Crystal structure coordinates
        edges: Tuple of (edge_src, edge_dst) for bonds
        epoch: Current epoch number
        pdb_id: PDB ID for naming (optional)
    """
    visualizer = MolecularVisualizer(vis_dir)
    visualizer.create_sampling_gif(trajectory, crystal_coords, edges, epoch, pdb_id)


if __name__ == "__main__":
    # Test the visualization functions
    print("Testing molecular visualization...")
    
    # Create dummy data
    n_steps = 20
    n_atoms = 10
    
    # Random trajectory
    trajectory = []
    for i in range(n_steps):
        coords = np.random.randn(n_atoms, 3) * (1 - i/n_steps) + np.ones((n_atoms, 3)) * i/n_steps
        trajectory.append(coords)
    
    crystal_coords = np.ones((n_atoms, 3))
    edges = (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0]))  # Simple ring
    
    # Test visualizations
    create_visualizations(
        vis_dir="test_vis",
        trajectory=trajectory,
        crystal_coords=crystal_coords,
        edges=edges,
        epoch=999,
        pdb_id="TEST"
    )
