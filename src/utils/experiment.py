"""
Experiment Manager for Unified Directory Structure

Manages all experiment outputs (checkpoints, logs, wandb, visualizations)
in a single organized directory per experiment run.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ExperimentManager:
    """
    Manages unified experiment directory structure.

    Directory Structure:
        save/
          └── {run_id}/
              ├── checkpoints/      # Model checkpoints (.pt files)
              ├── logs/             # Training logs (text logs, tensorboard)
              ├── wandb/            # WandB files
              ├── visualizations/   # Trajectory animations, plots
              └── config.yaml       # Saved config for this run
    """

    def __init__(
        self,
        base_dir: str = "save",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment manager.

        Args:
            base_dir: Base directory for all experiments (default: "save")
            run_name: Optional custom run name. If None, auto-generates timestamp-based name
            config: Optional config dict to save with this experiment
        """
        self.base_dir = Path(base_dir)

        # Generate unique run ID
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        self.run_name = run_name
        self.run_dir = self.base_dir / run_name

        # Create all subdirectories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.wandb_dir = self.run_dir / "wandb"
        self.visualizations_dir = self.run_dir / "visualizations"

        # Create directories
        self._create_directories()

        # Setup logging
        self._setup_logging()

        # Save config if provided
        if config is not None:
            self.save_config(config)

        self.logger.info(f"✓ Experiment initialized: {self.run_name}")
        self.logger.info(f"  Directory: {self.run_dir}")

    def _create_directories(self):
        """Create all necessary directories."""
        for dir_path in [
            self.run_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.wandb_dir,
            self.visualizations_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging to file and console."""
        # Create logger
        self.logger = logging.getLogger(f"experiment_{self.run_name}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler
        log_file = self.logs_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def save_config(self, config: Dict[str, Any]):
        """
        Save experiment configuration.

        Args:
            config: Configuration dictionary
        """
        import yaml

        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"✓ Config saved: {config_path}")

    def save_metadata(self, metadata: Dict[str, Any]):
        """
        Save experiment metadata (git hash, environment info, etc.).

        Args:
            metadata: Metadata dictionary
        """
        metadata_path = self.run_dir / "metadata.json"

        # Add timestamp
        metadata['created_at'] = datetime.now().isoformat()
        metadata['run_name'] = self.run_name

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✓ Metadata saved: {metadata_path}")

    def get_checkpoint_path(self, filename: str) -> Path:
        """Get path for checkpoint file."""
        return self.checkpoints_dir / filename

    def get_visualization_path(self, filename: str) -> Path:
        """Get path for visualization file."""
        return self.visualizations_dir / filename

    def get_log_path(self, filename: str) -> Path:
        """Get path for log file."""
        return self.logs_dir / filename

    def get_wandb_dir(self) -> str:
        """Get WandB directory path (as string for wandb.init)."""
        return str(self.wandb_dir)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary information."""
        return {
            'run_name': self.run_name,
            'run_dir': str(self.run_dir),
            'checkpoints_dir': str(self.checkpoints_dir),
            'logs_dir': str(self.logs_dir),
            'wandb_dir': str(self.wandb_dir),
            'visualizations_dir': str(self.visualizations_dir),
        }

    def list_checkpoints(self) -> list:
        """List all checkpoint files."""
        return sorted([f.name for f in self.checkpoints_dir.glob('*.pt')])

    def list_visualizations(self) -> list:
        """List all visualization files."""
        return sorted([
            f.name for f in self.visualizations_dir.glob('*')
            if f.suffix in ['.gif', '.mp4', '.png', '.jpg']
        ])

    def cleanup_old_checkpoints(self, keep_last_n: int = 5, keep_best: bool = True):
        """
        Clean up old checkpoints, keeping only the most recent ones.

        Args:
            keep_last_n: Number of most recent checkpoints to keep
            keep_best: Whether to always keep 'best.pt' checkpoint
        """
        checkpoints = [f for f in self.checkpoints_dir.glob('epoch_*.pt')]

        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime)

        # Determine which to delete
        to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []

        # Delete old checkpoints
        for ckpt in to_delete:
            # Skip if it's the best checkpoint and keep_best is True
            if keep_best and ckpt.name == 'best.pt':
                continue

            ckpt.unlink()
            self.logger.info(f"Removed old checkpoint: {ckpt.name}")

        if to_delete:
            self.logger.info(f"✓ Cleaned up {len(to_delete)} old checkpoints")

    def __str__(self):
        """String representation of experiment manager."""
        return f"ExperimentManager(run_name='{self.run_name}', run_dir='{self.run_dir}')"

    def __repr__(self):
        """Detailed representation of experiment manager."""
        summary = self.get_summary()
        return f"ExperimentManager({json.dumps(summary, indent=2)})"


def create_experiment(
    base_dir: str = "save",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ExperimentManager:
    """
    Create a new experiment with unified directory structure.

    Args:
        base_dir: Base directory for all experiments
        run_name: Optional custom run name
        config: Optional config to save
        metadata: Optional metadata to save

    Returns:
        ExperimentManager instance
    """
    exp_manager = ExperimentManager(
        base_dir=base_dir,
        run_name=run_name,
        config=config
    )

    if metadata is not None:
        exp_manager.save_metadata(metadata)

    return exp_manager


if __name__ == "__main__":
    # Test experiment manager
    print("Testing ExperimentManager...")

    # Create test experiment
    test_config = {
        'model': {'hidden_dim': 256},
        'training': {'lr': 0.001, 'epochs': 100}
    }

    test_metadata = {
        'git_hash': 'abc123',
        'python_version': '3.11',
        'cuda_version': '12.1'
    }

    exp = create_experiment(
        base_dir="save",
        run_name="test_experiment",
        config=test_config,
        metadata=test_metadata
    )

    # Test logging
    exp.logger.info("This is a test log message")

    # Print summary
    print("\nExperiment Summary:")
    for key, value in exp.get_summary().items():
        print(f"  {key}: {value}")

    # Test path methods
    print("\nExample Paths:")
    print(f"  Checkpoint: {exp.get_checkpoint_path('best.pt')}")
    print(f"  Visualization: {exp.get_visualization_path('trajectory.gif')}")
    print(f"  WandB dir: {exp.get_wandb_dir()}")

    print("\n✓ ExperimentManager test complete!")
