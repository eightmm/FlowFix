"""
Data utilities for FlowFix training
"""

from torch.utils.data import DataLoader
from src.data.dataset import FlowFixDataset, collate_flowfix_batch


def create_datasets(data_config, seed=42):
    """
    Create training and validation datasets.

    Args:
        data_config: Data configuration dict
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, dataset_type)
    """
    # Get loading mode from config
    loading_mode = data_config.get('loading_mode', 'lazy')

    # Training dataset (dynamic pose sampling)
    train_dataset = FlowFixDataset(
        data_dir=data_config.get('data_dir', 'train_data'),
        split='train',
        split_file=data_config.get('split_file', None),
        max_samples=data_config.get('max_train_samples', None),
        seed=seed,
        loading_mode=loading_mode
    )

    # Validation dataset (dynamic pose sampling)
    val_dataset = FlowFixDataset(
        data_dir=data_config.get('data_dir', 'train_data'),
        split='valid',
        split_file=data_config.get('split_file', None),
        max_samples=data_config.get('max_val_samples', None),
        seed=seed,
        loading_mode=loading_mode
    )

    dataset_type = "Dynamic"

    return train_dataset, val_dataset, dataset_type


def create_overfit_datasets(data_config, seed=42):
    """
    Create datasets for overfitting test (train and val use same data).

    This is useful for convergence testing - if the model can't overfit
    to a small dataset, there's likely a bug or the model is too weak.

    Args:
        data_config: Data configuration dict
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, dataset_type)
    """
    # Get loading mode from config
    loading_mode = data_config.get('loading_mode', 'lazy')

    # Training dataset (use 'train' split but with limited samples)
    train_dataset = FlowFixDataset(
        data_dir=data_config.get('data_dir', 'train_data'),
        split='train',
        split_file=data_config.get('split_file', None),
        max_samples=data_config.get('max_train_samples', 5),  # Small dataset for overfitting
        seed=seed,
        loading_mode=loading_mode
    )

    # Validation dataset: Use SAME data as training for overfitting test
    # We use 'train' split again (not 'valid')
    val_dataset = FlowFixDataset(
        data_dir=data_config.get('data_dir', 'train_data'),
        split='train',  # Same as training!
        split_file=data_config.get('split_file', None),
        max_samples=data_config.get('max_train_samples', 5),  # Same samples as training
        seed=seed,
        loading_mode=loading_mode
    )

    dataset_type = "Dynamic (Overfit)"

    return train_dataset, val_dataset, dataset_type


def create_dataloaders(train_dataset, val_dataset, training_config, data_config):
    """
    Create training and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_config: Training configuration dict
        data_config: Data configuration dict

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=collate_flowfix_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('val_batch_size', 4),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=collate_flowfix_batch
    )

    return train_loader, val_loader
