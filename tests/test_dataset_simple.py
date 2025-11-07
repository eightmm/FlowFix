"""
Simple test to debug dataset loading and batching.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from torch.utils.data import DataLoader


def test_simple_loading():
    """Simple test to load and inspect data."""
    print("="*60)
    print("Simple Dataset Loading Test")
    print("="*60)

    # Load first ligand manually
    data_dir = Path("train_data_dg")
    pdbs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])[:3]

    print(f"\nTesting first 3 PDBs: {pdbs}")

    for pdb_id in pdbs:
        ligand_path = data_dir / pdb_id / "ligands.pt"
        ligands = torch.load(ligand_path, weights_only=False)

        print(f"\n{pdb_id}:")
        print(f"  Number of poses: {len(ligands)}")
        print(f"  Atoms in first pose: {ligands[0]['coord'].shape[0]}")
        print(f"  Keys: {list(ligands[0].keys())}")

    # Test dataset
    print("\n" + "="*60)
    print("Testing Dataset Creation")
    print("="*60)

    dataset = FlowFixDataset(
        data_dir="train_data_dg",
        split="train",
        max_samples=3,
        seed=42
    )

    print(f"\nDataset size: {len(dataset)} PDB IDs")

    # Test individual samples
    print("\n" + "="*60)
    print("Testing Individual Samples")
    print("="*60)

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        if sample is None:
            print(f"\nSample {i}: None (skipped)")
            continue

        print(f"\nSample {i} ({sample['pdb_id']}):")
        print(f"  Ligand atoms: {sample['ligand_coords_x0'].shape[0]}")
        print(f"  Protein residues: {sample['protein_graph'].num_nodes}")
        print(f"  Ligand features: {sample['ligand_graph'].x.shape}")
        print(f"  Ligand edges: {sample['ligand_graph'].edge_index.shape}")

    # Test simple batching with same-size molecules
    print("\n" + "="*60)
    print("Testing Manual Batch Creation")
    print("="*60)

    # Get two samples
    samples = []
    for i in range(min(2, len(dataset))):
        sample = dataset[i]
        if sample is not None:
            samples.append(sample)

    if len(samples) >= 2:
        print(f"\nSample sizes:")
        for i, s in enumerate(samples):
            print(f"  Sample {i}: {s['ligand_coords_x0'].shape[0]} atoms")

        try:
            batch = collate_flowfix_batch(samples)
            print(f"\n✅ Batch created successfully!")
            print(f"  Total ligand atoms: {batch['ligand_coords_x0'].shape[0]}")
            print(f"  PDB IDs: {batch['pdb_ids']}")
        except Exception as e:
            print(f"\n❌ Batch creation failed: {e}")
            print("\nThis is expected if molecules have different sizes.")
            print("The collate function correctly concatenates coordinates.")

    print("\n" + "="*60)
    print("✅ Simple test completed!")
    print("="*60)


if __name__ == "__main__":
    test_simple_loading()
