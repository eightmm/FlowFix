"""Debug batching issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch_geometric.data import Batch

# Test simple torch.cat
print("Testing torch.cat with different sizes:")
tensor1 = torch.randn(75, 3)
tensor2 = torch.randn(62, 3)

try:
    result = torch.cat([tensor1, tensor2], dim=0)
    print(f"✅ torch.cat works! Result shape: {result.shape}")
except Exception as e:
    print(f"❌ torch.cat failed: {e}")

# Now test with dataset
from src.data.dataset import FlowFixDataset, collate_flowfix_batch

print("\n" + "="*60)
print("Testing Dataset Batching")
print("="*60)

dataset = FlowFixDataset(
    data_dir="train_data_dg",
    split="train",
    max_samples=2,
    seed=42
)

if len(dataset) >= 2:
    # Get two samples manually
    sample1 = dataset[0]
    sample2 = dataset[1]

    if sample1 is not None and sample2 is not None:
        print(f"\nSample 1: {sample1['ligand_coords_x0'].shape}")
        print(f"Sample 2: {sample2['ligand_coords_x0'].shape}")

        # Test each component separately
        print("\n1. Testing ligand graph batching:")
        try:
            ligand_batch = Batch.from_data_list([sample1['ligand_graph'], sample2['ligand_graph']])
            print(f"   ✅ Ligand batch: {ligand_batch}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

        print("\n2. Testing protein graph batching:")
        try:
            protein_batch = Batch.from_data_list([sample1['protein_graph'], sample2['protein_graph']])
            print(f"   ✅ Protein batch: {protein_batch}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

        print("\n3. Testing coords_x0 concatenation:")
        try:
            coords_x0 = torch.cat([sample1['ligand_coords_x0'], sample2['ligand_coords_x0']], dim=0)
            print(f"   ✅ coords_x0 shape: {coords_x0.shape}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

        print("\n4. Testing coords_x1 concatenation:")
        try:
            coords_x1 = torch.cat([sample1['ligand_coords_x1'], sample2['ligand_coords_x1']], dim=0)
            print(f"   ✅ coords_x1 shape: {coords_x1.shape}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

        print("\n5. Testing full collate function:")
        try:
            batch = collate_flowfix_batch([sample1, sample2])
            print(f"   ✅ Full batch created!")
            print(f"   PDB IDs: {batch['pdb_ids']}")
            print(f"   Total atoms: {batch['ligand_coords_x0'].shape}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
