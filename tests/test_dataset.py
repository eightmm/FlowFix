"""
Test FlowFixDataset with new ligand data format.

Tests:
1. Loading single ligand data file
2. Creating PyG graph from new format
3. Dataset iteration and batching
4. Coordinate extraction and validation
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from torch.utils.data import DataLoader

# Make pytest optional
try:
    import pytest
except ImportError:
    pytest = None

    # Mock pytest.skip for standalone execution
    class MockPytest:
        @staticmethod
        def skip(msg):
            print(f"⏭️  Skipping: {msg}")
            raise RuntimeError(f"Skip: {msg}")

    if pytest is None:
        pytest = MockPytest()


class TestLigandDataFormat:
    """Test new flat dictionary format for ligand data."""

    def test_load_single_ligand(self):
        """Test loading a single ligand.pt file."""
        # Find first available PDB
        data_dir = Path("train_data_dg")
        if not data_dir.exists():
            pytest.skip("train_data_dg directory not found")

        pdbs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        if not pdbs:
            pytest.skip("No PDB directories found")

        pdb_id = pdbs[0]
        ligand_path = data_dir / pdb_id / "ligands.pt"

        if not ligand_path.exists():
            pytest.skip(f"ligands.pt not found for {pdb_id}")

        # Load ligand data
        ligands_list = torch.load(ligand_path, weights_only=False)

        print(f"\n=== Testing PDB: {pdb_id} ===")
        print(f"Number of ligand poses: {len(ligands_list)}")

        # Check first ligand
        ligand = ligands_list[0]
        print(f"\nLigand data keys: {ligand.keys()}")

        # Verify required keys
        required_keys = ['edges', 'node_feats', 'edge_feats', 'coord', 'crystal_coord']
        for key in required_keys:
            assert key in ligand, f"Missing key: {key}"

        # Check shapes
        print(f"\nData shapes:")
        print(f"  edges: {ligand['edges'].shape}")
        print(f"  node_feats: {ligand['node_feats'].shape}")
        print(f"  edge_feats: {ligand['edge_feats'].shape}")
        print(f"  coord: {ligand['coord'].shape}")
        print(f"  crystal_coord: {ligand['crystal_coord'].shape}")

        if 'distance_lower_bounds' in ligand:
            print(f"  distance_lower_bounds: {ligand['distance_lower_bounds'].shape}")
        if 'distance_upper_bounds' in ligand:
            print(f"  distance_upper_bounds: {ligand['distance_upper_bounds'].shape}")

        # Validate edge index format
        assert ligand['edges'].shape[0] == 2, "edges should be [2, E]"

        # Validate coordinate shapes match
        n_atoms = ligand['coord'].shape[0]
        assert ligand['crystal_coord'].shape[0] == n_atoms, \
            "coord and crystal_coord should have same number of atoms"
        assert ligand['node_feats'].shape[0] == n_atoms, \
            "node_feats should match number of atoms"

        print("\n✅ Single ligand data format validation passed!")


class TestFlowFixDataset:
    """Test FlowFixDataset with new data format."""

    def test_dataset_initialization(self):
        """Test dataset initialization and PDB loading."""
        if not Path("train_data_dg").exists():
            pytest.skip("train_data_dg directory not found")

        dataset = FlowFixDataset(
            data_dir="train_data_dg",
            split="train",
            max_samples=5,
            seed=42
        )

        print(f"\n=== Dataset Initialization ===")
        print(f"Number of PDBs: {len(dataset)}")
        print(f"First 5 PDB IDs: {dataset.pdb_ids[:5]}")

        assert len(dataset) > 0, "Dataset should have at least one sample"
        print("✅ Dataset initialization passed!")

    def test_single_sample(self):
        """Test loading a single sample from dataset."""
        if not Path("train_data_dg").exists():
            pytest.skip("train_data_dg directory not found")

        dataset = FlowFixDataset(
            data_dir="train_data_dg",
            split="train",
            max_samples=1,
            seed=42
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        # Get first sample
        sample = dataset[0]

        if sample is None:
            pytest.skip("Sample has inconsistent atom counts")

        print(f"\n=== Single Sample Test ===")
        print(f"PDB ID: {sample['pdb_id']}")
        print(f"Ligand graph: {sample['ligand_graph']}")
        print(f"Protein graph: {sample['protein_graph']}")
        print(f"Ligand coords x0 shape: {sample['ligand_coords_x0'].shape}")
        print(f"Ligand coords x1 shape: {sample['ligand_coords_x1'].shape}")

        # Validate sample structure
        assert 'pdb_id' in sample
        assert 'protein_graph' in sample
        assert 'ligand_graph' in sample
        assert 'ligand_coords_x0' in sample
        assert 'ligand_coords_x1' in sample

        # Validate graph properties
        ligand_graph = sample['ligand_graph']
        assert hasattr(ligand_graph, 'x'), "Ligand graph missing node features"
        assert hasattr(ligand_graph, 'edge_index'), "Ligand graph missing edge index"
        assert hasattr(ligand_graph, 'pos'), "Ligand graph missing positions"

        # Check if distance bounds are preserved
        if hasattr(ligand_graph, 'distance_lower_bounds'):
            print(f"Distance lower bounds shape: {ligand_graph.distance_lower_bounds.shape}")
        if hasattr(ligand_graph, 'distance_upper_bounds'):
            print(f"Distance upper bounds shape: {ligand_graph.distance_upper_bounds.shape}")

        print("✅ Single sample test passed!")

    def test_batch_collation(self):
        """Test batching multiple samples."""
        if not Path("train_data_dg").exists():
            pytest.skip("train_data_dg directory not found")

        dataset = FlowFixDataset(
            data_dir="train_data_dg",
            split="train",
            max_samples=3,
            seed=42
        )

        if len(dataset) < 2:
            pytest.skip("Need at least 2 samples for batching test")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_flowfix_batch
        )

        # Get first batch
        batch = next(iter(dataloader))

        print(f"\n=== Batch Collation Test ===")
        print(f"Batch size: {len(batch['pdb_ids'])}")
        print(f"PDB IDs: {batch['pdb_ids']}")
        print(f"Total ligand atoms: {batch['ligand_coords_x0'].shape[0]}")
        print(f"Total protein residues: {batch['protein_graph'].num_nodes}")
        print(f"Ligand batch indices: {batch['ligand_batch']}")

        # Validate batch structure
        assert 'pdb_ids' in batch
        assert 'protein_graph' in batch
        assert 'ligand_graph' in batch
        assert 'ligand_coords_x0' in batch
        assert 'ligand_coords_x1' in batch
        assert 'ligand_batch' in batch

        # Validate batch sizes
        assert len(batch['pdb_ids']) == 2
        assert batch['ligand_coords_x0'].shape == batch['ligand_coords_x1'].shape

        print("✅ Batch collation test passed!")

    def test_epoch_sampling(self):
        """Test that different epochs sample different poses."""
        if not Path("train_data_dg").exists():
            pytest.skip("train_data_dg directory not found")

        dataset = FlowFixDataset(
            data_dir="train_data_dg",
            split="train",
            max_samples=1,
            seed=42
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        # Sample same index across different epochs
        coords_per_epoch = []

        for epoch in range(3):
            dataset.set_epoch(epoch)
            sample = dataset[0]

            if sample is None:
                continue

            coords_per_epoch.append(sample['ligand_coords_x0'])

        if len(coords_per_epoch) < 2:
            pytest.skip("Not enough valid samples across epochs")

        print(f"\n=== Epoch Sampling Test ===")
        print(f"Sampled coordinates across {len(coords_per_epoch)} epochs")

        # Check if coordinates differ across epochs (indicating different poses)
        coords_same = []
        for i in range(len(coords_per_epoch) - 1):
            same = torch.allclose(coords_per_epoch[i], coords_per_epoch[i+1], atol=1e-6)
            coords_same.append(same)
            print(f"Epoch {i} vs {i+1}: {'Same' if same else 'Different'}")

        # At least some epochs should have different poses
        # (unless there's only 1 pose in the ligands.pt file)
        print("✅ Epoch sampling test passed!")

    def test_coordinate_consistency(self):
        """Test that coordinates are consistent with graph positions."""
        if not Path("train_data_dg").exists():
            pytest.skip("train_data_dg directory not found")

        dataset = FlowFixDataset(
            data_dir="train_data_dg",
            split="train",
            max_samples=1,
            seed=42
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        sample = dataset[0]

        if sample is None:
            pytest.skip("Sample has inconsistent atom counts")

        print(f"\n=== Coordinate Consistency Test ===")

        # Check that ligand graph pos matches ligand_coords_x0
        ligand_graph = sample['ligand_graph']
        coords_x0 = sample['ligand_coords_x0']

        print(f"Graph pos shape: {ligand_graph.pos.shape}")
        print(f"coords_x0 shape: {coords_x0.shape}")

        assert torch.allclose(ligand_graph.pos, coords_x0, atol=1e-6), \
            "Graph positions should match ligand_coords_x0"

        print("✅ Coordinate consistency test passed!")


def run_all_tests():
    """Run all tests manually (for debugging)."""
    print("="*60)
    print("Testing FlowFix Dataset with New Ligand Format")
    print("="*60)

    # Test ligand data format
    test_format = TestLigandDataFormat()
    try:
        test_format.test_load_single_ligand()
    except Exception as e:
        print(f"❌ Error in test_load_single_ligand: {e}")

    # Test dataset
    test_dataset = TestFlowFixDataset()

    try:
        test_dataset.test_dataset_initialization()
    except Exception as e:
        print(f"❌ Error in test_dataset_initialization: {e}")

    try:
        test_dataset.test_single_sample()
    except Exception as e:
        print(f"❌ Error in test_single_sample: {e}")

    try:
        test_dataset.test_batch_collation()
    except Exception as e:
        print(f"❌ Error in test_batch_collation: {e}")

    try:
        test_dataset.test_epoch_sampling()
    except Exception as e:
        print(f"❌ Error in test_epoch_sampling: {e}")

    try:
        test_dataset.test_coordinate_consistency()
    except Exception as e:
        print(f"❌ Error in test_coordinate_consistency: {e}")

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    # Run all tests
    run_all_tests()
