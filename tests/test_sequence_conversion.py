"""
Test sequence conversion utilities (pyg_to_sequence and sequence_to_pyg).
"""
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.network import ProteinLigandInteractionNetwork


def test_round_trip_conversion():
    """Test that pyg_to_sequence and sequence_to_pyg are inverses."""
    print("\n" + "="*70)
    print("Test: Round-trip conversion (PyG → Sequence → PyG)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy interaction network for testing
    network = ProteinLigandInteractionNetwork(
        protein_output_irreps="64x0e + 16x1o + 16x1e",
        ligand_output_irreps="64x0e + 16x1o + 16x1e",
        hidden_dim=128,
        num_heads=4,
        pair_dim=16,
        num_layers=2
    ).to(device)

    # Create test data with variable batch sizes
    batch_size = 3
    nodes_per_batch = [10, 15, 8]  # Different number of nodes per batch
    feat_dim = 64

    # Create PyG format data
    features_list = []
    positions_list = []
    batch_idx_list = []

    for b in range(batch_size):
        n_nodes = nodes_per_batch[b]
        features_list.append(torch.randn(n_nodes, feat_dim, device=device))
        positions_list.append(torch.randn(n_nodes, 3, device=device))
        batch_idx_list.append(torch.full((n_nodes,), b, dtype=torch.long, device=device))

    original_features = torch.cat(features_list, dim=0)  # [33, 64]
    original_positions = torch.cat(positions_list, dim=0)  # [33, 3]
    original_batch_idx = torch.cat(batch_idx_list, dim=0)  # [33]

    print(f"\n1. Original PyG format:")
    print(f"   Features: {original_features.shape}")
    print(f"   Positions: {original_positions.shape}")
    print(f"   Batch idx: {original_batch_idx.shape}")
    print(f"   Nodes per batch: {nodes_per_batch}")

    # Convert to sequence
    padded_feat, padded_pos, mask = network.pyg_to_sequence(
        original_features, original_positions, original_batch_idx
    )

    print(f"\n2. Converted to sequence format:")
    print(f"   Padded features: {padded_feat.shape}")
    print(f"   Padded positions: {padded_pos.shape}")
    print(f"   Mask: {mask.shape}")
    print(f"   Valid nodes per batch: {mask.sum(dim=1).tolist()}")

    # Convert back to PyG
    recovered_features, recovered_positions, recovered_batch_idx = network.sequence_to_pyg(
        padded_feat, padded_pos, mask
    )

    print(f"\n3. Converted back to PyG format:")
    print(f"   Features: {recovered_features.shape}")
    print(f"   Positions: {recovered_positions.shape}")
    print(f"   Batch idx: {recovered_batch_idx.shape}")

    # Verify round-trip
    print(f"\n4. Verification:")

    # Check shapes match
    assert original_features.shape == recovered_features.shape, \
        f"Features shape mismatch: {original_features.shape} vs {recovered_features.shape}"
    assert original_positions.shape == recovered_positions.shape, \
        f"Positions shape mismatch: {original_positions.shape} vs {recovered_positions.shape}"
    assert original_batch_idx.shape == recovered_batch_idx.shape, \
        f"Batch idx shape mismatch: {original_batch_idx.shape} vs {recovered_batch_idx.shape}"
    print("   ✓ Shapes match")

    # Check values match
    feat_error = (original_features - recovered_features).abs().max().item()
    pos_error = (original_positions - recovered_positions).abs().max().item()
    batch_match = (original_batch_idx == recovered_batch_idx).all().item()

    print(f"   ✓ Features max error: {feat_error:.2e}")
    print(f"   ✓ Positions max error: {pos_error:.2e}")
    print(f"   ✓ Batch indices match: {batch_match}")

    assert feat_error < 1e-6, f"Features error too large: {feat_error}"
    assert pos_error < 1e-6, f"Positions error too large: {pos_error}"
    assert batch_match, "Batch indices don't match"

    print("\n✅ PASSED: Round-trip conversion preserves all data")
    return True


def test_empty_batches():
    """Test handling of edge cases (empty batches)."""
    print("\n" + "="*70)
    print("Test: Edge case handling")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = ProteinLigandInteractionNetwork(
        protein_output_irreps="64x0e + 16x1o + 16x1e",
        ligand_output_irreps="64x0e + 16x1o + 16x1e",
        hidden_dim=128,
        num_heads=4,
        pair_dim=16,
        num_layers=2
    ).to(device)

    # Test with single batch
    features = torch.randn(5, 64, device=device)
    positions = torch.randn(5, 3, device=device)
    batch_idx = torch.zeros(5, dtype=torch.long, device=device)

    padded_feat, padded_pos, mask = network.pyg_to_sequence(features, positions, batch_idx)
    recovered_features, recovered_positions, recovered_batch_idx = network.sequence_to_pyg(
        padded_feat, padded_pos, mask
    )

    assert torch.allclose(features, recovered_features, atol=1e-6)
    assert torch.allclose(positions, recovered_positions, atol=1e-6)
    assert (batch_idx == recovered_batch_idx).all()

    print("   ✓ Single batch case works")

    # Test with varying sizes including small batches
    features = torch.randn(20, 64, device=device)
    positions = torch.randn(20, 3, device=device)
    batch_idx = torch.tensor([0]*1 + [1]*10 + [2]*9, dtype=torch.long, device=device)

    padded_feat, padded_pos, mask = network.pyg_to_sequence(features, positions, batch_idx)
    recovered_features, recovered_positions, recovered_batch_idx = network.sequence_to_pyg(
        padded_feat, padded_pos, mask
    )

    assert torch.allclose(features, recovered_features, atol=1e-6)
    assert torch.allclose(positions, recovered_positions, atol=1e-6)
    assert (batch_idx == recovered_batch_idx).all()

    print("   ✓ Varying batch sizes work")

    print("\n✅ PASSED: Edge cases handled correctly")
    return True


if __name__ == "__main__":
    print("\n🧪 Testing Sequence Conversion Utilities")
    print("="*70)

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")

    results = []

    try:
        results.append(("Round-trip conversion", test_round_trip_conversion()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Round-trip conversion", False))
        import traceback
        traceback.print_exc()

    try:
        results.append(("Edge cases", test_empty_batches()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Edge cases", False))
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<50} {status}")
    print("="*70)

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n🎉 All tests passed! sequence_to_pyg works correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review.")
