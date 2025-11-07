"""
Test unified network to verify it works for both protein and ligand cases.
"""
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.network import UnifiedEquivariantNetwork


def create_dummy_batch(num_nodes, num_edges, scalar_dim, vector_dim=0, edge_scalar_dim=44, edge_vector_dim=0, batch_size=2):
    """Create dummy PyG-like batch for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_data = type('Batch', (), {})()
    batch_data.x = torch.randn(num_nodes, scalar_dim, device=device)
    batch_data.pos = torch.randn(num_nodes, 3, device=device)
    batch_data.edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    batch_data.edge_attr = torch.randn(num_edges, edge_scalar_dim, device=device)
    batch_data.batch = torch.repeat_interleave(torch.arange(batch_size, device=device), num_nodes // batch_size)

    if vector_dim > 0:
        batch_data.node_vector_features = torch.randn(num_nodes, vector_dim, 3, device=device)

    if edge_vector_dim > 0:
        batch_data.edge_vector_features = torch.randn(num_edges, edge_vector_dim, 3, device=device)

    return batch_data


def test_ligand_scalar_only():
    """Test ligand network (scalar-only input)."""
    print("\n" + "="*70)
    print("Test 1: Ligand Network (Scalar-only)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create ligand network (scalar-only input)
    network = UnifiedEquivariantNetwork(
        input_scalar_dim=121,
        input_vector_dim=0,  # No vectors
        input_edge_scalar_dim=44,
        input_edge_vector_dim=0,
        hidden_scalar_dim=128,
        hidden_vector_dim=16,
        output_scalar_dim=128,
        output_vector_dim=16,
        num_layers=3,
        use_time_conditioning=False
    ).to(device)
    network.eval()

    print(f"Input irreps: {network.in_irreps}")
    print(f"Hidden irreps: {network.hidden_irreps}")
    print(f"Output irreps: {network.out_irreps}")

    # Create dummy data (scalar-only)
    batch = create_dummy_batch(
        num_nodes=20,
        num_edges=60,
        scalar_dim=121,
        vector_dim=0,  # No vectors
        edge_scalar_dim=44
    )

    # Forward pass
    with torch.no_grad():
        output = network(batch)

    print(f"\nInput shape: {batch.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output dim: {network.out_irreps.dim}")

    assert output.shape[0] == 20, "Wrong batch dimension"
    assert output.shape[1] == network.out_irreps.dim, "Wrong feature dimension"

    print("✅ PASSED: Ligand network (scalar-only) works!")
    return True


def test_protein_with_vectors():
    """Test protein network (with vector features)."""
    print("\n" + "="*70)
    print("Test 2: Protein Network (With Vectors)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create protein network (with vector features)
    network = UnifiedEquivariantNetwork(
        input_scalar_dim=76,
        input_vector_dim=31,  # Has vectors
        input_edge_scalar_dim=39,
        input_edge_vector_dim=8,
        hidden_scalar_dim=128,
        hidden_vector_dim=32,
        output_scalar_dim=128,
        output_vector_dim=32,
        num_layers=4,
        use_time_conditioning=False
    ).to(device)
    network.eval()

    print(f"Input irreps: {network.in_irreps}")
    print(f"Hidden irreps: {network.hidden_irreps}")
    print(f"Output irreps: {network.out_irreps}")

    # Create dummy data (with vectors)
    batch = create_dummy_batch(
        num_nodes=50,
        num_edges=150,
        scalar_dim=76,
        vector_dim=31,  # Has vectors!
        edge_scalar_dim=39,
        edge_vector_dim=8
    )

    # Forward pass
    with torch.no_grad():
        output = network(batch)

    print(f"\nInput scalar shape: {batch.x.shape}")
    print(f"Input vector shape: {batch.node_vector_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output dim: {network.out_irreps.dim}")

    assert output.shape[0] == 50, "Wrong batch dimension"
    assert output.shape[1] == network.out_irreps.dim, "Wrong feature dimension"

    print("✅ PASSED: Protein network (with vectors) works!")
    return True


def test_time_conditioning():
    """Test time conditioning."""
    print("\n" + "="*70)
    print("Test 3: Time Conditioning")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create network with time conditioning
    network = UnifiedEquivariantNetwork(
        input_scalar_dim=64,
        input_vector_dim=0,
        input_edge_scalar_dim=44,  # Match edge dimension
        hidden_scalar_dim=128,
        hidden_vector_dim=16,
        num_layers=2,
        use_time_conditioning=True,
        time_embed_dim=64
    ).to(device)
    network.eval()

    batch = create_dummy_batch(
        num_nodes=30,
        num_edges=90,
        scalar_dim=64,
        edge_scalar_dim=44,  # Match network
        batch_size=3
    )

    # Create time conditioning
    time_condition = torch.randn(3, 64, device=device)

    # Forward pass with time
    with torch.no_grad():
        output = network(batch, time_condition=time_condition)

    print(f"Time condition shape: {time_condition.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape[0] == 30, "Wrong batch dimension"

    print("✅ PASSED: Time conditioning works!")
    return True


def test_flexible_dimensions():
    """Test custom dimensions."""
    print("\n" + "="*70)
    print("Test 4: Flexible Custom Dimensions")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create network with custom dimensions
    network = UnifiedEquivariantNetwork(
        input_scalar_dim=100,
        input_vector_dim=10,  # Custom vector count
        input_edge_scalar_dim=50,  # Custom edge dimension
        hidden_scalar_dim=256,
        hidden_vector_dim=64,
        output_scalar_dim=64,
        output_vector_dim=32,
        num_layers=5,
        use_time_conditioning=False
    ).to(device)
    network.eval()

    batch = create_dummy_batch(
        num_nodes=40,
        num_edges=120,
        scalar_dim=100,
        vector_dim=10,
        edge_scalar_dim=50  # Match network
    )

    with torch.no_grad():
        output = network(batch)

    print(f"Input irreps: {network.in_irreps}")
    print(f"Output irreps: {network.out_irreps}")
    print(f"Output shape: {output.shape}")

    # Output should be 64 scalars + 32*3 vectors (1o) + 32*3 vectors (1e)
    expected_dim = 64 + 32*3 + 32*3
    assert output.shape[1] == expected_dim, f"Expected {expected_dim}, got {output.shape[1]}"

    print("✅ PASSED: Flexible dimensions work!")
    return True


def test_unified_network_direct():
    """Test direct UnifiedEquivariantNetwork instantiation for protein/ligand."""
    print("\n" + "="*70)
    print("Test 5: Direct Network Instantiation")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create protein-style network (with vectors)
    protein = UnifiedEquivariantNetwork(
        input_scalar_dim=76,
        input_vector_dim=31,
        input_edge_scalar_dim=39,
        input_edge_vector_dim=8,
        hidden_scalar_dim=128,
        hidden_vector_dim=32,
        output_scalar_dim=128,
        output_vector_dim=32,
        num_layers=4,
        use_time_conditioning=False
    ).to(device)

    # Create ligand-style network (scalar-only)
    ligand = UnifiedEquivariantNetwork(
        input_scalar_dim=121,
        input_vector_dim=0,
        input_edge_scalar_dim=44,
        input_edge_vector_dim=0,
        hidden_scalar_dim=128,
        hidden_vector_dim=16,
        output_scalar_dim=128,
        output_vector_dim=16,
        num_layers=3,
        use_time_conditioning=True,
        time_embed_dim=128
    ).to(device)

    print("Protein-style network (with vectors):")
    print(f"  in_irreps: {protein.in_irreps}")
    print(f"  out_irreps: {protein.out_irreps}")
    print(f"  num_layers: {protein.num_layers}")
    print(f"  Type: {type(protein).__name__}")

    print("\nLigand-style network (scalar-only):")
    print(f"  in_irreps: {ligand.in_irreps}")
    print(f"  out_irreps: {ligand.out_irreps}")
    print(f"  num_layers: {ligand.num_layers}")
    print(f"  Type: {type(ligand).__name__}")

    # Verify they are both UnifiedEquivariantNetwork instances
    assert type(protein).__name__ == 'UnifiedEquivariantNetwork', "Protein should be UnifiedEquivariantNetwork"
    assert type(ligand).__name__ == 'UnifiedEquivariantNetwork', "Ligand should be UnifiedEquivariantNetwork"

    # Verify expected irreps
    assert str(protein.in_irreps) == "76x0e+31x1o", "Protein in_irreps mismatch"
    assert str(ligand.in_irreps) == "121x0e", "Ligand in_irreps mismatch"

    print("\n✅ PASSED: Direct network instantiation works!")
    return True


if __name__ == "__main__":
    print("\n" + "🧪 Testing Unified Equivariant Network")
    print("="*70)

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")

    results = []

    try:
        results.append(("Ligand (scalar-only)", test_ligand_scalar_only()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Ligand (scalar-only)", False))

    try:
        results.append(("Protein (with vectors)", test_protein_with_vectors()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Protein (with vectors)", False))

    try:
        results.append(("Time conditioning", test_time_conditioning()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Time conditioning", False))

    try:
        results.append(("Flexible dimensions", test_flexible_dimensions()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Flexible dimensions", False))

    try:
        results.append(("Direct instantiation", test_unified_network_direct()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Direct instantiation", False))

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
        print("\n🎉 All tests passed! Unified network works correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review.")
