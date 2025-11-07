"""
Test that ProteinLigandInteractionNetwork returns node-level features in PyG format.
"""
import torch
import sys
from pathlib import Path
from torch_geometric.data import Data, Batch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.network import ProteinLigandInteractionNetwork, UnifiedEquivariantNetwork


def test_node_feature_shapes():
    """Test that interaction network returns PyG node features, not padded sequences."""
    print("\n" + "="*70)
    print("Test: Interaction Network Node Feature Format")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create networks
    protein_network = UnifiedEquivariantNetwork(
        input_scalar_dim=76,
        input_vector_dim=31,
        input_edge_scalar_dim=39,
        input_edge_vector_dim=8,
        hidden_scalar_dim=128,
        hidden_vector_dim=32,
        output_scalar_dim=128,
        output_vector_dim=32,
        num_layers=2,
        use_time_conditioning=True,
        time_embed_dim=128
    ).to(device)

    ligand_network = UnifiedEquivariantNetwork(
        input_scalar_dim=121,
        input_vector_dim=0,
        input_edge_scalar_dim=44,
        input_edge_vector_dim=0,
        hidden_scalar_dim=128,
        hidden_vector_dim=16,
        output_scalar_dim=128,
        output_vector_dim=16,
        num_layers=2,
        use_time_conditioning=True,
        time_embed_dim=128
    ).to(device)

    interaction_network = ProteinLigandInteractionNetwork(
        protein_output_irreps="128x0e + 32x1o + 32x1e",
        ligand_output_irreps="128x0e + 16x1o + 16x1e",
        hidden_dim=256,
        num_heads=8,
        pair_dim=16,
        num_layers=2,
        time_embed_dim=128
    ).to(device)

    # Create test data with variable sizes
    batch_size = 3
    protein_nodes = [50, 40, 60]
    ligand_nodes = [20, 25, 18]

    # Protein batch
    protein_data_list = []
    for i in range(batch_size):
        n_nodes = protein_nodes[i]
        n_edges = n_nodes * 8
        protein_data = Data(
            x=torch.randn(n_nodes, 76, device=device),
            pos=torch.randn(n_nodes, 3, device=device),
            edge_index=torch.randint(0, n_nodes, (2, n_edges), device=device),
            edge_attr=torch.randn(n_edges, 39, device=device),
            node_vector_features=torch.randn(n_nodes, 31, 3, device=device),
            edge_vector_features=torch.randn(n_edges, 8, 3, device=device)
        )
        protein_data_list.append(protein_data)
    protein_batch = Batch.from_data_list(protein_data_list).to(device)

    # Ligand batch
    ligand_data_list = []
    for i in range(batch_size):
        n_nodes = ligand_nodes[i]
        n_edges = n_nodes * 6
        ligand_data = Data(
            x=torch.randn(n_nodes, 121, device=device),
            pos=torch.randn(n_nodes, 3, device=device),
            edge_index=torch.randint(0, n_nodes, (2, n_edges), device=device),
            edge_attr=torch.randn(n_edges, 44, device=device)
        )
        ligand_data_list.append(ligand_data)
    ligand_batch = Batch.from_data_list(ligand_data_list).to(device)

    print(f"\n1. Input sizes:")
    print(f"   Protein nodes per batch: {protein_nodes}")
    print(f"   Ligand nodes per batch: {ligand_nodes}")
    print(f"   Total protein nodes: {sum(protein_nodes)}")
    print(f"   Total ligand nodes: {sum(ligand_nodes)}")
    print(f"   Total nodes: {sum(protein_nodes) + sum(ligand_nodes)}")

    # Time embedding
    time_emb = torch.randn(batch_size, 128, device=device)

    # Encode
    with torch.no_grad():
        protein_output = protein_network(protein_batch, time_condition=time_emb)
        ligand_output = ligand_network(ligand_batch, time_condition=time_emb)

        print(f"\n2. Encoder outputs (PyG format):")
        print(f"   Protein output: {protein_output.shape}")
        print(f"   Ligand output: {ligand_output.shape}")

        # Interaction network
        (prot_out, lig_out, interaction_out), (prot_global, lig_global, interaction_global), pair_bias = interaction_network(
            protein_output, ligand_output, protein_batch, ligand_batch, time_embed=time_emb
        )

    print(f"\n3. Interaction network outputs:")
    print(f"   Node features (should be PyG format):")
    print(f"     prot_out: {prot_out.shape}")
    print(f"     lig_out: {lig_out.shape}")
    print(f"     interaction_out: {interaction_out.shape}")
    print(f"\n   Global features (batch-level):")
    print(f"     prot_global: {prot_global.shape}")
    print(f"     lig_global: {lig_global.shape}")
    print(f"     interaction_global: {interaction_global.shape}")
    print(f"\n   Pair bias: {pair_bias.shape}")

    # Verify shapes
    total_protein = sum(protein_nodes)
    total_ligand = sum(ligand_nodes)
    total_nodes = total_protein + total_ligand

    print(f"\n4. Shape verification:")

    # Node features should be [total_nodes, D], NOT [B, max_N, D]
    assert prot_out.dim() == 2, f"prot_out should be 2D (node features), got {prot_out.dim()}D"
    assert lig_out.dim() == 2, f"lig_out should be 2D (node features), got {lig_out.dim()}D"
    assert interaction_out.dim() == 2, f"interaction_out should be 2D (node features), got {interaction_out.dim()}D"
    print("   ✓ Node features are 2D (not padded sequences)")

    assert prot_out.shape[0] == total_protein, f"Expected {total_protein} protein nodes, got {prot_out.shape[0]}"
    assert lig_out.shape[0] == total_ligand, f"Expected {total_ligand} ligand nodes, got {lig_out.shape[0]}"
    assert interaction_out.shape[0] == total_nodes, f"Expected {total_nodes} total nodes, got {interaction_out.shape[0]}"
    print(f"   ✓ Correct number of nodes: {total_protein} (prot) + {total_ligand} (lig) = {total_nodes} (total)")

    # Global features should be [B, D]
    assert prot_global.shape == (batch_size, 256), f"Expected prot_global [{batch_size}, 256], got {prot_global.shape}"
    assert lig_global.shape == (batch_size, 256), f"Expected lig_global [{batch_size}, 256], got {lig_global.shape}"
    assert interaction_global.shape == (batch_size, 256), f"Expected interaction_global [{batch_size}, 256], got {interaction_global.shape}"
    print(f"   ✓ Global features are batch-level [{batch_size}, 256]")

    # Pair bias should be [B, max_N, max_N, pair_dim]
    assert pair_bias.dim() == 4, f"Pair bias should be 4D, got {pair_bias.dim()}D"
    assert pair_bias.shape[0] == batch_size, f"Pair bias batch size should be {batch_size}, got {pair_bias.shape[0]}"
    print(f"   ✓ Pair bias shape: {pair_bias.shape}")

    print("\n✅ PASSED: Interaction network returns PyG node features correctly")
    return True


def test_node_feature_correspondence():
    """Test that node features correspond to correct batch assignments."""
    print("\n" + "="*70)
    print("Test: Node Feature Batch Correspondence")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simple 2-batch test
    protein_network = UnifiedEquivariantNetwork(
        input_scalar_dim=76,
        input_vector_dim=31,
        input_edge_scalar_dim=39,
        input_edge_vector_dim=8,
        hidden_scalar_dim=64,
        hidden_vector_dim=16,
        output_scalar_dim=64,
        output_vector_dim=16,
        num_layers=2,
        use_time_conditioning=True,
        time_embed_dim=64
    ).to(device)

    ligand_network = UnifiedEquivariantNetwork(
        input_scalar_dim=121,
        input_vector_dim=0,
        input_edge_scalar_dim=44,
        input_edge_vector_dim=0,
        hidden_scalar_dim=64,
        hidden_vector_dim=8,
        output_scalar_dim=64,
        output_vector_dim=8,
        num_layers=2,
        use_time_conditioning=True,
        time_embed_dim=64
    ).to(device)

    interaction_network = ProteinLigandInteractionNetwork(
        protein_output_irreps="64x0e + 16x1o + 16x1e",
        ligand_output_irreps="64x0e + 8x1o + 8x1e",
        hidden_dim=128,
        num_heads=4,
        pair_dim=16,
        num_layers=2,
        time_embed_dim=64
    ).to(device)

    # Create data: batch 0 has 10 protein + 5 ligand, batch 1 has 15 protein + 8 ligand
    protein_data = [
        Data(x=torch.randn(10, 76, device=device),
             pos=torch.randn(10, 3, device=device),
             edge_index=torch.randint(0, 10, (2, 40), device=device),
             edge_attr=torch.randn(40, 39, device=device),
             node_vector_features=torch.randn(10, 31, 3, device=device),
             edge_vector_features=torch.randn(40, 8, 3, device=device)),
        Data(x=torch.randn(15, 76, device=device),
             pos=torch.randn(15, 3, device=device),
             edge_index=torch.randint(0, 15, (2, 60), device=device),
             edge_attr=torch.randn(60, 39, device=device),
             node_vector_features=torch.randn(15, 31, 3, device=device),
             edge_vector_features=torch.randn(60, 8, 3, device=device))
    ]
    protein_batch = Batch.from_data_list(protein_data).to(device)

    ligand_data = [
        Data(x=torch.randn(5, 121, device=device),
             pos=torch.randn(5, 3, device=device),
             edge_index=torch.randint(0, 5, (2, 20), device=device),
             edge_attr=torch.randn(20, 44, device=device)),
        Data(x=torch.randn(8, 121, device=device),
             pos=torch.randn(8, 3, device=device),
             edge_index=torch.randint(0, 8, (2, 32), device=device),
             edge_attr=torch.randn(32, 44, device=device))
    ]
    ligand_batch = Batch.from_data_list(ligand_data).to(device)

    print(f"\n1. Batch structure:")
    print(f"   Batch 0: 10 protein + 5 ligand = 15 nodes")
    print(f"   Batch 1: 15 protein + 8 ligand = 23 nodes")
    print(f"   Total: 25 protein + 13 ligand = 38 nodes")

    # Process
    time_emb = torch.randn(2, 64, device=device)
    with torch.no_grad():
        protein_output = protein_network(protein_batch, time_condition=time_emb)
        ligand_output = ligand_network(ligand_batch, time_condition=time_emb)

        (prot_out, lig_out, interaction_out), _, _ = interaction_network(
            protein_output, ligand_output, protein_batch, ligand_batch, time_embed=time_emb
        )

    print(f"\n2. Output shapes:")
    print(f"   prot_out: {prot_out.shape} (expected [25, D])")
    print(f"   lig_out: {lig_out.shape} (expected [13, D])")
    print(f"   interaction_out: {interaction_out.shape} (expected [38, D])")

    # Verify
    assert prot_out.shape[0] == 25, "Should have 25 protein nodes"
    assert lig_out.shape[0] == 13, "Should have 13 ligand nodes"
    assert interaction_out.shape[0] == 38, "Should have 38 total nodes"

    print("\n✅ PASSED: Node features have correct batch correspondence")
    return True


if __name__ == "__main__":
    print("\n🧪 Testing Interaction Network Node Features")
    print("="*70)

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")

    results = []

    try:
        results.append(("Node feature shapes", test_node_feature_shapes()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Node feature shapes", False))
        import traceback
        traceback.print_exc()

    try:
        results.append(("Node feature correspondence", test_node_feature_correspondence()))
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(("Node feature correspondence", False))
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
        print("\n🎉 All tests passed! Node features are in PyG format.")
    else:
        print("\n⚠️  Some tests failed. Please review.")
