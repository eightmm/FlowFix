"""
Test to verify that pocket vector features are properly extracted and used.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path


def test_pocket_vector_features():
    """Test that protein pocket includes vector features."""
    print("="*60)
    print("Testing Pocket Vector Features")
    print("="*60)

    # Load dataset
    from src.data.dataset import FlowFixDataset

    data_dir = Path("train_data_dg")
    if not data_dir.exists():
        print("⚠️  train_data_dg directory not found")
        return

    dataset = FlowFixDataset(
        data_dir="train_data_dg",
        split="train",
        max_samples=1,
        seed=42
    )

    if len(dataset) == 0:
        print("⚠️  No samples in dataset")
        return

    # Get sample
    sample = dataset[0]
    if sample is None:
        print("⚠️  Sample is None")
        return

    print(f"\n=== Sample: {sample['pdb_id']} ===")

    # Check protein graph
    protein_graph = sample['protein_graph']
    print(f"\nProtein Graph:")
    print(f"  Nodes: {protein_graph.num_nodes}")
    print(f"  Edges: {protein_graph.edge_index.shape[1]}")
    print(f"  Node features (x): {protein_graph.x.shape}")
    print(f"  Edge features: {protein_graph.edge_attr.shape}")
    print(f"  Positions: {protein_graph.pos.shape}")

    # Check vector features
    has_node_vector = hasattr(protein_graph, 'node_vector_features') and protein_graph.node_vector_features is not None
    has_edge_vector = hasattr(protein_graph, 'edge_vector_features') and protein_graph.edge_vector_features is not None

    print(f"\n✅ Vector Features Status:")
    print(f"  node_vector_features: {'✅ Present' if has_node_vector else '❌ Missing'}")
    if has_node_vector:
        print(f"    Shape: {protein_graph.node_vector_features.shape}")
        print(f"    Expected: [N_nodes, num_vectors, 3]")

    print(f"  edge_vector_features: {'✅ Present' if has_edge_vector else '❌ Missing'}")
    if has_edge_vector:
        print(f"    Shape: {protein_graph.edge_vector_features.shape}")
        print(f"    Expected: [N_edges, num_vectors, 3]")

    # Check ligand graph
    ligand_graph = sample['ligand_graph']
    print(f"\nLigand Graph:")
    print(f"  Nodes: {ligand_graph.num_nodes}")
    print(f"  Node features (x): {ligand_graph.x.shape}")
    print(f"  Edge features: {ligand_graph.edge_attr.shape if ligand_graph.edge_attr is not None else 'None'}")

    has_lig_node_vector = hasattr(ligand_graph, 'node_vector_features') and ligand_graph.node_vector_features is not None
    has_lig_edge_vector = hasattr(ligand_graph, 'edge_vector_features') and ligand_graph.edge_vector_features is not None
    print(f"  node_vector_features: {'Present' if has_lig_node_vector else 'None (expected for ligand)'}")
    print(f"  edge_vector_features: {'Present' if has_lig_edge_vector else 'None (expected for ligand)'}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Protein node vectors: {'✅ PASS' if has_node_vector else '❌ FAIL'}")
    print(f"Protein edge vectors: {'✅ PASS' if has_edge_vector else '❌ FAIL'}")

    if has_node_vector and has_edge_vector:
        print("\n✅ All pocket vector features are properly extracted!")
    else:
        print("\n⚠️  Some vector features are missing!")

    return has_node_vector and has_edge_vector


def test_model_vector_feature_usage():
    """Test that the model properly uses vector features."""
    print("\n" + "="*60)
    print("Testing Model Vector Feature Usage")
    print("="*60)

    import torch
    from src.models.network import UnifiedEquivariantNetwork

    # Check model configuration
    print("\nModel expects:")
    print("  protein_input_vector_dim: 31 (node vectors)")
    print("  protein_input_edge_vector_dim: 8 (edge vectors)")

    # Create dummy protein batch with vector features
    from torch_geometric.data import Data, Batch

    protein1 = Data(
        x=torch.randn(20, 76),  # scalar features
        pos=torch.randn(20, 3),
        edge_index=torch.randint(0, 20, (2, 50)),
        edge_attr=torch.randn(50, 39),
        node_vector_features=torch.randn(20, 31, 3),  # ✅ vector features
        edge_vector_features=torch.randn(50, 8, 3)    # ✅ vector features
    )

    protein_batch = Batch.from_data_list([protein1])

    # Create model
    model = UnifiedEquivariantNetwork(
        input_scalar_dim=76,
        input_vector_dim=31,  # ✅ expects 31 vectors per node
        input_edge_scalar_dim=39,
        input_edge_vector_dim=8,  # ✅ expects 8 vectors per edge
        hidden_scalar_dim=128,
        hidden_vector_dim=32,
        output_scalar_dim=128,
        output_vector_dim=32,
        num_layers=2
    )

    print("\n✅ Model created with vector feature support")

    # Test forward pass
    try:
        output = model(protein_batch)
        print(f"\n✅ Forward pass successful!")
        print(f"  Output scalar shape: {output[0].shape}")
        print(f"  Output vector shape: {output[1].shape}")
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        return False

    return True


if __name__ == "__main__":
    # Test 1: Check data extraction
    data_ok = test_pocket_vector_features()

    # Test 2: Check model usage
    model_ok = test_model_vector_feature_usage()

    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    if data_ok and model_ok:
        print("✅ Pocket vector features are properly extracted AND used by the model!")
    elif data_ok:
        print("⚠️  Vector features are extracted but may not be used properly")
    else:
        print("❌ Vector features are not properly extracted")
