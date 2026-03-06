"""
Test script for the Joint Graph Architecture.
Verifies forward pass shapes and basic functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch_geometric.data import Data, Batch


def create_dummy_protein_batch(batch_size=2, device='cpu'):
    """Create dummy protein batch for testing."""
    graphs = []
    for i in range(batch_size):
        n_residues = 30 + i * 10  # Variable sizes
        n_edges = n_residues * 8  # ~8 edges per node

        graph = Data(
            x=torch.randn(n_residues, 76, device=device),
            pos=torch.randn(n_residues, 3, device=device) * 10,
            edge_index=torch.randint(0, n_residues, (2, n_edges), device=device),
            edge_attr=torch.randn(n_edges, 39, device=device),
            node_vector_features=torch.randn(n_residues, 31, 3, device=device),
            edge_vector_features=torch.randn(n_edges, 8, 3, device=device),
        )
        graphs.append(graph)

    return Batch.from_data_list(graphs)


def create_dummy_ligand_batch(batch_size=2, device='cpu'):
    """Create dummy ligand batch for testing."""
    graphs = []
    for i in range(batch_size):
        n_atoms = 20 + i * 5  # Variable sizes
        n_edges = n_atoms * 4  # ~4 edges per atom

        graph = Data(
            x=torch.randn(n_atoms, 122, device=device),
            pos=torch.randn(n_atoms, 3, device=device) * 5,
            edge_index=torch.randint(0, n_atoms, (2, n_edges), device=device),
            edge_attr=torch.randn(n_edges, 44, device=device),
        )
        graphs.append(graph)

    return Batch.from_data_list(graphs)


def test_build_cross_edges():
    """Test cross-edge construction."""
    from src.models.network import build_cross_edges

    protein_pos = torch.randn(30, 3) * 10
    ligand_pos = torch.randn(20, 3) * 5

    edge_index = build_cross_edges(
        protein_pos, ligand_pos,
        distance_cutoff=10.0, max_neighbors=8
    )

    print(f"Cross-edge construction:")
    print(f"  Protein nodes: 30, Ligand nodes: 20")
    print(f"  Cross edges: {edge_index.shape[1]} (bidirectional)")
    print(f"  Edge index shape: {edge_index.shape}")

    # Verify bidirectional: first half P→L, second half L→P
    n_half = edge_index.shape[1] // 2
    print(f"  P→L edges: {n_half}, L→P edges: {edge_index.shape[1] - n_half}")

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    print("  PASSED")


def test_build_intra_edges():
    """Test intra-edge construction."""
    from src.models.network import build_intra_edges

    n_nodes = 30
    pos = torch.randn(n_nodes, 3) * 10

    # Create some existing edges (simulating pre-computed backbone edges)
    existing_src = torch.arange(0, n_nodes - 1)
    existing_dst = torch.arange(1, n_nodes)
    existing_edge_index = torch.stack([
        torch.cat([existing_src, existing_dst]),
        torch.cat([existing_dst, existing_src])
    ])  # Bidirectional sequential edges

    edge_index = build_intra_edges(
        pos, existing_edge_index,
        distance_cutoff=15.0, max_neighbors=8
    )

    print(f"\nIntra-edge construction:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Existing edges: {existing_edge_index.shape[1]}")
    print(f"  New intra edges: {edge_index.shape[1]}")
    print(f"  Edge index shape: {edge_index.shape}")

    assert edge_index.shape[0] == 2
    # Verify no overlap with existing edges
    if edge_index.shape[1] > 0:
        existing_keys = set((existing_edge_index[0].long() * n_nodes + existing_edge_index[1].long()).tolist())
        new_keys = (edge_index[0].long() * n_nodes + edge_index[1].long()).tolist()
        overlap = sum(1 for k in new_keys if k in existing_keys)
        print(f"  Overlap with existing: {overlap} (should be 0)")
        assert overlap == 0
    print("  PASSED")


def test_joint_network():
    """Test JointProteinLigandNetwork forward pass."""
    from src.models.network import JointProteinLigandNetwork

    device = 'cpu'
    batch_size = 2

    network = JointProteinLigandNetwork(
        protein_input_scalar_dim=76,
        protein_input_vector_dim=31,
        protein_input_edge_scalar_dim=39,
        protein_input_edge_vector_dim=8,
        ligand_input_scalar_dim=122,
        ligand_input_edge_scalar_dim=44,
        hidden_scalar_dim=32,  # Small for testing
        hidden_vector_dim=8,
        hidden_edge_dim=32,
        cross_edge_distance_cutoff=15.0,  # Larger cutoff for random data
        cross_edge_max_neighbors=8,
        cross_edge_num_rbf=16,
        intra_edge_distance_cutoff=15.0,
        intra_edge_max_neighbors=8,
        num_layers=2,  # Few layers for speed
        dropout=0.0,
        condition_dim=64,
    ).to(device)

    protein_batch = create_dummy_protein_batch(batch_size, device)
    ligand_batch = create_dummy_ligand_batch(batch_size, device)

    # Create dummy time condition [B, condition_dim]
    time_condition = torch.randn(batch_size, 64, device=device)

    print(f"\nJointProteinLigandNetwork forward pass:")
    print(f"  Protein nodes: {protein_batch.num_nodes}")
    print(f"  Ligand nodes: {ligand_batch.num_nodes}")

    velocity = network(protein_batch, ligand_batch, time_condition=time_condition)

    print(f"  velocity shape: {velocity.shape} (expected [{ligand_batch.num_nodes}, 3])")

    assert velocity.shape == (ligand_batch.num_nodes, 3)
    print("  PASSED")


def test_full_model():
    """Test ProteinLigandFlowMatchingJoint end-to-end."""
    from src.models.flowmatching import ProteinLigandFlowMatchingJoint

    device = 'cpu'
    batch_size = 2

    model = ProteinLigandFlowMatchingJoint(
        protein_input_scalar_dim=76,
        protein_input_vector_dim=31,
        protein_input_edge_scalar_dim=39,
        protein_input_edge_vector_dim=8,
        ligand_input_scalar_dim=122,
        ligand_input_edge_scalar_dim=44,
        hidden_scalar_dim=32,
        hidden_vector_dim=8,
        hidden_edge_dim=32,
        cross_edge_distance_cutoff=15.0,
        cross_edge_max_neighbors=8,
        cross_edge_num_rbf=16,
        intra_edge_distance_cutoff=15.0,
        intra_edge_max_neighbors=8,
        joint_num_layers=4,
        hidden_dim=64,
        dropout=0.0,
        use_esm_embeddings=False,
    ).to(device)

    protein_batch = create_dummy_protein_batch(batch_size, device)
    ligand_batch = create_dummy_ligand_batch(batch_size, device)
    t = torch.rand(batch_size, device=device)

    print(f"\nProteinLigandFlowMatchingJoint full forward pass:")
    print(f"  Protein nodes: {protein_batch.num_nodes}")
    print(f"  Ligand nodes: {ligand_batch.num_nodes}")
    print(f"  Batch size: {batch_size}")

    velocity = model(protein_batch, ligand_batch, t)

    print(f"  Velocity shape: {velocity.shape} (expected [{ligand_batch.num_nodes}, 3])")
    assert velocity.shape == (ligand_batch.num_nodes, 3)

    # Test backward pass
    loss = velocity.pow(2).mean()
    loss.backward()
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Backward pass: OK")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
    print("  PASSED")


def test_model_builder():
    """Test model builder with joint architecture config."""
    from src.utils.model_builder import build_model

    config = {
        'architecture': 'joint',
        'protein_input_scalar_dim': 76,
        'protein_input_vector_dim': 31,
        'protein_input_edge_scalar_dim': 39,
        'protein_input_edge_vector_dim': 8,
        'ligand_input_scalar_dim': 122,
        'ligand_input_edge_scalar_dim': 44,
        'hidden_scalar_dim': 32,
        'hidden_vector_dim': 8,
        'hidden_edge_dim': 32,
        'cross_edge_distance_cutoff': 15.0,
        'cross_edge_max_neighbors': 8,
        'cross_edge_num_rbf': 16,
        'intra_edge_distance_cutoff': 15.0,
        'intra_edge_max_neighbors': 8,
        'joint_num_layers': 4,
        'hidden_dim': 64,
        'dropout': 0.0,
        'use_esm_embeddings': False,
    }

    device = 'cpu'
    model = build_model(config, device)

    print(f"\nModel builder test:")
    print(f"  Model type: {type(model).__name__}")
    assert type(model).__name__ == 'ProteinLigandFlowMatchingJoint'
    print("  PASSED")


def test_esm_integration():
    """Test ESM gated concatenation integration."""
    from src.models.flowmatching import ProteinLigandFlowMatchingJoint

    device = 'cpu'
    batch_size = 2

    model = ProteinLigandFlowMatchingJoint(
        protein_input_scalar_dim=76,
        protein_input_vector_dim=31,
        protein_input_edge_scalar_dim=39,
        protein_input_edge_vector_dim=8,
        ligand_input_scalar_dim=122,
        ligand_input_edge_scalar_dim=44,
        hidden_scalar_dim=32,
        hidden_vector_dim=8,
        hidden_edge_dim=32,
        cross_edge_distance_cutoff=15.0,
        cross_edge_max_neighbors=8,
        cross_edge_num_rbf=16,
        intra_edge_distance_cutoff=15.0,
        intra_edge_max_neighbors=8,
        joint_num_layers=4,
        hidden_dim=64,
        dropout=0.0,
        use_esm_embeddings=True,
        esmc_dim=1152,
        esm3_dim=1536,
        esm_proj_dim=64,  # Small for testing
    ).to(device)

    protein_batch = create_dummy_protein_batch(batch_size, device)
    ligand_batch = create_dummy_ligand_batch(batch_size, device)
    t = torch.rand(batch_size, device=device)

    # Add dummy ESM embeddings
    n_protein = protein_batch.num_nodes
    protein_batch.esmc_embeddings = torch.randn(n_protein, 1152, device=device)
    protein_batch.esm3_embeddings = torch.randn(n_protein, 1536, device=device)

    print(f"\nESM gated concatenation test:")
    print(f"  Protein nodes: {n_protein}")
    print(f"  ESM proj dim: 64")
    print(f"  Effective protein scalar dim: 76 + 64 = 140")

    velocity = model(protein_batch, ligand_batch, t)

    print(f"  Velocity shape: {velocity.shape} (expected [{ligand_batch.num_nodes}, 3])")
    assert velocity.shape == (ligand_batch.num_nodes, 3)

    # Test backward
    loss = velocity.pow(2).mean()
    loss.backward()
    print(f"  Backward pass: OK")

    # Verify ESM gate is in computation graph (gradients assigned)
    gate_has_grad = all(p.grad is not None for p in model.esm_gate.parameters())
    print(f"  ESM gate gradients: {'OK' if gate_has_grad else 'MISSING'}")
    assert gate_has_grad

    # Test without ESM embeddings (should zero-pad)
    model.zero_grad()
    protein_batch2 = create_dummy_protein_batch(batch_size, device)
    velocity2 = model(protein_batch2, ligand_batch, t)
    assert velocity2.shape == (ligand_batch.num_nodes, 3)
    print(f"  Without ESM (zero-pad): OK")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Joint Graph Architecture")
    print("=" * 60)

    test_build_cross_edges()
    test_build_intra_edges()
    test_joint_network()
    test_full_model()
    test_esm_integration()
    test_model_builder()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
