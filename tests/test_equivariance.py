"""
Test SE(3) equivariance of FlowFix models.

SE(3) equivariance means:
- Rotation equivariance: R(f(x)) = f(R(x))
- Translation invariance: f(x + t) = f(x) (for scalars) or f(x + t) + t (for vectors)

For flow matching models, velocity predictions should be rotation equivariant:
- v(R(x)) = R(v(x))
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.flowmatching import ProteinLigandFlowMatching
from torch_geometric.data import Data, Batch


def random_rotation_matrix():
    """Generate a random 3D rotation matrix using QR decomposition."""
    M = torch.randn(3, 3)
    Q, R = torch.linalg.qr(M)
    # Ensure det(Q) = 1 (proper rotation)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def apply_rotation_to_graph(data, R):
    """Apply rotation R to graph coordinates and vector features."""
    data_rotated = data.clone()
    # Rotate positions
    data_rotated.pos = (R @ data.pos.T).T

    # Rotate node vector features if present
    if hasattr(data, 'node_vector_features') and data.node_vector_features is not None:
        N, vector_dim, _ = data.node_vector_features.shape
        vectors_flat = data.node_vector_features.reshape(-1, 3)
        vectors_rotated = (R @ vectors_flat.T).T
        data_rotated.node_vector_features = vectors_rotated.reshape(N, vector_dim, 3)

    # Rotate edge vector features if present
    if hasattr(data, 'edge_vector_features') and data.edge_vector_features is not None:
        E, edge_vector_dim, _ = data.edge_vector_features.shape
        edge_vectors_flat = data.edge_vector_features.reshape(-1, 3)
        edge_vectors_rotated = (R @ edge_vectors_flat.T).T
        data_rotated.edge_vector_features = edge_vectors_rotated.reshape(E, edge_vector_dim, 3)

    return data_rotated


def create_random_protein_graph(num_nodes=30):
    """Create a random protein graph for testing (PyG batch format)."""
    pos = torch.randn(num_nodes, 3)
    x = torch.randn(num_nodes, 76)
    node_vector_features = torch.randn(num_nodes, 31, 3)

    # Create edges (k-NN)
    k = 10
    dist = torch.cdist(pos, pos)
    _, edge_index_rows = torch.topk(dist, k + 1, largest=False, dim=1)
    edge_index = []
    for i in range(num_nodes):
        for j in edge_index_rows[i]:
            if i != j:
                edge_index.append([i, j.item()])
    edge_index = torch.tensor(edge_index).T

    edge_attr = torch.randn(edge_index.size(1), 39)
    edge_vector_features = torch.randn(edge_index.size(1), 8, 3)

    return Data(
        pos=pos,
        x=x,
        node_vector_features=node_vector_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_vector_features=edge_vector_features
    )


def create_random_ligand_graph(num_nodes=20):
    """Create a random ligand graph for testing (PyG batch format)."""
    pos = torch.randn(num_nodes, 3)
    x = torch.randn(num_nodes, 122)

    # Create edges (k-NN)
    k = 5
    dist = torch.cdist(pos, pos)
    _, edge_index_rows = torch.topk(dist, k + 1, largest=False, dim=1)
    edge_index = []
    for i in range(num_nodes):
        for j in edge_index_rows[i]:
            if i != j:
                edge_index.append([i, j.item()])
    edge_index = torch.tensor(edge_index).T

    edge_attr = torch.randn(edge_index.size(1), 44)

    return Data(
        pos=pos,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )


def test_full_model_equivariance():
    """Test full ProteinLigandFlowMatching model SE(3) equivariance."""
    print("\n" + "="*70)
    print("Testing SE(3) Equivariance of ProteinLigandFlowMatching")
    print("="*70)

    device = torch.device('cuda')

    # Create model
    model = ProteinLigandFlowMatching(
        protein_input_scalar_dim=76,
        protein_input_vector_dim=31,
        protein_input_edge_scalar_dim=39,
        protein_input_edge_vector_dim=8,
        protein_hidden_scalar_dim=64,
        protein_hidden_vector_dim=16,
        protein_output_scalar_dim=64,
        protein_output_vector_dim=16,
        protein_num_layers=2,
        ligand_input_scalar_dim=122,
        ligand_input_edge_scalar_dim=44,
        ligand_hidden_scalar_dim=64,
        ligand_hidden_vector_dim=16,
        ligand_output_scalar_dim=64,
        ligand_output_vector_dim=16,
        ligand_num_layers=2,
        velocity_hidden_scalar_dim=64,
        velocity_hidden_vector_dim=16,
        velocity_num_layers=2,
        time_embed_dim=64
    )
    model = model.to(device)
    model.eval()

    # Create random rotation
    R = random_rotation_matrix().to(device)
    print(f"\nRotation matrix:\n{R}\n")
    print(f"det(R) = {torch.det(R):.6f} (should be 1.0)")

    # Create test data
    protein_graph = create_random_protein_graph(num_nodes=30)
    ligand_graph = create_random_ligand_graph(num_nodes=20)

    # Time
    t = torch.tensor([0.5], device=device)

    # Create batches
    protein_batch = Batch.from_data_list([protein_graph]).to(device)
    ligand_batch = Batch.from_data_list([ligand_graph]).to(device)

    # Rotate data (move to GPU first, then rotate)
    protein_graph_gpu = protein_graph.clone()
    protein_graph_gpu.pos = protein_graph.pos.to(device)
    protein_graph_gpu.node_vector_features = protein_graph.node_vector_features.to(device)
    protein_graph_gpu.edge_vector_features = protein_graph.edge_vector_features.to(device)

    ligand_graph_gpu = ligand_graph.clone()
    ligand_graph_gpu.pos = ligand_graph.pos.to(device)

    protein_batch_rot = Batch.from_data_list([apply_rotation_to_graph(protein_graph_gpu, R)]).to(device)
    ligand_batch_rot = Batch.from_data_list([apply_rotation_to_graph(ligand_graph_gpu, R)]).to(device)

    print(f"\nProtein nodes: {protein_batch.num_nodes}")
    print(f"Ligand nodes: {ligand_batch.num_nodes}")

    # Forward pass
    print("\nRunning forward pass on original data...")
    with torch.no_grad():
        velocity_orig = model(protein_batch, ligand_batch, t)

    print(f"  Velocity shape: {velocity_orig.shape}")
    print(f"  Velocity norm: {torch.norm(velocity_orig).item():.6f}")

    print("\nRunning forward pass on rotated data...")
    with torch.no_grad():
        velocity_rot = model(protein_batch_rot, ligand_batch_rot, t)

    print(f"  Velocity shape: {velocity_rot.shape}")
    print(f"  Velocity norm: {torch.norm(velocity_rot).item():.6f}")

    # Rotate original velocity
    print("\nRotating original velocity...")
    velocity_orig_rotated = (R @ velocity_orig.T).T

    print(f"  Rotated velocity norm: {torch.norm(velocity_orig_rotated).item():.6f}")

    # Check equivariance: R(v(x)) should equal v(R(x))
    velocity_error = torch.abs(velocity_orig_rotated - velocity_rot).max().item()
    velocity_error_mean = torch.abs(velocity_orig_rotated - velocity_rot).mean().item()

    print("\n" + "-"*70)
    print("EQUIVARIANCE TEST RESULTS")
    print("-"*70)
    print(f"Max absolute error:  {velocity_error:.10f}")
    print(f"Mean absolute error: {velocity_error_mean:.10f}")

    tolerance = 1e-4
    if velocity_error < tolerance:
        print(f"\n✅ PASSED: Model is SE(3)-equivariant (error < {tolerance})")
        print("="*70)
        return True
    else:
        print(f"\n❌ FAILED: Model is NOT SE(3)-equivariant (error >= {tolerance})")
        print("\nDetailed error analysis:")
        error_per_atom = torch.abs(velocity_orig_rotated - velocity_rot).norm(dim=1)
        print(f"  Max per-atom error: {error_per_atom.max().item():.10f}")
        print(f"  Min per-atom error: {error_per_atom.min().item():.10f}")
        print(f"  Median per-atom error: {error_per_atom.median().item():.10f}")
        print("="*70)
        return False


def test_translation_invariance():
    """Test that scalar outputs are translation invariant."""
    print("\n" + "="*70)
    print("Testing Translation Invariance")
    print("="*70)

    device = torch.device('cuda')

    model = ProteinLigandFlowMatching(
        protein_input_scalar_dim=76,
        protein_input_vector_dim=31,
        protein_input_edge_scalar_dim=39,
        protein_input_edge_vector_dim=8,
        protein_hidden_scalar_dim=64,
        protein_hidden_vector_dim=16,
        protein_output_scalar_dim=64,
        protein_output_vector_dim=16,
        protein_num_layers=2,
        ligand_input_scalar_dim=122,
        ligand_input_edge_scalar_dim=44,
        ligand_hidden_scalar_dim=64,
        ligand_hidden_vector_dim=16,
        ligand_output_scalar_dim=64,
        ligand_output_vector_dim=16,
        ligand_num_layers=2,
        velocity_hidden_scalar_dim=64,
        velocity_hidden_vector_dim=16,
        velocity_num_layers=2,
        time_embed_dim=64
    )
    model = model.to(device)
    model.eval()

    # Create test data
    protein_graph = create_random_protein_graph(num_nodes=30)
    ligand_graph = create_random_ligand_graph(num_nodes=20)

    # Translation vector
    translation = torch.randn(3, device=device)
    print(f"\nTranslation vector: {translation}")

    # Translate data
    protein_graph_trans = protein_graph.clone()
    protein_graph_trans.pos = protein_graph.pos + translation.cpu()

    ligand_graph_trans = ligand_graph.clone()
    ligand_graph_trans.pos = ligand_graph.pos + translation.cpu()

    # Create batches
    protein_batch = Batch.from_data_list([protein_graph]).to(device)
    ligand_batch = Batch.from_data_list([ligand_graph]).to(device)

    protein_batch_trans = Batch.from_data_list([protein_graph_trans]).to(device)
    ligand_batch_trans = Batch.from_data_list([ligand_graph_trans]).to(device)

    # Time
    t = torch.tensor([0.5], device=device)

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        velocity_orig = model(protein_batch, ligand_batch, t)
        velocity_trans = model(protein_batch_trans, ligand_batch_trans, t)

    # Check that velocities are the same (translation invariant)
    velocity_error = torch.abs(velocity_orig - velocity_trans).max().item()
    velocity_error_mean = torch.abs(velocity_orig - velocity_trans).mean().item()

    print("\n" + "-"*70)
    print("TRANSLATION INVARIANCE TEST RESULTS")
    print("-"*70)
    print(f"Max absolute error:  {velocity_error:.10f}")
    print(f"Mean absolute error: {velocity_error_mean:.10f}")

    tolerance = 1e-4
    if velocity_error < tolerance:
        print(f"\n✅ PASSED: Model is translation-invariant (error < {tolerance})")
        print("="*70)
        return True
    else:
        print(f"\n❌ FAILED: Model is NOT translation-invariant (error >= {tolerance})")
        print("="*70)
        return False


if __name__ == "__main__":
    print("\n🔬 FlowFix SE(3) Equivariance Tests\n")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        print("   cuEquivariance requires CUDA to run.")
        print("   Please run tests on a GPU-enabled machine.\n")
        sys.exit(1)

    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}\n")

    # Run tests
    results = []

    print("\nTest 1: Rotation Equivariance")
    results.append(("Rotation Equivariance", test_full_model_equivariance()))

    print("\nTest 2: Translation Invariance")
    results.append(("Translation Invariance", test_translation_invariance()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    print("="*70 + "\n")

    # Exit code
    all_passed = all(passed for _, passed in results)
    sys.exit(0 if all_passed else 1)
