"""
Simple check for pocket vector features in loaded data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path


def main():
    print("="*70)
    print("Checking Pocket Vector Features in Dataset")
    print("="*70)

    # Find available data directory
    for data_dir_name in ['train_data_dg', 'train_data']:
        data_dir = Path(data_dir_name)
        if data_dir.exists():
            print(f"\n✅ Found data directory: {data_dir_name}")
            break
    else:
        print("\n❌ No data directory found!")
        return

    # Load one protein file manually
    pdbs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])[:1]
    if not pdbs:
        print("❌ No PDB directories found!")
        return

    pdb_id = pdbs[0]
    protein_path = data_dir / pdb_id / "protein.pt"

    if not protein_path.exists():
        print(f"❌ protein.pt not found for {pdb_id}")
        return

    print(f"\n📂 Loading protein: {pdb_id}")
    protein_data = torch.load(protein_path, weights_only=False)

    # Check structure
    print(f"\nProtein data type: {type(protein_data)}")

    if isinstance(protein_data, dict):
        print(f"Keys: {protein_data.keys()}")

        if 'node' in protein_data:
            node_data = protein_data['node']
            print(f"\nNode keys: {node_data.keys()}")

            # Check vector features
            if 'node_vector_features' in node_data:
                vec_feat = node_data['node_vector_features']
                if isinstance(vec_feat, tuple):
                    print(f"\n✅ node_vector_features found (tuple with {len(vec_feat)} elements)")
                    for i, v in enumerate(vec_feat):
                        if isinstance(v, torch.Tensor):
                            print(f"    [{i}] shape: {v.shape}")
                else:
                    print(f"\n✅ node_vector_features found: {vec_feat.shape}")
            else:
                print(f"\n❌ node_vector_features NOT found")

        if 'edge' in protein_data:
            edge_data = protein_data['edge']
            print(f"\nEdge keys: {edge_data.keys()}")

            # Check edge vector features
            if 'edge_vector_features' in edge_data:
                edge_vec_feat = edge_data['edge_vector_features']
                if isinstance(edge_vec_feat, tuple):
                    print(f"\n✅ edge_vector_features found (tuple with {len(edge_vec_feat)} elements)")
                    for i, v in enumerate(edge_vec_feat):
                        if isinstance(v, torch.Tensor):
                            print(f"    [{i}] shape: {v.shape}")
                else:
                    print(f"\n✅ edge_vector_features found: {edge_vec_feat.shape}")
            else:
                print(f"\n❌ edge_vector_features NOT found")

    # Now test with dataset
    print("\n" + "="*70)
    print("Testing with FlowFixDataset")
    print("="*70)

    from src.data.dataset import FlowFixDataset

    dataset = FlowFixDataset(
        data_dir=str(data_dir),
        split="train",
        max_samples=1,
        seed=42
    )

    if len(dataset) == 0:
        print("❌ No samples in dataset")
        return

    sample = dataset[0]
    if sample is None:
        print("❌ Sample is None")
        return

    print(f"\n✅ Loaded sample: {sample['pdb_id']}")

    # Check protein graph
    protein_graph = sample['protein_graph']
    print(f"\nProtein Graph attributes:")
    for attr in ['x', 'pos', 'edge_index', 'edge_attr', 'node_vector_features', 'edge_vector_features']:
        if hasattr(protein_graph, attr):
            val = getattr(protein_graph, attr)
            if val is not None:
                print(f"  ✅ {attr}: {val.shape}")
            else:
                print(f"  ⚠️  {attr}: None")
        else:
            print(f"  ❌ {attr}: Not present")

    # Final check
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    has_node_vec = hasattr(protein_graph, 'node_vector_features') and protein_graph.node_vector_features is not None
    has_edge_vec = hasattr(protein_graph, 'edge_vector_features') and protein_graph.edge_vector_features is not None

    if has_node_vec and has_edge_vec:
        print("✅ SUCCESS: Pocket vector features are properly extracted!")
        print(f"   - Node vectors: {protein_graph.node_vector_features.shape}")
        print(f"   - Edge vectors: {protein_graph.edge_vector_features.shape}")
    elif has_node_vec:
        print("⚠️  PARTIAL: Node vectors present, but edge vectors missing")
    elif has_edge_vec:
        print("⚠️  PARTIAL: Edge vectors present, but node vectors missing")
    else:
        print("❌ FAIL: No vector features found in pocket!")


if __name__ == "__main__":
    main()
