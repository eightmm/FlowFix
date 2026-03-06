#!/usr/bin/env python
"""
Inspect generated data in detail to verify all fields are correct.
"""

import torch
from pathlib import Path


def inspect_sample():
    """Inspect a single sample in detail."""

    # Load one sample
    pdb_id = "2f7o"  # Smallest molecule (11 atoms)

    print("="*80)
    print(f"Inspecting Sample: {pdb_id}")
    print("="*80)

    # Load protein
    print("\n" + "="*80)
    print("PROTEIN DATA")
    print("="*80)

    protein_path = Path(f"test_data/{pdb_id}/protein.pt")
    protein_data = torch.load(protein_path, weights_only=False)

    print(f"\nTop-level keys: {list(protein_data.keys())}")

    # Node data
    if 'node' in protein_data:
        print("\n[Node Data]")
        node_data = protein_data['node']
        print(f"  Keys: {list(node_data.keys())}")

        if 'coord' in node_data:
            coords = node_data['coord']
            print(f"  coord: shape={coords.shape}, dtype={coords.dtype}")
            print(f"    Sample values: {coords[0]}")

        if 'node_scalar_features' in node_data:
            feats = node_data['node_scalar_features']
            if isinstance(feats, tuple):
                print(f"  node_scalar_features: tuple of {len(feats)} tensors")
                for i, f in enumerate(feats):
                    if isinstance(f, torch.Tensor):
                        print(f"    [{i}] shape={f.shape}, dtype={f.dtype}")
            else:
                print(f"  node_scalar_features: shape={feats.shape}, dtype={feats.dtype}")

    # Edge data
    if 'edge' in protein_data:
        print("\n[Edge Data]")
        edge_data = protein_data['edge']
        print(f"  Keys: {list(edge_data.keys())}")

        if 'edges' in edge_data:
            edges = edge_data['edges']
            if isinstance(edges, tuple):
                print(f"  edges: tuple of {len(edges)} tensors")
                for i, e in enumerate(edges):
                    if isinstance(e, torch.Tensor):
                        print(f"    [{i}] shape={e.shape}, dtype={e.dtype}")
            else:
                print(f"  edges: shape={edges.shape}, dtype={edges.dtype}")

    # Embeddings
    if 'embeddings' in protein_data:
        print("\n[ESM Embeddings]")
        emb_data = protein_data['embeddings']
        print(f"  Keys: {list(emb_data.keys())}")

        if 'esmc' in emb_data:
            esmc = emb_data['esmc']
            print(f"\n  ESMC:")
            print(f"    Keys: {list(esmc.keys())}")
            if 'mean' in esmc:
                print(f"    mean: shape={esmc['mean'].shape}, dtype={esmc['mean'].dtype}")
            if 'per_residue' in esmc:
                print(f"    per_residue: shape={esmc['per_residue'].shape}, dtype={esmc['per_residue'].dtype}")

        if 'esm3' in emb_data:
            esm3 = emb_data['esm3']
            print(f"\n  ESM3:")
            print(f"    Keys: {list(esm3.keys())}")
            if 'mean' in esm3:
                print(f"    mean: shape={esm3['mean'].shape}, dtype={esm3['mean'].dtype}")
            if 'per_residue' in esm3:
                print(f"    per_residue: shape={esm3['per_residue'].shape}, dtype={esm3['per_residue'].dtype}")

    # Load ligands
    print("\n" + "="*80)
    print("LIGAND DATA")
    print("="*80)

    ligands_path = Path(f"test_data/{pdb_id}/ligands.pt")
    ligands_data = torch.load(ligands_path, weights_only=False)

    print(f"\nNumber of poses: {len(ligands_data)}")

    # Inspect first pose
    pose = ligands_data[0]
    print(f"\n[Pose 0]")
    print(f"Top-level keys: {list(pose.keys())}")

    print("\n[Coordinates]")
    print(f"  coord: shape={pose['coord'].shape}, dtype={pose['coord'].dtype}")
    print(f"    Sample: {pose['coord'][0]}")
    print(f"  crystal_coord: shape={pose['crystal_coord'].shape}, dtype={pose['crystal_coord'].dtype}")
    print(f"    Sample: {pose['crystal_coord'][0]}")

    print("\n[Graph Structure]")
    print(f"  node_feats: shape={pose['node_feats'].shape}, dtype={pose['node_feats'].dtype}")
    print(f"  edges: shape={pose['edges'].shape}, dtype={pose['edges'].dtype}")
    print(f"  edge_feats: shape={pose['edge_feats'].shape}, dtype={pose['edge_feats'].dtype}")

    print("\n[Fragment Decomposition]")
    print(f"  n_fragments: {pose['n_fragments']}")
    print(f"  fragment_id: shape={pose['fragment_id'].shape}, unique={pose['fragment_id'].unique().tolist()}")
    print(f"  fragment_local_coords: shape={pose['fragment_local_coords'].shape}")
    print(f"  fragment_coms: shape={pose['fragment_coms'].shape}")
    print(f"    Sample COMs: {pose['fragment_coms'][:3]}")
    print(f"  torsional_edge_index: shape={pose['torsional_edge_index'].shape}")
    print(f"  torsional_edge_attr: shape={pose['torsional_edge_attr'].shape}")

    print("\n[Torsion Information]")
    print(f"  mask_rotate: shape={pose['mask_rotate'].shape}, dtype={pose['mask_rotate'].dtype}")
    print(f"  rotatable_edges: shape={pose['rotatable_edges'].shape}, values={pose['rotatable_edges'].tolist()}")
    print(f"  torsion_angles_x0: shape={pose['torsion_angles_x0'].shape}")
    print(f"    Values (rad): {pose['torsion_angles_x0'].tolist()}")
    print(f"  torsion_angles_x1: shape={pose['torsion_angles_x1'].shape}")
    print(f"    Values (rad): {pose['torsion_angles_x1'].tolist()}")
    print(f"  torsion_changes: shape={pose['torsion_changes'].shape}")
    print(f"    Values (rad): {pose['torsion_changes'].tolist()}")
    print(f"  torsion_atoms: {len(pose['torsion_atoms'])} entries")
    print(f"    Sample: {pose['torsion_atoms'][0] if pose['torsion_atoms'] else 'N/A'}")

    print("\n[Rigid Transformation]")
    print(f"  translation: shape={pose['translation'].shape}, dtype={pose['translation'].dtype}")
    print(f"    Values: {pose['translation'].tolist()}")
    print(f"  rotation: shape={pose['rotation'].shape}, dtype={pose['rotation'].dtype}")
    print(f"    Values (axis-angle, rad): {pose['rotation'].tolist()}")

    print("\n[Distance Bounds]")
    print(f"  distance_lower_bounds: shape={pose['distance_lower_bounds'].shape}")
    print(f"  distance_upper_bounds: shape={pose['distance_upper_bounds'].shape}")

    # Check fragments field
    if 'fragments' in pose:
        print(f"\n[Fragment Objects]")
        print(f"  Number of fragment dicts: {len(pose['fragments'])}")
        if pose['fragments']:
            frag = pose['fragments'][0]
            print(f"  Sample fragment keys: {list(frag.keys())}")

    # Compute RMSD between docked and crystal
    rmsd = torch.sqrt(torch.mean(torch.sum((pose['coord'] - pose['crystal_coord'])**2, dim=1)))
    print(f"\n[Quality Check]")
    print(f"  RMSD (docked vs crystal): {rmsd:.3f} Å")

    print("\n" + "="*80)
    print("✅ Data inspection complete!")
    print("="*80)


if __name__ == "__main__":
    inspect_sample()
