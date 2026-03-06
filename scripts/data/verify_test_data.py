#!/usr/bin/env python
"""
Verify that generated test data has all required fields and correct formats.
"""

import torch
from pathlib import Path
import numpy as np


def verify_protein_data(protein_path: Path) -> dict:
    """Verify protein data structure."""
    try:
        protein_data = torch.load(protein_path, weights_only=False)

        results = {
            'status': 'success',
            'errors': [],
            'info': {}
        }

        # Check node data
        if 'node' in protein_data:
            node_data = protein_data['node']

            # Check coordinates
            if 'coord' in node_data:
                coords = node_data['coord']
                results['info']['n_residues'] = coords.shape[0]
                results['info']['coord_shape'] = tuple(coords.shape)

                if coords.shape[1] != 3:
                    results['errors'].append(f"Invalid coord shape: {coords.shape}, expected [N, 3]")
            else:
                results['errors'].append("Missing 'coord' in node data")

            # Check node features
            if 'node_scalar_features' in node_data:
                results['info']['has_node_features'] = True
            else:
                results['errors'].append("Missing 'node_scalar_features'")
        else:
            results['errors'].append("Missing 'node' key in protein data")

        # Check edge data
        if 'edge' in protein_data:
            edge_data = protein_data['edge']

            if 'edges' in edge_data:
                edges = edge_data['edges']
                results['info']['n_edges'] = edges.shape[1] if edges.dim() == 2 else edges[0].shape[0]
                results['info']['edge_shape'] = tuple(edges.shape) if edges.dim() == 2 else f"tuple: {len(edges)}"
            else:
                results['errors'].append("Missing 'edges' in edge data")
        else:
            results['errors'].append("Missing 'edge' key in protein data")

        # Check embeddings (optional but expected)
        if 'embeddings' in protein_data:
            emb_data = protein_data['embeddings']
            results['info']['embeddings'] = {}

            if 'esmc' in emb_data:
                esmc = emb_data['esmc']
                if 'mean' in esmc:
                    results['info']['embeddings']['esmc_mean'] = tuple(esmc['mean'].shape)
                if 'per_residue' in esmc:
                    results['info']['embeddings']['esmc_per_res'] = tuple(esmc['per_residue'].shape)

            if 'esm3' in emb_data:
                esm3 = emb_data['esm3']
                if 'mean' in esm3:
                    results['info']['embeddings']['esm3_mean'] = tuple(esm3['mean'].shape)
                if 'per_residue' in esm3:
                    results['info']['embeddings']['esm3_per_res'] = tuple(esm3['per_residue'].shape)
        else:
            results['errors'].append("Missing 'embeddings' (ESM embeddings)")

        if results['errors']:
            results['status'] = 'failed'

        return results

    except Exception as e:
        return {
            'status': 'error',
            'errors': [str(e)],
            'info': {}
        }


def verify_ligand_data(ligands_path: Path) -> dict:
    """Verify ligand data structure."""
    try:
        ligands_data = torch.load(ligands_path, weights_only=False)

        results = {
            'status': 'success',
            'errors': [],
            'info': {}
        }

        if not isinstance(ligands_data, list):
            results['errors'].append(f"Ligands data should be a list, got {type(ligands_data)}")
            results['status'] = 'failed'
            return results

        results['info']['n_poses'] = len(ligands_data)

        # Check first pose in detail
        if len(ligands_data) > 0:
            pose = ligands_data[0]

            # Required fields
            required_fields = [
                'coord', 'crystal_coord', 'node_feats', 'edges', 'edge_feats',
                'fragment_id', 'n_fragments', 'fragment_local_coords', 'fragment_coms',
                'torsional_edge_index', 'torsional_edge_attr',
                'mask_rotate', 'rotatable_edges', 'torsion_angles_x0', 'torsion_angles_x1',
                'torsion_changes', 'torsion_atoms',
                'translation', 'rotation',
                'distance_lower_bounds', 'distance_upper_bounds'
            ]

            missing_fields = []
            for field in required_fields:
                if field not in pose:
                    missing_fields.append(field)

            if missing_fields:
                results['errors'].append(f"Missing fields: {missing_fields}")

            # Check shapes
            if 'coord' in pose:
                coord_shape = tuple(pose['coord'].shape)
                results['info']['coord_shape'] = coord_shape
                results['info']['n_atoms'] = coord_shape[0]

                if len(coord_shape) != 2 or coord_shape[1] != 3:
                    results['errors'].append(f"Invalid coord shape: {coord_shape}, expected [N, 3]")

            if 'crystal_coord' in pose:
                crystal_shape = tuple(pose['crystal_coord'].shape)
                results['info']['crystal_coord_shape'] = crystal_shape

                if 'coord' in pose and crystal_shape != tuple(pose['coord'].shape):
                    results['errors'].append(
                        f"coord and crystal_coord shape mismatch: {tuple(pose['coord'].shape)} vs {crystal_shape}"
                    )

            if 'node_feats' in pose:
                results['info']['node_feats_shape'] = tuple(pose['node_feats'].shape)

            if 'edges' in pose:
                edges_shape = tuple(pose['edges'].shape)
                results['info']['edges_shape'] = edges_shape

                if len(edges_shape) != 2 or edges_shape[0] != 2:
                    results['errors'].append(f"Invalid edges shape: {edges_shape}, expected [2, E]")

            if 'edge_feats' in pose:
                results['info']['edge_feats_shape'] = tuple(pose['edge_feats'].shape)

            # Fragment info
            if 'fragment_id' in pose:
                results['info']['fragment_id_shape'] = tuple(pose['fragment_id'].shape)

            if 'n_fragments' in pose:
                results['info']['n_fragments'] = pose['n_fragments']

            if 'fragment_coms' in pose:
                results['info']['fragment_coms_shape'] = tuple(pose['fragment_coms'].shape)

            if 'torsional_edge_index' in pose:
                results['info']['torsional_edge_index_shape'] = tuple(pose['torsional_edge_index'].shape)

            if 'torsional_edge_attr' in pose:
                results['info']['torsional_edge_attr_shape'] = tuple(pose['torsional_edge_attr'].shape)

            # Torsion info
            if 'rotatable_edges' in pose:
                results['info']['n_rotatable_edges'] = len(pose['rotatable_edges'])

            if 'torsion_changes' in pose:
                results['info']['torsion_changes_shape'] = tuple(pose['torsion_changes'].shape)

            # Rigid transformation
            if 'translation' in pose:
                trans_shape = tuple(pose['translation'].shape)
                results['info']['translation_shape'] = trans_shape

                if trans_shape != (3,):
                    results['errors'].append(f"Invalid translation shape: {trans_shape}, expected (3,)")

            if 'rotation' in pose:
                rot_shape = tuple(pose['rotation'].shape)
                results['info']['rotation_shape'] = rot_shape

                if rot_shape != (3,):
                    results['errors'].append(f"Invalid rotation shape: {rot_shape}, expected (3,)")

            # Distance bounds
            if 'distance_lower_bounds' in pose:
                results['info']['distance_lower_bounds_shape'] = tuple(pose['distance_lower_bounds'].shape)

            if 'distance_upper_bounds' in pose:
                results['info']['distance_upper_bounds_shape'] = tuple(pose['distance_upper_bounds'].shape)

        if results['errors']:
            results['status'] = 'failed'

        return results

    except Exception as e:
        return {
            'status': 'error',
            'errors': [str(e)],
            'info': {}
        }


def main():
    """Verify all test data."""
    test_data_dir = Path("./test_data")

    if not test_data_dir.exists():
        print("❌ test_data directory not found!")
        return

    print("="*80)
    print("Verifying Test Data")
    print("="*80)

    pdb_dirs = sorted([d for d in test_data_dir.iterdir() if d.is_dir()])

    all_passed = True

    for pdb_dir in pdb_dirs:
        pdb_id = pdb_dir.name

        print(f"\n{'='*80}")
        print(f"PDB: {pdb_id}")
        print(f"{'='*80}")

        # Verify protein
        protein_path = pdb_dir / "protein.pt"
        if protein_path.exists():
            print("\n📊 Protein Data:")
            protein_results = verify_protein_data(protein_path)

            if protein_results['status'] == 'success':
                print("  ✅ Status: PASSED")
            else:
                print(f"  ❌ Status: {protein_results['status'].upper()}")
                all_passed = False

            # Print info
            for key, value in protein_results['info'].items():
                print(f"    {key}: {value}")

            # Print errors
            if protein_results['errors']:
                print("  ⚠️  Errors:")
                for error in protein_results['errors']:
                    print(f"    - {error}")
        else:
            print("\n❌ protein.pt not found!")
            all_passed = False

        # Verify ligands
        ligands_path = pdb_dir / "ligands.pt"
        if ligands_path.exists():
            print("\n📊 Ligand Data:")
            ligand_results = verify_ligand_data(ligands_path)

            if ligand_results['status'] == 'success':
                print("  ✅ Status: PASSED")
            else:
                print(f"  ❌ Status: {ligand_results['status'].upper()}")
                all_passed = False

            # Print info
            for key, value in ligand_results['info'].items():
                print(f"    {key}: {value}")

            # Print errors
            if ligand_results['errors']:
                print("  ⚠️  Errors:")
                for error in ligand_results['errors']:
                    print(f"    - {error}")
        else:
            print("\n❌ ligands.pt not found!")
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✅ All test data verified successfully!")
    else:
        print("❌ Some data files have errors!")
    print("="*80)


if __name__ == "__main__":
    main()
