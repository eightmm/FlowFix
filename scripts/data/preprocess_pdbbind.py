#!/usr/bin/env python
"""
PDBbind Data Preprocessing for FlowFix

Converts PDBbind protein PDB and ligand mol2 files into graph format
with specified distance cutoffs for protein (6A) and ligand (10A).

Output structure:
    output_dir/
        {pdb_id}/
            protein.pt   # Protein graph features
            ligand.pt    # Ligand graph features (crystal pose only)
        splits.json      # Train/val/test splits
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem

from featurizer import ProteinFeaturizer, MoleculeFeaturizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mol2(mol2_path: str) -> Optional[Chem.Mol]:
    """Load molecule from mol2 file."""
    try:
        mol = Chem.MolFromMol2File(mol2_path, removeHs=False)
        if mol is None:
            # Try without sanitization
            mol = Chem.MolFromMol2File(mol2_path, removeHs=False, sanitize=False)
        return mol
    except Exception as e:
        logger.warning(f"Failed to load mol2 {mol2_path}: {e}")
        return None


def load_sdf(sdf_path: str) -> Optional[Chem.Mol]:
    """Load molecule from SDF file."""
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = next(iter(suppl), None)
        return mol
    except Exception as e:
        logger.warning(f"Failed to load sdf {sdf_path}: {e}")
        return None


def extract_protein_features(
    pdb_path: str,
    distance_cutoff: float = 6.0,
) -> Optional[Dict]:
    """
    Extract protein graph features from PDB file.

    Args:
        pdb_path: Path to protein PDB file
        distance_cutoff: Distance cutoff for edge construction (Angstroms)

    Returns:
        Dictionary with protein features or None if failed
    """
    try:
        featurizer = ProteinFeaturizer(pdb_path)
        res_node, res_edge = featurizer.get_residue_features(
            distance_cutoff=distance_cutoff
        )

        # Get chain sequences for later ESM embedding extraction
        sequences = featurizer.get_sequence_by_chain()

        return {
            'node': res_node,
            'edge': res_edge,
            'sequences': sequences,
            'distance_cutoff': distance_cutoff,
        }
    except Exception as e:
        logger.warning(f"Failed to extract protein features from {pdb_path}: {e}")
        return None


def extract_ligand_features(
    mol: Chem.Mol,
    distance_cutoff: float = 10.0,
    use_hydrogen: bool = False,
) -> Optional[Dict]:
    """
    Extract ligand graph features from RDKit molecule.

    Args:
        mol: RDKit molecule with conformer
        distance_cutoff: Distance cutoff for edge construction (Angstroms)
        use_hydrogen: Whether to include hydrogen atoms

    Returns:
        Dictionary with ligand features or None if failed
    """
    try:
        # Remove hydrogens if not needed
        if not use_hydrogen:
            mol = Chem.RemoveHs(mol)

        # Check for conformer
        if mol.GetNumConformers() == 0:
            logger.warning("Molecule has no conformer")
            return None

        # Get featurizer with custom distance cutoff
        featurizer = MoleculeFeaturizer(mol, hydrogen=use_hydrogen)

        # Get graph features
        data = featurizer.get_graph(distance_cutoff=distance_cutoff)
        nodes = data[0]
        edges = data[1]

        # Get coordinates
        conf = mol.GetConformer()
        coords = torch.tensor(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())],
            dtype=torch.float32
        )

        # Compute distance bounds for conformational constraints
        from rdkit.Chem import rdDistGeom
        try:
            bounds_matrix = rdDistGeom.GetMoleculeBoundsMatrix(
                mol,
                set15bounds=True,
                scaleVDW=False,
                doTriangleSmoothing=True,
                useMacrocycle14config=False
            )

            num_atoms = mol.GetNumAtoms()
            distance_lower_bounds = np.zeros((num_atoms, num_atoms), dtype=np.float32)
            distance_upper_bounds = np.zeros((num_atoms, num_atoms), dtype=np.float32)

            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i > j:
                        distance_lower_bounds[i, j] = bounds_matrix[i, j]
                        distance_lower_bounds[j, i] = bounds_matrix[i, j]
                    elif i < j:
                        distance_upper_bounds[i, j] = bounds_matrix[i, j]
                        distance_upper_bounds[j, i] = bounds_matrix[i, j]

            distance_lower_bounds = torch.tensor(distance_lower_bounds, dtype=torch.float32)
            distance_upper_bounds = torch.tensor(distance_upper_bounds, dtype=torch.float32)
        except:
            # If bounds computation fails, set to None
            distance_lower_bounds = None
            distance_upper_bounds = None

        return {
            'edges': edges['edges'],
            'node_feats': nodes['node_feats'],
            'edge_feats': edges['edge_feats'],
            'coord': coords,
            'crystal_coord': coords.clone(),  # Same as coord for crystal structure
            'distance_lower_bounds': distance_lower_bounds,
            'distance_upper_bounds': distance_upper_bounds,
            'num_atoms': mol.GetNumAtoms(),
            'distance_cutoff': distance_cutoff,
        }
    except Exception as e:
        logger.warning(f"Failed to extract ligand features: {e}")
        traceback.print_exc()
        return None


def process_single_pdb(
    pdb_id: str,
    data_dir: Path,
    output_dir: Path,
    protein_cutoff: float = 6.0,
    ligand_cutoff: float = 10.0,
    use_hydrogen: bool = False,
) -> Tuple[str, bool, str]:
    """
    Process a single PDB entry.

    Args:
        pdb_id: PDB ID to process
        data_dir: Input data directory (PDBbind)
        output_dir: Output directory for processed data
        protein_cutoff: Distance cutoff for protein edges
        ligand_cutoff: Distance cutoff for ligand edges
        use_hydrogen: Include hydrogen atoms in ligand

    Returns:
        Tuple of (pdb_id, success, message)
    """
    try:
        pdb_dir = data_dir / pdb_id

        # Find protein and ligand files
        protein_pdb = pdb_dir / f"{pdb_id}_protein.pdb"
        ligand_mol2 = pdb_dir / f"{pdb_id}_ligand.mol2"
        ligand_sdf = pdb_dir / f"{pdb_id}_ligand.sdf"

        # Check files exist
        if not protein_pdb.exists():
            return pdb_id, False, f"Protein PDB not found: {protein_pdb}"

        # Load ligand (prefer mol2, fallback to sdf)
        mol = None
        if ligand_mol2.exists():
            mol = load_mol2(str(ligand_mol2))
        if mol is None and ligand_sdf.exists():
            mol = load_sdf(str(ligand_sdf))

        if mol is None:
            return pdb_id, False, "Failed to load ligand"

        # Extract protein features
        protein_features = extract_protein_features(
            str(protein_pdb),
            distance_cutoff=protein_cutoff
        )
        if protein_features is None:
            return pdb_id, False, "Failed to extract protein features"

        # Extract ligand features
        ligand_features = extract_ligand_features(
            mol,
            distance_cutoff=ligand_cutoff,
            use_hydrogen=use_hydrogen
        )
        if ligand_features is None:
            return pdb_id, False, "Failed to extract ligand features"

        # Create output directory
        out_pdb_dir = output_dir / pdb_id
        out_pdb_dir.mkdir(parents=True, exist_ok=True)

        # Save features
        torch.save(protein_features, out_pdb_dir / "protein.pt")
        torch.save(ligand_features, out_pdb_dir / "ligand.pt")

        return pdb_id, True, "Success"

    except Exception as e:
        return pdb_id, False, f"Error: {str(e)}"


def create_splits(
    pdb_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits.

    Args:
        pdb_ids: List of PDB IDs
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Dictionary with 'train', 'valid', 'test' keys
    """
    np.random.seed(seed)

    # Shuffle
    pdb_ids = np.array(pdb_ids)
    np.random.shuffle(pdb_ids)

    n = len(pdb_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = pdb_ids[:n_train].tolist()
    val_ids = pdb_ids[n_train:n_train + n_val].tolist()
    test_ids = pdb_ids[n_train + n_val:].tolist()

    return {
        'train': train_ids,
        'valid': val_ids,
        'test': test_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDBbind data for FlowFix")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to PDBbind data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--protein_cutoff', type=float, default=6.0,
                        help='Distance cutoff for protein edges (Angstroms)')
    parser.add_argument('--ligand_cutoff', type=float, default=10.0,
                        help='Distance cutoff for ligand edges (Angstroms)')
    parser.add_argument('--use_hydrogen', action='store_true',
                        help='Include hydrogen atoms in ligand')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing PDBbind data from: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Protein cutoff: {args.protein_cutoff} A")
    logger.info(f"Ligand cutoff: {args.ligand_cutoff} A")
    logger.info(f"Include hydrogens: {args.use_hydrogen}")

    # Find all PDB directories
    pdb_ids = []
    for item in data_dir.iterdir():
        if item.is_dir() and len(item.name) == 4:  # PDB IDs are 4 characters
            pdb_ids.append(item.name)

    pdb_ids = sorted(pdb_ids)
    logger.info(f"Found {len(pdb_ids)} PDB entries")

    if args.max_samples:
        pdb_ids = pdb_ids[:args.max_samples]
        logger.info(f"Processing first {args.max_samples} samples")

    # Process in parallel
    successful_ids = []
    failed_ids = []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    process_single_pdb,
                    pdb_id, data_dir, output_dir,
                    args.protein_cutoff, args.ligand_cutoff, args.use_hydrogen
                ): pdb_id
                for pdb_id in pdb_ids
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                pdb_id, success, message = future.result()
                if success:
                    successful_ids.append(pdb_id)
                else:
                    failed_ids.append((pdb_id, message))
    else:
        for pdb_id in tqdm(pdb_ids, desc="Processing"):
            pdb_id, success, message = process_single_pdb(
                pdb_id, data_dir, output_dir,
                args.protein_cutoff, args.ligand_cutoff, args.use_hydrogen
            )
            if success:
                successful_ids.append(pdb_id)
            else:
                failed_ids.append((pdb_id, message))

    logger.info(f"\nProcessing complete:")
    logger.info(f"  Successful: {len(successful_ids)}")
    logger.info(f"  Failed: {len(failed_ids)}")

    # Create splits
    if len(successful_ids) > 0:
        splits = create_splits(
            successful_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        splits_path = output_dir / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)

        logger.info(f"\nSplits saved to: {splits_path}")
        logger.info(f"  Train: {len(splits['train'])}")
        logger.info(f"  Valid: {len(splits['valid'])}")
        logger.info(f"  Test: {len(splits['test'])}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'protein_cutoff': args.protein_cutoff,
        'ligand_cutoff': args.ligand_cutoff,
        'use_hydrogen': args.use_hydrogen,
        'total_processed': len(successful_ids),
        'failed': len(failed_ids),
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save failed list
    if failed_ids:
        with open(output_dir / "failed.txt", 'w') as f:
            for pdb_id, msg in failed_ids:
                f.write(f"{pdb_id}\t{msg}\n")

    logger.info(f"\nDone! Processed data saved to: {output_dir}")


if __name__ == '__main__':
    main()
