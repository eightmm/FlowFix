import os
from pathlib import Path
from typing import Dict, List, Optional

from featurizer import MoleculeFeaturizer
from featurizer import ProteinFeaturizer



def load_canonical_set_files(canonical_set_path: str = "/mnt/data/PROJECT/canonical-set") -> Dict[str, Dict[str, List[str]]]:
    """
    Load canonical set files organized by PDB ID
    
    Args:
        canonical_set_path: Path to canonical set directory
        
    Returns:
        Dictionary with PDB ID as key and file paths as values
        Format: {pdb_id: {'sdf': [sdf_files], 'pdb': [pdb_files]}}
    """
    canonical_set_path = Path(canonical_set_path)
    pdb_files_dict = {}
    
    if not canonical_set_path.exists():
        print(f"Warning: Canonical set path does not exist: {canonical_set_path}")
        return pdb_files_dict
    
    # Iterate through each PDB ID directory
    for pdb_dir in canonical_set_path.iterdir():
        if pdb_dir.is_dir():
            pdb_id = pdb_dir.name
            
            # Find SDF and PDB files in the directory
            sdf_files = list(pdb_dir.glob("*.sdf"))
            pdb_files = list(pdb_dir.glob("*.pdb"))
            
            # Convert Path objects to strings
            pdb_files_dict[pdb_id] = {
                'sdf': [str(f) for f in sdf_files],
                'pdb': [str(f) for f in pdb_files]
            }
    
    return pdb_files_dict

def get_pdb_files_by_id(pdb_id: str, canonical_set_path: str = "/mnt/data/PROJECT/canonical-set") -> Optional[Dict[str, List[str]]]:
    """
    Get files for specific PDB ID
    
    Args:
        pdb_id: PDB ID to search for
        canonical_set_path: Path to canonical set directory
        
    Returns:
        Dictionary with file paths or None if not found
        Format: {'sdf': [sdf_files], 'pdb': [pdb_files]}
    """
    pdb_dir = Path(canonical_set_path) / pdb_id
    
    if not pdb_dir.exists():
        return None
    
    sdf_files = list(pdb_dir.glob("*.sdf"))
    pdb_files = list(pdb_dir.glob("*.pdb"))
    
    return {
        'sdf': [str(f) for f in sdf_files],
        'pdb': [str(f) for f in pdb_files]
    }


from rdkit import Chem
from rdkit.Chem import rdDistGeom
import torch
import numpy as np
from tqdm import tqdm
from featurizer import standardize_pdb

def process_and_save_features(canonical_set_path: str = "/mnt/data/PROJECT/pdbbind_redocked_pose",
                            output_dir: str = "./train_data_dg"):
    """
    Process all canonical set files and save features as .pt files

    Args:
        canonical_set_path: Path to canonical set directory
        output_dir: Output directory for processed features
    """
    # Toggle this to show/hide PDB ID during processing
    SHOW_PDB_ID = True  # Set to False to hide PDB ID output

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load all canonical set files
    canonical_files = load_canonical_set_files(canonical_set_path)

    # Count skipped files
    skipped_count = 0
    processed_count = 0

    # Process each PDB ID with progress bar
    for pdb_id, files in tqdm(canonical_files.items(), desc="Processing PDB files", unit="file"):

        # Create output directory for this PDB ID
        pdb_output_dir = output_path / pdb_id
        pdb_output_dir.mkdir(exist_ok=True)

        # Check if already processed
        protein_file = pdb_output_dir / "protein.pt"
        ligands_file = pdb_output_dir / "ligands.pt"

        if protein_file.exists() and ligands_file.exists():
            skipped_count += 1
            continue

        # Show PDB ID if enabled
        if SHOW_PDB_ID:
            tqdm.write(f"Processing: {pdb_id}")

        processed_count += 1
        
        # Process protein
        if files['pdb']:
            pdb_file_path = files['pdb'][0]
            
            # Standardize PDB file and save to temporary location
            temp_pdb_path = f"/tmp/{pdb_id}_tmp.pdb"
            standardize_pdb(pdb_file_path, temp_pdb_path)
            
            # Use ProteinFeaturizer with standardized PDB
            protein_featurizer = ProteinFeaturizer(temp_pdb_path)
            
            # Get residue-level features
            res_node, res_edge = protein_featurizer.get_residue_features(distance_cutoff=4.0)
            
            # Save protein features
            protein_features = {
                'node': res_node,
                'edge': res_edge
            }
            torch.save(protein_features, pdb_output_dir / "protein.pt")
            
            # Clean up temporary file
            os.remove(temp_pdb_path)
        
        # Process ligands
        if files['sdf']:
            sdf_file_path = files['sdf'][0]

            # Force heavy atoms only by removing hydrogens
            suppl = Chem.SDMolSupplier(sdf_file_path, removeHs=True)
            ligand_features_list = []
            molecules = []
            for mol in suppl:
                if mol is not None:
                    molecules.append(mol)

            if len(molecules) > 0:
                # Use first molecule as crystal structure reference
                reference_mol = molecules[0]
                ref_conf = reference_mol.GetConformer()
                crystal_coords = torch.tensor(
                    [[ref_conf.GetAtomPosition(i).x,
                      ref_conf.GetAtomPosition(i).y,
                      ref_conf.GetAtomPosition(i).z]
                     for i in range(reference_mol.GetNumAtoms())],
                    dtype=torch.float32
                )

                bounds_matrix = rdDistGeom.GetMoleculeBoundsMatrix(
                    reference_mol,
                    set15bounds=True,           # Include 1-5 atom distances (not just 1-4)
                    scaleVDW=False,             # Don't scale VDW radii for close atoms
                    doTriangleSmoothing=True,   # Apply triangle smoothing for consistency
                    useMacrocycle14config=False # Standard configuration
                )
                num_atoms = reference_mol.GetNumAtoms()
                
                distance_lower_bounds = np.zeros((num_atoms, num_atoms), dtype=np.float32)
                distance_upper_bounds = np.zeros((num_atoms, num_atoms), dtype=np.float32)

                for i in range(num_atoms):
                    for j in range(num_atoms):
                        if i > j:
                            # Lower triangle: minimum distance
                            distance_lower_bounds[i, j] = bounds_matrix[i, j]
                            distance_lower_bounds[j, i] = bounds_matrix[i, j]
                        elif i < j:
                            # Upper triangle: maximum distance
                            distance_upper_bounds[i, j] = bounds_matrix[i, j]
                            distance_upper_bounds[j, i] = bounds_matrix[i, j]
                        # Diagonal (i == j) stays 0

                # Convert to tensors
                distance_lower_bounds = torch.tensor(distance_lower_bounds, dtype=torch.float32)
                distance_upper_bounds = torch.tensor(distance_upper_bounds, dtype=torch.float32)

                # Process each molecule with inner progress bar (only show for large sets)
                mol_iterator = molecules
                if len(molecules) > 10:
                    mol_iterator = tqdm(molecules, desc=f"  Processing ligands for {pdb_id}", unit="mol", leave=False)

                skipped_count = 0
                for i, mol in enumerate(mol_iterator):
                    try:
                        mol_featurizer = MoleculeFeaturizer(mol, hydrogen=False)
                        data = mol_featurizer.get_graph()

                        nodes = data[0]
                        edges = data[1]

                        ligand_data = {
                            'edges': edges['edges'],
                            'node_feats': nodes['node_feats'],
                            'edge_feats': edges['edge_feats'],
                            'coord': nodes['coords'],                      # Current pose coordinates
                            'crystal_coord': crystal_coords,           # Crystal structure coordinates (reference)
                            'distance_lower_bounds': distance_lower_bounds,  # [N, N] min distance constraints
                            'distance_upper_bounds': distance_upper_bounds   # [N, N] max distance constraints
                        }

                        ligand_features_list.append(ligand_data)

                    except Exception as E:
                        skipped_count += 1
                        if isinstance(mol_iterator, tqdm):
                            mol_iterator.set_postfix(skipped=skipped_count)
                        continue

            # Only save if we have valid ligand features
            if ligand_features_list:
                torch.save(ligand_features_list, pdb_output_dir / "ligands.pt")

    # Show final summary
    tqdm.write(f"\nProcessing complete: {processed_count} processed, {skipped_count} skipped (already existed)")

# Execute the processing
if __name__ == "__main__":
    process_and_save_features()
