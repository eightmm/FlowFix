import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms
from scipy.spatial.transform import Rotation
from typing import Tuple, List, Dict
import os


def get_atom_positions(mol):
    """Extract 3D coordinates from RDKit molecule"""
    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    return np.array(positions)


def calculate_centroid(positions):
    """Calculate centroid of atomic positions"""
    return np.mean(positions, axis=0)


def find_rotatable_bonds(mol):
    """
    Find rotatable bonds in molecule
    Returns list of (atom1_idx, atom2_idx) tuples for rotatable bonds
    """
    rotatable_bonds = []
    
    for bond in mol.GetBonds():
        # Only consider single bonds
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
            
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        
        atom1 = mol.GetAtomWithIdx(atom1_idx)
        atom2 = mol.GetAtomWithIdx(atom2_idx)
        
        # Skip bonds involving hydrogen
        if atom1.GetSymbol() == 'H' or atom2.GetSymbol() == 'H':
            continue
            
        # Skip bonds where either atom has degree 1 (terminal)
        if atom1.GetDegree() == 1 or atom2.GetDegree() == 1:
            continue
            
        # Skip bonds in rings (not rotatable)
        if mol.GetRingInfo().NumBondRings(bond.GetIdx()) > 0:
            continue
            
        rotatable_bonds.append((atom1_idx, atom2_idx))
    
    return rotatable_bonds


def get_torsion_for_bond(mol, bond_atoms):
    """
    Get torsion angle for a rotatable bond
    Returns (torsion_atoms, angle) where torsion_atoms is [a1, bond1, bond2, a2]
    """
    atom1_idx, atom2_idx = bond_atoms
    
    atom1 = mol.GetAtomWithIdx(atom1_idx)
    atom2 = mol.GetAtomWithIdx(atom2_idx)
    
    # Find neighbors for torsion calculation
    neighbors1 = [n.GetIdx() for n in atom1.GetNeighbors() if n.GetIdx() != atom2_idx]
    neighbors2 = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != atom1_idx]
    
    if not neighbors1 or not neighbors2:
        return None, None
    
    # Use first available neighbor for each atom
    torsion_atoms = [neighbors1[0], atom1_idx, atom2_idx, neighbors2[0]]
    
    try:
        angle = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *torsion_atoms)
        return torsion_atoms, angle
    except:
        return None, None


def calculate_torsion_differences(crystal_mol, docked_mol):
    """
    Calculate torsion angle differences between crystal and docked conformations
    Returns list of torsion changes needed to convert docked to crystal
    """
    # Find rotatable bonds
    rotatable_bonds = find_rotatable_bonds(crystal_mol)
    
    torsion_changes = []
    
    for bond in rotatable_bonds:
        # Get torsion angles for both molecules
        crystal_torsion_atoms, crystal_angle = get_torsion_for_bond(crystal_mol, bond)
        docked_torsion_atoms, docked_angle = get_torsion_for_bond(docked_mol, bond)
        
        if crystal_torsion_atoms is None or docked_torsion_atoms is None:
            continue
            
        if crystal_angle is None or docked_angle is None:
            continue
        
        # Calculate angle difference
        angle_diff = crystal_angle - docked_angle
        
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        torsion_changes.append({
            'bond': bond,
            'torsion_atoms': crystal_torsion_atoms,
            'crystal_angle': crystal_angle,
            'docked_angle': docked_angle,
            'angle_change': angle_diff
        })
    
    return torsion_changes


def apply_torsion_changes(mol, torsion_changes):
    """
    Apply torsion angle changes to molecule to match crystal conformer
    This adjusts internal conformation without affecting overall position/orientation
    Returns modified molecule
    """
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()

    # Apply torsion changes in order to match crystal conformation
    for torsion_change in torsion_changes:
        try:
            # Set the torsion angle to match crystal structure
            # This rotates around the bond to match the target angle
            target_angle = torsion_change['crystal_angle']
            rdMolTransforms.SetDihedralDeg(conf, *torsion_change['torsion_atoms'], target_angle)
        except Exception as e:
            print(f"Failed to apply torsion change for bond {torsion_change['bond']}: {e}")
            continue

    return mol_copy


def calculate_rigid_transformation(crystal_mol, torsion_corrected_mol):
    """
    Calculate rigid body transformation (rotation + translation) needed
    to align torsion-corrected molecule to crystal structure
    Uses Kabsch algorithm for optimal superposition
    """
    crystal_pos = get_atom_positions(crystal_mol)
    query_pos = get_atom_positions(torsion_corrected_mol)

    # Calculate centroids
    crystal_centroid = calculate_centroid(crystal_pos)
    query_centroid = calculate_centroid(query_pos)

    # Center both structures at origin
    crystal_centered = crystal_pos - crystal_centroid
    query_centered = query_pos - query_centroid

    # Calculate optimal rotation using Kabsch algorithm
    # Covariance matrix H = query.T @ crystal
    H = query_centered.T @ crystal_centered

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix R
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation vector: move to crystal position after rotation
    translation = crystal_centroid

    return R, translation, query_centroid


def apply_rigid_transformation(mol, rotation_matrix, translation_vector, original_centroid):
    """
    Apply rigid body transformation to molecule
    Steps:
    1. Center molecule at origin (using its original centroid)
    2. Apply rotation
    3. Translate to final position
    """
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()

    positions = get_atom_positions(mol_copy)

    # Step 1: Center at origin
    positions_centered = positions - original_centroid

    # Step 2: Apply rotation
    positions_rotated = positions_centered @ rotation_matrix.T

    # Step 3: Translate to final position
    positions_final = positions_rotated + translation_vector

    # Update atomic positions in the molecule
    for i in range(mol_copy.GetNumAtoms()):
        conf.SetAtomPosition(i, positions_final[i])

    return mol_copy


def transform_docked_to_crystal(crystal_mol, docked_mol, verbose=True):
    """
    Transform docked molecule to crystal conformation

    Process:
    1. Adjust internal conformation: Apply torsion changes to match crystal conformer
    2. Apply rigid body transformation: Rotation followed by translation

    The key insight is that docking = torsion changes + rigid body movement
    So to reverse: first fix torsions, then apply rigid transformation

    Returns:
        - transformed_mol: Final transformed molecule
        - torsion_changes: List of torsion changes applied
        - rotation_matrix: 3x3 rotation matrix
        - translation_vector: Translation vector
        - rmsd_initial: Initial RMSD
        - rmsd_after_torsion: RMSD after torsion correction
        - rmsd_final: Final RMSD after complete transformation
    """

    if verbose:
        print(f"Crystal atoms: {crystal_mol.GetNumAtoms()}, Docked atoms: {docked_mol.GetNumAtoms()}")

    # Calculate initial RMSD (direct comparison without alignment)
    try:
        rmsd_initial = rdMolAlign.CalcRMS(crystal_mol, docked_mol)
    except:
        rmsd_initial = None

    if verbose:
        print(f"Initial RMSD: {rmsd_initial:.3f} Å" if rmsd_initial else "Initial RMSD: calculation failed")

    # Step 1: Adjust internal conformation by matching torsion angles to crystal
    torsion_changes = calculate_torsion_differences(crystal_mol, docked_mol)

    if verbose:
        print(f"\n[Step 1] Conformational Adjustment")
        print(f"Found {len(torsion_changes)} rotatable bonds")
        if len(torsion_changes) > 0:
            print("Torsion adjustments:")
            for i, tc in enumerate(torsion_changes[:5]):  # Show first 5
                print(f"  Bond {tc['bond']}: {tc['docked_angle']:.1f}° → {tc['crystal_angle']:.1f}° (Δ = {tc['angle_change']:.1f}°)")

    # Apply torsion changes to match crystal conformer
    torsion_corrected_mol = apply_torsion_changes(docked_mol, torsion_changes)

    # Calculate RMSD after torsion correction (direct comparison)
    try:
        rmsd_after_torsion = rdMolAlign.CalcRMS(crystal_mol, torsion_corrected_mol)
        if verbose:
            print(f"RMSD after torsion correction: {rmsd_after_torsion:.3f} Å")
    except:
        rmsd_after_torsion = None

    # Step 2: Calculate and apply rigid body transformation
    rotation_matrix, translation_vector, query_centroid = calculate_rigid_transformation(
        crystal_mol, torsion_corrected_mol
    )

    if verbose:
        print(f"\n[Step 2] Rigid Body Transformation")
        print(f"Rotation matrix determinant: {np.linalg.det(rotation_matrix):.3f} (should be 1.0)")
        print(f"Translation vector: [{translation_vector[0]:.3f}, {translation_vector[1]:.3f}, {translation_vector[2]:.3f}]")
        print(f"Translation magnitude: {np.linalg.norm(translation_vector):.3f} Å")

    # Apply rigid transformation (rotation then translation)
    final_mol = apply_rigid_transformation(
        torsion_corrected_mol, rotation_matrix, translation_vector, query_centroid
    )

    # Calculate final RMSD (direct comparison to verify transformation)
    try:
        rmsd_final = rdMolAlign.CalcRMS(crystal_mol, final_mol)
    except:
        rmsd_final = None

    if verbose:
        print(f"\n[Results]")
        print(f"Final RMSD: {rmsd_final:.3f} Å" if rmsd_final else "Final RMSD: calculation failed")

        if rmsd_initial and rmsd_final:
            improvement = rmsd_initial - rmsd_final
            improvement_percent = (improvement / rmsd_initial) * 100
            print(f"Improvement: {improvement:.3f} Å ({improvement_percent:.1f}%)")
            print(f"Transformation successful: {rmsd_final < 0.5}")  # Success if RMSD < 0.5 Å

    return {
        'transformed_mol': final_mol,
        'torsion_changes': torsion_changes,
        'rotation_matrix': rotation_matrix,
        'translation_vector': translation_vector,
        'rmsd_initial': rmsd_initial,
        'rmsd_after_torsion': rmsd_after_torsion,
        'rmsd_final': rmsd_final,
        'improvement': rmsd_initial - rmsd_final if (rmsd_initial and rmsd_final) else None,
        'success': rmsd_final < 0.5 if rmsd_final else False  # Consider successful if RMSD < 0.5 Å
    }


def analyze_sdf_poses(sdf_file):
    """
    Analyze all docked poses in SDF file
    First molecule = crystal structure
    Rest = docked poses
    """
    if not os.path.exists(sdf_file):
        print(f"Error: SDF file not found: {sdf_file}")
        return None
    
    # Load molecules
    supplier = Chem.SDMolSupplier(sdf_file)
    molecules = [mol for mol in supplier if mol is not None]
    
    if len(molecules) == 0:
        print("Error: No molecules found in SDF file")
        return None
    
    print(f"Loaded {len(molecules)} molecules from {sdf_file}")
    print("Molecule 0: Crystal structure (reference)")
    print(f"Molecules 1-{len(molecules)-1}: Docked poses")
    print("="*60)
    
    crystal_mol = molecules[0]
    results = []
    
    # Analyze each docked pose (limit to first 5 for testing)
    for i in range(1, min(len(molecules), 6)):
        print(f"\nAnalyzing Pose {i}:")
        print("-" * 30)
        
        docked_mol = molecules[i]
        
        try:
            result = transform_docked_to_crystal(crystal_mol, docked_mol, verbose=True)
            result['pose_id'] = i
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing pose {i}: {str(e)}")
            continue
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        successful_transforms = [r for r in results if r['success']]
        initial_rmsds = [r['rmsd_initial'] for r in results if r['rmsd_initial']]
        final_rmsds = [r['rmsd_final'] for r in results if r['rmsd_final']]
        improvements = [r['improvement'] for r in results if r['improvement']]
        
        print(f"Poses analyzed: {len(results)}")
        print(f"Successful transformations: {len(successful_transforms)}")
        
        if initial_rmsds:
            print(f"Average initial RMSD: {np.mean(initial_rmsds):.3f} ± {np.std(initial_rmsds):.3f} Å")
        if final_rmsds:
            print(f"Average final RMSD: {np.mean(final_rmsds):.3f} ± {np.std(final_rmsds):.3f} Å")
        if improvements:
            print(f"Average improvement: {np.mean(improvements):.3f} ± {np.std(improvements):.3f} Å")
    
    return results


def main():
    """Main function to run the analysis"""
    sdf_file = "/mnt/data/PROJECT/canonical-set/10gs/10gs_selected.sdf"
    
    print("Protein-Ligand Conformation Analysis")
    print("Transforming docked poses to crystal conformation")
    print("="*50)
    
    results = analyze_sdf_poses(sdf_file)
    
    if results:
        # Save results
        output_file = "/home/jaemin/project/protein-ligand/PredRMSD/transformation_results.txt"
        
        with open(output_file, 'w') as f:
            f.write("Pose_ID\tRMSD_Initial\tRMSD_After_Torsion\tRMSD_Final\tImprovement\tTranslation_Mag\tNum_Torsions\tSuccess\n")
            for r in results:
                f.write(f"{r['pose_id']}\t")
                f.write(f"{r['rmsd_initial']:.3f}\t" if r['rmsd_initial'] else "N/A\t")
                f.write(f"{r['rmsd_after_torsion']:.3f}\t" if r['rmsd_after_torsion'] else "N/A\t")
                f.write(f"{r['rmsd_final']:.3f}\t" if r['rmsd_final'] else "N/A\t")
                f.write(f"{r['improvement']:.3f}\t" if r['improvement'] else "N/A\t")
                f.write(f"{np.linalg.norm(r['translation_vector']):.3f}\t")
                f.write(f"{len(r['torsion_changes'])}\t")
                f.write(f"{r['success']}\n")
        
        print(f"\nResults saved to: {output_file}")


def extract_transformation_components(crystal_mol, docked_mol):
    """
    Extract individual transformation components for regression model training

    Returns dictionary with:
    - total_rmsd: Overall RMSD between crystal and docked
    - translation_magnitude: Magnitude of translation vector
    - rotation_angles: Euler angles from rotation matrix (in degrees)
    - torsion_angles: List of torsion angle differences
    - num_rotatable_bonds: Number of rotatable bonds
    """

    # Calculate initial RMSD
    try:
        total_rmsd = rdMolAlign.CalcRMS(crystal_mol, docked_mol)
    except:
        total_rmsd = None

    # Get torsion angle differences
    torsion_changes = calculate_torsion_differences(crystal_mol, docked_mol)
    torsion_angles = [tc['angle_change'] for tc in torsion_changes]

    # Apply torsion correction to isolate rigid body transformation
    torsion_corrected_mol = apply_torsion_changes(docked_mol, torsion_changes)

    # Calculate rigid body transformation components
    rotation_matrix, translation_vector, query_centroid = calculate_rigid_transformation(
        crystal_mol, torsion_corrected_mol
    )

    # Extract translation magnitude
    translation_magnitude = np.linalg.norm(translation_vector - query_centroid)

    # Convert rotation matrix to Euler angles (in degrees)
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

    # Calculate rotation angle magnitude (axis-angle representation)
    rotation_vector = r.as_rotvec(degrees=True)
    rotation_angle = np.linalg.norm(rotation_vector)

    return {
        'total_rmsd': total_rmsd,
        'translation_magnitude': translation_magnitude,
        'translation_vector': translation_vector.tolist(),
        'rotation_euler_angles': euler_angles.tolist(),
        'rotation_angle_magnitude': rotation_angle,
        'rotation_matrix': rotation_matrix.tolist(),
        'torsion_angles': torsion_angles,
        'torsion_rmsd': np.sqrt(np.mean(np.array(torsion_angles)**2)) if torsion_angles else 0.0,
        'num_rotatable_bonds': len(torsion_changes)
    }


def decompose_rmsd_contributions(crystal_mol, docked_mol):
    """
    Decompose RMSD into individual contributions from different transformation components

    Returns dictionary with:
    - initial_rmsd: Starting RMSD
    - torsion_rmsd: RMSD contribution from torsion angles
    - translation_rmsd: RMSD contribution from translation
    - rotation_rmsd: RMSD contribution from rotation
    - final_rmsd: Final RMSD after all transformations
    """

    # Initial RMSD
    try:
        initial_rmsd = rdMolAlign.CalcRMS(crystal_mol, docked_mol)
    except:
        initial_rmsd = None

    # Step 1: Apply torsion correction
    torsion_changes = calculate_torsion_differences(crystal_mol, docked_mol)
    torsion_corrected_mol = apply_torsion_changes(docked_mol, torsion_changes)

    try:
        rmsd_after_torsion = rdMolAlign.CalcRMS(crystal_mol, torsion_corrected_mol)
    except:
        rmsd_after_torsion = None

    # Step 2: Apply translation only
    rotation_matrix, translation_vector, query_centroid = calculate_rigid_transformation(
        crystal_mol, torsion_corrected_mol
    )

    # Create molecule with only translation applied
    translation_only_mol = Chem.Mol(torsion_corrected_mol)
    conf = translation_only_mol.GetConformer()
    positions = get_atom_positions(translation_only_mol)

    # Apply only translation (no rotation)
    positions_translated = positions - query_centroid + translation_vector
    for i in range(translation_only_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, positions_translated[i])

    try:
        rmsd_after_translation = rdMolAlign.CalcRMS(crystal_mol, translation_only_mol)
    except:
        rmsd_after_translation = None

    # Step 3: Apply full transformation (rotation + translation)
    final_mol = apply_rigid_transformation(
        torsion_corrected_mol, rotation_matrix, translation_vector, query_centroid
    )

    try:
        final_rmsd = rdMolAlign.CalcRMS(crystal_mol, final_mol)
    except:
        final_rmsd = None

    # Calculate individual contributions
    torsion_contribution = abs(initial_rmsd - rmsd_after_torsion) if (initial_rmsd and rmsd_after_torsion) else None
    translation_contribution = abs(rmsd_after_torsion - rmsd_after_translation) if (rmsd_after_torsion and rmsd_after_translation) else None
    rotation_contribution = abs(rmsd_after_translation - final_rmsd) if (rmsd_after_translation and final_rmsd) else None

    return {
        'initial_rmsd': initial_rmsd,
        'rmsd_after_torsion': rmsd_after_torsion,
        'rmsd_after_translation': rmsd_after_translation,
        'final_rmsd': final_rmsd,
        'torsion_contribution': torsion_contribution,
        'translation_contribution': translation_contribution,
        'rotation_contribution': rotation_contribution,
        'total_improvement': initial_rmsd - final_rmsd if (initial_rmsd and final_rmsd) else None
    }


def prepare_regression_features(crystal_mol, docked_mol):
    """
    Prepare features for regression model training
    Combines transformation components and RMSD decomposition

    Returns flattened feature vector suitable for regression
    """

    # Get transformation components
    transform_components = extract_transformation_components(crystal_mol, docked_mol)

    # Get RMSD decomposition
    rmsd_decomp = decompose_rmsd_contributions(crystal_mol, docked_mol)

    # Prepare feature vector
    features = {
        # Target variables (what to predict)
        'target_total_rmsd': transform_components['total_rmsd'],
        'target_translation_mag': transform_components['translation_magnitude'],
        'target_rotation_angle': transform_components['rotation_angle_magnitude'],
        'target_torsion_rmsd': transform_components['torsion_rmsd'],

        # Detailed components
        'translation_x': transform_components['translation_vector'][0],
        'translation_y': transform_components['translation_vector'][1],
        'translation_z': transform_components['translation_vector'][2],

        'rotation_roll': transform_components['rotation_euler_angles'][0],
        'rotation_pitch': transform_components['rotation_euler_angles'][1],
        'rotation_yaw': transform_components['rotation_euler_angles'][2],

        'num_rotatable_bonds': transform_components['num_rotatable_bonds'],
        'num_torsion_changes': len(transform_components['torsion_angles']),

        # RMSD contributions
        'rmsd_torsion_contrib': rmsd_decomp['torsion_contribution'],
        'rmsd_translation_contrib': rmsd_decomp['translation_contribution'],
        'rmsd_rotation_contrib': rmsd_decomp['rotation_contribution'],

        # Torsion angle statistics
        'torsion_mean': np.mean(transform_components['torsion_angles']) if transform_components['torsion_angles'] else 0,
        'torsion_std': np.std(transform_components['torsion_angles']) if transform_components['torsion_angles'] else 0,
        'torsion_max': np.max(np.abs(transform_components['torsion_angles'])) if transform_components['torsion_angles'] else 0,
    }

    return features


if __name__ == "__main__":
    main()