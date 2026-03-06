"""
Ligand Feature Extraction with Fragment Decomposition

This module provides ligand feature extraction including:
1. Atom-level geometric features (MoleculeFeaturizer)
2. Fragment-based SE(3) decomposition (SigmaDock approach)
3. Torsion angle decomposition (DiffDock approach, legacy)
4. Distance bounds for conformational constraints
5. Rigid body transformation computation

Following SigmaDock's fragment-based approach for batched graph handling.
"""

import torch
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms
from scipy.spatial.transform import Rotation as R

from featurizer import MoleculeFeaturizer


# ============================================================================
# Fragment Classes (SigmaDock approach)
# ============================================================================

class Fragment:
    """
    Single molecular fragment.

    A fragment is a rigid substructure connected by rotatable bonds.
    """
    def __init__(
        self,
        fragment_id: int,
        atom_indices: List[int],
        local_coords: np.ndarray,
        com: np.ndarray
    ):
        """
        Initialize fragment.

        Args:
            fragment_id: Unique fragment identifier
            atom_indices: List of atom indices in this fragment
            local_coords: [N, 3] coordinates in fragment's local frame
            com: [3] center of mass of fragment
        """
        self.id = fragment_id
        self.atom_indices = atom_indices
        self.local_coords = local_coords
        self.com = com
        self.torsional_bonds = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'atom_indices': self.atom_indices,
            'local_coords': self.local_coords.tolist(),
            'com': self.com.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Fragment':
        """Load from dictionary."""
        return cls(
            d['id'],
            d['atom_indices'],
            np.array(d['local_coords']),
            np.array(d['com'])
        )


class TorsionalBond:
    """
    Rotatable bond connecting two fragments.
    """
    def __init__(
        self,
        bond_idx: int,
        frag_1: int,
        frag_2: int,
        atom_1: int,
        atom_2: int
    ):
        """
        Initialize torsional bond.

        Args:
            bond_idx: Bond index in molecule
            frag_1: First fragment ID
            frag_2: Second fragment ID
            atom_1: First atom index (in frag_1)
            atom_2: Second atom index (in frag_2)
        """
        self.bond_idx = bond_idx
        self.frag_1 = frag_1
        self.frag_2 = frag_2
        self.atom_1 = atom_1
        self.atom_2 = atom_2


def find_rotatable_bonds_rdkit(mol: Chem.Mol) -> List[Tuple[int, int]]:
    """
    Find rotatable bonds using RDKit's standard Lipinski definition.

    Uses the same definition as Chem.Lipinski.NumRotatableBonds:
    - Single bonds
    - Not in rings
    - Not terminal
    - Excludes amides, esters, etc.

    Args:
        mol: RDKit molecule

    Returns:
        List of (atom1_idx, atom2_idx) tuples for rotatable bonds
    """
    from rdkit.Chem import Lipinski

    # Use Lipinski's SMARTS pattern for rotatable bonds
    # This excludes amides, esters, and other non-rotatable functional groups
    # RotatableBondSmarts is already a Mol object
    rot_pattern = Lipinski.RotatableBondSmarts

    # Find all rotatable bonds
    matches = mol.GetSubstructMatches(rot_pattern)

    rotatable_bonds = []
    seen = set()

    for match in matches:
        # Each match is a pair of atom indices
        atom1_idx, atom2_idx = match[0], match[1]

        # Ensure consistent ordering and avoid duplicates
        bond_tuple = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
        if bond_tuple not in seen:
            rotatable_bonds.append(bond_tuple)
            seen.add(bond_tuple)

    return rotatable_bonds


def fragment_molecule(mol: Chem.Mol, coords: np.ndarray) -> List[Fragment]:
    """
    Fragment molecule by rotatable bonds (SigmaDock approach).

    Each fragment is a rigid substructure. Fragments are connected by
    rotatable bonds, which define the torsional degrees of freedom.

    Uses graph connectivity to find fragments by removing rotatable bonds.

    Args:
        mol: RDKit molecule
        coords: [N, 3] atomic coordinates

    Returns:
        List of Fragment objects
    """
    num_atoms = mol.GetNumAtoms()

    # Find rotatable bonds
    rot_bonds_tuples = find_rotatable_bonds_rdkit(mol)

    # If no rotatable bonds, entire molecule is one fragment
    if len(rot_bonds_tuples) == 0:
        com = coords.mean(axis=0)
        local_coords = coords - com
        return [Fragment(
            fragment_id=0,
            atom_indices=list(range(num_atoms)),
            local_coords=local_coords,
            com=com
        )]

    # Build molecular graph WITHOUT rotatable bonds
    G = nx.Graph()
    G.add_nodes_from(range(num_atoms))

    # Add all bonds except rotatable ones
    rot_bonds_set = set(rot_bonds_tuples)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge = (min(i, j), max(i, j))

        # Skip rotatable bonds
        if edge not in rot_bonds_set:
            G.add_edge(i, j)

    # Find connected components = fragments
    connected_components = list(nx.connected_components(G))

    # Build Fragment objects
    fragments = []
    for frag_idx, atom_set in enumerate(connected_components):
        atom_indices = sorted(list(atom_set))

        if len(atom_indices) == 0:
            continue

        # Get fragment coordinates
        frag_coords = coords[atom_indices]

        # Compute center of mass and local coordinates
        com = frag_coords.mean(axis=0)
        local_coords = frag_coords - com

        # Create fragment
        fragment = Fragment(
            fragment_id=frag_idx,
            atom_indices=atom_indices,
            local_coords=local_coords,
            com=com
        )
        fragments.append(fragment)

    # Get bond indices for rotatable bonds
    rot_bond_indices = []
    for atom1_idx, atom2_idx in rot_bonds_tuples:
        bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is not None:
            rot_bond_indices.append(bond.GetIdx())

    # Add connectivity between fragments
    _add_fragment_connectivity(fragments, rot_bond_indices, mol)

    return fragments


def _add_fragment_connectivity(
    fragments: List[Fragment],
    rot_bond_indices: List[int],
    mol: Chem.Mol
):
    """
    Add torsional bond connectivity between fragments.

    Args:
        fragments: List of Fragment objects
        rot_bond_indices: List of rotatable bond indices
        mol: RDKit molecule
    """
    for bond_idx in rot_bond_indices:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_1 = bond.GetBeginAtomIdx()
        atom_2 = bond.GetEndAtomIdx()

        # Find which fragments contain these atoms
        frag_1_id = None
        frag_2_id = None

        for frag in fragments:
            if atom_1 in frag.atom_indices:
                frag_1_id = frag.id
            if atom_2 in frag.atom_indices:
                frag_2_id = frag.id

        # Create torsional bond if both atoms found
        if frag_1_id is not None and frag_2_id is not None:
            tor_bond = TorsionalBond(
                bond_idx=bond_idx,
                frag_1=frag_1_id,
                frag_2=frag_2_id,
                atom_1=atom_1,
                atom_2=atom_2
            )

            # Add to both fragments
            fragments[frag_1_id].torsional_bonds.append(tor_bond)
            fragments[frag_2_id].torsional_bonds.append(tor_bond)


def get_transformation_mask(
    edges: torch.Tensor,
    num_atoms: int
) -> Tuple[torch.Tensor, List[int]]:
    """
    Generate edge-wise rotation mask (DiffDock style).

    Args:
        edges: [2, E] edge index tensor
        num_atoms: Number of atoms

    Returns:
        mask_rotate: [E, N] boolean tensor indicating which atoms rotate around each edge
        rotatable_edges: List of rotatable edge indices
    """
    # Convert edges to numpy for easier processing
    edges_np = edges.numpy() if torch.is_tensor(edges) else edges

    # Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(num_atoms))

    # Add edges (undirected)
    edge_list = []
    for i in range(edges_np.shape[1]):
        u, v = int(edges_np[0, i]), int(edges_np[1, i])
        edge_list.append((min(u, v), max(u, v)))

    # Remove duplicates for undirected graph
    edge_list = list(set(edge_list))
    G.add_edges_from(edge_list)

    mask_rotate = []
    rotatable_edges = []

    # Process edge pairs (bi-directional: i and i+1)
    for i in range(0, edges_np.shape[1], 2):
        u, v = int(edges_np[0, i]), int(edges_np[1, i])

        # Check if removing this edge disconnects graph (rotatable bond)
        G_temp = G.copy()
        G_temp.remove_edge(min(u, v), max(u, v))

        if not nx.is_connected(G_temp):
            # Rotatable! Find which component is smaller
            components = list(nx.connected_components(G_temp))

            # Determine which side of the bond to rotate
            comp_with_u = [c for c in components if u in c][0]
            comp_with_v = [c for c in components if v in c][0]

            # Rotate the smaller component
            if len(comp_with_u) <= len(comp_with_v):
                rotating_atoms = comp_with_u
            else:
                rotating_atoms = comp_with_v

            # Create mask for this edge
            mask = torch.zeros(num_atoms, dtype=torch.bool)
            for atom_idx in rotating_atoms:
                mask[atom_idx] = True

            mask_rotate.append(mask)
            mask_rotate.append(torch.zeros(num_atoms, dtype=torch.bool))  # Empty mask for reverse edge

            rotatable_edges.append(i)
        else:
            # Non-rotatable: both directions get empty masks
            mask_rotate.append(torch.zeros(num_atoms, dtype=torch.bool))
            mask_rotate.append(torch.zeros(num_atoms, dtype=torch.bool))

    mask_rotate = torch.stack(mask_rotate)  # [E, N]
    return mask_rotate, rotatable_edges


def compute_torsion_angles_rdkit(
    mol: Chem.Mol,
    edges: torch.Tensor,
    rotatable_edges: List[int]
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Compute torsion angles for rotatable bonds using RDKit.

    Args:
        mol: RDKit molecule with conformer
        edges: [2, E] edge index tensor
        rotatable_edges: List of rotatable edge indices

    Returns:
        torsion_angles: [M] tensor of torsion angles in radians
        torsion_atoms: List of [a, b, c, d] atom indices for each torsion
    """
    edges_np = edges.numpy() if torch.is_tensor(edges) else edges
    conf = mol.GetConformer()

    torsion_angles = []
    torsion_atoms_list = []

    for edge_idx in rotatable_edges:
        u, v = int(edges_np[0, edge_idx]), int(edges_np[1, edge_idx])

        # Find neighbors for torsion calculation
        atom_u = mol.GetAtomWithIdx(u)
        atom_v = mol.GetAtomWithIdx(v)

        # Get neighbors (excluding the bond itself)
        neighbors_u = [n.GetIdx() for n in atom_u.GetNeighbors() if n.GetIdx() != v]
        neighbors_v = [n.GetIdx() for n in atom_v.GetNeighbors() if n.GetIdx() != u]

        if not neighbors_u or not neighbors_v:
            # Cannot compute torsion (terminal atoms)
            torsion_angles.append(0.0)
            torsion_atoms_list.append([u, u, v, v])  # Dummy
            continue

        # Use first available neighbor for each
        a = neighbors_u[0]
        b = u
        c = v
        d = neighbors_v[0]

        try:
            # GetDihedralDeg returns degrees, convert to radians
            angle_deg = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
            angle_rad = np.deg2rad(angle_deg)
            torsion_angles.append(angle_rad)
            torsion_atoms_list.append([a, b, c, d])
        except:
            # If calculation fails, use 0
            torsion_angles.append(0.0)
            torsion_atoms_list.append([a, b, c, d])

    return torch.tensor(torsion_angles, dtype=torch.float32), torsion_atoms_list


def apply_torsion_updates(
    coords: torch.Tensor,
    edges: torch.Tensor,
    mask_rotate: torch.Tensor,
    rotatable_edges: List[int],
    torsion_updates: torch.Tensor
) -> torch.Tensor:
    """
    Apply torsion angle updates to coordinates (DiffDock style).

    Args:
        coords: [N, 3] coordinates
        edges: [2, E] edge index
        mask_rotate: [E, N] boolean mask
        rotatable_edges: List of rotatable edge indices
        torsion_updates: [M] torsion angle changes in radians

    Returns:
        updated_coords: [N, 3] coordinates after torsion updates
    """
    coords = coords.clone() if torch.is_tensor(coords) else torch.tensor(coords)
    coords_np = coords.numpy()
    edges_np = edges.numpy() if torch.is_tensor(edges) else edges
    mask_np = mask_rotate.numpy() if torch.is_tensor(mask_rotate) else mask_rotate
    torsion_np = torsion_updates.numpy() if torch.is_tensor(torsion_updates) else torsion_updates

    # Apply each torsion update
    for i, edge_idx in enumerate(rotatable_edges):
        if abs(torsion_np[i]) < 1e-6:
            continue  # Skip negligible rotations

        u, v = int(edges_np[0, edge_idx]), int(edges_np[1, edge_idx])

        # Rotation axis: vector along the bond
        axis = coords_np[v] - coords_np[u]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            continue
        axis = axis / axis_norm

        # Rotation angle
        angle = torsion_np[i]

        # Create rotation vector (axis-angle representation)
        rot_vec = axis * angle
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        # Apply rotation to atoms that should rotate
        mask = mask_np[edge_idx]
        coords_np[mask] = (coords_np[mask] - coords_np[v]) @ rot_mat.T + coords_np[v]

    return torch.tensor(coords_np, dtype=torch.float32)


def compute_rigid_transform(
    coords_source: torch.Tensor,
    coords_target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rigid body transformation (translation + rotation) from source to target.

    Uses Kabsch algorithm for optimal rotation.

    Args:
        coords_source: [N, 3] source coordinates
        coords_target: [N, 3] target coordinates

    Returns:
        translation: [3] translation vector (target_com - source_com)
        rotation: [3] rotation in axis-angle representation (SO(3))
    """
    coords_source = coords_source.numpy() if torch.is_tensor(coords_source) else coords_source
    coords_target = coords_target.numpy() if torch.is_tensor(coords_target) else coords_target

    # Center of mass
    com_source = coords_source.mean(axis=0)
    com_target = coords_target.mean(axis=0)

    # Center coordinates
    centered_source = coords_source - com_source
    centered_target = coords_target - com_target

    # Kabsch algorithm: compute optimal rotation matrix
    # H = source^T @ target
    H = centered_source.T @ centered_target
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    R_mat = Vt.T @ U.T

    # Ensure proper rotation (det = 1, not reflection)
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    # Convert rotation matrix to axis-angle representation
    rotation_obj = R.from_matrix(R_mat)
    rotation_rotvec = rotation_obj.as_rotvec()  # [3] axis-angle

    # Translation (after rotation)
    translation = com_target - com_source

    return torch.tensor(translation, dtype=torch.float32), torch.tensor(rotation_rotvec, dtype=torch.float32)


def compute_distance_bounds(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance bounds matrix for conformational constraints.

    Args:
        mol: RDKit molecule

    Returns:
        distance_lower_bounds: [N, N] minimum distance constraints (Angstroms)
        distance_upper_bounds: [N, N] maximum distance constraints (Angstroms)
    """
    bounds_matrix = rdDistGeom.GetMoleculeBoundsMatrix(
        mol,
        set15bounds=True,           # Include 1-5 atom distances (not just 1-4)
        scaleVDW=False,             # Don't scale VDW radii for close atoms
        doTriangleSmoothing=True,   # Apply triangle smoothing for consistency
        useMacrocycle14config=False # Standard configuration
    )

    num_atoms = mol.GetNumAtoms()

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

    return (
        torch.tensor(distance_lower_bounds, dtype=torch.float32),
        torch.tensor(distance_upper_bounds, dtype=torch.float32)
    )


class LigandFeatureExtractor:
    """
    Unified ligand feature extractor with torsion decomposition.

    Features extracted:
    - Atom-level geometric features
    - Torsion angle decomposition (SE(3) + Torsion space)
    - Distance bounds for conformational constraints
    - Rigid body transformation to crystal structure
    """

    def __init__(self, use_hydrogen: bool = False):
        """
        Initialize ligand feature extractor.

        Args:
            use_hydrogen: Whether to include hydrogen atoms
        """
        self.use_hydrogen = use_hydrogen

    def extract_features(
        self,
        mol: Chem.Mol,
        crystal_mol: Chem.Mol,
        mol_featurizer: Optional[MoleculeFeaturizer] = None
    ) -> Dict:
        """
        Extract all ligand features including torsion decomposition.

        Args:
            mol: RDKit molecule (docked pose)
            crystal_mol: RDKit molecule (crystal structure reference)
            mol_featurizer: Pre-initialized MoleculeFeaturizer (optional)

        Returns:
            Dictionary with ligand features including:
                - edges, node_feats, edge_feats: Graph structure
                - coord, crystal_coord: Coordinates
                - distance_lower_bounds, distance_upper_bounds: Conformational constraints
                - mask_rotate, rotatable_edges: Torsion decomposition
                - torsion_angles_x0, torsion_angles_x1, torsion_changes: Torsion values
                - translation, rotation: Rigid body transformation
        """
        # Initialize featurizer if not provided
        if mol_featurizer is None:
            mol_featurizer = MoleculeFeaturizer(mol, hydrogen=self.use_hydrogen)

        # Get graph features
        data = mol_featurizer.get_graph()
        nodes = data[0]
        edges = data[1]

        # Get crystal coordinates
        ref_conf = crystal_mol.GetConformer()
        crystal_coords = torch.tensor(
            [[ref_conf.GetAtomPosition(i).x,
              ref_conf.GetAtomPosition(i).y,
              ref_conf.GetAtomPosition(i).z]
             for i in range(crystal_mol.GetNumAtoms())],
            dtype=torch.float32
        )

        # Compute distance bounds
        distance_lower_bounds, distance_upper_bounds = compute_distance_bounds(crystal_mol)

        # Get transformation mask and rotatable edges
        num_atoms = nodes['coords'].shape[0]
        mask_rotate, rotatable_edges = get_transformation_mask(
            edges['edges'], num_atoms
        )

        # Compute torsion angles for current pose (docked)
        torsion_angles_x0, torsion_atoms_x0 = compute_torsion_angles_rdkit(
            mol, edges['edges'], rotatable_edges
        )

        # Compute torsion angles for crystal structure
        torsion_angles_x1, torsion_atoms_x1 = compute_torsion_angles_rdkit(
            crystal_mol, edges['edges'], rotatable_edges
        )

        # Compute torsion changes FIRST
        torsion_changes = torsion_angles_x1 - torsion_angles_x0
        # Wrap to [-π, π] for shortest rotation path
        torsion_changes = torch.atan2(torch.sin(torsion_changes), torch.cos(torsion_changes))

        # Apply torsion changes to get torsion-corrected coordinates
        coords_torsion_corrected = apply_torsion_updates(
            nodes['coords'], edges['edges'], mask_rotate,
            rotatable_edges, torsion_changes
        )

        # Then compute rigid body transformation
        # From torsion-corrected coordinates to crystal
        translation, rotation = compute_rigid_transform(
            coords_torsion_corrected, crystal_coords
        )

        # ====================================================================
        # Fragment Decomposition (SigmaDock approach)
        # ====================================================================
        # Fragment molecule by rotatable bonds
        coords_np = nodes['coords'].numpy()
        fragments = fragment_molecule(mol, coords_np)

        # Build fragment ID mapping: [N] atom -> fragment ID
        fragment_id = torch.full((num_atoms,), -1, dtype=torch.long)
        fragment_local_coords = torch.zeros(num_atoms, 3, dtype=torch.float32)

        for frag in fragments:
            for atom_idx in frag.atom_indices:
                fragment_id[atom_idx] = frag.id
            # Store local coordinates for this fragment's atoms
            fragment_local_coords[frag.atom_indices] = torch.tensor(
                frag.local_coords, dtype=torch.float32
            )

        # Extract fragment centers of mass: [M, 3]
        fragment_coms = torch.tensor(
            np.stack([frag.com for frag in fragments]),
            dtype=torch.float32
        )

        # Build torsional edge index: connections between fragments
        torsional_edge_indices = []
        torsional_edge_attrs = []

        for frag in fragments:
            for tor_bond in frag.torsional_bonds:
                atom_i = tor_bond.atom_1
                atom_j = tor_bond.atom_2

                # Add bidirectional edges
                torsional_edge_indices.extend([[atom_i, atom_j], [atom_j, atom_i]])

                # Edge features: [type, frag_1, frag_2, distance]
                dist = float(np.linalg.norm(coords_np[atom_i] - coords_np[atom_j]))
                feat = [1.0, float(tor_bond.frag_1), float(tor_bond.frag_2), dist]
                torsional_edge_attrs.extend([feat, feat])

        # Convert to tensors
        if len(torsional_edge_indices) > 0:
            torsional_edge_index = torch.tensor(torsional_edge_indices, dtype=torch.long).T
            torsional_edge_attr = torch.tensor(torsional_edge_attrs, dtype=torch.float32)
        else:
            torsional_edge_index = torch.empty((2, 0), dtype=torch.long)
            torsional_edge_attr = torch.empty((0, 4), dtype=torch.float32)

        return {
            # Graph structure
            'edges': edges['edges'],
            'node_feats': nodes['node_feats'],
            'edge_feats': edges['edge_feats'],

            # Coordinates
            'coord': nodes['coords'],                      # Current pose coordinates
            'crystal_coord': crystal_coords,                # Crystal structure coordinates (reference)

            # Distance constraints
            'distance_lower_bounds': distance_lower_bounds, # [N, N] min distance constraints
            'distance_upper_bounds': distance_upper_bounds, # [N, N] max distance constraints

            # Torsion information (legacy, for compatibility)
            'mask_rotate': mask_rotate,                     # [E, N] boolean mask
            'rotatable_edges': torch.tensor(rotatable_edges, dtype=torch.long),  # [M]
            'torsion_angles_x0': torsion_angles_x0,        # [M] docked pose torsions (radians)
            'torsion_angles_x1': torsion_angles_x1,        # [M] crystal torsions (radians)
            'torsion_changes': torsion_changes,            # [M] target changes (radians)
            'torsion_atoms': torsion_atoms_x0,             # List of [a,b,c,d] atom indices

            # Rigid body transformation (after torsion correction)
            'translation': translation,                     # [3] translation vector
            'rotation': rotation,                           # [3] rotation in axis-angle (SO(3))

            # Fragment decomposition (SigmaDock approach)
            'fragment_id': fragment_id,                    # [N] fragment ID for each atom
            'n_fragments': len(fragments),                 # int: total number of fragments
            'fragment_local_coords': fragment_local_coords, # [N, 3] local coords in fragment frame
            'fragment_coms': fragment_coms,                # [M, 3] center of mass for each fragment
            'torsional_edge_index': torsional_edge_index,  # [2, E_tor] fragment connections
            'torsional_edge_attr': torsional_edge_attr,    # [E_tor, 4] torsional edge features
            'fragments': [frag.to_dict() for frag in fragments],  # List of fragment dicts (for serialization)
        }


# Convenience function for backward compatibility
def extract_ligand_features(
    mol: Chem.Mol,
    crystal_mol: Chem.Mol,
    use_hydrogen: bool = False
) -> Dict:
    """
    Extract ligand features with torsion decomposition.

    Args:
        mol: RDKit molecule (docked pose)
        crystal_mol: RDKit molecule (crystal structure)
        use_hydrogen: Whether to include hydrogen atoms

    Returns:
        Dictionary with ligand features
    """
    extractor = LigandFeatureExtractor(use_hydrogen=use_hydrogen)
    return extractor.extract_features(mol, crystal_mol)
