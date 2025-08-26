import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# Atom type mapping
ATOM_TYPES = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8,
    'P': 9, 'B': 10, 'Si': 11, 'Se': 12, 'Unknown': 13
}

# Hybridization mapping
HYBRIDIZATION_TYPES = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 6
}

# Bond type mapping
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def read_mol2(mol2_path: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int, str]]]:
    """
    Read MOL2 file and extract atom coordinates, types, and bonds.
    
    Args:
        mol2_path: Path to MOL2 file
        
    Returns:
        coords: (N, 3) array of atomic coordinates
        atom_types: List of atom type strings
        bonds: List of (atom1_idx, atom2_idx, bond_type) tuples
    """
    coords = []
    atom_types = []
    bonds = []
    
    with open(mol2_path, 'r') as f:
        lines = f.readlines()
    
    in_atom_section = False
    in_bond_section = False
    
    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            in_atom_section = True
            in_bond_section = False
            continue
        elif line.startswith('@<TRIPOS>BOND'):
            in_atom_section = False
            in_bond_section = True
            continue
        elif line.startswith('@<TRIPOS>'):
            in_atom_section = False
            in_bond_section = False
            continue
        
        if in_atom_section and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                atom_type = parts[5].split('.')[0]  # Remove hybridization info
                coords.append([x, y, z])
                atom_types.append(atom_type)
        
        elif in_bond_section and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                atom1 = int(parts[1]) - 1  # Convert to 0-indexed
                atom2 = int(parts[2]) - 1
                bond_type = parts[3]
                bonds.append((atom1, atom2, bond_type))
    
    return np.array(coords), atom_types, bonds


def read_sdf(sdf_path: str) -> Chem.Mol:
    """
    Read SDF file and return RDKit molecule object.
    
    Args:
        sdf_path: Path to SDF file
        
    Returns:
        RDKit Mol object
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(suppl)
    if mol is None:
        # Try without removing hydrogens
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
        mol = next(suppl)
    return mol


def get_atom_features(atom: Chem.Atom) -> torch.Tensor:
    """
    Get atom-level features for a single atom.
    
    Args:
        atom: RDKit Atom object
        
    Returns:
        Feature vector for the atom
    """
    # Atom type one-hot
    atom_type = atom.GetSymbol()
    atom_type_idx = ATOM_TYPES.get(atom_type, ATOM_TYPES['Unknown'])
    atom_type_onehot = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(ATOM_TYPES))
    
    # Degree (number of bonds)
    degree = atom.GetDegree()
    degree_onehot = F.one_hot(torch.tensor(min(degree, 5)), num_classes=6)
    
    # Hybridization
    hybridization = atom.GetHybridization()
    hybridization_idx = HYBRIDIZATION_TYPES.get(hybridization, 6)
    hybridization_onehot = F.one_hot(torch.tensor(hybridization_idx), num_classes=7)
    
    # Formal charge
    formal_charge = atom.GetFormalCharge()
    charge_encoded = torch.tensor([formal_charge], dtype=torch.float32)
    
    # Aromaticity
    is_aromatic = torch.tensor([atom.GetIsAromatic()], dtype=torch.float32)
    
    # Ring membership
    is_in_ring = torch.tensor([atom.IsInRing()], dtype=torch.float32)
    
    # Implicit valence
    implicit_valence = atom.GetImplicitValence()
    valence_onehot = F.one_hot(torch.tensor(min(implicit_valence, 5)), num_classes=6)
    
    # Number of hydrogens
    num_h = atom.GetTotalNumHs()
    num_h_onehot = F.one_hot(torch.tensor(min(num_h, 4)), num_classes=5)
    
    # Concatenate all features
    features = torch.cat([
        atom_type_onehot.float(),
        degree_onehot.float(),
        hybridization_onehot.float(),
        charge_encoded,
        is_aromatic,
        is_in_ring,
        valence_onehot.float(),
        num_h_onehot.float()
    ])
    
    return features


def get_bond_features(bond: Chem.Bond) -> torch.Tensor:
    """
    Get bond-level features for a single bond.
    
    Args:
        bond: RDKit Bond object
        
    Returns:
        Feature vector for the bond
    """
    # Bond type
    bond_type = bond.GetBondType()
    bond_type_idx = BOND_TYPES.get(bond_type, 0)
    bond_type_onehot = F.one_hot(torch.tensor(bond_type_idx), num_classes=4)
    
    # Conjugation
    is_conjugated = torch.tensor([bond.GetIsConjugated()], dtype=torch.float32)
    
    # Ring membership
    is_in_ring = torch.tensor([bond.IsInRing()], dtype=torch.float32)
    
    # Stereochemistry
    stereo = bond.GetStereo()
    stereo_onehot = F.one_hot(torch.tensor(int(stereo)), num_classes=6)
    
    # Concatenate all features
    features = torch.cat([
        bond_type_onehot.float(),
        is_conjugated,
        is_in_ring,
        stereo_onehot.float()
    ])
    
    return features


def mol_to_graph(mol: Chem.Mol) -> Dict[str, torch.Tensor]:
    """
    Convert RDKit molecule to graph representation.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary containing graph data
    """
    if mol is None:
        raise ValueError("Invalid molecule")
    
    # Get 3D coordinates (generate if not present)
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    coords = torch.tensor(coords, dtype=torch.float32)
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    atom_features = torch.stack(atom_features)
    
    # Get bonds and edge features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions for undirected graph
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)  # Same features for both directions
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.stack(edge_features)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 12), dtype=torch.float32)
    
    # Calculate additional molecular properties
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    
    graph_data = {
        'coords': coords,  # (N, 3)
        'x': atom_features,  # (N, feat_dim)
        'edge_index': edge_index,  # (2, E)
        'edge_attr': edge_attr,  # (E, edge_feat_dim)
        'num_atoms': torch.tensor(mol.GetNumAtoms(), dtype=torch.long),
        'mol_weight': torch.tensor(mol_weight, dtype=torch.float32),
        'logp': torch.tensor(logp, dtype=torch.float32),
        'tpsa': torch.tensor(tpsa, dtype=torch.float32),
        'num_rotatable_bonds': torch.tensor(num_rotatable_bonds, dtype=torch.long),
    }
    
    return graph_data


def process_ligand_mol2(mol2_path: str) -> Dict[str, torch.Tensor]:
    """
    Process ligand from MOL2 file.
    
    Args:
        mol2_path: Path to MOL2 file
        
    Returns:
        Dictionary containing ligand graph data
    """
    coords, atom_types, bonds = read_mol2(mol2_path)
    coords = torch.tensor(coords, dtype=torch.float32)
    
    # Create atom features
    atom_features = []
    for atom_type in atom_types:
        atom_type_idx = ATOM_TYPES.get(atom_type, ATOM_TYPES['Unknown'])
        atom_type_onehot = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(ATOM_TYPES))
        atom_features.append(atom_type_onehot.float())
    atom_features = torch.stack(atom_features)
    
    # Create edge indices
    edge_indices = []
    for atom1, atom2, bond_type in bonds:
        edge_indices.append([atom1, atom2])
        edge_indices.append([atom2, atom1])  # Undirected
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return {
        'coords': coords,
        'x': atom_features,
        'edge_index': edge_index,
        'num_atoms': torch.tensor(len(atom_types), dtype=torch.long),
        'center': coords.mean(dim=0)
    }


def process_ligand_sdf(sdf_path: str) -> Dict[str, torch.Tensor]:
    """
    Process ligand from SDF file using RDKit.
    
    Args:
        sdf_path: Path to SDF file
        
    Returns:
        Dictionary containing ligand graph data
    """
    mol = read_sdf(sdf_path)
    if mol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")
    
    # Add hydrogens if missing
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates if not present
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
    
    graph_data = mol_to_graph(mol)
    graph_data['center'] = graph_data['coords'].mean(dim=0)
    
    return graph_data


def get_ligand_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    """
    Get Morgan fingerprint for a molecule.
    
    Args:
        mol: RDKit Mol object
        radius: Radius for Morgan fingerprint
        n_bits: Number of bits in fingerprint
        
    Returns:
        Fingerprint as tensor
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    fp_array = np.array(fp)
    return torch.tensor(fp_array, dtype=torch.float32)