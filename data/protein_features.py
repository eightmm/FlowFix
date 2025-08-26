import pandas as pd
import torch, os
import numpy as np
import pandas as pd
import freesasa as fs
import torch.nn.functional as F
import warnings
import contextlib
import sys
from io import StringIO

amino_acid_mapping = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                      'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                      'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                      'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',}

amino_acid_mapping_reverse = {v: k for k, v in amino_acid_mapping.items()}
amino_acid_1_to_int = { k: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys())) }
amino_acid_3_to_int = { amino_acid_mapping_reverse[k]: i for i, k in enumerate( sorted( amino_acid_mapping_reverse.keys() ) ) }

amino_acid_1_to_int['X'] = 20
amino_acid_3_to_int['UNK'] = 20


@contextlib.contextmanager
def suppress_freesasa_warnings():
    """
    Context manager to suppress FreeSASA warnings about unknown atoms.
    Captures both Python warnings and stderr output.
    """
    # Suppress Python warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Capture stderr to suppress FreeSASA warnings
        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            yield
        finally:
            sys.stderr = old_stderr


STANDARD_ATOMS = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'UNK': ['N', 'CA', 'C', 'O', 'CB']
}

def standardize_pdb(input_pdb_path, output_pdb_path, remove_hydrogens=True):
    """
    Standardize and reorder PDB file in one step.
    Handles residue numbers with insertion codes (e.g., 123A).
    Removes DNA and RNA residues.
    """
    # DNA and RNA residue names to exclude
    nucleic_acid_residues = {
        # DNA
        'DA', 'DT', 'DG', 'DC', 'DI', 'DU',
        # RNA
        'A', 'U', 'G', 'C', 'I',
        # Modified nucleotides
        'ADE', 'THY', 'GUA', 'CYT', 'URA',
        '1MA', '2MG', '4SU', '5MC', '5MU', 'PSU', 'H2U', 'M2G', 'OMC', 'OMG'
    }

    output_dir = os.path.dirname(output_pdb_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(input_pdb_path, 'r') as f:
        lines = f.readlines()

    protein_residues = {}
    hetatm_residues = {}

    for line in lines:
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_num_str = line[22:27].strip()  # Include insertion code
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]

            if remove_hydrogens and (atom_name.startswith('H') or element.upper() == 'H'):
                continue

            if res_name in ['HOH', 'WAT']:
                continue

            # Skip DNA and RNA residues
            if res_name in nucleic_acid_residues:
                continue

            residue_key = (chain_id, res_num_str, res_name)
            if residue_key not in protein_residues:
                protein_residues[residue_key] = {}
            protein_residues[residue_key][atom_name] = line

        elif line.startswith('HETATM'):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_num_str = line[22:27].strip()  # Include insertion code
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]

            if remove_hydrogens and (atom_name.startswith('H') or element.upper() == 'H'):
                continue

            if res_name in ['HOH', 'WAT']:
                continue

            # Skip DNA and RNA residues
            if res_name in nucleic_acid_residues:
                continue

            residue_key = (chain_id, res_num_str, res_name)
            if residue_key not in hetatm_residues:
                hetatm_residues[residue_key] = {}
            hetatm_residues[residue_key][atom_name] = line

    standardized_lines = []
    atom_counter = 1

    # Sort residues by chain and residue number (handling insertion codes)
    def sort_key(residue_key):
        chain_id, res_num_str, res_name = residue_key
        # Extract numeric part and insertion code
        res_num = int(''.join(filter(str.isdigit, res_num_str)) or 0)
        insertion_code = ''.join(filter(str.isalpha, res_num_str))
        return (chain_id, res_num, insertion_code)

    # Group residues by chain
    protein_by_chain = {}
    for residue_key in protein_residues.keys():
        chain_id = residue_key[0]
        if chain_id not in protein_by_chain:
            protein_by_chain[chain_id] = []
        protein_by_chain[chain_id].append(residue_key)

    # Process protein residues chain by chain
    for chain_id in sorted(protein_by_chain.keys()):
        sorted_residues = sorted(protein_by_chain[chain_id], key=sort_key)
        res_counter = 1  # Reset residue counter for each chain

        for residue_key in sorted_residues:
            chain_id, res_num_str, res_name = residue_key
            residue_atoms = protein_residues[residue_key]

            if res_name in STANDARD_ATOMS:
                for standard_atom in STANDARD_ATOMS[res_name]:
                    if standard_atom in residue_atoms:
                        line = residue_atoms[standard_atom]
                        atom_name = line[12:16].strip()
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
                        temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
                        element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                        new_line = f"ATOM  {atom_counter:5d}  {atom_name:<4s}{res_name} {chain_id}{res_counter:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:>6s}{temp_factor:>6s}          {element:>2s}\n"
                        standardized_lines.append(new_line)
                        atom_counter += 1
            else:
                for atom_name, line in residue_atoms.items():
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
                    temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
                    element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                    new_line = f"ATOM  {atom_counter:5d}  {atom_name:<4s}{res_name} {chain_id}{res_counter:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:>6s}{temp_factor:>6s}          {element:>2s}\n"
                    standardized_lines.append(new_line)
                    atom_counter += 1

            res_counter += 1

    # Group HETATM residues by chain
    hetatm_by_chain = {}
    for residue_key in hetatm_residues.keys():
        chain_id = residue_key[0]
        if chain_id not in hetatm_by_chain:
            hetatm_by_chain[chain_id] = []
        hetatm_by_chain[chain_id].append(residue_key)

    # Process HETATM residues chain by chain
    for chain_id in sorted(hetatm_by_chain.keys()):
        sorted_residues = sorted(hetatm_by_chain[chain_id], key=sort_key)
        hetatm_counter = 1  # Reset HETATM counter for each chain

        for residue_key in sorted_residues:
            chain_id, res_num_str, res_name = residue_key
            residue_atoms = hetatm_residues[residue_key]

            for atom_name, line in residue_atoms.items():
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = line[54:60].strip() if len(line) > 54 else "1.00"
                temp_factor = line[60:66].strip() if len(line) > 60 else "0.00"
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                new_line = f"HETATM{atom_counter:5d} {atom_name:<4s}  {res_name}  {hetatm_counter:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  {occupancy:>6s}{temp_factor:>6s}           {element:>2s}\n"
                standardized_lines.append(new_line)
                atom_counter += 1

            hetatm_counter += 1

    with open(output_pdb_path, 'w') as f:
        f.writelines(standardized_lines)

    return output_pdb_path

class PDBParserSim:
    def __init__(self, pdb):
        self.pdb = pdb
        self.protein_indices, self.hetero_indices, self.protein_atom_info, self.hetero_atom_info = self._get_data_from_pdb(pdb)

    def _get_data_from_pdb(self, pdb):
        with open(pdb, 'r') as f:
            lines = f.read().split('\n')

        protein_index, hetero_index = [], []
        protein_data,  hetero_data = { 'coord': [] }, { 'coord': [] }

        for line in lines:
            record_type = line[:6].strip()  # Changed to handle HETATM properly
            if record_type in ['ATOM', 'HETATM'] and len(line) > 13 and line[13] != 'H' and line[17:20].strip() != 'HOH':
                atom_type = line[12:17].strip()
                res_type  = amino_acid_3_to_int.get( line[17:20].strip(), 20 )
                chain_id  = line[21]
                res_num   = int( line[22:26] )
                # res_num   = line[22:27].strip()
                xyz       = [ float( line[idx:idx + 8] ) for idx in range(30, 54, 8) ]

                if record_type == 'ATOM':
                    protein_index.append( (chain_id, res_num, res_type, atom_type ) )
                    protein_data['coord'].append( xyz )

                elif record_type == 'HETATM':
                    hetero_index.append( ('HETERO', res_num, res_type, atom_type ) )
                    hetero_data['coord'].append( xyz )

        protein_index = pd.MultiIndex.from_tuples(protein_index, names=['chain', 'res_num', 'AA', 'atom'])
        hetero_index  = pd.MultiIndex.from_tuples(hetero_index,  names=['chain', 'res_num', 'AA', 'atom'])

        protein_atom_info = pd.DataFrame(protein_data, index=protein_index)
        hetero_atom_info  = pd.DataFrame(hetero_data,  index=hetero_index)

        return (protein_index, hetero_index, protein_atom_info, hetero_atom_info)

    def get_atom(self, prot=True, hetero=False):
        if prot and hetero:
            return list( self.protein_indices ) + list( self.hetero_indices )
        elif prot:
            return list( self.protein_indices )
        elif hetero:
            return list( self.hetero_indices )

    def get_all_atom_info(self, prot=True, hetero=False):
        if prot and hetero:
            return pd.concat( (self.protein_atom_info, self.hetero_atom_info) )
        elif prot:
            return self.protein_atom_info
        elif hetero:
            return self.hetero_atom_info

    def get_residue(self):
        return sorted( set( list( [ (chain, num, res) for chain, num, res, atom in self.protein_indices ] ) ) )

    def get_residue_coord(self, index):
        return self.get_all_atom_info(prot=True, hetero=False).coord.xs(index)

    def get_atom_with_name(self, name):
        return sorted( set( list( [ (chain, num, res) for chain, num, res, atom in self.protein_indices if atom == name ]  ) ) )

    def get_terminal_flag(self):
        """
        Get terminal flags for each residue.
        Returns N-terminal and C-terminal flags separately.
        """
        # Get unique residues (chain, res_num, res_type)
        residues = self.get_residue()

        n_terminal = []
        c_terminal = []

        # Group residues by chain
        chain_residues = {}
        for chain, num, res in residues:

            if chain not in chain_residues:
                chain_residues[chain] = []
            chain_residues[chain].append((num, res))

        # Sort residues within each chain by residue number
        for chain in chain_residues:
            chain_residues[chain].sort(key=lambda x: x[0])

        for chain, num, res in residues:
            chain_res_list = chain_residues[chain]

            # Check if this is N-terminal (first residue in chain)
            is_n_terminal = (num, res) == chain_res_list[0]
            # Check if this is C-terminal (last residue in chain)
            is_c_terminal = (num, res) == chain_res_list[-1]

            n_terminal.append(is_n_terminal)
            c_terminal.append(is_c_terminal)

        return torch.tensor(n_terminal, dtype=torch.bool), torch.tensor(c_terminal, dtype=torch.bool)

    def get_relative_position(self, cutoff=32, onehot=True):
        """
        Get relative position of each residue within each chain.
        Returns a tensor where same chain residues have relative positions and different chains have -1.
        """
        residues = self.get_residue()
        num_residues = len(residues)

        # Initialize with -1 (for different chains)
        relative_positions = torch.ones((num_residues, num_residues)) * 33

        # Group residues by chain with their indices
        chain_residue_indices = {}
        for idx, (chain, num, res) in enumerate(residues):
            if chain not in chain_residue_indices:
                chain_residue_indices[chain] = []
            chain_residue_indices[chain].append(idx)

        # Calculate relative positions within each chain
        for chain, indices in chain_residue_indices.items():
            num_chain_residues = len(indices)
            arrange = torch.arange(num_chain_residues)
            chain_relative_positions = (arrange[:, None] - arrange[None, :]).abs()
            chain_relative_positions = torch.where(chain_relative_positions > cutoff, 33, chain_relative_positions)

            # Fill in the relative positions for same chain residues
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    relative_positions[idx_i, idx_j] = chain_relative_positions[i, j]

        if onehot:
            relative_positions = relative_positions.long()
            relative_positions = F.one_hot(relative_positions, num_classes=cutoff + 2)
            relative_positions = relative_positions.float()

        return relative_positions

    def _dihedral(self, X, eps=1e-8):
        shape_X = X.shape
        X = X.reshape( shape_X[0] * shape_X[1], shape_X[2] )

        U = F.normalize( X[1:, :] - X[:-1,:], dim=-1)
        u_2 = U[:-2,:]
        u_1 = U[1:-1,:]
        u_0 = U[2:,:]

        n_2 = F.normalize(torch.cross(u_2, u_1, dim=1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=1), dim=-1)

        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)

        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)

        return D.view( (int(D.size(0)/shape_X[1]), shape_X[1] ) )

    def get_SASA(self):
        # Suppress FreeSASA warnings about unknown atoms
        with suppress_freesasa_warnings():
            sasas = [
                [
                    values.total / 350,
                    values.polar / 350,
                    values.apolar / 350,
                    values.mainChain / 350,
                    values.sideChain / 350,
                    values.relativeTotal,
                    values.relativePolar,
                    values.relativeApolar,
                    values.relativeMainChain,
                    values.relativeSideChain
                ]
                for chain, residues in fs.calc( fs.Structure( self.pdb ) ).residueAreas().items()
                for residue, values in residues.items()
            ]

        return torch.nan_to_num( torch.as_tensor( sasas ) )

    def get_dihedral_angle(self, coords, res_type):
        """
        coords: (num_residues, 15, 3)
        res_type: (num_residues, )

        chi1: N A B G == [0, 1, 4, 5]
        chi2: A B G D == [1, 4, 5, 6]
        chi3: B G D E == [4, 5, 6, 7]
        chi4: G D E Z == [5, 6, 7, 8]
        chi5: D E Z H == [6, 7, 8, 9]
        return: (num_residues, 8[:3 == bb, 3: == sc]), (num_residues, 5)
        """

        chi_indices = {
            'chi1': torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            'chi2': torch.tensor([2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19]),
            'chi3': torch.tensor([3, 8, 10, 13, 14]),
            'chi4': torch.tensor([8, 14]),
            'chi5': torch.tensor([14])
        }

        is_ILE = torch.isin(res_type, torch.tensor([7])).int().unsqueeze(1).unsqueeze(2)
        is_not_ILE = 1 - is_ILE

        has_chi = torch.stack([torch.isin(res_type, chi_indices[f'chi{i}']).int() for i in range(1, 6)], dim=1)

        N_CA_C = coords[:, :3, :]
        backbone_dihedrals = self._dihedral(N_CA_C)

        N_A_B_G_D_E_Z_ILE = torch.cat( [coords[:, :2, :], coords[:, 4:6, :], coords[:, 7:11, :]], dim=1 ) * is_ILE
        N_A_B_G_D_E_Z_no_ILE = torch.cat( [coords[:, :2, :], coords[:, 4:10, :]], dim=1 ) * is_not_ILE

        N_A_B_G_D_E_Z = N_A_B_G_D_E_Z_ILE + N_A_B_G_D_E_Z_no_ILE

        side_chain_dihedrals = self._dihedral(N_A_B_G_D_E_Z)[:, 1:-2] * has_chi

        dihedrals = torch.cat( [backbone_dihedrals, side_chain_dihedrals], dim=1 )

        return dihedrals, has_chi

    def _self_distance(self, coords):
        coords = torch.cat( [ coords[:, :4, :], coords[:, -1:, :] ], dim=1 )
        distance = torch.cdist( coords, coords )
        mask_sca = torch.triu( torch.ones_like( distance ), diagonal=1 ).bool()
        distance = torch.masked_select( distance, mask_sca ).view( distance.shape[0], -1 )

        return torch.nan_to_num(distance)

    def _self_vector(self, coords):
        coords = torch.cat( [ coords[:, :4, :], coords[:, -1:, :] ], dim=1 )
        vectors = coords[:, None] - coords[:, :, None]
        vectors = vectors.view( coords.shape[0], 25, 3 )
        vectors = torch.index_select(vectors, 1, torch.tensor([ 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]))

        return torch.nan_to_num( vectors )

    def _forward_reverse(self, coord, terminal_flags):
        ca_coords = coord[:, 1, :]  # CA coordinates
        sc_coords = coord[:, -1, :] # SC coordinates

        n_terminal, c_terminal = terminal_flags

        forward_vector = torch.zeros(coord.shape[0], 4, 3)
        forward_distance = torch.zeros(coord.shape[0], 4)
        reverse_vector = torch.zeros(coord.shape[0], 4, 3)
        reverse_distance = torch.zeros(coord.shape[0], 4)

        if coord.shape[0] > 1:
            # Forward vectors (current to next residue)
            ca_diff = ca_coords[1:] - ca_coords[:-1]
            sc_diff = sc_coords[1:] - sc_coords[:-1]
            ca_sc_diff = sc_coords[1:] - ca_coords[:-1]
            sc_ca_diff = ca_coords[1:] - sc_coords[:-1]

            forward_vector[:-1] = torch.stack([ca_diff, sc_diff, ca_sc_diff, sc_ca_diff], dim=1)
            forward_distance[:-1] = torch.norm(forward_vector[:-1], dim=-1)

            # Mask out forward vectors for C-terminal residues
            c_mask = ~c_terminal[:-1]
            forward_vector[:-1] *= c_mask[:, None, None]
            forward_distance[:-1] *= c_mask[:, None]

            # Reverse vectors (current to previous residue)
            reverse_vector[1:] = torch.stack([-ca_diff, -sc_diff, ca_coords[:-1] - sc_coords[1:], sc_coords[:-1] - ca_coords[1:]], dim=1)
            reverse_distance[1:] = torch.norm(reverse_vector[1:], dim=-1)

            # Mask out reverse vectors for N-terminal residues
            n_mask = (~n_terminal[1:])
            reverse_vector[1:] *= n_mask[:, None, None]
            reverse_distance[1:] *= n_mask[:, None]

        forward_vector = torch.nan_to_num(forward_vector)
        reverse_vector = torch.nan_to_num(reverse_vector)
        forward_distance = torch.nan_to_num(forward_distance)
        reverse_distance = torch.nan_to_num(reverse_distance)

        return forward_vector, forward_distance, reverse_vector, reverse_distance

    def _interaction_distance(self, coords, cutoff=8):
        coord_CA = coords[:, 1:2, :].transpose(0, 1)
        coord_SC = coords[:, -1:, :].transpose(0, 1)
        mask = ( 1 - torch.eye( coords.shape[0] ) ).int()

        dm_CA_CA = torch.cdist( coord_CA, coord_CA )[0]
        dm_SC_SC = torch.cdist( coord_SC, coord_SC )[0]
        dm_CA_SC = torch.cdist( coord_CA, coord_SC )[0]
        dm_SC_CA = torch.cdist( coord_SC, coord_CA )[0]

        adj_CA_CA = (dm_CA_CA < cutoff) * mask
        adj_SC_SC = (dm_SC_SC < cutoff) * mask
        adj_CA_SC = (dm_CA_SC < cutoff) * mask
        adj_SC_CA = (dm_SC_CA < cutoff) * mask

        adj = adj_CA_CA | adj_SC_SC | adj_CA_SC | adj_SC_CA

        dm_all = torch.stack( (dm_CA_CA, dm_SC_SC, dm_CA_SC, dm_SC_CA), dim=-1 )
        dm_select = dm_all * adj[:, :, None]

        return torch.nan_to_num(dm_select), adj

    def _interaction_vectors(self, coords, adj):
        coord_CA_SC = torch.cat( [coords[:, 1:2, :], coords[:, -1:, :]], dim=1 )
        coord_SC_CA = torch.cat( [coords[:, -1:, :], coords[:, 1:2, :]], dim=1 )

        vector1 = coord_CA_SC[:, None, :] - coord_CA_SC[:, :, :]
        vector3 = coord_CA_SC[:, None, :] - coord_SC_CA[:, :, :]
        vectors = torch.cat( [vector1, -vector1, vector3, -vector3], dim=2).nan_to_num()
        vectors = vectors * adj[:, :, None, None]

        return vectors

    def _residue_features(self, coords, residue_types):
        residue_one_hot = F.one_hot(residue_types, num_classes=21)

        self_distance = self._self_distance(coords)
        self_vector = self._self_vector(coords)

        dihedrals, has_chi_angles = self.get_dihedral_angle(coords, residue_types)
        dihedrals = torch.cat([torch.cos(dihedrals), torch.sin(dihedrals)], dim=-1)

        sasa = self.get_SASA()

        terminal_flags = self.get_terminal_flag()

        forward_vector, forward_distance, reverse_vector, reverse_distance = self._forward_reverse(coords, terminal_flags)

        rf_vector = torch.cat([forward_vector, reverse_vector], dim=1) # 8
        rf_distance = torch.cat([forward_distance, reverse_distance], dim=1)

        node_scalar_features = (
            residue_one_hot,
            self_distance,
            dihedrals,
            has_chi_angles,
            sasa,
            rf_distance,
            torch.stack(terminal_flags, dim=1)
            )

        node_vector_features = (
            self_vector,
            rf_vector,
            )

        return node_scalar_features, node_vector_features

    def _interaction_features(self, coords, distance_cutoff=8, relative_position_cutoff=32):
        relative_position = self.get_relative_position(cutoff=relative_position_cutoff, onehot=True)
        distance_adj, adj = self._interaction_distance(coords, cutoff=distance_cutoff)
        interaction_vectors = self._interaction_vectors(coords, adj)

        sparse = distance_adj.to_sparse(sparse_dim=2)
        src, dst = sparse.indices()
        distance = sparse.values()

        relative_position = relative_position[src, dst]
        vectors = interaction_vectors[src, dst, :]

        edges = (src, dst)
        edge_scalar_features = (distance, relative_position)
        edge_vector_features = (vectors, )

        return edges, edge_scalar_features, edge_vector_features

    def get_features(self):
        residues = self.get_residue()
        coords = torch.zeros( len(residues), 15, 3 )
        residue_types = torch.from_numpy( np.array( residues )[:, 2].astype(int) )

        for idx, residue in enumerate( residues ):
            residue_coord = torch.as_tensor( self.get_residue_coord(residue).tolist() )
            coords[idx, :residue_coord.shape[0], :] = residue_coord
            coords[idx, -1, :] = residue_coord[4:, :].mean(0) if residue_coord.shape[0] > 4 else residue_coord[:, :].mean(0)

        coords_CA = coords[:, 1:2, :]
        coords_SC = coords[:, -1:, :]

        coord = torch.cat( [coords_CA, coords_SC], dim=1 )
        node_scalar_features, node_vector_features = self._residue_features( coords, residue_types )
        edges, edge_scalar_features, edge_vector_features = self._interaction_features( coords, distance_cutoff=8, relative_position_cutoff=32 )

        node = {'coord': coord, 'node_scalar_features': node_scalar_features, 'node_vector_features': node_vector_features}
        edge = {'edges': edges, 'edge_scalar_features': edge_scalar_features, 'edge_vector_features': edge_vector_features}
        return node, edge