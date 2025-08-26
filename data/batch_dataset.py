"""
Improved batching for protein-ligand complexes using PyTorch Geometric.
"""
import torch
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any


def create_graph_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Create properly batched graph data for protein-ligand complexes.
    
    This handles variable-sized proteins and ligands by creating batch indices
    and concatenating features along the node dimension.
    """
    batch_size = len(batch_list)
    device = batch_list[0]['ligand_coords_t'].device if torch.is_tensor(batch_list[0]['ligand_coords_t']) else 'cpu'
    
    # Collect all data
    batched_data = {}
    
    # Protein data - concatenate along node dimension
    protein_coords_ca_list = []
    protein_coords_sc_list = []
    protein_features_list = []
    protein_batch_list = []
    
    # Ligand data - concatenate along node dimension  
    ligand_coords_t_list = []
    ligand_coords_0_list = []
    ligand_features_list = []
    vector_field_list = []
    ligand_batch_list = []
    
    # Time and other scalars
    t_list = []
    
    # Track offsets for edge indices
    protein_offset = 0
    ligand_offset = 0
    
    for i, sample in enumerate(batch_list):
        # Protein data
        if 'protein_coord_CA' in sample:
            if isinstance(sample['protein_coord_CA'], list):
                protein_ca = sample['protein_coord_CA'][0] if len(sample['protein_coord_CA']) > 0 else torch.empty(0, 3)
            else:
                protein_ca = sample['protein_coord_CA']
            
            if isinstance(sample['protein_coord_SC'], list):
                protein_sc = sample['protein_coord_SC'][0] if len(sample['protein_coord_SC']) > 0 else torch.empty(0, 3)
            else:
                protein_sc = sample['protein_coord_SC']
                
            if isinstance(sample['protein_x'], list):
                protein_feat = sample['protein_x'][0] if len(sample['protein_x']) > 0 else torch.empty(0, sample['protein_x'][0].shape[-1])
            else:
                protein_feat = sample['protein_x']
            
            n_protein = protein_ca.shape[0]
            
            protein_coords_ca_list.append(protein_ca)
            protein_coords_sc_list.append(protein_sc)
            protein_features_list.append(protein_feat)
            protein_batch_list.append(torch.full((n_protein,), i, dtype=torch.long))
            
        # Ligand data
        if 'ligand_coords_t' in sample:
            ligand_coords_t = sample['ligand_coords_t']
            if ligand_coords_t.dim() == 3:  # Remove batch dim if present
                ligand_coords_t = ligand_coords_t[0]
            
            ligand_coords_0 = sample.get('ligand_coords_0', ligand_coords_t.clone())
            if ligand_coords_0.dim() == 3:
                ligand_coords_0 = ligand_coords_0[0]
            
            ligand_features = sample['ligand_x']
            if ligand_features.dim() == 3:
                ligand_features = ligand_features[0]
            
            vector_field = sample.get('vector_field', torch.zeros_like(ligand_coords_t))
            if vector_field.dim() == 3:
                vector_field = vector_field[0]
            
            n_ligand = ligand_coords_t.shape[0]
            
            ligand_coords_t_list.append(ligand_coords_t)
            ligand_coords_0_list.append(ligand_coords_0)
            ligand_features_list.append(ligand_features)
            vector_field_list.append(vector_field)
            ligand_batch_list.append(torch.full((n_ligand,), i, dtype=torch.long))
        
        # Time
        t = sample['t']
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_list.append(t)
    
    # Concatenate all
    if len(protein_coords_ca_list) > 0:
        batched_data['protein_coords'] = torch.stack([
            torch.cat(protein_coords_ca_list, dim=0),
            torch.cat(protein_coords_sc_list, dim=0)
        ], dim=1)  # (N_total_protein, 2, 3)
        batched_data['protein_features'] = torch.cat(protein_features_list, dim=0)
        batched_data['protein_batch'] = torch.cat(protein_batch_list, dim=0)
    
    if len(ligand_coords_t_list) > 0:
        batched_data['ligand_coords_t'] = torch.cat(ligand_coords_t_list, dim=0)
        batched_data['ligand_coords_0'] = torch.cat(ligand_coords_0_list, dim=0)
        batched_data['ligand_features'] = torch.cat(ligand_features_list, dim=0)
        batched_data['vector_field'] = torch.cat(vector_field_list, dim=0)
        batched_data['ligand_batch'] = torch.cat(ligand_batch_list, dim=0)
    
    batched_data['t'] = torch.cat(t_list, dim=0)  # (batch_size,)
    batched_data['batch_size'] = batch_size
    
    return batched_data


def collate_fn_batched(batch: List[Dict]) -> Dict:
    """
    Improved collate function that properly batches protein-ligand data.
    """
    return create_graph_batch(batch)