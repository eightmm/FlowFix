"""
Model building utilities for FlowFix
"""

import torch
from src.models.flowmatching import ProteinLigandFlowMatching


def build_model(model_config, device):
    """
    Build ProteinLigandFlowMatching model from configuration.

    Args:
        model_config: Model configuration dict
        device: torch device

    Returns:
        Initialized model on specified device
    """
    model = ProteinLigandFlowMatching(
        # Protein network parameters (with vector features)
        protein_input_scalar_dim=model_config.get('protein_input_scalar_dim', 76),
        protein_input_vector_dim=model_config.get('protein_input_vector_dim', 31),
        protein_input_edge_scalar_dim=model_config.get('protein_input_edge_scalar_dim', 39),
        protein_input_edge_vector_dim=model_config.get('protein_input_edge_vector_dim', 8),
        protein_hidden_scalar_dim=model_config.get('protein_hidden_scalar_dim', 128),
        protein_hidden_vector_dim=model_config.get('protein_hidden_vector_dim', 32),
        protein_output_scalar_dim=model_config.get('protein_output_scalar_dim', 128),
        protein_output_vector_dim=model_config.get('protein_output_vector_dim', 32),
        protein_num_layers=model_config.get('protein_num_layers', model_config.get('protein_num_blocks', 3)),

        # Ligand network parameters (NO vector features)
        ligand_input_scalar_dim=model_config.get('ligand_input_scalar_dim', 121),
        ligand_input_edge_scalar_dim=model_config.get('ligand_input_edge_scalar_dim', 44),
        ligand_hidden_scalar_dim=model_config.get('ligand_hidden_scalar_dim', 128),
        ligand_hidden_vector_dim=model_config.get('ligand_hidden_vector_dim', 16),
        ligand_output_scalar_dim=model_config.get('ligand_output_scalar_dim', 128),
        ligand_output_vector_dim=model_config.get('ligand_output_vector_dim', 16),
        ligand_num_layers=model_config.get('ligand_num_layers', model_config.get('ligand_num_blocks', 3)),

        # Interaction network parameters
        interaction_num_heads=model_config.get('interaction_num_heads', 8),
        interaction_num_layers=model_config.get('interaction_num_layers', 2),
        interaction_num_rbf=model_config.get('interaction_num_rbf', 32),
        interaction_pair_dim=model_config.get('interaction_pair_dim', 64),

        # Velocity predictor parameters
        velocity_hidden_scalar_dim=model_config.get('velocity_hidden_scalar_dim', 128),
        velocity_hidden_vector_dim=model_config.get('velocity_hidden_vector_dim', 16),
        velocity_num_layers=model_config.get('velocity_num_layers', 4),

        # General parameters (unified hidden_dim for non-equivariant features)
        hidden_dim=model_config.get('hidden_dim', model_config.get('interaction_hidden_dim', 256)),
        dropout=model_config.get('dropout', 0.1),

        # ESM embedding parameters
        use_esm_embeddings=model_config.get('use_esm_embeddings', True),
        esmc_dim=model_config.get('esmc_dim', 1152),
        esm3_dim=model_config.get('esm3_dim', 1536)
    ).to(device)

    return model
