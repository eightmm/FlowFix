"""
Protein Feature Extraction with ESM Integration

Simplified module for dataset usage:
1. Geometric features from PDB structures
2. ESM embeddings (ESMC + ESM3) - always extracted together

No caching, no file I/O, no standardization - just pure feature extraction.
"""

import torch
from pathlib import Path
from typing import Dict, Union, Literal
import logging

from featurizer import ProteinFeaturizer

logger = logging.getLogger(__name__)


# ============================================================================
# ESM Embedding Extraction
# ============================================================================

class ESMEmbeddingExtractor:
    """Extract protein embeddings using ESM3 or ESMC models."""

    def __init__(
        self,
        model_type: Literal["esm3", "esmc"] = "esmc",
        model_name: str = "esmc_600m",
        device: str = "cuda",
    ):
        """
        Initialize ESM embedding extractor.

        Args:
            model_type: "esm3" or "esmc"
            model_name: Model variant name
                ESMC: "esmc_600m" (1152-dim), "esmc_300m" (960-dim)
                ESM3: "esm3-open" (1536-dim)
            device: "cuda" or "cpu"
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load ESM3 or ESMC model."""
        try:
            if self.model_type == "esm3":
                from esm.models.esm3 import ESM3
                logger.info(f"Loading ESM3 model: {self.model_name}")

                self.model = ESM3.from_pretrained(self.model_name)
                if self.device == "cpu":
                    self.model = self.model.float()
                self.model = self.model.to(self.device)

            elif self.model_type == "esmc":
                from esm.models.esmc import ESMC
                logger.info(f"Loading ESMC model: {self.model_name}")

                self.model = ESMC.from_pretrained(self.model_name).to(self.device)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            logger.info(f"Model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Failed to import ESM models. Please install: pip install esm\n"
                f"For faster inference: pip install flash-attn --no-build-isolation\n"
                f"Error: {e}"
            )

    @torch.no_grad()
    def extract_embeddings(
        self,
        sequence: str,
    ) -> torch.Tensor:
        """
        Extract per-residue embeddings from a protein sequence.

        Args:
            sequence: Protein sequence (e.g., "MKTIIALSYIFCLVFA")

        Returns:
            Per-residue embeddings [L, D]
        """
        from esm.sdk.api import ESMProtein, LogitsConfig

        protein = ESMProtein(sequence=sequence)

        # Encode and extract embeddings
        if self.model_type == "esmc":
            protein_tensor = self.model.encode(protein)
            logits_output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            embeddings = logits_output.embeddings  # [L, D]

        elif self.model_type == "esm3":
            from esm.sdk.api import SamplingConfig

            protein_tensor = self.model.encode(protein)
            output = self.model.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=True)
            )
            embeddings = output.per_residue_embedding  # [B, L, D]

            if embeddings.dim() == 3 and embeddings.size(0) == 1:
                embeddings = embeddings.squeeze(0)  # [L, D]

        # Convert to tensor and move to CPU
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        if embeddings.device != torch.device("cpu"):
            embeddings = embeddings.cpu()

        # Remove batch dimension if present
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)  # [L, D]

        # Remove BOS/EOS tokens (first and last positions)
        if embeddings.shape[0] == len(sequence) + 2:
            embeddings = embeddings[1:-1]

        return embeddings


# ============================================================================
# Protein Geometric Feature Extraction with Dual ESM Integration
# ============================================================================

class ProteinFeatureExtractor:
    """Extract geometric features with ESMC + ESM3 embeddings from protein structures."""

    def __init__(
        self,
        distance_cutoff: float = 4.0,
        esmc_model: str = "esmc_600m",
        esm3_model: str = "esm3-open",
        device: str = "cuda",
    ):
        """
        Initialize protein feature extractor with dual ESM models.

        Args:
            distance_cutoff: Distance cutoff for edge construction (Angstroms)
            esmc_model: ESMC model variant (1152-dim for esmc_600m)
            esm3_model: ESM3 model variant (1536-dim for esm3-open)
            device: "cuda" or "cpu"
        """
        self.distance_cutoff = distance_cutoff

        # Initialize both ESM extractors
        logger.info("Initializing ESMC and ESM3 extractors...")
        self.esmc_extractor = ESMEmbeddingExtractor(
            model_type="esmc",
            model_name=esmc_model,
            device=device
        )
        self.esm3_extractor = ESMEmbeddingExtractor(
            model_type="esm3",
            model_name=esm3_model,
            device=device
        )

    def extract_from_pdb(
        self,
        pdb_path: Union[str, Path],
    ) -> Dict:
        """
        Extract geometric features and dual ESM embeddings from PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Dictionary with:
                - node: Dict with node features
                    - coord: [N, 3] residue coordinates
                    - node_scalar_features: scalar features
                    - node_vector_features: vector features
                    - esmc_embeddings: [N, 1152] ESMC per-residue embeddings
                    - esm3_embeddings: [N, 1536] ESM3 per-residue embeddings
                - edge: Dict with edge features
                    - edges: edge indices
                    - edge_scalar_features: scalar features
                    - edge_vector_features: vector features
        """
        pdb_path = Path(pdb_path)

        # Extract geometric features
        protein_featurizer = ProteinFeaturizer(str(pdb_path))
        res_node, res_edge = protein_featurizer.get_residue_features(
            distance_cutoff=self.distance_cutoff
        )

        # Get chain-wise sequences
        sequences = protein_featurizer.get_sequence_by_chain()

        # Extract ESMC embeddings for each chain
        esmc_embeddings_list = []
        for chain_id, sequence in sequences.items():
            logger.info(f"  Extracting ESMC for chain {chain_id}: {len(sequence)} residues")
            emb = self.esmc_extractor.extract_embeddings(sequence)
            esmc_embeddings_list.append(emb)

        # Extract ESM3 embeddings for each chain
        esm3_embeddings_list = []
        for chain_id, sequence in sequences.items():
            logger.info(f"  Extracting ESM3 for chain {chain_id}: {len(sequence)} residues")
            emb = self.esm3_extractor.extract_embeddings(sequence)
            esm3_embeddings_list.append(emb)

        # Concatenate embeddings from all chains
        esmc_embeddings = torch.cat(esmc_embeddings_list, dim=0)  # [N, 960]
        esm3_embeddings = torch.cat(esm3_embeddings_list, dim=0)  # [N, 1536]

        # Add to node features
        res_node['esmc_embeddings'] = esmc_embeddings
        res_node['esm3_embeddings'] = esm3_embeddings

        logger.info(f"  ESMC embeddings: {esmc_embeddings.shape}")
        logger.info(f"  ESM3 embeddings: {esm3_embeddings.shape}")

        return {
            'node': res_node,
            'edge': res_edge
        }


def extract_protein_features(
    pdb_path: Union[str, Path],
    distance_cutoff: float = 4.0,
    esmc_model: str = "esmc_600m",
    esm3_model: str = "esm3-open",
    device: str = "cuda",
) -> Dict:
    """
    Extract protein features (geometry + ESMC + ESM3) from PDB file.

    Args:
        pdb_path: Path to PDB file
        distance_cutoff: Distance cutoff for edges
        esmc_model: ESMC model variant (1152-dim for esmc_600m)
        esm3_model: ESM3 model variant (1536-dim for esm3-open)
        device: "cuda" or "cpu"

    Returns:
        Dictionary with node and edge features
        node['esmc_embeddings']: [N, 1152] ESMC embeddings
        node['esm3_embeddings']: [N, 1536] ESM3 embeddings
    """
    extractor = ProteinFeatureExtractor(
        distance_cutoff=distance_cutoff,
        esmc_model=esmc_model,
        esm3_model=esm3_model,
        device=device
    )
    return extractor.extract_from_pdb(pdb_path=pdb_path)
