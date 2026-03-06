"""
SE(3)-Equivariant Flow Matching for Protein-Ligand Pose Refinement.

This module implements a flow matching model that refines ligand docked poses
to crystal poses using SE(3)-equivariant neural networks.
"""

import torch
import torch.nn as nn
import cuequivariance as cue_base

from .cue_layers import EquivariantMLP, GatingEquivariantLayer
from .torch_layers import MLP
from .network import UnifiedEquivariantNetwork, ProteinLigandInteractionNetwork


class ProteinLigandFlowMatching(nn.Module):
    """
    Complete SE(3)-equivariant flow matching model for protein-ligand pose refinement.

    Architecture:
    1. ProteinNetwork: Encodes fixed protein structure
    2. LigandNetwork: Encodes ligand at current time t with time conditioning
    3. ProteinLigandInteractionNetwork: Computes protein-ligand interactions via cross-attention
    4. Velocity Prediction: Predicts per-atom velocity field using:
       - Ligand node features (from LigandNetwork)
       - Protein context (from ProteinLigandInteractionNetwork)
       - Pair features and time conditioning

    Output: [N_ligand, 3] velocity vectors for each ligand atom
    """

    def __init__(self,
                 # Protein network parameters
                 protein_input_scalar_dim: int = 76,
                 protein_input_vector_dim: int = 31,
                 protein_input_edge_scalar_dim: int = 39,
                 protein_input_edge_vector_dim: int = 8,
                 protein_hidden_scalar_dim: int = 128,
                 protein_hidden_vector_dim: int = 32,
                 protein_output_scalar_dim: int = 128,
                 protein_output_vector_dim: int = 32,
                 protein_num_layers: int = 3,

                 # Ligand network parameters
                 ligand_input_scalar_dim: int = 121,
                 ligand_input_edge_scalar_dim: int = 44,
                 ligand_hidden_scalar_dim: int = 128,
                 ligand_hidden_vector_dim: int = 16,
                 ligand_output_scalar_dim: int = 128,
                 ligand_output_vector_dim: int = 16,
                 ligand_num_layers: int = 3,

                 # Interaction network parameters
                 interaction_num_heads: int = 8,
                 interaction_num_layers: int = 2,
                 interaction_num_rbf: int = 32,
                 interaction_pair_dim: int = 64,

                 # Velocity predictor parameters
                 velocity_hidden_scalar_dim: int = 128,
                 velocity_hidden_vector_dim: int = 16,
                 velocity_num_layers: int = 4,

                 # General parameters (unified hidden_dim for non-equivariant features)
                 hidden_dim: int = 256,  # Unified dimension for time, interaction, conditioning
                 dropout: float = 0.1,

                 # ESM embedding parameters
                 use_esm_embeddings: bool = True,
                 esmc_dim: int = 1152,  # ESMC 600M embedding dimension
                 esm3_dim: int = 1536):  # ESM3 embedding dimension
        super().__init__()

        # ESM embedding projection layers
        self.use_esm_embeddings = use_esm_embeddings
        if use_esm_embeddings:
            self.esmc_projection = MLP(
                in_dim=esmc_dim,
                hidden_dim=protein_hidden_scalar_dim,
                out_dim=protein_hidden_scalar_dim,
                num_layers=2,
                activation='silu',
                dropout=dropout
            )
            self.esm3_projection = MLP(
                in_dim=esm3_dim,
                hidden_dim=protein_hidden_scalar_dim,
                out_dim=protein_hidden_scalar_dim,
                num_layers=2,
                activation='silu',
                dropout=dropout
            )

            # Learnable weights for combining ESM projections
            self.esm_weight = nn.Parameter(torch.ones(2) * 0.5)  # [ESMC weight, ESM3 weight]

        self.protein_network = UnifiedEquivariantNetwork(
            input_scalar_dim=protein_input_scalar_dim,
            input_vector_dim=protein_input_vector_dim,
            input_edge_scalar_dim=protein_input_edge_scalar_dim,
            input_edge_vector_dim=protein_input_edge_vector_dim,
            hidden_scalar_dim=protein_hidden_scalar_dim,
            hidden_vector_dim=protein_hidden_vector_dim,
            output_scalar_dim=protein_output_scalar_dim,
            output_vector_dim=protein_output_vector_dim,
            num_layers=protein_num_layers,
            dropout=dropout
        )

        self.ligand_network = UnifiedEquivariantNetwork(
            input_scalar_dim=ligand_input_scalar_dim,
            input_vector_dim=0,  # Ligand has no vector features
            input_edge_scalar_dim=ligand_input_edge_scalar_dim,
            input_edge_vector_dim=0,  # Ligand edges have no vector features
            hidden_scalar_dim=ligand_hidden_scalar_dim,
            hidden_vector_dim=ligand_hidden_vector_dim,
            output_scalar_dim=ligand_output_scalar_dim,
            output_vector_dim=ligand_output_vector_dim,
            num_layers=ligand_num_layers,
            dropout=dropout
        )

        # Protein-ligand interaction network
        protein_irreps_str = f"{protein_output_scalar_dim}x0e + {protein_output_vector_dim}x1o + {protein_output_vector_dim}x1e"
        ligand_irreps_str = f"{ligand_output_scalar_dim}x0e + {ligand_output_vector_dim}x1o + {ligand_output_vector_dim}x1e"

        self.interaction_network = ProteinLigandInteractionNetwork(
            protein_output_irreps=protein_irreps_str,
            ligand_output_irreps=ligand_irreps_str,
            hidden_dim=hidden_dim,
            num_heads=interaction_num_heads,
            num_layers=interaction_num_layers,
            dropout=dropout,
            num_rbf=interaction_num_rbf,
            pair_dim=interaction_pair_dim
        )

        # ✅ TimeEmbedding removed - time is implicit in x_t coordinates

        # Parse ligand output irreps
        ligand_output_irreps_obj = cue_base.Irreps("O3", ligand_irreps_str)

        # Get scalar dimension from ligand irreps
        self.ligand_scalar_dim = sum(mul for mul, ir in ligand_output_irreps_obj if ir.l == 0)  # Count 0e
        self.ligand_output_vector_dim = ligand_output_vector_dim

        # Use only ligand network output (no interaction features)
        combined_input_irreps = cue_base.Irreps(
            "O3",
            f"{self.ligand_scalar_dim}x0e + {self.ligand_output_vector_dim}x1o + {self.ligand_output_vector_dim}x1e"
        )

        vel_hidden_irreps = cue_base.Irreps(
            "O3",
            f"{velocity_hidden_scalar_dim}x0e + {velocity_hidden_vector_dim}x1o + {velocity_hidden_vector_dim}x1e"
        )

        # Output is 3D velocity vector (1x1o = single odd-parity vector)
        velocity_output_irreps = cue_base.Irreps("O3", "1x1o")

        # Spherical harmonics for edge encoding
        sh_lmax = 2
        sh_components = " + ".join([f"1x{l}{'o' if l % 2 == 1 else 'e'}" for l in range(sh_lmax + 1)])
        vel_sh_irreps = cue_base.Irreps("O3", sh_components)

        # Conditioning from combined global + local features
        # protein_context: [B, hidden_dim*2] - global protein features (mean+std pooling)
        # lig_out: [N_ligand, hidden_dim] - atom-wise interaction features
        atom_condition_dim = hidden_dim * 3  # protein_context (256*2) + lig_out (256) = 768

        # Atom-level condition projection (protein_context + lig_out)
        self.vel_atom_condition_proj = MLP(
            in_dim=atom_condition_dim,
            hidden_dim=atom_condition_dim,
            out_dim=hidden_dim,  # Project back to hidden_dim for velocity blocks
            num_layers=2,
            activation='silu'
        )

        # Input projection (accepts ligand features + interaction context)
        self.vel_input_projection = EquivariantMLP(
            irreps_in=combined_input_irreps,  # Ligand scalars + interaction scalars + vectors
            irreps_hidden=vel_hidden_irreps,
            irreps_out=vel_hidden_irreps,
            num_layers=2,
            dropout=dropout
        )

        # Equivariant velocity prediction blocks
        # Note: atom_condition_dim is the input to projection (512),
        # but after projection it becomes hidden_dim (256)
        self.velocity_blocks = nn.ModuleList([
            GatingEquivariantLayer(
                in_irreps=vel_hidden_irreps,
                out_irreps=vel_hidden_irreps,
                sh_irreps=vel_sh_irreps,
                edge_dim=ligand_input_edge_scalar_dim,
                use_skip=True,
                mlp_num_layers=2,
                dropout=dropout,
                condition_dim=hidden_dim  # Projected atom condition dimension
            )
            for _ in range(velocity_num_layers)
        ])

        # Final projection to velocity (unified MLP)
        # Include pseudo-vectors (1e) in intermediate for richer geometric representation
        intermediate_irreps = cue_base.Irreps("O3", f"{velocity_hidden_scalar_dim}x0e + {velocity_hidden_vector_dim}x1o + {velocity_hidden_vector_dim}x1e")
        self.vel_output = EquivariantMLP(
            irreps_in=vel_hidden_irreps,
            irreps_hidden=intermediate_irreps,
            irreps_out=velocity_output_irreps,  # Final output: 1x1o (polar vector for velocity)
            num_layers=3,  # Maintains same depth as before (2-layer MLP + 1 linear)
            dropout=dropout
        )

        # Zero initialize final layer for stable training
        with torch.no_grad():
            final_linear = self.vel_output.layers[-2]  # Last Linear before final BatchNorm
            if hasattr(final_linear, 'weight'):
                nn.init.zeros_(final_linear.weight)

        # Learnable velocity scale
        self.velocity_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Apply custom initialization
        self._init_weights()

    def forward(self,
                protein_batch,
                ligand_batch,
                t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field for ligand refinement.

        Args:
            protein_batch: PyG Batch for protein (fixed)
            ligand_batch: PyG Batch for ligand at time t
            t: [B] time values in [0, 1] (kept for API compatibility, not used)

        Returns:
            [N_ligand, 3] velocity vectors for ligand atoms

        Note:
            Time information is implicit in ligand_batch.pos (x_t coordinates).
            For linear interpolation: v = x_1 - x_0 (constant, time-independent).
        """
        # 1. Encode Protein and Ligand (time-free)
        # Protein is fixed, ligand coordinates x_t contain implicit time information

        # Process ESM embeddings if available
        if self.use_esm_embeddings:
            protein_batch = self._integrate_esm_embeddings(protein_batch)

        protein_output = self.protein_network(protein_batch)
        ligand_output = self.ligand_network(ligand_batch)

        # 2. Protein-Ligand Interaction (time-free)
        (_, lig_out), (protein_context, _), _ = self.interaction_network(
            protein_output, ligand_output, protein_batch, ligand_batch
        )

        # 3. Velocity Prediction with combined global + local conditioning
        protein_context_expanded = protein_context[ligand_batch.batch]  # [N_ligand, hidden_dim*2]
        combined_condition = torch.cat([protein_context_expanded, lig_out], dim=-1)  # [N_ligand, hidden_dim*3]
        atom_condition = self.vel_atom_condition_proj(combined_condition)  # [N_ligand, hidden_dim]

        h = self.vel_input_projection(ligand_output)

        h_initial = h

        for block in self.velocity_blocks:
            h = block(
                h,
                ligand_batch.pos,
                ligand_batch.edge_index,
                ligand_batch.edge_attr,
                condition=atom_condition  # Atom-wise condition with interaction info
            )

        h = h + h_initial

        velocity = self.vel_output(h) * self.velocity_scale

        return velocity

    def _integrate_esm_embeddings(self, protein_batch):
        """
        Integrate ESMC and ESM3 embeddings into protein node features.

        Args:
            protein_batch: PyG Batch with optional esmc_embeddings and esm3_embeddings

        Returns:
            Modified protein_batch with enhanced node features
        """
        # Check if ESM embeddings are available
        has_esmc = hasattr(protein_batch, 'esmc_embeddings') and protein_batch.esmc_embeddings is not None
        has_esm3 = hasattr(protein_batch, 'esm3_embeddings') and protein_batch.esm3_embeddings is not None

        if not (has_esmc or has_esm3):
            return protein_batch

        # Get original scalar dimension
        original_scalar_dim = protein_batch.x.shape[1]

        # Project ESM embeddings (this creates the computational graph)
        esm_features = []
        weights = []

        if has_esmc:
            # Convert to float and enable gradient if needed
            esmc_emb = protein_batch.esmc_embeddings.float()
            esmc_proj = self.esmc_projection(esmc_emb)  # [N, hidden_scalar_dim]
            esm_features.append(esmc_proj)
            weights.append(self.esm_weight[0])

        if has_esm3:
            # Convert to float and enable gradient if needed
            esm3_emb = protein_batch.esm3_embeddings.float()
            esm3_proj = self.esm3_projection(esm3_emb)  # [N, hidden_scalar_dim]
            esm_features.append(esm3_proj)
            weights.append(self.esm_weight[1])

        # Weighted combination of ESM projections
        weights_tensor = torch.stack(weights)
        weights_normalized = torch.softmax(weights_tensor, dim=0)

        combined_esm = sum(w * feat for w, feat in zip(weights_normalized, esm_features))
        # combined_esm: [N, hidden_scalar_dim]

        # Project combined ESM back to original scalar dimension to preserve UnifiedEquivariantNetwork input
        if not hasattr(self, '_esm_to_input'):
            # Create projection layer on first call (lazy init to avoid device issues)
            device = protein_batch.x.device
            self._esm_to_input = MLP(
                in_dim=self.protein_network.hidden_scalar_dim,
                hidden_dim=original_scalar_dim,
                out_dim=original_scalar_dim,
                num_layers=1,
                activation='silu'
            ).to(device)

        esm_to_input = self._esm_to_input(combined_esm)  # [N, original_scalar_dim]

        # Create new node features with ESM enhancement (residual connection)
        # This creates a new tensor in the computational graph
        enhanced_x = protein_batch.x + esm_to_input  # [N, original_scalar_dim]

        # Update batch.x directly (PyG allows this and preserves computational graph)
        protein_batch.x = enhanced_x

        return protein_batch

    def _init_weights(self):
        """
        Initialize weights for stable training.

        Strategy:
        1. MLP layers: Xavier uniform for hidden layers, smaller init for conditioning paths
        2. Conditioning networks: Conservative initialization for stable training
        3. Final velocity projection: Already zero-initialized in __init__
        """
        # Initialize atom condition projection MLP (conservative for conditioning)
        self._init_mlp(self.vel_atom_condition_proj, gain=0.5)

        # Initialize ESM projection MLPs (if enabled)
        if self.use_esm_embeddings:
            self._init_mlp(self.esmc_projection, gain=0.5)
            self._init_mlp(self.esm3_projection, gain=0.5)

    def _init_mlp(self, mlp_module, gain=1.0):
        """
        Initialize Linear layers in an MLP module.

        Args:
            mlp_module: MLP module (has .layers Sequential)
            gain: Initialization gain (smaller for conditioning paths)
        """
        for module in mlp_module.layers:
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for SiLU activation
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

