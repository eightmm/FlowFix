"""
Unified SE(3)-equivariant networks for protein-ligand modeling.
"""

import torch
import torch.nn as nn
import cuequivariance as cue_base

from .cue_layers import (
    EquivariantMLP,
    GatingEquivariantLayer,
    EquivariantBatchNorm,
    PairBiasAttentionLayer,
    parse_irreps_dims
)
from .torch_layers import ConditionedTransitionBlock, MLP


class UnifiedEquivariantNetwork(nn.Module):
    """
    Unified SE(3)-equivariant network for processing molecular graphs.

    Supports both scalar-only and scalar+vector inputs.
    Can be used for proteins (with vector features) or ligands (scalar-only).
    """

    def __init__(self,
                 input_scalar_dim: int = 121,
                 input_vector_dim: int = 0,
                 input_edge_scalar_dim: int = 44,
                 input_edge_vector_dim: int = 0,
                 hidden_scalar_dim: int = 128,
                 hidden_vector_dim: int = 16,
                 hidden_edge_dim: int = 64,
                 output_scalar_dim: int = 128,
                 output_vector_dim: int = 16,
                 num_layers: int = 3,
                 sh_lmax: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_edge_dim = hidden_edge_dim
        self.input_vector_dim = input_vector_dim
        self.input_edge_vector_dim = input_edge_vector_dim

        # Store output dimensions
        self.output_scalar_dim = output_scalar_dim
        self.output_vector_dim = output_vector_dim
        self.hidden_scalar_dim = hidden_scalar_dim

        # Build input irreps
        if input_vector_dim > 0:
            self.in_irreps = cue_base.Irreps("O3", f"{input_scalar_dim}x0e + {input_vector_dim}x1o")
        else:
            self.in_irreps = cue_base.Irreps("O3", f"{input_scalar_dim}x0e")

        # Hidden and output irreps
        self.hidden_irreps = cue_base.Irreps(
            "O3", f"{hidden_scalar_dim}x0e + {hidden_vector_dim}x1o + {hidden_vector_dim}x1e"
        )
        self.out_irreps = cue_base.Irreps(
            "O3", f"{output_scalar_dim}x0e + {output_vector_dim}x1o + {output_vector_dim}x1e"
        )

        # Edge irreps
        if input_edge_vector_dim > 0:
            self.edge_in_irreps = cue_base.Irreps("O3", f"{input_edge_scalar_dim}x0e + {input_edge_vector_dim}x1o")
        else:
            self.edge_in_irreps = cue_base.Irreps("O3", f"{input_edge_scalar_dim}x0e")

        self.edge_hidden_irreps = cue_base.Irreps("O3", f"{hidden_edge_dim}x0e")

        # Spherical harmonics
        sh_components = " + ".join([f"1x{l}{'o' if l % 2 == 1 else 'e'}" for l in range(sh_lmax + 1)])
        self.sh_irreps = cue_base.Irreps("O3", sh_components)

        # Networks
        self.node_processor = EquivariantMLP(
            irreps_in=self.in_irreps,
            irreps_hidden=self.hidden_irreps,
            irreps_out=self.hidden_irreps,
            num_layers=3,
            dropout=dropout
        )

        self.edge_processor = EquivariantMLP(
            irreps_in=self.edge_in_irreps,
            irreps_hidden=self.edge_hidden_irreps,
            irreps_out=self.edge_hidden_irreps,
            num_layers=2,
            dropout=dropout
        )

        self.layers = nn.ModuleList([
            GatingEquivariantLayer(
                in_irreps=self.hidden_irreps,
                out_irreps=self.hidden_irreps,
                sh_irreps=self.sh_irreps,
                edge_dim=hidden_edge_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.output_projection = EquivariantMLP(
            irreps_in=self.hidden_irreps,
            irreps_hidden=self.hidden_irreps,
            irreps_out=self.out_irreps,
            num_layers=2,
            dropout=dropout
        )

        self.skip_norms = nn.ModuleList([
            EquivariantBatchNorm(self.hidden_irreps, layout=cue_base.mul_ir)
            for _ in range(num_layers)
        ])

    def forward(self, batch):
        """Forward pass through network."""
        # Extract node features
        node_scalar = batch.x
        if self.input_vector_dim > 0 and hasattr(batch, 'node_vector_features'):
            node_vector = batch.node_vector_features
            node_vector_flat = node_vector.reshape(node_vector.shape[0], -1)
            x = torch.cat([node_scalar, node_vector_flat], dim=-1)
        else:
            x = node_scalar

        h = self.node_processor(x)

        # Extract edge features
        edge_scalar = batch.edge_attr
        if self.input_edge_vector_dim > 0 and hasattr(batch, 'edge_vector_features'):
            edge_vector = batch.edge_vector_features
            edge_vector_flat = edge_vector.reshape(edge_vector.shape[0], -1)
            edge_combined = torch.cat([edge_scalar, edge_vector_flat], dim=-1)
        else:
            edge_combined = edge_scalar

        edge_attr  = self.edge_processor(edge_combined) # only scaler output 

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, batch.pos, batch.edge_index, edge_attr)
            h = self.skip_norms[layer_idx](h)

        return self.output_projection(h)



class ProteinLigandInteractionNetwork(nn.Module):
    """
    Protein-ligand interaction network using cross-attention.
    Processes protein and ligand features jointly through attention layers.
    """

    def __init__(self,
                 protein_output_irreps: str = "128x0e + 32x1o + 32x1e",
                 ligand_output_irreps: str = "128x0e + 16x1o + 16x1e",
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 num_rbf: int = 32,
                 pair_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.pair_dim = pair_dim

        # Parse irreps
        protein_irreps = cue_base.Irreps("O3", protein_output_irreps)
        ligand_irreps = cue_base.Irreps("O3", ligand_output_irreps)

        # Projection to scalar-only features using EquivariantMLP
        # This is better than simple norm: learns optimal projection of vectors to scalars
        scalar_only_irreps = cue_base.Irreps("O3", f"{hidden_dim}x0e")

        self.protein_to_scalar = EquivariantMLP(
            irreps_in=protein_irreps,
            irreps_hidden=protein_irreps,  # Keep vectors in hidden layers
            irreps_out=scalar_only_irreps,  # Output scalars only
            num_layers=2,
            dropout=dropout
        )

        self.ligand_to_scalar = EquivariantMLP(
            irreps_in=ligand_irreps,
            irreps_hidden=ligand_irreps,  # Keep vectors in hidden layers
            irreps_out=scalar_only_irreps,  # Output scalars only
            num_layers=2,
            dropout=dropout
        )

        # RBF parameters
        self.rbf_centers = nn.Parameter(torch.linspace(0, 20, num_rbf))
        self.rbf_width = nn.Parameter(torch.tensor(2.5))

        # Project raw pair features (num_rbf + 6) to pair_dim
        raw_pair_dim = num_rbf + 6  # RBF + 3 interaction types + inv_dist + norm_dist + node_mask
        self.pair_projection = nn.Sequential(
            nn.Linear(raw_pair_dim, self.pair_dim),
            nn.SiLU(),
            nn.Linear(self.pair_dim, self.pair_dim)
        )

        # Attention layers
        self.attention_layers = nn.ModuleList([
            PairBiasAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                pair_dim=self.pair_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # FFN blocks
        self.ffn_blocks = nn.ModuleList([
            MLP(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                out_dim=hidden_dim,
                num_layers=2,
                activation='silu',
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Separate layer norms for attention and FFN
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Apply custom weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for interaction network components.

        Strategy:
        - EquivariantMLPs: Use default cuequivariance initialization (already optimal)
        - FFN blocks: Xavier uniform for SiLU activation
        - Attention layers: Already initialized in PairBiasAttentionLayer
        """
        # Note: protein_to_scalar and ligand_to_scalar are EquivariantMLPs
        # They use cuequivariance's built-in initialization which is already optimal

        # Initialize FFN blocks
        for ffn in self.ffn_blocks:
            self._init_mlp(ffn, gain=1.0)

    def _init_mlp(self, mlp_module, gain=1.0):
        """
        Initialize Linear layers in an MLP module.

        Args:
            mlp_module: MLP module (has .layers Sequential)
            gain: Initialization gain
        """
        for module in mlp_module.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def pyg_to_sequence(self, features, positions, batch_idx):
        """Convert PyG batch to padded sequence with dynamic padding."""
        device = features.device
        batch_size = batch_idx.max().item() + 1
        feat_dim = features.shape[1]

        nodes_per_batch = torch.bincount(batch_idx, minlength=batch_size)
        max_n = nodes_per_batch.max().item()  # Dynamic padding

        padded_feat = torch.zeros(batch_size, max_n, feat_dim, device=device)
        padded_pos = torch.zeros(batch_size, max_n, 3, device=device)
        mask = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)

        for b in range(batch_size):
            b_mask = (batch_idx == b)
            n = min(b_mask.sum().item(), max_n)
            padded_feat[b, :n] = features[b_mask][:n]
            padded_pos[b, :n] = positions[b_mask][:n]
            mask[b, :n] = True

        return padded_feat, padded_pos, mask

    def sequence_to_pyg(self, padded_feat, padded_pos, mask):
        """Convert padded sequence back to PyG batch format.

        Inverse operation of pyg_to_sequence. Converts padded tensors with masks
        back to PyG's concatenated node representation.

        Args:
            padded_feat: [B, max_N, D] padded node features
            padded_pos: [B, max_N, 3] padded positions
            mask: [B, max_N] boolean mask indicating valid nodes

        Returns:
            features: [total_nodes, D] concatenated node features
            positions: [total_nodes, 3] concatenated positions
            batch_idx: [total_nodes] batch assignment for each node
        """
        device = padded_feat.device
        batch_size = padded_feat.shape[0]

        features_list = []
        positions_list = []
        batch_idx_list = []

        for b in range(batch_size):
            valid_mask = mask[b]  # [max_N]
            n_valid = valid_mask.sum().item()

            if n_valid > 0:
                features_list.append(padded_feat[b, valid_mask])  # [n_valid, D]
                positions_list.append(padded_pos[b, valid_mask])  # [n_valid, 3]
                batch_idx_list.append(torch.full((n_valid,), b, dtype=torch.long, device=device))

        # Concatenate all batches
        features = torch.cat(features_list, dim=0)  # [total_nodes, D]
        positions = torch.cat(positions_list, dim=0)  # [total_nodes, 3]
        batch_idx = torch.cat(batch_idx_list, dim=0)  # [total_nodes]

        return features, positions, batch_idx

    def create_pair_bias(self, N_p, combined_pos, combined_mask):
        """
        Create distance-based pair bias features.

        Args:
            N_p: int, number of protein nodes
            combined_pos: [B, N_p+N_l, 3] concatenated protein+ligand positions
            combined_mask: [B, N_p+N_l] concatenated protein+ligand mask
        """
        dist = torch.cdist(combined_pos, combined_pos)
        node_mask = combined_mask.unsqueeze(1) * combined_mask.unsqueeze(2)

        # Interaction types (protein-protein, protein-ligand, ligand-ligand)
        is_pp = torch.zeros_like(dist)
        is_pl = torch.zeros_like(dist)
        is_ll = torch.zeros_like(dist)
        is_pp[:, :N_p, :N_p] = 1
        is_pl[:, :N_p, N_p:] = 1
        is_pl[:, N_p:, :N_p] = 1
        is_ll[:, N_p:, N_p:] = 1

        # RBF features
        rbf_width_safe = torch.abs(self.rbf_width) + 1e-8
        rbf_features = [
            torch.exp(-((dist - c) ** 2) / (2 * rbf_width_safe ** 2)) * node_mask
            for c in self.rbf_centers
        ]

        # Additional features
        eps = 1e-8
        inv_dist = (1.0 / (eps + dist)) * node_mask
        norm_dist = torch.clamp(dist / 20.0, 0, 1) * node_mask

        features = rbf_features + [
            is_pp * node_mask, is_pl * node_mask, is_ll * node_mask,
            inv_dist, norm_dist, node_mask
        ]

        pair_features = torch.stack(features, dim=-1)  # [B, N, N, num_rbf + 6]

        # Project to pair_dim (16)
        pair_features = self.pair_projection(pair_features)  # [B, N, N, 16]

        return pair_features

    def forward(self, protein_output, ligand_output, protein_batch, ligand_batch):
        """
        Forward pass through interaction network.

        Returns:
            (prot_out, lig_out): Tuple of atom-wise features
                - prot_out: [N_protein, hidden_dim] protein node features (unused in velocity prediction)
                - lig_out: [N_ligand, hidden_dim] ligand node features with protein interaction
            prot_global: [B, hidden_dim*2] global protein features (mean+std pooling)
            pair_bias: [B, N, N, pair_dim] pairwise bias features
        """
        # Project to scalar-only features using learnable equivariant projection
        # This learns optimal way to convert vectors to scalars (better than simple norm)
        prot_feat = self.protein_to_scalar(protein_output)  # [N_protein, hidden_dim] scalars only
        lig_feat = self.ligand_to_scalar(ligand_output)     # [N_ligand, hidden_dim] scalars only

        # Convert to sequences with dynamic padding
        prot_seq, prot_pos, prot_mask = self.pyg_to_sequence(
            prot_feat, protein_batch.pos, protein_batch.batch
        )
        lig_seq, lig_pos, lig_mask = self.pyg_to_sequence(
            lig_feat, ligand_batch.pos, ligand_batch.batch
        )
        # Pre-compute combined tensors (used in multiple places)
        N_p = prot_seq.shape[1]
        combined_feat = torch.cat([prot_seq, lig_seq], dim=1)
        combined_pos = torch.cat([prot_pos, lig_pos], dim=1)
        combined_mask = torch.cat([prot_mask, lig_mask], dim=1)

        # Pair bias (reuses pre-computed tensors - no redundant computations)
        pair_bias = self.create_pair_bias(N_p, combined_pos, combined_mask)

        # Attention + FFN blocks with global skip connection
        h = combined_feat
        h_initial = h  # Save for global residual connection

        for attn, ffn, attn_norm, ffn_norm, dropout in zip(
            self.attention_layers, self.ffn_blocks,
            self.attn_norms, self.ffn_norms, self.dropouts
        ):
            # Attention block (local skip within layer)
            h_attn, _ = attn(h, pair_bias, combined_mask)
            h = attn_norm(h + dropout(h_attn))

            # FFN block (local skip within layer)
            h_ffn = ffn(h)
            h = ffn_norm(h + dropout(h_ffn))

        # Global residual connection: add initial features back
        # This helps gradient flow through the entire attention stack
        h = h + h_initial

        # Split into protein/ligand sequences (reuse pre-computed N_p)
        prot_seq_out = h[:, :N_p] * prot_mask.unsqueeze(-1)  # [B, N_p, D]
        lig_seq_out = h[:, N_p:] * lig_mask.unsqueeze(-1)    # [B, N_l, D]

        # Convert back to PyG node features
        prot_out, _, _ = self.sequence_to_pyg(prot_seq_out, prot_pos, prot_mask)  # [total_prot_nodes, D]
        lig_out, _, _ = self.sequence_to_pyg(lig_seq_out, lig_pos, lig_mask)      # [total_lig_nodes, D]

        # Global pooling with mean + std for protein only
        # Only protein global context is needed for conditioning
        prot_global = self._mean_std_pooling(prot_seq_out, prot_mask)  # [B, D*2]
        lig_global = self._mean_std_pooling(lig_seq_out, lig_mask)  # [B, D*2]

        return (prot_out, lig_out), (prot_global, lig_global), pair_bias

    def _mean_std_pooling(self, features, mask):
        """
        Compute mean and std pooling with masking.

        Args:
            features: [B, N, D] node features
            mask: [B, N] boolean mask

        Returns:
            pooled: [B, D*2] concatenated mean and std
        """
        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]

        # Count valid nodes per batch
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        # Mean pooling
        mean = (features * mask_expanded).sum(dim=1) / count  # [B, D]

        # Std pooling (variance-based)
        # Compute variance: E[(x - mean)^2]
        diff = (features - mean.unsqueeze(1)) * mask_expanded  # [B, N, D]
        variance = (diff ** 2).sum(dim=1) / count  # [B, D]
        std = torch.sqrt(variance + 1e-8)  # [B, D], add epsilon for numerical stability

        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=-1)  # [B, D*2]

        return pooled
