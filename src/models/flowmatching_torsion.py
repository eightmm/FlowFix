"""
SE(3) + Torsion Decomposition Flow Matching Model.

Instead of predicting per-atom Cartesian velocity [N, 3], this model predicts:
- Translation [B, 3]: CoM displacement
- Rotation [B, 3]: Axis-angle rotation around CoM
- Torsion [M]: One scalar per rotatable bond

Inherits the encoder and interaction network from ProteinLigandFlowMatching,
replacing only the output heads.
"""

import torch
import torch.nn as nn
import cuequivariance as cue_base
from torch_scatter import scatter_mean

from .cue_layers import EquivariantMLP
from .torch_layers import MLP
from .flowmatching import ProteinLigandFlowMatching


class ProteinLigandFlowMatchingTorsion(ProteinLigandFlowMatching):
    """
    SE(3) + Torsion decomposition variant of ProteinLigandFlowMatching.

    Shares encoder, interaction network, and velocity blocks with the base class.
    Replaces the output heads with:
    - Translation head: mean-pool → EquivariantMLP → [B, 3]
    - Rotation head: mean-pool → EquivariantMLP → [B, 3] (axis-angle)
    - Torsion head: src/dst node scalar concat → MLP → [M, 1]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get dimensions from parent
        vel_hidden_scalar_dim = kwargs.get('velocity_hidden_scalar_dim', 128)
        vel_hidden_vector_dim = kwargs.get('velocity_hidden_vector_dim', 16)
        dropout = kwargs.get('dropout', 0.1)

        self._vel_hidden_scalar_dim = vel_hidden_scalar_dim

        vel_hidden_irreps = cue_base.Irreps(
            "O3",
            f"{vel_hidden_scalar_dim}x0e + {vel_hidden_vector_dim}x1o + {vel_hidden_vector_dim}x1e"
        )
        intermediate_irreps = cue_base.Irreps(
            "O3",
            f"{vel_hidden_scalar_dim}x0e + {vel_hidden_vector_dim}x1o + {vel_hidden_vector_dim}x1e"
        )
        vector_output_irreps = cue_base.Irreps("O3", "1x1o")

        # Translation head: pooled features → 3D vector
        self.translation_output = EquivariantMLP(
            irreps_in=vel_hidden_irreps,
            irreps_hidden=intermediate_irreps,
            irreps_out=vector_output_irreps,
            num_layers=2,
            dropout=dropout,
        )

        # Rotation head: pooled features → 3D axis-angle
        self.rotation_output = EquivariantMLP(
            irreps_in=vel_hidden_irreps,
            irreps_hidden=intermediate_irreps,
            irreps_out=vector_output_irreps,
            num_layers=2,
            dropout=dropout,
        )

        # Torsion head: src/dst node scalars → 1 scalar per rotatable bond
        torsion_input_dim = vel_hidden_scalar_dim * 2
        self.torsion_output = MLP(
            in_dim=torsion_input_dim,
            hidden_dim=vel_hidden_scalar_dim,
            out_dim=1,
            num_layers=2,
            activation='silu',
        )

        # Zero-initialize all output heads for stable training
        self._zero_init_output_heads()

        # Learnable scales
        self.translation_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.rotation_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.torsion_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _zero_init_output_heads(self):
        """Zero-initialize output heads for stable training start."""
        for head in [self.torsion_output]:
            for module in head.layers:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _encode(self, protein_batch, ligand_batch, t):
        """
        Run shared encoder + interaction + velocity blocks.

        Returns h: [N_ligand, hidden_irreps] features after message passing.
        """
        # ESM embeddings
        if self.use_esm_embeddings:
            protein_batch = self._integrate_esm_embeddings(protein_batch)

        protein_output = self.protein_network(protein_batch)
        ligand_output = self.ligand_network(ligand_batch)

        # Interaction
        (_, lig_out), (protein_context, _), _ = self.interaction_network(
            protein_output, ligand_output, protein_batch, ligand_batch
        )

        # Conditioning
        protein_context_expanded = protein_context[ligand_batch.batch]
        combined_condition = torch.cat([protein_context_expanded, lig_out], dim=-1)
        atom_condition = self.vel_atom_condition_proj(combined_condition)

        # Velocity blocks (shared backbone)
        h = self.vel_input_projection(ligand_output)
        h_initial = h

        for block in self.velocity_blocks:
            h = block(
                h,
                ligand_batch.pos,
                ligand_batch.edge_index,
                ligand_batch.edge_attr,
                condition=atom_condition,
            )

        h = h + h_initial
        return h

    def forward(
        self,
        protein_batch,
        ligand_batch,
        t: torch.Tensor,
        rotatable_edges: torch.Tensor = None,
    ) -> dict:
        """
        Predict SE(3) + Torsion velocity.

        Args:
            protein_batch: Protein PyG batch
            ligand_batch: Ligand PyG batch at time t
            t: [B] timesteps
            rotatable_edges: [M, 2] atom indices of rotatable bonds

        Returns:
            Dict with 'translation' [B, 3], 'rotation' [B, 3], 'torsion' [M]
        """
        h = self._encode(protein_batch, ligand_batch, t)

        # Translation: mean-pool → 3D
        h_pooled = scatter_mean(h, ligand_batch.batch, dim=0)
        translation = self.translation_output(h_pooled) * self.translation_scale

        # Rotation: mean-pool → 3D axis-angle
        rotation = self.rotation_output(h_pooled) * self.rotation_scale

        # Torsion: src/dst scalar features → 1 scalar per bond
        if rotatable_edges is not None and rotatable_edges.shape[0] > 0:
            src_idx = rotatable_edges[:, 0]
            dst_idx = rotatable_edges[:, 1]

            # Extract scalar part (first scalar_dim components of irreps)
            h_scalar = h[:, :self._vel_hidden_scalar_dim]
            edge_feat = torch.cat([h_scalar[src_idx], h_scalar[dst_idx]], dim=-1)
            torsion = self.torsion_output(edge_feat).squeeze(-1) * self.torsion_scale
        else:
            torsion = torch.zeros(0, device=h.device)

        return {
            'translation': translation,
            'rotation': rotation,
            'torsion': torsion,
        }
