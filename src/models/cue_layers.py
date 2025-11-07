"""
cuEquivariance-based SE(3)-equivariant layers.

This module contains all layers that use cuequivariance_torch for SE(3)-equivariant operations.
"""

import torch
import torch.nn as nn
import cuequivariance as cue_base
from torch_scatter import scatter
from typing import Dict

from cuequivariance_torch import (
    SphericalHarmonics,
    FullyConnectedTensorProduct,
    Linear,
    attention_pair_bias
)
from cuequivariance_torch.layers import BatchNorm as EquivariantBatchNorm

from .torch_layers import MLP


# ============================================================================
# Utility Functions
# ============================================================================

def parse_irreps_dims(irreps: cue_base.Irreps) -> Dict[str, int]:
    """
    Extract scalar and vector dimensions from irreps.

    Args:
        irreps: cuEquivariance Irreps object

    Returns:
        Dictionary with keys: 'scalar', 'vector_1o', 'vector_1e', 'total_vector'
    """
    scalar = sum(mul for mul, ir in irreps if ir.l == 0)
    vector_1o = sum(mul for mul, ir in irreps if ir.l == 1 and ir.p == 1)
    vector_1e = sum(mul for mul, ir in irreps if ir.l == 1 and ir.p == -1)

    return {
        'scalar': scalar,
        'vector_1o': vector_1o,
        'vector_1e': vector_1e,
        'total_vector': vector_1o + vector_1e
    }


class EquivariantAdaLN(nn.Module):
    """
    SE(3)-Equivariant Adaptive Layer Normalization.

    Applies AdaLN to scalar features (l=0) and optional time-conditioned gating
    to vector features (l=1) to preserve SE(3) equivariance.

    Args:
        irreps: Input/output irreps (must be same for residual structure)
        time_embed_dim: Dimension of time conditioning
        apply_to_vectors: If True, apply time-conditioned gating to vectors
    """

    def __init__(self,
                 irreps: cue_base.Irreps,
                 time_embed_dim: int,
                 apply_to_vectors: bool = True):
        super().__init__()

        self.irreps = irreps
        self.time_embed_dim = time_embed_dim
        self.apply_to_vectors = apply_to_vectors

        # Parse irreps to get dimensions
        dims = parse_irreps_dims(irreps)
        self.scalar_dim = dims['scalar']
        self.vector_1o_dim = dims['vector_1o']
        self.vector_1e_dim = dims['vector_1e']
        self.total_vector_dim = dims['total_vector']

        # AdaLN for scalar features
        if self.scalar_dim > 0:
            # Layer norm for scalars (no learnable parameters)
            self.scalar_norm = nn.LayerNorm(self.scalar_dim, elementwise_affine=False, bias=False)

            # Time conditioning for scalars
            self.time_norm = nn.LayerNorm(time_embed_dim, bias=False)
            self.time_to_scale = nn.Linear(time_embed_dim, self.scalar_dim)
            self.time_to_bias = nn.Linear(time_embed_dim, self.scalar_dim, bias=False)
        else:
            self.scalar_norm = None
            self.time_norm = None
            self.time_to_scale = None
            self.time_to_bias = None

        # Time-conditioned gating for vectors (preserves equivariance)
        if self.apply_to_vectors and self.total_vector_dim > 0:
            # Project time to gate values for each vector channel
            self.time_to_vector_gate = MLP(
                time_embed_dim, time_embed_dim, self.total_vector_dim,
                num_layers=2, activation='silu', final_activation='sigmoid'
            )
        else:
            self.time_to_vector_gate = None

        # Apply custom weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for AdaLN components.

        Strategy:
        - Scale/bias projections: Zero initialization for stable identity-like start
        - Vector gating: Conservative initialization for stable gating
        """
        # Zero initialization for scale and bias (DiT-style)
        if self.time_to_scale is not None:
            nn.init.zeros_(self.time_to_scale.weight)
            nn.init.zeros_(self.time_to_scale.bias)

        if self.time_to_bias is not None:
            nn.init.zeros_(self.time_to_bias.weight)

        # Conservative initialization for vector gating MLP
        if self.time_to_vector_gate is not None:
            for module in self.time_to_vector_gate.layers:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        """
        Apply equivariant AdaLN conditioning.

        Args:
            x: [N, irreps.dim] input features (scalars + vectors flattened)
            time_embed: [N, time_embed_dim] or [B, time_embed_dim] time conditioning

        Returns:
            [N, irreps.dim] conditioned features
        """
        # Handle broadcasting if time_embed is [B, D] and we have batch indices
        # For now assume time_embed is already [N, D]

        components = []
        offset = 0

        # Process scalars with AdaLN
        if self.scalar_dim > 0:
            scalar_features = x[:, :self.scalar_dim]

            # Normalize scalars
            scalar_norm = self.scalar_norm(scalar_features)

            # Normalize time
            time_norm = self.time_norm(time_embed)

            # Compute scale and bias from time
            scale = torch.sigmoid(self.time_to_scale(time_norm))  # [N, scalar_dim]
            bias = self.time_to_bias(time_norm)  # [N, scalar_dim]

            # Apply AdaLN: scale * norm(x) + bias
            scalar_conditioned = scale * scalar_norm + bias
            components.append(scalar_conditioned)
            offset += self.scalar_dim

        # Process vectors with time-conditioned gating
        if self.total_vector_dim > 0:
            vector_dim_flat = self.total_vector_dim * 3
            vector_features = x[:, offset:offset + vector_dim_flat]

            if self.apply_to_vectors and self.time_to_vector_gate is not None:
                # Reshape to [N, total_vector_dim, 3]
                vectors = vector_features.reshape(-1, self.total_vector_dim, 3)

                # Compute time-conditioned gates [N, total_vector_dim]
                gates = self.time_to_vector_gate(time_embed)  # [N, total_vector_dim]

                # Apply gates: [N, total_vector_dim, 1] * [N, total_vector_dim, 3]
                vectors_gated = vectors * gates.unsqueeze(-1)

                # Flatten back
                vector_conditioned = vectors_gated.reshape(-1, vector_dim_flat)
            else:
                # No gating, pass through
                vector_conditioned = vector_features

            components.append(vector_conditioned)
            offset += vector_dim_flat

        # Concatenate all components
        output = torch.cat(components, dim=-1) if len(components) > 1 else components[0]

        return output


class ScalarActivation(nn.Module):
    def __init__(self, activation_fn: nn.Module, irreps: cue_base.Irreps):
        super().__init__()
        self.activation_fn = activation_fn
        self.scalar_dim = irreps[0].mul if irreps[0].ir.l == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scalar_dim > 0:
            result = x.clone()
            result[:, :self.scalar_dim] = self.activation_fn(x[:, :self.scalar_dim])
            return result
        return x


class ScalarDropout(nn.Module):
    def __init__(self, dropout_rate: float, irreps: cue_base.Irreps):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scalar_dim = irreps[0].mul if irreps[0].ir.l == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scalar_dim > 0:
            result = x.clone()
            result[:, :self.scalar_dim] = self.dropout(x[:, :self.scalar_dim])
            return result
        return x


class EquivariantDropout(nn.Module):
    def __init__(self, p: float, scalar_dim: int, vector_dim: int):
        super().__init__()
        self.p = p
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        x_scalar = x[:, :self.scalar_dim]
        x_vector = x[:, self.scalar_dim:].reshape(x.shape[0], self.vector_dim, 3)

        scalar_mask = torch.bernoulli(torch.ones_like(x_scalar) * (1 - self.p))
        x_scalar = x_scalar * scalar_mask / (1 - self.p)

        vector_mask = torch.bernoulli(
            torch.ones(x.shape[0], self.vector_dim, 1, device=x.device) * (1 - self.p)
        )
        x_vector = x_vector * vector_mask / (1 - self.p)

        x_out = torch.cat([x_scalar, x_vector.reshape(x.shape[0], -1)], dim=-1)
        return x_out


class EquivariantMLP(nn.Module):
    """
    Equivariant MLP using cuequivariance_torch.Linear for proper SE(3)-equivariance.

    This MLP maintains SE(3) equivariance by using equivariant linear layers
    instead of regular nn.Linear layers.
    """
    def __init__(self,
                 irreps_in: cue_base.Irreps,
                 irreps_hidden: cue_base.Irreps,
                 irreps_out: cue_base.Irreps,
                 num_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_hidden = irreps_hidden
        self.num_layers = num_layers

        layers = []

        # Input to hidden with normalization
        layers.append(Linear(
            irreps_in=irreps_in,
            irreps_out=self.irreps_hidden,
            layout=cue_base.mul_ir
        ))

        layers.append(EquivariantBatchNorm(
            irreps=self.irreps_hidden,
            layout=cue_base.mul_ir,
        ))

        layers.append(ScalarActivation(nn.SiLU(), self.irreps_hidden))

        if dropout > 0:
            layers.append(ScalarDropout(dropout, self.irreps_hidden))

        for _ in range(num_layers - 2):
            layers.append(Linear(
                irreps_in=self.irreps_hidden,
                irreps_out=self.irreps_hidden,
                layout=cue_base.mul_ir
            ))

            layers.append(EquivariantBatchNorm(
                irreps=self.irreps_hidden,
                layout=cue_base.mul_ir,
            ))

            layers.append(ScalarActivation(nn.SiLU(), self.irreps_hidden))

            if dropout > 0:
                layers.append(ScalarDropout(dropout, self.irreps_hidden))

        # Hidden to output with normalization
        if num_layers > 1:
            layers.append(Linear(
                irreps_in=self.irreps_hidden,
                irreps_out=irreps_out,
                layout=cue_base.mul_ir
            ))

            layers.append(EquivariantBatchNorm(
                irreps=irreps_out,
                layout=cue_base.mul_ir,
            ))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GatingEquivariantLayer(nn.Module):
    """
    SE(3)-Equivariant layer with element-wise scalar and adaptive vector gating.

    This is the final production-ready layer with:
    - Flexible irreps input/output
    - Element-wise scalar gating (most expressive)
    - Adaptive vector gating with norm-based features
    - Edge importance weighting for message selection
    - Skip connections for gradient flow
    - Self-interaction for isolated nodes
    - Always uses sum aggregation for perfect SE(3) equivariance
    """

    def __init__(
        self,
        in_irreps: cue_base.Irreps,
        out_irreps: cue_base.Irreps,
        sh_irreps: cue_base.Irreps,
        edge_dim: int = 32,
        use_skip: bool = True,
        mlp_num_layers: int = 2,
        dropout: float = 0.0,
        condition_dim: int = None,
    ):
        """
        Initialize the gated SE(3)-equivariant layer.

        Args:
            in_irreps: Input irreducible representations
            out_irreps: Output irreducible representations
            sh_irreps: Spherical harmonics irreps for position encoding
            edge_dim: Dimension of edge attributes
            use_skip: Whether to use skip connections
            mlp_num_layers: Number of layers in EquivariantMLP
            dropout: Dropout rate
            condition_dim: Dimension of context conditioning (protein+ligand+interaction info)
        """
        super().__init__()

        # Store irreps
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps

        # Parse dimensions from irreps
        self._parse_irreps_dimensions()

        # Store other parameters
        self.edge_dim = edge_dim
        self.dropout_rate = dropout
        self.aggregation = 'sum'  # Always use sum aggregation for stability
        self.condition_dim = condition_dim

        # Store flags
        self.use_skip = use_skip
        self.mlp_num_layers = mlp_num_layers

        # Initialize components
        self._init_edge_embedding()
        self._init_tensor_products()
        self._init_edge_importance()
        self._init_gating()
        self._init_node_update_mlp()
        self._init_normalization()
        self._init_spherical_harmonics()
        self._init_context_conditioning()

        # Apply custom weight initialization
        self._init_weights()

    def _parse_irreps_dimensions(self):
        """Parse scalar and vector dimensions from irreps."""
        # Parse input irreps
        dims_in = parse_irreps_dims(self.in_irreps)
        self.scalar_dim_in = dims_in['scalar']
        self.vector_dim_in = dims_in['vector_1e']  # Using 1e for consistency

        # Parse output irreps
        dims_out = parse_irreps_dims(self.out_irreps)
        self.scalar_dim_out = dims_out['scalar']
        self.vector_dim_out = dims_out['vector_1e']  # Using 1e for consistency

        # For backward compatibility and gating networks
        self.scalar_dim = self.scalar_dim_in
        self.vector_dim = self.vector_dim_in

    def _init_spherical_harmonics(self):
        """Initialize spherical harmonics computation."""
        self.sh_lmax = max(ir.l for _, ir in self.sh_irreps)

        ls = list(range(self.sh_lmax + 1))
        self.spherical_harmonics = SphericalHarmonics(ls=ls, normalize=True)

    def _init_edge_embedding(self):
        """Initialize edge embedding network."""
        # Use edge_dim as the target hidden dimension for edge features
        # This is the dimension passed from network (hidden_edge_dim parameter)
        hidden_dim = self.edge_dim

        self.edge_embedding = MLP(
            self.edge_dim, hidden_dim, hidden_dim,
            num_layers=3, activation='silu', dropout=self.dropout_rate
        )

        self.irreps_edge_attr = cue_base.Irreps("O3", f"{hidden_dim}x0e")
        self.hidden_dim = hidden_dim

    def _init_tensor_products(self):
        """Initialize tensor product operations."""
        self.tp_message = FullyConnectedTensorProduct(
            irreps_in1=self.in_irreps,
            irreps_in2=self.irreps_edge_attr + self.sh_irreps,
            irreps_out=self.out_irreps,
            layout_in1=cue_base.mul_ir,
            layout_in2=cue_base.mul_ir,
            layout_out=cue_base.mul_ir,
        )

        self.tp_self = FullyConnectedTensorProduct(
            irreps_in1=self.in_irreps,
            irreps_in2=cue_base.Irreps("O3", "1x0e"),
            irreps_out=self.out_irreps,
            layout_in1=cue_base.mul_ir,
            layout_in2=cue_base.mul_ir,
            layout_out=cue_base.mul_ir,
        )

    def _init_edge_importance(self):
        """Initialize edge importance network for message weighting."""
        # Use edge_emb (hidden_dim) instead of raw edge_attr for invariance
        input_dim = 2 * self.scalar_dim_in + self.hidden_dim  # src + dst + edge_emb
        self.edge_importance_net = MLP(
            input_dim, self.hidden_dim, 1,
            num_layers=2, activation='silu', final_activation='sigmoid',
            dropout=self.dropout_rate
        )

    def _init_gating(self):
        """Initialize gating networks."""
        gate_base_dim = self.hidden_dim  # edge_emb (+ optional context projection)

        # Scalar gating
        if self.scalar_dim_out > 0:
            self.scalar_gate_net = MLP(
                gate_base_dim, self.hidden_dim, self.scalar_dim_out,
                num_layers=2, activation='silu', final_activation='sigmoid',
                dropout=self.dropout_rate
            )
        else:
            self.scalar_gate_net = None

        # Vector gating
        if self.vector_dim_out > 0:
            self.vector_norm_net = MLP(
                self.vector_dim_out, self.hidden_dim // 2, self.hidden_dim // 2,
                num_layers=1, activation='silu'
            )

            vector_gate_dim = self.hidden_dim + self.hidden_dim // 2  # edge_emb + norm features

            self.vector_gate_net = MLP(
                vector_gate_dim, self.hidden_dim, self.vector_dim_out,
                num_layers=2, activation='silu', final_activation='sigmoid',
                dropout=self.dropout_rate
            )
        else:
            self.vector_norm_net = None
            self.vector_gate_net = None

    def _init_context_conditioning(self):
        """
        Initialize context conditioning projections.

        Context contains protein+ligand+interaction information and is used at 2 stages:
        1. Input node features (main conditioning - context propagates through entire layer)
        2. Output features (final refinement after aggregation)

        Note: Edge conditioning removed - redundant since nodes already have context
        and it naturally propagates through message passing.
        """
        if self.condition_dim is not None:
            # 1. EquivariantAdaLN for input node conditioning
            # Applied BEFORE message passing so that:
            # - tp_message uses context-aware node features
            # - tp_self uses context-aware node features
            # - edge_importance_net sees context-aware src/dst
            # Context propagates naturally through all operations
            self.context_node_adaln = EquivariantAdaLN(
                irreps=self.in_irreps,
                time_embed_dim=self.condition_dim,
                apply_to_vectors=True
            )

            # 2. EquivariantAdaLN for output feature conditioning
            # Applied after batch norm for final context-aware refinement
            self.context_output_adaln = EquivariantAdaLN(
                irreps=self.out_irreps,
                time_embed_dim=self.condition_dim,
                apply_to_vectors=True
            )
        else:
            self.context_node_adaln = None
            self.context_output_adaln = None

    def _init_node_update_mlp(self):
        """Initialize EquivariantMLP for node feature updates."""
        hidden_scalar_dim = max(self.scalar_dim_out * 2, 64)
        hidden_vector_dim = self.vector_dim_out

        hidden_irreps_str = f"{hidden_scalar_dim}x0e"
        if hidden_vector_dim > 0:
            hidden_irreps_str += f" + {hidden_vector_dim}x1o"
        irreps_hidden = cue_base.Irreps("O3", hidden_irreps_str)

        self.node_update_mlp = EquivariantMLP(
            irreps_in=self.out_irreps,
            irreps_hidden=irreps_hidden,
            irreps_out=self.out_irreps,
            num_layers=self.mlp_num_layers,
            dropout=self.dropout_rate
        )

        self.message_mlp = EquivariantMLP(
            irreps_in=self.out_irreps,
            irreps_hidden=self.out_irreps,
            irreps_out=self.out_irreps,
            num_layers=1,
            dropout=0.0
        )

    def _init_normalization(self):
        """Initialize normalization layers."""
        self.batch_norm = EquivariantBatchNorm(
            self.out_irreps,
            layout=cue_base.mul_ir
        )

        # Calculate actual feature dimension from out_irreps
        # out_irreps.dim gives total flattened dimension
        actual_dim = self.out_irreps.dim
        expected_dim = self.scalar_dim_out + self.vector_dim_out * 3

        if actual_dim != expected_dim:
            # Recalculate based on actual irreps structure
            # Count actual scalars and vectors from irreps
            actual_scalar = sum(mul for mul, ir in self.out_irreps if ir.l == 0)
            actual_vector = sum(mul for mul, ir in self.out_irreps if ir.l == 1)

            self.dropout = EquivariantDropout(
                p=self.dropout_rate,
                scalar_dim=actual_scalar,
                vector_dim=actual_vector
            )
        else:
            self.dropout = EquivariantDropout(
                p=self.dropout_rate,
                scalar_dim=self.scalar_dim_out,
                vector_dim=self.vector_dim_out
            )

    def _init_weights(self):
        """
        Initialize weights for MLP components.

        Strategy:
        - Edge and gating networks: Xavier uniform for SiLU activation
        - Smaller gain for gating networks (conservative for stability)
        - Tensor products and equivariant layers: handled by cuequivariance
        """
        # Initialize edge embedding MLP
        self._init_mlp(self.edge_embedding, gain=1.0)

        # Initialize edge importance network (conservative for attention-like behavior)
        self._init_mlp(self.edge_importance_net, gain=0.5)

        # Initialize gating networks (conservative for stable gating)
        if self.scalar_gate_net is not None:
            self._init_mlp(self.scalar_gate_net, gain=0.5)

        if self.vector_norm_net is not None:
            self._init_mlp(self.vector_norm_net, gain=0.5)

        if self.vector_gate_net is not None:
            self._init_mlp(self.vector_gate_net, gain=0.5)

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

    def _apply_scalar_gates(self, messages: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        """
        Apply scalar gating to messages.

        Args:
            messages: [E, features] message features
            gate_input: [E, gate_dim] input for gate network

        Returns:
            [E, features] gated messages
        """
        if self.scalar_gate_net is None:
            return messages

        scalar_gates = self.scalar_gate_net(gate_input)
        scalar_part = messages[:, :self.scalar_dim_out] * scalar_gates
        remaining_part = messages[:, self.scalar_dim_out:]
        return torch.cat([scalar_part, remaining_part], dim=-1)

    def _apply_vector_gates(
        self,
        messages: torch.Tensor,
        edge_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply vector gating with norm-based features.

        Args:
            messages: [E, features] message features
            edge_emb: [E, hidden_dim] edge embeddings (contains context info)

        Returns:
            [E, features] gated messages
        """
        if self.vector_gate_net is None:
            return messages

        vector_start = self.scalar_dim_out
        vector_end = vector_start + self.vector_dim_out * 3
        vector_flat = messages[:, vector_start:vector_end]
        vectors = vector_flat.reshape(-1, self.vector_dim_out, 3)

        # Compute norm features
        vector_norms = vectors.norm(dim=-1)
        norm_features = self.vector_norm_net(vector_norms)

        # Build vector gate input (edge_emb already contains context)
        vector_gate_input = torch.cat([edge_emb, norm_features], dim=-1)
        vector_gates = self.vector_gate_net(vector_gate_input).unsqueeze(-1)

        # Apply gates
        vectors_gated = vectors * vector_gates
        vector_flat_new = vectors_gated.reshape(-1, self.vector_dim_out * 3)

        # Reassemble
        before = messages[:, :vector_start]
        after = messages[:, vector_end:]
        return torch.cat([before, vector_flat_new, after], dim=-1)

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_edge_features: bool = False,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with gated SE(3)-equivariant message passing.

        Args:
            node_features: [N, in_irreps.dim] node features
            positions: [N, 3] node positions
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] edge attributes
            return_edge_features: If True, return (node_features, edge_features)
            condition: [N, condition_dim] context conditioning (protein+ligand+interaction info)

        Returns:
            If return_edge_features is False:
                [N, out_irreps.dim] updated node features
            If return_edge_features is True:
                ([N, out_irreps.dim], [E, edge_feature_dim]) tuple of node and edge features
        """
        src, dst = edge_index
        N = node_features.shape[0]
        device = node_features.device

        edge_vec = positions[dst] - positions[src]
        edge_sh = self.spherical_harmonics(edge_vec)

        # Save original for skip connection
        identity = node_features

        # 0. Apply context to input node features (main conditioning)
        # This makes ALL subsequent operations context-aware:
        # - tp_message: node[src] has context
        # - tp_self: node has context
        # - edge_importance: src/dst have context
        # - messages naturally carry context through the layer
        if condition is not None and self.context_node_adaln is not None:
            node_features = self.context_node_adaln(node_features, condition)

        # Edge embedding (no context needed - nodes already have it)
        edge_emb = self.edge_embedding(edge_attr)
        edge_features = torch.cat([edge_emb, edge_sh], dim=-1)

        messages = self.tp_message(node_features[src], edge_features)

        # Apply message MLP
        messages = self.message_mlp(messages)

        # Apply edge importance weighting (message-level selection)
        # Note: src_scalar and dst_scalar already have context from node conditioning
        if self.edge_importance_net is not None:
            src_scalar = node_features[src, :self.scalar_dim_in]
            dst_scalar = node_features[dst, :self.scalar_dim_in]
            importance_input = torch.cat([src_scalar, dst_scalar, edge_emb], dim=-1)
            importance_weights = self.edge_importance_net(importance_input)  # [E, 1]
            messages = messages * importance_weights

        # 2. Apply scalar and vector gating
        # Note: messages already carry context from context-aware node features
        messages = self._apply_scalar_gates(messages, edge_emb)
        messages = self._apply_vector_gates(messages, edge_emb)

        aggregated = scatter(messages, dst, dim=0, dim_size=N, reduce='sum')

        ones = torch.ones(N, 1, device=device)
        self_update = self.tp_self(node_features, ones)

        output = aggregated + self_update

        # Apply node update MLP
        output = self.node_update_mlp(output)

        output = self.batch_norm(output)

        # 3. Apply output AdaLN context conditioning AFTER batch norm (final refinement)
        if self.context_output_adaln is not None and condition is not None:
            output = self.context_output_adaln(output, condition)

        output = self.dropout(output)

        if self.use_skip and self.in_irreps == self.out_irreps:
            output = output + identity

        if return_edge_features:
            # Return both node features and the processed edge features (messages before aggregation)
            return output, messages
        else:
            return output


class PairBiasAttentionLayer(nn.Module):
    """
    Single attention layer using cuequivariance_torch.attention_pair_bias.

    This layer implements efficient attention with pair bias for protein-ligand interactions.
    Uses NVIDIA's optimized attention_pair_bias kernel.

    Args:
        hidden_dim: Hidden dimension for input features
        num_heads: Number of attention heads
        pair_dim: Dimension of pairwise bias features (z)
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        pair_dim: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.pair_dim = pair_dim

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Input projection (s -> q, k, v)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        # Pair bias projection weights (following cuequivariance API)
        # w_proj_z: (H, z_dim) for pair bias projection
        self.w_proj_z = nn.Parameter(torch.randn(num_heads, pair_dim) / (pair_dim ** 0.5))
        # w_proj_g: (D, D) for gating projection
        self.w_proj_g = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / (hidden_dim ** 0.5))
        # w_proj_o: (D, D) for output projection
        self.w_proj_o = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / (hidden_dim ** 0.5))

        # Layer norm for pair features
        self.w_ln_z = nn.Parameter(torch.ones(pair_dim))
        self.b_ln_z = nn.Parameter(torch.zeros(pair_dim))

        # Optional bias terms
        self.b_proj_z = nn.Parameter(torch.zeros(num_heads))
        self.b_proj_g = nn.Parameter(torch.zeros(hidden_dim))
        self.b_proj_o = nn.Parameter(torch.zeros(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Apply custom weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for attention components.

        Strategy:
        - QKV projection: Xavier uniform for stability
        - Parameter weights: Already initialized with proper scaling in __init__
        - Layer norm: PyTorch default (ones for weight, zeros for bias)
        """
        # Initialize QKV projection
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass with pair bias attention.

        Args:
            x: Input sequence features [B, N, D] where N is sequence length
            z: Pairwise bias features [B, N, N, pair_dim]
            mask: Attention mask [B, N] or [B, N, N]

        Returns:
            output: Attention output [B, N, D]
            proj_z: Projected pair bias [B, H, N, N]
        """
        B, N, D = x.shape

        # Normalize input
        x_norm = self.layer_norm(x)

        # Project to Q, K, V
        qkv = self.qkv_proj(x_norm)  # [B, N, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, N, D]

        # Reshape for multi-head: [B, N, D] -> [B, H, N, DH]
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # Handle mask: convert to [B, N] if needed
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        elif mask.dim() == 3:  # [B, N, N] -> take diagonal or use column
            mask = mask[:, 0, :]  # [B, N] - assume symmetric mask

        # Call attention_pair_bias
        s_input = x_norm  # [B, N, D] - multiplicity M=1

        # Ensure all inputs are contiguous (cuequivariance backward requires this)
        s_input = s_input.contiguous()
        z = z.contiguous()
        mask = mask.contiguous()

        output, proj_z = attention_pair_bias(
            s=s_input,
            q=q,
            k=k,
            v=v,
            z=z,
            mask=mask,
            num_heads=self.num_heads,
            w_proj_z=self.w_proj_z,
            w_proj_g=self.w_proj_g,
            w_proj_o=self.w_proj_o,
            w_ln_z=self.w_ln_z,
            b_ln_z=self.b_ln_z,
            b_proj_z=self.b_proj_z,
            b_proj_g=self.b_proj_g,
            b_proj_o=self.b_proj_o
        )

        # output: [B, N, D]
        # proj_z: [B, H, N, N]

        # Residual connection
        output = x + self.dropout(output)

        return output, proj_z
