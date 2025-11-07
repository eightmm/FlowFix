"""
Pure PyTorch layers without cuequivariance dependencies.

This module contains layers that use standard PyTorch operations,
primarily for attention mechanisms and conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


LinearNoBias = partial(nn.Linear, bias=False)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit with chunked input"""
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class AdaLN(nn.Module):
    """Adaptive Layer Normalization"""

    def __init__(self, dim, dim_single_cond):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        dim : int
            The input dimension.
        dim_single_cond : int
            The single condition dimension.
        """
        super().__init__()
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

        # Zero initialization for stable training (DiT-style)
        # Initial: scale=sigmoid(0)=0.5, bias=0 → stable identity-like transformation
        nn.init.zeros_(self.s_scale.weight)
        nn.init.zeros_(self.s_scale.bias)
        nn.init.zeros_(self.s_bias.weight)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)

        # Handle broadcasting: s is [B, D_cond], a is [B, N, D]
        scale = torch.sigmoid(self.s_scale(s))  # [B, D]
        bias = self.s_bias(s)  # [B, D]

        # Expand to match a's dimensions
        if a.dim() == 3 and scale.dim() == 2:
            scale = scale.unsqueeze(1)  # [B, 1, D]
            bias = bias.unsqueeze(1)    # [B, 1, D]

        a = scale * a + bias
        return a


class MLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron with configurable architecture.

    Supports various activation functions, dropout, and optional final activation.
    Used throughout the codebase to replace repetitive MLP patterns.

    Args:
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension(s). Can be int or list of ints
        out_dim: Output dimension
        num_layers: Number of layers (including output layer). Must be >= 1
        activation: Activation function name ('silu', 'relu', 'gelu', 'sigmoid', 'tanh')
        final_activation: Final activation function name (same options as activation, or None)
        dropout: Dropout probability (applied after each hidden layer)
        bias: Whether to use bias in linear layers

    Examples:
        # Simple 2-layer MLP with SiLU
        mlp = MLP(64, 128, 32, num_layers=2, activation='silu')

        # 3-layer MLP with dropout and sigmoid output
        mlp = MLP(64, 128, 1, num_layers=3, activation='silu',
                  final_activation='sigmoid', dropout=0.1)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        activation: str = 'silu',
        final_activation: str = None,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Get activation functions
        self.activation = self._get_activation(activation)
        self.final_activation = self._get_activation(final_activation) if final_activation else None

        # Build layers
        layers = []

        if num_layers == 1:
            # Single layer: direct in -> out
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
        else:
            # First layer: in -> hidden
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Middle layers: hidden -> hidden
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Final layer: hidden -> out
            layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))

        # Add final activation if specified
        if self.final_activation is not None:
            layers.append(self.final_activation)

        self.layers = nn.Sequential(*layers)

    def _get_activation(self, name: str):
        """Get activation function by name."""
        if name is None:
            return None

        activations = {
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }

        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")

        return activations[name.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Create embedding projection
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] time values in [0, 1]
        Returns:
            [B, embed_dim] time embeddings
        """
        # Sinusoidal embedding
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Project to embedding dimension
        emb = self.projection(emb)
        return emb


class ConditionedTransitionBlock(nn.Module):
    """Conditioned Transition Block with AdaLN and SwiGLU"""

    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        super().__init__()

        self.adaln = AdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = nn.Sequential(
            LinearNoBias(dim_single, dim_inner * 2),
            SwiGLU(),
        )
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        self.b_to_a = LinearNoBias(dim_inner, dim_single)

        output_projection_linear = nn.Linear(dim_single_cond, dim_single)
        nn.init.zeros_(output_projection_linear.weight)
        nn.init.constant_(output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(output_projection_linear, nn.Sigmoid())

    def forward(self, a, s):
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)

        # Handle output projection broadcasting
        gate = self.output_projection(s)  # [B, dim_single]
        b_output = self.b_to_a(b)  # [B, N, dim_single]

        # Expand gate to match b_output dimensions
        if a.dim() == 3 and gate.dim() == 2:
            gate = gate.unsqueeze(1)  # [B, 1, dim_single]

        return gate * b_output
