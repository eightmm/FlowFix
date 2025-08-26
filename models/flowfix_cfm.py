import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import math

# cuEquivariance imports
import cuequivariance as cue_base
from cuequivariance_torch import (
    SphericalHarmonics,
    Linear as EquivariantLinear,
)
from cuequivariance_torch.layers import (
    FullyConnectedTensorProductConv,
    BatchNorm as EquivariantBatchNorm
)


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching (CFM) model for protein-ligand binding pose refinement.
    Implements optimal transport flow matching with SE(3)-equivariance.
    """
    
    def __init__(
        self,
        protein_feat_dim: int = 72,
        ligand_feat_dim: int = 14,
        edge_dim: int = 64,
        hidden_scalars: int = 128,
        hidden_vectors: int = 32,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 8,
        max_ell: int = 2,
        cutoff: float = 10.0,
        time_embedding_dim: int = 128,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_layer_norm: bool = True,
        use_gate: bool = True
    ):
        super().__init__()
        
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_scalars = hidden_scalars
        self.use_gate = use_gate
        self.out_dim = out_dim
        
        layout = cue_base.ir_mul
        
        # Define irreps with richer representations
        self.irreps_protein = cue_base.Irreps("O3", f"{protein_feat_dim}x0e")
        self.irreps_ligand = cue_base.Irreps("O3", f"{ligand_feat_dim}x0e")
        # Richer hidden representation for better expressivity
        self.irreps_hidden = cue_base.Irreps("O3", 
            f"{hidden_scalars}x0e + {hidden_vectors}x1o + {hidden_vectors//2}x1e + {hidden_vectors//4}x2e")
        self.irreps_out = cue_base.Irreps("O3", f"{out_dim}x0e + 3x1o")
        
        # Spherical harmonics
        self.spherical_harmonics = SphericalHarmonics(list(range(max_ell + 1)), normalize=True)
        sh_irreps_str = "+".join([f"1x{l}e" if l % 2 == 0 else f"1x{l}o" for l in range(max_ell + 1)])
        self.irreps_sh = cue_base.Irreps("O3", sh_irreps_str)
        
        # Improved time encoding with FiLM conditioning
        self.time_encoder = ImprovedTimeEncoder(time_embedding_dim)
        
        # Feature encoders with skip connections
        self.protein_encoder = ResidualMLP(
            protein_feat_dim, hidden_scalars, hidden_dim, dropout
        )
        
        self.ligand_encoder = ResidualMLP(
            ligand_feat_dim, hidden_scalars, hidden_dim, dropout
        )
        
        # Time-conditioned feature modulation
        self.time_modulation = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_scalars * 2),
            nn.SiLU(),
            nn.Linear(hidden_scalars * 2, hidden_scalars * 2)
        )
        
        # Input embedding
        self.input_embedding = EquivariantLinear(
            cue_base.Irreps("O3", f"{hidden_scalars}x0e"),
            self.irreps_hidden,
            layout=layout
        )
        
        # Edge encoder with attention mechanism
        self.edge_encoder = nn.Sequential(
            nn.Linear(1 + time_embedding_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_dim * 2, edge_dim)
        )
        
        # Multi-scale equivariant blocks
        self.conv_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            block = EquivariantBlock(
                irreps_hidden=self.irreps_hidden,
                irreps_sh=self.irreps_sh,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                layout=layout,
                dropout=dropout,
                use_gate=use_gate,
                use_layer_norm=use_layer_norm
            )
            self.conv_blocks.append(block)
        
        # Global attention pooling for context
        self.global_attention = GlobalAttentionPooling(
            self.irreps_hidden, 
            num_heads=num_heads,
            layout=layout
        )
        
        # Output projection with skip connection
        self.output_projection = nn.ModuleList([
            EquivariantLinear(self.irreps_hidden, self.irreps_hidden, layout=layout),
            EquivariantLinear(self.irreps_hidden, self.irreps_out, layout=layout)
        ])
        
        # Learnable scale parameter for vector field
        self.vector_scale = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        protein_coords: torch.Tensor,
        protein_features: torch.Tensor,
        ligand_coords: torch.Tensor,
        ligand_features: torch.Tensor,
        t: torch.Tensor,
        protein_batch: Optional[torch.Tensor] = None,
        ligand_batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with improved flow matching.
        """
        # Ensure correct dtype
        protein_coords = protein_coords.float()
        protein_features = protein_features.float()
        ligand_coords = ligand_coords.float()
        ligand_features = ligand_features.float()
        t = t.float()
        
        # Encode time with improved encoder
        time_embedding = self.time_encoder(t)
        
        # Extract protein CA coordinates
        if protein_coords.dim() == 3:
            protein_ca = protein_coords[:, 0, :]
        else:
            protein_ca = protein_coords
        
        # Encode features
        protein_h = self.protein_encoder(protein_features)
        ligand_h = self.ligand_encoder(ligand_features)
        
        # Apply time-conditioned modulation (FiLM)
        time_mod = self.time_modulation(time_embedding[0] if time_embedding.shape[0] == 1 else time_embedding.mean(dim=0))
        scale, shift = time_mod.chunk(2, dim=-1)
        scale = scale.unsqueeze(0)
        shift = shift.unsqueeze(0)
        
        ligand_h = ligand_h * (1 + scale) + shift
        
        # Build edges with improved connectivity
        edges, edge_distances, edge_vectors = self._build_edges(
            protein_ca, ligand_coords, protein_batch, ligand_batch
        )
        
        if edges.shape[1] == 0:
            # No edges - return zero vector field
            return {
                'vector_field': torch.zeros_like(ligand_coords),
                'confidence': torch.zeros(ligand_coords.shape[0], 1, device=ligand_coords.device)
            }
        
        # Prepare edge features
        if ligand_batch is not None and time_embedding.shape[0] > 1:
            edge_batch = ligand_batch[edges[0]]
            time_emb_expanded = time_embedding[edge_batch]
        else:
            time_emb_expanded = time_embedding[0].unsqueeze(0).expand(edge_distances.shape[0], -1)
        
        edge_features = self.edge_encoder(
            torch.cat([edge_distances.unsqueeze(-1), time_emb_expanded], dim=-1)
        )
        
        # Calculate spherical harmonics
        edge_sh = self.spherical_harmonics(edge_vectors.float())
        
        # Combine features
        num_ligand = ligand_coords.shape[0]
        num_protein = protein_ca.shape[0]
        
        combined_features = torch.cat([ligand_h, protein_h], dim=0)
        h = self.input_embedding(combined_features.float())
        
        # Adjust edge indices
        adjusted_edges = edges.clone()
        adjusted_edges[1] = adjusted_edges[1] + num_ligand
        
        graph = (adjusted_edges, (num_ligand + num_protein, num_ligand + num_protein))
        
        # Multi-scale processing with skip connections
        h_skip = []
        
        for i, block in enumerate(self.conv_blocks):
            h = block(h, edge_sh, edge_features, graph)
            
            # Store intermediate features for skip connections
            if i % 2 == 0:
                h_skip.append(h[:num_ligand])
        
        # Extract ligand features
        ligand_h_final = h[:num_ligand]
        
        # Apply global attention for context
        if len(h_skip) > 0:
            ligand_h_final = self.global_attention(ligand_h_final, h_skip)
        
        # Output projection with residual
        out = self.output_projection[0](ligand_h_final.float())
        out = F.silu(out)
        out = self.output_projection[1](out.float())
        
        # Split into scalars and vectors
        # The output irreps are "256x0e + 3x1o" which means:
        # - 256 scalars (256 dimensions)
        # - 3 vectors (3x3=9 dimensions)
        out_scalars = out[:, :self.out_dim]
        vectors = out[:, self.out_dim:self.out_dim+9].view(-1, 3, 3)
        
        # Compute vector field with learned scale
        vector_field = vectors.mean(dim=1) * self.vector_scale
        
        # Compute confidence score from scalar features
        confidence = torch.sigmoid(out_scalars[:, :1])
        
        return {
            'vector_field': vector_field,
            'confidence': confidence,
            'scalar_features': out_scalars
        }
    
    def _build_edges(
        self,
        protein_coords: torch.Tensor,
        ligand_coords: torch.Tensor,
        protein_batch: Optional[torch.Tensor] = None,
        ligand_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build edges with KNN fallback for connectivity."""
        
        if protein_batch is None or ligand_batch is None:
            # Single sample
            distances = torch.cdist(ligand_coords, protein_coords)
            edge_mask = distances < self.cutoff
            
            # Ensure minimum connectivity with KNN
            if edge_mask.sum() < ligand_coords.shape[0] * 3:
                k = min(5, protein_coords.shape[0])
                _, knn_indices = distances.topk(k, largest=False, dim=1)
                for i in range(ligand_coords.shape[0]):
                    edge_mask[i, knn_indices[i]] = True
            
            edges = torch.nonzero(edge_mask, as_tuple=False).T
            edge_distances = distances[edge_mask]
            edge_vectors = ligand_coords[edges[0]] - protein_coords[edges[1]]
            edge_vectors_norm = F.normalize(edge_vectors, dim=-1)
            
            return edges, edge_distances, edge_vectors_norm
        
        # Batched case
        edge_list = []
        distance_list = []
        vector_list = []
        
        batch_ids = torch.unique(ligand_batch)
        
        for batch_id in batch_ids:
            ligand_mask = ligand_batch == batch_id
            protein_mask = protein_batch == batch_id
            
            ligand_idx = torch.where(ligand_mask)[0]
            protein_idx = torch.where(protein_mask)[0]
            
            if len(ligand_idx) == 0 or len(protein_idx) == 0:
                continue
            
            batch_ligand_coords = ligand_coords[ligand_mask]
            batch_protein_coords = protein_coords[protein_mask]
            
            distances = torch.cdist(batch_ligand_coords, batch_protein_coords)
            edge_mask = distances < self.cutoff
            
            # Ensure connectivity
            if edge_mask.sum() < batch_ligand_coords.shape[0] * 3:
                k = min(5, batch_protein_coords.shape[0])
                _, knn_indices = distances.topk(k, largest=False, dim=1)
                for i in range(batch_ligand_coords.shape[0]):
                    edge_mask[i, knn_indices[i]] = True
            
            local_edges = torch.nonzero(edge_mask, as_tuple=False).T
            
            if local_edges.shape[1] == 0:
                continue
            
            global_edges = torch.stack([
                ligand_idx[local_edges[0]],
                protein_idx[local_edges[1]]
            ])
            
            edge_list.append(global_edges)
            distance_list.append(distances[edge_mask])
            
            edge_vecs = ligand_coords[global_edges[0]] - protein_coords[global_edges[1]]
            vector_list.append(F.normalize(edge_vecs, dim=-1))
        
        if len(edge_list) == 0:
            return (torch.zeros((2, 0), dtype=torch.long, device=ligand_coords.device),
                    torch.zeros(0, device=ligand_coords.device),
                    torch.zeros((0, 3), device=ligand_coords.device))
        
        edges = torch.cat(edge_list, dim=1)
        edge_distances = torch.cat(distance_list)
        edge_vectors_norm = torch.cat(vector_list)
        
        return edges, edge_distances, edge_vectors_norm


class EquivariantBlock(nn.Module):
    """Single equivariant processing block with gating and normalization."""
    
    def __init__(
        self,
        irreps_hidden,
        irreps_sh,
        edge_dim: int,
        hidden_dim: int,
        layout,
        dropout: float = 0.1,
        use_gate: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.use_gate = use_gate
        self.use_layer_norm = use_layer_norm
        
        # Message passing
        self.conv = FullyConnectedTensorProductConv(
            in_irreps=irreps_hidden,
            sh_irreps=irreps_sh,
            out_irreps=irreps_hidden,
            mlp_channels=[edge_dim, hidden_dim, hidden_dim],
            mlp_activation=nn.SiLU(),
            layout=layout
        )
        
        # Normalization
        if use_layer_norm:
            self.norm = EquivariantBatchNorm(irreps_hidden, layout=layout)
        
        # Self-interaction
        self.self_interaction = EquivariantLinear(irreps_hidden, irreps_hidden, layout=layout)
        
        # Gating mechanism
        if use_gate:
            self.gate = EquivariantLinear(irreps_hidden, irreps_hidden, layout=layout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, edge_sh, edge_features, graph):
        # Message passing
        messages = self.conv(
            src_features=h.float(),
            edge_sh=edge_sh.float(),
            edge_emb=edge_features.float(),
            graph=graph
        )
        
        # Self-interaction
        self_update = self.self_interaction(h.float())
        
        # Combine
        update = messages + self_update
        
        # Normalization
        if self.use_layer_norm:
            update = self.norm(update)
        
        # Gating
        if self.use_gate:
            gate = torch.sigmoid(self.gate(h.float()))
            update = gate * update
        
        # Residual connection with dropout
        h = h + self.dropout(update)
        
        return h


class ResidualMLP(nn.Module):
    """MLP with residual connections and layer normalization."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        ])
        
        # Skip connection if dimensions match
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x + identity * 0.1  # Scaled residual


class ImprovedTimeEncoder(nn.Module):
    """Enhanced time encoding with learnable fourier features."""
    
    def __init__(self, dim: int, num_fourier: int = 32):
        super().__init__()
        self.dim = dim
        self.num_fourier = num_fourier
        
        # Learnable fourier frequencies
        self.fourier_freqs = nn.Parameter(torch.randn(num_fourier) * 0.1)
        
        # Projection network
        self.proj = nn.Sequential(
            nn.Linear(num_fourier * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Learnable fourier features
        freqs = self.fourier_freqs.unsqueeze(0)
        t_expanded = t.unsqueeze(-1)
        
        # Compute fourier features
        fourier_features = torch.cat([
            torch.sin(2 * math.pi * t_expanded * freqs),
            torch.cos(2 * math.pi * t_expanded * freqs)
        ], dim=-1)
        
        return self.proj(fourier_features)


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling for aggregating multi-scale features."""
    
    def __init__(self, irreps, num_heads: int, layout):
        super().__init__()
        self.num_heads = num_heads
        
        # Simple aggregation for equivariant features
        self.out_proj = EquivariantLinear(irreps, irreps, layout=layout)
    
    def forward(self, query, keys):
        if len(keys) == 0:
            return query
        
        # Simple mean pooling of multi-scale features
        # This preserves equivariance
        keys_mean = torch.stack(keys, dim=0).mean(dim=0)
        
        # Combine with query
        combined = query + keys_mean * 0.1
        
        # Project
        out = self.out_proj(combined.float())
        
        return query + out * 0.1  # Scaled residual