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


class FlowFixEquivariantModel(nn.Module):
    """
    Flow matching model for protein-ligand binding pose refinement.
    Uses cuEquivariance for SE(3)-equivariant processing.
    """
    
    def __init__(
        self,
        protein_feat_dim: int = 96,  # Protein scalar features
        ligand_feat_dim: int = 52,   # Ligand scalar features  
        edge_dim: int = 32,
        hidden_scalars: int = 48,
        hidden_vectors: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 6,
        max_ell: int = 2,
        cutoff: float = 10.0,
        time_embedding_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            protein_feat_dim: Dimension of protein scalar features
            ligand_feat_dim: Dimension of ligand scalar features
            edge_dim: Edge feature dimension
            hidden_scalars: Number of scalar channels in hidden layers
            hidden_vectors: Number of vector channels in hidden layers
            hidden_dim: Hidden dimension for MLPs
            out_dim: Output dimension
            num_layers: Number of equivariant layers
            max_ell: Maximum angular momentum
            cutoff: Distance cutoff for interactions
            time_embedding_dim: Dimension for time embedding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_scalars = hidden_scalars
        
        layout = cue_base.ir_mul
        
        # Define irreps for different stages
        self.irreps_protein = cue_base.Irreps("O3", f"{protein_feat_dim}x0e")
        self.irreps_ligand = cue_base.Irreps("O3", f"{ligand_feat_dim}x0e + 3x1o")  # Include 3D coords as vectors
        self.irreps_hidden = cue_base.Irreps("O3", f"{hidden_scalars}x0e + {hidden_scalars}x1o + {hidden_vectors}x1e")
        self.irreps_out = cue_base.Irreps("O3", f"{out_dim}x0e + 3x1o")  # Output includes vector field
        
        # Spherical harmonics for edge features
        self.spherical_harmonics = SphericalHarmonics(list(range(max_ell + 1)), normalize=True)
        sh_irreps_str = "+".join([f"1x{l}e" for l in list(range(max_ell + 1))])
        self.irreps_sh = cue_base.Irreps("O3", sh_irreps_str)
        
        # Time encoding
        self.time_encoder = TimeEncoder(time_embedding_dim)
        
        # Protein encoder - processes residue features
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_scalars),
            nn.Dropout(dropout)
        )
        
        # Ligand encoder - processes atom features
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_scalars),
            nn.Dropout(dropout)
        )
        
        # Scalar preprocessing to match expected dimensions
        self.scalar_proj = nn.Linear(hidden_scalars, hidden_scalars)
        
        # Input embedding for equivariant processing
        # Only use scalar features for simplicity
        self.input_embedding = EquivariantLinear(
            cue_base.Irreps("O3", f"{hidden_scalars}x0e"),
            self.irreps_hidden,
            layout=layout
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1 + time_embedding_dim, edge_dim),  # distance + time
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Equivariant message passing layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.self_interaction_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = FullyConnectedTensorProductConv(
                in_irreps=self.irreps_hidden,
                sh_irreps=self.irreps_sh,
                out_irreps=self.irreps_hidden,
                mlp_channels=[edge_dim, hidden_dim, hidden_dim],
                mlp_activation=nn.SiLU(),
                layout=layout
            )
            self.conv_layers.append(conv)
            self.norm_layers.append(EquivariantBatchNorm(self.irreps_hidden, layout=layout))
            self.self_interaction_layers.append(
                EquivariantLinear(self.irreps_hidden, self.irreps_hidden, layout=layout)
            )
        
        # Output projection for vector field prediction
        self.output_projection = EquivariantLinear(self.irreps_hidden, self.irreps_out, layout=layout)
        
        # Remove physics module - focus purely on pose refinement
        
        self.dropout = nn.Dropout(dropout)
    
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
        Forward pass of the flow matching model.
        
        Args:
            protein_coords: Protein CA/SC coordinates (N_protein, 2, 3)
            protein_features: Protein node features (N_protein, F_protein)
            ligand_coords: Ligand atom coordinates (N_ligand, 3)
            ligand_features: Ligand atom features (N_ligand, F_ligand)
            t: Time step (B,) or (1,)
            protein_batch: Batch indices for proteins
            ligand_batch: Batch indices for ligands
            
        Returns:
            Dictionary with predicted vector field and auxiliary outputs
        """
        # Ensure all inputs are float32 for cuEquivariance
        protein_coords = protein_coords.float()
        protein_features = protein_features.float()
        ligand_coords = ligand_coords.float()
        ligand_features = ligand_features.float()
        t = t.float()
        
        # Encode time
        time_embedding = self.time_encoder(t)
        
        # Process protein features (use CA coordinates)
        protein_ca = protein_coords[:, 0, :]  # (N_protein, 3)
        protein_h = self.protein_encoder(protein_features)  # (N_protein, hidden_scalars)
        
        # Process ligand features
        ligand_h = self.ligand_encoder(ligand_features)  # (N_ligand, hidden_scalars)
        
        # Build edges and calculate distances
        edges, edge_distances, edge_vectors = self._build_edges(
            protein_ca, ligand_coords,
            protein_batch, ligand_batch
        )
        
        # Check if we have edges
        if edges.shape[1] == 0:
            # No edges found - return zero vector field
            return {
                'vector_field': torch.zeros_like(ligand_coords),
                'ligand_features': torch.zeros(ligand_coords.shape[0], self.out_dim, device=ligand_coords.device)
            }
        
        # Prepare edge features with time embedding
        # For batched inputs, we need to assign the correct time embedding to each edge
        if ligand_batch is not None and time_embedding.shape[0] > 1:
            # Get the batch ID for each edge from the ligand node
            edge_batch = ligand_batch[edges[0]]
            time_emb_expanded = time_embedding[edge_batch]
        else:
            # Single sample or single time value
            if time_embedding.shape[0] == 1:
                time_emb_expanded = time_embedding.expand(edge_distances.shape[0], -1)
            else:
                time_emb_expanded = time_embedding[0].unsqueeze(0).expand(edge_distances.shape[0], -1)
        
        edge_features = self.edge_encoder(
            torch.cat([edge_distances.unsqueeze(-1), time_emb_expanded], dim=-1)
        )
        
        # Combine all ligand nodes with protein context
        # For simplicity, we'll focus on ligand-protein interactions
        src, dst = edges  # src: ligand, dst: protein
        
        # Calculate spherical harmonics for edge vectors
        edge_sh = self.spherical_harmonics(edge_vectors.float())
        
        # Prepare input for equivariant processing
        # Combine both protein and ligand features into a single node feature tensor
        # We'll concatenate them and keep track of which are which
        num_ligand = ligand_coords.shape[0]
        num_protein = protein_ca.shape[0]
        
        # Project features to same dimension
        ligand_input = self.scalar_proj(ligand_h)  # (N_ligand, hidden_scalars)
        protein_input = self.scalar_proj(protein_h)  # (N_protein, hidden_scalars)
        
        # Initialize node features for both ligand and protein
        # Ligand nodes come first, then protein nodes
        combined_features = torch.cat([ligand_input, protein_input], dim=0)
        
        # Apply input embedding to all nodes
        h = self.input_embedding(combined_features.float())
        
        # Adjust edge indices for combined node ordering
        # Ligand indices stay the same (0 to num_ligand-1)
        # Protein indices need to be shifted by num_ligand
        adjusted_edges = edges.clone()
        adjusted_edges[1] = adjusted_edges[1] + num_ligand  # Shift protein indices
        
        # Graph structure for message passing - note total nodes is ligand + protein
        graph = (adjusted_edges, (num_ligand + num_protein, num_ligand + num_protein))
        
        # Apply equivariant message passing layers
        for i in range(self.num_layers):
            conv = self.conv_layers[i]
            norm = self.norm_layers[i]
            self_int = self.self_interaction_layers[i]
            
            # Message passing from protein to ligand
            # Create extended features for source nodes
            src_features = h.float()
            
            # Apply convolution
            messages = conv(
                src_features=src_features,
                edge_sh=edge_sh.float(),
                edge_emb=edge_features.float(),
                graph=graph
            )
            
            # Self-interaction
            self_update = self_int(h.float())
            
            # Normalize and update with residual
            h = h + norm(messages + self_update)
        
        # Output projection - only for ligand nodes
        ligand_h = h[:num_ligand]  # Extract ligand node features
        output = self.output_projection(ligand_h.float())
        
        # Split output into scalar and vector components
        # The output has scalars (0e) and vectors (1o)
        # For irreps "256x0e + 3x1o":
        # - 256x0e: 256 scalars (l=0, even) -> 256 dimensions
        # - 3x1o: 3 vectors (l=1, odd) -> 3*3=9 dimensions
        # Total: 265 dimensions
        
        out_scalars = output[:, :256]  # First 256 are scalars
        # Vectors are stored as 3 sets of 3D vectors (9 dims total)
        # Reshape and take mean to get a single 3D vector field
        all_vectors = output[:, 256:265].view(-1, 3, 3)  # (N, 3, 3) - 3 vectors of dim 3
        vector_field = all_vectors.mean(dim=1)  # Average the 3 vectors to get (N, 3)
        
        # Return only the vector field for pose refinement
        return {
            'vector_field': vector_field,
            'ligand_features': out_scalars
        }
    
    def _build_edges(
        self,
        protein_coords: torch.Tensor,
        ligand_coords: torch.Tensor,
        protein_batch: Optional[torch.Tensor] = None,
        ligand_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build edges between protein and ligand atoms within cutoff distance.
        Handles batched inputs by only connecting atoms within the same batch.
        
        Returns:
            edges: Edge indices (2, E) [ligand_idx, protein_idx]
            edge_distances: Distances for each edge (E,)
            edge_vectors: Normalized vectors for each edge (E, 3)
        """
        if protein_batch is None or ligand_batch is None:
            # Single sample case
            distances = torch.cdist(ligand_coords, protein_coords)  # (N_ligand, N_protein)
            edge_mask = distances < self.cutoff
            edges = torch.nonzero(edge_mask, as_tuple=False).T  # (2, E)
            
            edge_distances = distances[edge_mask]
            edge_vectors = ligand_coords[edges[0]] - protein_coords[edges[1]]
            edge_vectors_norm = F.normalize(edge_vectors, dim=-1)
            
            return edges, edge_distances, edge_vectors_norm
        
        # Batched case - only connect within same batch
        edge_list = []
        distance_list = []
        vector_list = []
        
        batch_ids = torch.unique(ligand_batch)
        
        for batch_id in batch_ids:
            # Get indices for this batch
            ligand_mask = ligand_batch == batch_id
            protein_mask = protein_batch == batch_id
            
            ligand_idx = torch.where(ligand_mask)[0]
            protein_idx = torch.where(protein_mask)[0]
            
            if len(ligand_idx) == 0 or len(protein_idx) == 0:
                continue
            
            # Get coordinates for this batch
            batch_ligand_coords = ligand_coords[ligand_mask]
            batch_protein_coords = protein_coords[protein_mask]
            
            # Calculate distances within batch
            distances = torch.cdist(batch_ligand_coords, batch_protein_coords)
            edge_mask = distances < self.cutoff
            
            # Get local edges
            local_edges = torch.nonzero(edge_mask, as_tuple=False).T
            
            if local_edges.shape[1] == 0:
                continue
            
            # Convert to global indices
            global_edges = torch.stack([
                ligand_idx[local_edges[0]],
                protein_idx[local_edges[1]]
            ])
            
            edge_list.append(global_edges)
            distance_list.append(distances[edge_mask])
            
            # Calculate edge vectors
            edge_vecs = ligand_coords[global_edges[0]] - protein_coords[global_edges[1]]
            vector_list.append(F.normalize(edge_vecs, dim=-1))
        
        if len(edge_list) == 0:
            # No edges found
            return (torch.zeros((2, 0), dtype=torch.long, device=ligand_coords.device),
                    torch.zeros(0, device=ligand_coords.device),
                    torch.zeros((0, 3), device=ligand_coords.device))
        
        # Concatenate all edges
        edges = torch.cat(edge_list, dim=1)
        edge_distances = torch.cat(distance_list)
        edge_vectors_norm = torch.cat(vector_list)
        
        return edges, edge_distances, edge_vectors_norm


class TimeEncoder(nn.Module):
    """Sinusoidal time position encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time step in [0, 1], shape (B,) or (1,)
            
        Returns:
            Time embedding of shape (B, dim) or (1, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.proj(emb)


# PhysicsGuidanceModule removed - focusing purely on flow matching for pose refinement


def test_equivariance(model: FlowFixEquivariantModel, device: str = 'cuda', initialize_weights: bool = True):
    """
    Test SE(3) equivariance of the FlowFix model.
    Tests both rotation and translation equivariance.
    
    Args:
        model: The FlowFixEquivariantModel to test
        device: Device to run tests on
        initialize_weights: Whether to initialize weights randomly
    
    Returns:
        Dictionary with test results and error metrics
    """
    import scipy.spatial.transform as R
    
    model.eval()
    model = model.to(device)
    
    # Initialize weights if requested to ensure non-zero outputs
    if initialize_weights:
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            elif hasattr(m, 'parameters'):
                # Initialize any layer with parameters (EquivariantLinear, Conv, etc.)
                try:
                    with torch.no_grad():
                        for param in m.parameters():
                            if param.dim() > 1:  # Weight matrices
                                param.data.uniform_(-0.1, 0.1)
                            else:  # Biases or other 1D params
                                param.data.uniform_(-0.01, 0.01)
                except:
                    pass  # Skip if initialization fails
        model.apply(init_weights)
    
    with torch.no_grad():
        # Create test data
        n_protein = 50
        n_ligand = 20
        
        # Random protein and ligand coordinates
        protein_coords = torch.randn(n_protein, 2, 3, device=device) * 10
        ligand_coords = torch.randn(n_ligand, 3, device=device) * 5
        
        # Random features - use the actual feature dimensions expected by the model
        protein_features = torch.randn(n_protein, 72, device=device)  # protein_feat_dim
        ligand_features = torch.randn(n_ligand, 14, device=device)    # ligand_feat_dim
        
        # Time step
        t = torch.tensor([0.5], device=device)
        
        # Get original output
        output_original = model(
            protein_coords=protein_coords,
            protein_features=protein_features,
            ligand_coords=ligand_coords,
            ligand_features=ligand_features,
            t=t
        )
        
        # Test 1: Rotation Equivariance
        print("\n" + "="*50)
        print("Testing Rotation Equivariance")
        print("="*50)
        
        # Generate random rotation matrix
        rotation = R.Rotation.random()
        rot_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=device)
        
        # Rotate coordinates
        protein_coords_rot = torch.matmul(protein_coords, rot_matrix.T)
        ligand_coords_rot = torch.matmul(ligand_coords, rot_matrix.T)
        
        # Get output with rotated coordinates
        output_rotated = model(
            protein_coords=protein_coords_rot,
            protein_features=protein_features,
            ligand_coords=ligand_coords_rot,
            ligand_features=ligand_features,
            t=t
        )
        
        # Rotate original output vector field
        vector_field_original_rot = torch.matmul(output_original['vector_field'], rot_matrix.T)
        
        # Calculate rotation equivariance error
        rotation_error = torch.mean(torch.abs(output_rotated['vector_field'] - vector_field_original_rot))
        vector_magnitude = torch.mean(torch.abs(output_original['vector_field']))
        
        # Handle zero or very small vector fields
        if vector_magnitude < 1e-8:
            print(f"Warning: Vector field magnitude is very small ({vector_magnitude.item():.6e})")
            print("This may indicate model initialization issues. Trying with different random initialization...")
            rotation_relative_error = float('inf')
        else:
            rotation_relative_error = rotation_error / vector_magnitude
        
        print(f"Vector field magnitude (original): {vector_magnitude.item():.6f}")
        print(f"Rotation equivariance error (absolute): {rotation_error.item():.6e}")
        print(f"Rotation equivariance error (relative): {rotation_relative_error:.6e}" if rotation_relative_error != float('inf') else "Rotation equivariance error (relative): N/A (zero vector field)")
        
        if rotation_relative_error < 1e-5 and rotation_relative_error != float('inf'):
            print("✓ Rotation equivariance test PASSED")
        else:
            print("✗ Rotation equivariance test FAILED")
        
        # Test 2: Translation Equivariance
        print("\n" + "="*50)
        print("Testing Translation Equivariance")
        print("="*50)
        
        # Generate random translation
        translation = torch.randn(1, 3, device=device) * 10
        
        # Translate coordinates
        protein_coords_trans = protein_coords + translation.unsqueeze(1)
        ligand_coords_trans = ligand_coords + translation
        
        # Get output with translated coordinates
        output_translated = model(
            protein_coords=protein_coords_trans,
            protein_features=protein_features,
            ligand_coords=ligand_coords_trans,
            ligand_features=ligand_features,
            t=t
        )
        
        # Translation should not affect the vector field (it's a difference vector)
        translation_error = torch.mean(torch.abs(output_translated['vector_field'] - output_original['vector_field']))
        if vector_magnitude < 1e-8:
            translation_relative_error = float('inf')
        else:
            translation_relative_error = translation_error / vector_magnitude
        
        print(f"Translation equivariance error (absolute): {translation_error.item():.6e}")
        print(f"Translation equivariance error (relative): {translation_relative_error:.6e}" if translation_relative_error != float('inf') else "Translation equivariance error (relative): N/A (zero vector field)")
        
        if translation_relative_error < 1e-5 and translation_relative_error != float('inf'):
            print("✓ Translation equivariance test PASSED")
        else:
            print("✗ Translation equivariance test FAILED")
        
        # Test 3: Combined Transformation
        print("\n" + "="*50)
        print("Testing Combined SE(3) Equivariance")
        print("="*50)
        
        # Apply both rotation and translation
        protein_coords_se3 = torch.matmul(protein_coords, rot_matrix.T) + translation.unsqueeze(1)
        ligand_coords_se3 = torch.matmul(ligand_coords, rot_matrix.T) + translation
        
        # Get output with SE(3) transformed coordinates
        output_se3 = model(
            protein_coords=protein_coords_se3,
            protein_features=protein_features,
            ligand_coords=ligand_coords_se3,
            ligand_features=ligand_features,
            t=t
        )
        
        # Expected output: rotated vector field
        vector_field_expected = torch.matmul(output_original['vector_field'], rot_matrix.T)
        
        # Calculate SE(3) equivariance error
        se3_error = torch.mean(torch.abs(output_se3['vector_field'] - vector_field_expected))
        if vector_magnitude < 1e-8:
            se3_relative_error = float('inf')
        else:
            se3_relative_error = se3_error / vector_magnitude
        
        print(f"SE(3) equivariance error (absolute): {se3_error.item():.6e}")
        print(f"SE(3) equivariance error (relative): {se3_relative_error:.6e}" if se3_relative_error != float('inf') else "SE(3) equivariance error (relative): N/A (zero vector field)")
        
        if se3_relative_error < 1e-5 and se3_relative_error != float('inf'):
            print("✓ SE(3) equivariance test PASSED")
        else:
            print("✗ SE(3) equivariance test FAILED")
        
        # Test 4: Permutation Equivariance (within protein/ligand groups)
        print("\n" + "="*50)
        print("Testing Permutation Equivariance")
        print("="*50)
        
        # Permute ligand atoms
        perm_ligand = torch.randperm(n_ligand, device=device)
        ligand_coords_perm = ligand_coords[perm_ligand]
        ligand_features_perm = ligand_features[perm_ligand]
        
        # Get output with permuted ligand
        output_perm = model(
            protein_coords=protein_coords,
            protein_features=protein_features,
            ligand_coords=ligand_coords_perm,
            ligand_features=ligand_features_perm,
            t=t
        )
        
        # Expected: permuted vector field
        vector_field_expected_perm = output_original['vector_field'][perm_ligand]
        
        # Calculate permutation equivariance error
        perm_error = torch.mean(torch.abs(output_perm['vector_field'] - vector_field_expected_perm))
        if vector_magnitude < 1e-8:
            perm_relative_error = float('inf')
        else:
            perm_relative_error = perm_error / vector_magnitude
        
        print(f"Permutation equivariance error (absolute): {perm_error.item():.6e}")
        print(f"Permutation equivariance error (relative): {perm_relative_error:.6e}" if perm_relative_error != float('inf') else "Permutation equivariance error (relative): N/A (zero vector field)")
        
        if perm_relative_error < 1e-5 and perm_relative_error != float('inf'):
            print("✓ Permutation equivariance test PASSED")
        else:
            print("✗ Permutation equivariance test FAILED")
        
        print("\n" + "="*50)
        print("Equivariance Test Summary")
        print("="*50)
        
        results = {
            'rotation_error': rotation_relative_error if rotation_relative_error != float('inf') else float('nan'),
            'translation_error': translation_relative_error if translation_relative_error != float('inf') else float('nan'),
            'se3_error': se3_relative_error if se3_relative_error != float('inf') else float('nan'),
            'permutation_error': perm_relative_error if perm_relative_error != float('inf') else float('nan'),
            'all_passed': all([
                rotation_relative_error < 1e-5 and rotation_relative_error != float('inf'),
                translation_relative_error < 1e-5 and translation_relative_error != float('inf'),
                se3_relative_error < 1e-5 and se3_relative_error != float('inf'),
                perm_relative_error < 1e-5 and perm_relative_error != float('inf')
            ])
        }
        
        if results['all_passed']:
            print("✓ ALL EQUIVARIANCE TESTS PASSED!")
        else:
            print("✗ Some equivariance tests failed. Check errors above.")
        
        return results


if __name__ == "__main__":
    """
    Run equivariance tests when module is executed directly.
    """
    print("Initializing FlowFix Equivariant Model for testing...")
    
    # Create model with default parameters
    model = FlowFixEquivariantModel(
        protein_feat_dim=72,
        ligand_feat_dim=14,
        edge_dim=32,
        hidden_scalars=48,
        hidden_vectors=16,
        hidden_dim=128,
        out_dim=256,
        num_layers=4,  # Fewer layers for faster testing
        max_ell=2,
        cutoff=10.0,
        time_embedding_dim=64,
        dropout=0.0  # No dropout for testing
    )
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on device: {device}")
    
    # Run equivariance tests
    results = test_equivariance(model, device=device)
    
    # Print final summary
    print("\n" + "="*50)
    print("Final Test Results:")
    print("="*50)
    for key, value in results.items():
        if key != 'all_passed':
            print(f"{key}: {value:.6e}")
    print(f"\nAll tests passed: {results['all_passed']}")