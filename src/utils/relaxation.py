
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, Any

from src.utils.losses import compute_clash_loss, compute_distance_geometry_loss

class RelaxationEngine:
    """
    Post-processing relaxation engine using L-BFGS and physical force fields.
    
    Minimizes energy E = E_clash + E_bond + E_restraint
    - E_clash: VdW repulsion between protein and ligand
    - E_bond: Distance constraints (bond lengths/angles)
    - E_restraint: Harmonic restraint to initial predicted pose
    """
    def __init__(
        self,
        clash_weight: float = 1.0,
        dg_weight: float = 1.0,
        restraint_weight: float = 5.0,
        max_steps: int = 100,
        lr: float = 0.5,
        tolerance: float = 1e-4
    ):
        self.clash_weight = clash_weight
        self.dg_weight = dg_weight
        self.restraint_weight = restraint_weight
        self.max_steps = max_steps
        self.lr = lr
        self.tolerance = tolerance

    def relax(
        self,
        ligand_coords: torch.Tensor,
        protein_batch: Optional[Any] = None,
        distance_bounds: Optional[Dict[str, torch.Tensor]] = None,
        ligand_batch_indices: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run relaxation on ligand coordinates.

        Args:
            ligand_coords: [N_ligand, 3] Initial predicted coordinates
            protein_batch: PyG Batch/Data for protein (optional, for clash loss)
            distance_bounds: Dict with 'lower', 'upper' bounds (optional, for DG loss)
            ligand_batch_indices: [N_ligand] Batch indices (optional, defaults to zeros)
            device: Torch device

        Returns:
            relaxed_coords: [N_ligand, 3] Optimized coordinates
            metrics: Dict with final energy values
        """
        if device is None:
            device = ligand_coords.device

        # Clone to avoid modifying original
        x = ligand_coords.clone().detach().to(device)
        x.requires_grad_(True)
        
        # Original coordinates for restraint
        x_init = ligand_coords.clone().detach().to(device)

        # Batch indices
        if ligand_batch_indices is None:
            ligand_batch_indices = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        
        # optimizer = optim.LBFGS([x], lr=self.lr, max_iter=20, history_size=10) # L-BFGS often needs closure
        optimizer = optim.LBFGS([x], 
                                lr=self.lr, 
                                max_iter=self.max_steps, 
                                tolerance_change=self.tolerance, 
                                history_size=10,
                                line_search_fn='strong_wolfe')

        metrics = {}
        
        # dummy t for loss functions (t=1.0 for maximum constraint enforcement)
        t = torch.ones(ligand_batch_indices.max() + 1, device=device)

        def closure():
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            
            # 1. Restraint Loss (Harmonic potential)
            # E = k * (x - x_0)^2
            restraint_loss = torch.sum((x - x_init) ** 2) * self.restraint_weight
            loss = loss + restraint_loss
            metrics['restraint'] = restraint_loss.item()
            
            # 2. Clash Loss
            if protein_batch is not None:
                clash_ca, clash_sc = compute_clash_loss(
                    x_pred=x,
                    ligand_batch_indices=ligand_batch_indices,
                    protein_batch=protein_batch,
                    t=t,
                    device=device,
                    weight=self.clash_weight,
                    margin=0.5  # Stricter margin for refinement
                )
                loss = loss + clash_ca + clash_sc
                metrics['clash'] = (clash_ca + clash_sc).item()
                
            # 3. Distance Geometry Loss (Bond lengths/angles)
            if distance_bounds is not None:
                dg_loss = compute_distance_geometry_loss(
                    x_pred=x,
                    batch_indices=ligand_batch_indices,
                    distance_bounds=distance_bounds,
                    t=t,
                    device=device,
                    weight=self.dg_weight
                )
                loss = loss + dg_loss
                metrics['dg'] = dg_loss.item()
                
            loss.backward()
            metrics['total'] = loss.item()
            return loss

        # Run optimization
        optimizer.step(closure)
        
        return x.detach(), metrics
