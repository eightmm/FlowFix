
import torch
import unittest
from src.utils.relaxation import RelaxationEngine

class TestRelaxationEngine(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        
    def test_distance_geometry_relaxation(self):
        """Test if relaxation fixes bond length violation."""
        print("\nTesting Distance Geometry Relaxation...")
        
        # 1. Setup two atoms at distance 2.0 along X axis
        # Atom 0: [0, 0, 0], Atom 1: [2, 0, 0]
        init_coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=self.device)
        
        # 2. Define distance bounds: d(0,1) must be in [0.9, 1.1] (target ~1.0)
        # Shape: [Batch=1, MaxAtoms=2, MaxAtoms=2]
        lower_bound = torch.tensor([[[0.0, 0.9], [0.9, 0.0]]], device=self.device)
        upper_bound = torch.tensor([[[0.0, 1.1], [1.1, 0.0]]], device=self.device)
        num_atoms = torch.tensor([2], device=self.device)
        
        distance_bounds = {
            "lower": lower_bound,
            "upper": upper_bound,
            "num_atoms": num_atoms
        }
        
        # 3. Initialize Engine
        # strong DG weight, weak restraint to allow movement
        engine = RelaxationEngine(
            dg_weight=10.0, 
            restraint_weight=0.1, 
            lr=1.0, 
            max_steps=50
        )
        
        # 4. Run Relax
        relaxed_coords, metrics = engine.relax(
            ligand_coords=init_coords,
            distance_bounds=distance_bounds,
            device=self.device
        )
        
        # 5. Check result
        final_dist = torch.norm(relaxed_coords[0] - relaxed_coords[1]).item()
        print(f"Initial Distance: 2.0")
        print(f"Target Bound: [0.9, 1.1]")
        print(f"Final Distance: {final_dist:.4f}")
        print(f"Metrics: {metrics}")
        
        # Use simple assert
        self.assertTrue(final_dist < 1.5, "Distance should decrease significantly")
        self.assertTrue(final_dist > 0.8, "Distance should stay reasonable")
        self.assertTrue(metrics['total'] < 1.0, "Total energy should be low")

    def test_restraint_keeps_position(self):
        """Test if restraint prevents movement when no other forces exist."""
        print("\nTesting Restraint...")
        init_coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=self.device)
        
        engine = RelaxationEngine(
            restraint_weight=100.0,  # Very strong restraint
            lr=1.0
        )
        
        relaxed_coords, metrics = engine.relax(
            ligand_coords=init_coords,
            device=self.device
        )
        
        drift = torch.norm(relaxed_coords - init_coords).item()
        print(f"Drift with strong restraint: {drift:.6f}")
        self.assertTrue(drift < 1e-4, "Drift should be negligible with strong restraint")

if __name__ == '__main__':
    unittest.main()
