#!/usr/bin/env python
"""
Generate training data for FlowFix

This script processes PDB files and generates protein.pt and ligands.pt files
with all necessary features including:
- Protein: ESM embeddings (ESMC + ESM3) + geometric features
- Ligand: Fragment decomposition + Torsion angles + Distance bounds
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.make_feat import process_and_save_features


def main():
    """Generate training data with customizable parameters."""

    # Configuration
    config = {
        # Data paths
        "canonical_set_path": "/mnt/data/PROJECT/pdbbind_redocked_pose",
        "output_dir": "./train_data",

        # ESM models
        "esmc_model": "esmc_600m",      # Options: esmc_300m (960-dim), esmc_600m (1152-dim)
        "esm3_model": "esm3-open",       # ESM3 model (1536-dim)

        # Processing
        "device": "cuda",                # "cuda" or "cpu"
        "distance_cutoff": 4.0,          # Distance cutoff for protein edges (Å)
    }

    print("="*80)
    print("FlowFix Training Data Generation")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n" + "="*80)

    # Process data
    process_and_save_features(**config)

    print("\n" + "="*80)
    print("✅ Data generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
