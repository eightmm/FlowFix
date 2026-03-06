#!/usr/bin/env python
"""
Generate small test dataset for FlowFix (fast testing)

Processes only a few PDB files for quick testing and validation.
"""

import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.make_feat import load_canonical_set_files, ProteinFeatureExtractor, LigandFeatureExtractor
from rdkit import Chem
import torch
from tqdm import tqdm


def process_test_samples(
    canonical_set_path: str = "/mnt/data/PROJECT/pdbbind_redocked_pose",
    output_dir: str = "./test_data",
    num_samples: int = 5,
    esmc_model: str = "esmc_300m",
    esm3_model: str = "esm3-open",
    device: str = "cuda"
):
    """
    Process a small number of samples for testing.

    Args:
        canonical_set_path: Path to canonical set directory
        output_dir: Output directory for processed features
        num_samples: Number of PDB samples to process
        esmc_model: ESMC model variant
        esm3_model: ESM3 model variant
        device: Device for ESM models
    """
    output_path = Path(output_dir)

    # Clean up existing test data
    if output_path.exists():
        print(f"🧹 Cleaning up existing test data: {output_dir}")
        shutil.rmtree(output_path)

    output_path.mkdir(exist_ok=True)

    # Load canonical set files
    canonical_files = load_canonical_set_files(canonical_set_path)

    # Get first N samples
    sample_ids = list(canonical_files.keys())[:num_samples]

    print(f"📦 Processing {num_samples} test samples: {sample_ids}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🧬 ESM models: {esmc_model} + {esm3_model}")
    print(f"🔧 Device: {device}")
    print()

    # Initialize protein feature extractor
    print("Initializing protein feature extractor...")
    protein_extractor = ProteinFeatureExtractor(
        distance_cutoff=4.0,
        esmc_model=esmc_model,
        esm3_model=esm3_model,
        device=device
    )
    print("✅ Protein extractor initialized!\n")

    # Process each sample
    for pdb_id in tqdm(sample_ids, desc="Processing test samples"):
        files = canonical_files[pdb_id]

        # Create output directory
        pdb_output_dir = output_path / pdb_id
        pdb_output_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {pdb_id}")
        print(f"{'='*60}")

        # Process protein
        if files['pdb']:
            pdb_file_path = files['pdb'][0]
            print(f"  📄 Protein: {Path(pdb_file_path).name}")

            try:
                protein_features = protein_extractor.extract_from_pdb(
                    pdb_path=pdb_file_path
                )
                torch.save(protein_features, pdb_output_dir / "protein.pt")
                print(f"  ✅ Protein saved: {pdb_output_dir / 'protein.pt'}")

                # Print protein info
                if isinstance(protein_features, dict):
                    if 'node' in protein_features:
                        print(f"     Residues: {protein_features['node']['coord'].shape[0]}")
                    if 'embeddings' in protein_features:
                        emb = protein_features['embeddings']
                        if 'esmc' in emb:
                            print(f"     ESMC embedding: {emb['esmc']['mean'].shape}")
                        if 'esm3' in emb:
                            print(f"     ESM3 embedding: {emb['esm3']['mean'].shape}")

            except Exception as e:
                print(f"  ❌ Error processing protein: {e}")
                continue

        # Process ligands
        if files['sdf']:
            sdf_file_path = files['sdf'][0]
            print(f"  📄 Ligands: {Path(sdf_file_path).name}")

            try:
                suppl = Chem.SDMolSupplier(sdf_file_path, removeHs=True)
                ligand_features_list = []
                molecules = []

                for mol in suppl:
                    if mol is not None:
                        molecules.append(mol)

                if len(molecules) > 0:
                    reference_mol = molecules[0]
                    ligand_extractor = LigandFeatureExtractor(use_hydrogen=False)

                    print(f"     Total poses: {len(molecules)}")

                    for i, mol in enumerate(molecules):
                        try:
                            ligand_data = ligand_extractor.extract_features(
                                mol=mol,
                                crystal_mol=reference_mol
                            )
                            ligand_features_list.append(ligand_data)
                        except Exception as e:
                            print(f"     ⚠️  Skipped pose {i}: {e}")
                            continue

                    if ligand_features_list:
                        torch.save(ligand_features_list, pdb_output_dir / "ligands.pt")
                        print(f"  ✅ Ligands saved: {pdb_output_dir / 'ligands.pt'}")
                        print(f"     Valid poses: {len(ligand_features_list)}")

                        # Print ligand info
                        sample_ligand = ligand_features_list[0]
                        print(f"     Atoms: {sample_ligand['coord'].shape[0]}")
                        print(f"     Fragments: {sample_ligand.get('n_fragments', 'N/A')}")
                        print(f"     Rotatable bonds: {len(sample_ligand.get('rotatable_edges', []))}")

            except Exception as e:
                print(f"  ❌ Error processing ligands: {e}")
                continue

    print(f"\n{'='*60}")
    print(f"✅ Test data generation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Processed samples: {num_samples}")


if __name__ == "__main__":
    process_test_samples(
        num_samples=5,           # Process 5 PDBs for testing
        output_dir="./test_data",
        esmc_model="esmc_600m",  # 1152-dim embeddings
        device="cuda"
    )
