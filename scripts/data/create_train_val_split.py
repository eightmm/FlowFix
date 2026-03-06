#!/usr/bin/env python3
"""
Create train/val split from pdbbind_data_by_year.tsv
- Exclude test set (Split=='test')
- Exclude core set (In_Core_Set==True)
- Exclude excluded samples (Split=='exclude')
- Randomly sample 200 for validation
- Use rest for training
"""

import pandas as pd
import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
tsv_file = Path("src/data/pdbbind_data_by_year.tsv")
data_dir = Path("train_data")
output_file = Path("train_data/splits.json")

print("=" * 80)
print("Creating Train/Val Split from PDBbind Data")
print("=" * 80)

# Read TSV file
print(f"\n📂 Reading {tsv_file}...")
df = pd.read_csv(tsv_file, sep='\t')
print(f"   Total entries in TSV: {len(df)}")

# Show column info
print(f"\n📊 Columns: {list(df.columns)}")
print(f"\n   Set distribution:")
print(df['Set'].value_counts())
print(f"\n   In_Core_Set distribution:")
print(df['In_Core_Set'].value_counts())
print(f"\n   Split distribution:")
print(df['Split'].value_counts())

# Filter out ONLY test set (ignore train/valid/exclude labels)
print(f"\n🔍 Filtering data...")
print(f"   Before filtering: {len(df)}")

# Only exclude test set - ignore all other Split labels
mask_not_test = df['Split'] != 'test'

df_filtered = df[mask_not_test].copy()
print(f"   After removing only test set: {len(df_filtered)}")

# Get list of available PDB IDs in train_data
print(f"\n📁 Checking available PDBs in {data_dir}...")
available_pdbs = set()
if data_dir.exists():
    for pdb_dir in data_dir.iterdir():
        if pdb_dir.is_dir() and (pdb_dir / "ligands.pt").exists():
            available_pdbs.add(pdb_dir.name)
    print(f"   Found {len(available_pdbs)} PDBs with ligands.pt")
else:
    print(f"   ⚠️  {data_dir} does not exist!")
    exit(1)

# Filter to only include available PDBs
df_available = df_filtered[df_filtered['PDB_ID'].isin(available_pdbs)].copy()
print(f"   PDBs in TSV and available in train_data: {len(df_available)}")

if len(df_available) < 200:
    print(f"\n❌ ERROR: Not enough data for validation! Only {len(df_available)} available.")
    print(f"   Need at least 200 samples for validation.")
    exit(1)

# Get all available PDB IDs
all_pdb_ids = df_available['PDB_ID'].tolist()
print(f"\n🎲 Randomly sampling validation set...")
print(f"   Total available PDBs: {len(all_pdb_ids)}")

# Randomly sample 200 for validation
val_pdb_ids = random.sample(all_pdb_ids, 200)
train_pdb_ids = [pdb for pdb in all_pdb_ids if pdb not in val_pdb_ids]

print(f"   Validation PDBs: {len(val_pdb_ids)}")
print(f"   Training PDBs: {len(train_pdb_ids)}")

# Create split dictionary
splits = {
    "train": sorted(train_pdb_ids),
    "val": sorted(val_pdb_ids),
    "test": []  # No test set for now
}

# Save to JSON
print(f"\n💾 Saving splits to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"   ✓ Saved!")

# Summary statistics
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print(f"Training samples:   {len(splits['train'])}")
print(f"Validation samples: {len(splits['val'])}")
print(f"Test samples:       {len(splits['test'])}")
print(f"Total:              {len(splits['train']) + len(splits['val']) + len(splits['test'])}")
print("\nFirst 10 validation PDB IDs:", splits['val'][:10])
print("\n✅ Split creation complete!")
