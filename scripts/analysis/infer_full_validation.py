#!/usr/bin/env python
"""
Full validation inference for FlowFix.
Runs all poses for all validation PDBs using batched DataLoader for speed.
"""
import argparse
import json
import signal
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from src.utils.model_builder import build_model


def per_molecule_rmsd(coords_a, coords_b, batch_idx):
    """Compute per-molecule RMSD given batched coordinates and batch indices."""
    rmsds = []
    for i in batch_idx.unique():
        mask = batch_idx == i
        diff = coords_a[mask] - coords_b[mask]
        rmsds.append(torch.sqrt(torch.mean(diff ** 2)).item())
    return rmsds


def summarize(rows):
    if not rows:
        return {"count": 0}
    n = len(rows)
    init_arr = np.array([r["initial_rmsd"] for r in rows])
    final_arr = np.array([r["final_rmsd"] for r in rows])
    imp_arr = np.array([r["improvement"] for r in rows])

    return {
        "count": n,
        "avg_initial_rmsd": float(np.mean(init_arr)),
        "avg_final_rmsd": float(np.mean(final_arr)),
        "avg_improvement": float(np.mean(imp_arr)),
        "median_initial_rmsd": float(np.median(init_arr)),
        "median_final_rmsd": float(np.median(final_arr)),
        "median_improvement": float(np.median(imp_arr)),
        "std_final_rmsd": float(np.std(final_arr)),
        "improved_count": int(np.sum(imp_arr > 0)),
        "improved_pct": float(np.mean(imp_arr > 0) * 100),
        "worsened_count": int(np.sum(imp_arr < 0)),
        "worsened_pct": float(np.mean(imp_arr < 0) * 100),
        "success_rate_2A": float(np.mean(final_arr < 2.0) * 100),
        "success_rate_1A": float(np.mean(final_arr < 1.0) * 100),
        "success_rate_05A": float(np.mean(final_arr < 0.5) * 100),
        "init_success_rate_2A": float(np.mean(init_arr < 2.0) * 100),
        "init_success_rate_1A": float(np.mean(init_arr < 1.0) * 100),
        "init_success_rate_05A": float(np.mean(init_arr < 0.5) * 100),
    }


def print_report(result):
    """Print a formatted report of the inference results."""
    print("\n" + "=" * 70)
    print("FULL VALIDATION INFERENCE REPORT")
    print("=" * 70)
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"EMA applied: {result['ema_applied']}")
    print(f"Num steps: {result['num_steps']}")
    print(f"Total time: {result['total_time_sec']:.0f}s")

    s = result["summary"]
    print(f"\n--- Overall Summary ({s['count']} poses, {result['num_pdbs']} PDBs) ---")
    print(f"  Initial RMSD:  {s['avg_initial_rmsd']:.3f} (median {s['median_initial_rmsd']:.3f}) A")
    print(f"  Final RMSD:    {s['avg_final_rmsd']:.3f} +/- {s['std_final_rmsd']:.3f} (median {s['median_final_rmsd']:.3f}) A")
    print(f"  Improvement:   {s['avg_improvement']:.3f} A (median {s['median_improvement']:.3f})")
    print(f"  Improved:      {s['improved_count']}/{s['count']} ({s['improved_pct']:.1f}%)")
    print(f"  Worsened:      {s['worsened_count']}/{s['count']} ({s['worsened_pct']:.1f}%)")

    print(f"\n--- Success Rates (Final RMSD < threshold) ---")
    print(f"  {'Threshold':>10s} | {'Initial':>10s} | {'Final':>10s} | {'Delta':>10s}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for label, ki, kf in [
        ("< 0.5 A", "init_success_rate_05A", "success_rate_05A"),
        ("< 1.0 A", "init_success_rate_1A", "success_rate_1A"),
        ("< 2.0 A", "init_success_rate_2A", "success_rate_2A"),
    ]:
        delta = s[kf] - s[ki]
        print(f"  {label:>10s} | {s[ki]:>9.1f}% | {s[kf]:>9.1f}% | {delta:>+9.1f}%")

    print(f"\n--- Per-PDB Summary (sorted by improvement) ---")
    per_pdb = result["per_pdb_summary"]
    sorted_pdbs = sorted(per_pdb.items(), key=lambda x: x[1]["avg_improvement"], reverse=True)

    print(f"  {'PDB':>6s} | {'#poses':>6s} | {'Init':>7s} | {'Final':>7s} | {'Improv':>8s} | {'<2A':>5s} | {'Worse%':>6s}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*7} | {'-'*7} | {'-'*8} | {'-'*5} | {'-'*6}")

    for pdb_id, ps in sorted_pdbs:
        print(
            f"  {pdb_id:>6s} | {ps['count']:>6d} | "
            f"{ps['avg_initial_rmsd']:>7.3f} | {ps['avg_final_rmsd']:>7.3f} | "
            f"{ps['avg_improvement']:>+8.3f} | {ps['success_rate_2A']:>4.0f}% | "
            f"{ps['worsened_pct']:>5.1f}%"
        )

    print(f"\n--- Top 10 Best Improved PDBs ---")
    for pdb_id, ps in sorted_pdbs[:10]:
        print(f"  {pdb_id}: {ps['avg_initial_rmsd']:.3f} -> {ps['avg_final_rmsd']:.3f} ({ps['avg_improvement']:+.3f} A, {ps['count']} poses)")

    print(f"\n--- Top 10 Worst PDBs (most worsened) ---")
    for pdb_id, ps in sorted_pdbs[-10:]:
        print(f"  {pdb_id}: {ps['avg_initial_rmsd']:.3f} -> {ps['avg_final_rmsd']:.3f} ({ps['avg_improvement']:+.3f} A, {ps['count']} poses)")

    print("=" * 70)


def plot_results(result, plot_dir: Path):
    """Generate scatter plots and histograms for inference results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    init_rmsds = np.array([r["initial_rmsd"] for r in rows])
    final_rmsds = np.array([r["final_rmsd"] for r in rows])
    improvements = np.array([r["improvement"] for r in rows])
    num_atoms = np.array([r["num_atoms"] for r in rows])

    # --- 1. Scatter: Initial vs Final RMSD (per pose) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(init_rmsds, final_rmsds, alpha=0.3, s=15, c="steelblue", edgecolors="none")
    lim = max(init_rmsds.max(), final_rmsds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (no change)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Initial RMSD (A)", fontsize=13)
    ax.set_ylabel("Final RMSD (A)", fontsize=13)
    ax.set_title("Per-Pose: Initial vs Final RMSD", fontsize=14)
    ax.set_aspect("equal")
    below = np.sum(final_rmsds < init_rmsds)
    above = np.sum(final_rmsds > init_rmsds)
    ax.text(0.05, 0.95, f"Improved: {below} ({below/len(rows)*100:.1f}%)",
            transform=ax.transAxes, fontsize=11, va="top", color="green")
    ax.text(0.05, 0.90, f"Worsened: {above} ({above/len(rows)*100:.1f}%)",
            transform=ax.transAxes, fontsize=11, va="top", color="red")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_init_vs_final_pose.png", dpi=150)
    plt.close(fig)

    # --- 2. Scatter: Initial vs Final RMSD (per PDB, averaged) ---
    per_pdb = result["per_pdb_summary"]
    pdb_ids = list(per_pdb.keys())
    pdb_init = np.array([per_pdb[p]["avg_initial_rmsd"] for p in pdb_ids])
    pdb_final = np.array([per_pdb[p]["avg_final_rmsd"] for p in pdb_ids])
    pdb_counts = np.array([per_pdb[p]["count"] for p in pdb_ids])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pdb_init, pdb_final, alpha=0.6, s=pdb_counts * 3, c="darkorange", edgecolors="k", linewidths=0.5)
    lim = max(pdb_init.max(), pdb_final.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Avg Initial RMSD (A)", fontsize=13)
    ax.set_ylabel("Avg Final RMSD (A)", fontsize=13)
    ax.set_title("Per-PDB: Avg Initial vs Final RMSD", fontsize=14)
    ax.set_aspect("equal")
    below_pdb = np.sum(pdb_final < pdb_init)
    ax.text(0.05, 0.95, f"Improved PDBs: {below_pdb}/{len(pdb_ids)} ({below_pdb/len(pdb_ids)*100:.1f}%)",
            transform=ax.transAxes, fontsize=11, va="top", color="green")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_init_vs_final_pdb.png", dpi=150)
    plt.close(fig)

    # --- 3. Scatter: Initial RMSD vs Improvement ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(improvements > 0, "steelblue", "tomato")
    ax.scatter(init_rmsds, improvements, alpha=0.3, s=15, c=colors, edgecolors="none")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Initial RMSD (A)", fontsize=13)
    ax.set_ylabel("Improvement (A)  [positive = better]", fontsize=13)
    ax.set_title("Initial RMSD vs Improvement", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_init_vs_improvement.png", dpi=150)
    plt.close(fig)

    # --- 4. Scatter: Num Atoms vs Improvement ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(num_atoms, improvements, alpha=0.3, s=15, c=colors, edgecolors="none")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Num Ligand Atoms", fontsize=13)
    ax.set_ylabel("Improvement (A)  [positive = better]", fontsize=13)
    ax.set_title("Ligand Size vs Improvement", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_atoms_vs_improvement.png", dpi=150)
    plt.close(fig)

    # --- 5. Histogram: RMSD distributions (before/after) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(init_rmsds.max(), final_rmsds.max()), 60)
    ax.hist(init_rmsds, bins=bins, alpha=0.5, label="Initial RMSD", color="gray", edgecolor="gray")
    ax.hist(final_rmsds, bins=bins, alpha=0.5, label="Final RMSD", color="steelblue", edgecolor="steelblue")
    ax.axvline(np.mean(init_rmsds), color="gray", ls="--", lw=1.5, label=f"Init mean: {np.mean(init_rmsds):.2f}")
    ax.axvline(np.mean(final_rmsds), color="steelblue", ls="--", lw=1.5, label=f"Final mean: {np.mean(final_rmsds):.2f}")
    ax.set_xlabel("RMSD (A)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("RMSD Distribution: Before vs After Refinement", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / "hist_rmsd_distribution.png", dpi=150)
    plt.close(fig)

    # --- 6. Histogram: Improvement distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(improvements, bins=60, alpha=0.7, color="steelblue", edgecolor="steelblue")
    ax.axvline(0, color="k", lw=1.5, ls="--")
    ax.axvline(np.mean(improvements), color="red", lw=1.5, ls="--",
               label=f"Mean: {np.mean(improvements):.3f} A")
    ax.axvline(np.median(improvements), color="orange", lw=1.5, ls="--",
               label=f"Median: {np.median(improvements):.3f} A")
    ax.set_xlabel("Improvement (A)  [positive = better]", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of RMSD Improvement", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / "hist_improvement.png", dpi=150)
    plt.close(fig)

    print(f"  Generated 6 plots in {plot_dir}")


@torch.no_grad()
def run_batched_inference(model, dataloader, num_steps, device, output_path):
    """Run ODE inference in batches using DataLoader. Saves partial results on SIGTERM."""
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    rows = []
    total_poses = 0
    start_time = time.time()
    interrupted = False

    def save_partial(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\nSIGTERM received! Saving partial results ({len(rows)} poses)...")

    signal.signal(signal.SIGTERM, save_partial)

    for batch_idx, batch in enumerate(dataloader):
        if interrupted:
            break

        ligand_batch = batch["ligand_graph"].to(device)
        protein_batch = batch["protein_graph"].to(device)
        x0 = batch["ligand_coords_x0"].to(device)
        x1 = batch["ligand_coords_x1"].to(device)
        pdb_ids = batch["pdb_ids"]
        batch_size = len(pdb_ids)

        current = x0.clone()

        # ODE integration
        for step in range(num_steps):
            t_val = timesteps[step]
            dt = timesteps[step + 1] - timesteps[step]
            t = torch.ones(batch_size, device=device) * t_val

            ligand_batch_t = ligand_batch.clone()
            ligand_batch_t.pos = current.clone()
            v = model(protein_batch, ligand_batch_t, t)
            current = current + dt * v

        # Per-molecule RMSD
        init_rmsds = per_molecule_rmsd(x0, x1, ligand_batch.batch)
        final_rmsds = per_molecule_rmsd(current, x1, ligand_batch.batch)

        for i in range(batch_size):
            mask = ligand_batch.batch == i
            rows.append({
                "idx": total_poses + i,
                "pdb_id": pdb_ids[i],
                "initial_rmsd": init_rmsds[i],
                "final_rmsd": final_rmsds[i],
                "improvement": init_rmsds[i] - final_rmsds[i],
                "num_atoms": int(mask.sum().item()),
            })

        total_poses += batch_size

        # Progress log
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            rate = total_poses / elapsed
            remaining = len(dataloader.dataset) - total_poses
            eta = remaining / rate if rate > 0 else 0
            partial = summarize(rows)
            print(
                f"  [{total_poses}/{len(dataloader.dataset)}] "
                f"init={partial['avg_initial_rmsd']:.3f} "
                f"final={partial['avg_final_rmsd']:.3f} "
                f"imp={partial['avg_improvement']:+.3f} "
                f"improved={partial['improved_pct']:.1f}% "
                f"{rate:.1f} poses/s "
                f"ETA={eta/60:.1f}min"
            )

    total_time = time.time() - start_time
    return rows, total_time, interrupted


def main():
    p = argparse.ArgumentParser(description="Full validation inference")
    p.add_argument("--config", default="configs/train_joint.yaml")
    p.add_argument("--checkpoint", default="save/rectified-flow-full-v4/checkpoints/latest.pt")
    p.add_argument("--output", default="inference_results/full_validation_results.json")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--num_steps", type=int, default=None, help="Override sampling steps")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get("model", cfg["model"])
    model = build_model(model_cfg, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    ema_applied = False
    if (not args.no_ema) and (ckpt.get("ema_state_dict") is not None):
        shadow = ckpt["ema_state_dict"].get("shadow", {})
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(param.device))
        ema_applied = True
        print("applied_ema=True")
    else:
        print("applied_ema=False")

    model.eval()

    num_steps = args.num_steps or int(cfg.get("sampling", {}).get("num_steps", 20))

    # Load full validation dataset
    ds = FlowFixDataset(
        data_dir=cfg["data"]["data_dir"],
        split_file=cfg["data"].get("split_file"),
        split="valid",
        max_samples=None,
        seed=42,
        loading_mode="lazy",
        rank=0,
        fix_pose=False,
        fix_pose_high_rmsd=False,
    )
    print(f"Validation dataset: {len(ds)} poses")

    # DataLoader for batched inference
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=collate_flowfix_batch,
    )
    print(f"Batches: {len(loader)} (batch_size={args.batch_size})")

    # Run batched inference
    out = Path(args.output)
    print(f"\nRunning {num_steps}-step Euler ODE integration...")
    rows, total_time, interrupted = run_batched_inference(model, loader, num_steps, device, out)

    # Per-PDB breakdown
    per_pdb_rows = defaultdict(list)
    for r in rows:
        per_pdb_rows[r["pdb_id"]].append(r)
    per_pdb_summary = {
        pdb_id: summarize(v) for pdb_id, v in per_pdb_rows.items()
    }

    # Build result
    result = {
        "checkpoint": args.checkpoint,
        "ema_applied": ema_applied,
        "device": str(device),
        "num_steps": num_steps,
        "batch_size": args.batch_size,
        "num_pdbs": len(per_pdb_summary),
        "total_poses": len(rows),
        "total_dataset_poses": len(ds),
        "completed": not interrupted,
        "total_time_sec": total_time,
        "summary": summarize(rows),
        "per_pdb_summary": per_pdb_summary,
        "rows": rows,
    }

    # Save JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {out} ({'COMPLETE' if not interrupted else 'PARTIAL'})")

    # Print report
    print_report(result)

    # Generate plots
    plot_dir = out.parent / out.stem
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_results(result, plot_dir)
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
