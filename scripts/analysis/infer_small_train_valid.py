#!/usr/bin/env python
import argparse
import json
import tempfile
from pathlib import Path

import torch
import yaml

from src.data.dataset import FlowFixDataset, collate_flowfix_batch
from src.utils.model_builder import build_model


def rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


@torch.no_grad()
def infer_one(model, sample, timesteps, device):
    batch = collate_flowfix_batch([sample])
    ligand_batch = batch["ligand_graph"].to(device)
    protein_batch = batch["protein_graph"].to(device)
    x0 = batch["ligand_coords_x0"].to(device)
    x1 = batch["ligand_coords_x1"].to(device)

    current = x0.clone()
    init_r = rmsd(x0, x1)

    for i in range(len(timesteps) - 1):
        t = torch.ones(1, device=device) * timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        ligand_batch_t = ligand_batch.clone()
        ligand_batch_t.pos = current.clone()
        v = model(protein_batch, ligand_batch_t, t)
        current = current + dt * v

    final_r = rmsd(current, x1)
    return init_r, final_r, batch["pdb_ids"][0]


def run_split(model, cfg, split, n, timesteps, device):
    ds = FlowFixDataset(
        data_dir=cfg["data"]["data_dir"],
        split_file=cfg["data"].get("split_file"),
        split=split,
        max_samples=None,
        seed=42,
        loading_mode="lazy",
        rank=0,
        fix_pose=cfg["data"].get("fix_pose", False),
        fix_pose_high_rmsd=cfg["data"].get("fix_pose_high_rmsd", False),
    )

    rows = []
    m = min(n, len(ds))
    for i in range(m):
        init_r, final_r, pdb_id = infer_one(model, ds[i], timesteps, device)
        rows.append(
            {
                "idx": i,
                "pdb_id": pdb_id,
                "initial_rmsd": init_r,
                "final_rmsd": final_r,
                "improvement": init_r - final_r,
            }
        )
    return rows


def summarize(rows):
    if not rows:
        return {"count": 0}
    n = len(rows)
    avg_init = sum(x["initial_rmsd"] for x in rows) / n
    avg_final = sum(x["final_rmsd"] for x in rows) / n
    avg_impr = sum(x["improvement"] for x in rows) / n
    return {
        "count": n,
        "avg_initial_rmsd": avg_init,
        "avg_final_rmsd": avg_final,
        "avg_improvement": avg_impr,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_rectified_flow_full.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_train", type=int, default=5)
    p.add_argument("--n_valid", type=int, default=5)
    p.add_argument("--output", default="inference_results/latest_train_valid_5.json")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument(
        "--train_pdb_all_poses",
        type=int,
        default=0,
        help="If > 0, take first N train PDB IDs and run inference on all poses for those PDBs.",
    )
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get("model", cfg["model"])
    model = build_model(model_cfg, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    use_ema = (not args.no_ema) and (ckpt.get("ema_state_dict") is not None)
    if use_ema:
        shadow = ckpt["ema_state_dict"].get("shadow", {})
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(param.device))
        print("applied_ema=True")
    else:
        print("applied_ema=False")

    model.eval()

    num_steps = int(cfg.get("sampling", {}).get("num_steps", 20))
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_steps": num_steps,
    }

    if args.train_pdb_all_poses > 0:
        with open(cfg["data"]["split_file"], "r") as f:
            split_data = json.load(f)
        selected_train_pdbs = split_data["train"][: args.train_pdb_all_poses]

        tmp_split = {
            "train": [],
            "val": selected_train_pdbs,
            "test": [],
        }
        tmp_path = Path(tempfile.gettempdir()) / "flowfix_train_allposes_split.json"
        tmp_path.write_text(json.dumps(tmp_split))

        ds_all_poses = FlowFixDataset(
            data_dir=cfg["data"]["data_dir"],
            split_file=str(tmp_path),
            split="valid",
            max_samples=None,
            seed=42,
            loading_mode="lazy",
            rank=0,
            fix_pose=False,
            fix_pose_high_rmsd=False,
        )

        rows = []
        for i in range(len(ds_all_poses)):
            init_r, final_r, pdb_id = infer_one(model, ds_all_poses[i], timesteps, device)
            rows.append(
                {
                    "idx": i,
                    "pdb_id": pdb_id,
                    "initial_rmsd": init_r,
                    "final_rmsd": final_r,
                    "improvement": init_r - final_r,
                }
            )

        per_pdb = {}
        for r in rows:
            per_pdb.setdefault(r["pdb_id"], []).append(r)
        per_pdb_summary = {
            pdb_id: summarize(v) for pdb_id, v in per_pdb.items()
        }

        result["train_pdb_all_poses"] = {
            "selected_train_pdbs": selected_train_pdbs,
            "rows": rows,
            "summary": summarize(rows),
            "per_pdb_summary": per_pdb_summary,
        }
        print("train_pdb_all_poses_summary", result["train_pdb_all_poses"]["summary"])
    else:
        train_rows = run_split(model, cfg, "train", args.n_train, timesteps, device)
        valid_rows = run_split(model, cfg, "valid", args.n_valid, timesteps, device)
        result["train"] = {"rows": train_rows, "summary": summarize(train_rows)}
        result["valid"] = {"rows": valid_rows, "summary": summarize(valid_rows)}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved={out}")

    if "train_pdb_all_poses" in result:
        print("selected_train_pdbs", result["train_pdb_all_poses"]["selected_train_pdbs"])
    else:
        print("train_summary", result["train"]["summary"])
        print("valid_summary", result["valid"]["summary"])


if __name__ == "__main__":
    main()
