#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from data.loader import load_transition_dataset
from models.networks import Actor
from eval.fqe import run_fqe


def build_actor_from_checkpoint(ckpt_path: Path, obs_dim: int, n_actions: int) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["nets"]["actor"]
    # infer hidden and layers from state dict
    hid = sd["logits.weight"].shape[1]
    n_act_sd = sd["logits.weight"].shape[0]
    assert n_act_sd == n_actions, f"n_actions mismatch: ckpt {n_act_sd} vs dataset {n_actions}"
    # count layers from backbone.fcs.*.weight keys
    layers = len([k for k in sd.keys() if k.startswith("backbone.fcs.") and k.endswith(".weight")])
    actor = Actor(obs_dim, n_actions, hidden=hid, layers=layers)
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def parse_args():
    p = argparse.ArgumentParser(description="Run quick FQE evaluation on a trained policy",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dirs", nargs='+', required=False, default=[])
    p.add_argument("--data-base", type=str, default="Omnet_Sims/dc_simulations/simulations/sims/tmp/data", help="Base directory where data_1G_<policy> folders live")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint_*.pt")
    p.add_argument("--out", required=False, help="Output JSON path (defaults to <ckpt_dir>/fqe_eval_quick.json)")
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-transitions", type=int, default=150000, help="Subsample transitions for faster FQE")
    p.add_argument("--t-min", type=float, default=None, help="Min timestamp (seconds) for filtering transitions")
    p.add_argument("--t-max", type=float, default=None, help="Max timestamp (seconds) for filtering transitions")
    p.add_argument("--require-fct", action="store_true", help="Keep only transitions from flows with known FCT")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "fqe_eval_quick.json"

    # Resolve dataset directories
    if args.data_dirs:
        data_dirs = [Path(d) for d in args.data_dirs]
    else:
        base = Path(args.data_base)
        POLICY_NAMES = [
            "dibs","ecmp","probabilistic","probabilistic_tb","random","random_tb","sd","threshold","threshold_tb","vertigo"
        ]
        expanded = [base / f"data_1G_{p}" for p in POLICY_NAMES]
        data_dirs = [p for p in expanded if p.exists()]
        if not data_dirs:
            data_dirs = sorted(base.glob("data_1G_*"))

    # Limit CSVs for quick FQE
    import os
    os.environ["PD_MAX_CSV_FILES"] = "3"
    os.environ["PD_ROW_LIMIT"] = "100000"
    ds, normalizer, meta = load_transition_dataset(data_dirs)
    obs_dim = ds["s"].shape[1]
    n_actions = int(ds["a"].max()) + 1

    # Build subset mask: time filter then subsample for speed
    N = ds["s"].shape[0]
    subset_mask = np.ones(N, dtype=bool)
    if args.t_min is not None:
        subset_mask &= (ds.get("t", np.zeros(N)) >= args.t_min)
    if args.t_max is not None:
        subset_mask &= (ds.get("t", np.zeros(N)) <= args.t_max)
    if args.require_fct and "has_fct" in ds:
        subset_mask &= (ds["has_fct"] == 1)
    idxs = np.where(subset_mask)[0]
    if len(idxs) == 0:
        print("[run_fqe] No transitions in the requested time window.")
        return
    if args.max_transitions and len(idxs) > args.max_transitions:
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, size=args.max_transitions, replace=False)
    subset = idxs

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    actor = build_actor_from_checkpoint(Path(args.ckpt), obs_dim, n_actions).to(device)

    res = run_fqe(ds, actor, gamma=args.gamma, device=device, steps=args.steps, batch_size=args.batch_size, subset=subset)
    with out_path.open("w") as f:
        json.dump(res, f, indent=2)
    print("[run_fqe] Wrote", out_path)


if __name__ == "__main__":
    main()
