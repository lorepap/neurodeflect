#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

from data.loader import load_transition_dataset
from data.replay import TransitionDataset
from models.networks import make_nets
from algos.iql import IQL
from algos.cql import CQL
from algos.awr import AWR
from eval.metrics import compute_behavior_kl


ALGOS = {
    "iql": IQL,
    "cql": CQL,
    "awr": AWR,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    def describe_cuda(index: int) -> None:
        name = torch.cuda.get_device_name(index)
        cap_major, cap_minor = torch.cuda.get_device_capability(index)
        print(f"[train] Using CUDA device cuda:{index} — {name} (SM {cap_major}.{cap_minor})")

    if requested == "auto":
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            torch.cuda.set_device(index)
            describe_cuda(index)
            return torch.device(f"cuda:{index}")
        print("[train] Auto-selected CPU (CUDA unavailable).")
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError):
        print(f"[train] Invalid device '{requested}'. Falling back to CPU.")
        return torch.device("cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("[train] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        describe_cuda(index)
        device = torch.device(f"cuda:{index}")
    else:
        print(f"[train] Using device: {device}")

    return device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline RL training for per-switch deflection policies",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = Path(__file__).resolve().parents[1]
    default_data_base = project_root / "Omnet_Sims/dc_simulations/simulations/sims/tmp/data"

    p.add_argument("--data-dirs", nargs='+', required=False, default=[], help="Dataset directories containing per-switch CSVs (optional if --data-base is provided)")
    p.add_argument("--data-base", type=str, default=str(default_data_base), help="Base directory where data_1G_<policy> folders live")
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="iql")
    p.add_argument("--out-dir", required=True, help="Output directory for logs and checkpoints")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.7, help="Expectile for IQL value net")
    p.add_argument("--beta", type=float, default=3.0, help="Temperature for advantage weighting")
    p.add_argument("--polyak", type=float, default=0.995)
    p.add_argument("--cql-alpha", type=float, default=0.5)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--save-interval", type=int, default=10000)
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from (continues for --steps more updates)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")

    # Reward weights
    p.add_argument("--w-q", type=float, default=1.0)
    p.add_argument("--w-l", type=float, default=0.3, help="Penalty weight for total queue utilization")
    p.add_argument("--w-o", type=float, default=0.2)
    p.add_argument("--w-d", type=float, default=0.3)
    p.add_argument("--w-f", type=float, default=0.5, help="Terminal FCT shaping weight")

    # Feature/history config
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--ema-half-life-us", type=float, default=80.0)

    # Network size
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = out_dir / "logs.csv"

    # Resolve dataset directories
    POLICY_NAMES = [
        "dibs",
        "ecmp",
        "probabilistic",
        "probabilistic_tb",
        "random",
        "random_tb",
        "sd",
        "threshold",
        "threshold_tb",
        "vertigo",
    ]
    if args.data_dirs:
        data_dirs = [Path(d).expanduser().resolve() for d in args.data_dirs]
    else:
        base = Path(args.data_base).expanduser().resolve()
        expanded = [base / f"data_1G_{p}" for p in POLICY_NAMES]
        data_dirs = [p for p in expanded if p.exists()]
        if not data_dirs:
            # Fallback: discover any data_1G_* under base
            data_dirs = sorted(base.glob("data_1G_*"))
        if not data_dirs:
            raise SystemExit(f"No dataset directories found under {base}. Provide --data-dirs or correct --data-base/--policies.")

    print("[train] Loading dataset…")
    ds, normalizer, meta = load_transition_dataset(
        data_dirs,
        history=args.history,
        ema_half_life_us=args.ema_half_life_us,
        reward_weights={"w_q": args.w_q, "w_l": args.w_l, "w_o": args.w_o, "w_d": args.w_d, "w_F": args.w_f},
    )

    print(f"[train] Transitions: {len(ds['s'])} | obs_dim={ds['s'].shape[1]} | actions={int(ds['a'].max())+1}")
    print(f"[train] Episodes: {int(ds['episode_starts'].sum())} (starts) | Policies: {sorted(set(meta['policy_labels']))}")

    # Save normalizer stats
    with (out_dir / "normalization.json").open("w") as f:
        json.dump(normalizer, f, indent=2)

    if args.dry_run:
        print("[train] Dry-run requested; exiting before training.")
        return

    # Build dataset and model
    dataset = TransitionDataset(ds)
    # Device selection
    device = resolve_device(args.device)
    if device.type == "cuda" and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    pin = (device.type == "cuda" and torch.cuda.is_available())
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=max(0, args.num_workers),
        pin_memory=pin,
    )

    obs_dim = ds["s"].shape[1]
    n_actions = int(ds["a"].max()) + 1
    nets = make_nets(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, layers=args.layers)

    base_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("nets", {})
            if state:
                for k, v in nets.items():
                    if k in state:
                        v.load_state_dict(state[k])
            base_step = int(ckpt.get("step", 0))
            print(f"[train] Resumed from {ckpt_path} at step {base_step}")
        else:
            print(f"[train] Resume checkpoint not found: {ckpt_path}")

    AlgoCls = ALGOS[args.algo]
    algo = AlgoCls(nets=nets, lr=args.lr, weight_decay=args.weight_decay, gamma=args.gamma,
                   tau=args.tau, beta=args.beta, polyak=args.polyak, cql_alpha=args.cql_alpha,
                   device=device, use_amp=(device.type == "cuda" and args.amp))

    # Logging setup
    if not log_csv_path.exists():
        with log_csv_path.open("w") as f:
            f.write("step,critic_loss,actor_loss,mean_Q,std_Q,policy_entropy,kl_to_behavior,deflect_rate\n")

    step = 0
    while step < args.steps:
        for batch in loader:
            metrics = algo.update(batch)
            step += 1
            global_step = base_step + step

            if step % args.log_interval == 0 or step == args.steps:
                # Diagnostics
                with torch.no_grad():
                    idx = np.random.choice(len(dataset), size=min(5000, len(dataset)), replace=False)
                    s_sample = torch.from_numpy(ds['s'][idx]).float().to(device)
                    pi = nets['actor'](s_sample)
                    probs = torch.softmax(pi, dim=-1)
                    probs_cpu = probs.detach().cpu()
                    deflect_rate = probs[:, 1].mean().item() if n_actions >= 2 else 0.0
                    kl = compute_behavior_kl(probs_cpu.numpy(), ds['a'][idx])
                row = {
                    "step": global_step,
                    "critic_loss": float(metrics.get("critic_loss", np.nan)),
                    "actor_loss": float(metrics.get("actor_loss", np.nan)),
                    "mean_Q": float(metrics.get("mean_Q", np.nan)),
                    "std_Q": float(metrics.get("std_Q", np.nan)),
                    "policy_entropy": float(metrics.get("policy_entropy", np.nan)),
                    "kl_to_behavior": float(kl),
                    "deflect_rate": float(deflect_rate),
                }
                with log_csv_path.open("a") as f:
                    f.write(
                        ",".join(str(row[k]) for k in ["step","critic_loss","actor_loss","mean_Q","std_Q","policy_entropy","kl_to_behavior","deflect_rate"]) + "\n"
                    )
                print("[train]", row)

            if step % args.save_interval == 0 or step == args.steps:
                ckpt = {
                    "step": global_step,
                    "algo": args.algo,
                    "nets": {k: v.state_dict() for k, v in nets.items()},
                    "normalizer": normalizer,
                    "meta": meta,
                }
                torch.save(ckpt, out_dir / f"checkpoint_{global_step}.pt")

            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
