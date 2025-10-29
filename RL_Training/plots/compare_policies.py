#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot RL vs Behavior FQE per policy",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--eval-json", required=True, help="Path to run_fqe output JSON containing per_policy and per_policy_behavior")
    p.add_argument("--out", required=True, help="Output PNG path")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.eval_json) as f:
        data = json.load(f)
    per = data.get("per_policy", {})
    per_b = data.get("per_policy_behavior", {})
    policies = sorted(set(list(per.keys()) + list(per_b.keys())))
    rl_vals = [per.get(p, {}).get("fqe_return", float('nan')) for p in policies]
    beh_vals = [per_b.get(p, {}).get("fqe_return", float('nan')) for p in policies]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(policies))
    width = 0.35
    ax.bar([i - width/2 for i in x], beh_vals, width, label='Behavior (logged)')
    ax.bar([i + width/2 for i in x], rl_vals, width, label='RL (new policy)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies, rotation=30, ha='right')
    ax.set_ylabel('FQE Return (higher better)')
    ax.set_title('Per-policy FQE: Behavior vs New RL Policy')
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("[plot] Wrote", args.out)


if __name__ == '__main__':
    main()

