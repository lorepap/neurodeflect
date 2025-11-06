import argparse
from pathlib import Path

import torch

from networks import Actor


def infer_model_hyperparams(state_dict):
    first_fc = state_dict["backbone.fcs.0.weight"]
    obs_dim = first_fc.shape[1]
    hidden_dim = first_fc.shape[0]
    num_layers = len([k for k in state_dict.keys() if k.startswith("backbone.fcs") and k.endswith(".weight")])
    n_actions = state_dict["logits.weight"].shape[0]
    return obs_dim, hidden_dim, num_layers, n_actions


def main():
    parser = argparse.ArgumentParser(description="Convert training checkpoint to TorchScript actor")
    parser.add_argument("--ckpt", required=True, help="Path to training checkpoint (torch.save) with actor state")
    parser.add_argument("--out", help="Destination TorchScript path (defaults to <ckpt_dir>/actor_<step>_scripted.pt)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "nets" not in ckpt or "actor" not in ckpt["nets"]:
        raise SystemExit("Checkpoint does not contain nets['actor'] state")

    state = ckpt["nets"]["actor"]
    obs_dim, hidden_dim, num_layers, n_actions = infer_model_hyperparams(state)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Inferred actor dimensions: obs_dim={obs_dim}, hidden_dim={hidden_dim}, layers={num_layers}, actions={n_actions}")

    actor = Actor(obs_dim, n_actions, hidden=hidden_dim, layers=num_layers)
    actor.load_state_dict(state)
    actor.eval()

    scripted = torch.jit.script(actor)

    if args.out:
        out_path = Path(args.out)
    else:
        suffix = ckpt_path.stem.replace("checkpoint_", "actor_") + "_scripted.pt"
        out_path = ckpt_path.with_name(suffix)

    scripted.save(str(out_path))
    print("Saved TorchScript actor to", out_path)


if __name__ == "__main__":
    main()
