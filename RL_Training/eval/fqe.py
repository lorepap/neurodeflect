from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import QNet, Actor


def _fqe_core(s, a, r, sp, done, actor: nn.Module, gamma: float, device: torch.device, steps: int, batch_size: int = 4096) -> Dict[str, float]:
    n_actions = int(a.max().item()) + 1
    obs_dim = s.shape[1]

    q = QNet(obs_dim, n_actions).to(device)
    opt = torch.optim.AdamW(q.parameters(), lr=3e-4, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(s, a, r, sp, done)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    actor.eval()
    updates = 0
    while updates < max(1, steps):
        for sb, ab, rb, spb, db in loader:
            with torch.no_grad():
                logits = actor(spb)
                probs = torch.softmax(logits, dim=-1)
                q_next = q(spb)
                v_next = (probs * q_next).sum(dim=-1)
                target = rb + gamma * (1.0 - db) * v_next
            q_sa = q(sb).gather(1, ab.view(-1, 1)).squeeze(1)
            loss = F.mse_loss(q_sa, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()
            updates += 1
            if updates >= steps:
                break

    with torch.no_grad():
        logits = actor(s)
        probs = torch.softmax(logits, dim=-1)
        q_sa = q(s)
        v = (probs * q_sa).sum(dim=-1)
        V_hat = float(v.mean().item())
        vals = v.cpu().numpy()
    return {"V_hat": V_hat, "values": vals}


def run_fqe(ds: Dict[str, np.ndarray], actor: nn.Module, gamma: float, device: torch.device, steps: int = 50000, batch_size: int = 4096, subset: np.ndarray | None = None) -> Dict[str, Any]:
    if subset is None:
        s_np, a_np, r_np, sp_np, d_np = ds["s"], ds["a"], ds["r"], ds["sp"], ds["done"]
    else:
        s_np, a_np, r_np, sp_np, d_np = ds["s"][subset], ds["a"][subset], ds["r"][subset], ds["sp"][subset], ds["done"][subset]
    s = torch.from_numpy(s_np).float().to(device)
    a = torch.from_numpy(a_np).long().to(device)
    r = torch.from_numpy(r_np).float().to(device)
    sp = torch.from_numpy(sp_np).float().to(device)
    done = torch.from_numpy(d_np).float().to(device)

    core = _fqe_core(s, a, r, sp, done, actor, gamma, device, steps, batch_size=batch_size)
    V_hat = core["V_hat"]
    vals = core["values"]
    n = len(vals)
    B = 200
    boots = []
    rng = np.random.default_rng(0)
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        boots.append(float(vals[idx].mean()))
    mu = float(np.mean(boots))
    lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()

    # Behavior baseline via a_next if available
    result = {"fqe_return": V_hat, "bootstrap_mean": mu, "ci95": [lo, hi]}
    if "a_next" in ds:
        a_next_all = ds["a_next"][subset] if subset is not None else ds["a_next"]
        a_next = torch.from_numpy(a_next_all).long().to(device)
        n_actions = int(a.max().item()) + 1
        obs_dim = s.shape[1]
        q_b = QNet(obs_dim, n_actions).to(device)
        opt_b = torch.optim.AdamW(q_b.parameters(), lr=3e-4, weight_decay=1e-4)
        dataset_b = torch.utils.data.TensorDataset(s, a, r, sp, done, a_next)
        loader_b = torch.utils.data.DataLoader(dataset_b, batch_size=batch_size, shuffle=True, drop_last=True)
        updates = 0
        while updates < max(1, steps):
            for sb, ab, rb, spb, db, anb in loader_b:
                with torch.no_grad():
                    target_b = rb + gamma * (1.0 - db) * q_b(spb).gather(1, anb.view(-1,1)).squeeze(1)
                q_b_sa = q_b(sb).gather(1, ab.view(-1,1)).squeeze(1)
                loss_b = F.mse_loss(q_b_sa, target_b)
                opt_b.zero_grad(set_to_none=True)
                loss_b.backward()
                torch.nn.utils.clip_grad_norm_(q_b.parameters(), 5.0)
                opt_b.step()
                updates += 1
                if updates >= steps:
                    break
        with torch.no_grad():
            Vb = q_b(s).gather(1, a.view(-1,1)).squeeze(1).cpu().numpy()
        rngb = np.random.default_rng(0)
        bootsb = [float(Vb[rngb.choice(len(Vb), size=len(Vb), replace=True)].mean()) for _ in range(200)]
        result["behavior_return"] = {
            "fqe_return": float(np.mean(Vb)),
            "bootstrap_mean": float(np.mean(bootsb)),
            "ci95": np.percentile(bootsb, [2.5, 97.5]).tolist(),
        }
    if "policy_idx" in ds and "policy_to_idx" in ds:
        pol_idx = ds["policy_idx"]
        inv = {v: k for k, v in ds["policy_to_idx"].items()}
        per = {}
        per_behavior = {}
        for pid in sorted(set(pol_idx.tolist())):
            mask = (pol_idx == pid)
            if mask.sum() == 0:
                continue
            s2 = torch.from_numpy(ds["s"][mask]).float().to(device)
            a2 = torch.from_numpy(ds["a"][mask]).long().to(device)
            r2 = torch.from_numpy(ds["r"][mask]).float().to(device)
            sp2 = torch.from_numpy(ds["sp"][mask]).float().to(device)
            d2 = torch.from_numpy(ds["done"][mask]).float().to(device)
            core2 = _fqe_core(s2, a2, r2, sp2, d2, actor, gamma, device, int(max(100, steps//4)), batch_size=batch_size)
            vals2 = core2["values"]
            rng = np.random.default_rng(0)
            n2 = len(vals2)
            boots2 = [float(vals2[rng.choice(n2, size=n2, replace=True)].mean()) for _ in range(200)]
            per[inv.get(pid, str(pid))] = {
                "fqe_return": float(np.mean(vals2)),
                "bootstrap_mean": float(np.mean(boots2)),
                "ci95": np.percentile(boots2, [2.5, 97.5]).tolist(),
            }
            # Behavior per-policy if a_next available
            if "a_next" in ds:
                a_next2 = torch.from_numpy(ds["a_next"][mask]).long().to(device)
                q_b2 = QNet(s2.shape[1], int(a2.max().item()) + 1).to(device)
                opt_b2 = torch.optim.AdamW(q_b2.parameters(), lr=3e-4, weight_decay=1e-4)
                dataset_b2 = torch.utils.data.TensorDataset(s2, a2, r2, sp2, d2, a_next2)
                loader_b2 = torch.utils.data.DataLoader(dataset_b2, batch_size=batch_size, shuffle=True, drop_last=True)
                upd = 0
                while upd < max(1, steps//4):
                    for sb, ab, rb, spb, db, anb in loader_b2:
                        with torch.no_grad():
                            targ = rb + gamma * (1.0 - db) * q_b2(spb).gather(1, anb.view(-1,1)).squeeze(1)
                        q_sa_b = q_b2(sb).gather(1, ab.view(-1,1)).squeeze(1)
                        loss = F.mse_loss(q_sa_b, targ)
                        opt_b2.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(q_b2.parameters(), 5.0)
                        opt_b2.step()
                        upd += 1
                        if upd >= steps//4:
                            break
                with torch.no_grad():
                    Vb2 = q_b2(s2).gather(1, a2.view(-1,1)).squeeze(1).cpu().numpy()
                boots_b2 = [float(Vb2[rng.choice(len(Vb2), size=len(Vb2), replace=True)].mean()) for _ in range(200)]
                per_behavior[inv.get(pid, str(pid))] = {
                    "fqe_return": float(np.mean(Vb2)),
                    "bootstrap_mean": float(np.mean(boots_b2)),
                    "ci95": np.percentile(boots_b2, [2.5, 97.5]).tolist(),
                }
        result["per_policy"] = per
        if per_behavior:
            result["per_policy_behavior"] = per_behavior
    return result
