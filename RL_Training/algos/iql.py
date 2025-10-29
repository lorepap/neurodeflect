from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def expectile_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.abs(tau - (u < 0).float())
    return (weight * (u ** 2)).mean()


class IQL:
    def __init__(self, nets: Dict[str, nn.Module], lr: float, weight_decay: float, gamma: float, tau: float, beta: float, polyak: float, cql_alpha: float = 0.0, device: torch.device = torch.device("cpu"), use_amp: bool = False):
        self.nets = nets
        self.device = device
        for m in nets.values():
            m.to(self.device)
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.polyak = polyak
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.use_amp:
            from torch.cuda.amp import GradScaler, autocast  # type: ignore
            self.scaler = GradScaler()
            self.autocast = autocast
        else:
            self.scaler = None
            from contextlib import nullcontext
            self.autocast = nullcontext

        params = list(nets["value"].parameters()) + list(nets["q"].parameters()) + list(nets["actor"].parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s = batch["s"].to(self.device)
        a = batch["a"].to(self.device)
        r = batch["r"].to(self.device)
        sp = batch["sp"].to(self.device)
        done = batch["done"].to(self.device)

        actor, value, q, target_value = self.nets["actor"], self.nets["value"], self.nets["q"], self.nets["target_value"]

        with self.autocast():
            # 1) Value update via expectile regression towards stop(Q - V) advantage
            with torch.no_grad():
                q_sa = q(s)
                q_selected = q_sa.gather(1, a.view(-1, 1)).squeeze(1)
            v_s = value(s)
            adv = q_selected - v_s
            v_loss = expectile_loss(adv, self.tau)

            # 2) Q update to r + gamma * V(sp)
            with torch.no_grad():
                v_sp = target_value(sp)
                target = r + self.gamma * (1.0 - done) * v_sp
            q_sa_all = q(s)
            q_a = q_sa_all.gather(1, a.view(-1, 1)).squeeze(1)
            q_loss = F.mse_loss(q_a, target)

            # 3) Actor update via advantage-weighted regression
            with torch.no_grad():
                v_s_detached = value(s)
                q_all = q(s)
                q_sel = q_all.gather(1, a.view(-1, 1)).squeeze(1)
                A = q_sel - v_s_detached
                weights = torch.clamp(torch.exp(A / max(self.beta, 1e-6)), max=100.0)
            logits = actor(s)
            logp = F.log_softmax(logits, dim=-1)
            logp_a = logp.gather(1, a.view(-1, 1)).squeeze(1)
            actor_loss = -(weights * logp_a).mean()

            loss = v_loss + q_loss + actor_loss
        self.opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(self.nets["value"].parameters()) + list(self.nets["q"].parameters()) + list(self.nets["actor"].parameters()), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.nets["value"].parameters()) + list(self.nets["q"].parameters()) + list(self.nets["actor"].parameters()), 5.0)
            self.opt.step()

        # Target value update
        with torch.no_grad():
            for p, tp in zip(self.nets["value"].parameters(), self.nets["target_value"].parameters()):
                tp.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)

        with torch.no_grad():
            probs = torch.softmax(actor(s), dim=-1)
            ent = (-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            mean_Q = q_sa_all.mean()
            std_Q = q_sa_all.std()

        return {
            "critic_loss": float((v_loss + q_loss).item()),
            "actor_loss": float(actor_loss.item()),
            "policy_entropy": float(ent.item()),
            "mean_Q": float(mean_Q.item()),
            "std_Q": float(std_Q.item()),
        }
