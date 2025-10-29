from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AWR:
    def __init__(self, nets: Dict[str, nn.Module], lr: float, weight_decay: float, gamma: float, tau: float, beta: float, polyak: float, cql_alpha: float = 0.0, device: torch.device = torch.device("cpu"), use_amp: bool = False):
        self.nets = nets
        self.device = device
        for m in nets.values():
            m.to(self.device)
        self.gamma = gamma
        self.beta = beta
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.use_amp:
            from torch.cuda.amp import GradScaler, autocast  # type: ignore
            self.scaler_q = GradScaler()
            self.scaler_pi = GradScaler()
            self.autocast = autocast
        else:
            self.scaler_q = None
            self.scaler_pi = None
            from contextlib import nullcontext
            self.autocast = nullcontext

        self.opt_q = torch.optim.AdamW(nets["q"].parameters(), lr=lr, weight_decay=weight_decay)
        self.opt_pi = torch.optim.AdamW(nets["actor"].parameters(), lr=lr, weight_decay=weight_decay)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s = batch["s"].to(self.device)
        a = batch["a"].to(self.device)
        r = batch["r"].to(self.device)
        sp = batch["sp"].to(self.device)
        done = batch["done"].to(self.device)

        actor, q = self.nets["actor"], self.nets["q"]

        # Critic: TD(0) with v(s') approximated by max_a Q(s',a) (greedy)
        with self.autocast():
            with torch.no_grad():
                q_next = q(sp)
                v_next = q_next.max(dim=-1).values
                target = r + self.gamma * (1.0 - done) * v_next
            q_sa = q(s)
            q_a = q_sa.gather(1, a.view(-1, 1)).squeeze(1)
            td_loss = F.mse_loss(q_a, target)
        self.opt_q.zero_grad(set_to_none=True)
        if self.scaler_q is not None:
            self.scaler_q.scale(td_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.nets["q"].parameters(), 5.0)
            self.scaler_q.step(self.opt_q)
            self.scaler_q.update()
        else:
            td_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nets["q"].parameters(), 5.0)
            self.opt_q.step()

        # Actor: advantage-weighted BC with A = Q(s,a) - mean_a Q(s,a)
        with torch.no_grad():
            q_all = q(s)
            v_s = (torch.softmax(q_all, dim=-1) * q_all).sum(dim=-1)
            A = q_a - v_s
            weights = torch.clamp(torch.exp(A / max(self.beta, 1e-6)), max=100.0)
        with self.autocast():
            logits = actor(s)
            logp = F.log_softmax(logits, dim=-1)
            logp_a = logp.gather(1, a.view(-1, 1)).squeeze(1)
            actor_loss = -(weights * logp_a).mean()
        self.opt_pi.zero_grad(set_to_none=True)
        if self.scaler_pi is not None:
            self.scaler_pi.scale(actor_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.nets["actor"].parameters(), 5.0)
            self.scaler_pi.step(self.opt_pi)
            self.scaler_pi.update()
        else:
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nets["actor"].parameters(), 5.0)
            self.opt_pi.step()

        with torch.no_grad():
            probs = torch.softmax(actor(s), dim=-1)
            ent = (-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            mean_Q = q_sa.mean()
            std_Q = q_sa.std()

        return {
            "critic_loss": float(td_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "policy_entropy": float(ent.item()),
            "mean_Q": float(mean_Q.item()),
            "std_Q": float(std_Q.item()),
        }
