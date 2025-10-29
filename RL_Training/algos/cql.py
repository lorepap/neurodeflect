from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CQL:
    def __init__(self, nets: Dict[str, nn.Module], lr: float, weight_decay: float, gamma: float, tau: float, beta: float, polyak: float, cql_alpha: float = 0.5, device: torch.device = torch.device("cpu"), use_amp: bool = False):
        self.nets = nets
        self.device = device
        for m in nets.values():
            m.to(self.device)
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = cql_alpha
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.use_amp:
            from torch.cuda.amp import GradScaler, autocast  # type: ignore
            self.scaler = GradScaler()
            self.autocast = autocast
        else:
            self.scaler = None
            from contextlib import nullcontext
            self.autocast = nullcontext

        params = list(nets["q"].parameters()) + list(nets["actor"].parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s = batch["s"].to(self.device)
        a = batch["a"].to(self.device)
        r = batch["r"].to(self.device)
        sp = batch["sp"].to(self.device)
        done = batch["done"].to(self.device)

        actor, q = self.nets["actor"], self.nets["q"]

        with self.autocast():
            # Q TD update
            with torch.no_grad():
                logits_next = actor(sp)
                prob_next = torch.softmax(logits_next, dim=-1)
                q_next = q(sp)
                v_next = (prob_next * q_next).sum(dim=-1)
                target = r + self.gamma * (1.0 - done) * v_next
            q_sa = q(s)
            q_a = q_sa.gather(1, a.view(-1, 1)).squeeze(1)
            td_loss = F.mse_loss(q_a, target)

            # CQL conservative term
            lse = torch.logsumexp(q_sa, dim=-1).mean()
            q_b = q_a.mean()
            cql_pen = self.alpha * (lse - q_b)

            # Actor (BC-style towards max-Q)
            logits = actor(s)
            logp = torch.log_softmax(logits, dim=-1)
            with torch.no_grad():
                a_star = torch.argmax(q_sa, dim=-1)
            actor_loss = F.nll_loss(logp, a_star)

            loss = td_loss + cql_pen + actor_loss
        self.opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(self.nets["q"].parameters()) + list(self.nets["actor"].parameters()), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.nets["q"].parameters()) + list(self.nets["actor"].parameters()), 5.0)
            self.opt.step()

        with torch.no_grad():
            probs = torch.softmax(actor(s), dim=-1)
            ent = (-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            mean_Q = q_sa.mean()
            std_Q = q_sa.std()

        return {
            "critic_loss": float((td_loss + cql_pen).item()),
            "actor_loss": float(actor_loss.item()),
            "policy_entropy": float(ent.item()),
            "mean_Q": float(mean_Q.item()),
            "std_Q": float(std_Q.item()),
        }
