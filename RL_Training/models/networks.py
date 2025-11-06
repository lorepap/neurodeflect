from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            x = F.relu(x, inplace=True)
        return self.ln(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden=hidden, layers=layers)
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.backbone(s)
        return self.logits(z)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden=hidden, layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.backbone(s)
        return self.head(z).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden=hidden, layers=layers)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.backbone(s)
        return self.head(z)


def make_nets(obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 3) -> Dict[str, nn.Module]:
    actor = Actor(obs_dim, n_actions, hidden=hidden, layers=layers)
    value = ValueNet(obs_dim, hidden=hidden, layers=layers)
    q = QNet(obs_dim, n_actions, hidden=hidden, layers=layers)
    target_value = ValueNet(obs_dim, hidden=hidden, layers=layers)
    target_value.load_state_dict(value.state_dict())
    return {"actor": actor, "value": value, "q": q, "target_value": target_value}
