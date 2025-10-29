from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, ds: Dict[str, np.ndarray]):
        self.s = ds["s"]
        self.a = ds["a"]
        self.r = ds["r"]
        self.sp = ds["sp"]
        self.done = ds["done"]

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "s": torch.from_numpy(self.s[idx]).float(),
            "a": torch.tensor(self.a[idx], dtype=torch.long),
            "r": torch.tensor(self.r[idx], dtype=torch.float32),
            "sp": torch.from_numpy(self.sp[idx]).float(),
            "done": torch.tensor(self.done[idx], dtype=torch.float32),
        }

