from __future__ import annotations

import numpy as np


def compute_behavior_kl(policy_probs: np.ndarray, behavior_actions: np.ndarray) -> float:
    # policy_probs: [N, A], behavior_actions: [N]
    N, A = policy_probs.shape
    # behavior empirical distribution
    counts = np.bincount(behavior_actions.astype(int), minlength=A).astype(float)
    p_b = counts / max(counts.sum(), 1.0)
    p_pi = policy_probs.mean(axis=0)
    eps = 1e-8
    kl = np.sum(p_pi * (np.log(p_pi + eps) - np.log(p_b + eps)))
    return float(kl)

