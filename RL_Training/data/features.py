from __future__ import annotations

import math
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


def compute_normalization(df: pd.DataFrame) -> Dict[str, Any]:
    # Per-switch z-score for congestion/latency
    stats = {}
    for col in ["queue_util", "queues_tot_util", "packet_latency"]:
        if col not in df.columns:
            continue
        grouped = df.groupby("switch_id")[col]
        mean = grouped.mean().to_dict()
        std = grouped.std().replace({0.0: 1.0}).fillna(1.0).to_dict()
        stats[col] = {"mean": mean, "std": std}
    return {"per_switch": stats}


def zscore_per_switch(switch_id: str, value: float, stats: Dict[str, Any], col: str) -> float:
    col_stats = stats.get("per_switch", {}).get(col)
    if col_stats is None:
        return float(value) if not pd.isna(value) else 0.0
    m = col_stats["mean"].get(switch_id, 0.0)
    sd = col_stats["std"].get(switch_id, 1.0)
    if sd == 0:
        sd = 1.0
    if pd.isna(value):
        return 0.0
    return float((value - m) / sd)


def ema(series: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(series, dtype=float)
    accum = 0.0
    for i, x in enumerate(series):
        accum = alpha * x + (1 - alpha) * accum
        out[i] = accum
    return out


def build_features_and_rewards(
    g: pd.DataFrame,
    history: int,
    ema_half_life_us: float,
    normalizer: Dict[str, Any],
    reward_weights: Dict[str, float],
    fct_min_max: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    # Extract arrays (ensure no NaNs by fallback)
    switch_id = str(g["switch_id"].iloc[0]) if "switch_id" in g.columns else "switch"
    ts = g["timestamp"].to_numpy(dtype=float)
    queue_util = g.get("queue_util", pd.Series(np.zeros(len(g)))).to_numpy(dtype=float)
    queues_tot_util = g.get("queues_tot_util", pd.Series(np.zeros(len(g)))).to_numpy(dtype=float)
    seq_num = g.get("seq_num", pd.Series(np.arange(len(g)))).to_numpy(dtype=float)
    flow_start_col = g.get("flow_start_time", pd.Series(np.full(len(g), ts[0] if len(ts)>0 else 0.0)))
    flow_end_col = g.get("flow_end_time", pd.Series(np.full(len(g), ts[-1] if len(ts)>0 else 1.0)))
    # replace NaNs with episode min/max timestamps
    fs_fill = ts[0] if len(ts) > 0 else 0.0
    fe_fill = ts[-1] if len(ts) > 0 else 1.0
    flow_start = pd.to_numeric(flow_start_col, errors="coerce").fillna(fs_fill).to_numpy(dtype=float)
    flow_end = pd.to_numeric(flow_end_col, errors="coerce").fillna(fe_fill).to_numpy(dtype=float)
    # packet latency may not be consistently available online; omit from feature set
    ooo = g.get("ooo", pd.Series(np.zeros(len(g)))).to_numpy(dtype=float)
    action = g.get("action_label", pd.Series(np.zeros(len(g)))).to_numpy(dtype=int)
    qid = g.get("query_id", pd.Series(np.full(len(g), np.nan))).to_numpy()
    qst = g.get("query_start_time", pd.Series(np.full(len(g), np.nan))).to_numpy(dtype=float)
    qet = g.get("query_end_time", pd.Series(np.full(len(g), np.nan))).to_numpy(dtype=float)
    fct = g.get("FCT", pd.Series(np.full(len(g), np.nan))).to_numpy(dtype=float)

    n = len(ts)
    if n <= 1:
        return {"s": np.zeros((0, 1), dtype=np.float32), "a": np.zeros((0,), dtype=np.int64),
                "r": np.zeros((0,), dtype=np.float32), "sp": np.zeros((0, 1), dtype=np.float32), "done": np.zeros((0,), dtype=bool)}

    # Normalized features
    q_util_z = np.array([zscore_per_switch(switch_id, v, normalizer, "queue_util") for v in queue_util])
    qt_util_z = np.array([zscore_per_switch(switch_id, v, normalizer, "queues_tot_util") for v in queues_tot_util])
    # [0,1] features (online-computable)
    # normalized sequence rank within episode
    ranks = (seq_num - np.nanmin(seq_num))
    denom = np.nanmax(seq_num) - np.nanmin(seq_num) + 1e-9
    seq_norm = np.clip(ranks / denom, 0.0, 1.0)

    # flow age
    flow_dur = (flow_end - flow_start)
    flow_age = np.clip((ts - flow_start) / (flow_dur + 1e-9), 0.0, 1.0)

    # EMA signals
    # convert half-life in microseconds to alpha per step assuming sorted timestamps
    if n >= 2:
        dt_us = np.clip(np.diff(ts) * 1e6, 1.0, None)
        median_dt = float(np.median(dt_us))
    else:
        median_dt = 1.0
    hl = max(ema_half_life_us, 1e-6)
    # continuous-time approximation: alpha = 1 - exp(-dt / hl)
    alpha = 1.0 - math.exp(-median_dt / hl)
    deflect_indicator = (action == 1).astype(float)
    deflect_ema = ema(deflect_indicator, alpha)
    ooo_recent = ema((ooo > 0).astype(float), alpha)

    # History stack for k observations (t-3..t)
    k = max(1, history)
    stack_feats = [q_util_z, qt_util_z, deflect_ema, ooo_recent]
    stacked = []
    for arr in stack_feats:
        # pad with zeros for first k-1 steps
        pad = np.zeros(k - 1)
        arr_pad = np.concatenate([pad, arr])
        # build rolling windows ending at t
        windows = []
        for i in range(n):
            w = arr_pad[i:i + k]
            windows.append(w)
        stacked.append(np.stack(windows, axis=0))  # [n, k]
    hist_stack = np.concatenate(stacked, axis=1)  # [n, k*len(stack_feats)]

    inst = np.stack([q_util_z, qt_util_z, seq_norm, flow_age, ooo_recent, deflect_ema], axis=1)
    X = np.concatenate([inst, hist_stack], axis=1)

    # actions
    A = action.astype(np.int64)

    # rewards
    w_q, w_l, w_o, w_d, w_F = reward_weights["w_q"], reward_weights["w_l"], reward_weights["w_o"], reward_weights["w_d"], reward_weights["w_F"]
    step_r = - (w_q * np.nan_to_num(queue_util, nan=0.0)
                + w_l * np.nan_to_num(queues_tot_util, nan=0.0)
                + w_o * (ooo > 0).astype(float))
    step_r = step_r - w_d * (A == 1).astype(float)

    # terminal FCT shaping on last row (apply only when FCT is present)
    r = step_r.copy()
    fct_min, fct_max = fct_min_max
    has_fct = not np.isnan(fct[-1])
    if has_fct:
        fct_norm = (fct[-1] - fct_min) / (max(fct_max - fct_min, 1e-9))
        r[-1] += - w_F * float(np.clip(fct_norm, 0.0, 1.0))

    # transitions
    S = X[:-1]
    SP = X[1:]
    A_t = A[:-1]
    A_next = A[1:]
    R = r[:-1]
    done = np.zeros_like(R, dtype=bool)
    done[-1] = True
    T = ts[:-1]

    # flag transitions that belong to flows with known FCT
    has_fct_arr = np.full(S.shape[0], 1 if has_fct else 0, dtype=np.int64)

    return {"s": S.astype(np.float32), "a": A_t.astype(np.int64), "a_next": A_next.astype(np.int64), "r": R.astype(np.float32),
            "sp": SP.astype(np.float32), "done": done, "t": T.astype(np.float32), "has_fct": has_fct_arr}
