from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .features import build_features_and_rewards, compute_normalization

RUN_TIMESTAMP_PATTERN = re.compile(r"(?P<date>\d{8})-(?P<time>\d{2}:\d{2}:\d{2})")


def infer_policy_label_from_path(path: Path) -> str:
    # Expect patterns like data_1G_<policy>
    name = path.name
    if "data_1G_" in name:
        return name.split("data_1G_")[-1]
    # Fallback to trailing component
    return name


def extract_run_id_from_path(path: Path) -> str | None:
    stem = path.stem
    if "__" not in stem:
        return None
    return stem.split("__", 1)[0]


def parse_timestamp_from_run(run_id: str) -> datetime | None:
    match = RUN_TIMESTAMP_PATTERN.search(run_id)
    if not match:
        return None
    ts_str = f"{match.group('date')}-{match.group('time')}"
    try:
        return datetime.strptime(ts_str, "%Y%m%d-%H:%M:%S")
    except ValueError:
        return None


def select_latest_run_files(files: List[Path]) -> List[Path]:
    runs: Dict[str, List[Path]] = {}
    run_ts: Dict[str, datetime | None] = {}
    unassigned: List[Path] = []
    for p in files:
        run_id = extract_run_id_from_path(p)
        if run_id is None:
            unassigned.append(p)
            continue
        runs.setdefault(run_id, []).append(p)
        if run_id not in run_ts:
            run_ts[run_id] = parse_timestamp_from_run(run_id)
    if not runs:
        return files
    latest_run_id = max(
        runs.keys(),
        key=lambda rid: (
            run_ts.get(rid) or datetime.min,
            rid,
        ),
    )
    selected = sorted(runs[latest_run_id])
    if unassigned:
        selected.extend(sorted(unassigned))
    return selected


def read_dataset_dirs(data_dirs: List[Path], max_files: int | None = None, row_limit_per_file: int | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames: List[pd.DataFrame] = []
    policy_labels: List[str] = []
    for d in data_dirs:
        label = infer_policy_label_from_path(d)
        csvs = sorted(d.glob("*.csv"))
        if not csvs:
            # Allow nested structure
            csvs = sorted(d.rglob("*.csv"))
        if csvs:
            csvs = select_latest_run_files(csvs)
        if max_files is not None and len(csvs) > max_files:
            csvs = csvs[:max_files]
        if not csvs:
            print(f"[loader] warning: no CSVs found under {d}")
            continue
        for p in csvs:
            try:
                if row_limit_per_file is not None and row_limit_per_file > 0:
                    df = pd.read_csv(p, nrows=row_limit_per_file)
                else:
                    df = pd.read_csv(p)
                if df.empty:
                    continue
                df["_source_path"] = str(p)
                df["_policy"] = label
                frames.append(df)
                policy_labels.append(label)
            except Exception as e:
                print(f"[loader] failed to read {p}: {e}")
                continue

    if not frames:
        raise RuntimeError("No CSVs loaded from provided data directories")

    all_df = pd.concat(frames, ignore_index=True)
    # basic type coercion
    for col in [
        "timestamp", "seq_num", "queue_util", "queues_tot_util", "packet_latency",
        "RequesterID", "FlowID", "flow_start_time", "flow_end_time",
        "query_id", "query_start_time", "query_end_time", "QCT", "FCT",
    ]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
    # normalize action strings to int labels
    if "action" in all_df.columns:
        def map_action(a: Any) -> int:
            if pd.isna(a):
                return 0
            # numerical 0/1 or string
            try:
                v = float(a)
                return 1 if abs(v - 1.0) < 1e-6 else 0
            except Exception:
                s = str(a).upper()
                if "DEFLECT" in s:
                    return 1
                if s in ("1", "1.0", "DEFLECT", "D"):
                    return 1
                return 0
        all_df["action_label"] = all_df["action"].map(map_action)
    else:
        all_df["action_label"] = 0

    return all_df, {"policy_labels": sorted(set(policy_labels))}


def group_by_switch_flow(df: pd.DataFrame) -> List[Tuple[str, int, pd.DataFrame]]:
    required = ["switch_id", "FlowID"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} in dataset")
    groups = []
    for (switch_id, flow_id), g in df.groupby(["switch_id", "FlowID"], sort=False):
        g = g.sort_values(["timestamp", "seq_num"], kind="mergesort").reset_index(drop=True)
        groups.append((str(switch_id), int(flow_id) if pd.notna(flow_id) else -1, g))
    return groups


def load_transition_dataset(
    data_dirs: List[Path],
    history: int = 4,
    ema_half_life_us: float = 80.0,
    reward_weights: Dict[str, float] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    # Allow sampling control via env for quick runs
    import os
    max_files_env = os.environ.get("PD_MAX_CSV_FILES")
    row_limit_env = os.environ.get("PD_ROW_LIMIT")
    max_files = int(max_files_env) if max_files_env else None
    row_limit = int(row_limit_env) if row_limit_env else None

    df, meta = read_dataset_dirs(data_dirs, max_files=max_files, row_limit_per_file=row_limit)
    print(f"[loader] rows={len(df)} files≈{df['_source_path'].nunique()} switches≈{df['switch_id'].nunique()} flows≈{df['FlowID'].nunique()}")

    # Build per-(switch,flow) episodes
    groups = group_by_switch_flow(df)
    # Compute normalization using all rows
    normalizer = compute_normalization(df)

    S_list: List[np.ndarray] = []
    A_list: List[np.ndarray] = []
    R_list: List[np.ndarray] = []
    SP_list: List[np.ndarray] = []
    D_list: List[np.ndarray] = []
    T_list: List[np.ndarray] = []
    HF_list: List[np.ndarray] = []
    P_list: List[np.ndarray] = []  # policy index per transition
    AN_list: List[np.ndarray] = []  # a_next per transition
    E_starts: List[int] = []

    # Precompute FCT normalization by flow size bin
    # Use number of packets per (switch,flow) as proxy for size
    per_group_sizes = [len(g.index) for (_, _, g) in groups]
    quantiles = np.quantile(per_group_sizes, [0.33, 0.66]) if len(per_group_sizes) >= 3 else [np.median(per_group_sizes)]*2
    def size_bin(n: int) -> int:
        if n <= quantiles[0]:
            return 0
        if n <= quantiles[1]:
            return 1
        return 2

    # For each bin, compute FCT min/max across groups
    fct_bins: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    for (_, _, g) in groups:
        if "FCT" in g.columns and g["FCT"].notna().any():
            n = len(g.index)
            b = size_bin(n)
            # take FCT from last row if present
            fct_val = float(g["FCT"].dropna().iloc[-1])
            fct_bins[b].append(fct_val)
    fct_stats: Dict[int, Tuple[float, float]] = {}
    for b in fct_bins:
        if fct_bins[b]:
            fct_stats[b] = (min(fct_bins[b]), max(fct_bins[b]))
        else:
            fct_stats[b] = (0.0, 1.0)

    total_transitions = 0
    # map policy label to index
    all_policies = sorted(df["_policy"].dropna().unique().tolist())
    policy_to_idx = {p: i for i, p in enumerate(all_policies)}
    for (switch_id, flow_id, g) in groups:
        feats = build_features_and_rewards(
            g,
            history=history,
            ema_half_life_us=ema_half_life_us,
            normalizer=normalizer,
            reward_weights=reward_weights or {"w_q": 1.0, "w_l": 0.3, "w_o": 0.2, "w_d": 0.05, "w_F": 0.5},
            fct_min_max=fct_stats.get(size_bin(len(g.index)), (0.0, 1.0)),
        )

        s, a, r, sp, done = feats["s"], feats["a"], feats["r"], feats["sp"], feats["done"]
        if len(s) == 0:
            continue
        S_list.append(s)
        A_list.append(a)
        R_list.append(r)
        SP_list.append(sp)
        D_list.append(done.astype(np.float32))
        if "a_next" in feats:
            AN_list.append(feats["a_next"].astype(np.int64))
        if "t" in feats:
            T_list.append(feats["t"].astype(np.float32))
        if "has_fct" in feats:
            HF_list.append(feats["has_fct"].astype(np.int64))
        pol = str(g["_policy"].iloc[0]) if "_policy" in g.columns else "unknown"
        P_list.append(np.full(len(s), policy_to_idx.get(pol, -1), dtype=np.int64))
        E_starts.append(1)
        total_transitions += len(s)

    if not S_list:
        raise RuntimeError("No transitions constructed — check dataset columns and grouping")

    S = np.vstack(S_list).astype(np.float32)
    A = np.concatenate(A_list).astype(np.int64)
    R = np.concatenate(R_list).astype(np.float32)
    SP = np.vstack(SP_list).astype(np.float32)
    D = np.concatenate(D_list).astype(np.float32)
    episode_starts = np.array(E_starts, dtype=np.int64)
    P = np.concatenate(P_list).astype(np.int64) if P_list else np.zeros_like(A)
    T = np.concatenate(T_list).astype(np.float32) if T_list else np.zeros((S.shape[0],), dtype=np.float32)
    HF = np.concatenate(HF_list).astype(np.int64) if HF_list else np.zeros((S.shape[0],), dtype=np.int64)

    if AN_list:
        A_next = np.concatenate(AN_list).astype(np.int64)
    else:
        A_next = np.zeros_like(A)

    ds = {"s": S, "a": A, "a_next": A_next, "r": R, "sp": SP, "done": D, "t": T, "has_fct": HF, "episode_starts": episode_starts, "policy_idx": P, "policy_to_idx": policy_to_idx}
    meta = {**meta, "policy_to_idx": policy_to_idx}
    return ds, normalizer, meta
