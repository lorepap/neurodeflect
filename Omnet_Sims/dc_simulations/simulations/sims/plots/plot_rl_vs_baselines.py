#!/usr/bin/env python3
"""
RL vs Baselines Plotter
=======================

This script produces publication-ready figures comparing the RL-based
deflection policy against classic baselines (Vertigo, DIBS, ECMP) and a
random policy.  It also visualises the deflection decision rate over time
for the RL policy.

Input directories follow the naming convention produced by the extraction
pipeline:
  - results_1G/                  (RL policy)
  - results_1G_vertigo/
  - results_1G_dibs/
  - results_1G_ecmp/
  - results_1G_random/

Each directory contains per-metric CSV files (FLOW_STARTED, FLOW_ENDED,
REQUEST_SENT, PACKET_ACTION, ...).  The script parses those vector CSVs
and computes:
  * Flow Completion Time (FCT) CDFs
  * Query Completion Time (QCT) CDFs (approximated as REQUEST_SENT -> FLOW_ENDED)
  * Deflection rate over time for the RL policy (from PACKET_ACTION vectors)

Usage examples:
    python plot_rl_vs_baselines.py \
        --rl-dir ../results_rl_policy \
        --out-dir plots

    python plot_rl_vs_baselines.py \
        --rl-dir ../results_rl_policy \
        --baseline vertigo:../results_1G_vertigo \
        --baseline dibs:../results_1G_dibs \
        --baseline ecmp:../results_1G_ecmp \
        --random ../results_1G_random \
        --out-dir plots

The script will create three PNG files inside --out-dir:
  * rl_vs_baselines_fct_qct.png
  * rl_vs_random_fct_qct.png
  * rl_deflection_timeline.png

"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims")
BACKGROUND_APP_CONNECTIONS = 1  # matches BASE_NUM_BACKGROUND_CONNECTIONS_TO_OTHER_SERVERS in sample_qct.py
DEFAULT_INCAST_SIZE = 40  # matches NUM_REQUESTS_PER_BURST for 1G configs unless overridden
APP_INDEX_PATTERN = re.compile(r"app\[(\d+)\]")


def parse_vector_csv(directory: Path) -> Iterable[Tuple[float, float]]:
    """Yield (time, value) pairs from all vector rows in CSV files under directory."""

    if not directory.exists():
        return []

    target_files = sorted(directory.glob("*.csv"))[:1]
    if not target_files:
        return []

    pairs: List[Tuple[float, float]] = []
    for csv_file in target_files:
        df = pd.read_csv(csv_file)
        for _, row in df[df["type"] == "vector"].iterrows():
            times = str(row["vectime"]).strip()
            values = str(row["vecvalue"]).strip()
            if not times or times == "nan" or not values or values == "nan":
                continue

            time_list = [float(t) for t in times.split()]
            value_list = [float(v) for v in values.split()]
            if len(time_list) != len(value_list):
                # Skip malformed rows but continue processing others
                continue

            pairs.extend(zip(time_list, value_list))
    return pairs


@dataclass
class FlowRecord:
    flow_id: int
    module: str
    send_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    query_id: int | None = None
    app_index: int | None = None
    is_burst: bool = False
    length: float | None = None
    request_index: int | None = None


@dataclass
class FlowDataset:
    flows: Dict[int, FlowRecord]
    request_sequences: Dict[str, List[int]]


@dataclass
class CompletionStats:
    flow_rate: float
    query_rate: float
    total_flows: int
    total_queries: int


def load_vector_values(directory: Path) -> Dict[str, List[float]]:
    """Return per-module flattened vector values from OMNeT++ CSV exports."""

    data: Dict[str, List[float]] = {}
    if not directory.exists():
        return data

    target_files = sorted(directory.glob("*.csv"))[:1]
    for csv_file in target_files:
        df = pd.read_csv(csv_file)
        for _, row in df[df["type"] == "vector"].iterrows():
            values = str(row["vecvalue"]).strip()
            if not values or values == "nan":
                continue
            try:
                value_list = [float(v) for v in values.split()]
            except ValueError:
                continue
            module = row["module"]
            data.setdefault(module, []).extend(value_list)
    return data


def compute_queue_utilization(results_dir: Path) -> np.ndarray:
    """Compute total queue utilization samples (QueuesTotLen / QueuesTotCapacity)."""

    len_dir = results_dir / "QUEUES_TOT_LEN"
    cap_dir = results_dir / "QUEUES_TOT_CAPACITY"
    if not len_dir.exists() or not cap_dir.exists():
        return np.array([])

    len_map = load_vector_values(len_dir)
    cap_map = load_vector_values(cap_dir)
    samples: List[float] = []

    for module, lengths in len_map.items():
        capacities = cap_map.get(module)
        if not capacities:
            continue
        min_len = min(len(lengths), len(capacities))
        if min_len == 0:
            continue
        length_arr = np.asarray(lengths[:min_len], dtype=float)
        capacity_arr = np.asarray(capacities[:min_len], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.divide(
                length_arr,
                capacity_arr,
                out=np.zeros_like(length_arr, dtype=float),
                where=capacity_arr > 0,
            )
        samples.extend(ratios.tolist())

    return np.array(samples, dtype=float)


def count_vector_events(directory: Path) -> int:
    """Count number of vector samples (events) emitted in all CSVs under directory."""

    if not directory.exists():
        return 0

    total = 0
    target_files = sorted(directory.glob("*.csv"))[:1]
    for csv_file in target_files:
        df = pd.read_csv(csv_file)
        for _, row in df[df["type"] == "vector"].iterrows():
            times = str(row["vectime"]).strip()
            if not times or times == "nan":
                continue
            total += len(times.split())
    return total


def sum_vector_values(directory: Path) -> float:
    """Sum all values from vector entries under directory (used for counting deflections)."""

    if not directory.exists():
        return 0.0

    total = 0.0
    target_files = sorted(directory.glob("*.csv"))[:1]
    for csv_file in target_files:
        df = pd.read_csv(csv_file)
        for _, row in df[df["type"] == "vector"].iterrows():
            values = str(row["vecvalue"]).strip()
            if not values or values == "nan":
                continue
            for token in values.split():
                try:
                    total += float(token)
                except ValueError:
                    continue
    return total


def compute_fct(results_dir: Path) -> np.ndarray:
    flow_data = load_flows(results_dir)
    flows = flow_data.flows
    fcts = []
    for flow in flows.values():
        if flow.is_burst:
            continue
        if flow.start_time is None or flow.end_time is None:
            continue
        if flow.end_time < flow.start_time:
            continue
        fcts.append(flow.end_time - flow.start_time)
    return np.array(fcts, dtype=float)


def load_flows(results_dir: Path) -> FlowDataset:
    flow_map: Dict[int, FlowRecord] = {}
    app_indices_seen: set[int] = set()
    request_sequences: Dict[str, List[int]] = {}
    reply_length_offsets: Dict[str, int] = {}
    flow_end_sequences: Dict[str, List[int]] = {}
    flow_end_offsets: Dict[str, int] = {}

    def parse_id_tokens(raw: str) -> List[int]:
        ids: List[int] = []
        for token in raw.split():
            try:
                ids.append(int(float(token)))
            except ValueError:
                continue
        return ids

    def parse_float_tokens(raw: str) -> List[float]:
        values: List[float] = []
        for token in raw.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    def ensure_flow(module: str, flow_id: int) -> FlowRecord:
        app_idx: int | None = None
        match = APP_INDEX_PATTERN.search(module)
        if match:
            app_idx = int(match.group(1))
            app_indices_seen.add(app_idx)

        flow = flow_map.get(flow_id)
        if flow is None:
            flow = FlowRecord(flow_id=flow_id, module=module, app_index=app_idx)
            flow_map[flow_id] = flow
            return flow

        if app_idx is not None:
            app_indices_seen.add(app_idx)
            if flow.app_index is None:
                flow.app_index = app_idx
        return flow

    # REQUEST_SENT -> initialise flows and set send_time
    request_dir = results_dir / "REQUEST_SENT"
    if request_dir.exists():
        for csv_file in request_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            for _, row in df[df["type"] == "vector"].iterrows():
                module = row["module"]
                times = str(row["vectime"]).strip()
                values = str(row["vecvalue"]).strip()
                if not times or not values or times == "nan" or values == "nan":
                    continue
                time_list = [float(t) for t in times.split()]
                value_list = parse_id_tokens(values)
                seq = request_sequences.setdefault(module, [])
                for t, fid in zip(time_list, value_list):
                    flow = ensure_flow(module, fid)
                    flow.send_time = t if flow.send_time is None else min(flow.send_time, t)
                    flow.request_index = len(seq)
                    seq.append(fid)

    # REPLY_LENGTH_ASKED -> capture per-flow reply sizes (parity with sample_qct)
    reply_dir = results_dir / "REPLY_LENGTH_ASKED"
    if reply_dir.exists():
        for csv_file in reply_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            for _, row in df[df["type"] == "vector"].iterrows():
                module = row["module"]
                values = str(row["vecvalue"]).strip()
                if not values or values == "nan":
                    continue
                seq = request_sequences.get(module)
                if not seq:
                    continue
                offset = reply_length_offsets.get(module, 0)
                for length in parse_float_tokens(values):
                    if offset >= len(seq):
                        break
                    fid = seq[offset]
                    offset += 1
                    flow = ensure_flow(module, fid)
                    flow.length = length
                reply_length_offsets[module] = offset

    # FLOW_STARTED -> record start time
    flow_started_dir = results_dir / "FLOW_STARTED"
    if flow_started_dir.exists():
        for csv_file in flow_started_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            for _, row in df[df["type"] == "vector"].iterrows():
                module = row["module"]
                times = str(row["vectime"]).strip()
                values = str(row["vecvalue"]).strip()
                if not times or not values or times == "nan" or values == "nan":
                    continue
                time_list = [float(t) for t in times.split()]
                value_list = parse_id_tokens(values)
                for t, fid in zip(time_list, value_list):
                    flow = ensure_flow(module, fid)
                    flow.start_time = t

    # FLOW_ENDED -> record end time
    flow_ended_dir = results_dir / "FLOW_ENDED"
    if flow_ended_dir.exists():
        for csv_file in flow_ended_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            for _, row in df[df["type"] == "vector"].iterrows():
                module = row["module"]
                times = str(row["vectime"]).strip()
                values = str(row["vecvalue"]).strip()
                if not times or not values or times == "nan" or values == "nan":
                    continue
                time_list = [float(t) for t in times.split()]
                value_list = parse_id_tokens(values)
                for t, fid in zip(time_list, value_list):
                    flow = ensure_flow(module, fid)
                    flow.end_time = t
                    flow_end_sequences.setdefault(module, []).append(fid)

    # FLOW_ENDED_QUERY_ID -> align query IDs with flow completions
    flow_query_dir = results_dir / "FLOW_ENDED_QUERY_ID"
    if flow_query_dir.exists():
        for csv_file in flow_query_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            for _, row in df[df["type"] == "vector"].iterrows():
                module = row["module"]
                values = str(row["vecvalue"]).strip()
                if not values or values == "nan":
                    continue
                sequence = flow_end_sequences.get(module)
                if not sequence:
                    continue
                offset = flow_end_offsets.get(module, 0)
                for qid in parse_id_tokens(values):
                    if offset >= len(sequence):
                        break
                    fid = sequence[offset]
                    offset += 1
                    flow = flow_map.get(fid)
                    if flow is not None:
                        flow.query_id = qid
                flow_end_offsets[module] = offset

    burst_threshold: int | None = None
    if app_indices_seen:
        lowest_app = min(app_indices_seen)
        burst_threshold = lowest_app + BACKGROUND_APP_CONNECTIONS

    for flow in flow_map.values():
        if flow.app_index is None or burst_threshold is None:
            flow.is_burst = False
        else:
            flow.is_burst = flow.app_index >= burst_threshold

    return FlowDataset(flows=flow_map, request_sequences=request_sequences)


def compute_qct(results_dir: Path) -> np.ndarray:
    flow_data = load_flows(results_dir)
    flow_map = flow_data.flows

    queries: Dict[Tuple[str, int], List[FlowRecord]] = {}
    for flow in flow_map.values():
        if not flow.is_burst or flow.query_id is None:
            continue
        queries.setdefault((flow.module, flow.query_id), []).append(flow)

    qcts: List[float] = []
    for flows in queries.values():
        if any(flow.send_time is None or flow.end_time is None for flow in flows):
            continue
        start = min(flow.send_time for flow in flows if flow.send_time is not None)
        end = max(flow.end_time for flow in flows if flow.end_time is not None)
        if end > start:
            qcts.append(end - start)
    return np.array(qcts, dtype=float)


def compute_completion_stats(results_dir: Path) -> CompletionStats:
    flow_data = load_flows(results_dir)
    flow_map = flow_data.flows
    request_sequences = flow_data.request_sequences

    background_flows = [flow for flow in flow_map.values() if not flow.is_burst and flow.send_time is not None]
    completed_background = [flow for flow in background_flows if flow.end_time is not None and flow.start_time is not None]
    flow_rate = (len(completed_background) / len(background_flows)) if background_flows else 0.0

    queries: Dict[Tuple[str, int], List[FlowRecord]] = {}
    for flow in flow_map.values():
        if not flow.is_burst or flow.query_id is None or flow.send_time is None:
            continue
        queries.setdefault((flow.module, flow.query_id), []).append(flow)

    completed_queries = 0
    total_queries = 0
    if queries:
        total_queries = len(queries)
        completed_queries = sum(
            1
            for group in queries.values()
            if all(f.start_time is not None and f.end_time is not None for f in group)
        )
    else:
        incast_size_hint = DEFAULT_INCAST_SIZE

        def iter_query_chunks(flow_ids: List[int], incast_size: int):
            for idx in range(0, len(flow_ids), incast_size):
                yield flow_ids[idx:idx + incast_size]

        for module, seq in request_sequences.items():
            burst_seq = [fid for fid in seq if fid in flow_map and flow_map[fid].is_burst]
            if not burst_seq or incast_size_hint <= 0:
                continue
            for group in iter_query_chunks(burst_seq, incast_size_hint):
                if not group or len(group) < incast_size_hint:
                    continue
                total_queries += 1
                if all(flow_map[fid].start_time is not None and flow_map[fid].end_time is not None for fid in group):
                    completed_queries += 1

    query_rate = (completed_queries / total_queries) if total_queries else 0.0

    return CompletionStats(flow_rate=flow_rate,
                           query_rate=query_rate,
                           total_flows=len(background_flows),
                           total_queries=total_queries)


def load_deflection_series(results_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    action_dir = results_dir / "PACKET_ACTION"
    pairs = list(parse_vector_csv(action_dir))
    if not pairs:
        return np.array([]), np.array([])

    times = np.array([t for t, _ in pairs], dtype=float)
    values = np.array([v for _, v in pairs], dtype=float)

    order = np.argsort(times)
    return times[order], values[order]


def load_deflection_events(results_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted (time, action) arrays without binning for deflection comparisons."""

    action_dir = results_dir / "PACKET_ACTION"
    pairs = list(parse_vector_csv(action_dir))
    if not pairs:
        return np.array([]), np.array([])

    times = np.array([t for t, _ in pairs], dtype=float)
    values = np.array([v for _, v in pairs], dtype=float)
    order = np.argsort(times)
    return times[order], values[order]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    sorted_vals = np.sort(values)
    probs = np.linspace(0, 1, len(sorted_vals), endpoint=False)
    probs = (np.arange(1, len(sorted_vals) + 1)) / len(sorted_vals)
    return sorted_vals, probs


def plot_cdf(ax, data: Dict[str, np.ndarray], title: str, xlabel: str):
    for label, values in data.items():
        x, y = compute_cdf(values)
        if x.size == 0:
            continue
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def summarize_dataset(name: str, values: np.ndarray):
    if values.size == 0:
        return f"{name}: no samples"
    return (f"{name}: n={len(values)}, mean={values.mean():.6f}, "
            f"median={np.median(values):.6f}, p95={np.percentile(values,95):.6f}")


def plot_rate_bars(ax, rates: Dict[str, float], title: str):
    labels = list(rates.keys())
    values = [rates[label] for label in labels]
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels))) if labels else []
    ax.bar(labels, values, color=colors[:len(labels)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Completion Rate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for idx, val in enumerate(values):
        ax.text(idx, min(val + 0.03, 1.02), f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=9)


def summarize_completion(name: str, stats: CompletionStats) -> str:
    return (f"{name}: flow={stats.flow_rate*100:.2f}% (n={stats.total_flows}), "
            f"query={stats.query_rate*100:.2f}% (n={stats.total_queries})")


def summarize_count(name: str, count: int) -> str:
    return f"{name}: {count}"


def plot_bar_metrics(ax, metrics: Dict[str, float], title: str, ylabel: str,
                     value_format: str = "{:.3f}", ymax: float | None = None):
    if not metrics:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    colors = plt.cm.Paired(np.linspace(0, 1, len(labels))) if labels else []
    ax.bar(labels, values, color=colors[:len(labels)])
    if ymax is None:
        max_val = max(values) if values else 1.0
        ymax = max_val * 1.15 if max_val > 0 else 1.0
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    offset = ymax * 0.02
    for idx, val in enumerate(values):
        ax.text(idx, min(val + offset, ymax * 0.98), value_format.format(val),
                ha="center", va="bottom", fontsize=9)


def summarize_queue(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: no samples"
    return (f"{name}: n={len(values)}, avg={values.mean():.4f}, "
            f"p95={np.percentile(values, 95):.4f}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Plot RL vs baseline comparisons (FCT, QCT, deflections)")
    parser.add_argument("--rl-dir", default=str(BASE_DIR / "results_rl_policy"),
                        help="Directory containing RL policy results (default: results_rl_policy under sims root)")
    parser.add_argument("--baseline", action="append", nargs="+", default=[],
                        help=("Baseline specification (name or name:path). "
                              "Supports multiple names per flag, e.g. "
                              "'--baseline probabilistic threshold'. "
                              "Defaults: vertigo,dibs,ecmp under sims root"))
    parser.add_argument("--random", default=str(BASE_DIR / "results_1G_random"),
                        help="Random baseline directory (default: results_1G_random under sims root)")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "plots"),
                        help="Output directory for figures")
    return parser


def parse_baselines(baseline_args: List[List[str]]) -> Dict[str, Path]:
    baselines: Dict[str, Path] = {
        "vertigo": BASE_DIR / "results_1G_vertigo",
        "dibs": BASE_DIR / "results_1G_dibs",
        "ecmp": BASE_DIR / "results_1G_ecmp",
        "probabilistic": BASE_DIR / "results_1G_probabilistic",
    }
    flattened: List[str] = []
    for entry in baseline_args:
        if isinstance(entry, str):
            flattened.append(entry)
        else:
            flattened.extend(entry)

    for item in flattened:
        if ":" in item:
            name, path = item.split(":", 1)
            baselines[name.strip()] = Path(path.strip()).expanduser().resolve()
        else:
            name = item.strip()
            if not name:
                raise ValueError("Baseline name cannot be empty")
            inferred = BASE_DIR / f"results_1G_{name}"
            baselines[name] = inferred.resolve()
    return baselines


def main():
    parser = build_argparser()
    args = parser.parse_args()

    rl_dir = Path(args.rl_dir).expanduser().resolve()
    baselines = parse_baselines(args.baseline)
    random_dir = Path(args.random).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    # Load RL datasets
    rl_fct = compute_fct(rl_dir)
    rl_qct = compute_qct(rl_dir)
    rl_completion = compute_completion_stats(rl_dir)
    rl_queue_util = compute_queue_utilization(rl_dir)
    rl_retransmissions = count_vector_events(rl_dir / "RETRANSMITTED")
    rl_ooo_segments = count_vector_events(rl_dir / "OOO_SEG")
    rl_deflections = sum_vector_values(rl_dir / "PACKET_ACTION")

    print("RL dataset summary:")
    print("  " + summarize_dataset("RL FCT", rl_fct))
    print("  " + summarize_dataset("RL QCT", rl_qct))
    print("  " + summarize_completion("RL completion", rl_completion))
    print("  " + summarize_queue("RL queue util", rl_queue_util))
    print("  " + summarize_count("RL retransmissions", rl_retransmissions))
    print("  " + summarize_count("RL out-of-order segments", rl_ooo_segments))
    print("  " + summarize_count("RL deflections", int(rl_deflections)))

    # Load baseline datasets
    baseline_fcts: Dict[str, np.ndarray] = {}
    baseline_qcts: Dict[str, np.ndarray] = {}
    flow_rates: Dict[str, float] = {"RL": rl_completion.flow_rate}
    query_rates: Dict[str, float] = {"RL": rl_completion.query_rate}
    queue_avg_metrics: Dict[str, float] = {}
    queue_p95_metrics: Dict[str, float] = {}
    queue_cdf_data: Dict[str, np.ndarray] = {}
    retransmission_counts: Dict[str, int] = {"RL": rl_retransmissions}
    ooo_counts: Dict[str, int] = {"RL": rl_ooo_segments}
    deflection_totals: Dict[str, float] = {"RL": rl_deflections}
    if rl_queue_util.size > 0:
        queue_avg_metrics["RL"] = float(rl_queue_util.mean())
        queue_p95_metrics["RL"] = float(np.percentile(rl_queue_util, 95))
        queue_cdf_data["RL"] = rl_queue_util

    for name, path in baselines.items():
        path = path.resolve()
        if not path.exists():
            print(f"Warning: baseline '{name}' directory not found at {path}; skipping")
            continue
        baseline_fcts[name] = compute_fct(path)
        baseline_qcts[name] = compute_qct(path)
        completion_stats = compute_completion_stats(path)
        print("  " + summarize_dataset(f"{name.upper()} FCT", baseline_fcts[name]))
        print("  " + summarize_dataset(f"{name.upper()} QCT", baseline_qcts[name]))
        print("  " + summarize_completion(f"{name.upper()} completion", completion_stats))
        queue_util = compute_queue_utilization(path)
        print("  " + summarize_queue(f"{name.upper()} queue util", queue_util))
        retrans_count = count_vector_events(path / "RETRANSMITTED")
        ooo_count = count_vector_events(path / "OOO_SEG")
        deflection_total = sum_vector_values(path / "PACKET_ACTION")
        print("  " + summarize_count(f"{name.upper()} retransmissions", retrans_count))
        print("  " + summarize_count(f"{name.upper()} out-of-order segments", ooo_count))
        print("  " + summarize_count(f"{name.upper()} deflections", int(deflection_total)))
        flow_rates[name.upper()] = completion_stats.flow_rate
        query_rates[name.upper()] = completion_stats.query_rate
        if queue_util.size > 0:
            queue_avg_metrics[name.upper()] = float(queue_util.mean())
            queue_p95_metrics[name.upper()] = float(np.percentile(queue_util, 95))
            queue_cdf_data[name.upper()] = queue_util
        retransmission_counts[name.upper()] = retrans_count
        ooo_counts[name.upper()] = ooo_count
        deflection_totals[name.upper()] = deflection_total

    # Load random baseline
    random_fct = compute_fct(random_dir) if random_dir.exists() else np.array([])
    random_qct = compute_qct(random_dir) if random_dir.exists() else np.array([])
    random_completion: CompletionStats | None = None
    random_queue_util = compute_queue_utilization(random_dir) if random_dir.exists() else np.array([])
    random_retransmissions = count_vector_events(random_dir / "RETRANSMITTED") if random_dir.exists() else 0
    random_ooo_segments = count_vector_events(random_dir / "OOO_SEG") if random_dir.exists() else 0
    random_deflections = sum_vector_values(random_dir / "PACKET_ACTION") if random_dir.exists() else 0.0
    if random_dir.exists():
        random_completion = compute_completion_stats(random_dir)
        print("  " + summarize_dataset("RANDOM FCT", random_fct))
        print("  " + summarize_dataset("RANDOM QCT", random_qct))
        print("  " + summarize_completion("RANDOM completion", random_completion))
        print("  " + summarize_queue("RANDOM queue util", random_queue_util))
        print("  " + summarize_count("RANDOM retransmissions", random_retransmissions))
        print("  " + summarize_count("RANDOM out-of-order segments", random_ooo_segments))
        print("  " + summarize_count("RANDOM deflections", int(random_deflections)))
        flow_rates["RANDOM"] = random_completion.flow_rate
        query_rates["RANDOM"] = random_completion.query_rate
        if random_queue_util.size > 0:
            queue_avg_metrics["RANDOM"] = float(random_queue_util.mean())
            queue_p95_metrics["RANDOM"] = float(np.percentile(random_queue_util, 95))
            queue_cdf_data["RANDOM"] = random_queue_util
        retransmission_counts["RANDOM"] = random_retransmissions
        ooo_counts["RANDOM"] = random_ooo_segments
        deflection_totals["RANDOM"] = random_deflections
    else:
        print(f"Warning: random baseline directory not found at {random_dir}")

    # Figure: Boxplots for FCT and QCT across policies
    boxplot_labels: List[str] = []
    boxplot_fct: List[np.ndarray] = []
    boxplot_qct: List[np.ndarray] = []

    def add_boxplot_entry(label: str, fct_values: np.ndarray, qct_values: np.ndarray):
        if (fct_values.size == 0) and (qct_values.size == 0):
            return
        boxplot_labels.append(label)
        boxplot_fct.append(fct_values if fct_values.size > 0 else np.array([np.nan]))
        boxplot_qct.append(qct_values if qct_values.size > 0 else np.array([np.nan]))

    add_boxplot_entry("RL", rl_fct, rl_qct)
    for name in sorted(baseline_fcts.keys()):
        add_boxplot_entry(name.upper(), baseline_fcts[name], baseline_qcts.get(name, np.array([])))
    add_boxplot_entry("RANDOM", random_fct, random_qct)

    if boxplot_labels:
        fig_box, axes_box = plt.subplots(1, 2, figsize=(12, 6))
        axes_box[0].boxplot(boxplot_fct, labels=boxplot_labels, showmeans=True)
        axes_box[0].set_title("Flow Completion Time")
        axes_box[0].set_ylabel("Time (s)")
        axes_box[0].tick_params(axis="x", rotation=45)
        for label in axes_box[0].get_xticklabels():
            label.set_ha("right")
        axes_box[0].grid(True, axis="y", alpha=0.3)

        axes_box[1].boxplot(boxplot_qct, labels=boxplot_labels, showmeans=True)
        axes_box[1].set_title("Query Completion Time")
        axes_box[1].set_ylabel("Time (s)")
        axes_box[1].tick_params(axis="x", rotation=45)
        for label in axes_box[1].get_xticklabels():
            label.set_ha("right")
        axes_box[1].grid(True, axis="y", alpha=0.3)

        fig_box.suptitle("Completion Time Distribution Comparison")
        fig_box.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_box.savefig(out_dir / "fct_qct_boxplot.png", dpi=300)
        plt.close(fig_box)

    p95_fct_metrics: Dict[str, float] = {}
    p95_qct_metrics: Dict[str, float] = {}
    if rl_fct.size > 0:
        p95_fct_metrics["RL"] = float(np.percentile(rl_fct, 95))
    if rl_qct.size > 0:
        p95_qct_metrics["RL"] = float(np.percentile(rl_qct, 95))
    for name, values in baseline_fcts.items():
        if values.size > 0:
            p95_fct_metrics[name.upper()] = float(np.percentile(values, 95))
    for name, values in baseline_qcts.items():
        if values.size > 0:
            p95_qct_metrics[name.upper()] = float(np.percentile(values, 95))
    if random_fct.size > 0:
        p95_fct_metrics["RANDOM"] = float(np.percentile(random_fct, 95))
    if random_qct.size > 0:
        p95_qct_metrics["RANDOM"] = float(np.percentile(random_qct, 95))

    # Figure 1: RL vs Vertigo/DIBS/ECMP (FCT & QCT)
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    baseline_fct_data = {"RL": rl_fct}
    baseline_qct_data = {"RL": rl_qct}
    for name, values in baseline_fcts.items():
        if values.size > 0:
            baseline_fct_data[name.upper()] = values
    for name, values in baseline_qcts.items():
        if values.size > 0:
            baseline_qct_data[name.upper()] = values

    plot_cdf(axes[0], baseline_fct_data, "FCT CDF", "Flow Completion Time (s)")
    plot_cdf(axes[1], baseline_qct_data, "QCT CDF", "Query Completion Time (s)")
    fig1.suptitle("RL vs Baseline Policies")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(out_dir / "rl_vs_baselines_fct_qct.png", dpi=300)

    # Figure 2: RL vs Random baseline
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    plot_cdf(axes2[0], {"RL": rl_fct, "Random": random_fct}, "FCT CDF", "Flow Completion Time (s)")
    plot_cdf(axes2[1], {"RL": rl_qct, "Random": random_qct}, "QCT CDF", "Query Completion Time (s)")
    fig2.suptitle("RL vs Random Baseline")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(out_dir / "rl_vs_random_fct_qct.png", dpi=300)

    # Figure 3: p95 FCT & QCT comparison
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    plot_bar_metrics(axes3[0], p95_fct_metrics, "p95 FCT Comparison", "Seconds")
    plot_bar_metrics(axes3[1], p95_qct_metrics, "p95 QCT Comparison", "Seconds")
    fig3.suptitle("p95 Completion Time Comparison")
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(out_dir / "p95_fct_qct_comparison.png", dpi=300)

    # Figure 4: Completion rate comparison
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    plot_rate_bars(axes4[0], flow_rates, "Flow Completion Rate")
    plot_rate_bars(axes4[1], query_rates, "Query Completion Rate")
    fig4.suptitle("Completion Rate Comparison")
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig4.savefig(out_dir / "completion_rate_comparison.png", dpi=300)

    # Figure 5: Queue utilization average & p95 comparison
    queue_avg_percent = {label: value * 100 for label, value in queue_avg_metrics.items()}
    queue_p95_percent = {label: value * 100 for label, value in queue_p95_metrics.items()}
    if queue_avg_metrics or queue_p95_metrics:
        fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
        plot_bar_metrics(axes5[0], queue_avg_percent, "Average Queue Utilization", "Utilization (%)",
                         value_format="{:.1f}%")
        plot_bar_metrics(axes5[1], queue_p95_percent, "p95 Queue Utilization", "Utilization (%)",
                         value_format="{:.1f}%")
        fig5.suptitle("Queue Utilization Comparison")
        fig5.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig5.savefig(out_dir / "queue_utilization_summary.png", dpi=300)

    # Figure 6: Queue utilization CDF
    if queue_cdf_data:
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        plot_cdf(ax6, queue_cdf_data, "Queue Utilization CDF", "Utilization (fraction)")
        fig6.tight_layout()
        fig6.savefig(out_dir / "queue_utilization_cdf.png", dpi=300)

    # Figure 7: Retransmissions and out-of-order segments
    if retransmission_counts or ooo_counts:
        fig7, axes7 = plt.subplots(1, 2, figsize=(12, 5))
        plot_bar_metrics(axes7[0], {k: float(v) for k, v in retransmission_counts.items()},
                         "Retransmissions", "Count", value_format="{:.0f}")
        plot_bar_metrics(axes7[1], {k: float(v) for k, v in ooo_counts.items()},
                         "Out-of-Order Segments", "Count", value_format="{:.0f}")
        fig7.suptitle("Retransmissions and OOO Segments")
        fig7.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig7.savefig(out_dir / "retransmissions_ooo_comparison.png", dpi=300)

    # Figure 8: Deflection rate over time
    def compute_deflection_series(directory: Path) -> Tuple[np.ndarray, np.ndarray]:
        times, actions = load_deflection_series(directory)
        if times.size == 0:
            return np.array([]), np.array([])
        num_bins = min(200, max(10, int(math.sqrt(len(times)))))
        bin_edges = np.linspace(times.min(), times.max(), num_bins + 1)
        bin_indices = np.digitize(times, bin_edges) - 1

        bin_centers = []
        binned_rates = []
        for idx in range(num_bins):
            mask = bin_indices == idx
            if not np.any(mask):
                continue
            center = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])
            rate = actions[mask].mean()
            bin_centers.append(center)
            binned_rates.append(rate)
        return np.array(bin_centers), np.array(binned_rates)

    rl_times, rl_rates = compute_deflection_series(rl_dir)
    if rl_times.size > 0:
        fig8, ax8 = plt.subplots(figsize=(10, 5))
        ax8.plot(rl_times, rl_rates, marker='o', linewidth=1.5, label="RL")

        series_specs: List[Tuple[str, Path]] = []
        for name in ("vertigo", "dibs"):
            directory = baselines.get(name)
            if directory is not None:
                series_specs.append((name.upper(), directory))
        if random_dir.exists():
            series_specs.append(("RANDOM", random_dir))

        for label, directory in series_specs:
            t, r = compute_deflection_series(directory)
            if t.size == 0:
                print(f"Warning: unable to locate PACKET_ACTION data for {label}; skipping")
                continue
            ax8.plot(t, r, linewidth=1.2, label=label)

        ax8.set_xlabel("Time (s)")
        ax8.set_ylabel("Deflection Rate")
        ax8.set_title("Deflection Rate Over Time")
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        fig8.tight_layout()
        fig8.savefig(out_dir / "deflection_rate_comparison.png", dpi=300)
    else:
        print("Warning: unable to locate PACKET_ACTION data for RL policy; skipping deflection plots")

    # Figure 9: Cumulative deflections over time
    event_specs: List[Tuple[str, Path]] = [("RL", rl_dir)]
    for name in ("vertigo", "dibs", "probabilistic"):
        directory = baselines.get(name)
        if directory and directory.exists():
            event_specs.append((name.upper(), directory))
    if random_dir.exists():
        event_specs.append(("RANDOM", random_dir))

    fig9, ax9 = plt.subplots(figsize=(10, 5))
    plotted_any = False
    for label, directory in event_specs:
        times, actions = load_deflection_events(directory)
        if times.size == 0:
            print(f"Warning: unable to locate PACKET_ACTION data for {label}; skipping cumulative chart")
            continue
        cumulative = np.cumsum(actions)
        ax9.step(times, cumulative, where="post", label=label)
        plotted_any = True

    if plotted_any:
        ax9.set_xlabel("Time (s)")
        ax9.set_ylabel("Cumulative Deflections")
        ax9.set_title("Deflections Over Time")
        ax9.grid(True, alpha=0.3)
        ax9.legend()
        fig9.tight_layout()
        fig9.savefig(out_dir / "deflections_over_time.png", dpi=300)
    else:
        plt.close(fig9)
        print("Warning: no PACKET_ACTION data available to plot cumulative deflections.")

    plt.close('all')
    print(f"Plots saved under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
