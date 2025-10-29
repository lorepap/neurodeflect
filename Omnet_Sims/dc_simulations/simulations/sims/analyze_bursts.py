#!/usr/bin/env python3
"""Analyze queue bursts and deflection activity from scavetool CSV exports."""

from __future__ import annotations

import argparse
import csv
import pathlib
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Tuple, Optional
import sys

import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)


@dataclass
class QueueSample:
    time: float
    util: float


@dataclass
class Burst:
    start: float
    end: float
    duration: float
    deflections: int


CSV_COLUMNS = [
    "run",
    "type",
    "module",
    "name",
    "attrname",
    "attrvalue",
    "value",
    "vectime",
    "vecvalue",
]

def load_csv_vectors(directory: pathlib.Path, vector_name: str, module_filter) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Load vectors grouped by run and module from scavetool CSV-R exports."""
    vectors: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    if not directory.exists():
        return {}
    for csv_path in sorted(directory.glob("*.csv")):
        with csv_path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                continue
            for row in reader:
                if len(row) < len(CSV_COLUMNS):
                    continue
                run_id, row_type, module, name = row[0], row[1], row[2], row[3]
                if row_type != "vector" or name != vector_name:
                    continue
                if module_filter and not module_filter(module):
                    continue
                times_str = row[7].strip()
                values_str = row[8].strip()
                if not times_str or not values_str:
                    continue
                times = [float(tok) for tok in times_str.split()]
                values = [float(tok) for tok in values_str.split()]
                if not times or not values:
                    continue
                if len(times) != len(values):
                    length = min(len(times), len(values))
                    times = times[:length]
                    values = values[:length]
                vectors[run_id][module].extend(zip(times, values))
    return vectors

def collect_queue_series(queue_dir: pathlib.Path, capacity_dir: pathlib.Path) -> Dict[str, Dict[str, List[QueueSample]]]:
    module_filter = lambda module: ".relayUnit" in module
    length_vectors = load_csv_vectors(queue_dir, "QueueLen:vector", module_filter)
    capacity_vectors = load_csv_vectors(capacity_dir, "QueueCapacity:vector", module_filter)

    queue_series: Dict[str, Dict[str, List[QueueSample]]] = {}
    for run_id, modules in length_vectors.items():
        run_result: Dict[str, List[QueueSample]] = {}
        for module, samples in modules.items():
            cap_samples = capacity_vectors.get(run_id, {}).get(module, [])
            if not samples or not cap_samples:
                continue
            samples.sort(key=lambda x: x[0])
            cap_samples.sort(key=lambda x: x[0])
            cap_times = [t for t, _ in cap_samples]
            cap_values = [v for _, v in cap_samples]
            if not cap_times:
                continue

            def capacity_at(time: float) -> float:
                idx = bisect_right(cap_times, time) - 1
                if idx < 0:
                    idx = 0
                return cap_values[idx]

            normalised: List[QueueSample] = []
            for time, length in samples:
                cap_val = capacity_at(time)
                if cap_val <= 0:
                    continue
                util = length / cap_val
                normalised.append(QueueSample(time=time, util=max(0.0, min(util, 1.0))))
            if normalised:
                run_result[module] = normalised
        if run_result:
            queue_series[run_id] = run_result
    return queue_series

def collect_deflections(packet_dir: pathlib.Path) -> Dict[str, Dict[str, List[float]]]:
    module_filter = lambda module: ".relayUnit" in module
    vectors = load_csv_vectors(packet_dir, "PacketAction:vector", module_filter)
    deflections: Dict[str, Dict[str, List[float]]] = {}
    for run_id, modules in vectors.items():
        run_defs: Dict[str, List[float]] = {}
        for module, samples in modules.items():
            times = [time for time, value in samples if value > 0.0]
            if times:
                times.sort()
                run_defs[module] = times
        if run_defs:
            deflections[run_id] = run_defs
    return deflections

def detect_bursts(samples: List[QueueSample], base_threshold: float, slope_threshold: float, cooldown: float) -> List[Tuple[float, float]]:
    bursts: List[Tuple[float, float]] = []
    if len(samples) < 2:
        return bursts
    in_burst = False
    start = None
    cooldown_start = None
    prev = samples[0]
    for curr in samples[1:]:
        dt = curr.time - prev.time
        if dt <= 0:
            prev = curr
            continue
        slope = (curr.util - prev.util) / dt
        if not in_burst:
            trigger = False
            if prev.util < base_threshold <= curr.util:
                trigger = True
                frac = (base_threshold - prev.util) / max(curr.util - prev.util, 1e-12)
                start = prev.time + frac * dt
            elif slope >= slope_threshold:
                trigger = True
                start = prev.time
            if trigger:
                in_burst = True
                cooldown_start = None
        else:
            if curr.util < base_threshold and slope <= 0.0:
                if cooldown_start is None:
                    cooldown_start = curr.time
                elif curr.time - cooldown_start >= cooldown:
                    bursts.append((start, curr.time))
                    in_burst = False
                    start = None
                    cooldown_start = None
            else:
                cooldown_start = None
        prev = curr
    if in_burst and start is not None:
        bursts.append((start, samples[-1].time))
    return bursts

def sliding_counts(times: List[float], window: float) -> List[int]:
    counts: List[int] = []
    head = 0
    for tail, timestamp in enumerate(times):
        while timestamp - times[head] > window:
            head += 1
        counts.append(tail - head + 1)
    return counts

def summarize(values: List[float], quantiles: Iterable[float]) -> Dict[float, float]:
    if not values:
        return {q: 0.0 for q in quantiles}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    summary = {}
    for q in quantiles:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        summary[q] = sorted_vals[idx]
    return summary

def plot_bursts(bursts_local: List[Burst], samples: List[QueueSample], 
                run_id: str, module: str, plot_dir: pathlib.Path, base_threshold: float) -> None:
    times = [sample.time for sample in samples]
    utils = [sample.util for sample in samples]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, utils, linewidth=1, color="steelblue", label="Utilization")

    for burst in bursts_local:
        ax.axvspan(burst.start, burst.end, color="orangered", alpha=0.25)

    ax.axhline(base_threshold, linestyle="--", color="gray", linewidth=0.8)

    ax.set_title(f"{run_id} — {module}")
    ax.set_ylabel("Queue Utilization")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.3, alpha=0.7)

    module_id = module.replace(".", "_").replace("[", "_").replace("]", "_")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{run_id}__{module_id}.png", dpi=150)
    plt.close(fig)

    print("    Plot saved to:", plot_dir / f"{run_id}__{module_id}.png")


def analyze_runs(queue_series: Dict[str, Dict[str, List[QueueSample]]], deflections: Dict[str, Dict[str, List[float]]], base_threshold: float, slope_threshold: float, cooldown: float, burst_window: float, 
                 rate_windows: List[float], quantiles: Iterable[float], plot_dir: Optional[pathlib.Path]) -> None:
    run_ids = sorted(set(queue_series.keys()) | set(deflections.keys()))
    if not run_ids:
        print("No runs found.")
        return
    for run_id in run_ids:
        modules = queue_series.get(run_id, {})
        run_deflections = deflections.get(run_id, {})
        print(f"\n=== Run {run_id} ===")
        if not modules:
            print("No queue data available.")
            continue
        total_deflections = sum(len(run_deflections.get(module, [])) for module in modules)
        print(f"Modules analyzed: {len(modules)}")
        print(f"Total deflection attempts observed: {total_deflections}")
        for module in sorted(modules.keys()):
            samples = modules[module]
            intervals = detect_bursts(samples, base_threshold, slope_threshold, cooldown)
            if not intervals:
                print(f"\n  Module {module}\n  No bursts detected.")
                continue
            port_defs = run_deflections.get(module, [])
            bursts_local: List[Burst] = []
            burst_window_counts: List[int] = []
            window_rates: Dict[float, List[float]] = {window: [] for window in rate_windows}
            for start, end in intervals:
                left = bisect_left(port_defs, start) if port_defs else 0
                right = bisect_right(port_defs, end) if port_defs else 0
                count = max(right - left, 0)
                duration = max(end - start, 1e-12)
                bursts_local.append(Burst(start=start, end=end, duration=duration, deflections=count))
                if count:
                    burst_window_counts.extend(sliding_counts(port_defs[left:right], burst_window))
            if port_defs:
                for window in rate_windows:
                    window_counts = sliding_counts(port_defs, window)
                    window_rates[window].extend(c / window for c in window_counts)
            print(f"\n  Module {module}")
            print(f"  Deflection attempts observed: {len(port_defs)}")
            print(f"  Bursts detected: {len(bursts_local)}")
            burst_lengths = [b.duration for b in bursts_local]
            if plot_dir:
                plot_bursts(bursts_local, samples, run_id, module, plot_dir, base_threshold)
            length_stats = summarize(burst_lengths, quantiles)
            print("    Burst duration quantiles (seconds):")
            for q in quantiles:
                print(f"      q={q:.2f}: {length_stats[q]:.6f} s")
            burst_counts = [b.deflections for b in bursts_local]
            count_stats = summarize(burst_counts, quantiles)
            print("    Deflection attempts per burst (A_{τ_b} empirical):")
            for q in quantiles:
                print(f"      q={q:.2f}: {count_stats[q]:.2f} attempts")
            suggested_B = {q: int(ceil(count_stats[q])) for q in quantiles}
            print(f"    Suggested B (ceil of quantiles): {suggested_B}")
            burst_rates = [
                (b.deflections / b.duration) if b.deflections and b.duration > 0 else 0.0
                for b in bursts_local
            ]
            rate_stats = summarize(burst_rates, quantiles)
            print("    Deflection rate within bursts (attempts per second):")
            for q in quantiles:
                print(f"      q={q:.2f}: {rate_stats[q]:.2f} attempts/s")
            for window in rate_windows:
                stats = summarize(window_rates.get(window, []), quantiles)
                print(f"    D_W quantiles for window {window*1e3:.0f} ms (attempts per second):")
                for q in quantiles:
                    print(f"      q={q:.2f}: {stats[q]:.2f}")
                suggested_r = {q: round(stats[q], 2) for q in quantiles}
                print(f"    Suggested r (rounded): {suggested_r}")
            if burst_window_counts:
                burst_window_stats = summarize([c / burst_window for c in burst_window_counts], quantiles)
                print(f"    Sliding burst-window rate quantiles:")
                for q in quantiles:
                    print(f"      q={q:.2f}: {burst_window_stats[q]:.2f} attempts/s")

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze queue bursts from scavetool CSV exports.")
    parser.add_argument("--results-dir", required=True, type=pathlib.Path,
                        help="Policy directory (e.g., results_1G_threshold_tb)")
    parser.add_argument("--queue-len-dir", type=pathlib.Path,
                        help="Queue length CSV directory (default: <results-dir>/QUEUE_LEN)")
    parser.add_argument("--queue-cap-dir", type=pathlib.Path,
                        help="Queue capacity CSV directory (default: <results-dir>/QUEUE_CAPACITY)")
    parser.add_argument("--packet-action-dir", type=pathlib.Path,
                        help="Packet action CSV directory (default: <results-dir>/PACKET_ACTION)")
    parser.add_argument("--base-threshold", type=float, default=0.55)
    parser.add_argument("--slope-threshold", type=float, default=2e5)
    parser.add_argument("--cooldown-us", type=float, default=30.0)
    parser.add_argument("--burst-window-us", type=float, default=100.0)
    parser.add_argument("--rate-windows-ms", nargs="+", type=float, default=[1.0, 5.0])
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.5, 0.9, 0.99])
    parser.add_argument("--plot", action="store_true",
                    help="Generate burst plots for each module.")
    parser.add_argument("--plot-dir", type=pathlib.Path, default=None,
                        help="Where to save plots (default: <results-dir>/plots).")

    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir.is_absolute() else pathlib.Path.cwd() / args.results_dir
    queue_len_dir = args.queue_len_dir or (results_dir / "QUEUE_LEN")
    queue_cap_dir = args.queue_cap_dir or (results_dir / "QUEUE_CAPACITY")
    packet_action_dir = args.packet_action_dir or (results_dir / "PACKET_ACTION")

    if args.plot:
        plot_dir = args.plot_dir or (results_dir / "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

    queue_series = collect_queue_series(queue_len_dir, queue_cap_dir)
    if not queue_series:
        print("No queue data found in CSV directories.")
        return
    deflections = collect_deflections(packet_action_dir)

    cooldown = args.cooldown_us * 1e-6
    burst_window = args.burst_window_us * 1e-6
    rate_windows = [ms * 1e-3 for ms in args.rate_windows_ms]

    analyze_runs(
        queue_series,
        deflections,
        args.base_threshold,
        args.slope_threshold,
        cooldown,
        burst_window,
        rate_windows,
        args.quantiles,
        plot_dir if args.plot else None
    )


if __name__ == "__main__":
    main()
