#!/usr/bin/env python3
"""Dataset builder for per-switch RL training with flow context."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass
import re
from typing import Dict, Iterable, Iterator, List, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

csv.field_size_limit(sys.maxsize)

@dataclass
class VectorSeries:
    run: str
    module: str
    name: str
    times: List[float]
    values: List[float]

    def pair_iter(self) -> Iterator[Tuple[float, float]]:
        return zip(self.times, self.values)

    def to_frame(self, column: str) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": self.times, column: self.values})


VECTOR_COLUMN_MAP = {
    "QueueLen:vector": "queue_len",
    "QueuesTotLen:vector": "queues_tot_len",
    "QueueCapacity:vector": "queue_capacity",
    "QueuesTotCapacity:vector": "queues_tot_capacity",
    "switchSeqNum:vector": "seq_num",
    "switchTtl:vector": "ttl",
    "PacketSize:vector": "packet_size",
    "PacketAction:vector": "action",
    "RequesterID:vector": "RequesterID",
    "FlowID:vector": "FlowID",
}

FLOW_START_VECTOR = "flowStartedRequesterID:vector"
FLOW_START_REQ_COL = "flowStartedRequesterID"
FLOW_END_VECTOR = "flowEndedRequesterID:vector"
FLOW_END_REQ_COL = "flowEndedRequesterID"
FLOW_QUERY_VECTOR = "flowEndedQueryID:vector"
REQUEST_SENT_VECTOR = "requestSentRequesterID:vector"

BURSTY_APP_MIN_INDEX = 2
APP_INDEX_PATTERN = re.compile(r"\.app\[(\d+)\]")

BASE_VECTOR_PRIORITY = [
    "PacketAction:vector",
    "PacketSize:vector",
    "QueueLen:vector",
]



def build_flow_event_df(vectors: List[VectorSeries]) -> pd.DataFrame:
    rows: List[Tuple[float, float]] = []
    for vec in vectors:
        rows.extend(vec.pair_iter())
    if not rows:
        return pd.DataFrame(columns=["timestamp", "RequesterID"])
    df = pd.DataFrame(rows, columns=["timestamp", "RequesterID"])
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df

def assign_flow_event_times(df: pd.DataFrame, events: pd.DataFrame, forward: bool) -> pd.Series:
    """Assign nearest flow event time for each packet row."""
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if events is None or events.empty or "RequesterID" not in events.columns:
        return result
    events_sorted = events.sort_values(["RequesterID", "timestamp"]).dropna(subset=["RequesterID"])
    events_sorted["RequesterID"] = pd.to_numeric(events_sorted["RequesterID"], errors="coerce")
    events_sorted = events_sorted.dropna(subset=["RequesterID"]).astype({"RequesterID": float})
    if events_sorted.empty:
        return result
    for requester_id, group in events_sorted.groupby("RequesterID"):
        mask = df["RequesterID"].to_numpy() == requester_id
        if not mask.any():
            continue
        packet_times = df.loc[mask, "timestamp"].to_numpy()
        event_times = group["timestamp"].to_numpy()
        assigned = np.full(packet_times.shape, np.nan)
        if forward:
            pos = np.searchsorted(event_times, packet_times, side="left")
            valid = pos < len(event_times)
            assigned[valid] = event_times[pos[valid]]
        else:
            pos = np.searchsorted(event_times, packet_times, side="right") - 1
            valid = pos >= 0
            assigned[valid] = event_times[pos[valid]]
        result.iloc[np.where(mask)[0]] = assigned
    return result


def build_sequence_df(vectors: list[VectorSeries]) -> pd.DataFrame:
    rows: list[tuple[float, float]] = []
    for vec in vectors:
        rows.extend(vec.pair_iter())
    if not rows:
        return pd.DataFrame(columns=["timestamp", "seq_num"])
    df = pd.DataFrame(rows, columns=["timestamp", "seq_num"])
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")
    df = df.dropna(subset=["seq_num"])
    df["seq_num"] = df["seq_num"].astype("int64", copy=False)
    return df


def build_receiver_ooo_set(vectors: list[VectorSeries]) -> set[int]:
    if not vectors:
        return set()
    seqs: set[int] = set()
    for vec in vectors:
        seqs.update(int(val) for val in vec.values if pd.notna(val))
    return seqs


def build_requester_query_table(
    request_vectors: List[VectorSeries],
    query_vectors: List[VectorSeries],
    flow_start_df: pd.DataFrame | None,
    flow_end_df: pd.DataFrame | None,
) -> pd.DataFrame:
    empty_cols = ["RequesterID", "query_id", "query_start_time", "query_end_time", "QCT"]
    if not request_vectors or not query_vectors:
        return pd.DataFrame(columns=empty_cols)

    requests_by_module: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for vec in request_vectors:
        requests_by_module[vec.module].extend(vec.pair_iter())

    queries_by_module: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for vec in query_vectors:
        queries_by_module[vec.module].extend(vec.pair_iter())

    rows: List[Dict[str, float]] = []
    for module, req_pairs in requests_by_module.items():
        query_pairs = queries_by_module.get(module)
        if not query_pairs:
            continue
        match = APP_INDEX_PATTERN.search(module)
        if not match:
            continue
        app_idx = int(match.group(1))
        if app_idx < BURSTY_APP_MIN_INDEX:
            continue
        req_pairs = sorted(req_pairs, key=lambda x: x[0])
        query_pairs = sorted(query_pairs, key=lambda x: x[0])
        n = min(len(req_pairs), len(query_pairs))
        if n == 0:
            continue
        if len(req_pairs) != len(query_pairs):
            print(f"[builder] warning: query vector length mismatch in {module}: requests={len(req_pairs)} queries={len(query_pairs)}")
        for idx in range(n):
            req_time, requester_id = req_pairs[idx]
            _, query_id = query_pairs[idx]
            rows.append(
                {
                    "RequesterID": requester_id,
                    "query_id": query_id,
                    "request_time": req_time,
                    "app_index": app_idx,
                }
            )

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    mapping_df = pd.DataFrame(rows)
    mapping_df["RequesterID"] = pd.to_numeric(mapping_df["RequesterID"], errors="coerce")
    mapping_df["query_id"] = pd.to_numeric(mapping_df["query_id"], errors="coerce")
    mapping_df = mapping_df.dropna(subset=["RequesterID", "query_id"])
    if mapping_df.empty:
        return pd.DataFrame(columns=empty_cols)

    if flow_start_df is not None and not flow_start_df.empty:
        start_df = flow_start_df.rename(columns={"timestamp": "flow_start_time"})
        mapping_df = mapping_df.merge(start_df, on="RequesterID", how="left")
    else:
        mapping_df["flow_start_time"] = np.nan

    if flow_end_df is not None and not flow_end_df.empty:
        end_df = flow_end_df.rename(columns={"timestamp": "flow_end_time"})
        mapping_df = mapping_df.merge(end_df, on="RequesterID", how="left")
    else:
        mapping_df["flow_end_time"] = np.nan

    if mapping_df["query_id"].notna().any():
        query_rows = mapping_df.dropna(subset=["query_id"]).copy()
        query_rows["request_time"] = pd.to_numeric(query_rows["request_time"], errors="coerce")
        query_rows["flow_end_time"] = pd.to_numeric(query_rows["flow_end_time"], errors="coerce")
        grouped = (
            query_rows.groupby("query_id")
            .agg(
                query_start_time=("request_time", "min"),
                query_end_time=("flow_end_time", "max"),
            )
            .reset_index()
        )
        grouped["QCT"] = grouped["query_end_time"] - grouped["query_start_time"]
        mapping_df = mapping_df.merge(grouped, on="query_id", how="left")
    else:
        mapping_df["query_start_time"] = np.nan
        mapping_df["query_end_time"] = np.nan
        mapping_df["QCT"] = np.nan

    mapping_df = mapping_df.drop(columns=["request_time", "flow_start_time", "flow_end_time"], errors="ignore")
    mapping_df = mapping_df.drop(columns=["app_index"], errors="ignore")
    return mapping_df

def parse_vector_row(row: List[str]) -> VectorSeries | None:
    try:
        run_id, row_type, module, name = row[0], row[1], row[2], row[3]
        if row_type != "vector":
            return None
        times_str = row[7].strip()
        values_str = row[8].strip()
        if not times_str or not values_str:
            return None
        times = [float(tok) for tok in times_str.split()]
        values = [float(tok) for tok in values_str.split()]
        if len(times) != len(values):
            n = min(len(times), len(values))
            times = times[:n]
            values = values[:n]
        return VectorSeries(run=run_id, module=module, name=name, times=times, values=values)
    except (IndexError, ValueError):
        return None


def load_csvr_directory(directory: pathlib.Path, name_filter: str | None = None) -> List[VectorSeries]:
    series: List[VectorSeries] = []
    for csv_path in sorted(directory.glob("*.csv")):
        try:
            with csv_path.open(newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    vec = parse_vector_row(row)
                    if vec is None:
                        continue
                    if name_filter and vec.name != name_filter:
                        continue
                    series.append(vec)
        except FileNotFoundError:
            continue
    return series


def discover_runs(series: Iterable[VectorSeries]) -> Dict[str, List[VectorSeries]]:
    runs: Dict[str, List[VectorSeries]] = {}
    for vec in series:
        runs.setdefault(vec.run, []).append(vec)
    return runs


def module_is_relay(module_path: str) -> bool:
    return ".relayUnit" in module_path


def module_identifier(module_path: str) -> str:
    safe = module_path.replace('.', '__').replace('[', '_').replace(']', '')
    return safe


def build_dataframe(series_map: Dict[str, VectorSeries], flow_starts: pd.DataFrame | None = None, flow_ends: pd.DataFrame | None = None, requester_query_df: pd.DataFrame | None = None, switch_seq_df: pd.DataFrame | None = None, send_seq_df: pd.DataFrame | None = None, receiver_ooo_set: set[int] | None = None) -> pd.DataFrame | None:
    base_series: VectorSeries | None = None
    for name in BASE_VECTOR_PRIORITY:
        if name in series_map:
            base_series = series_map[name]
            break
    if base_series is None:
        return None

    base_col = VECTOR_COLUMN_MAP.get(base_series.name, "value")  
    df = base_series.to_frame(base_col).sort_values("timestamp").reset_index(drop=True)

    for name, column in VECTOR_COLUMN_MAP.items():
        if name == base_series.name:
            continue
        vec = series_map.get(name)
        if vec is None:
            continue
        feature_df = vec.to_frame(column).sort_values("timestamp")
        df = pd.merge_asof(
            df,
            feature_df,
            on="timestamp",
            direction="nearest",
            tolerance=1e-9,
        )

    if switch_seq_df is not None and not switch_seq_df.empty:
        seq_df = switch_seq_df.sort_values("timestamp")
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            seq_df,
            on="timestamp",
            direction="nearest",
            tolerance=1e-9,
        )

    # Compute queue utilizations and drop raw capacities/lengths
    if "queue_len" in df.columns and "queue_capacity" in df.columns:
        denom = pd.to_numeric(df["queue_capacity"], errors="coerce")
        numer = pd.to_numeric(df["queue_len"], errors="coerce")
        df["queue_util"] = numer / denom.replace({0: pd.NA})
        df = df.drop(columns=["queue_len", "queue_capacity"])
    if "queues_tot_len" in df.columns and "queues_tot_capacity" in df.columns:
        denom = pd.to_numeric(df["queues_tot_capacity"], errors="coerce")
        numer = pd.to_numeric(df["queues_tot_len"], errors="coerce")
        df["queues_tot_util"] = numer / denom.replace({0: pd.NA})
        df = df.drop(columns=["queues_tot_len", "queues_tot_capacity"])

    # Flow start/end association from run-level events
    if "RequesterID" in df.columns:
        df["RequesterID"] = pd.to_numeric(df["RequesterID"], errors="coerce")
        df["flow_start_time"] = assign_flow_event_times(df, flow_starts, forward=False)
        df["flow_end_time"] = assign_flow_event_times(df, flow_ends, forward=True)

    if requester_query_df is not None and not requester_query_df.empty and "RequesterID" in df.columns:
        merge_cols = [c for c in ["RequesterID", "query_id", "query_start_time", "query_end_time", "QCT"] if c in requester_query_df.columns]
        if merge_cols:
            df = df.merge(requester_query_df[merge_cols].drop_duplicates(subset=["RequesterID"]), on="RequesterID", how="left")

    if "flow_start_time" in df.columns and "flow_end_time" in df.columns:
        df["FCT"] = df["flow_end_time"] - df["flow_start_time"]
    elif "flow_start_time" in df.columns and "FCT" not in df.columns:
        df["FCT"] = pd.NA

    # if "FlowID" in df.columns and df["FlowID"].notna().any():
    #     df["packet_position"] = df.groupby("FlowID", dropna=False).cumcount() + 1
    # elif "RequesterID" in df.columns:
    #     df["packet_position"] = df.groupby("RequesterID", dropna=False).cumcount() + 1

    if "seq_num" in df.columns:
        df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")
        df["packet_latency"] = pd.NA
        if send_seq_df is not None and not send_seq_df.empty:
            seq_map_df = send_seq_df.dropna().drop_duplicates("seq_num", keep="first")
            seq_map_df["seq_num"] = pd.to_numeric(seq_map_df["seq_num"], errors="coerce")
            seq_map_df = seq_map_df.dropna(subset=["seq_num"])
            if not seq_map_df.empty:
                seq_map_df["seq_num"] = seq_map_df["seq_num"].astype("int64", copy=False)
                send_map = dict(zip(seq_map_df["seq_num"], seq_map_df["timestamp"]))
                valid_seq = df["seq_num"].notna()
                if valid_seq.any():
                    seq_values = df.loc[valid_seq, "seq_num"].astype("int64", copy=False)
                    send_times = seq_values.map(send_map)
                    df.loc[valid_seq, "packet_latency"] = df.loc[valid_seq, "timestamp"] - send_times
                    df["packet_latency"] = pd.to_numeric(df["packet_latency"], errors="coerce")
        if receiver_ooo_set:
            df["ooo"] = 0
            mask = df["seq_num"].notna()
            if mask.any():
                seq_values = df.loc[mask, "seq_num"].astype("int64", copy=False)
                df.loc[mask, "ooo"] = seq_values.isin(receiver_ooo_set).astype(int)
        else:
            df["ooo"] = 0
    else:
        df["packet_latency"] = pd.NA
        df["ooo"] = 0


    return df


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-switch RL datasets from CSV-R exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", required=True, type=pathlib.Path,
                        help="Root directory containing per-run CSV-R exports.")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path,
                        help="Destination directory for per-switch datasets.")
    parser.add_argument("--runs", nargs="*",
                        help="Optional subset of run IDs to process (defaults to all runs found).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List detected runs and switches without writing output.")
    parser.add_argument("--vector", action="append",
                        help="Optional specific vector names to inspect during dry-run.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists():
        print(f"[builder] results directory not found: {results_dir}", file=sys.stderr)
        return 1

    print(f"[builder] results-dir: {results_dir}")
    print(f"[builder] output-dir : {output_dir}")
    print(f"[builder] dry-run    : {args.dry_run}")

    subdirs = [p for p in results_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print("[builder] no metric directories found (QUEUE_LEN, etc.)", file=sys.stderr)
        return 1

    vectors_by_subdir: Dict[pathlib.Path, List[VectorSeries]] = {}
    for subdir in subdirs:
        names = args.vector or []
        if names:
            aggregate: List[VectorSeries] = []
            for name in names:
                aggregate.extend(load_csvr_directory(subdir, name_filter=name))
        else:
            aggregate = load_csvr_directory(subdir)
        if aggregate:
            vectors_by_subdir[subdir] = aggregate

    if not vectors_by_subdir:
        print("[builder] no vectors parsed (check CSV format).", file=sys.stderr)
        return 1

    run_flow_start_vectors = defaultdict(list)
    run_flow_end_vectors = defaultdict(list)
    run_switch_seq_vectors = defaultdict(list)
    run_send_seq_vectors = defaultdict(list)
    run_receiver_ooo_vectors = defaultdict(list)
    run_request_sent_vectors = defaultdict(list)
    run_query_id_vectors = defaultdict(list)
    for vectors in vectors_by_subdir.values():
        for vec in vectors:
            if vec.name == FLOW_START_VECTOR:
                run_flow_start_vectors[vec.run].append(vec)
            elif vec.name == FLOW_END_VECTOR:
                run_flow_end_vectors[vec.run].append(vec)
            elif vec.name == "switchSeqNum:vector":
                run_switch_seq_vectors[vec.run].append(vec)
            elif vec.name == "sndNxt:vector":
                run_send_seq_vectors[vec.run].append(vec)
            elif vec.name == "rcvOooSeg:vector":
                run_receiver_ooo_vectors[vec.run].append(vec)
            elif vec.name == REQUEST_SENT_VECTOR:
                run_request_sent_vectors[vec.run].append(vec)
            elif vec.name == FLOW_QUERY_VECTOR:
                run_query_id_vectors[vec.run].append(vec)

    run_flow_start_df = {run: build_flow_event_df(vecs) for run, vecs in run_flow_start_vectors.items()}
    run_flow_end_df = {run: build_flow_event_df(vecs) for run, vecs in run_flow_end_vectors.items()}
    run_switch_seq_df = {run: build_sequence_df(vecs) for run, vecs in run_switch_seq_vectors.items()}
    run_send_seq_df = {run: build_sequence_df(vecs) for run, vecs in run_send_seq_vectors.items()}
    run_receiver_ooo_set = {run: build_receiver_ooo_set(vecs) for run, vecs in run_receiver_ooo_vectors.items()}

    if args.dry_run:
        print("[builder] Dry-run summary:")
        for subdir, vectors in vectors_by_subdir.items():
            runs = discover_runs(vectors)
            run_list = sorted(runs.keys())
            print(f"  - {subdir.name}: {len(vectors)} vectors, runs={len(run_list)}")
            for run_id in run_list[:5]:
                modules = {vec.module for vec in runs[run_id]}
                sample_names = {vec.name for vec in runs[run_id]}
                print(f"      run {run_id}: modules={len(modules)}, names={sorted(sample_names)[:5]}")
        return 0

    run_module_map: Dict[str, Dict[str, Dict[str, VectorSeries]]] = {}
    for vectors in vectors_by_subdir.values():
        for vec in vectors:
            run_entry = run_module_map.setdefault(vec.run, {})
            module_entry = run_entry.setdefault(vec.module, {})
            module_entry.setdefault(vec.name, vec)

    run_requester_query_df: Dict[str, pd.DataFrame] = {}
    for run_id in run_module_map.keys():
        rq_df = build_requester_query_table(
            run_request_sent_vectors.get(run_id, []),
            run_query_id_vectors.get(run_id, []),
            run_flow_start_df.get(run_id),
            run_flow_end_df.get(run_id),
        )
        run_requester_query_df[run_id] = rq_df

    selected_runs = args.runs if args.runs else sorted(run_module_map.keys())
    if not selected_runs:
        print("[builder] no runs selected.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for run_id in selected_runs:
        module_map = run_module_map.get(run_id)
        if not module_map:
            print(f"[builder] warning: run {run_id} not found in parsed data")
            continue
        for module_path, series_map in module_map.items():
            if not module_is_relay(module_path):
                continue
            flow_start_df = run_flow_start_df.get(run_id)
            flow_end_df = run_flow_end_df.get(run_id)
            switch_seq_df = run_switch_seq_df.get(run_id)
            send_seq_df = run_send_seq_df.get(run_id)
            receiver_ooo_set = run_receiver_ooo_set.get(run_id)
            requester_query_df = run_requester_query_df.get(run_id)
            df = build_dataframe(series_map, flow_start_df, flow_end_df, requester_query_df, switch_seq_df, send_seq_df, receiver_ooo_set)
            if df is None or df.empty:
                continue
            df = df.sort_values("timestamp").reset_index(drop=True)
            df.insert(0, "run", run_id)
            df.insert(1, "module", module_path)
            df.insert(2, "switch_id", module_identifier(module_path))
            output_path = output_dir / f"{run_id}__{module_identifier(module_path)}.csv"
            df.to_csv(output_path, index=False)
            written += 1
            print(f"[builder] wrote {output_path.relative_to(output_dir)} rows={len(df)}")

    print(f"[builder] completed writing {written} module files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
