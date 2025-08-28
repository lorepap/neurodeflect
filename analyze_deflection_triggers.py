#!/usr/bin/env python3
"""
Analyze deflection triggers by parsing simulation .out logs for
explicit DEFLECTION_CAUSE markers added in the relay.

Looks in:
  Omnet_Sims/dc_simulations/simulations/sims/logs/deflection_threshold_*/

Reports per-threshold counts for:
  - THRESHOLD
  - OVERFLOW (generic)
  - OVERFLOW_METHOD=RANDOM | NAIVE | SAME_PATH | V2

Usage:
  python3 analyze_deflection_triggers.py
Optionally override logs root:
  python3 analyze_deflection_triggers.py --logs <path>
"""

import argparse
import os
import re
from collections import defaultdict, Counter


def find_threshold_dirs(root: str):
	dirs = []
	if not os.path.isdir(root):
		return dirs
	for name in os.listdir(root):
		if name.startswith("deflection_threshold_"):
			full = os.path.join(root, name)
			if os.path.isdir(full):
				try:
					thr = int(name.split("_")[-1])
				except ValueError:
					continue
				dirs.append((thr, full))
	return sorted(dirs)


def parse_out_file(path: str):
	"""Parse a single .out log and count deflection cause markers."""
	counts = Counter()
	# Patterns:
	# DEFLECTION_CAUSE=THRESHOLD ...
	# DEFLECTION_CAUSE=OVERFLOW ...
	# DEFLECTION_CAUSE=OVERFLOW_METHOD=RANDOM|NAIVE|SAME_PATH|V2 ...
	pat_threshold = re.compile(r"DEFLECTION_CAUSE=THRESHOLD")
	pat_overflow = re.compile(r"DEFLECTION_CAUSE=OVERFLOW(?!_METHOD)")
	pat_overflow_method = re.compile(r"DEFLECTION_CAUSE=OVERFLOW_METHOD=([A-Z_0-9]+)")

	try:
		with open(path, "r", errors="ignore") as f:
			for line in f:
				if pat_threshold.search(line):
					counts["THRESHOLD"] += 1
				m = pat_overflow_method.search(line)
				if m:
					counts[f"OVERFLOW_METHOD={m.group(1)}"] += 1
				elif pat_overflow.search(line):
					counts["OVERFLOW"] += 1
	except FileNotFoundError:
		pass
	return counts


def analyze_threshold_dir(threshold: int, dir_path: str):
	counts = Counter()
	out_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.out')]
	for of in out_files:
		counts += parse_out_file(of)
	total_overflow = counts.get("OVERFLOW", 0) \
		+ counts.get("OVERFLOW_METHOD=RANDOM", 0) \
		+ counts.get("OVERFLOW_METHOD=NAIVE", 0) \
		+ counts.get("OVERFLOW_METHOD=SAME_PATH", 0) \
		+ counts.get("OVERFLOW_METHOD=V2", 0)
	return {
		"threshold": threshold,
		"threshold_deflections": counts.get("THRESHOLD", 0),
		"overflow_deflections": total_overflow,
		"breakdown": {
			"OVERFLOW": counts.get("OVERFLOW", 0),
			"RANDOM": counts.get("OVERFLOW_METHOD=RANDOM", 0),
			"NAIVE": counts.get("OVERFLOW_METHOD=NAIVE", 0),
			"SAME_PATH": counts.get("OVERFLOW_METHOD=SAME_PATH", 0),
			"V2": counts.get("OVERFLOW_METHOD=V2", 0),
		}
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--logs", default=os.path.join(
		os.path.dirname(__file__),
		"Omnet_Sims",
		"dc_simulations",
		"simulations",
		"sims",
		"logs",
	), help="Path to logs directory containing deflection_threshold_* folders")
	args = parser.parse_args()

	roots = find_threshold_dirs(args.logs)
	if not roots:
		print(f"No threshold logs found under {args.logs}")
		return

	print("DEFLECTION TRIGGERS SUMMARY")
	print("=" * 60)
	rows = []
	for thr, path in roots:
		res = analyze_threshold_dir(thr, path)
		rows.append(res)
		print(f"Threshold {thr:>6}: THRESHOLD={res['threshold_deflections']:,}  "
			  f"OVERFLOW={res['overflow_deflections']:,}  "
			  f"[breakdown: gen={res['breakdown']['OVERFLOW']:,}, rand={res['breakdown']['RANDOM']:,}, "
			  f"naive={res['breakdown']['NAIVE']:,}, same={res['breakdown']['SAME_PATH']:,}, v2={res['breakdown']['V2']:,}]")

	print("-" * 60)
	# Simple check: does threshold deflections go down as threshold grows?
	monotonic = all(rows[i]["threshold_deflections"] >= rows[i+1]["threshold_deflections"] for i in range(len(rows)-1))
	trend = "monotonic non-increasing" if monotonic else "not monotonic"
	print(f"Trend of THRESHOLD deflections vs threshold: {trend}")


if __name__ == "__main__":
	main()

