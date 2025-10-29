#!/usr/bin/env python3
"""
Utility to inspect the first non-null value stored in a simulation distribution
SQLite database. These DB files back the per-server burst inter-arrival times.

Example:
    python3 extract_first_burst.py \\
        distributions/cache_bursty_inter_arrival_time_db_1_flowmult_0.11_intermult_40_flows_per_incast_20000_incast_flow_size_0.db \\
        --column server0app2

By default the script auto-detects the table name (most files use
`inter_arrival`). Use --table to override it explicitly.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from typing import List


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cur.fetchall()]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract the first non-null entry for a given column from a distribution DB.")
    parser.add_argument(
        "db_path",
        type=str,
        help="Path to the .db file (e.g. distributions/cache_bursty_inter_arrival_time_db_... .db)",
    )
    parser.add_argument(
        "--column",
        required=True,
        help="Column name to inspect (e.g. server0app2)",
    )
    parser.add_argument(
        "--table",
        help="Optional table name; if omitted the script auto-detects the first table in the DB.",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="Only list tables in the database and exit.",
    )
    args = parser.parse_args(argv)

    try:
        conn = sqlite3.connect(args.db_path)
    except sqlite3.Error as exc:
        print(f"[error] failed to open database: {exc}", file=sys.stderr)
        return 1

    with conn:
        tables = list_tables(conn)
        if not tables:
            print("[error] no tables found in database (is this a valid SQLite file?)", file=sys.stderr)
            return 1

        if args.list_tables:
            print("Tables:")
            for name in tables:
                print(f"  {name}")
            return 0

        table_name = args.table or tables[0]
        if table_name not in tables:
            print(f"[error] table '{table_name}' not present in database. Available: {', '.join(tables)}", file=sys.stderr)
            return 1

        query = (
            f"SELECT {args.column} FROM {table_name} "
            f"WHERE {args.column} IS NOT NULL AND {args.column} <> '' "
            "ORDER BY rowid LIMIT 1"
        )

        cur = conn.cursor()
        try:
            cur.execute(query)
        except sqlite3.Error as exc:
            print(f"[error] query failed: {exc}", file=sys.stderr)
            return 1

        row = cur.fetchone()
        if row is None:
            print(f"[warn] no non-null values found for column '{args.column}' in table '{table_name}'", file=sys.stderr)
            return 2

        value = row[0]
        print(value)

    return 0


if __name__ == "__main__":
    sys.exit(main())
