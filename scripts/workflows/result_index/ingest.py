"""Build a SQLite experiment result index from runs reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.workflows.result_index.dashboard_export import (
    DEFAULT_DB_PATH,
    write_dashboard_bundle,
)
from scripts.workflows.result_index.report_loader import (
    discover_candidate_report_paths,
    discover_report_paths,
    load_result_index_records,
)
from scripts.workflows.result_index.sqlite_store import (
    clear_database,
    write_result_index_records,
)

DEFAULT_RUNS_ROOT = Path("runs")


def ingest_reports(
    *,
    runs_root: Path,
    db_path: Path,
    reset: bool = False,
) -> int:
    report_paths = discover_report_paths(runs_root)
    records = [load_result_index_records(path) for path in report_paths]
    if reset:
        clear_database(db_path)
    write_result_index_records(db_path=db_path, records=records)
    return len(records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest experiment report.json files into a SQLite result index.",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing indexed rows before ingesting this runs root.",
    )
    parser.add_argument(
        "--dashboard-json",
        type=Path,
        default=None,
        help="Optionally export a static dashboard JSON bundle after ingest.",
    )
    args = parser.parse_args(argv)

    indexed_count = ingest_reports(
        runs_root=args.runs_root,
        db_path=args.db,
        reset=args.reset,
    )
    candidate_count = len(discover_candidate_report_paths(args.runs_root))
    excluded_count = max(candidate_count - indexed_count, 0)
    output_parts = [
        f"db={args.db}",
        f"candidate_reports={candidate_count}",
        f"indexed_runs={indexed_count}",
        f"excluded_reports={excluded_count}",
    ]
    if args.dashboard_json is not None:
        bundle = write_dashboard_bundle(
            db_path=args.db,
            output_path=args.dashboard_json,
        )
        output_parts.append(f"dashboard_json={args.dashboard_json}")
        output_parts.append(f"dashboard_runs={len(bundle['runs'])}")
        output_parts.append(
            f"dashboard_algorithms={len(bundle['filters']['algorithms'])}"
        )
    print(" ".join(output_parts), flush=True)


if __name__ == "__main__":
    main()
