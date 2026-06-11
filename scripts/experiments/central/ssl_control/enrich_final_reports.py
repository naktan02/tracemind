"""중앙 SSL report의 final 항목을 보강/재작성하는 스크립트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.support.query_ssl_text_encoder.result_utils import (
    extract_final_selection_report,
    merge_results_with_best_and_final,
)


def main() -> None:
    args = _parse_args()
    report_paths = _collect_report_paths(
        explicit_reports=tuple(args.report or []),
        run_dirs=tuple(args.run_dir or []),
        report_suffix=args.report_suffix,
    )
    if not report_paths:
        raise ValueError(
            "No report.json files found from --report or --run-dir inputs."
        )

    changed_count = 0
    no_change_count = 0
    for report_path in report_paths:
        changed = _enrich_single_report(
            report_path=report_path,
            dry_run=bool(args.dry_run),
        )
        if changed:
            changed_count += 1
        else:
            no_change_count += 1

    print(f"enriched={changed_count}")
    print(f"unchanged={no_change_count}")


def _collect_report_paths(
    *,
    explicit_reports: tuple[str, ...],
    run_dirs: tuple[str, ...],
    report_suffix: str,
) -> tuple[Path, ...]:
    """입력 경로에서 대상 report.json 목록을 수집한다."""

    report_paths: set[Path] = set()
    for explicit in explicit_reports:
        path = Path(explicit)
        if path.is_file():
            report_paths.add(path.resolve())
            continue
        if path.is_dir():
            candidate = path / report_suffix
            if candidate.is_file():
                report_paths.add(candidate.resolve())
            else:
                raise ValueError(f"run report not found in directory: {candidate}")

    for run_dir in run_dirs:
        path = Path(run_dir)
        if not path.is_dir():
            raise ValueError(f"run_dir must be a directory: {path}")
        for report_path in path.rglob(report_suffix):
            if report_path.is_file():
                report_paths.add(report_path.resolve())

    return tuple(sorted(report_paths))


def _enrich_single_report(
    *,
    report_path: Path,
    dry_run: bool,
) -> bool:
    """하나의 report.json을 읽어 final 정보를 보강한다."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report must be a JSON object: {report_path}")

    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        return False

    results = payload.get("results")
    if not isinstance(results, dict):
        results = {}

    history = manifest.get("history")
    if not isinstance(history, list):
        history = []

    final_selection_report = manifest.get("final_selection_report")
    if final_selection_report is None:
        final_selection_report = extract_final_selection_report(history)

    selection_set = str(manifest.get("selection_set", "test"))
    final_results = merge_results_with_best_and_final(
        results=results,
        selection_set=selection_set,
        final_selection_report=(
            final_selection_report if isinstance(final_selection_report, dict) else None
        ),
    )

    updated_manifest = dict(manifest)
    updated_results = dict(payload.get("results", {}))
    updated_payload = dict(payload)
    updated_manifest["final_selection_report"] = (
        dict(final_selection_report)
        if isinstance(final_selection_report, dict)
        else None
    )
    updated_results.update(final_results)
    updated_payload["manifest"] = updated_manifest
    updated_payload["results"] = updated_results

    if _payload_changed(payload=payload, updated_payload=updated_payload):
        if not dry_run:
            report_path.write_text(
                json.dumps(updated_payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        print(
            f"[updated] {report_path}"
            + (" (dry-run)" if dry_run else "")
        )
        return True

    print(f"[unchanged] {report_path}")
    return False


def _payload_changed(
    *,
    payload: dict[str, Any],
    updated_payload: dict[str, Any],
) -> bool:
    """주요 필드 기준 변경 여부를 판정한다."""

    return (
        payload.get("manifest", {}).get("final_selection_report")
        != updated_payload.get("manifest", {}).get("final_selection_report")
        or payload.get("results", {}) != updated_payload.get("results", {})
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Central SSL report 파일에 final 값을 보강한다."
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help=(
            "직접 지정한 report.json 경로를 추가합니다. run_dir보다 우선순위가 "
            "높습니다."
        ),
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help=(
            "디렉터리를 지정하면 하위에 있는 모든 */reports/report.json를 "
            "스캔합니다."
        ),
    )
    parser.add_argument(
        "--report-suffix",
        default="reports/report.json",
        help=(
            "run-dir scan 시 사용할 report 파일 상대 경로입니다. 기본값: "
            "reports/report.json"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 파일 쓰기 없이 변경 대상만 조회합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
