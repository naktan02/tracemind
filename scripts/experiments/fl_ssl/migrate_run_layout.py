"""Migrate existing FL SSL run directories to the current comparison layout."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scripts.experiments.fl_ssl.support.layout import build_fl_ssl_run_dir

REPORT_NAME = "fl_ssl_main_comparison.report.json"
PEFT_TEXT_ENCODER_COMPOSITION_SLUG_BUILDER = (
    "methods.adaptation.peft_text_encoder.update_family_runtime."
    "build_peft_text_encoder_composition_slug"
)
LEGACY_PEFT_UPDATE_FAMILY_NAMES = frozenset({"lora_classifier", "peft_classifier"})


@dataclass(frozen=True, slots=True)
class FlSslRunLayoutMigrationEntry:
    source_dir: Path
    target_dir: Path
    report_path: Path
    run_id: str

    def to_json(self, *, status: str) -> dict[str, str]:
        return {
            "status": status,
            "run_id": self.run_id,
            "source_dir": str(self.source_dir),
            "target_dir": str(self.target_dir),
            "report_path": str(self.report_path),
        }


def build_migration_plan(*, runs_root: Path) -> list[FlSslRunLayoutMigrationEntry]:
    """Return source/target directory moves for historical FL SSL runs."""

    reports = sorted(runs_root.glob(f"**/reports/{REPORT_NAME}"))
    entries: list[FlSslRunLayoutMigrationEntry] = []
    for report_path in reports:
        run_dir = report_path.parent.parent
        payload = _read_json_object(report_path)
        cfg = OmegaConf.create(_layout_cfg_from_report_payload(payload))
        target_dir = build_fl_ssl_run_dir(
            runs_root,
            cfg=cfg,
            run_id=run_dir.name,
        )
        if run_dir.resolve() == target_dir.resolve():
            continue
        entries.append(
            FlSslRunLayoutMigrationEntry(
                source_dir=run_dir,
                target_dir=target_dir,
                report_path=report_path,
                run_id=run_dir.name,
            )
        )
    _validate_migration_plan(entries=entries, runs_root=runs_root)
    return entries


def execute_migration_plan(
    *,
    entries: list[FlSslRunLayoutMigrationEntry],
    manifest_path: Path,
    prune_empty_parents: bool,
    runs_root: Path,
) -> None:
    """Move directories and keep a progress manifest for rollback inspection."""

    moved: list[FlSslRunLayoutMigrationEntry] = []
    _write_manifest(
        manifest_path=manifest_path,
        entries=entries,
        moved=moved,
        dry_run=False,
    )
    for entry in entries:
        entry.target_dir.parent.mkdir(parents=True, exist_ok=True)
        entry.source_dir.rename(entry.target_dir)
        moved.append(entry)
        _write_manifest(
            manifest_path=manifest_path,
            entries=entries,
            moved=moved,
            dry_run=False,
        )
    if prune_empty_parents:
        _prune_empty_source_parents(entries=entries, runs_root=runs_root)


def _layout_cfg_from_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    protocol = _mapping(payload.get("protocol"))
    fl_data_source = _mapping(protocol.get("fl_data_source"))
    source_selection = _mapping(fl_data_source.get("source_selection"))
    labeled_policy = _mapping(fl_data_source.get("labeled_policy"))
    labeled_exposure_policy = _mapping(
        fl_data_source.get("labeled_exposure_policy")
    )
    local_update_budget = _mapping(protocol.get("local_update_budget"))
    round_runtime = _mapping(protocol.get("round_runtime"))
    fl_method = _mapping(protocol.get("fl_method"))
    manual_axes = _mapping(fl_method.get("manual_axes"))
    ssl_method = _mapping(protocol.get("ssl_method"))
    objective = _mapping(protocol.get("objective"))

    update_family_name = _canonical_update_family_name(
        round_runtime.get("update_family_name")
        or manual_axes.get("update_family")
        or round_runtime.get("adapter_family_name")
        or round_runtime.get("payload_adapter_kind")
    )
    payload_adapter_kind = str(
        round_runtime.get("payload_adapter_kind")
        or round_runtime.get("adapter_family_name")
        or update_family_name
    )
    if payload_adapter_kind in LEGACY_PEFT_UPDATE_FAMILY_NAMES:
        payload_adapter_kind = "peft_classifier"

    round_runtime_cfg: dict[str, Any] = {
        "update_family_name": update_family_name,
        "payload_adapter_kind": payload_adapter_kind,
        "aggregation_backend_name": round_runtime.get("aggregation_backend_name"),
    }
    peft_adapter_name = _peft_adapter_name_from_objective(objective)
    if update_family_name == "peft_text_encoder" and peft_adapter_name is not None:
        round_runtime_cfg.update(
            {
                "composition_slug_builder": PEFT_TEXT_ENCODER_COMPOSITION_SLUG_BUILDER,
                "runtime_payload_key": "peft_text_encoder",
                "runtime_payloads": {
                    "peft_text_encoder": {
                        "peft_adapter_name": peft_adapter_name,
                    }
                },
            }
        )

    query_ssl_method_name = (
        objective.get("query_ssl.method_name")
        or ssl_method.get("name")
        or manual_axes.get("client_ssl_objective")
    )

    return {
        "seed": protocol.get("seed"),
        "federated_run_budget": {
            "client_count": protocol.get("client_count"),
            "rounds": protocol.get("round_budget"),
        },
        "query_data_selection": {
            "labeled": source_selection.get("labeled"),
            "unlabeled": source_selection.get("unlabeled"),
        },
        "fl_data": {
            "split_manifest": fl_data_source.get("split_manifest_path")
            or fl_data_source.get("split_id"),
        },
        "fl_client_split_materialization": {
            "labeled_policy": {
                "count_per_class": labeled_policy.get("count_per_class"),
            }
        },
        "labeled_exposure_policy": {
            "name": labeled_exposure_policy.get("name"),
        },
        "query_ssl_method": {
            "name": query_ssl_method_name,
        },
        "ssl_method": {
            "name": ssl_method.get("name"),
            "scenario": ssl_method.get("scenario"),
        },
        "round_runtime": round_runtime_cfg,
        "training_task": {
            "local_epochs": local_update_budget.get("local_epochs"),
            "batch_size": local_update_budget.get("batch_size"),
            "max_steps": local_update_budget.get("max_steps"),
            "objective": objective,
        },
        "fl_method": {
            "composition_mode": fl_method.get("composition_mode"),
        },
    }


def _validate_migration_plan(
    *,
    entries: list[FlSslRunLayoutMigrationEntry],
    runs_root: Path,
) -> None:
    resolved_root = runs_root.resolve()
    target_to_source: dict[Path, Path] = {}
    for entry in entries:
        _require_under_root(path=entry.source_dir, root=resolved_root)
        _require_under_root(path=entry.target_dir, root=resolved_root)
        if not entry.source_dir.exists():
            raise FileNotFoundError(
                f"Source run dir does not exist: {entry.source_dir}"
            )
        if entry.target_dir.exists():
            raise FileExistsError(
                f"Target run dir already exists; refusing to overwrite: "
                f"{entry.target_dir}"
            )
        previous_source = target_to_source.get(entry.target_dir)
        if previous_source is not None:
            raise ValueError(
                "Multiple source runs map to the same target: "
                f"{previous_source} and {entry.source_dir} -> {entry.target_dir}"
            )
        target_to_source[entry.target_dir] = entry.source_dir


def _require_under_root(*, path: Path, root: Path) -> None:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes runs root: {path}")


def _write_manifest(
    *,
    manifest_path: Path,
    entries: list[FlSslRunLayoutMigrationEntry],
    moved: list[FlSslRunLayoutMigrationEntry],
    dry_run: bool,
) -> None:
    moved_sources = {entry.source_dir for entry in moved}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "fl_ssl_run_layout_migration.v1",
                "created_at": datetime.now(UTC).isoformat(),
                "dry_run": dry_run,
                "entry_count": len(entries),
                "moved_count": len(moved),
                "entries": [
                    entry.to_json(
                        status=(
                            "moved" if entry.source_dir in moved_sources else "planned"
                        )
                    )
                    for entry in entries
                ],
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _prune_empty_source_parents(
    *,
    entries: list[FlSslRunLayoutMigrationEntry],
    runs_root: Path,
) -> None:
    resolved_root = runs_root.resolve()
    candidates = sorted(
        {parent for entry in entries for parent in entry.source_dir.parents},
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for candidate in candidates:
        if candidate.resolve() == resolved_root:
            continue
        if resolved_root not in candidate.resolve().parents:
            continue
        try:
            candidate.rmdir()
        except OSError:
            continue


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report must be a JSON object: {path}")
    return payload


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _canonical_update_family_name(value: object) -> str:
    normalized = str(value or "").strip()
    if normalized in LEGACY_PEFT_UPDATE_FAMILY_NAMES:
        return "peft_text_encoder"
    return normalized or "unknown_family"


def _peft_adapter_name_from_objective(objective: dict[str, Any]) -> str | None:
    for key in (
        "peft_classifier.peft_adapter_name",
        "lora_classifier.peft_adapter_name",
        "peft_text_encoder.peft_adapter_name",
    ):
        value = objective.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for value in objective.values():
        if isinstance(value, dict):
            nested = _peft_adapter_name_from_objective(value)
            if nested is not None:
                return nested
    return None


def _default_manifest_path(runs_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return runs_root / f".layout_migration_{timestamp}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move historical FL SSL runs into the current layout.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/fl_ssl"),
        help="FL SSL runs root.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Move directories. Without this flag only a dry-run manifest is written.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest output path. Defaults under runs-root.",
    )
    parser.add_argument(
        "--prune-empty-parents",
        action="store_true",
        help="Remove empty historical parent directories after successful moves.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs_root = args.runs_root
    manifest_path = args.manifest or _default_manifest_path(runs_root)
    entries = build_migration_plan(runs_root=runs_root)
    if args.execute:
        execute_migration_plan(
            entries=entries,
            manifest_path=manifest_path,
            prune_empty_parents=bool(args.prune_empty_parents),
            runs_root=runs_root,
        )
        print(f"moved {len(entries)} FL SSL run dirs")
        print(f"manifest: {manifest_path}")
        return
    _write_manifest(
        manifest_path=manifest_path,
        entries=entries,
        moved=[],
        dry_run=True,
    )
    print(f"planned {len(entries)} FL SSL run dir moves")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
