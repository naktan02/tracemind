"""완료된 FL SSL run의 최종 global state에서 projection artifact를 재생성한다."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.io.client_split_manifest import (
    load_materialized_client_split,
)
from scripts.experiments.fl_ssl.federated_simulation.io.final_projection import (
    build_final_projection_artifacts,
)
from scripts.experiments.fl_ssl.federated_simulation.io.rows import load_jsonl_rows
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedFinalProjectionConfig,
    FederatedLocalTrainerRuntimeConfig,
)
from shared.src.contracts.adapter_contract_families.io import (
    load_shared_adapter_state_payload,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import load_model_manifest_payload
from shared.src.contracts.training_contracts import TrainingObjectiveConfig

REPORT_NAME = "fl_ssl_main_comparison.report.json"


@dataclass(slots=True)
class _ProjectionTrainingTaskConfig:
    objective_config: TrainingObjectiveConfig
    batch_size: int


@dataclass(slots=True)
class _ProjectionRequest:
    final_projection_config: FederatedFinalProjectionConfig
    training_task_config: _ProjectionTrainingTaskConfig
    local_trainer_runtime_config: FederatedLocalTrainerRuntimeConfig
    seed: int
    output_dir: Path
    validation_rows: list[LabeledQueryRow]
    test_rows: list[LabeledQueryRow]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill final validation/test projections for FL SSL runs."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/fl_ssl"),
        help="FL SSL run root used for report discovery.",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Target fl_ssl_main_comparison.report.json. Can be repeated.",
    )
    parser.add_argument(
        "--datasets",
        default="validation,test",
        help="Comma-separated projection datasets. Supports validation,test.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override local trainer device. Default uses the report runtime device.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild projection artifacts even when projection_manifest.json exists.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also process runs whose completed_rounds is lower than round_budget.",
    )
    args = parser.parse_args()

    dataset_names = tuple(
        value.strip() for value in str(args.datasets).split(",") if value.strip()
    )
    report_paths = _resolve_report_paths(
        explicit_reports=[Path(value) for value in args.report],
        runs_root=args.runs_root,
    )
    for report_path in report_paths:
        result = backfill_final_projection(
            report_path=report_path,
            dataset_names=dataset_names,
            force=bool(args.force),
            include_incomplete=bool(args.include_incomplete),
            device_override=args.device,
        )
        print(
            "report="
            f"{report_path} "
            f"status={result['status']} "
            f"revision={result.get('model_revision', '-')} "
            f"datasets={result.get('datasets', '-')}"
        )


def backfill_final_projection(
    *,
    report_path: Path,
    dataset_names: tuple[str, ...],
    force: bool = False,
    include_incomplete: bool = False,
    device_override: str | None = None,
) -> dict[str, object]:
    """저장된 최종 state와 eval rows로 projection artifact를 만든다."""

    payload = _load_json_object(report_path)
    protocol = _as_mapping(payload.get("protocol"))
    if not _is_completed_run(protocol) and not include_incomplete:
        return {"status": "skipped_incomplete"}

    run_dir = report_path.parent.parent
    projection_manifest_path = run_dir / "projections" / "projection_manifest.json"
    if projection_manifest_path.exists() and not force:
        return {
            "status": "skipped_existing",
            "model_revision": _final_model_revision(payload, protocol),
        }

    model_revision = _final_model_revision(payload, protocol)
    if not model_revision:
        return {"status": "skipped_missing_final_revision"}

    manifest_path = (
        run_dir
        / "main_server"
        / "model_manifests"
        / "versions"
        / (f"{model_revision}.json")
    )
    state_path = (
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / (f"{model_revision}.json")
    )
    if not manifest_path.exists() or not state_path.exists():
        return {
            "status": "skipped_missing_final_state",
            "model_revision": model_revision,
        }

    validation_rows, test_rows = _load_projection_rows(protocol)
    request = _build_projection_request(
        run_dir=run_dir,
        protocol=protocol,
        dataset_names=dataset_names,
        validation_rows=validation_rows,
        test_rows=test_rows,
        device_override=device_override,
    )
    active = ActiveSimulationState(
        manifest=load_model_manifest_payload(manifest_path),
        adapter_state=load_shared_adapter_state_payload(state_path),
    )
    artifacts = build_final_projection_artifacts(request=request, active=active)
    if not _as_mapping(artifacts).get("enabled"):
        return {
            "status": "failed",
            "model_revision": model_revision,
            "reason": _as_mapping(artifacts).get("reason"),
            "error": _as_mapping(artifacts).get("error"),
        }
    return {
        "status": "written",
        "model_revision": model_revision,
        "datasets": ",".join(sorted(_as_mapping(artifacts).get("datasets", {}))),
    }


def _resolve_report_paths(
    *,
    explicit_reports: list[Path],
    runs_root: Path,
) -> list[Path]:
    if explicit_reports:
        return sorted(explicit_reports)
    if runs_root.is_file():
        return [runs_root]
    return sorted(path for path in runs_root.rglob(REPORT_NAME) if path.is_file())


def _build_projection_request(
    *,
    run_dir: Path,
    protocol: dict[str, Any],
    dataset_names: tuple[str, ...],
    validation_rows: list[LabeledQueryRow],
    test_rows: list[LabeledQueryRow],
    device_override: str | None,
) -> _ProjectionRequest:
    local_update_budget = _as_mapping(protocol.get("local_update_budget"))
    local_runtime = _as_mapping(protocol.get("local_trainer_runtime"))
    return _ProjectionRequest(
        final_projection_config=FederatedFinalProjectionConfig(
            enabled=True,
            dataset_names=dataset_names or ("validation", "test"),
            fail_on_error=False,
        ),
        training_task_config=_ProjectionTrainingTaskConfig(
            objective_config=TrainingObjectiveConfig.from_mapping(
                _as_mapping(protocol.get("objective"))
            ),
            batch_size=int(local_update_budget.get("batch_size") or 16),
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device=str(device_override or local_runtime.get("device") or "cpu"),
            local_files_only=_bool_value(
                local_runtime.get("local_files_only"),
                default=True,
            ),
            cache_dir=str(local_runtime.get("cache_dir") or "hf_cache"),
            trust_remote_code=_bool_value(
                local_runtime.get("trust_remote_code"),
                default=False,
            ),
            classifier_dropout=float(local_runtime.get("classifier_dropout") or 0.1),
        ),
        seed=int(protocol.get("seed") or 42),
        output_dir=run_dir,
        validation_rows=validation_rows,
        test_rows=test_rows,
    )


def _load_projection_rows(
    protocol: dict[str, Any],
) -> tuple[list[LabeledQueryRow], list[LabeledQueryRow]]:
    fl_data_source = _as_mapping(protocol.get("fl_data_source"))
    split_manifest_path = fl_data_source.get("split_manifest_path")
    if split_manifest_path:
        loaded_split = load_materialized_client_split(
            _project_path(str(split_manifest_path))
        )
        return loaded_split.validation_rows, loaded_split.test_rows

    source_jsonl = _as_mapping(fl_data_source.get("source_jsonl"))
    validation_jsonl = source_jsonl.get("validation")
    test_jsonl = source_jsonl.get("test") or fl_data_source.get("test_jsonl")
    validation_rows = (
        load_jsonl_rows(_project_path(str(validation_jsonl)))
        if validation_jsonl
        else []
    )
    test_rows = load_jsonl_rows(_project_path(str(test_jsonl))) if test_jsonl else []
    return validation_rows, test_rows


def _is_completed_run(protocol: dict[str, Any]) -> bool:
    completed_rounds = _optional_int(protocol.get("completed_rounds"))
    round_budget = _optional_int(protocol.get("round_budget"))
    if completed_rounds is None or round_budget is None:
        return False
    return completed_rounds >= round_budget


def _final_model_revision(
    payload: dict[str, Any],
    protocol: dict[str, Any],
) -> str | None:
    rounds = [
        _as_mapping(round_payload)
        for round_payload in payload.get("rounds") or []
        if isinstance(round_payload, dict)
    ]
    if rounds:
        revision = rounds[-1].get("model_revision")
        if revision:
            return str(revision)
    completed_rounds = _optional_int(protocol.get("completed_rounds"))
    if completed_rounds is None:
        return None
    return f"sim_rev_{completed_rounds:04d}"


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


if __name__ == "__main__":
    main()
