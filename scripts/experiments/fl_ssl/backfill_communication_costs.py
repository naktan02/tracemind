"""기존 FL SSL run report에 후처리 통신량 추정치를 병합한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

POSTHOC_SCHEMA_VERSION = "fl_ssl_posthoc_communication_cost.v1"
REPORT_NAME = "fl_ssl_main_comparison.report.json"
SIDECAR_NAME = "fl_ssl_posthoc_communication_cost.json"
SPARSE_VALUE_BYTES = 4
SPARSE_INDEX_BYTES = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Backfill C2S/S2C byte estimates into existing FL SSL reports.")
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Target fl_ssl_main_comparison.report.json. Can be repeated.",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Discover FL SSL reports below this runs root.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write sidecar and merge posthoc fields into canonical report.",
    )
    args = parser.parse_args()

    report_paths = _resolve_report_paths(
        explicit_reports=[Path(value) for value in args.report],
        runs_root=None if args.runs_root is None else Path(args.runs_root),
    )
    for report_path in report_paths:
        payload = _load_json_object(report_path)
        posthoc = build_posthoc_communication_cost(
            report_path=report_path,
            payload=payload,
        )
        if args.write:
            write_posthoc_communication_cost(
                report_path=report_path,
                payload=payload,
                posthoc=posthoc,
            )
        print(
            "report="
            f"{report_path} "
            f"c2s={posthoc['c2s_total_bytes']} "
            f"s2c={posthoc['s2c_total_bytes_estimated']} "
            f"total={posthoc['bidirectional_total_bytes_estimated']} "
            f"write={bool(args.write)}"
        )


def build_posthoc_communication_cost(
    *,
    report_path: Path,
    payload: dict[str, Any],
) -> dict[str, object]:
    """저장된 run artifact 크기로 가능한 C2S/S2C byte 추정치를 계산한다."""

    run_dir = report_path.parent.parent
    rounds = _list_of_mappings(payload.get("rounds"))
    c2s_payload_bytes = 0
    c2s_artifact_bytes = 0
    s2c_global_state_bytes = 0
    s2c_partitioned_sparse_bytes = 0
    s2c_manifest_task_bytes = 0
    per_round: list[dict[str, object]] = []

    for round_payload in rounds:
        round_index = _int_value(round_payload.get("round_index")) or 0
        selected_client_count = (
            _int_value(round_payload.get("selected_client_count"))
            or len(_list_of_mappings(round_payload.get("clients")))
            or _int_value(round_payload.get("update_count"))
            or 0
        )
        round_c2s_payload = _round_c2s_payload_bytes(round_payload)
        round_c2s_artifact = _round_c2s_artifact_bytes(round_payload)
        active_revision = f"sim_rev_{max(0, round_index - 1):04d}"
        state_bytes = _global_state_material_bytes(
            run_dir=run_dir,
            model_revision=active_revision,
        )
        sparse_state_bytes = (
            _global_partitioned_sparse_transport_bytes(
                run_dir=run_dir,
                model_revision=active_revision,
            )
            if round_index > 1
            else 0
        )
        manifest_task_bytes = _round_manifest_task_bytes(
            run_dir=run_dir,
            round_id=str(round_payload.get("round_id") or ""),
        )
        round_s2c_global = state_bytes * selected_client_count
        round_s2c_sparse = sparse_state_bytes * selected_client_count
        round_s2c_manifest_task = manifest_task_bytes * selected_client_count
        round_s2c_total = round_s2c_global + round_s2c_sparse + round_s2c_manifest_task

        c2s_payload_bytes += round_c2s_payload
        c2s_artifact_bytes += round_c2s_artifact
        s2c_global_state_bytes += round_s2c_global
        s2c_partitioned_sparse_bytes += round_s2c_sparse
        s2c_manifest_task_bytes += round_s2c_manifest_task
        per_round.append(
            {
                "round_id": round_payload.get("round_id"),
                "round_index": round_index,
                "selected_client_count": selected_client_count,
                "active_model_revision_sent_to_clients": active_revision,
                "c2s_payload_bytes": round_c2s_payload,
                "c2s_artifact_bytes": round_c2s_artifact,
                "c2s_total_bytes": round_c2s_payload + round_c2s_artifact,
                "s2c_global_state_bytes_estimated": round_s2c_global,
                "s2c_partitioned_sparse_transport_bytes_estimated": (round_s2c_sparse),
                "s2c_manifest_task_bytes_estimated": round_s2c_manifest_task,
                "s2c_total_bytes_estimated": round_s2c_total,
            }
        )

    c2s_total = c2s_payload_bytes + c2s_artifact_bytes
    s2c_total = (
        s2c_global_state_bytes + s2c_partitioned_sparse_bytes + s2c_manifest_task_bytes
    )
    return {
        "schema_version": POSTHOC_SCHEMA_VERSION,
        "status": "estimated_from_saved_run_artifacts",
        "basis": {
            "c2s": (
                "client update JSON payload bytes plus server-owned update "
                "artifact bytes saved under the run directory"
            ),
            "s2c": (
                "active global LoRA/classifier state artifact bytes multiplied "
                "by selected clients per round, plus partitioned sparse "
                "transport estimates when saved metadata is available; base "
                "transformer is assumed pre-cached and excluded"
            ),
            "limitations": [
                "runtime-only final evaluation timing cannot be recovered posthoc",
                "actual network transport framing/compression is not measured",
                "in-memory helper snapshot exchange is not directly measured",
                "partitioned sparse transport uses non-zero value plus flat index "
                "byte estimates from saved artifact metadata, not measured packets",
                "round 1 has no previous client partition snapshot, so sparse S2C "
                "is estimated only from round 2 onward",
            ],
        },
        "c2s_payload_bytes": c2s_payload_bytes,
        "c2s_artifact_bytes": c2s_artifact_bytes,
        "c2s_total_bytes": c2s_total,
        "s2c_global_state_bytes_estimated": s2c_global_state_bytes,
        "s2c_partitioned_sparse_transport_bytes_estimated": (
            s2c_partitioned_sparse_bytes
        ),
        "s2c_manifest_task_bytes_estimated": s2c_manifest_task_bytes,
        "s2c_total_bytes_estimated": s2c_total,
        "bidirectional_total_bytes_estimated": c2s_total + s2c_total,
        "per_round": per_round,
    }


def write_posthoc_communication_cost(
    *,
    report_path: Path,
    payload: dict[str, Any],
    posthoc: dict[str, object],
) -> None:
    """sidecar와 canonical report 양쪽에 후처리 추정치를 쓴다."""

    sidecar_path = report_path.parent / SIDECAR_NAME
    sidecar_path.write_text(
        json.dumps(posthoc, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = _ensure_mapping(payload, "diagnostics")
    diagnostic_cost = _ensure_mapping(diagnostics, "communication_cost")
    diagnostic_cost["posthoc_byte_estimates"] = posthoc
    metrics = _ensure_mapping(payload, "metrics")
    secondary = _ensure_mapping(metrics, "secondary")
    secondary_cost = secondary.get("communication_cost")
    if not isinstance(secondary_cost, dict):
        secondary_cost = dict(diagnostic_cost)
        secondary["communication_cost"] = secondary_cost
    secondary_cost["posthoc_byte_estimates"] = posthoc
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _resolve_report_paths(
    *,
    explicit_reports: list[Path],
    runs_root: Path | None,
) -> list[Path]:
    paths = list(explicit_reports)
    if runs_root is not None:
        paths.extend(sorted(runs_root.rglob(f"reports/{REPORT_NAME}")))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    if not unique_paths:
        raise SystemExit("No FL SSL reports matched.")
    return unique_paths


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _round_c2s_payload_bytes(round_payload: dict[str, Any]) -> int:
    clients = _list_of_mappings(round_payload.get("clients"))
    measured = sum(
        _int_value(client.get("client_payload_bytes")) or 0 for client in clients
    )
    if measured:
        return measured
    return _int_value(round_payload.get("total_payload_bytes")) or 0


def _round_c2s_artifact_bytes(round_payload: dict[str, Any]) -> int:
    return sum(
        _int_value(client.get("client_artifact_bytes")) or 0
        for client in _list_of_mappings(round_payload.get("clients"))
    )


def _global_state_material_bytes(*, run_dir: Path, model_revision: str) -> int:
    state_path = (
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / f"{model_revision}.json"
    )
    state = _load_json_object(state_path) if state_path.exists() else {}
    artifact_refs = (
        _optional_str(state.get("lora_adapter_artifact_ref")),
        _optional_str(state.get("classifier_head_artifact_ref")),
    )
    return sum(
        _artifact_ref_size(run_dir=run_dir, artifact_ref=artifact_ref)
        for artifact_ref in artifact_refs
        if artifact_ref is not None
    )


def _global_partitioned_sparse_transport_bytes(
    *,
    run_dir: Path,
    model_revision: str,
) -> int:
    state_path = (
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / f"{model_revision}.json"
    )
    state = _load_json_object(state_path) if state_path.exists() else {}
    artifact_refs = (
        _optional_str(state.get("lora_adapter_artifact_ref")),
        _optional_str(state.get("classifier_head_artifact_ref")),
    )
    nonzero_count = sum(
        _partitioned_artifact_nonzero_count(
            run_dir=run_dir,
            artifact_ref=artifact_ref,
        )
        for artifact_ref in artifact_refs
        if artifact_ref is not None
    )
    return nonzero_count * (SPARSE_VALUE_BYTES + SPARSE_INDEX_BYTES)


def _partitioned_artifact_nonzero_count(*, run_dir: Path, artifact_ref: str) -> int:
    path = _server_aggregate_artifact_path(run_dir=run_dir, artifact_ref=artifact_ref)
    if not path.exists():
        return 0
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if not isinstance(artifact, dict):
        return 0
    return sum(
        _nested_nonzero_float_count(artifact.get(key))
        for key in (
            "partitioned_lora_parameters",
            "partitioned_classifier_head_weights",
            "partitioned_classifier_head_biases",
        )
    )


def _nested_nonzero_float_count(value: object) -> int:
    if isinstance(value, dict):
        return sum(_nested_nonzero_float_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_nested_nonzero_float_count(child) for child in value)
    try:
        return int(float(value) != 0.0)
    except (TypeError, ValueError):
        return 0


def _round_manifest_task_bytes(*, run_dir: Path, round_id: str) -> int:
    if not round_id:
        return 0
    round_path = run_dir / "main_server" / "rounds" / "records" / f"{round_id}.json"
    if not round_path.exists():
        return 0
    payload = _load_json_object(round_path)
    return len(
        json.dumps(
            {
                "active_manifest": payload.get("active_manifest"),
                "training_task": payload.get("training_task"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _artifact_ref_size(*, run_dir: Path, artifact_ref: str) -> int:
    path = _server_aggregate_artifact_path(
        run_dir=run_dir,
        artifact_ref=artifact_ref,
    )
    return path.stat().st_size if path.exists() else 0


def _server_aggregate_artifact_path(*, run_dir: Path, artifact_ref: str) -> Path:
    prefix = "server-aggregate://"
    if not artifact_ref.startswith(prefix):
        return run_dir / artifact_ref
    relative = artifact_ref[len(prefix) :]
    return (
        run_dir
        / "main_server"
        / "aggregation_artifacts"
        / "versions"
        / (relative + ".json")
    )


def _ensure_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    mapping[key] = value
    return value


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _int_value(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    main()
