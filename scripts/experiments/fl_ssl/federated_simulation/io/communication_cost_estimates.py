"""FL SSL report artifact 기반 통신량 추정 helper."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.model_revisions import (
    build_active_model_revision_for_round,
)

COMMUNICATION_ESTIMATE_SCHEMA_VERSION = "fl_ssl_artifact_communication_cost.v1"
SPARSE_VALUE_BYTES = 4
SPARSE_INDEX_BYTES = 4


def build_artifact_communication_estimate(
    *,
    run_dir: Path,
    rounds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """저장된 run artifact 크기로 가능한 C2S/S2C byte 추정치를 계산한다."""

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
        active_revision = build_active_model_revision_for_round(round_index)
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
                "s2c_partitioned_sparse_transport_bytes_estimated": (
                    round_s2c_sparse
                ),
                "s2c_manifest_task_bytes_estimated": round_s2c_manifest_task,
                "s2c_total_bytes_estimated": round_s2c_total,
            }
        )

    c2s_total = c2s_payload_bytes + c2s_artifact_bytes
    s2c_total = (
        s2c_global_state_bytes + s2c_partitioned_sparse_bytes + s2c_manifest_task_bytes
    )
    return {
        "schema_version": COMMUNICATION_ESTIMATE_SCHEMA_VERSION,
        "status": "estimated_from_saved_run_artifacts",
        "basis": {
            "c2s": (
                "client update JSON payload bytes plus server-owned update "
                "artifact bytes saved under the run directory"
            ),
            "s2c": (
                "active global update-family state artifact bytes multiplied "
                "by selected clients per round, plus partitioned sparse "
                "transport estimates when saved metadata is available; base "
                "model material is assumed pre-cached and excluded"
            ),
            "limitations": [
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


def attach_artifact_communication_estimate(
    *,
    communication_cost: dict[str, object],
    run_dir: Path | None,
    rounds: Sequence[Mapping[str, object]],
) -> None:
    """report communication_cost에 artifact 기반 추정치를 붙인다."""

    if run_dir is None:
        return
    communication_cost["artifact_byte_estimates"] = (
        build_artifact_communication_estimate(
            run_dir=run_dir,
            rounds=rounds,
        )
    )


def _round_c2s_payload_bytes(round_payload: Mapping[str, object]) -> int:
    clients = _list_of_mappings(round_payload.get("clients"))
    measured = sum(
        _int_value(client.get("client_payload_bytes")) or 0 for client in clients
    )
    if measured:
        return measured
    return _int_value(round_payload.get("total_payload_bytes")) or 0


def _round_c2s_artifact_bytes(round_payload: Mapping[str, object]) -> int:
    return sum(
        _int_value(client.get("client_artifact_bytes")) or 0
        for client in _list_of_mappings(round_payload.get("clients"))
    )


def _global_state_material_bytes(*, run_dir: Path, model_revision: str) -> int:
    state = _load_global_state(run_dir=run_dir, model_revision=model_revision)
    artifact_refs = _global_state_artifact_refs(state)
    return sum(
        _artifact_ref_size(run_dir=run_dir, artifact_ref=artifact_ref)
        for artifact_ref in artifact_refs
    )


def _global_partitioned_sparse_transport_bytes(
    *,
    run_dir: Path,
    model_revision: str,
) -> int:
    state = _load_global_state(run_dir=run_dir, model_revision=model_revision)
    nonzero_count = sum(
        _partitioned_artifact_nonzero_count(
            run_dir=run_dir,
            artifact_ref=artifact_ref,
        )
        for artifact_ref in _global_state_artifact_refs(state)
    )
    return nonzero_count * (SPARSE_VALUE_BYTES + SPARSE_INDEX_BYTES)


def _global_state_artifact_refs(state: Mapping[str, object]) -> tuple[str, ...]:
    refs = tuple(
        _optional_str(value)
        for key, value in state.items()
        if str(key).endswith("_artifact_ref")
    )
    return tuple(ref for ref in refs if ref is not None)


def _load_global_state(*, run_dir: Path, model_revision: str) -> Mapping[str, object]:
    state_path = (
        run_dir
        / "main_server"
        / "shared_adapter_states"
        / "versions"
        / f"{model_revision}.json"
    )
    return _load_json_object(state_path) if state_path.exists() else {}


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
    partitioned_values = [
        value for key, value in artifact.items() if str(key).startswith("partitioned_")
    ]
    return sum(_nested_nonzero_float_count(value) for value in partitioned_values)


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
    for prefix in ("server-aggregate://", "aggregation_artifact::"):
        if artifact_ref.startswith(prefix):
            relative = artifact_ref[len(prefix) :]
            break
    else:
        return run_dir / artifact_ref
    artifact_path = (
        run_dir / "main_server" / "aggregation_artifacts" / "versions" / relative
    )
    if artifact_path.suffix:
        return artifact_path
    json_path = artifact_path.with_suffix(".json")
    if json_path.exists():
        return json_path
    safetensors_path = artifact_path.with_suffix(".safetensors")
    if safetensors_path.exists():
        return safetensors_path
    return json_path


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


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
