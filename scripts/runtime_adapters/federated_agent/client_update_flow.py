"""FL simulation client update submit/summary flow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from methods.common.timing import TimingRecorder
from methods.evaluation.pseudo_label_quality import PseudoLabelQualitySummary
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    client_update_submission,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.diagnostic_view import (
    build_client_diagnostic_unlabeled_view,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    BootstrappedSimulation,
    ClientRoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent.artifact_store import (
    SimulationClientArtifactStore,
)
from shared.src.contracts.training_contracts import ClientMetricKeys


class LocalTrainingRoundResult(Protocol):
    """client update submit/summary에 필요한 local training result 표면."""

    update_envelope: Any
    update_payload: Any
    candidate_count: int
    accepted_count: int
    client_metrics: Mapping[str, float]
    pseudo_label_quality: PseudoLabelQualitySummary


class UploadClientUpdate(Protocol):
    """agent-local update payload를 server-owned payload로 변환한다."""

    def __call__(
        self,
        *,
        artifact_store: SimulationClientArtifactStore,
        update_payload: Any,
    ) -> Any:
        """server에 제출 가능한 payload를 반환한다."""


class ClientArtifactByteCounter(Protocol):
    """server-owned client artifact byte size를 계산한다."""

    def __call__(
        self,
        *,
        artifact_store: SimulationClientArtifactStore,
        update_payload: Any,
    ) -> int:
        """artifact byte count를 반환한다."""


def submit_local_training_result(
    *,
    bootstrapped: BootstrappedSimulation,
    round_id: str,
    output_dir: Path,
    client_id: str,
    diagnostic_candidate_count: int,
    client_train_time_seconds: float,
    timing_recorder: TimingRecorder,
    local_result: LocalTrainingRoundResult,
    upload_client_update: UploadClientUpdate,
    client_artifact_byte_counter: ClientArtifactByteCounter,
    method_diagnostics: Mapping[str, float] | None = None,
    peer_client_snapshot: Any | None = None,
    client_partition_snapshot: Mapping[str, Any] | None = None,
    query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> ClientRoundExecution:
    """local training 결과를 server에 제출하고 client round summary를 만든다."""

    artifact_store = SimulationClientArtifactStore(output_dir=output_dir)
    with timing_recorder.measure("update_upload_materialize_seconds"):
        server_update_payload = upload_client_update(
            artifact_store=artifact_store,
            update_payload=local_result.update_payload,
        )
    with timing_recorder.measure("server_update_submit_seconds"):
        update_submitted = client_update_submission.accept_client_update(
            server_runtime=bootstrapped.server_runtime,
            round_id=round_id,
            update_envelope=local_result.update_envelope,
            update_payload=server_update_payload,
        )
    pseudo_label_quality = _pseudo_label_quality(local_result)
    summary = ClientRoundSummary(
        client_id=client_id,
        candidate_count=local_result.candidate_count,
        diagnostic_candidate_count=diagnostic_candidate_count,
        accepted_count=local_result.accepted_count,
        update_generated=update_submitted,
        delta_l2_norm=client_update_submission.extract_delta_l2_norm(
            local_result.update_envelope
        ),
        aggregation_example_count=(
            client_update_submission.extract_aggregation_example_count(
                local_result.update_envelope
            )
        ),
        client_train_time_seconds=client_train_time_seconds,
        client_payload_bytes=(
            client_update_submission.payload_byte_count(server_update_payload)
            if update_submitted
            else None
        ),
        client_artifact_bytes=(
            client_artifact_byte_counter(
                artifact_store=artifact_store,
                update_payload=server_update_payload,
            )
            if update_submitted
            else None
        ),
        pseudo_label_confidence_mean=(
            pseudo_label_quality.pseudo_label_confidence_mean
            if pseudo_label_quality.pseudo_label_confidence_mean is not None
            else local_result.client_metrics.get(ClientMetricKeys.MEAN_CONFIDENCE)
        ),
        pseudo_label_margin_mean=(
            pseudo_label_quality.pseudo_label_margin_mean
            if pseudo_label_quality.pseudo_label_margin_mean is not None
            else local_result.client_metrics.get(ClientMetricKeys.MEAN_MARGIN)
        ),
        pseudo_label_correct_count=pseudo_label_quality.pseudo_label_correct_count,
        pseudo_label_evaluated_count=pseudo_label_quality.pseudo_label_evaluated_count,
        accepted_label_distribution=pseudo_label_quality.accepted_label_distribution,
        rejected_label_distribution=pseudo_label_quality.rejected_label_distribution,
        method_diagnostics=dict(method_diagnostics or {}),
        timing_breakdown=timing_recorder.to_mapping(),
    )
    write_client_timing_snapshot(
        output_dir=output_dir,
        round_id=round_id,
        update_id=str(local_result.update_envelope.update_id),
        summary=summary,
    )
    return ClientRoundExecution(
        summary=summary,
        update_submitted=update_submitted,
        peer_client_snapshot=peer_client_snapshot,
        client_partition_snapshot=client_partition_snapshot or {},
        query_ssl_algorithm_state=query_ssl_algorithm_state or {},
    )


def write_client_timing_snapshot(
    *,
    output_dir: Path,
    round_id: str,
    update_id: str,
    summary: ClientRoundSummary,
) -> Path:
    """round 종료 전에도 client별 runtime timing을 확인할 수 있게 저장한다."""

    payload = {
        "schema_version": "fl_client_timing_snapshot.v1",
        "round_id": round_id,
        "client_id": summary.client_id,
        "update_id": update_id,
        "client_train_time_seconds": summary.client_train_time_seconds,
        "candidate_count": summary.candidate_count,
        "diagnostic_candidate_count": summary.diagnostic_candidate_count,
        "accepted_count": summary.accepted_count,
        "update_generated": summary.update_generated,
        "client_payload_bytes": summary.client_payload_bytes,
        "client_artifact_bytes": summary.client_artifact_bytes,
        "timing_breakdown": dict(summary.timing_breakdown),
    }
    path = (
        output_dir
        / "diagnostics"
        / "client_timing"
        / _safe_path_part(round_id)
        / f"{_safe_path_part(summary.client_id)}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _safe_path_part(value: str) -> str:
    normalized = str(value).strip().replace("/", "_")
    if not normalized or normalized in {".", ".."}:
        raise ValueError("timing snapshot path part must not be empty or traversal.")
    return normalized


def build_round_diagnostic_unlabeled_rows(
    *,
    request: SimulationRunRequest,
    round_id: str,
    shard: FederatedClientShard,
) -> list[Any]:
    """client round diagnostic unlabeled view를 만든다."""

    return build_client_diagnostic_unlabeled_view(
        rows=shard.unlabeled_rows,
        config=request.diagnostic_view_config,
        run_seed=request.seed,
        round_index=round_index_from_id(round_id),
        client_id=shard.client_id,
    )


def round_index_from_id(round_id: str) -> int:
    return int(round_id.rsplit("_", maxsplit=1)[-1])


def _pseudo_label_quality(
    local_result: LocalTrainingRoundResult,
) -> PseudoLabelQualitySummary:
    return local_result.pseudo_label_quality
