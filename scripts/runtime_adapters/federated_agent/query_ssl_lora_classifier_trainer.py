"""FL simulation Query SSL LoRA-classifier local trainer adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
    materialize_base_lora_classifier_state,
)
from methods.adaptation.lora_classifier.config import (
    LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL,
    LORA_CLASSIFIER_DELTA_FORMAT_INLINE,
    LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED,
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    QuerySslLoraClientTrainingResult,
    QuerySslLoraDeltaMaterialization,
    run_query_ssl_lora_classifier_training_core,
)
from methods.federated.aggregation.base import FederatedAggregationContext
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
    FederatedQuerySslObjectiveConfig,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LoraClassifierDelta,
    LoraClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

_DeltaMaterializationPlan = QuerySslLoraDeltaMaterialization

_AGENT_LOCAL_ARTIFACT_REF_PREFIX = "agent-local://"
_LORA_DELTA_ARTIFACT_SCHEMA_VERSION = "lora_classifier_client_delta_artifact.v1"
_HEAD_DELTA_ARTIFACT_SCHEMA_VERSION = "lora_classifier_client_head_delta_artifact.v1"


@dataclass(frozen=True, slots=True)
class _SimulationQuerySslLoraDeltaMaterializer:
    """simulation output directory를 delta artifact 저장소로 쓰는 bridge."""

    output_dir: Path

    def prepare(
        self,
        *,
        update_id: str,
        training_task: TrainingTask,
        client_id: str,
        delta_format: str,
        artifact_ref_prefix: str,
        lora_parameter_deltas: Mapping[str, Sequence[float]],
        classifier_head_weight_deltas: Mapping[str, Sequence[float]],
        classifier_head_bias_deltas: Mapping[str, float],
    ) -> QuerySslLoraDeltaMaterialization:
        return _prepare_delta_materialization(
            output_dir=self.output_dir,
            update_id=update_id,
            training_task=training_task,
            client_id=client_id,
            delta_format=delta_format,
            artifact_ref_prefix=artifact_ref_prefix,
            lora_parameter_deltas=lora_parameter_deltas,
            classifier_head_weight_deltas=classifier_head_weight_deltas,
            classifier_head_bias_deltas=classifier_head_bias_deltas,
        )


def run_query_ssl_lora_classifier_local_training(
    *,
    client_id: str,
    seed: int,
    output_dir: Path,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    active_adapter_state: object,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    query_ssl_config: FederatedQuerySslObjectiveConfig,
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
    created_at: datetime | None = None,
) -> QuerySslLoraClientTrainingResult:
    """simulation runtime state를 method-owned Query SSL LoRA core에 연결한다."""

    if not isinstance(active_adapter_state, LoraClassifierState):
        raise ValueError(
            "Query SSL LoRA local training requires active LoraClassifierState."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    base_parameters = _load_base_parameters(
        active_adapter_state=active_adapter_state,
        output_dir=output_dir,
        aggregated_at=effective_created_at,
    )
    return run_query_ssl_lora_classifier_training_core(
        client_id=client_id,
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        labels=tuple(str(label) for label in active_adapter_state.label_schema),
        base_parameters=base_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
        query_ssl_config=query_ssl_config,
        lora_config=lora_config,
        trainer_runtime_config=trainer_runtime_config,
        created_at=effective_created_at,
        delta_materializer=_SimulationQuerySslLoraDeltaMaterializer(
            output_dir=output_dir
        ),
    )


def _load_base_parameters(
    *,
    active_adapter_state: LoraClassifierState,
    output_dir: Path,
    aggregated_at: datetime,
) -> LoraClassifierMaterializedState:
    return materialize_base_lora_classifier_state(
        base_state=active_adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=active_adapter_state.model_revision,
            aggregated_at=aggregated_at,
            artifact_loader=AggregationArtifactStore(
                state_root=output_dir / "main_server" / "aggregation_artifacts"
            ),
        ),
    )


def _prepare_delta_materialization(
    *,
    output_dir: Path,
    update_id: str,
    training_task: TrainingTask,
    client_id: str,
    delta_format: str,
    artifact_ref_prefix: str = "agent-local://lora_classifier",
    lora_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> QuerySslLoraDeltaMaterialization:
    normalized_delta_format = str(delta_format).strip()
    if normalized_delta_format == LORA_CLASSIFIER_DELTA_FORMAT_INLINE:
        return QuerySslLoraDeltaMaterialization(
            delta_format=normalized_delta_format,
            lora_delta_artifact_ref=None,
            classifier_head_delta_artifact_ref=None,
            include_inline_deltas=True,
        )
    if normalized_delta_format == LORA_CLASSIFIER_DELTA_FORMAT_AGENT_LOCAL:
        lora_delta_ref = _agent_local_ref_for_artifact(
            artifact_ref_prefix=artifact_ref_prefix,
            training_task=training_task,
            client_id=client_id,
            update_id=update_id,
            artifact_name="lora_delta",
        )
        head_delta_ref = _agent_local_ref_for_artifact(
            artifact_ref_prefix=artifact_ref_prefix,
            training_task=training_task,
            client_id=client_id,
            update_id=update_id,
            artifact_name="classifier_head_delta",
        )
        _save_agent_local_json_artifact(
            output_dir=output_dir,
            artifact_ref=lora_delta_ref,
            payload=_build_lora_delta_artifact_payload(
                update_id=update_id,
                training_task=training_task,
                client_id=client_id,
                lora_parameter_deltas=lora_parameter_deltas,
            ),
        )
        _save_agent_local_json_artifact(
            output_dir=output_dir,
            artifact_ref=head_delta_ref,
            payload=_build_classifier_head_delta_artifact_payload(
                update_id=update_id,
                training_task=training_task,
                client_id=client_id,
                classifier_head_weight_deltas=classifier_head_weight_deltas,
                classifier_head_bias_deltas=classifier_head_bias_deltas,
            ),
        )
        return QuerySslLoraDeltaMaterialization(
            delta_format=normalized_delta_format,
            lora_delta_artifact_ref=lora_delta_ref,
            classifier_head_delta_artifact_ref=head_delta_ref,
            include_inline_deltas=False,
        )
    if normalized_delta_format != LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED:
        raise ValueError(
            f"Unsupported Query SSL LoRA delta_format: {normalized_delta_format!r}."
        )

    store = AggregationArtifactStore(
        state_root=output_dir / "main_server" / "aggregation_artifacts"
    )
    artifact_id_prefix = (
        f"client_updates/{training_task.round_id}/{client_id}/{update_id}"
    )
    lora_delta_ref = store.ref_for_artifact(f"{artifact_id_prefix}/lora_delta")
    head_delta_ref = store.ref_for_artifact(
        f"{artifact_id_prefix}/classifier_head_delta"
    )
    store.save_json_artifact_ref(
        artifact_ref=lora_delta_ref,
        payload=_build_lora_delta_artifact_payload(
            update_id=update_id,
            training_task=training_task,
            client_id=client_id,
            lora_parameter_deltas=lora_parameter_deltas,
        ),
    )
    store.save_json_artifact_ref(
        artifact_ref=head_delta_ref,
        payload=_build_classifier_head_delta_artifact_payload(
            update_id=update_id,
            training_task=training_task,
            client_id=client_id,
            classifier_head_weight_deltas=classifier_head_weight_deltas,
            classifier_head_bias_deltas=classifier_head_bias_deltas,
        ),
    )
    return QuerySslLoraDeltaMaterialization(
        delta_format=normalized_delta_format,
        lora_delta_artifact_ref=lora_delta_ref,
        classifier_head_delta_artifact_ref=head_delta_ref,
        include_inline_deltas=False,
    )


def upload_agent_local_lora_classifier_update(
    *,
    output_dir: Path,
    update_payload: LoraClassifierDelta,
) -> LoraClassifierDelta:
    """agent-local delta artifact ref를 server-owned ref로 materialize한다."""

    update_fields: dict[str, object] = {}
    if _is_agent_local_ref(update_payload.lora_delta_artifact_ref):
        update_fields["lora_delta_artifact_ref"] = _upload_agent_local_artifact(
            output_dir=output_dir,
            agent_local_ref=update_payload.lora_delta_artifact_ref,
        )
    if _is_agent_local_ref(update_payload.classifier_head_delta_artifact_ref):
        update_fields["classifier_head_delta_artifact_ref"] = (
            _upload_agent_local_artifact(
                output_dir=output_dir,
                agent_local_ref=update_payload.classifier_head_delta_artifact_ref,
            )
        )
    if not update_fields:
        return update_payload
    update_fields["delta_format"] = LORA_CLASSIFIER_DELTA_FORMAT_SERVER_UPLOADED
    return update_payload.model_copy(update=update_fields)


def _upload_agent_local_artifact(
    *,
    output_dir: Path,
    agent_local_ref: str | None,
) -> str:
    if agent_local_ref is None:
        raise ValueError("agent_local_ref must not be None.")
    payload = _load_agent_local_json_artifact(
        output_dir=output_dir,
        artifact_ref=agent_local_ref,
    )
    store = AggregationArtifactStore(
        state_root=output_dir / "main_server" / "aggregation_artifacts"
    )
    server_ref = store.ref_for_artifact(
        "client_uploads/" + "/".join(_agent_local_artifact_parts(agent_local_ref))
    )
    store.save_json_artifact_ref(artifact_ref=server_ref, payload=payload)
    return server_ref


def _agent_local_ref_for_artifact(
    *,
    artifact_ref_prefix: str,
    training_task: TrainingTask,
    client_id: str,
    update_id: str,
    artifact_name: str,
) -> str:
    normalized_prefix = artifact_ref_prefix.strip().rstrip("/")
    if not normalized_prefix.startswith(_AGENT_LOCAL_ARTIFACT_REF_PREFIX):
        raise ValueError(
            "LoRA-classifier agent-local artifact_ref_prefix must start with "
            f"{_AGENT_LOCAL_ARTIFACT_REF_PREFIX!r}."
        )
    return "/".join(
        (
            normalized_prefix,
            _safe_ref_part(training_task.round_id),
            _safe_ref_part(client_id),
            _safe_ref_part(update_id),
            _safe_ref_part(artifact_name),
        )
    )


def _save_agent_local_json_artifact(
    *,
    output_dir: Path,
    artifact_ref: str,
    payload: dict[str, object],
) -> None:
    path = _path_for_agent_local_artifact(
        output_dir=output_dir,
        artifact_ref=artifact_ref,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_agent_local_json_artifact(
    *,
    output_dir: Path,
    artifact_ref: str,
) -> dict[str, object]:
    path = _path_for_agent_local_artifact(
        output_dir=output_dir,
        artifact_ref=artifact_ref,
    )
    if not path.exists():
        raise FileNotFoundError(f"Agent-local artifact not found: {artifact_ref}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Agent-local artifact must be a JSON object: {artifact_ref}")
    return payload


def _path_for_agent_local_artifact(
    *,
    output_dir: Path,
    artifact_ref: str,
) -> Path:
    parts = _agent_local_artifact_parts(artifact_ref)
    return (
        output_dir
        / "agents"
        / "local_artifacts"
        / "versions"
        / Path(*parts[:-1])
        / f"{parts[-1]}.json"
    )


def _agent_local_artifact_parts(artifact_ref: str) -> tuple[str, ...]:
    if not artifact_ref.startswith(_AGENT_LOCAL_ARTIFACT_REF_PREFIX):
        raise ValueError(
            f"Expected LoRA-classifier agent-local artifact ref, got: {artifact_ref!r}."
        )
    raw_artifact_id = artifact_ref.removeprefix(_AGENT_LOCAL_ARTIFACT_REF_PREFIX)
    parts = tuple(part.strip() for part in raw_artifact_id.split("/") if part.strip())
    if not parts:
        raise ValueError("agent-local artifact ref must contain an artifact id.")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("agent-local artifact ref must not contain path traversal.")
    return parts


def _is_agent_local_ref(artifact_ref: str | None) -> bool:
    return artifact_ref is not None and artifact_ref.startswith(
        _AGENT_LOCAL_ARTIFACT_REF_PREFIX
    )


def _safe_ref_part(value: str) -> str:
    normalized = str(value).strip().replace("/", "_")
    if not normalized or normalized in {".", ".."}:
        raise ValueError("artifact ref path parts must not be empty or traversal.")
    return normalized


def _build_lora_delta_artifact_payload(
    *,
    update_id: str,
    training_task: TrainingTask,
    client_id: str,
    lora_parameter_deltas: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    return {
        "schema_version": _LORA_DELTA_ARTIFACT_SCHEMA_VERSION,
        "update_id": update_id,
        "round_id": training_task.round_id,
        "client_id": client_id,
        "lora_parameter_deltas": {
            str(key): [float(value) for value in values]
            for key, values in lora_parameter_deltas.items()
        },
    }


def _build_classifier_head_delta_artifact_payload(
    *,
    update_id: str,
    training_task: TrainingTask,
    client_id: str,
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> dict[str, object]:
    return {
        "schema_version": _HEAD_DELTA_ARTIFACT_SCHEMA_VERSION,
        "update_id": update_id,
        "round_id": training_task.round_id,
        "client_id": client_id,
        "classifier_head_weight_deltas": {
            str(key): [float(value) for value in values]
            for key, values in classifier_head_weight_deltas.items()
        },
        "classifier_head_bias_deltas": {
            str(key): float(value) for key, value in classifier_head_bias_deltas.items()
        },
    }
