"""PEFT text encoder simulation supervised seed projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from methods.adaptation.peft_text_encoder.aggregation import (
    peft_encoder_fedavg_projection as peft_fedavg_projection,
)
from methods.adaptation.peft_text_encoder.aggregation import (
    peft_encoder_state_projection as peft_state_projection,
)
from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.federated_ssl import (
    supervised_seed_step,
)
from methods.adaptation.peft_text_encoder.training.query_ssl_local_training import (
    PeftEncoderTrainerRuntimeConfig,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    materialize_base_peft_encoder_state,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.federated.aggregation.base import FederatedAggregationContext
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

PEFT_ENCODER_SEED_ADAPTER_ARTIFACT_SLOT = (
    peft_fedavg_projection.PEFT_ADAPTER_ARTIFACT_SLOT
)
PEFT_ENCODER_SEED_CLASSIFIER_HEAD_ARTIFACT_SLOT = (
    peft_fedavg_projection.CLASSIFIER_HEAD_ARTIFACT_SLOT
)
PEFT_ENCODER_SUPERVISED_SEED_STEP_SEED_OFFSET = 7919
PEFT_ENCODER_SUPERVISED_SEED_REVISION_SUFFIX = "server_seed"


def build_peft_encoder_supervised_seed_revision(
    *,
    base_model_revision: str,
) -> str:
    """server supervised seed step publication model revision을 만든다."""

    return f"{base_model_revision}_{PEFT_ENCODER_SUPERVISED_SEED_REVISION_SUFFIX}"


def peft_encoder_supervised_seed_step_seed(
    *,
    base_seed: int,
    round_index: int,
) -> int:
    """server supervised seed step의 deterministic seed를 계산한다."""

    return (
        int(base_seed)
        + PEFT_ENCODER_SUPERVISED_SEED_STEP_SEED_OFFSET
        + int(round_index)
    )


def peft_encoder_supervised_seed_artifact_names() -> tuple[str, str]:
    """server seed projection이 publish할 PEFT artifact slot 이름을 반환한다."""

    return (
        PEFT_ENCODER_SEED_ADAPTER_ARTIFACT_SLOT,
        PEFT_ENCODER_SEED_CLASSIFIER_HEAD_ARTIFACT_SLOT,
    )


@dataclass(frozen=True, slots=True)
class PeftEncoderSupervisedSeedProjection:
    """server publication runtime이 저장할 seed-step projection 결과."""

    next_state: PeftClassifierState
    artifacts: dict[str, dict[str, object]]
    metrics: dict[str, float]


def build_peft_encoder_supervised_seed_projection(
    *,
    adapter_state: PeftClassifierState,
    bootstrap_rows: list[LabeledQueryRow],
    aggregation_context: FederatedAggregationContext,
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_clip_norm: float | None,
    next_model_revision: str,
    updated_at: datetime,
    artifact_refs_by_name: Mapping[str, str],
    artifact_format: str,
) -> PeftEncoderSupervisedSeedProjection:
    """server bootstrap rows로 다음 PEFT encoder shared state projection을 만든다."""

    labels = tuple(str(label) for label in adapter_state.label_schema)
    base_parameters = materialize_base_peft_encoder_state(
        base_state=adapter_state,
        context=aggregation_context,
    )
    seed_result = supervised_seed_step.run_peft_encoder_supervised_seed_step_core(
        labels=labels,
        base_parameters=base_parameters,
        bootstrap_rows=bootstrap_rows,
        peft_config=peft_config,
        trainer_runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
    )
    projection = peft_state_projection.build_peft_encoder_state_projection(
        base_state=adapter_state,
        base_parameters=base_parameters,
        next_model_revision=next_model_revision,
        updated_at=updated_at,
        peft_adapter_artifact_ref=artifact_refs_by_name[
            PEFT_ENCODER_SEED_ADAPTER_ARTIFACT_SLOT
        ],
        classifier_head_artifact_ref=artifact_refs_by_name[
            PEFT_ENCODER_SEED_CLASSIFIER_HEAD_ARTIFACT_SLOT
        ],
        artifact_format=artifact_format,
        peft_parameter_deltas=seed_result.peft_parameter_deltas,
        classifier_head_weight_deltas=seed_result.classifier_head_weight_deltas,
        classifier_head_bias_deltas=seed_result.classifier_head_bias_deltas,
    )
    return PeftEncoderSupervisedSeedProjection(
        next_state=projection.next_state,
        artifacts=projection.artifacts,
        metrics=seed_result.metrics,
    )


def build_peft_encoder_supervised_seed_projection_from_runtime_payload(
    *,
    adapter_state: PeftClassifierState,
    bootstrap_rows: list[LabeledQueryRow],
    aggregation_context: FederatedAggregationContext,
    runtime_payload: object,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_clip_norm: float | None,
    next_model_revision: str,
    updated_at: datetime,
    artifact_refs_by_name: Mapping[str, str],
    artifact_format: str,
) -> PeftEncoderSupervisedSeedProjection:
    """config-declared server bridge용 PEFT seed projection entrypoint."""

    peft_config = getattr(runtime_payload, "training_backend_config", None)
    if peft_config is None:
        raise TypeError(
            "PEFT supervised seed runtime payload must expose training_backend_config."
        )
    return build_peft_encoder_supervised_seed_projection(
        adapter_state=adapter_state,
        bootstrap_rows=bootstrap_rows,
        aggregation_context=aggregation_context,
        peft_config=peft_config,
        trainer_runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        next_model_revision=next_model_revision,
        updated_at=updated_at,
        artifact_refs_by_name=artifact_refs_by_name,
        artifact_format=artifact_format,
    )
