"""Method-owned PEFT-classifier local training core resolver."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

from methods.adaptation.peft_text_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.peft_text_classifier.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_classifier.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated_ssl.peer_context import FederatedSslPeerContext
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

PeftEncoderTrainerRuntimeConfig = qssl_training.PeftEncoderTrainerRuntimeConfig
QuerySslPeftEncoderClientTrainingResult = (
    qssl_training.QuerySslPeftEncoderClientTrainingResult
)
QuerySslPeftEncoderDeltaMaterializer = (
    qssl_training.QuerySslPeftEncoderDeltaMaterializer
)
QuerySslPeftEncoderObjectiveRuntimeConfig = (
    qssl_training.QuerySslPeftEncoderObjectiveRuntimeConfig
)


class FederatedSslMethodLocalTrainingConfig(Protocol):
    """method-owned PEFT local core가 필요한 method config surface."""

    name: str
    scenario: str | None
    effective_parameters: Mapping[str, object]


MethodOwnedPeftEncoderTrainingCore = Callable[
    ...,
    QuerySslPeftEncoderClientTrainingResult,
]


def resolve_method_owned_peft_encoder_training_core(
    method_name: str,
) -> MethodOwnedPeftEncoderTrainingCore:
    """method descriptor의 명시 entrypoint로 PEFT local training core를 resolve한다."""

    normalized_name = method_name.strip().lower().replace("-", "_")
    if not normalized_name:
        raise ValueError("method_name must not be empty.")
    descriptor = resolve_federated_ssl_method_descriptor(normalized_name)
    entrypoint = descriptor.local_step.runtime_entrypoint
    if entrypoint is None:
        raise NotImplementedError(
            "Method-owned PEFT-classifier local training core is not declared: "
            f"{method_name}"
        )
    return _load_method_owned_peft_encoder_training_core(entrypoint)


def _load_method_owned_peft_encoder_training_core(
    entrypoint: str,
) -> MethodOwnedPeftEncoderTrainingCore:
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name.strip() or not function_name.strip():
        raise ValueError(
            "method-owned PEFT-classifier runtime_entrypoint must use "
            "'module:function' format."
        )
    try:
        module = importlib.import_module(module_name.strip())
    except ModuleNotFoundError as error:
        if error.name == module_name.strip():
            raise NotImplementedError(
                "Method-owned PEFT-classifier local training core module is not wired: "
                f"{module_name}"
            ) from error
        raise

    core = getattr(module, function_name.strip(), None)
    if core is None:
        raise NotImplementedError(
            "Method-owned PEFT-classifier local training core function is missing: "
            f"{entrypoint}"
        )
    return core


def run_method_owned_peft_encoder_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    base_partition_parameters: Mapping[str, PeftEncoderMaterializedState] | None = None,
    previous_client_partition_parameters: Mapping[str, PeftEncoderMaterializedState]
    | None = None,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: FederatedSslMethodLocalTrainingConfig,
    local_ssl_policy_name: str,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig | None,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer,
    peer_context: FederatedSslPeerContext | None = None,
    helper_weak_probability_provider: object | None = None,
    peer_probe_rows: Sequence[LabeledQueryRow] | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """선택된 method-owned PEFT-classifier local training core를 실행한다."""

    core = resolve_method_owned_peft_encoder_training_core(ssl_method_config.name)
    return core(
        client_id=client_id,
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
        base_partition_parameters=base_partition_parameters,
        previous_client_partition_parameters=previous_client_partition_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
        ssl_method_config=ssl_method_config,
        local_ssl_policy_name=local_ssl_policy_name,
        query_ssl_config=query_ssl_config,
        peer_context=peer_context,
        strong_view_policy=strong_view_policy,
        unlabeled_batch_size=unlabeled_batch_size,
        lora_config=lora_config,
        trainer_runtime_config=trainer_runtime_config,
        created_at=created_at,
        delta_materializer=delta_materializer,
        helper_weak_probability_provider=helper_weak_probability_provider,
        peer_probe_rows=peer_probe_rows,
        runtime_resource_cache=runtime_resource_cache,
        timing_recorder=timing_recorder,
        initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
    )
