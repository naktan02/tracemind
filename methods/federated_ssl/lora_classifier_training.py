"""Method-owned LoRA-classifier local training core resolver."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    LoraClassifierTrainerRuntimeConfig,
    QuerySslLoraClientTrainingResult,
    QuerySslLoraDeltaMaterializer,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask


class FederatedSslMethodLocalTrainingConfig(Protocol):
    """method-owned LoRA local core가 필요한 method config surface."""

    name: str
    scenario: str | None
    effective_parameters: Mapping[str, object]


MethodOwnedLoraClassifierTrainingCore = Callable[
    ...,
    QuerySslLoraClientTrainingResult,
]


def resolve_method_owned_lora_classifier_training_core(
    method_name: str,
) -> MethodOwnedLoraClassifierTrainingCore:
    """method 이름으로 LoRA-classifier local training core를 convention resolve한다."""

    normalized_name = method_name.strip().lower().replace("-", "_")
    if not normalized_name:
        raise ValueError("method_name must not be empty.")
    module_name = f"methods.federated_ssl.{normalized_name}.lora_classifier_training"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise NotImplementedError(
                "Method-owned LoRA-classifier local training core is not wired: "
                f"{method_name}"
            ) from error
        raise

    core = getattr(module, "run_method_owned_lora_classifier_training_core", None)
    if core is None:
        raise NotImplementedError(
            "Method-owned LoRA-classifier local training core is missing standard "
            f"entrypoint in {module_name}."
        )
    return core


def run_method_owned_lora_classifier_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    base_parameters: LoraClassifierMaterializedState,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: FederatedSslMethodLocalTrainingConfig,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslLoraDeltaMaterializer,
) -> QuerySslLoraClientTrainingResult:
    """선택된 method-owned LoRA-classifier local training core를 실행한다."""

    core = resolve_method_owned_lora_classifier_training_core(ssl_method_config.name)
    return core(
        client_id=client_id,
        seed=seed,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
        ssl_method_config=ssl_method_config,
        strong_view_policy=strong_view_policy,
        unlabeled_batch_size=unlabeled_batch_size,
        lora_config=lora_config,
        trainer_runtime_config=trainer_runtime_config,
        created_at=created_at,
        delta_materializer=delta_materializer,
    )
