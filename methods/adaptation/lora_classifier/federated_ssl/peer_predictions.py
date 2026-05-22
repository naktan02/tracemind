"""LoRA-classifier family의 FL SSL peer prediction primitive."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.training.delta_extraction import (
    load_lora_classifier_base_parameters_into_model,
)
from methods.adaptation.lora_classifier.training.modeling import (
    LoraTextClassifier,
    build_lora_text_classifier_from_config,
)
from methods.adaptation.lora_classifier.training.query_ssl_local_training import (
    LoraClassifierTrainerRuntimeConfig,
)
from methods.adaptation.query_classifier_adaptation.data import build_weak_dataloader
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

LORA_CLASSIFIER_PEER_SNAPSHOT_KIND = "lora_classifier_materialized_state.v1"


@dataclass(frozen=True, slots=True)
class LoraClassifierHelperWeakProbabilityProvider:
    """선택된 helper LoRA-classifier snapshot으로 weak-view 확률을 계산한다."""

    helper_models: tuple[LoraTextClassifier, ...]
    device: str

    @property
    def helper_count(self) -> int:
        return len(self.helper_models)

    def __call__(
        self,
        *,
        unlabeled_batch: Mapping[str, Tensor],
    ) -> Tensor | None:
        if not self.helper_models:
            return None
        input_ids = unlabeled_batch["weak_input_ids"].to(self.device)
        attention_mask = unlabeled_batch["weak_attention_mask"].to(self.device)
        probabilities: list[Tensor] = []
        with torch.no_grad():
            for model in self.helper_models:
                model.eval()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities.append(torch.softmax(logits, dim=-1).detach())
        return torch.stack(probabilities, dim=0)


def build_lora_classifier_peer_client_snapshot(
    *,
    client_id: str,
    model: LoraTextClassifier,
    tokenizer: Any,
    probe_rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    probe_batch_size: int,
) -> FederatedSslPeerClientSnapshot | None:
    """final local model에서 helper selection vector와 reloadable state를 만든다."""

    selection_vector = compute_lora_classifier_probe_vector(
        model=model,
        tokenizer=tokenizer,
        probe_rows=probe_rows,
        lora_config=lora_config,
        trainer_runtime_config=trainer_runtime_config,
        probe_batch_size=probe_batch_size,
    )
    if selection_vector is None:
        return None
    return FederatedSslPeerClientSnapshot(
        client_id=client_id,
        selection_vector=selection_vector,
        payload_kind=LORA_CLASSIFIER_PEER_SNAPSHOT_KIND,
        payload=extract_lora_classifier_materialized_state(
            model=model,
            labels=labels,
        ),
        metadata={
            "probe_row_count": len(probe_rows),
            "probe_source": "simulation_validation_rows",
            "probability_vector": "mean_weak_view_class_probability",
        },
    )


def compute_lora_classifier_probe_vector(
    *,
    model: LoraTextClassifier,
    tokenizer: Any,
    probe_rows: Sequence[LabeledQueryRow],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    probe_batch_size: int,
) -> tuple[float, ...] | None:
    """고정 probe row에 대한 평균 class probability vector를 만든다."""

    effective_rows = list(probe_rows)
    if not effective_rows:
        return None
    loader = build_weak_dataloader(
        rows=effective_rows,
        tokenizer=tokenizer,
        batch_size=max(1, int(probe_batch_size)),
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=False,
    )
    probability_sum: Tensor | None = None
    row_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["weak_input_ids"].to(trainer_runtime_config.device)
            attention_mask = batch["weak_attention_mask"].to(
                trainer_runtime_config.device
            )
            probabilities = torch.softmax(
                model(input_ids=input_ids, attention_mask=attention_mask),
                dim=-1,
            )
            batch_sum = probabilities.detach().sum(dim=0).cpu()
            probability_sum = (
                batch_sum if probability_sum is None else probability_sum + batch_sum
            )
            row_count += int(probabilities.shape[0])
    if probability_sum is None or row_count <= 0:
        return None
    mean_probability = probability_sum / float(row_count)
    return tuple(float(value) for value in mean_probability.reshape(-1).tolist())


def extract_lora_classifier_materialized_state(
    *,
    model: LoraTextClassifier,
    labels: Sequence[str],
) -> LoraClassifierMaterializedState:
    """현재 LoRA-classifier trainable parameter를 materialized state로 만든다."""

    lora_parameters: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("classifier."):
            continue
        lora_parameters[name] = [
            float(value) for value in parameter.detach().cpu().reshape(-1).tolist()
        ]
    if not lora_parameters:
        raise ValueError("LoRA peer snapshot requires trainable LoRA parameters.")

    weight = model.classifier.weight.detach().cpu()
    bias = model.classifier.bias.detach().cpu()
    classifier_head_weights: dict[str, list[float]] = {}
    classifier_head_biases: dict[str, float] = {}
    for label_index, label in enumerate(labels):
        key = str(label)
        classifier_head_weights[key] = [
            float(value) for value in weight[label_index].reshape(-1).tolist()
        ]
        classifier_head_biases[key] = float(bias[label_index].item())

    return LoraClassifierMaterializedState(
        lora_parameters=lora_parameters,
        classifier_head_weights=classifier_head_weights,
        classifier_head_biases=classifier_head_biases,
    )


def build_lora_classifier_helper_probability_provider(
    *,
    peer_context: FederatedSslPeerContext | None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None,
    labels: Sequence[str],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
) -> LoraClassifierHelperWeakProbabilityProvider | None:
    """선택된 helper snapshot을 weak probability provider로 materialize한다."""

    if peer_context is None or peer_context.helper_count == 0 or not peer_snapshots:
        return None
    helper_models: list[LoraTextClassifier] = []
    for helper_client_id in peer_context.helper_client_ids:
        snapshot = peer_snapshots.get(helper_client_id)
        if snapshot is None:
            continue
        if snapshot.payload_kind != LORA_CLASSIFIER_PEER_SNAPSHOT_KIND:
            continue
        if not isinstance(snapshot.payload, LoraClassifierMaterializedState):
            raise TypeError(
                "LoRA-classifier helper snapshot payload must be "
                "LoraClassifierMaterializedState."
            )
        model, _tokenizer = build_lora_text_classifier_from_config(
            labels=[str(label) for label in labels],
            lora_config=lora_config,
            runtime_config=trainer_runtime_config,
        )
        load_lora_classifier_base_parameters_into_model(
            model=model,
            labels=tuple(str(label) for label in labels),
            base_parameters=snapshot.payload,
            device=trainer_runtime_config.device,
        )
        model.eval()
        helper_models.append(model)
    if not helper_models:
        return None
    return LoraClassifierHelperWeakProbabilityProvider(
        helper_models=tuple(helper_models),
        device=trainer_runtime_config.device,
    )
