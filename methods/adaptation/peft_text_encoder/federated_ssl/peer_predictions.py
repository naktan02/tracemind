"""PEFT text encoder/head FL SSL peer prediction primitive."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.resource_cache import (
    peft_encoder_resource_cache_key,
)
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    load_peft_encoder_base_parameters_into_model,
)
from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftTextEncoderWithLinearHead,
    build_peft_text_encoder_with_linear_head_from_config,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
    compact_peft_encoder_materialized_state,
)
from methods.adaptation.query_text_views.data import build_weak_dataloader
from methods.common.runtime_resources import RuntimeResourceCache
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

PeftEncoderTrainerRuntimeConfig = qssl_training.PeftEncoderTrainerRuntimeConfig

PEFT_ENCODER_PEER_SNAPSHOT_KIND = "peft_encoder_materialized_state.v1"
PEFT_ENCODER_ACCEPTED_PEER_SNAPSHOT_KINDS = (PEFT_ENCODER_PEER_SNAPSHOT_KIND,)


@dataclass(slots=True)
class PeftEncoderHelperWeakProbabilityProvider:
    """선택된 helper PEFT encoder snapshot으로 weak-view 확률을 계산한다."""

    helper_snapshots: tuple[FederatedSslPeerClientSnapshot, ...]
    labels: tuple[str, ...]
    peft_config: PeftEncoderTrainingBackendConfig
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig
    device: str
    runtime_resource_cache: RuntimeResourceCache | None = None
    _helper_models: tuple[PeftTextEncoderWithLinearHead, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def helper_count(self) -> int:
        return len(self.helper_snapshots)

    @property
    def materialized_helper_count(self) -> int:
        return 0 if self._helper_models is None else len(self._helper_models)

    @property
    def helper_models(self) -> tuple[PeftTextEncoderWithLinearHead, ...]:
        """호출 시점에만 helper model을 GPU에 materialize한다."""

        if self._helper_models is None:
            self._helper_models = tuple(
                _materialize_helper_model(
                    snapshot=snapshot,
                    labels=self.labels,
                    peft_config=self.peft_config,
                    trainer_runtime_config=self.trainer_runtime_config,
                    runtime_resource_cache=self.runtime_resource_cache,
                )
                for snapshot in self.helper_snapshots
            )
        return self._helper_models

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


def build_peft_encoder_peer_client_snapshot(
    *,
    client_id: str,
    model: PeftTextEncoderWithLinearHead,
    tokenizer: Any,
    probe_rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    probe_batch_size: int,
) -> FederatedSslPeerClientSnapshot | None:
    """final local model에서 helper selection vector와 reloadable state를 만든다."""

    selection_vector = compute_peft_encoder_probe_vector(
        model=model,
        tokenizer=tokenizer,
        probe_rows=probe_rows,
        peft_config=peft_config,
        trainer_runtime_config=trainer_runtime_config,
        probe_batch_size=probe_batch_size,
    )
    if selection_vector is None:
        return None
    return FederatedSslPeerClientSnapshot(
        client_id=client_id,
        selection_vector=selection_vector,
        payload_kind=PEFT_ENCODER_PEER_SNAPSHOT_KIND,
        payload=extract_peft_encoder_materialized_state(
            model=model,
            labels=labels,
        ),
        metadata={
            "probe_row_count": len(probe_rows),
            "probe_source": "simulation_validation_rows",
            "probability_vector": "mean_weak_view_class_probability",
        },
    )


def compute_peft_encoder_probe_vector(
    *,
    model: PeftTextEncoderWithLinearHead,
    tokenizer: Any,
    probe_rows: Sequence[LabeledQueryRow],
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
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
        max_length=peft_config.max_length,
        task_prefix=peft_config.task_prefix,
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


def extract_peft_encoder_materialized_state(
    *,
    model: PeftTextEncoderWithLinearHead,
    labels: Sequence[str],
) -> PeftEncoderMaterializedState:
    """현재 PEFT text encoder/head trainable state를 materialize한다."""

    peft_parameters: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("classifier."):
            continue
        peft_parameters[name] = [
            float(value) for value in parameter.detach().cpu().reshape(-1).tolist()
        ]
    if not peft_parameters:
        raise ValueError("PEFT peer snapshot requires trainable adapter parameters.")

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

    return compact_peft_encoder_materialized_state(
        PeftEncoderMaterializedState(
            peft_parameters=peft_parameters,
            classifier_head_weights=classifier_head_weights,
            classifier_head_biases=classifier_head_biases,
        )
    )


def build_peft_encoder_helper_probability_provider(
    *,
    peer_context: FederatedSslPeerContext | None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None,
    labels: Sequence[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> PeftEncoderHelperWeakProbabilityProvider | None:
    """선택된 helper snapshot을 weak probability provider로 materialize한다."""

    if peer_context is None or peer_context.helper_count == 0 or not peer_snapshots:
        return None
    helper_snapshots: list[FederatedSslPeerClientSnapshot] = []
    for helper_client_id in peer_context.helper_client_ids:
        snapshot = peer_snapshots.get(helper_client_id)
        if snapshot is None:
            continue
        if snapshot.payload_kind not in PEFT_ENCODER_ACCEPTED_PEER_SNAPSHOT_KINDS:
            continue
        if not isinstance(snapshot.payload, PeftEncoderMaterializedState):
            raise TypeError(
                "PEFT text encoder/head helper snapshot payload must be "
                "PeftEncoderMaterializedState."
            )
        helper_snapshots.append(snapshot)
    if not helper_snapshots:
        return None
    return PeftEncoderHelperWeakProbabilityProvider(
        helper_snapshots=tuple(helper_snapshots),
        labels=tuple(str(label) for label in labels),
        peft_config=peft_config,
        trainer_runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
        device=trainer_runtime_config.device,
    )


def _materialize_helper_model(
    *,
    snapshot: FederatedSslPeerClientSnapshot,
    labels: tuple[str, ...],
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
) -> PeftTextEncoderWithLinearHead:
    if not isinstance(snapshot.payload, PeftEncoderMaterializedState):
        raise TypeError("snapshot payload must be PeftEncoderMaterializedState.")
    cache_key = _helper_model_cache_key(
        snapshot=snapshot,
        labels=labels,
        peft_config=peft_config,
        trainer_runtime_config=trainer_runtime_config,
    )
    if runtime_resource_cache is not None:
        cached = runtime_resource_cache.get_resource(cache_key)
        if cached is not None:
            if not isinstance(cached, PeftTextEncoderWithLinearHead):
                raise TypeError(
                    "Cached helper model must be PeftTextEncoderWithLinearHead."
                )
            return cached

    model, _tokenizer = build_peft_text_encoder_with_linear_head_from_config(
        labels=list(labels),
        peft_config=peft_config,
        runtime_config=trainer_runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    load_peft_encoder_base_parameters_into_model(
        model=model,
        labels=labels,
        base_parameters=snapshot.payload,
        device=trainer_runtime_config.device,
    )
    model.eval()
    if runtime_resource_cache is not None:
        runtime_resource_cache.set_resource(cache_key, model)
    return model


def _helper_model_cache_key(
    *,
    snapshot: FederatedSslPeerClientSnapshot,
    labels: tuple[str, ...],
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
) -> str:
    if not isinstance(snapshot.payload, PeftEncoderMaterializedState):
        raise TypeError("snapshot payload must be PeftEncoderMaterializedState.")
    payload = {
        "client_id": snapshot.client_id,
        "payload_hash": _materialized_state_hash(snapshot.payload),
        "labels": labels,
        "backbone": peft_config.to_backbone_payload(),
        "peft_adapter_config": peft_config.to_peft_adapter_config_payload(),
        "device": trainer_runtime_config.device,
        "classifier_dropout": trainer_runtime_config.classifier_dropout,
    }
    return peft_encoder_resource_cache_key(kind="helper_model", values=payload)


def _materialized_state_hash(
    state: PeftEncoderMaterializedState,
) -> str:
    payload = {
        "peft_parameters": _sorted_numeric_mapping(state.peft_parameters),
        "classifier_head_weights": _sorted_numeric_mapping(
            state.classifier_head_weights
        ),
        "classifier_head_biases": dict(sorted(state.classifier_head_biases.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sorted_numeric_mapping(
    value: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    return {key: list(items) for key, items in sorted(value.items())}
