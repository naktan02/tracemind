"""LoRA-classifier FL SSL helper probability provider resolution."""

from __future__ import annotations

from collections.abc import Mapping

from methods.adaptation.text_classifier.peft_encoder.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.text_classifier.peft_encoder.federated_ssl import (
    peer_predictions,
)
from methods.adaptation.text_classifier.peft_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated_ssl.local_objective import (
    requires_method_helper_probability_provider,
)
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)

LoraClassifierTrainerRuntimeConfig = qssl_training.LoraClassifierTrainerRuntimeConfig
build_lora_classifier_helper_probability_provider = (
    peer_predictions.build_lora_classifier_helper_probability_provider
)


def build_lora_classifier_helper_provider_for_local_ssl_policy(
    *,
    method_name: str,
    local_ssl_policy_name: str,
    peer_context: FederatedSslPeerContext | None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None,
    labels: tuple[str, ...],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: LoraClassifierTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
    timing_recorder: TimingRecorder | None,
) -> object | None:
    """local_ssl_policy가 helper model 확률을 요구하면 provider를 만든다."""

    if not requires_method_helper_probability_provider(
        method_name=method_name,
        local_ssl_policy_name=local_ssl_policy_name,
    ):
        return None
    if timing_recorder is None:
        return build_lora_classifier_helper_probability_provider(
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            labels=labels,
            lora_config=lora_config,
            trainer_runtime_config=trainer_runtime_config,
            runtime_resource_cache=runtime_resource_cache,
        )
    with timing_recorder.measure("adapter_helper_provider_prepare_seconds"):
        return build_lora_classifier_helper_probability_provider(
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            labels=labels,
            lora_config=lora_config,
            trainer_runtime_config=trainer_runtime_config,
            runtime_resource_cache=runtime_resource_cache,
        )
