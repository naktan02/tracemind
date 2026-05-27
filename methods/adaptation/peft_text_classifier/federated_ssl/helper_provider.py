"""PEFT-encoder classifier FL SSL helper probability provider resolution."""

from __future__ import annotations

from collections.abc import Mapping

from methods.adaptation.peft_text_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.peft_text_classifier.federated_ssl import (
    peer_predictions,
)
from methods.adaptation.peft_text_classifier.training import (
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

PeftEncoderTrainerRuntimeConfig = qssl_training.PeftEncoderTrainerRuntimeConfig
build_lora_classifier_helper_probability_provider = (
    peer_predictions.build_peft_encoder_helper_probability_provider
)
build_peft_encoder_helper_probability_provider = (
    peer_predictions.build_peft_encoder_helper_probability_provider
)


def build_peft_encoder_helper_provider_for_local_ssl_policy(
    *,
    method_name: str,
    local_ssl_policy_name: str,
    peer_context: FederatedSslPeerContext | None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None,
    labels: tuple[str, ...],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
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
        return build_peft_encoder_helper_probability_provider(
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            labels=labels,
            lora_config=lora_config,
            trainer_runtime_config=trainer_runtime_config,
            runtime_resource_cache=runtime_resource_cache,
        )

    with timing_recorder.measure("adapter_helper_provider_prepare_seconds"):
        return build_peft_encoder_helper_probability_provider(
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            labels=labels,
            lora_config=lora_config,
            trainer_runtime_config=trainer_runtime_config,
            runtime_resource_cache=runtime_resource_cache,
        )


build_lora_classifier_helper_provider_for_local_ssl_policy = (
    build_peft_encoder_helper_provider_for_local_ssl_policy
)
