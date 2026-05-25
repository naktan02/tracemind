"""Local SSL policy별 helper probability provider resolver."""

from __future__ import annotations

from collections.abc import Mapping

from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.federated_ssl.peer_predictions import (
    build_lora_classifier_helper_probability_provider,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from methods.federated_ssl.capability_axes import LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedLocalTrainerRuntimeConfig,
)


def build_lora_classifier_helper_provider_for_local_ssl_policy(
    *,
    local_ssl_policy_name: str,
    peer_context: FederatedSslPeerContext | None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None,
    labels: tuple[str, ...],
    lora_config: LoraClassifierTrainingBackendConfig,
    trainer_runtime_config: FederatedLocalTrainerRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
    timing_recorder: TimingRecorder | None,
) -> object | None:
    """local_ssl_policy가 helper model 확률을 요구하면 provider를 만든다."""

    normalized_policy = local_ssl_policy_name.strip().lower().replace("-", "_")
    if normalized_policy != LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT:
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
