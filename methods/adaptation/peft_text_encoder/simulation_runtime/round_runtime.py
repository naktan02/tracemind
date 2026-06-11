"""PEFT encoder round-runtime config surface for FL simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.federated_ssl.method_training_surface import (
    FsslPeftEncoderMethodTrainingRequest,
)
from methods.adaptation.peft_text_encoder.training.query_ssl_local_training import (
    QuerySslPeftEncoderClientTrainingResult,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask


@dataclass(slots=True)
class FederatedPeftEncoderRuntimeConfig:
    """PEFT text encoder/head simulation bootstrap에 필요한 fixed scaffold snapshot."""

    training_backend_config: PeftEncoderTrainingBackendConfig
    artifact_format: str = "simulation_peft_classifier_state_ref"
    peft_adapter_artifact_ref: str | None = None
    classifier_head_artifact_ref: str | None = None

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
        *,
        default_artifact_format: str = "simulation_peft_classifier_state_ref",
    ) -> "FederatedPeftEncoderRuntimeConfig":
        """Hydra round_runtime classifier mapping을 typed config로 해석한다."""

        artifact_format = str(
            source.get("artifact_format", default_artifact_format)
        ).strip()
        if not artifact_format:
            raise ValueError("round_runtime classifier artifact_format invalid.")
        return cls(
            training_backend_config=PeftEncoderTrainingBackendConfig.from_mapping(
                {
                    key: value
                    for key, value in source.items()
                    if key not in _PEFT_ENCODER_RUNTIME_ARTIFACT_KEYS
                }
            ),
            artifact_format=artifact_format,
            peft_adapter_artifact_ref=_optional_str(
                source.get("peft_adapter_artifact_ref")
            ),
            classifier_head_artifact_ref=_optional_str(
                source.get("classifier_head_artifact_ref")
            ),
        )

    def backbone_payload(self) -> dict[str, str | int]:
        """shared PEFT text encoder/head state에 넣을 backbone/tokenizer snapshot."""

        return self.training_backend_config.to_backbone_payload()

    def peft_adapter_config_payload(self) -> dict[str, object]:
        """shared peft_classifier state에 넣을 PEFT mechanism config snapshot."""

        return self.training_backend_config.to_peft_adapter_config_payload()


def build_peft_encoder_round_runtime_payloads(
    round_runtime_mapping: Mapping[str, object],
) -> dict[str, object]:
    """Hydra round_runtime mapping에서 PEFT text encoder/head payload를 만든다."""

    payload_key = _runtime_payload_key(round_runtime_mapping)
    runtime_payloads = _required_runtime_payloads(round_runtime_mapping)
    peft_config = _required_runtime_payload_config(
        runtime_payloads=runtime_payloads,
        payload_key=payload_key,
    )
    return {
        payload_key: FederatedPeftEncoderRuntimeConfig.from_mapping(
            peft_config,
            default_artifact_format="simulation_peft_classifier_state_ref",
        )
    }


_PEFT_ENCODER_RUNTIME_ARTIFACT_KEYS = frozenset(
    {
        "artifact_format",
        "peft_adapter_artifact_ref",
        "classifier_head_artifact_ref",
    }
)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _runtime_payload_key(round_runtime_mapping: Mapping[str, object]) -> str:
    value = _optional_str(round_runtime_mapping.get("runtime_payload_key"))
    if value is None:
        value = _optional_str(round_runtime_mapping.get("update_family_name"))
    if value is None:
        raise ValueError(
            "round_runtime must define runtime_payload_key or update_family_name."
        )
    return value.lower().replace("-", "_")


def _required_runtime_payloads(
    round_runtime_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    runtime_payloads = _optional_mapping(round_runtime_mapping.get("runtime_payloads"))
    if runtime_payloads is None:
        raise ValueError("round_runtime.runtime_payloads must define PEFT payloads.")
    return runtime_payloads


def _required_runtime_payload_config(
    *,
    runtime_payloads: Mapping[str, object],
    payload_key: str,
) -> Mapping[str, object]:
    config = _optional_mapping(runtime_payloads.get(payload_key))
    if config is None:
        raise ValueError(
            f"round_runtime.runtime_payloads must include payload key: {payload_key!r}."
        )
    return config


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("round_runtime PEFT payload config must be a mapping.")
    return value


def release_transient_model_cache(runtime_resource_cache: object | None) -> int:
    """client 경계에서 FedMatch helper model materialization cache를 폐기한다."""

    import gc

    from methods.adaptation.peft_text_encoder.resource_cache import (
        clear_peft_encoder_helper_model_cache,
    )

    removed = (
        0
        if runtime_resource_cache is None
        else clear_peft_encoder_helper_model_cache(runtime_resource_cache)
    )
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency guard
        return removed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _trim_process_allocator()
    return removed


def _trim_process_allocator() -> None:
    """Linux allocator가 해제된 CPU tensor memory를 OS에 반환하도록 요청한다."""

    import sys

    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if callable(malloc_trim):
            malloc_trim(0)
    except Exception:
        return


def run_method_owned_peft_encoder_local_training_core(
    *,
    client_id: str,
    seed: int,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    diagnostic_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    active_adapter_state: object,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    ssl_method_config: Any,
    local_ssl_policy_name: str,
    query_ssl_config: Any | None,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    trainer_runtime_config: Any,
    peer_context: Any | None = None,
    peer_snapshots: Mapping[str, Any] | None = None,
    peer_probe_rows: Sequence[LabeledQueryRow] | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    created_at: datetime | None = None,
    base_parameters: PeftEncoderMaterializedState,
    base_partition_parameters: Mapping[str, PeftEncoderMaterializedState],
    previous_client_partition_parameters: (
        Mapping[str, PeftEncoderMaterializedState] | None
    ) = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
    timing_recorder: TimingRecorder | None = None,
    delta_materializer: Any,
) -> QuerySslPeftEncoderClientTrainingResult:
    """PEFT text encoder local training을 methods 레이어 안에서 실행한다."""

    from methods.adaptation.peft_text_encoder.federated_ssl import (
        helper_provider,
        method_owned_training,
    )
    from methods.adaptation.peft_text_encoder.update_family_runtime import (
        build_training_backend_config_for_peft_encoder_state,
    )
    from shared.src.contracts.adapter_contract_families.peft_classifier import (
        PeftClassifierState,
    )

    if not isinstance(active_adapter_state, PeftClassifierState):
        raise ValueError(
            "Method-owned PEFT text encoder local training requires active PEFT "
            "text encoder state."
        )
    effective_created_at = created_at or datetime.now(tz=timezone.utc)
    effective_peft_config = build_training_backend_config_for_peft_encoder_state(
        active_adapter_state=active_adapter_state,
        objective_config=training_task.objective_config,
    )
    labels = tuple(str(label) for label in active_adapter_state.label_schema)
    helper_weak_probability_provider = (
        helper_provider.build_peft_encoder_helper_provider_for_local_ssl_policy(
            method_name=ssl_method_config.name,
            local_ssl_policy_name=local_ssl_policy_name,
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            labels=labels,
            peft_config=effective_peft_config,
            trainer_runtime_config=trainer_runtime_config,
            runtime_resource_cache=runtime_resource_cache,
            timing_recorder=timing_recorder,
        )
    )
    return method_owned_training.run_method_owned_peft_encoder_training_request(
        FsslPeftEncoderMethodTrainingRequest(
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
            peft_config=effective_peft_config,
            trainer_runtime_config=trainer_runtime_config,
            created_at=effective_created_at,
            delta_materializer=delta_materializer,
            helper_weak_probability_provider=helper_weak_probability_provider,
            peer_probe_rows=peer_probe_rows,
            runtime_resource_cache=runtime_resource_cache,
            timing_recorder=timing_recorder,
            initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
        )
    )
