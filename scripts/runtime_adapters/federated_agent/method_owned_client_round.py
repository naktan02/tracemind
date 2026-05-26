"""FL simulation method-owned client-round runtime bridge."""

from __future__ import annotations

import gc
import sys
import time
from collections.abc import Mapping
from typing import Any

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.update.delta_artifacts import (
    server_owned_lora_classifier_update_artifact_byte_count,
    upload_agent_local_lora_classifier_update,
)
from methods.common.timing import TimingRecorder
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    client_update_submission,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.diagnostic_view import (
    build_client_diagnostic_unlabeled_view,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
    ClientRoundExecution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    FederatedClientShard,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_agent import local_training as method_trainer
from scripts.runtime_adapters.federated_agent.artifact_store import (
    SimulationClientArtifactStore,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
)
from shared.src.contracts.training_contracts import ClientMetricKeys


def run_method_owned_client_round_if_supported(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    capability_plan: FederatedSslCapabilityPlan,
    peer_context: FederatedSslPeerContext | None = None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None = None,
    previous_client_partition_parameters: (
        Mapping[str, LoraClassifierMaterializedState] | None
    ) = None,
) -> ClientRoundExecution | None:
    """method-owned LoRA raw-row training이 가능한 조합이면 실행한다."""

    if not _supports_method_owned_lora_client_training(request):
        return None
    return _run_method_owned_lora_client_round(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_id=round_id,
        shard=shard,
        training_task=training_task,
        capability_plan=capability_plan,
        peer_context=peer_context,
        peer_snapshots=peer_snapshots,
        previous_client_partition_parameters=previous_client_partition_parameters,
    )


def _supports_method_owned_lora_client_training(
    request: SimulationRunRequest,
) -> bool:
    return (
        request.ssl_method_config is not None
        and str(request.round_runtime_config.adapter_family_name).strip().lower()
        == LORA_CLASSIFIER_ADAPTER_KIND
        and request.round_runtime_config.lora_classifier is not None
    )


def _run_method_owned_lora_client_round(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_id: str,
    shard: FederatedClientShard,
    training_task: Any,
    capability_plan: FederatedSslCapabilityPlan,
    peer_context: FederatedSslPeerContext | None = None,
    peer_snapshots: Mapping[str, FederatedSslPeerClientSnapshot] | None = None,
    previous_client_partition_parameters: (
        Mapping[str, LoraClassifierMaterializedState] | None
    ) = None,
) -> ClientRoundExecution:
    if request.ssl_method_config is None:
        raise ValueError("ssl_method_config is required.")
    if request.round_runtime_config.lora_classifier is None:
        raise ValueError("LoRA-classifier runtime config is required.")

    query_ssl_config = request.query_ssl_objective_config
    timing = TimingRecorder()
    training_started_at = time.perf_counter()
    with timing.measure("diagnostic_view_select_seconds"):
        diagnostic_unlabeled_rows = build_client_diagnostic_unlabeled_view(
            rows=shard.unlabeled_rows,
            config=request.diagnostic_view_config,
            run_seed=request.seed,
            round_index=_round_index_from_id(round_id),
            client_id=shard.client_id,
        )
    with timing.measure("local_training_total_seconds"):
        local_result = method_trainer.run_method_owned_lora_classifier_local_training(
            client_id=shard.client_id,
            seed=request.seed,
            output_dir=request.output_dir,
            labeled_rows=shard.labeled_rows,
            unlabeled_rows=shard.unlabeled_rows,
            diagnostic_unlabeled_rows=diagnostic_unlabeled_rows,
            active_adapter_state=active.adapter_state,
            training_task=training_task,
            model_manifest=active.manifest,
            ssl_method_config=request.ssl_method_config,
            local_ssl_policy_name=capability_plan.local_ssl_policy_name,
            query_ssl_config=request.query_ssl_objective_config,
            peer_context=peer_context,
            peer_snapshots=peer_snapshots,
            previous_client_partition_parameters=previous_client_partition_parameters,
            runtime_resource_cache=bootstrapped.runtime_resource_cache,
            round_base_snapshot_cache=bootstrapped.round_base_snapshot_cache,
            peer_probe_rows=(
                bootstrapped.peer_probe_rows
                if bootstrapped.peer_probe_rows
                else tuple(request.validation_rows)
            ),
            strong_view_policy=(
                "first_aug"
                if query_ssl_config is None
                else query_ssl_config.strong_view_policy
            ),
            unlabeled_batch_size=(
                None
                if query_ssl_config is None
                else query_ssl_config.unlabeled_batch_size
            ),
            trainer_runtime_config=request.local_trainer_runtime_config,
            timing_recorder=timing,
            persist_agent_local_update=(
                request.artifact_persistence_config.persist_agent_local_updates
            ),
        )
    with timing.measure("helper_model_cache_release_seconds"):
        _release_helper_model_cache(bootstrapped.runtime_resource_cache)
    client_train_time_seconds = time.perf_counter() - training_started_at
    artifact_store = SimulationClientArtifactStore(output_dir=request.output_dir)
    with timing.measure("update_upload_materialize_seconds"):
        server_update_payload = upload_agent_local_lora_classifier_update(
            artifact_store=artifact_store,
            update_payload=local_result.update_payload,
        )
    with timing.measure("server_update_submit_seconds"):
        update_submitted = client_update_submission.accept_client_update(
            server_runtime=bootstrapped.server_runtime,
            round_id=round_id,
            update_envelope=local_result.update_envelope,
            update_payload=server_update_payload,
        )
    pseudo_label_quality = local_result.pseudo_label_quality
    return ClientRoundExecution(
        summary=ClientRoundSummary(
            client_id=shard.client_id,
            candidate_count=local_result.candidate_count,
            diagnostic_candidate_count=len(diagnostic_unlabeled_rows),
            accepted_count=local_result.accepted_count,
            update_generated=update_submitted,
            delta_l2_norm=client_update_submission.extract_delta_l2_norm(
                local_result.update_envelope
            ),
            aggregation_example_count=(
                client_update_submission.extract_aggregation_example_count(
                    local_result.update_envelope
                )
            ),
            client_train_time_seconds=client_train_time_seconds,
            client_payload_bytes=(
                client_update_submission.payload_byte_count(server_update_payload)
                if update_submitted
                else None
            ),
            client_artifact_bytes=(
                server_owned_lora_classifier_update_artifact_byte_count(
                    artifact_store=artifact_store,
                    update_payload=server_update_payload,
                )
                if update_submitted
                else None
            ),
            pseudo_label_confidence_mean=(
                pseudo_label_quality.pseudo_label_confidence_mean
                if pseudo_label_quality.pseudo_label_confidence_mean is not None
                else local_result.client_metrics.get(ClientMetricKeys.MEAN_CONFIDENCE)
            ),
            pseudo_label_margin_mean=(
                pseudo_label_quality.pseudo_label_margin_mean
                if pseudo_label_quality.pseudo_label_margin_mean is not None
                else local_result.client_metrics.get(ClientMetricKeys.MEAN_MARGIN)
            ),
            pseudo_label_correct_count=(
                pseudo_label_quality.pseudo_label_correct_count
            ),
            pseudo_label_evaluated_count=(
                pseudo_label_quality.pseudo_label_evaluated_count
            ),
            accepted_label_distribution=(
                pseudo_label_quality.accepted_label_distribution
            ),
            rejected_label_distribution=(
                pseudo_label_quality.rejected_label_distribution
            ),
            fedmatch_helper_count=_optional_float_metric(
                local_result.client_metrics.get("fedmatch_helper_count")
            ),
            fedmatch_peer_context_helper_count=_optional_float_metric(
                local_result.client_metrics.get("fedmatch_peer_context_helper_count")
            ),
            fedmatch_helper_provider_count=_optional_float_metric(
                local_result.client_metrics.get("fedmatch_helper_provider_count")
            ),
            fedmatch_missing_helper_snapshot_count=_optional_float_metric(
                local_result.client_metrics.get(
                    "fedmatch_missing_helper_snapshot_count"
                )
            ),
            fedmatch_materialized_helper_model_count=_optional_float_metric(
                local_result.client_metrics.get(
                    "fedmatch_materialized_helper_model_count"
                )
            ),
            fedmatch_peer_context_refreshed=_optional_float_metric(
                local_result.client_metrics.get("fedmatch_peer_context_refreshed")
            ),
            fedmatch_c2s_sparse_upload_value_count=_optional_float_metric(
                local_result.client_metrics.get(
                    "fedmatch_c2s_sparse_upload_value_count"
                )
            ),
            fedmatch_s2c_sparse_download_value_count=_optional_float_metric(
                local_result.client_metrics.get(
                    "fedmatch_s2c_sparse_download_value_count"
                )
            ),
            timing_breakdown=timing.to_mapping(),
        ),
        update_submitted=update_submitted,
        peer_client_snapshot=local_result.peer_client_snapshot,
        client_partition_snapshot=local_result.client_partition_parameters,
    )


def _round_index_from_id(round_id: str) -> int:
    return int(round_id.rsplit("_", maxsplit=1)[-1])


def _optional_float_metric(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _release_helper_model_cache(runtime_resource_cache: object | None) -> int:
    """client 경계에서 FedMatch helper model materialization만 폐기한다."""

    clear_resources = getattr(runtime_resource_cache, "clear_resources", None)
    removed = 0
    if callable(clear_resources):
        removed = int(clear_resources(key_prefix="lora_classifier:helper_model:"))
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
