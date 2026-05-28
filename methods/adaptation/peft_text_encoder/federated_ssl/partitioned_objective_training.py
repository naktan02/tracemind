"""PEFT text-classifier family의 partitioned objective training bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from methods.adaptation.peft_text_encoder.aggregation import (
    peft_encoder_partitioned_state as ps,
)
from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.federated_ssl import (
    peer_predictions,
)
from methods.adaptation.peft_text_encoder.training import (
    pseudo_label_diagnostics as pld,
)
from methods.adaptation.peft_text_encoder.training import (
    query_ssl_local_training as qssl_training,
)
from methods.adaptation.peft_text_encoder.training.delta_extraction import (
    extract_peft_encoder_parameter_deltas,
    load_peft_encoder_base_parameters_into_model,
)
from methods.adaptation.peft_text_encoder.training.loops import set_seed
from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftEncoderTextClassifier,
    build_peft_encoder_text_classifier_from_config,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
    compact_peft_encoder_materialized_state,
)
from methods.adaptation.peft_text_encoder.update.partitioned_delta import (
    PeftEncoderPartitionDelta,
)
from methods.adaptation.peft_text_encoder.update.query_ssl_update import (
    build_query_ssl_peft_encoder_update_payload,
)
from methods.adaptation.query_text_views.data import (
    build_dataloader,
    build_multiview_dataloader,
)
from methods.adaptation.query_text_views.local_training_budget import (
    LOCAL_BUDGET_POLICY_ORIGINAL_METHOD,
)
from methods.adaptation.query_text_views.tokenization import (
    resolve_text_tokenization_cache,
)
from methods.adaptation.query_text_views.view_rows import (
    USB_MULTIVIEW_BUILDER_NAME,
    validate_query_ssl_unlabeled_views,
)
from methods.common.runtime_resources import RuntimeResourceCache
from methods.common.timing import TimingRecorder, timing_mapping
from methods.federated_ssl.capability_axes import (
    LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT,
    LOCAL_SSL_POLICY_FIXMATCH,
)
from methods.federated_ssl.local_supervision import (
    FederatedSslLocalSupervisionRegime,
    require_rows_match_local_supervision_regime,
)
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)
from methods.ssl.base import (
    QuerySslAlgorithm,
    configure_query_ssl_algorithm_dataset,
    configure_query_ssl_algorithm_training,
)
from methods.ssl.registry import resolve_query_ssl_algorithm_descriptor
from methods.ssl.state import (
    export_query_ssl_algorithm_state,
    load_query_ssl_algorithm_state,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT,
    PeftClassifierDelta,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    make_training_update_envelope,
)

from .partitioned.budget import (
    normalize_partitioned_local_budget_policy,
    resolve_partitioned_local_budget,
)
from .partitioned.model_builder import (
    build_partitioned_peft_encoder_text_classifier_from_config,
)
from .partitioned.sparse_sync import (
    PartitionSparseSyncParameters,
    apply_partitioned_s2c_sparse_download,
    count_partition_delta_nonzero_values,
    project_partitioned_c2s_sparse_upload,
    project_partitioned_s2c_sparse_download,
)
from .partitioned.training_loop import (
    HelperWeakProbabilityProvider,
    PartitionedAdapterClassifierTrainingResult,
    train_partitioned_adapter_classifier,
    train_physical_partitioned_adapter_classifier,
)

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
PseudoLabelDiagnosticThreshold = pld.PseudoLabelDiagnosticThreshold
build_final_snapshot_pseudo_label_quality = (
    pld.build_final_snapshot_pseudo_label_quality
)
resolve_fixed_pseudo_label_diagnostic_threshold = (
    pld.resolve_fixed_pseudo_label_diagnostic_threshold
)
tokenization_cache_namespace = pld.tokenization_cache_namespace


class PartitionedMethodLocalTrainingConfig(Protocol):
    """partitioned local trainer가 읽는 method config surface."""

    name: str
    scenario: str | None
    local_budget_policy: str
    effective_parameters: Mapping[str, object]


class PartitionedLocalRuntimePlan(Protocol):
    """partitioned PEFT text-classifier runtime에 주입되는 method-owned plan."""

    scenario_name: str
    local_supervision_regime: FederatedSslLocalSupervisionRegime
    physical_objective: Any
    sequential_objective: Any
    partition_names: tuple[str, ...]
    supervised_partition: str
    unsupervised_partition: str
    upload_partitions: tuple[str, ...]
    l1_sparse_partitions: tuple[str, ...]
    psi_factor: float
    metric_prefix: str
    diagnostic_acceptance_threshold: float
    emit_supervised_partition: bool


def run_partitioned_peft_encoder_training_core(
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
    ssl_method_config: PartitionedMethodLocalTrainingConfig,
    partitioned_runtime_plan: PartitionedLocalRuntimePlan,
    local_ssl_policy_name: str,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig | None,
    strong_view_policy: str,
    unlabeled_batch_size: int | None,
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    created_at: datetime,
    delta_materializer: QuerySslPeftEncoderDeltaMaterializer,
    peer_context: FederatedSslPeerContext | None = None,
    helper_weak_probability_provider: HelperWeakProbabilityProvider | None = None,
    peer_probe_rows: Sequence[LabeledQueryRow] | None = None,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    timing_recorder: TimingRecorder | None = None,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslPeftEncoderClientTrainingResult:
    """method-owned partitioned objective를 PEFT text-classifier update로 실행한다."""

    effective_labeled_rows = list(labeled_rows)
    effective_unlabeled_rows = list(unlabeled_rows)
    require_rows_match_local_supervision_regime(
        regime=partitioned_runtime_plan.local_supervision_regime,
        labeled_rows=effective_labeled_rows,
        unlabeled_rows=effective_unlabeled_rows,
        context=(
            f"{ssl_method_config.name} {partitioned_runtime_plan.scenario_name} "
            "local runtime"
        ),
    )

    validate_query_ssl_unlabeled_views(
        rows=effective_unlabeled_rows,
        view_builder_name=USB_MULTIVIEW_BUILDER_NAME,
        algorithm_name=str(ssl_method_config.name),
    )
    effective_labels = tuple(str(label) for label in labels)
    if not effective_labels:
        raise ValueError("Partitioned PEFT text encoder label schema is empty.")
    if partitioned_runtime_plan.local_supervision_regime.uses_client_labeled_rows:
        _validate_labeled_rows_have_known_labels(
            rows=effective_labeled_rows,
            labels=effective_labels,
        )

    tokenization_cache = resolve_text_tokenization_cache(runtime_resource_cache)
    tokenization_cache_namespace_value = tokenization_cache_namespace(peft_config)
    with _measure(timing_recorder, "core_model_prepare_seconds"):
        with _measure(timing_recorder, "core_seed_seconds"):
            set_seed(int(seed))
        with _measure(timing_recorder, "core_model_build_seconds"):
            model, tokenizer = build_peft_encoder_text_classifier_from_config(
                labels=list(effective_labels),
                peft_config=peft_config,
                runtime_config=trainer_runtime_config,
                runtime_resource_cache=runtime_resource_cache,
            )
        with _measure(timing_recorder, "core_base_parameter_load_seconds"):
            load_peft_encoder_base_parameters_into_model(
                model=model,
                labels=effective_labels,
                base_parameters=base_parameters,
                device=trainer_runtime_config.device,
            )

    local_budget_policy = normalize_partitioned_local_budget_policy(
        ssl_method_config.local_budget_policy
    )
    with _measure(timing_recorder, "core_dataloader_prepare_seconds"):
        label_to_index = {label: index for index, label in enumerate(effective_labels)}
        uses_client_labeled_rows = (
            partitioned_runtime_plan.local_supervision_regime.uses_client_labeled_rows
        )
        labeled_batch_size, resolved_unlabeled_batch_size, step_plan = (
            resolve_partitioned_local_budget(
                policy_name=local_budget_policy,
                labeled_count=len(effective_labeled_rows),
                unlabeled_count=len(effective_unlabeled_rows),
                training_task=training_task,
                configured_unlabeled_batch_size=unlabeled_batch_size,
                effective_parameters=ssl_method_config.effective_parameters,
                uses_labeled_batches=uses_client_labeled_rows,
            )
        )
        train_loader = (
            build_dataloader(
                rows=effective_labeled_rows,
                label_to_index=label_to_index,
                tokenizer=tokenizer,
                batch_size=labeled_batch_size,
                max_length=peft_config.max_length,
                task_prefix=peft_config.task_prefix,
                shuffle=True,
                tokenization_cache=tokenization_cache,
                tokenization_cache_namespace=tokenization_cache_namespace_value,
            )
            if uses_client_labeled_rows
            else None
        )
        unlabeled_loader = build_multiview_dataloader(
            rows=effective_unlabeled_rows,
            tokenizer=tokenizer,
            batch_size=resolved_unlabeled_batch_size,
            max_length=peft_config.max_length,
            task_prefix=peft_config.task_prefix,
            shuffle=True,
            strong_view_policy=strong_view_policy,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
        )
        psi_query_ssl_algorithm = _build_psi_query_ssl_algorithm(
            local_ssl_policy_name=local_ssl_policy_name,
            query_ssl_config=query_ssl_config,
            train_loader_steps=step_plan.labeled_loader_steps,
            unlabeled_loader_steps=step_plan.unlabeled_loader_steps,
            total_steps=step_plan.total_steps,
            num_classes=len(effective_labels),
            unlabeled_row_count=len(effective_unlabeled_rows),
            initial_query_ssl_algorithm_state=initial_query_ssl_algorithm_state,
        )

    with _measure(timing_recorder, "core_training_loop_seconds"):
        uses_physical_partition_runtime = psi_query_ssl_algorithm is None
        server_partition_parameters = (
            {} if base_partition_parameters is None else base_partition_parameters
        )
        previous_client_partition_parameters = (
            {}
            if previous_client_partition_parameters is None
            else previous_client_partition_parameters
        )
        sparse_sync_parameters = PartitionSparseSyncParameters.from_mapping(
            ssl_method_config.effective_parameters,
            l1_sparse_partitions=partitioned_runtime_plan.l1_sparse_partitions,
        )
        uses_s2c_sparse_download = (
            uses_physical_partition_runtime
            and bool(server_partition_parameters)
            and bool(previous_client_partition_parameters)
        )
        effective_base_partition_parameters = server_partition_parameters
        partition_initialization_metrics: dict[str, float] = {}
        s2c_sparse_download_value_count = 0
        if uses_s2c_sparse_download:
            s2c_sparse_download_deltas = apply_partitioned_s2c_sparse_download(
                server_partition_parameters=server_partition_parameters,
                client_partition_parameters=previous_client_partition_parameters,
                parameters=sparse_sync_parameters,
            )
            s2c_sparse_download_value_count = count_partition_delta_nonzero_values(
                s2c_sparse_download_deltas
            )
            effective_base_partition_parameters = (
                project_partitioned_s2c_sparse_download(
                    server_partition_parameters=server_partition_parameters,
                    client_partition_parameters=previous_client_partition_parameters,
                    parameters=sparse_sync_parameters,
                )
            )
        elif (
            uses_physical_partition_runtime and not effective_base_partition_parameters
        ):
            split_published_state = ps.split_peft_encoder_state_by_residual_factor
            effective_base_partition_parameters = split_published_state(
                published_parameters=base_parameters,
                base_partition_name=partitioned_runtime_plan.supervised_partition,
                residual_partition_name=partitioned_runtime_plan.unsupervised_partition,
                residual_factor=partitioned_runtime_plan.psi_factor,
            )
            partition_initialization_metrics = {
                "fedmatch_initial_partition_from_published_state": 1.0,
                "fedmatch_initial_partition_psi_factor": (
                    partitioned_runtime_plan.psi_factor
                ),
            }
        if uses_physical_partition_runtime:
            partitioned_build = (
                build_partitioned_peft_encoder_text_classifier_from_config(
                    partition_names=partitioned_runtime_plan.partition_names,
                    labels=effective_labels,
                    base_parameters=base_parameters,
                    base_partition_parameters=effective_base_partition_parameters,
                    peft_config=peft_config,
                    runtime_config=trainer_runtime_config,
                    runtime_resource_cache=runtime_resource_cache,
                )
            )
            training_result = train_physical_partitioned_adapter_classifier(
                model=partitioned_build.model,
                train_loader=train_loader,
                unlabeled_loader=unlabeled_loader,
                labels=effective_labels,
                objective=partitioned_runtime_plan.physical_objective,
                step_plan=step_plan,
                device=trainer_runtime_config.device,
                learning_rate=float(training_task.learning_rate),
                classifier_learning_rate=float(training_task.learning_rate),
                weight_decay=0.0,
                max_grad_norm=(
                    0.0
                    if training_task.gradient_clip_norm is None
                    else float(training_task.gradient_clip_norm)
                ),
                supervised_partition=partitioned_runtime_plan.supervised_partition,
                unsupervised_partition=partitioned_runtime_plan.unsupervised_partition,
                helper_weak_probability_provider=helper_weak_probability_provider,
                enable_inter_client_consistency=(
                    helper_weak_probability_provider is not None
                ),
                use_supervised_steps=(
                    partitioned_runtime_plan.local_supervision_regime.uses_client_labeled_rows
                ),
                emit_supervised_partition=partitioned_runtime_plan.emit_supervised_partition,
                metric_prefix=partitioned_runtime_plan.metric_prefix,
            )
            if partition_initialization_metrics:
                training_result = PartitionedAdapterClassifierTrainingResult(
                    metrics={
                        **training_result.metrics,
                        **partition_initialization_metrics,
                    },
                    partition_deltas=training_result.partition_deltas,
                )
            client_partition_parameters = (
                ps.apply_peft_encoder_partition_deltas_to_partitioned_state(
                    base_parameters=base_parameters,
                    base_partition_parameters=effective_base_partition_parameters,
                    partition_deltas=training_result.partition_deltas,
                )
            )
            c2s_projection = project_partitioned_c2s_sparse_upload(
                base_parameters=base_parameters,
                server_partition_parameters=server_partition_parameters,
                client_partition_parameters=client_partition_parameters,
                parameters=sparse_sync_parameters,
            )
            c2s_sparse_upload_value_count = count_partition_delta_nonzero_values(
                c2s_projection.upload_partition_deltas
            )
            training_result = replace_partitioned_training_deltas(
                training_result=training_result,
                partition_deltas=c2s_projection.upload_partition_deltas,
            )
            client_partition_parameters = {
                partition_name: compact_peft_encoder_materialized_state(partition_state)
                for partition_name, partition_state in (
                    c2s_projection.client_partition_parameters.items()
                )
            }
            merged_partition_delta = ps.merge_partitioned_peft_encoder_deltas(
                training_result.partition_deltas
            )
            merged_parameters = ps.apply_peft_encoder_partition_delta_to_state(
                base_parameters=base_parameters,
                delta=merged_partition_delta,
            )
            load_peft_encoder_base_parameters_into_model(
                model=model,
                labels=effective_labels,
                base_parameters=merged_parameters,
                device=trainer_runtime_config.device,
            )
        else:
            client_partition_parameters = {}
            c2s_sparse_upload_value_count = 0
            training_result = train_partitioned_adapter_classifier(
                model=model,
                train_loader=train_loader,
                unlabeled_loader=unlabeled_loader,
                labels=effective_labels,
                objective=partitioned_runtime_plan.sequential_objective,
                step_plan=step_plan,
                device=trainer_runtime_config.device,
                learning_rate=float(training_task.learning_rate),
                classifier_learning_rate=float(training_task.learning_rate),
                weight_decay=0.0,
                max_grad_norm=(
                    0.0
                    if training_task.gradient_clip_norm is None
                    else float(training_task.gradient_clip_norm)
                ),
                helper_weak_probability_provider=helper_weak_probability_provider,
                psi_query_ssl_algorithm=psi_query_ssl_algorithm,
                enable_inter_client_consistency=(
                    helper_weak_probability_provider is not None
                ),
                use_supervised_steps=(
                    partitioned_runtime_plan.local_supervision_regime.uses_client_labeled_rows
                ),
                supervised_partition=partitioned_runtime_plan.supervised_partition,
                unsupervised_partition=partitioned_runtime_plan.unsupervised_partition,
                emit_sigma_partition=partitioned_runtime_plan.emit_supervised_partition,
                metric_prefix=partitioned_runtime_plan.metric_prefix,
            )
    history_record = training_result.metrics
    diagnostic_threshold = _resolve_partitioned_diagnostic_threshold(
        local_ssl_policy_name=local_ssl_policy_name,
        partitioned_runtime_plan=partitioned_runtime_plan,
        query_ssl_config=query_ssl_config,
    )
    with _measure(timing_recorder, "core_pseudo_label_diagnostics_seconds"):
        pseudo_label_quality = build_final_snapshot_pseudo_label_quality(
            model=model,
            tokenizer=tokenizer,
            rows=(
                effective_unlabeled_rows
                if diagnostic_unlabeled_rows is None
                else list(diagnostic_unlabeled_rows)
            ),
            labels=effective_labels,
            peft_config=peft_config,
            acceptance_threshold=diagnostic_threshold.threshold,
            trainer_runtime_config=trainer_runtime_config,
            unlabeled_batch_size=resolved_unlabeled_batch_size,
            tokenization_cache=tokenization_cache,
            tokenization_cache_namespace=tokenization_cache_namespace_value,
        )

    with _measure(timing_recorder, "core_delta_extract_seconds"):
        peft_parameter_deltas, head_weight_deltas, head_bias_deltas = (
            extract_peft_encoder_parameter_deltas(
                model=model,
                base_parameters=base_parameters,
                labels=effective_labels,
            )
        )
    update_id = f"update_{training_task.round_id}_{client_id}_{uuid4().hex[:12]}"
    with _measure(timing_recorder, "core_delta_materialization_seconds"):
        delta_materialization = delta_materializer.prepare(
            update_id=update_id,
            training_task=training_task,
            client_id=client_id,
            delta_format=peft_config.delta_format,
            artifact_ref_prefix=peft_config.artifact_ref_prefix,
            peft_parameter_deltas=peft_parameter_deltas,
            classifier_head_weight_deltas=head_weight_deltas,
            classifier_head_bias_deltas=head_bias_deltas,
            partitioned_deltas=training_result.partition_deltas,
            materialize_primary_deltas=False,
        )
    with _measure(timing_recorder, "core_update_payload_build_seconds"):
        update_build_result = build_query_ssl_peft_encoder_update_payload(
            training_task=training_task,
            model_manifest=model_manifest,
            peft_config=peft_config,
            labels=effective_labels,
            labeled_rows=effective_labeled_rows,
            unlabeled_rows=effective_unlabeled_rows,
            step_plan=step_plan,
            history_record=history_record,
            peft_parameter_deltas=peft_parameter_deltas,
            classifier_head_weight_deltas=head_weight_deltas,
            classifier_head_bias_deltas=head_bias_deltas,
            partitioned_deltas=training_result.partition_deltas,
            created_at=created_at,
            delta_format=delta_materialization.delta_format,
            peft_adapter_delta_artifact_ref=delta_materialization.peft_adapter_delta_artifact_ref,
            classifier_head_delta_artifact_ref=(
                delta_materialization.classifier_head_delta_artifact_ref
            ),
            partitioned_deltas_artifact_ref=(
                delta_materialization.partitioned_deltas_artifact_ref
            ),
            include_inline_deltas=delta_materialization.include_inline_deltas,
        )
    peer_context_helper_count = (
        0.0 if peer_context is None else float(peer_context.helper_count)
    )
    helper_provider_count = _optional_helper_provider_metric(
        helper_weak_probability_provider,
        "helper_count",
    )
    materialized_helper_model_count = _optional_helper_provider_metric(
        helper_weak_probability_provider,
        "materialized_helper_count",
    )
    missing_helper_snapshot_count = max(
        0.0,
        peer_context_helper_count - helper_provider_count,
    )
    client_metrics = {
        **dict(update_build_result.client_metrics),
        "fedmatch_local_runtime": 1.0,
        "fedmatch_physical_partition_runtime": float(uses_physical_partition_runtime),
        "fedmatch_c2s_sparse_upload": float(uses_physical_partition_runtime),
        "fedmatch_s2c_sparse_download": float(uses_s2c_sparse_download),
        "fedmatch_c2s_sparse_upload_value_count": float(c2s_sparse_upload_value_count),
        "fedmatch_s2c_sparse_download_value_count": float(
            s2c_sparse_download_value_count
        ),
        "fedmatch_local_ssl_policy_is_fixmatch": float(
            local_ssl_policy_name == LOCAL_SSL_POLICY_FIXMATCH
        ),
        "fedmatch_helper_count": _history_float(
            history_record,
            "train_fedmatch_psi_helper_count",
        ),
        "fedmatch_peer_context_helper_count": peer_context_helper_count,
        "fedmatch_helper_provider_count": helper_provider_count,
        "fedmatch_missing_helper_snapshot_count": missing_helper_snapshot_count,
        "fedmatch_materialized_helper_model_count": materialized_helper_model_count,
        "fedmatch_peer_context_refreshed": (
            0.0 if peer_context is None else float(peer_context.refreshed)
        ),
        "fedmatch_partitioned_delta_count": float(
            len(training_result.partition_deltas)
        ),
        "fedmatch_local_budget_policy_is_original": float(
            local_budget_policy == LOCAL_BUDGET_POLICY_ORIGINAL_METHOD
        ),
        "fedmatch_budget_client_epochs": float(step_plan.local_epochs),
        "fedmatch_budget_labeled_batch_size": float(labeled_batch_size),
        "fedmatch_budget_unlabeled_batch_size": float(resolved_unlabeled_batch_size),
        "fedmatch_budget_steps_per_epoch": float(step_plan.full_epoch_steps),
        "fedmatch_budget_total_steps": float(step_plan.total_steps),
        **diagnostic_threshold.to_client_metrics(),
    }
    update_envelope = make_training_update_envelope(
        update_id=update_id,
        round_id=training_task.round_id,
        task_id=training_task.task_id,
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        payload_ref=f"client-submission::{update_id}",
        payload_format=_payload_format_for_update(update_build_result.update_payload),
        example_count=update_build_result.update_payload.example_count,
        client_metrics=client_metrics,
        created_at=created_at,
    )

    return QuerySslPeftEncoderClientTrainingResult(
        update_envelope=update_envelope,
        update_payload=update_build_result.update_payload,
        candidate_count=len(effective_unlabeled_rows),
        accepted_count=update_build_result.accepted_unlabeled_count,
        local_step_plan=step_plan,
        client_metrics=client_metrics,
        pseudo_label_quality=pseudo_label_quality,
        peer_client_snapshot=_build_timed_peer_client_snapshot(
            timing_recorder=timing_recorder,
            client_id=client_id,
            model=model,
            tokenizer=tokenizer,
            peer_probe_rows=peer_probe_rows,
            labels=effective_labels,
            peft_config=peft_config,
            trainer_runtime_config=trainer_runtime_config,
            probe_batch_size=resolved_unlabeled_batch_size,
        ),
        client_partition_parameters=client_partition_parameters,
        query_ssl_algorithm_state=(
            {}
            if psi_query_ssl_algorithm is None
            else dict(export_query_ssl_algorithm_state(psi_query_ssl_algorithm))
        ),
        timing_breakdown=timing_mapping(timing_recorder),
    )


def _payload_format_for_update(update_payload: PeftClassifierDelta) -> str:
    return PEFT_CLASSIFIER_UPDATE_PAYLOAD_FORMAT


def _build_psi_query_ssl_algorithm(
    *,
    local_ssl_policy_name: str,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig | None,
    train_loader_steps: int,
    unlabeled_loader_steps: int,
    total_steps: int,
    num_classes: int,
    unlabeled_row_count: int,
    initial_query_ssl_algorithm_state: Mapping[str, Any] | None = None,
) -> QuerySslAlgorithm | None:
    normalized_policy = local_ssl_policy_name.strip().lower().replace("-", "_")
    if normalized_policy == LOCAL_SSL_POLICY_FEDMATCH_AGREEMENT:
        if initial_query_ssl_algorithm_state:
            raise ValueError(
                "Query SSL algorithm state cannot be loaded for "
                "local_ssl_policy=fedmatch_agreement."
            )
        return None
    if normalized_policy != LOCAL_SSL_POLICY_FIXMATCH:
        raise NotImplementedError(
            "FedMatch partitioned LoRA runtime currently supports "
            "local_ssl_policy=fedmatch_agreement or fixmatch."
        )
    if query_ssl_config is None:
        raise ValueError("local_ssl_policy=fixmatch requires query_ssl_config.")
    descriptor = resolve_query_ssl_algorithm_descriptor(query_ssl_config.algorithm_name)
    if descriptor.algorithm_name != LOCAL_SSL_POLICY_FIXMATCH:
        raise ValueError(
            "local_ssl_policy=fixmatch requires "
            "query_ssl_method.algorithm_name=fixmatch."
        )
    algorithm = descriptor.build_algorithm(query_ssl_config.parameters)
    algorithm.validate_loaders(
        train_loader_length=train_loader_steps,
        unlabeled_loader_length=unlabeled_loader_steps,
    )
    configure_query_ssl_algorithm_training(
        algorithm,
        num_train_iter=total_steps,
    )
    configure_query_ssl_algorithm_dataset(
        algorithm,
        num_classes=num_classes,
        unlabeled_row_count=unlabeled_row_count,
    )
    if initial_query_ssl_algorithm_state:
        load_query_ssl_algorithm_state(
            algorithm,
            initial_query_ssl_algorithm_state,
        )
    return algorithm


def replace_partitioned_training_deltas(
    *,
    training_result: PartitionedAdapterClassifierTrainingResult,
    partition_deltas: Mapping[str, PeftEncoderPartitionDelta],
) -> PartitionedAdapterClassifierTrainingResult:
    """학습 metric은 유지하고 upload 대상 partition delta만 교체한다."""

    return replace(training_result, partition_deltas=partition_deltas)


def _build_timed_peer_client_snapshot(
    *,
    timing_recorder: TimingRecorder | None,
    client_id: str,
    model: PeftEncoderTextClassifier,
    tokenizer: Any,
    peer_probe_rows: Sequence[LabeledQueryRow] | None,
    labels: Sequence[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    probe_batch_size: int,
) -> FederatedSslPeerClientSnapshot:
    with _measure(timing_recorder, "core_peer_snapshot_build_seconds"):
        return peer_predictions.build_peft_encoder_peer_client_snapshot(
            client_id=client_id,
            model=model,
            tokenizer=tokenizer,
            probe_rows=() if peer_probe_rows is None else peer_probe_rows,
            labels=labels,
            peft_config=peft_config,
            trainer_runtime_config=trainer_runtime_config,
            probe_batch_size=probe_batch_size,
        )


def _measure(timing_recorder: TimingRecorder | None, key: str) -> Any:
    if timing_recorder is None:
        return nullcontext()
    return timing_recorder.measure(key)


def _resolve_partitioned_diagnostic_threshold(
    *,
    local_ssl_policy_name: str,
    partitioned_runtime_plan: PartitionedLocalRuntimePlan,
    query_ssl_config: QuerySslPeftEncoderObjectiveRuntimeConfig | None,
) -> PseudoLabelDiagnosticThreshold:
    if (
        local_ssl_policy_name.strip().lower().replace("-", "_")
        == LOCAL_SSL_POLICY_FIXMATCH
    ):
        if query_ssl_config is None:
            raise ValueError("local_ssl_policy=fixmatch requires query_ssl_config.")
        return resolve_fixed_pseudo_label_diagnostic_threshold(
            dict(query_ssl_config.parameters)
        )
    return resolve_fixed_pseudo_label_diagnostic_threshold(
        {"p_cutoff": partitioned_runtime_plan.diagnostic_acceptance_threshold}
    )


def _history_float(
    history_record: Mapping[str, object],
    key: str,
) -> float:
    value = history_record.get(key)
    if value is None:
        return 0.0
    return float(value)


def _optional_helper_provider_metric(provider: object | None, key: str) -> float:
    if provider is None:
        return 0.0
    value = getattr(provider, key, 0.0)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _validate_labeled_rows_have_known_labels(
    *,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
) -> None:
    known_labels = {str(label) for label in labels}
    missing = sorted({str(row["mapped_label_4"]) for row in rows} - known_labels)
    if missing:
        raise ValueError(
            "Partitioned labeled_rows contain labels outside active label_schema: "
            f"{missing}."
        )
