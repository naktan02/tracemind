"""FL simulation protocol section payload helpers."""

from __future__ import annotations

from collections.abc import Mapping

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.capabilities.plan import FederatedSslCapabilityPlan
from methods.federated_ssl.execution_plan import FederatedSslExecutionPlan
from scripts.experiments.fl_ssl.federated_simulation.io.split_diagnostics import (
    build_client_pool_split_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedArtifactPersistenceConfig,
    FederatedClientPoolSplitConfig,
    FederatedDatasetSplit,
    FederatedDataSourceConfig,
    FederatedDiagnosticViewConfig,
    FederatedLocalTrainerRuntimeConfig,
    FederatedPeerProbeManifest,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationResult,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def build_protocol_payload(
    *,
    result: SimulationResult,
    report_config: FederatedReportConfig,
    client_count: int,
    round_budget: int,
    bootstrap_ratio: float,
    seed: int,
    run_budget_name: str | None = None,
    run_output_dir: str | None = None,
    shard_policy: FederatedShardPolicyConfig,
    dataset_split: FederatedDatasetSplit,
    ssl_method_config: FederatedSslMethodConfig | None,
    client_pool_split_config: FederatedClientPoolSplitConfig | None,
    training_task_config: FederatedTrainingTaskConfig,
    validation_config: FederatedValidationConfig,
    round_runtime_config: FederatedRoundRuntimeConfig,
    execution_plan: FederatedSslExecutionPlan | None = None,
    local_update_profile_name: str | None = None,
    capability_plan: FederatedSslCapabilityPlan | None = None,
    server_step_executor: str | None = None,
    data_source_config: FederatedDataSourceConfig | None = None,
    embedding_spec: EmbeddingAdapterSpec | None = None,
    local_trainer_runtime_config: FederatedLocalTrainerRuntimeConfig | None = None,
    artifact_persistence_config: FederatedArtifactPersistenceConfig | None = None,
    diagnostic_view_config: FederatedDiagnosticViewConfig | None = None,
    peer_probe_manifest: FederatedPeerProbeManifest | None = None,
) -> dict[str, object]:
    resolved_data_source_config = data_source_config or FederatedDataSourceConfig()
    payload: dict[str, object] = {
        "client_count": client_count,
        "round_budget": round_budget,
        "completed_rounds": len(result.rounds),
        "seed": seed,
        "run_control": _run_control_to_payload(
            run_budget_name=run_budget_name,
            run_output_dir=run_output_dir,
        ),
        "seed_count": report_config.seed_count,
        "bootstrap_ratio": bootstrap_ratio,
        "shard_policy": _shard_policy_to_payload(shard_policy),
        "fl_data_source": _fl_data_source_to_payload(resolved_data_source_config),
        "ssl_method": _ssl_method_to_payload(ssl_method_config),
        "labeled_unlabeled_split": build_client_pool_split_payload(
            dataset_split=dataset_split,
            client_pool_split_config=client_pool_split_config,
            report_config=report_config,
            data_source_config=resolved_data_source_config,
        ),
        "local_update_budget": {
            "local_epochs": training_task_config.local_epochs,
            "batch_size": training_task_config.batch_size,
            "learning_rate": training_task_config.learning_rate,
            "max_steps": training_task_config.max_steps,
            "min_required_examples": training_task_config.min_required_examples,
            "gradient_clip_norm": training_task_config.gradient_clip_norm,
            "selection_policy": config_to_mapping(
                training_task_config.selection_policy
            ),
        },
        "round_runtime": {
            "payload_adapter_kind": round_runtime_config.payload_adapter_kind,
            "update_family_name": round_runtime_config.update_family_name,
            "round_runtime_payload_builder": (
                round_runtime_config.round_runtime_payload_builder
            ),
            "local_objective_executors": list(
                round_runtime_config.local_objective_executors
            ),
            "client_round_runtime": dict(round_runtime_config.client_round_runtime),
            "server_round_runtime": dict(round_runtime_config.server_round_runtime),
            "initial_state_builder": round_runtime_config.initial_state_builder,
            "validation_evaluator": round_runtime_config.validation_evaluator,
            "final_projection_builder": (round_runtime_config.final_projection_builder),
            "transient_resource_cleaner": (
                round_runtime_config.transient_resource_cleaner
            ),
            "release_transient_model_cache_after_client": (
                round_runtime_config.release_transient_model_cache_after_client
            ),
            "aggregation_backend_name": (round_runtime_config.aggregation_backend_name),
        },
        "fl_capabilities": _capability_plan_to_payload(capability_plan),
        "server_step_runtime": {
            "executor": server_step_executor,
        },
        "embedding_adapter": _embedding_spec_to_payload(embedding_spec),
        "local_trainer_runtime": _local_trainer_runtime_to_payload(
            local_trainer_runtime_config
        ),
        "artifact_persistence": _artifact_persistence_to_payload(
            artifact_persistence_config
        ),
        "diagnostic_view": _diagnostic_view_to_payload(diagnostic_view_config),
        "peer_probe": _peer_probe_to_payload(peer_probe_manifest),
        "objective": config_to_mapping(training_task_config.objective_config),
        "validation": {
            "similarity_name": validation_config.similarity_name,
            "scorer_backend_name": validation_config.scorer_backend_name,
            "score_policy_name": validation_config.score_policy_name,
            "score_top_k": validation_config.score_top_k,
            "confidence_threshold": validation_config.confidence_threshold,
            "margin_threshold": validation_config.margin_threshold,
        },
    }
    if execution_plan is not None:
        payload["fl_method"] = execution_plan.to_mapping()
        payload["runtime_selection"] = execution_plan.runtime_selection(
            local_update_profile_name=local_update_profile_name,
        ).to_mapping()
    return payload


def _capability_plan_to_payload(
    capability_plan: FederatedSslCapabilityPlan | None,
) -> dict[str, object]:
    if capability_plan is None:
        return {"metadata_status": "not_recorded"}
    return {
        "metadata_status": "recorded",
        **capability_plan.to_payload(),
    }


def _run_control_to_payload(
    *,
    run_budget_name: str | None,
    run_output_dir: str | None,
) -> dict[str, object]:
    if run_budget_name is None and run_output_dir is None:
        return {"metadata_status": "not_recorded"}
    return {
        "metadata_status": "recorded",
        "budget_name": run_budget_name,
        "output_dir": run_output_dir,
    }


def _embedding_spec_to_payload(
    embedding_spec: EmbeddingAdapterSpec | None,
) -> dict[str, object]:
    if embedding_spec is None:
        return {"metadata_status": "not_recorded"}
    return {
        "metadata_status": "recorded",
        "backend": embedding_spec.backend,
        "model_id": embedding_spec.model_id,
        "revision": embedding_spec.revision,
        "device": embedding_spec.device,
        "batch_size": embedding_spec.batch_size,
        "cache_dir": embedding_spec.cache_dir,
        "task_prefix": embedding_spec.task_prefix,
        "normalize_embeddings": embedding_spec.normalize_embeddings,
        "hash_dim": embedding_spec.hash_dim,
        "local_files_only": embedding_spec.local_files_only,
    }


def _local_trainer_runtime_to_payload(
    runtime_config: FederatedLocalTrainerRuntimeConfig | None,
) -> dict[str, object]:
    if runtime_config is None:
        return {"metadata_status": "not_recorded"}
    return {
        "metadata_status": "recorded",
        "device": runtime_config.device,
        "local_files_only": runtime_config.local_files_only,
        "cache_dir": runtime_config.cache_dir,
        "trust_remote_code": runtime_config.trust_remote_code,
        "classifier_dropout": runtime_config.classifier_dropout,
    }


def _diagnostic_view_to_payload(
    config: FederatedDiagnosticViewConfig | None,
) -> dict[str, object]:
    if config is None:
        return {"metadata_status": "not_recorded"}
    return {
        "metadata_status": "recorded",
        "enabled": config.enabled,
        "selection_policy": config.selection_policy,
        "max_rows": config.max_rows,
        "seed_offset": config.seed_offset,
        "source": "client_unlabeled_pool",
        "scope": "pseudo_label_diagnostics_only",
    }


def _artifact_persistence_to_payload(
    config: FederatedArtifactPersistenceConfig | None,
) -> dict[str, object]:
    if config is None:
        return {"metadata_status": "not_recorded"}
    return {
        "metadata_status": "recorded",
        "persist_agent_local_updates": config.persist_agent_local_updates,
        "canonical_update_source": "server_owned_aggregation_artifact",
    }


def _peer_probe_to_payload(
    manifest: FederatedPeerProbeManifest | None,
) -> dict[str, object]:
    if manifest is None:
        return {"metadata_status": "disabled"}
    return manifest.to_payload()


def config_to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if hasattr(value, "to_mapping"):
        mapping = value.to_mapping()
        if not isinstance(mapping, Mapping):
            raise TypeError("to_mapping() must return a mapping.")
        return dict(mapping)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported config mapping type: {type(value).__name__}")


def _shard_policy_to_payload(
    shard_policy: FederatedShardPolicyConfig,
) -> dict[str, object]:
    return {
        "name": shard_policy.name,
        "client_id_prefix": shard_policy.client_id_prefix,
        "dominant_ratio": shard_policy.dominant_ratio,
        "alpha": shard_policy.alpha,
    }


def _ssl_method_to_payload(
    ssl_method_config: FederatedSslMethodConfig | None,
) -> dict[str, object]:
    if ssl_method_config is None:
        return {
            "metadata_status": "not_applicable",
            "reason": "manual_composition",
        }
    return {
        "schema_version": ssl_method_config.schema_version,
        "name": ssl_method_config.name,
        "display_name": ssl_method_config.display_name,
        "method_role": ssl_method_config.method_role,
        "implementation_status": ssl_method_config.implementation_status,
        "original_source": dict(ssl_method_config.original_source),
        "scenario": ssl_method_config.scenario,
        "use_original_parameters": ssl_method_config.use_original_parameters,
        "local_budget_policy": ssl_method_config.local_budget_policy,
        "original_parameters": dict(ssl_method_config.original_parameters),
        "parameter_overrides": dict(ssl_method_config.parameter_overrides),
        "effective_parameters": dict(ssl_method_config.effective_parameters),
        "parameter_override_status": ssl_method_config.parameter_override_status,
        "trace_mapping": dict(ssl_method_config.trace_mapping),
        "client_step": dict(ssl_method_config.client_step),
        "server_step": dict(ssl_method_config.server_step),
        "round_state_exchange": dict(ssl_method_config.round_state_exchange),
        "report_tags": list(ssl_method_config.report_tags),
        "notes": list(ssl_method_config.notes),
    }


def _fl_data_source_to_payload(
    data_source_config: FederatedDataSourceConfig,
) -> dict[str, object]:
    return {
        "source_mode": data_source_config.source_mode,
        "split_manifest_path": data_source_config.split_manifest_path,
        "split_manifest_sha256": data_source_config.split_manifest_sha256,
        "split_id": data_source_config.split_id,
        "source_selection": dict(data_source_config.source_selection),
        "source_jsonl": dict(data_source_config.source_jsonl),
        "labeled_policy": dict(data_source_config.labeled_policy),
        "labeled_exposure_policy": dict(data_source_config.labeled_exposure_policy),
        "view_schema": dict(data_source_config.view_schema),
        "test_jsonl": data_source_config.test_jsonl,
    }
