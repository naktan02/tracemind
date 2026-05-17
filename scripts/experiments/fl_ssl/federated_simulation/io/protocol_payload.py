"""FL simulation protocol section payload helpers."""

from __future__ import annotations

from collections.abc import Mapping

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.execution_plan import FederatedSslExecutionPlan
from scripts.experiments.fl_ssl.federated_simulation.io.split_diagnostics import (
    build_client_pool_split_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedDatasetSplit,
    FederatedDataSourceConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationResult,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)


def build_protocol_payload(
    *,
    result: SimulationResult,
    report_config: FederatedReportConfig,
    client_count: int,
    round_budget: int,
    bootstrap_ratio: float,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
    dataset_split: FederatedDatasetSplit,
    ssl_method_config: FederatedSslMethodConfig,
    client_pool_split_config: FederatedClientPoolSplitConfig | None,
    training_task_config: FederatedTrainingTaskConfig,
    validation_config: FederatedValidationConfig,
    round_runtime_config: FederatedRoundRuntimeConfig,
    execution_plan: FederatedSslExecutionPlan | None = None,
    data_source_config: FederatedDataSourceConfig | None = None,
) -> dict[str, object]:
    resolved_data_source_config = data_source_config or FederatedDataSourceConfig()
    payload: dict[str, object] = {
        "client_count": client_count,
        "round_budget": round_budget,
        "completed_rounds": len(result.rounds),
        "seed": seed,
        "seed_count": report_config.seed_count,
        "bootstrap_ratio": bootstrap_ratio,
        "shard_policy": _shard_policy_to_payload(shard_policy),
        "fl_data_source": _fl_data_source_to_payload(resolved_data_source_config),
        "ssl_method": _ssl_method_to_payload(ssl_method_config),
        "labeled_unlabeled_split": build_client_pool_split_payload(
            dataset_split=dataset_split,
            client_pool_split_config=client_pool_split_config,
            report_config=report_config,
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
            "adapter_family_name": round_runtime_config.adapter_family_name,
            "aggregation_backend_name": (round_runtime_config.aggregation_backend_name),
            "classifier_head_bootstrap_logit_scale": (
                round_runtime_config.classifier_head_bootstrap_logit_scale
            ),
        },
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
    return payload


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
    ssl_method_config: FederatedSslMethodConfig,
) -> dict[str, object]:
    return {
        "schema_version": ssl_method_config.schema_version,
        "name": ssl_method_config.name,
        "display_name": ssl_method_config.display_name,
        "method_role": ssl_method_config.method_role,
        "implementation_status": ssl_method_config.implementation_status,
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
        "view_schema": dict(data_source_config.view_schema),
        "test_jsonl": data_source_config.test_jsonl,
    }
