"""Hydra FL SSL config를 simulation request로 변환한다."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.federated_ssl.compatibility import (
    validate_federated_ssl_local_ssl_policy_alignment,
)
from methods.federated_ssl.execution_plan import (
    COMPOSITION_MODE_MANUAL,
    FederatedSslExecutionPlan,
    build_federated_ssl_execution_plan,
)
from methods.federated_ssl.local_update_profile import (
    LocalUpdateProfile,
    require_training_objective_matches_local_update_profile,
)
from methods.federated_ssl.method_parameters import (
    build_federated_ssl_method_parameter_snapshot,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from scripts.experiments.fl_ssl.federated_simulation.config_utils import (
    optional_plain_dict,
    to_plain_dict,
)
from scripts.experiments.fl_ssl.federated_simulation.data_source_request import (
    resolve_fl_data_source,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedArtifactPersistenceConfig,
    FederatedClientPoolSplitConfig,
    FederatedDiagnosticsConfig,
    FederatedDiagnosticViewConfig,
    FederatedFinalProjectionConfig,
    FederatedLocalTrainerRuntimeConfig,
    FederatedPeerProbeConfig,
    FederatedQuerySslObjectiveConfig,
    FederatedReportConfig,
    FederatedResumeConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_federated_training_task_config,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


def build_simulation_request_from_config(
    cfg: DictConfig,
    *,
    output_dir: Path,
    seed: int | None = None,
) -> SimulationRunRequest:
    """Hydra 실행 config를 typed simulation request로 해석한다."""

    embedding_spec = instantiate(cfg.embedding.spec)
    local_update_profile = LocalUpdateProfile.from_mapping(
        to_plain_dict(cfg.local_update_profile)
    )
    execution_plan = _build_execution_plan(cfg)
    training_task_config = _build_training_task_config(
        cfg.training_task,
        task_type=_resolve_training_task_type(cfg=cfg, execution_plan=execution_plan),
        local_update_profile=local_update_profile,
    )
    round_runtime_payloads = _build_round_runtime_payloads(cfg.round_runtime)
    round_runtime_config = FederatedRoundRuntimeConfig(
        adapter_family_name=str(cfg.round_runtime.adapter_family_name),
        aggregation_backend_name=str(cfg.round_runtime.aggregation_backend_name),
        update_family_name=str(cfg.round_runtime.update_family_name),
        runtime_payload_key=_optional_config_str(
            cfg.round_runtime,
            "runtime_payload_key",
        ),
        runtime_payloads=round_runtime_payloads,
        round_runtime_payload_builder=_optional_config_str(
            cfg.round_runtime,
            "round_runtime_payload_builder",
        ),
        local_objective_executors=_optional_config_str_tuple(
            cfg.round_runtime,
            "local_objective_executors",
        ),
        initial_state_builder=_optional_config_str(
            cfg.round_runtime,
            "initial_state_builder",
        ),
        validation_evaluator=_optional_config_str(
            cfg.round_runtime,
            "validation_evaluator",
        ),
        final_projection_builder=_optional_config_str(
            cfg.round_runtime,
            "final_projection_builder",
        ),
        transient_resource_cleaner=_optional_config_str(
            cfg.round_runtime,
            "transient_resource_cleaner",
        ),
        classifier_head_bootstrap_logit_scale=float(
            cfg.round_runtime.classifier_head_bootstrap_logit_scale
        ),
    )
    actual_seed = int(cfg.seed if seed is None else seed)
    shard_policy = FederatedShardPolicyConfig(**to_plain_dict(cfg.shard_policy))
    client_pool_split_config = FederatedClientPoolSplitConfig(
        **to_plain_dict(cfg.client_pool_split)
    )
    fl_data_source = resolve_fl_data_source(
        cfg=cfg,
        client_count=int(cfg.federated_run_budget.client_count),
        bootstrap_ratio=float(cfg.federated_run_budget.bootstrap_ratio),
        seed=actual_seed,
        shard_policy=shard_policy,
    )
    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=fl_data_source.data_source_config.labeled_exposure_policy,
        execution_plan=execution_plan,
    )
    query_ssl_objective_config = FederatedQuerySslObjectiveConfig.from_mapping(
        to_plain_dict(cfg.query_ssl_method),
        strong_view_policy=str(cfg.query_ssl_strong_view_policy),
    )
    validate_federated_ssl_local_ssl_policy_alignment(
        capability_plan=capability_plan,
        query_ssl_algorithm_name=query_ssl_objective_config.algorithm_name,
    )
    return SimulationRunRequest(
        train_rows=fl_data_source.train_rows,
        validation_rows=fl_data_source.validation_rows,
        test_rows=fl_data_source.test_rows,
        output_dir=output_dir,
        client_count=int(cfg.federated_run_budget.client_count),
        rounds=int(cfg.federated_run_budget.rounds),
        bootstrap_ratio=float(cfg.federated_run_budget.bootstrap_ratio),
        seed=actual_seed,
        run_budget_name=_optional_config_str(cfg.federated_run_budget, "name"),
        run_output_dir=_optional_config_str(cfg.federated_run_budget, "output_dir"),
        embedding_spec=embedding_spec,
        model_id=str(cfg.published_model_id),
        training_scope=local_update_profile.training_scope,
        round_runtime_config=round_runtime_config,
        shard_policy=shard_policy,
        training_task_config=training_task_config,
        validation_config=FederatedValidationConfig(**to_plain_dict(cfg.validation)),
        diagnostics_config=FederatedDiagnosticsConfig(**to_plain_dict(cfg.diagnostics)),
        artifact_persistence_config=FederatedArtifactPersistenceConfig.from_mapping(
            optional_plain_dict(cfg, "artifact_persistence")
        ),
        resume_config=FederatedResumeConfig.from_mapping(
            optional_plain_dict(cfg, "resume")
        ),
        ssl_method_config=_build_ssl_method_config(cfg, execution_plan=execution_plan),
        client_pool_split_config=client_pool_split_config,
        materialized_dataset_split=fl_data_source.materialized_dataset_split,
        data_source_config=fl_data_source.data_source_config,
        report_config=FederatedReportConfig(**to_plain_dict(cfg.report)),
        local_update_profile=local_update_profile,
        execution_plan=execution_plan,
        capability_plan=capability_plan,
        server_step_executor=_optional_config_str(cfg.server_step_policy, "executor"),
        query_ssl_objective_config=query_ssl_objective_config,
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device=str(cfg.runtime.device),
            local_files_only=bool(cfg.runtime.local_files_only),
            cache_dir=str(cfg.paper_backbone.cache_dir),
            trust_remote_code=bool(cfg.paper_backbone.trust_remote_code),
            classifier_dropout=float(cfg.paper_backbone.classifier_dropout),
        ),
        diagnostic_view_config=FederatedDiagnosticViewConfig.from_mapping(
            optional_plain_dict(cfg, "diagnostic_view")
        ),
        final_projection_config=FederatedFinalProjectionConfig.from_mapping(
            optional_plain_dict(cfg, "final_projection")
        ),
        peer_probe_config=FederatedPeerProbeConfig.from_mapping(
            optional_plain_dict(cfg, "peer_probe")
        ),
    )


def _build_capability_plan(
    *,
    cfg: DictConfig,
    labeled_exposure_policy: dict[str, object],
    execution_plan: FederatedSslExecutionPlan | None = None,
) -> FederatedSslCapabilityPlan:
    """Hydra FL strategy axes를 runtime capability plan으로 정규화한다."""

    resolved_execution_plan = execution_plan or _build_execution_plan(cfg)
    return FederatedSslCapabilityPlan.from_mappings(
        client_participation_policy=optional_plain_dict(
            cfg, "client_participation_policy"
        ),
        aggregation_weight_policy=optional_plain_dict(cfg, "aggregation_weight_policy"),
        labeled_exposure_policy=(
            labeled_exposure_policy
            or optional_plain_dict(cfg, "labeled_exposure_policy")
        ),
        local_supervision_regime=optional_plain_dict(cfg, "local_supervision_regime"),
        server_step_policy=optional_plain_dict(cfg, "server_step_policy"),
        peer_context_policy=optional_plain_dict(cfg, "peer_context_policy"),
        update_partition_policy=optional_plain_dict(cfg, "update_partition_policy"),
        local_ssl_policy=_resolve_local_ssl_policy_mapping(
            cfg=cfg,
            execution_plan=resolved_execution_plan,
        ),
        server_update_policy=optional_plain_dict(cfg, "server_update_policy"),
        query_multiview_source=optional_plain_dict(cfg, "query_multiview_source"),
    )


def _resolve_local_ssl_policy_mapping(
    *,
    cfg: DictConfig,
    execution_plan: FederatedSslExecutionPlan,
) -> dict[str, object] | None:
    """method-owned local SSL objective는 descriptor 요구사항에서 읽는다."""

    if execution_plan.composition_mode == COMPOSITION_MODE_MANUAL:
        return optional_plain_dict(cfg, "local_ssl_policy")
    if execution_plan.descriptor_name is None:
        return optional_plain_dict(cfg, "local_ssl_policy")
    descriptor = resolve_federated_ssl_method_descriptor(execution_plan.descriptor_name)
    local_ssl_policy_names = descriptor.required_capabilities.local_ssl_policy_names
    configured_policy_name = _optional_config_str(
        cfg.ssl_method.trace_mapping,
        "local_ssl_policy",
    )
    if configured_policy_name is not None:
        if configured_policy_name not in local_ssl_policy_names:
            raise ValueError(
                "ssl_method.trace_mapping.local_ssl_policy must be supported by "
                "descriptor.required_capabilities.local_ssl_policy_names: "
                f"method={descriptor.name}, value={configured_policy_name!r}, "
                f"supported={list(local_ssl_policy_names)!r}."
            )
        return {
            "name": configured_policy_name,
            "parameter_source": "method_descriptor",
        }
    if not local_ssl_policy_names:
        return optional_plain_dict(cfg, "local_ssl_policy")
    if len(local_ssl_policy_names) != 1:
        raise ValueError(
            "method-owned local_ssl_policy derivation requires exactly one "
            "descriptor.required_capabilities.local_ssl_policy_names entry: "
            f"method={descriptor.name}, values={list(local_ssl_policy_names)!r}."
        )
    return {
        "name": local_ssl_policy_names[0],
        "parameter_source": "method_descriptor",
    }


def _optional_config_str(cfg: DictConfig, key: str) -> str | None:
    value = getattr(cfg, key, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_config_str_tuple(cfg: DictConfig, key: str) -> tuple[str, ...]:
    value = getattr(cfg, key, None)
    if value is None:
        return ()
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    else:
        values = [str(item).strip() for item in value]
    return tuple(item for item in values if item)


def _build_training_task_config(
    cfg: DictConfig,
    *,
    task_type: str,
    local_update_profile: LocalUpdateProfile,
) -> FederatedTrainingTaskConfig:
    objective_config = to_plain_dict(cfg.objective)
    selection_policy = to_plain_dict(cfg.selection_policy)
    training_objective = TrainingObjectiveConfig.from_mapping(objective_config)
    require_training_objective_matches_local_update_profile(
        objective_config=training_objective,
        local_update_profile=local_update_profile,
    )
    return build_federated_training_task_config(
        task_type=task_type,
        local_epochs=int(cfg.local_epochs),
        batch_size=int(cfg.batch_size),
        learning_rate=float(cfg.learning_rate),
        max_steps=int(cfg.max_steps),
        min_required_examples=int(cfg.min_required_examples),
        gradient_clip_norm=(
            None if cfg.gradient_clip_norm is None else float(cfg.gradient_clip_norm)
        ),
        objective_config=training_objective,
        selection_policy=TrainingSelectionPolicy.from_mapping(selection_policy),
    )


def _build_round_runtime_payloads(cfg: DictConfig) -> dict[str, object]:
    builder_path = _optional_config_str(cfg, "round_runtime_payload_builder")
    if builder_path is None:
        return {}
    builder = _load_round_runtime_payload_builder(builder_path)
    payloads = builder(round_runtime_mapping=to_plain_dict(cfg))
    if not isinstance(payloads, dict):
        raise ValueError(
            "round_runtime.round_runtime_payload_builder must return a dict: "
            f"{builder_path!r}."
        )
    return payloads


def _load_round_runtime_payload_builder(builder_path: str) -> Any:
    module_name, separator, function_name = builder_path.rpartition(".")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "round_runtime.round_runtime_payload_builder must be a fully qualified "
            f"function path: {builder_path!r}."
        )
    module = import_module(module_name)
    builder = getattr(module, function_name, None)
    if not callable(builder):
        raise ValueError(
            "round_runtime.round_runtime_payload_builder must point to a callable: "
            f"{builder_path!r}."
        )
    return builder


def _build_execution_plan(cfg: DictConfig) -> FederatedSslExecutionPlan:
    fl_method = to_plain_dict(cfg.fl_method)
    if _is_manual_composition(fl_method):
        descriptor = None
    else:
        ssl_method = cfg.get("ssl_method")
        if ssl_method is None:
            raise ValueError(
                "method-owned FL SSL execution requires strategy_axes/fl/"
                "method_descriptor config."
            )
        descriptor = resolve_federated_ssl_method_descriptor(str(ssl_method.name))
    return build_federated_ssl_execution_plan(
        fl_method=_with_inferred_manual_axes(cfg=cfg, fl_method=fl_method),
        security_policy=to_plain_dict(cfg.security_policy),
        method_descriptor=descriptor,
    )


def _build_ssl_method_config(
    cfg: DictConfig,
    *,
    execution_plan: FederatedSslExecutionPlan,
) -> FederatedSslMethodConfig | None:
    if execution_plan.composition_mode == COMPOSITION_MODE_MANUAL:
        return None
    ssl_method = cfg.get("ssl_method")
    if ssl_method is None:
        raise ValueError("method-owned FL SSL execution requires ssl_method config.")
    ssl_method_mapping = to_plain_dict(ssl_method)
    parameter_snapshot = build_federated_ssl_method_parameter_snapshot(
        method_name=str(ssl_method_mapping["name"]),
        method_config=ssl_method_mapping,
    )
    ssl_method_mapping.update(parameter_snapshot.to_mapping())
    return FederatedSslMethodConfig(**ssl_method_mapping)


def _resolve_training_task_type(
    *,
    cfg: DictConfig,
    execution_plan: FederatedSslExecutionPlan,
) -> str:
    if execution_plan.composition_mode == COMPOSITION_MODE_MANUAL:
        return TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING.value
    ssl_method = cfg.get("ssl_method")
    if ssl_method is None:
        raise ValueError("method-owned FL SSL execution requires ssl_method config.")
    return str(ssl_method.client_step.task_type)


def _is_manual_composition(fl_method: dict[str, object]) -> bool:
    return (
        str(fl_method.get("composition_mode", "method_owned")).strip().lower()
        == COMPOSITION_MODE_MANUAL
    )


def _with_inferred_manual_axes(
    *,
    cfg: DictConfig,
    fl_method: dict[str, object],
) -> dict[str, object]:
    """manual FL 조합에서 실제 lower-axis 값을 report metadata로 채운다."""

    if not _is_manual_composition(fl_method):
        return fl_method

    raw_manual_axes = fl_method.get("manual_axes")
    manual_axes = raw_manual_axes if isinstance(raw_manual_axes, dict) else {}
    inferred_axes = {
        "client_ssl_objective": _infer_client_ssl_objective_name(cfg),
        "server_aggregation": str(cfg.round_runtime.aggregation_backend_name),
        "update_family": str(cfg.round_runtime.update_family_name),
    }
    explicit_axes = {
        key: value
        for key, value in manual_axes.items()
        if value is not None and str(value).strip()
    }
    return {
        **fl_method,
        "manual_axes": {
            **inferred_axes,
            **explicit_axes,
        },
    }


def _infer_client_ssl_objective_name(cfg: DictConfig) -> str:
    """manual FL 조합에서 실제 SSL algorithm 축을 report용 이름으로 쓴다."""

    query_ssl_method = cfg.get("query_ssl_method")
    if (
        query_ssl_method is not None
        and query_ssl_method.get("algorithm_name") is not None
    ):
        return str(query_ssl_method.algorithm_name)
    return TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING.value
