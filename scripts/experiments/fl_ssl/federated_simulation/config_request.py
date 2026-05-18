"""Hydra FL SSL config를 simulation request로 변환한다."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.execution_plan import (
    COMPOSITION_MODE_MANUAL,
    FederatedSslExecutionPlan,
    build_federated_ssl_execution_plan,
)
from methods.federated_ssl.local_update_profile import (
    LocalUpdateProfile,
    require_training_objective_matches_local_update_profile,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from scripts.experiments.fl_ssl.federated_simulation.config_utils import (
    to_plain_dict,
)
from scripts.experiments.fl_ssl.federated_simulation.data_source_request import (
    resolve_fl_data_source,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedDiagnosticsConfig,
    FederatedLocalTrainerRuntimeConfig,
    FederatedLoraClassifierRuntimeConfig,
    FederatedPrototypeRebuildConfig,
    FederatedQuerySslObjectiveConfig,
    FederatedReportConfig,
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
    prototype_build_strategy = instantiate(cfg.prototype_builder)
    local_update_profile = LocalUpdateProfile.from_mapping(
        to_plain_dict(cfg.local_update_profile)
    )
    execution_plan = _build_execution_plan(cfg)
    training_task_config = _build_training_task_config(
        cfg.training_task,
        task_type=_resolve_training_task_type(cfg=cfg, execution_plan=execution_plan),
        local_update_profile=local_update_profile,
    )
    round_runtime_config = FederatedRoundRuntimeConfig(
        adapter_family_name=str(cfg.round_runtime.adapter_family_name),
        aggregation_backend_name=str(cfg.round_runtime.aggregation_backend_name),
        classifier_head_bootstrap_logit_scale=float(
            cfg.round_runtime.classifier_head_bootstrap_logit_scale
        ),
        lora_classifier=_build_lora_classifier_runtime_config(cfg.round_runtime),
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
        shard_policy=shard_policy,
    )
    return SimulationRunRequest(
        train_rows=fl_data_source.train_rows,
        validation_rows=fl_data_source.validation_rows,
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
        prototype_build_strategy=prototype_build_strategy,
        shard_policy=shard_policy,
        training_task_config=training_task_config,
        validation_config=FederatedValidationConfig(**to_plain_dict(cfg.validation)),
        prototype_rebuild_config=FederatedPrototypeRebuildConfig(
            **to_plain_dict(cfg.prototype_rebuild)
        ),
        diagnostics_config=FederatedDiagnosticsConfig(**to_plain_dict(cfg.diagnostics)),
        ssl_method_config=_build_ssl_method_config(cfg, execution_plan=execution_plan),
        client_pool_split_config=client_pool_split_config,
        materialized_dataset_split=fl_data_source.materialized_dataset_split,
        data_source_config=fl_data_source.data_source_config,
        report_config=FederatedReportConfig(**to_plain_dict(cfg.report)),
        local_update_profile=local_update_profile,
        execution_plan=execution_plan,
        query_ssl_objective_config=FederatedQuerySslObjectiveConfig.from_mapping(
            to_plain_dict(cfg.query_ssl_method),
            strong_view_policy=str(cfg.query_ssl_strong_view_policy),
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(
            device=str(cfg.runtime.device),
            local_files_only=bool(cfg.runtime.local_files_only),
            cache_dir=str(cfg.paper_backbone.cache_dir),
            trust_remote_code=bool(cfg.paper_backbone.trust_remote_code),
            classifier_dropout=float(cfg.paper_backbone.classifier_dropout),
        ),
    )


def _optional_config_str(cfg: DictConfig, key: str) -> str | None:
    value = getattr(cfg, key, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _build_lora_classifier_runtime_config(
    cfg: DictConfig,
) -> FederatedLoraClassifierRuntimeConfig | None:
    if "lora_classifier" not in cfg or cfg.lora_classifier is None:
        return None
    return FederatedLoraClassifierRuntimeConfig.from_mapping(
        to_plain_dict(cfg.lora_classifier)
    )


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
    return FederatedSslMethodConfig(**to_plain_dict(ssl_method))


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
        "update_family": str(cfg.round_runtime.adapter_family_name),
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
