"""합성 federation simulation을 실행한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from methods.federated_ssl.execution_plan import (
    FederatedSslExecutionPlan,
    build_federated_ssl_execution_plan,
)
from methods.federated_ssl.local_update_profile import (
    LocalUpdateProfile,
    require_training_objective_matches_local_update_profile,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from scripts.experiments.fl_ssl.federated_simulation.io.client_split_manifest import (
    LoadedFlClientSplit,
    load_materialized_client_split,
)
from scripts.experiments.fl_ssl.federated_simulation.io.rows import load_jsonl_rows
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT,
    FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN,
    FederatedClientPoolSplitConfig,
    FederatedDatasetSplit,
    FederatedDataSourceConfig,
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
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    run_simulation_request,
)
from scripts.experiments.fl_ssl.run_layout import build_fl_ssl_run_dir
from scripts.experiments.fl_ssl.run_safety import require_fl_ssl_run_budget_allowed
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_federated_training_task_config,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


@dataclass(slots=True)
class _ResolvedFlDataSource:
    train_rows: list[LabeledQueryRow]
    validation_rows: list[LabeledQueryRow]
    materialized_dataset_split: FederatedDatasetSplit | None
    data_source_config: FederatedDataSourceConfig


def _to_plain_dict(cfg: DictConfig) -> dict[str, object]:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Expected DictConfig section to resolve to a dict.")
    return raw


def _build_training_task_config(
    cfg: DictConfig,
    *,
    task_type: str,
    local_update_profile: LocalUpdateProfile,
) -> FederatedTrainingTaskConfig:
    objective_config = _to_plain_dict(cfg.objective)
    selection_policy = _to_plain_dict(cfg.selection_policy)
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
        _to_plain_dict(cfg.lora_classifier)
    )


def _build_execution_plan(cfg: DictConfig) -> FederatedSslExecutionPlan:
    descriptor = resolve_federated_ssl_method_descriptor(str(cfg.ssl_method.name))
    fl_method = _to_plain_dict(cfg.fl_method)
    return build_federated_ssl_execution_plan(
        fl_method=_with_inferred_manual_axes(cfg=cfg, fl_method=fl_method),
        security_policy=_to_plain_dict(cfg.security_policy),
        method_descriptor=descriptor,
    )


def _with_inferred_manual_axes(
    *,
    cfg: DictConfig,
    fl_method: dict[str, object],
) -> dict[str, object]:
    if str(fl_method.get("composition_mode", "method_owned")) != "manual":
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
    return str(cfg.ssl_method.client_step.task_type)


def build_simulation_request_from_config(
    cfg: DictConfig,
    *,
    output_dir: Path,
    seed: int | None = None,
) -> SimulationRunRequest:
    embedding_spec = instantiate(cfg.embedding.spec)
    prototype_build_strategy = instantiate(cfg.prototype_builder)
    local_update_profile = LocalUpdateProfile.from_mapping(
        _to_plain_dict(cfg.local_update_profile)
    )
    training_task_config = _build_training_task_config(
        cfg.training_task,
        task_type=str(cfg.ssl_method.client_step.task_type),
        local_update_profile=local_update_profile,
    )
    execution_plan = _build_execution_plan(cfg)
    round_runtime_config = FederatedRoundRuntimeConfig(
        adapter_family_name=str(cfg.round_runtime.adapter_family_name),
        aggregation_backend_name=str(cfg.round_runtime.aggregation_backend_name),
        classifier_head_bootstrap_logit_scale=float(
            cfg.round_runtime.classifier_head_bootstrap_logit_scale
        ),
        lora_classifier=_build_lora_classifier_runtime_config(cfg.round_runtime),
    )
    actual_seed = int(cfg.seed if seed is None else seed)
    shard_policy = FederatedShardPolicyConfig(**_to_plain_dict(cfg.shard_policy))
    client_pool_split_config = FederatedClientPoolSplitConfig(
        **_to_plain_dict(cfg.client_pool_split)
    )
    fl_data_source = _resolve_fl_data_source(
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
        embedding_spec=embedding_spec,
        model_id=str(cfg.published_model_id),
        training_scope=local_update_profile.training_scope,
        round_runtime_config=round_runtime_config,
        prototype_build_strategy=prototype_build_strategy,
        shard_policy=shard_policy,
        training_task_config=training_task_config,
        validation_config=FederatedValidationConfig(**_to_plain_dict(cfg.validation)),
        prototype_rebuild_config=FederatedPrototypeRebuildConfig(
            **_to_plain_dict(cfg.prototype_rebuild)
        ),
        diagnostics_config=FederatedDiagnosticsConfig(
            **_to_plain_dict(cfg.diagnostics)
        ),
        ssl_method_config=FederatedSslMethodConfig(**_to_plain_dict(cfg.ssl_method)),
        client_pool_split_config=client_pool_split_config,
        materialized_dataset_split=fl_data_source.materialized_dataset_split,
        data_source_config=fl_data_source.data_source_config,
        report_config=FederatedReportConfig(**_to_plain_dict(cfg.report)),
        local_update_profile=local_update_profile,
        execution_plan=execution_plan,
        query_ssl_objective_config=FederatedQuerySslObjectiveConfig.from_mapping(
            _to_plain_dict(cfg.query_ssl_method),
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


def _resolve_fl_data_source(
    *,
    cfg: DictConfig,
    client_count: int,
    bootstrap_ratio: float,
    shard_policy: FederatedShardPolicyConfig,
) -> _ResolvedFlDataSource:
    fl_data_cfg = cfg.get("fl_data", {})
    source_mode = str(
        fl_data_cfg.get("source_mode", FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN)
    )
    if source_mode == FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN:
        return _ResolvedFlDataSource(
            train_rows=load_jsonl_rows(Path(str(cfg.train_jsonl))),
            validation_rows=load_jsonl_rows(Path(str(cfg.validation_jsonl))),
            materialized_dataset_split=None,
            data_source_config=FederatedDataSourceConfig(
                source_mode=source_mode,
                source_selection=_optional_plain_dict(cfg, "query_data_selection"),
                source_jsonl={
                    "train": str(cfg.train_jsonl),
                    "validation": str(cfg.validation_jsonl),
                },
            ),
        )
    if source_mode != FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT:
        return _unsupported_fl_data_source(source_mode)

    raw_manifest_path = fl_data_cfg.get("split_manifest")
    if raw_manifest_path is None:
        raise ValueError(
            "fl_data.split_manifest is required when fl_data.source_mode is "
            "materialized_client_split."
        )
    loaded_split = load_materialized_client_split(Path(str(raw_manifest_path)))
    _require_manifest_matches_config(
        loaded_split=loaded_split,
        cfg=cfg,
        client_count=client_count,
        bootstrap_ratio=bootstrap_ratio,
        shard_policy=shard_policy,
    )
    manifest = loaded_split.manifest
    return _ResolvedFlDataSource(
        train_rows=loaded_split.train_rows,
        validation_rows=loaded_split.validation_rows,
        materialized_dataset_split=loaded_split.dataset_split,
        data_source_config=FederatedDataSourceConfig(
            source_mode=source_mode,
            split_manifest_path=str(manifest.manifest_path or raw_manifest_path),
            split_manifest_sha256=manifest.manifest_sha256,
            split_id=manifest.split_id,
            source_selection=dict(manifest.source_selection),
            source_jsonl=dict(manifest.source_jsonl),
            labeled_policy=dict(manifest.labeled_policy),
            view_schema=manifest.view_schema.to_payload(),
            test_jsonl=manifest.test_jsonl,
        ),
    )


def _unsupported_fl_data_source(source_mode: str) -> _ResolvedFlDataSource:
    supported_modes = [
        FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN,
        FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT,
    ]
    raise ValueError(
        "Unsupported fl_data.source_mode. "
        f"Expected one of {supported_modes}, got {source_mode!r}."
    )


def _require_manifest_matches_config(
    *,
    loaded_split: LoadedFlClientSplit,
    cfg: DictConfig,
    client_count: int,
    bootstrap_ratio: float,
    shard_policy: FederatedShardPolicyConfig,
) -> None:
    manifest = loaded_split.manifest
    if manifest.client_count != client_count:
        raise ValueError(
            "fl_data materialized manifest client_count must match "
            f"federated_run_budget.client_count: {manifest.client_count} != "
            f"{client_count}."
        )
    if abs(manifest.bootstrap_ratio - bootstrap_ratio) > 1e-9:
        raise ValueError(
            "fl_data materialized manifest bootstrap_ratio must match "
            f"federated_run_budget.bootstrap_ratio: {manifest.bootstrap_ratio} != "
            f"{bootstrap_ratio}."
        )
    _require_manifest_shard_policy_matches_config(manifest.shard_policy, shard_policy)
    configured_source_selection = _optional_plain_dict(cfg, "query_data_selection")
    if (
        configured_source_selection
        and manifest.source_selection
        and manifest.source_selection != configured_source_selection
    ):
        raise ValueError(
            "fl_data materialized manifest source_selection must match "
            f"query_data_selection: {manifest.source_selection} != "
            f"{configured_source_selection}."
        )


def _require_manifest_shard_policy_matches_config(
    manifest_policy: dict[str, object],
    shard_policy: FederatedShardPolicyConfig,
) -> None:
    expected = {
        "name": shard_policy.name,
        "client_id_prefix": shard_policy.client_id_prefix,
        "dominant_ratio": shard_policy.dominant_ratio,
        "alpha": shard_policy.alpha,
    }
    for key, expected_value in expected.items():
        actual_value = manifest_policy.get(key)
        if actual_value != expected_value:
            raise ValueError(
                "fl_data materialized manifest shard_policy must match config: "
                f"{key} {actual_value!r} != {expected_value!r}."
            )


def _optional_plain_dict(cfg: DictConfig, section_name: str) -> dict[str, object]:
    if section_name not in cfg:
        return {}
    return _to_plain_dict(cfg[section_name])


def render_simulation_result_lines(
    *,
    output_dir: Path,
    result,
) -> list[str]:
    lines = [
        f"output_dir={output_dir}",
        f"initial_model_revision={result.initial_model_revision}",
        f"initial_prototype_version={result.initial_prototype_version}",
        (
            "initial_validation="
            f"accuracy:{result.initial_validation.top1_accuracy:.4f},"
            f"loss:{result.initial_validation.loss:.4f},"
            f"macro_f1:{result.initial_validation.macro_f1:.4f},"
            f"ece:{result.initial_validation.expected_calibration_error:.4f},"
            f"accepted_ratio:{result.initial_validation.accepted_ratio:.4f}"
        ),
    ]
    if result.rounds:
        last_round = result.rounds[-1]
        lines.extend(
            [
                f"final_model_revision={last_round.model_revision}",
                f"final_prototype_version={last_round.prototype_version}",
                (
                    "final_validation="
                    f"accuracy:{result.final_validation.top1_accuracy:.4f},"
                    f"loss:{result.final_validation.loss:.4f},"
                    f"macro_f1:{result.final_validation.macro_f1:.4f},"
                    f"ece:{result.final_validation.expected_calibration_error:.4f},"
                    f"accepted_ratio:{result.final_validation.accepted_ratio:.4f}"
                ),
                f"round_count={len(result.rounds)}",
            ]
        )
    else:
        lines.extend(
            [
                "round_count=0",
                "note=no client updates satisfied the pseudo-label selection criteria.",
            ]
        )
    if result.report_path is not None:
        lines.append(f"report_json={result.report_path}")
    return lines


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/fl_ssl/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    require_fl_ssl_run_budget_allowed(
        cfg,
        run_kind="single_simulation",
    )
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_fl_ssl_run_dir(
        cfg.federated_run_budget.output_dir,
        cfg=cfg,
        run_id=run_id,
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    result = run_simulation_request(
        build_simulation_request_from_config(
            cfg,
            output_dir=output_dir,
        )
    )
    for line in render_simulation_result_lines(output_dir=output_dir, result=result):
        print(line)


if __name__ == "__main__":
    main()
