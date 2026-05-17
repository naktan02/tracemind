"""합성 federation simulation을 실행한다."""

from __future__ import annotations

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
from scripts.artifacts.run_artifacts import build_run_dir
from scripts.experiments.fl_ssl.federated_simulation.io.rows import load_jsonl_rows
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedDiagnosticsConfig,
    FederatedLoraClassifierRuntimeConfig,
    FederatedPrototypeRebuildConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedValidationConfig,
    SimulationRunRequest,
)
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    run_simulation_request,
)
from scripts.runtime_adapters.federated_server.round_request_mapper import (
    build_federated_training_task_config,
)
from scripts.runtime_adapters.federated_server.task_config_surface import (
    FederatedTrainingTaskConfig,
)
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)


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
        "client_ssl_objective": str(cfg.ssl_method.client_step.task_type),
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
    return SimulationRunRequest(
        train_rows=load_jsonl_rows(Path(str(cfg.train_jsonl))),
        validation_rows=load_jsonl_rows(Path(str(cfg.validation_jsonl))),
        output_dir=output_dir,
        client_count=int(cfg.federated_run_budget.client_count),
        rounds=int(cfg.federated_run_budget.rounds),
        bootstrap_ratio=float(cfg.federated_run_budget.bootstrap_ratio),
        seed=int(cfg.seed if seed is None else seed),
        embedding_spec=embedding_spec,
        model_id=str(cfg.published_model_id),
        training_scope=local_update_profile.training_scope,
        round_runtime_config=round_runtime_config,
        prototype_build_strategy=prototype_build_strategy,
        shard_policy=FederatedShardPolicyConfig(**_to_plain_dict(cfg.shard_policy)),
        training_task_config=training_task_config,
        validation_config=FederatedValidationConfig(**_to_plain_dict(cfg.validation)),
        prototype_rebuild_config=FederatedPrototypeRebuildConfig(
            **_to_plain_dict(cfg.prototype_rebuild)
        ),
        diagnostics_config=FederatedDiagnosticsConfig(
            **_to_plain_dict(cfg.diagnostics)
        ),
        ssl_method_config=FederatedSslMethodConfig(**_to_plain_dict(cfg.ssl_method)),
        client_pool_split_config=FederatedClientPoolSplitConfig(
            **_to_plain_dict(cfg.client_pool_split)
        ),
        report_config=FederatedReportConfig(**_to_plain_dict(cfg.report)),
        local_update_profile=local_update_profile,
        execution_plan=execution_plan,
    )


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
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_run_dir(
        cfg.federated_run_budget.output_dir,
        run_id=run_id,
        created_at=created_at,
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
