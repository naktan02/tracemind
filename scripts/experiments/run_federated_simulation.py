"""합성 federation simulation을 실행한다."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.federated_simulation import (
    FederatedDiagnosticsConfig,
    FederatedPrototypeRebuildConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    load_jsonl_rows,
    run_simulation,
    split_rows_for_federation,
)
from scripts.run_artifacts import build_run_dir
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)

__all__ = [
    "load_jsonl_rows",
    "main",
    "run_simulation",
    "split_rows_for_federation",
]


def _to_plain_dict(cfg: DictConfig) -> dict[str, object]:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Expected DictConfig section to resolve to a dict.")
    return raw


def _build_training_task_config(cfg: DictConfig) -> FederatedTrainingTaskConfig:
    objective_config = _to_plain_dict(cfg.objective)
    selection_policy = _to_plain_dict(cfg.selection_policy)
    return FederatedTrainingTaskConfig(
        local_epochs=int(cfg.local_epochs),
        batch_size=int(cfg.batch_size),
        learning_rate=float(cfg.learning_rate),
        max_steps=int(cfg.max_steps),
        min_required_examples=int(cfg.min_required_examples),
        gradient_clip_norm=(
            None if cfg.gradient_clip_norm is None else float(cfg.gradient_clip_norm)
        ),
        objective_config=TrainingObjectiveConfig.from_mapping(objective_config),
        selection_policy=TrainingSelectionPolicy.from_mapping(selection_policy),
    )


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="jobs/experiments/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_run_dir(
        cfg.federated_run_preset.output_dir,
        run_id=run_id,
        created_at=created_at,
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    embedding_spec = instantiate(cfg.embedding.spec)
    prototype_build_strategy = instantiate(cfg.prototype_builder)

    result = run_simulation(
        train_rows=load_jsonl_rows(Path(str(cfg.train_jsonl))),
        validation_rows=load_jsonl_rows(Path(str(cfg.validation_jsonl))),
        output_dir=output_dir,
        client_count=int(cfg.federated_run_preset.client_count),
        rounds=int(cfg.federated_run_preset.rounds),
        bootstrap_ratio=float(cfg.federated_run_preset.bootstrap_ratio),
        seed=int(cfg.seed),
        embedding_spec=embedding_spec,
        model_id=str(cfg.published_model_id),
        training_scope=str(cfg.training_algorithm_profile.training_scope),
        round_runtime_config=FederatedRoundRuntimeConfig(
            adapter_family_name=str(cfg.round_runtime.adapter_family_name),
            aggregation_backend_name=str(cfg.round_runtime.aggregation_backend_name),
            classifier_head_bootstrap_logit_scale=float(
                cfg.round_runtime.classifier_head_bootstrap_logit_scale
            ),
        ),
        prototype_build_strategy=prototype_build_strategy,
        shard_policy=FederatedShardPolicyConfig(**_to_plain_dict(cfg.shard_policy)),
        training_task_config=_build_training_task_config(cfg.training_task),
        validation_config=FederatedValidationConfig(**_to_plain_dict(cfg.validation)),
        prototype_rebuild_config=FederatedPrototypeRebuildConfig(
            **_to_plain_dict(cfg.prototype_rebuild)
        ),
        diagnostics_config=FederatedDiagnosticsConfig(
            **_to_plain_dict(cfg.diagnostics)
        ),
        ssl_method_config=FederatedSslMethodConfig(**_to_plain_dict(cfg.ssl_method)),
        report_config=FederatedReportConfig(**_to_plain_dict(cfg.report)),
    )

    print(f"output_dir={output_dir}")
    print(f"initial_model_revision={result.initial_model_revision}")
    print(f"initial_prototype_version={result.initial_prototype_version}")
    print(
        "initial_validation="
        f"accuracy:{result.initial_validation.top1_accuracy:.4f},"
        f"accepted_ratio:{result.initial_validation.accepted_ratio:.4f}"
    )
    if result.rounds:
        last_round = result.rounds[-1]
        print(f"final_model_revision={last_round.model_revision}")
        print(f"final_prototype_version={last_round.prototype_version}")
        print(
            "final_validation="
            f"accuracy:{result.final_validation.top1_accuracy:.4f},"
            f"accepted_ratio:{result.final_validation.accepted_ratio:.4f}"
        )
        print(f"round_count={len(result.rounds)}")
    else:
        print("round_count=0")
        print("note=no client updates satisfied the pseudo-label selection criteria.")
    if result.report_path is not None:
        print(f"report_json={result.report_path}")


if __name__ == "__main__":
    main()
