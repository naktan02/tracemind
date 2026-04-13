"""scripts Hydra config group tests."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module


@pytest.mark.parametrize(
    "config_name",
    [
        "datasets/run_dataset_pipeline",
        "prototypes/seed_prototypes",
        "prototypes/evaluate_prototype_pack",
        "experiments/prototype_strategy",
        "experiments/prototype_threshold_sweep",
        "experiments/run_federated_simulation",
        "experiments/train_softmax_classifier",
        "experiments/train_lora_classifier",
        "experiments/train_lora_pseudo_label_classifier",
        "experiments/train_lora_bootstrap_classifier_teacher",
    ],
)
def test_script_configs_disable_hydra_file_logging(config_name: str) -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name=config_name, return_hydra_config=True)

    assert cfg.hydra.job_logging.root.level == "ERROR"
    assert cfg.hydra.job_logging.disable_existing_loggers is True
    assert "handlers" not in cfg.hydra.job_logging.root
    assert cfg.hydra.hydra_logging.root.level == "ERROR"
    assert cfg.hydra.hydra_logging.disable_existing_loggers is True
    assert "handlers" not in cfg.hydra.hydra_logging.root


def test_seed_prototypes_default_runtime_is_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="prototypes/seed_prototypes")

    assert cfg.dataset.name == "ourafla"
    assert cfg.embedding.backend == "transformers_mxbai"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is False
    assert cfg.prototype_builder.name == "single"


def test_seed_prototypes_supports_short_builder_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="prototypes/seed_prototypes",
            overrides=[
                "prototype_builder=kmeans",
                "prototype_builder.candidate_ks=[2]",
            ],
        )

    assert cfg.prototype_builder.name == "kmeans"
    assert list(cfg.prototype_builder.candidate_ks) == [2]


def test_prototype_strategy_supports_short_group_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/prototype_strategy",
            overrides=[
                "runtime=gpu_local",
                "embedding=hash_debug",
                "strategy.name=kmeans",
                "strategy.kmeans_candidate_ks=[2]",
                "runner.score_policy_name=topk_mean_cosine",
                "runner.score_top_k=2",
            ],
        )

    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is True
    assert cfg.embedding.backend == "hash_debug"
    assert cfg.embedding.model_id == "hash_debug"
    assert cfg.strategy.name == "kmeans"
    assert list(cfg.strategy.kmeans_candidate_ks) == [2]
    assert cfg.runner.score_policy_name == "topk_mean_cosine"
    assert cfg.runner.score_top_k == 2


def test_train_lora_classifier_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_classifier",
            overrides=["runtime=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_classifier_supports_train_source_and_run_preset_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_classifier",
            overrides=[
                "lora_train_source=bootstrap_teacher_split30_2026_04_14",
                "lora_run_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.lora_train_source.name == "bootstrap_teacher_split30_2026_04_14"
    assert cfg.train_jsonl.endswith("teacher_seed_train.jsonl")
    assert cfg.lora_run_preset.name == "smoke_verbose_e1"
    assert cfg.epochs == 1
    assert cfg.log_every_steps == 20


def test_train_lora_pseudo_label_classifier_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_pseudo_label_classifier",
            overrides=["runtime=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_pseudo_label_classifier_supports_train_source_and_run_preset_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_pseudo_label_classifier",
            overrides=[
                "lora_train_source=bootstrap_teacher_split30_2026_04_14",
                "lora_run_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.lora_train_source.name == "bootstrap_teacher_split30_2026_04_14"
    assert cfg.train_jsonl.endswith("teacher_seed_train.jsonl")
    assert cfg.lora_run_preset.name == "smoke_verbose_e1"
    assert cfg.epochs == 1
    assert cfg.log_every_steps == 20


def test_train_lora_bootstrap_classifier_teacher_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_bootstrap_classifier_teacher",
            overrides=["runtime=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_bootstrap_classifier_teacher_supports_source_and_run_preset_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_bootstrap_classifier_teacher",
            overrides=[
                "bootstrap_teacher_source=bootstrap_teacher_split30_2026_04_14",
                "lora_run_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.bootstrap_teacher_source.name == "bootstrap_teacher_split30_2026_04_14"
    assert cfg.teacher_train_jsonl.endswith("teacher_seed_train.jsonl")
    assert cfg.teacher_unlabeled_jsonl.endswith("teacher_unlabeled_pool.jsonl")
    assert cfg.lora_run_preset.name == "smoke_verbose_e1"
    assert cfg.epochs == 1
    assert cfg.log_every_steps == 20


def test_threshold_sweep_supports_short_leaf_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/prototype_threshold_sweep",
            overrides=[
                "strategy.name=single",
                "runner.score_policy_name=topk_mean_cosine",
                "runner.score_top_k=2",
            ],
        )

    assert cfg.strategy.name == "single"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runner.score_policy_name == "topk_mean_cosine"
    assert cfg.runner.score_top_k == 2
    assert len(cfg.threshold_policies) == 3
    assert (
        cfg.threshold_policies[0]._target_
        == (
            "scripts.experiments.prototype_strategy.threshold_policies."
            "FixMatchFixedConfidencePolicy"
        )
    )
    assert (
        cfg.threshold_policies[2]._target_
        == (
            "scripts.experiments.prototype_strategy.threshold_policies."
            "ClasswiseStaticConfidencePolicy"
        )
    )


def test_federated_simulation_uses_smoke_preset_by_default() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/run_federated_simulation")

    assert cfg.federated_run_preset.name == "smoke"
    assert cfg.training_algorithm_profile.algorithm_profile_name == (
        "prototype_pseudo_label_v1"
    )
    assert cfg.round_runtime.classifier_head_bootstrap_logit_scale == 8.0
    assert cfg.federated_run_preset.output_dir == "runs/federated_simulation_smoke"
    assert cfg.federated_run_preset.client_count == 4
    assert cfg.federated_run_preset.rounds == 1


def test_federated_simulation_supports_short_preset_and_leaf_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/run_federated_simulation",
            overrides=[
                "federated_run_preset=standard",
                "federated_run_preset.rounds=3",
                "federated_run_preset.client_count=8",
                "prototype_builder=kmeans",
                "prototype_builder.candidate_ks=[2]",
            ],
        )

    assert cfg.federated_run_preset.name == "standard"
    assert cfg.federated_run_preset.rounds == 3
    assert cfg.federated_run_preset.client_count == 8
    assert cfg.federated_run_preset.output_dir == "runs/federated_simulation"
    assert cfg.prototype_builder.name == "kmeans"
    assert list(cfg.prototype_builder.candidate_ks) == [2]


def test_federated_simulation_supports_detail_strategy_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/run_federated_simulation",
            overrides=[
                "training_algorithm_profile=prototype_top1_confidence_v1",
                "shard_policy.dominant_ratio=0.6",
                "training_task.objective.example_generation_backend_name=prototype_rescore",
                "training_task.objective.evidence_backend_name=prototype_similarity_evidence",
                "training_task.objective.scorer_backend_name=prototype_similarity",
                "training_task.objective.score_policy_name=topk_mean_cosine",
                "training_task.objective.score_top_k=2",
                "validation.scorer_backend_name=prototype_similarity",
                "validation.score_policy_name=topk_mean_cosine",
                "validation.score_top_k=2",
                "prototype_rebuild.mapping_version=custom_mapping.v1",
                "diagnostics.dump_dir_name=custom_dumps",
            ],
        )

    assert cfg.shard_policy.dominant_ratio == 0.6
    assert cfg.training_task.objective.algorithm_profile_name == (
        "prototype_top1_confidence_v1"
    )
    assert (
        cfg.training_task.objective.example_generation_backend_name
        == "prototype_rescore"
    )
    assert (
        cfg.training_task.objective.evidence_backend_name
        == "prototype_similarity_evidence"
    )
    assert cfg.training_task.objective.scorer_backend_name == "prototype_similarity"
    assert cfg.training_task.objective.score_policy_name == "topk_mean_cosine"
    assert cfg.training_task.objective.score_top_k == 2
    assert cfg.validation.scorer_backend_name == "prototype_similarity"
    assert cfg.validation.score_policy_name == "topk_mean_cosine"
    assert cfg.validation.score_top_k == 2
    assert cfg.prototype_rebuild.mapping_version == "custom_mapping.v1"
    assert cfg.diagnostics.dump_dir_name == "custom_dumps"



def test_train_lora_classifier_defaults_to_gpu_online_and_fixed_lora_scaffold() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/train_lora_classifier")

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.selection_set == "validation"


def test_train_lora_pseudo_label_classifier_defaults_to_gpu_online_and_fixed_lora_scaffold() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/train_lora_pseudo_label_classifier")

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.selection_set == "validation"
    assert cfg.pseudo_label_jsonl is None


def test_train_lora_bootstrap_classifier_teacher_defaults_to_classifier_teacher_then_fixed_lora_student() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_bootstrap_classifier_teacher"
        )

    assert cfg.dataset.name == "ourafla"
    assert cfg.embedding.backend == "transformers_mxbai"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.pseudo_label_confidence_threshold == 0.6
    assert cfg.pseudo_label_margin_threshold == 0.02
    assert cfg.bootstrap_split.enabled is False


def test_dataset_pipeline_defaults_to_ourafla_and_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="datasets/run_dataset_pipeline")

    assert cfg.dataset.name == "ourafla"
    assert cfg.dataset.test_jsonl == cfg.dataset.test_labeled_jsonl
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is False
    assert cfg.prototype_builder.name == "single"
