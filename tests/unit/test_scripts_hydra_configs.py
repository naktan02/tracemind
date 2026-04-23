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
        "experiments/train_lora_fixmatch",
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
    assert list(cfg.fixed_categories) == list(cfg.dataset.prototype_expected_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == "none"
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


def test_train_lora_pseudo_label_classifier_supports_auto_local_runtime_override() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_pseudo_label_classifier",
            overrides=["runtime=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_fixmatch_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_fixmatch",
            overrides=["runtime=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_fixmatch_supports_source_and_run_preset_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_fixmatch",
            overrides=[
                "query_ssl_train_source=bootstrap_teacher_split30_2026_04_14",
                "lora_run_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.query_ssl_train_source.name == "bootstrap_teacher_split30_2026_04_14"
    assert cfg.train_jsonl.endswith("teacher_seed_train.jsonl")
    assert cfg.unlabeled_jsonl.endswith("teacher_unlabeled_pool.jsonl")
    assert cfg.lora_run_preset.name == "smoke_verbose_e1"
    assert cfg.epochs == 1
    assert cfg.log_every_steps == 20
    assert list(cfg.fixed_categories) == list(cfg.dataset.prototype_expected_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == (
        "canonical_fixed_classifier_seed"
    )
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


def test_train_lora_fixmatch_supports_query_ssl_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_fixmatch",
            overrides=[
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.hard_label is True


def test_train_lora_fixmatch_supports_query_ssl_augmenter_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_fixmatch",
            overrides=["query_ssl_augmenter=precomputed_usb_candidates_v1"],
        )

    assert cfg.query_ssl_augmenter.name == "precomputed_usb_candidates_v1"
    assert cfg.query_ssl_augmenter.augmenter_type == "precomputed_usb_candidates"


def test_train_lora_pseudo_label_classifier_supports_train_source_and_run_preset_overrides(  # noqa: E501
) -> (
    None
):
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
    assert cfg.include_seed_train_rows is False
    assert list(cfg.fixed_categories) == list(cfg.dataset.prototype_expected_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == (
        "canonical_fixed_classifier_seed"
    )
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


def test_train_lora_pseudo_label_classifier_supports_pseudo_label_algorithm_override() -> (  # noqa: E501
    None
):
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_pseudo_label_classifier",
            overrides=["pseudo_label_algorithm=fixed_confidence_095"],
        )

    assert cfg.pseudo_label_algorithm.name == "fixed_confidence_095"
    assert cfg.pseudo_label_algorithm.confidence_threshold == 0.95
    assert cfg.pseudo_label_algorithm.margin_threshold == 0.0
    assert cfg.pseudo_label_algorithm.algorithm_name == (
        "top1_confidence_only"
    )


def test_train_lora_bootstrap_classifier_teacher_supports_auto_local_runtime_override(  # noqa: E501
) -> (
    None
):
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_bootstrap_classifier_teacher",
            overrides=["runtime=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_bootstrap_classifier_teacher_supports_source_and_run_preset_overrides(  # noqa: E501
) -> (
    None
):
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
    assert (
        cfg.teacher_reuse_manifest_path
        == "data/processed/classifier_heads/clf_2026_04_11_143138.manifest.json"
    )
    assert cfg.student_include_seed_train_rows is False
    assert list(cfg.fixed_categories) == list(cfg.dataset.prototype_expected_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == (
        "canonical_fixed_classifier_seed"
    )
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


def test_train_lora_bootstrap_classifier_teacher_supports_pseudo_label_algorithm_override() -> (  # noqa: E501
    None
):
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/train_lora_bootstrap_classifier_teacher",
            overrides=["pseudo_label_algorithm=fixed_confidence_095"],
        )

    assert cfg.pseudo_label_algorithm.name == "fixed_confidence_095"
    assert cfg.pseudo_label_algorithm.confidence_threshold == 0.95
    assert cfg.pseudo_label_algorithm.margin_threshold == 0.0
    assert cfg.pseudo_label_algorithm.algorithm_name == (
        "top1_confidence_only"
    )


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
    assert cfg.threshold_policies[0]._target_ == (
        "scripts.experiments.prototype_strategy.threshold_policies."
        "FixMatchFixedConfidencePolicy"
    )
    assert cfg.threshold_policies[2]._target_ == (
        "scripts.experiments.prototype_strategy.threshold_policies."
        "ClasswiseStaticConfidencePolicy"
    )


def test_federated_simulation_uses_smoke_preset_by_default() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/run_federated_simulation")

    assert cfg.federated_run_preset.name == "smoke"
    assert cfg.training_algorithm_profile.algorithm_profile_name == (
        "prototype_pseudo_label_v1"
    )
    assert cfg.round_runtime.classifier_head_bootstrap_logit_scale == 8.0
    assert cfg.training_task.objective.algorithm_profile_name == (
        "prototype_pseudo_label_v1"
    )
    assert cfg.validation.confidence_threshold == 0.6
    assert cfg.validation.margin_threshold == 0.02
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


def test_train_lora_pseudo_label_classifier_defaults_to_gpu_online_and_fixed_lora_scaffold(  # noqa: E501
) -> (
    None
):
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/train_lora_pseudo_label_classifier")

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.selection_set == "validation"
    assert cfg.pseudo_label_algorithm.name == "margin_threshold_v1"
    assert cfg.pseudo_label_jsonl is None


def test_train_lora_fixmatch_defaults_to_gpu_online_and_usb_fixmatch_method() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/train_lora_fixmatch")

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.selection_set == "validation"
    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "fixmatch"
    assert cfg.query_ssl_method.temperature == 0.5
    assert cfg.query_ssl_method.p_cutoff == 0.95
    assert cfg.query_ssl_method.lambda_u == 1.0
    assert cfg.query_ssl_method.supervised_loss_weight == 1.0
    assert cfg.query_ssl_augmenter.name == "backtranslation_nllb_en_de_fr_usb_v1"
    assert cfg.query_ssl_augmenter.source_lang == "eng_Latn"
    assert list(cfg.query_ssl_augmenter.pivot_languages) == [
        "deu_Latn",
        "fra_Latn",
    ]
    assert cfg.query_ssl_augmenter.torch_dtype == "auto"
    assert cfg.unlabeled_jsonl == cfg.query_ssl_train_source.unlabeled_jsonl


def test_train_lora_bootstrap_classifier_teacher_defaults_to_classifier_teacher_then_fixed_lora_student(  # noqa: E501
) -> (
    None
):
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/train_lora_bootstrap_classifier_teacher")

    assert cfg.dataset.name == "ourafla"
    assert cfg.embedding.backend == "transformers_mxbai"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.pseudo_label_algorithm.name == "margin_threshold_v1"
    assert cfg.pseudo_label_algorithm.confidence_threshold == 0.6
    assert cfg.pseudo_label_algorithm.margin_threshold == 0.02
    assert cfg.pseudo_label_algorithm.algorithm_name == (
        "top1_margin_threshold"
    )
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
