"""scripts Hydra config group tests."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from methods.federated_ssl.compatibility import (
    FederatedSslProfileCompatibilityContext,
    validate_federated_ssl_profile_compatibility,
)
from methods.federated_ssl.experiment_profile import FederatedSslExperimentProfile
from methods.federated_ssl.local_update_profile import (
    LocalUpdateProfile,
    require_training_objective_matches_local_update_profile,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def _plain_dict(source: DictConfig) -> dict[str, object]:
    raw = OmegaConf.to_container(source, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Expected DictConfig section to resolve to a dict.")
    return raw


@pytest.mark.parametrize(
    "config_name",
    [
        "entrypoints/data_pipeline/run_dataset_pipeline",
        "entrypoints/prototype_pack/seed_prototypes",
        "entrypoints/prototype_pack/evaluate_prototype_pack",
        "entrypoints/prototype_analysis/prototype_strategy",
        "entrypoints/prototype_analysis/prototype_threshold_sweep",
        "entrypoints/fl_ssl/run_federated_simulation",
        "entrypoints/central_classifier_seed/train_softmax_classifier",
        "entrypoints/central_ssl_control/train_lora_classifier",
        "entrypoints/central_ssl_control/train_lora_fixmatch",
        "entrypoints/central_ssl_control/train_lora_pseudo_label_classifier",
        "entrypoints/central_ssl_control/train_lora_bootstrap_classifier_teacher",
    ],
)
def test_script_configs_disable_hydra_file_logging(config_name: str) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name=config_name, return_hydra_config=True)

    assert cfg.hydra.job_logging.root.level == "ERROR"
    assert cfg.hydra.job_logging.disable_existing_loggers is True
    assert "handlers" not in cfg.hydra.job_logging.root
    assert cfg.hydra.hydra_logging.root.level == "ERROR"
    assert cfg.hydra.hydra_logging.disable_existing_loggers is True
    assert "handlers" not in cfg.hydra.hydra_logging.root


def test_seed_prototypes_default_runtime_is_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/prototype_pack/seed_prototypes")

    assert cfg.dataset.name == "ourafla"
    assert cfg.embedding.backend == "transformers_mxbai"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is False
    assert cfg.prototype_builder.name == "single"


@pytest.mark.parametrize(
    ("embedding_override", "expected_backend"),
    [
        ("execution_context/embedding_adapter=mxbai", "transformers_mxbai"),
        ("execution_context/embedding_adapter=hash_debug", "hash_debug"),
    ],
)
def test_embedding_adapter_spec_config_instantiates(
    embedding_override: str,
    expected_backend: str,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                embedding_override,
                "execution_context/runtime_env=cpu_local",
            ],
        )

    embedding_spec = instantiate(cfg.embedding.spec)

    assert isinstance(embedding_spec, EmbeddingAdapterSpec)
    assert embedding_spec.backend == expected_backend
    assert embedding_spec.device == "cpu"


def test_seed_prototypes_supports_short_builder_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/prototype_pack/seed_prototypes",
            overrides=[
                "strategy_axes/prototype/build_strategy=kmeans",
                "prototype_builder.candidate_ks=[2]",
            ],
        )

    assert cfg.prototype_builder.name == "kmeans"
    assert list(cfg.prototype_builder.candidate_ks) == [2]


def test_prototype_strategy_supports_short_group_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/prototype_analysis/prototype_strategy",
            overrides=[
                "execution_context/runtime_env=gpu_local",
                "execution_context/embedding_adapter=hash_debug",
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
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_classifier",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_classifier_supports_train_source_and_run_preset_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_classifier",
            overrides=[
                "track_presets/central_ssl_control/query_source=bootstrap_teacher_split30_2026_04_14",
                "track_presets/central_ssl_control/training_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.query_source.name == "bootstrap_teacher_split30_2026_04_14"
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
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_pseudo_label_classifier",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_fixmatch_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_fixmatch",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_fixmatch_supports_source_and_run_preset_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_fixmatch",
            overrides=[
                "track_presets/central_ssl_control/query_source=bootstrap_teacher_split30_2026_04_14",
                "track_presets/central_ssl_control/training_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.query_source.name == "bootstrap_teacher_split30_2026_04_14"
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
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_fixmatch",
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
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_fixmatch",
            overrides=["strategy_axes/ssl/augmentation=precomputed_usb_candidates_v1"],
        )

    assert cfg.query_ssl_augmenter.name == "precomputed_usb_candidates_v1"
    assert cfg.query_ssl_augmenter.augmenter_type == "precomputed_usb_candidates"


def test_train_lora_pseudo_label_classifier_supports_train_source_and_run_preset_overrides(  # noqa: E501
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_pseudo_label_classifier",
            overrides=[
                "track_presets/central_ssl_control/query_source=bootstrap_teacher_split30_2026_04_14",
                "track_presets/central_ssl_control/training_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.query_source.name == "bootstrap_teacher_split30_2026_04_14"
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
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_pseudo_label_classifier",
            overrides=["strategy_axes/ssl/pseudo_label_selection=fixed_confidence_095"],
        )

    assert cfg.pseudo_label_algorithm.name == "fixed_confidence_095"
    assert cfg.pseudo_label_algorithm.confidence_threshold == 0.95
    assert cfg.pseudo_label_algorithm.margin_threshold == 0.0
    assert cfg.pseudo_label_algorithm.algorithm_name == ("top1_confidence_only")


def test_train_lora_bootstrap_classifier_teacher_supports_auto_local_runtime_override(  # noqa: E501
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_bootstrap_classifier_teacher",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_bootstrap_classifier_teacher_supports_source_and_run_preset_overrides(  # noqa: E501
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_bootstrap_classifier_teacher",
            overrides=[
                "track_presets/central_ssl_control/query_source=bootstrap_teacher_split30_2026_04_14",
                "track_presets/central_ssl_control/training_preset=smoke_verbose_e1",
            ],
        )

    assert cfg.query_source.name == "bootstrap_teacher_split30_2026_04_14"
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
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_bootstrap_classifier_teacher",
            overrides=["strategy_axes/ssl/pseudo_label_selection=fixed_confidence_095"],
        )

    assert cfg.pseudo_label_algorithm.name == "fixed_confidence_095"
    assert cfg.pseudo_label_algorithm.confidence_threshold == 0.95
    assert cfg.pseudo_label_algorithm.margin_threshold == 0.0
    assert cfg.pseudo_label_algorithm.algorithm_name == ("top1_confidence_only")


def test_threshold_sweep_supports_short_leaf_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/prototype_analysis/prototype_threshold_sweep",
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
        "scripts.experiments.prototype_analysis.prototype_strategy.threshold_policies."
        "FixMatchFixedConfidencePolicy"
    )
    assert cfg.threshold_policies[2]._target_ == (
        "scripts.experiments.prototype_analysis.prototype_strategy.threshold_policies."
        "ClasswiseStaticConfidencePolicy"
    )


def test_federated_simulation_uses_smoke_preset_by_default() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    assert cfg.federated_run_preset.name == "smoke"
    assert cfg.local_update_profile.algorithm_profile_name == (
        "prototype_pseudo_label_v1"
    )
    assert cfg.fl_profile.name == "fedavg_pseudo_label_diagonal_scale_v1"
    assert cfg.fl_profile.method_name == "fedavg_pseudo_label"
    assert cfg.fl_profile.local_update_profile_name == "prototype_pseudo_label_v1"
    assert cfg.fl_profile.round_runtime_profile_name == "fedavg_diagonal_scale"
    assert cfg.round_runtime_profile.name == "fedavg_diagonal_scale"
    assert cfg.round_runtime.adapter_family_name == "diagonal_scale"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.round_runtime.classifier_head_bootstrap_logit_scale == 8.0
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.lora.name == "default"
    assert cfg.training_task.objective.algorithm_profile_name == (
        "prototype_pseudo_label_v1"
    )
    assert cfg.validation.confidence_threshold == 0.6
    assert cfg.validation.margin_threshold == 0.02
    assert cfg.federated_run_preset.output_dir == "runs/federated_simulation_smoke"
    assert cfg.federated_run_preset.client_count == 4
    assert cfg.federated_run_preset.rounds == 3
    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.local_files_only is True
    assert cfg.seed_sweep.output_dir == "runs/federated_simulation_seed_sweep"
    assert list(cfg.seed_sweep.seeds) == [42, 43, 44]
    assert cfg.shard_policy.name == "label_dominant"
    assert cfg.shard_policy.dominant_ratio == 0.75
    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert cfg.ssl_method.method_role == "baseline"
    assert cfg.ssl_method.implementation_status == "active_runtime"
    assert cfg.ssl_method.client_step.owner == "agent"
    assert cfg.ssl_method.server_step.aggregation_backend_name == "fedavg"
    assert cfg.ssl_method.round_state_exchange.exchange_name == "none"
    assert cfg.report.track == "fl_ssl_main_comparison"
    assert cfg.report.table_role == "main_comparison"
    assert cfg.client_pool_split.labeled_ratio == 0.1
    assert cfg.client_pool_split.unlabeled_ratio == 0.9
    assert cfg.report.labeled_ratio == 0.1
    assert cfg.report.unlabeled_ratio == 0.9


def test_federated_simulation_config_keeps_fl_semantic_axes_separate() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert cfg.ssl_method.client_step.uses_local_update_profile is True
    assert cfg.ssl_method.client_step.custom_method_runtime_required is False
    assert cfg.ssl_method.server_step.custom_round_policy_required is False
    assert cfg.ssl_method.round_state_exchange.custom_exchange_required is False
    assert "training_algorithm_profile" not in cfg
    assert "adapter_family_name" not in cfg.local_update_profile
    assert "aggregation_backend_name" not in cfg.local_update_profile
    assert (
        cfg.training_task.objective.algorithm_profile_name
        == cfg.local_update_profile.algorithm_profile_name
    )
    assert (
        cfg.round_runtime.adapter_family_name
        == cfg.round_runtime_profile.adapter_family_name
    )
    assert (
        cfg.round_runtime.aggregation_backend_name
        == cfg.round_runtime_profile.aggregation_backend_name
    )
    assert cfg.report.labeled_ratio == cfg.client_pool_split.labeled_ratio
    assert cfg.report.unlabeled_ratio == cfg.client_pool_split.unlabeled_ratio
    assert len(cfg.seed_sweep.seeds) == cfg.report.seed_count
    assert cfg.report.seed_count == 3


@pytest.mark.parametrize(
    "profile_name",
    [
        "prototype_pseudo_label_v1",
        "prototype_top1_confidence_v1",
        "lora_pseudo_label_v1",
    ],
)
def test_federated_simulation_local_update_profile_is_hydra_source_of_truth(
    profile_name: str,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                f"strategy_axes/fl/local_update_profile={profile_name}",
            ],
        )

    local_update_profile = LocalUpdateProfile.from_mapping(
        _plain_dict(cfg.local_update_profile)
    )
    objective_config = TrainingObjectiveConfig.from_mapping(
        _plain_dict(cfg.training_task.objective)
    )

    assert local_update_profile.algorithm_profile_name == profile_name
    require_training_objective_matches_local_update_profile(
        objective_config=objective_config,
        local_update_profile=local_update_profile,
    )


def test_federated_simulation_supports_lora_classifier_profiles() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/local_update_profile=lora_pseudo_label_v1",
                "strategy_axes/fl/round_runtime_profile=fedavg_lora_classifier",
            ],
        )

    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert cfg.local_update_profile.algorithm_profile_name == "lora_pseudo_label_v1"
    assert cfg.round_runtime_profile.name == "fedavg_lora_classifier"
    assert cfg.round_runtime.adapter_family_name == "lora_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.training_task.objective.training_backend_name == (
        "lora_classifier_trainer"
    )
    assert cfg.training_task.objective.privacy_guard_name == "noop"
    assert cfg.training_task.objective["lora_classifier.backbone_model_id"] == (
        "mixedbread-ai/mxbai-embed-large-v1"
    )
    assert cfg.training_task.objective["lora_classifier.rank"] == 8
    assert cfg.training_task.objective["lora_classifier.alpha"] == 16
    assert cfg.training_task.objective["lora_classifier.delta_format"] == (
        "agent_local_artifact_ref"
    )
    assert "adapter_family_name" not in cfg.local_update_profile
    assert "aggregation_backend_name" not in cfg.local_update_profile


def test_federated_simulation_supports_high_level_fl_profile() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/experiment_profile=fedavg_pseudo_label_lora_classifier_v1",
            ],
        )

    assert cfg.fl_profile.name == "fedavg_pseudo_label_lora_classifier_v1"
    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert cfg.local_update_profile.algorithm_profile_name == "lora_pseudo_label_v1"
    assert cfg.round_runtime_profile.name == "fedavg_lora_classifier"
    assert cfg.round_runtime.adapter_family_name == "lora_classifier"
    assert cfg.training_task.objective.training_backend_name == (
        "lora_classifier_trainer"
    )
    descriptor = resolve_federated_ssl_method_descriptor(str(cfg.ssl_method.name))
    local_update_profile = LocalUpdateProfile.from_mapping(
        _plain_dict(cfg.local_update_profile)
    )
    assert descriptor.recipe is not None
    assert descriptor.recipe.supports_profile_combination(
        local_update_profile_name=local_update_profile.algorithm_profile_name,
        round_runtime_profile_name=str(cfg.round_runtime_profile.name),
    )
    validate_federated_ssl_profile_compatibility(
        FederatedSslProfileCompatibilityContext(
            method_descriptor=descriptor,
            local_update_profile=local_update_profile,
            local_update_adapter_kind=str(cfg.round_runtime.adapter_family_name),
            round_adapter_family_name=str(cfg.round_runtime.adapter_family_name),
            round_aggregation_backend_name=str(
                cfg.round_runtime.aggregation_backend_name
            ),
            experiment_profile=FederatedSslExperimentProfile.from_mapping(
                _plain_dict(cfg.fl_profile)
            ),
            round_runtime_profile_name=str(cfg.round_runtime_profile.name),
        )
    )


def test_federated_simulation_ssl_method_config_matches_methods_spec() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    descriptor = resolve_federated_ssl_method_descriptor(str(cfg.ssl_method.name))
    local_update_profile = LocalUpdateProfile.from_mapping(
        _plain_dict(cfg.local_update_profile)
    )

    assert cfg.ssl_method.implementation_status == descriptor.implementation_status
    assert cfg.ssl_method.client_step.task_type == descriptor.local_step.step_name
    assert (
        cfg.ssl_method.client_step.custom_method_runtime_required
        is descriptor.runtime_capabilities.requires_custom_client_runtime
    )
    assert (
        cfg.ssl_method.server_step.custom_round_policy_required
        is descriptor.runtime_capabilities.requires_custom_server_runtime
    )
    assert cfg.ssl_method.round_state_exchange.exchange_name == (
        descriptor.round_state_exchange.exchange_name
    )
    assert list(
        cfg.ssl_method.round_state_exchange.required_client_metric_keys
    ) == list(descriptor.round_state_exchange.required_client_metric_keys)
    assert descriptor.recipe is not None
    assert descriptor.recipe.supports_local_update_profile(
        local_update_profile.algorithm_profile_name
    )
    assert descriptor.recipe.supports_runtime_pair(
        adapter_family_name=str(cfg.round_runtime_profile.adapter_family_name),
        aggregation_backend_name=str(
            cfg.round_runtime_profile.aggregation_backend_name
        ),
    )
    assert descriptor.recipe.supports_profile_combination(
        local_update_profile_name=local_update_profile.algorithm_profile_name,
        round_runtime_profile_name=str(cfg.round_runtime_profile.name),
    )


def test_federated_simulation_supports_short_preset_and_leaf_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "track_presets/fl_ssl/simulation_preset=standard",
                "federated_run_preset.rounds=3",
                "federated_run_preset.client_count=8",
                "strategy_axes/prototype/build_strategy=kmeans",
                "prototype_builder.candidate_ks=[2]",
            ],
        )

    assert cfg.federated_run_preset.name == "standard"
    assert cfg.federated_run_preset.rounds == 3
    assert cfg.federated_run_preset.client_count == 8
    assert cfg.federated_run_preset.output_dir == "runs/federated_simulation"
    assert cfg.prototype_builder.name == "kmeans"
    assert list(cfg.prototype_builder.candidate_ks) == [2]


def test_federated_simulation_standard_preset_fixes_main_comparison_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["track_presets/fl_ssl/simulation_preset=standard"],
        )

    assert cfg.federated_run_preset.client_count == 10
    assert cfg.federated_run_preset.rounds == 50
    assert cfg.training_task.local_epochs == 1
    assert cfg.training_task.max_steps == 50


def test_federated_simulation_supports_detail_strategy_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/local_update_profile=prototype_top1_confidence_v1",
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

    assert cfg.shard_policy.name == "label_dominant"
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


def test_federated_simulation_supports_dirichlet_shard_policy_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["strategy_axes/fl/shard_policy=dirichlet_alpha03"],
        )

    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.shard_policy.dominant_ratio is None


def test_federated_simulation_supports_ssl_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["strategy_axes/fl/method_descriptor=fedavg_pseudo_label"],
        )

    assert cfg.ssl_method.schema_version == "federated_ssl_method.v1"
    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert cfg.ssl_method.round_state_exchange.exchange_name == "none"
    assert list(cfg.ssl_method.report_tags) == [
        "baseline",
        "fedavg",
        "pseudo_label",
    ]


def test_train_lora_classifier_defaults_to_gpu_online_and_fixed_lora_scaffold() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_classifier"
        )

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.selection_set == "validation"


def test_train_lora_pseudo_label_classifier_defaults_to_gpu_online_and_fixed_lora_scaffold(  # noqa: E501
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_pseudo_label_classifier"
        )

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
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/central_ssl_control/train_lora_fixmatch")

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
    assert cfg.unlabeled_jsonl == cfg.query_source.unlabeled_jsonl


def test_train_lora_bootstrap_classifier_teacher_defaults_to_classifier_teacher_then_fixed_lora_student(  # noqa: E501
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_bootstrap_classifier_teacher"
        )

    assert cfg.dataset.name == "ourafla"
    assert cfg.embedding.backend == "transformers_mxbai"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.pseudo_label_algorithm.name == "margin_threshold_v1"
    assert cfg.pseudo_label_algorithm.confidence_threshold == 0.6
    assert cfg.pseudo_label_algorithm.margin_threshold == 0.02
    assert cfg.pseudo_label_algorithm.algorithm_name == ("top1_margin_threshold")
    assert cfg.bootstrap_split.enabled is False


def test_dataset_pipeline_defaults_to_ourafla_and_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/data_pipeline/run_dataset_pipeline")

    assert cfg.dataset.name == "ourafla"
    assert cfg.dataset.test_jsonl == cfg.dataset.test_labeled_jsonl
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is False
    assert cfg.prototype_builder.name == "single"
