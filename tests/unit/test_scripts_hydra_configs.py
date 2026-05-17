"""scripts Hydra config group tests."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.compatibility import (
    FederatedSslProfileCompatibilityContext,
    validate_federated_ssl_profile_compatibility,
)
from methods.federated_ssl.execution_plan import build_federated_ssl_execution_plan
from methods.federated_ssl.local_update_profile import (
    LocalUpdateProfile,
    require_training_objective_matches_local_update_profile,
)
from methods.federated_ssl.registry import (
    list_federated_ssl_method_descriptors,
    resolve_federated_ssl_method_descriptor,
)
from scripts.experiments.fl_ssl.run_federated_simulation import (
    _with_inferred_manual_axes,
)
from scripts.runtime_adapters.federated_agent.backend_resolver import (
    resolve_federated_training_backend_adapter_kind,
)
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def _plain_dict(source: DictConfig) -> dict[str, object]:
    raw = OmegaConf.to_container(source, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Expected DictConfig section to resolve to a dict.")
    return raw


def _assert_fl_ssl_method_config_matches_descriptor(cfg: DictConfig) -> None:
    descriptor = resolve_federated_ssl_method_descriptor(str(cfg.ssl_method.name))

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
    expected_exchange_name = (
        None
        if descriptor.round_state_exchange is None
        else descriptor.round_state_exchange.exchange_name
    )
    expected_metric_keys = (
        []
        if descriptor.round_state_exchange is None
        else list(descriptor.round_state_exchange.required_client_metric_keys)
    )
    assert cfg.ssl_method.round_state_exchange.exchange_name == (expected_exchange_name)
    assert (
        list(cfg.ssl_method.round_state_exchange.required_client_metric_keys)
        == expected_metric_keys
    )


def _assert_composed_fl_runtime_is_compatible(cfg: DictConfig) -> None:
    descriptor = resolve_federated_ssl_method_descriptor(str(cfg.ssl_method.name))
    local_update_profile = LocalUpdateProfile.from_mapping(
        _plain_dict(cfg.local_update_profile)
    )
    objective_config = TrainingObjectiveConfig.from_mapping(
        _plain_dict(cfg.training_task.objective)
    )

    require_training_objective_matches_local_update_profile(
        objective_config=objective_config,
        local_update_profile=local_update_profile,
    )
    assert descriptor.recipe is not None
    assert descriptor.recipe.supports_local_update_profile(
        local_update_profile.algorithm_profile_name
    )
    assert descriptor.recipe.supports_runtime_pair(
        adapter_family_name=str(cfg.round_runtime.adapter_family_name),
        aggregation_backend_name=str(cfg.round_runtime.aggregation_backend_name),
    )
    validate_federated_ssl_profile_compatibility(
        FederatedSslProfileCompatibilityContext(
            method_descriptor=descriptor,
            local_update_profile=local_update_profile,
            local_update_adapter_kind=resolve_federated_training_backend_adapter_kind(
                objective_config=objective_config
            ),
            round_adapter_family_name=str(cfg.round_runtime.adapter_family_name),
            round_aggregation_backend_name=str(
                cfg.round_runtime.aggregation_backend_name
            ),
        )
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "entrypoints/dataset_pipeline/run_dataset_pipeline",
        "entrypoints/dataset_pipeline/materialize_query_ssl_split",
        "entrypoints/dataset_pipeline/materialize_query_ssl_views",
        "entrypoints/prototype_pack/seed_prototypes",
        "entrypoints/prototype_pack/evaluate_prototype_pack",
        "entrypoints/prototype_analysis/prototype_strategy",
        "entrypoints/prototype_analysis/prototype_threshold_sweep",
        "entrypoints/fl_ssl/materialize_fl_client_split",
        "entrypoints/fl_ssl/run_federated_simulation",
        "entrypoints/central_classifier_seed/train_softmax_classifier",
        "entrypoints/central_ssl_control/train_lora_supervised_classifier",
        "entrypoints/central_ssl_control/train_lora_ssl_classifier",
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


def test_query_ssl_view_materialization_entrypoint_uses_szegeelim_nllb_preset() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/dataset_pipeline/materialize_query_ssl_views"
        )

    assert (
        cfg.query_view_materialization.name
        == "szegeelim_general4_ssl_labeled1024_per_class_seed42_nllb_v1"
    )
    assert cfg.query_view_materialization.split_name == (
        "labeled1024_per_class_seed42_v1"
    )
    assert cfg.query_view_materialization.split_dir.endswith(
        "data/datasets/szegeelim_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1"
    )
    assert cfg.query_view_materialization.output_root.endswith(
        "data/datasets/szegeelim_mental_health/views"
    )
    assert cfg.query_view_materialization.batch_size == 64
    assert cfg.query_view_materialization.chunk_size == 128
    assert cfg.query_view_materialization.device == "cuda"
    assert cfg.query_view_materialization.local_files_only is False
    assert cfg.query_view_materialization.pivot_languages == ["deu_Latn", "fra_Latn"]


def test_query_ssl_split_materialization_entrypoint_uses_dataset_scoped_root() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/dataset_pipeline/materialize_query_ssl_split",
            overrides=["execution_context/dataset_asset=mental_health_kaggle"],
        )

    assert cfg.query_ssl_split_materialization.name == (
        "labeled1024_per_class_seed42_v1"
    )
    assert cfg.query_ssl_split_materialization.source_train_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/splits/train_split.v1.train.jsonl"
    )
    assert cfg.query_ssl_split_materialization.source_validation_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/splits/train_split.v1.validation.jsonl"
    )
    assert cfg.query_ssl_split_materialization.source_test_jsonl.endswith(
        "data/datasets/ourafla_mental_health/mapped/"
        "ourafla_mental_health_text_classification_test.v1.jsonl"
    )
    assert cfg.query_ssl_split_materialization.output_root.endswith(
        "data/datasets/szegeelim_mental_health/query_ssl"
    )
    assert cfg.query_ssl_split_materialization.labeled_count_per_class == 1024
    assert cfg.query_ssl_split_materialization.seed == 42


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


def test_train_lora_supervised_classifier_supports_auto_local_runtime_override() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_supervised_classifier",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_supervised_classifier_supports_source_and_budget_overrides() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_supervised_classifier",
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "run_controls/central_ssl/budget=smoke",
            ],
        )

    assert cfg.query_data_selection.labeled == "szegeelim_general4"
    assert cfg.train_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
    )
    assert cfg.eval_sets.validation == cfg.query_source.validation_jsonl
    assert cfg.eval_sets.test == cfg.query_source.test_jsonl
    assert cfg.central_ssl_budget.name == "smoke"
    assert cfg.train_batch_size == 8
    assert cfg.eval_batch_size == 32
    assert cfg.epochs == 1
    assert cfg.max_train_steps == 100
    assert cfg.learning_rate == 0.0002
    assert cfg.classifier_learning_rate == 0.0002
    assert cfg.weight_decay == 0.01
    assert cfg.log_every_steps == 100
    assert list(cfg.fixed_categories) == list(cfg.dataset.prototype_expected_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == "none"
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


def test_train_lora_ssl_classifier_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_lora_ssl_classifier_supports_source_budget_and_leaf_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "query_data_selection.unlabeled=ourafla_reddit",
                "run_controls/central_ssl/budget=smoke",
                "log_every_steps=20",
            ],
        )

    assert cfg.query_data_selection.labeled == "szegeelim_general4"
    assert cfg.query_data_selection.unlabeled == "ourafla_reddit"
    assert cfg.train_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
    )
    assert cfg.unlabeled_jsonl.endswith(
        "data/datasets/ourafla_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/unlabeled_pool.with_views.jsonl"
    )
    assert cfg.eval_sets.validation == cfg.query_source.validation_jsonl
    assert cfg.eval_sets.test == cfg.query_source.test_jsonl
    assert cfg.central_ssl_budget.name == "smoke"
    assert cfg.train_batch_size == 8
    assert cfg.eval_batch_size == 32
    assert cfg.epochs == 1
    assert cfg.max_train_steps == 100
    assert cfg.learning_rate == 0.0002
    assert cfg.classifier_learning_rate == 0.0002
    assert cfg.weight_decay == 0.01
    assert cfg.log_every_steps == 20
    assert list(cfg.fixed_categories) == list(cfg.dataset.prototype_expected_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == "none"
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


def test_train_lora_ssl_classifier_supports_query_ssl_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.hard_label is True


def test_train_lora_ssl_classifier_supports_pseudolabel_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/consistency_method=pseudolabel_usb_v1",
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.unsup_warm_up=0.2",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "pseudolabel_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "pseudolabel"
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.unsup_warm_up == 0.2
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is False


def test_train_lora_ssl_classifier_supports_flexmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/consistency_method=flexmatch_usb_v1",
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.thresh_warmup=false",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "flexmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "flexmatch"
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.thresh_warmup is False
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_train_lora_ssl_classifier_supports_freematch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/consistency_method=freematch_usb_v1",
                "query_ssl_method.ema_p=0.9",
                "query_ssl_method.ent_loss_ratio=0.02",
                "query_ssl_method.use_quantile=true",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "freematch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "freematch"
    assert cfg.query_ssl_method.ema_p == 0.9
    assert cfg.query_ssl_method.ent_loss_ratio == 0.02
    assert cfg.query_ssl_method.use_quantile is True
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_train_lora_ssl_classifier_supports_adamatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/consistency_method=adamatch_usb_v1",
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.ema_p=0.9",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "adamatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "adamatch"
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.ema_p == 0.9
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_train_lora_ssl_classifier_uses_precomputed_query_views() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/augmentation_source=precomputed_usb_candidates_v1",
                "query_ssl_strong_view_policy=first_aug",
            ],
        )

    assert cfg.query_ssl_augmenter.name == "precomputed_usb_candidates_v1"
    assert cfg.query_ssl_augmenter.augmenter_type == "precomputed_usb_candidates"
    assert cfg.query_ssl_augmenter.cache_dir == "data/cache/query_ssl_augmentations"
    assert cfg.query_ssl_strong_view_policy == "first_aug"


def test_train_lora_ssl_classifier_supports_pseudo_label_replay_mode() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "ssl_input_mode=pseudo_label_replay",
                "pseudo_label_jsonl=data/artifacts/lora_pseudo_label/run/pseudo_label_train.jsonl",
                "include_seed_train_rows=true",
            ],
        )

    assert cfg.ssl_input_mode == "pseudo_label_replay"
    assert cfg.pseudo_label_jsonl.endswith("pseudo_label_train.jsonl")
    assert cfg.include_seed_train_rows is True
    assert cfg.pseudo_label_export_root == "data/artifacts/lora_pseudo_label"
    assert list(cfg.fixed_categories) == list(cfg.dataset.prototype_expected_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == "none"
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


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

    assert cfg.federated_run_budget.name == "smoke"
    assert cfg.local_update_profile.algorithm_profile_name == "lora_pseudo_label_v1"
    assert "fl_profile" not in cfg
    assert "round_runtime_profile" not in cfg
    assert cfg.round_runtime.adapter_family_name == "lora_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.round_runtime.classifier_head_bootstrap_logit_scale == 8.0
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.lora.name == "default"
    assert cfg.training_task.objective.algorithm_profile_name == (
        "lora_pseudo_label_v1"
    )
    assert cfg.training_task.objective.training_backend_name == (
        "lora_classifier_trainer"
    )
    assert cfg.training_task.objective.privacy_guard_name == "noop"
    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "fixmatch"
    assert cfg.query_ssl_method.unlabeled_batch_size == cfg.training_task.batch_size
    assert cfg.query_ssl_strong_view_policy == "first_aug"
    assert cfg.training_task.objective["query_ssl.method_name"] == "fixmatch_usb_v1"
    assert cfg.training_task.objective["query_ssl.algorithm_name"] == "fixmatch"
    assert cfg.training_task.objective["query_ssl.strong_view_policy"] == "first_aug"
    assert cfg.validation.confidence_threshold == 0.6
    assert cfg.validation.margin_threshold == 0.02
    assert cfg.federated_run_budget.output_dir == "runs/federated_simulation_smoke"
    assert cfg.federated_run_budget.client_count == 4
    assert cfg.federated_run_budget.rounds == 3
    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.local_files_only is True
    assert cfg.fl_data.source_mode == "runtime_split_from_train"
    assert cfg.fl_data.split_manifest is None
    assert cfg.seed_sweep.output_dir == "runs/federated_simulation_seed_sweep"
    assert list(cfg.seed_sweep.seeds) == [42, 43, 44]
    assert cfg.client_count_sweep.output_dir == (
        "runs/federated_simulation_client_count_sweep"
    )
    assert list(cfg.client_count_sweep.client_counts) == list(range(1, 11))
    assert cfg.shard_policy.name == "label_dominant"
    assert cfg.shard_policy.dominant_ratio == 0.75
    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert cfg.ssl_method.method_role == "baseline"
    assert cfg.ssl_method.implementation_status == "active_runtime"
    assert cfg.ssl_method.client_step.owner == "agent"
    assert cfg.ssl_method.server_step.aggregation_backend_name == "fedavg"
    assert cfg.ssl_method.round_state_exchange.exchange_name == "none"
    assert cfg.fl_method.composition_mode == "manual"
    assert "manual_axes" not in cfg.fl_method
    assert cfg.security_policy.name == "plaintext"
    assert cfg.security_policy.update_payload_visibility == "per_client_plaintext"
    assert cfg.security_policy.client_metric_visibility == "per_client_plaintext"
    assert cfg.report.track == "fl_ssl_main_comparison"
    assert cfg.report.table_role == "main_comparison"
    assert cfg.client_pool_split.labeled_ratio == 0.1
    assert cfg.client_pool_split.unlabeled_ratio == 0.9
    assert cfg.report.labeled_ratio == 0.1
    assert cfg.report.unlabeled_ratio == 0.9


def test_fl_client_split_materialization_uses_query_data_source_and_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/materialize_fl_client_split",
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "query_data_selection.unlabeled=ourafla_reddit",
                "run_controls/fl_ssl/budget=main",
                "federated_run_budget.client_count=8",
                "strategy_axes/fl/shard_policy=dirichlet_alpha03",
            ],
        )

    assert cfg.fl_client_split_materialization.source_labeled_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
    )
    assert cfg.fl_client_split_materialization.source_unlabeled_jsonl.endswith(
        "data/datasets/ourafla_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/unlabeled_pool.with_views.jsonl"
    )
    assert cfg.fl_client_split_materialization.source_validation_jsonl.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/validation.jsonl"
    )
    assert cfg.fl_client_split_materialization.client_count == 8
    assert cfg.fl_client_split_materialization.bootstrap_ratio == (
        cfg.federated_run_budget.bootstrap_ratio
    )
    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.fl_client_split_materialization.labeled_policy.mode == "all"
    assert cfg.fl_client_split_materialization.labeled_policy.count_per_class is None
    assert cfg.fl_client_split_materialization.labeled_policy.fraction is None
    assert cfg.fl_client_split_materialization.view_schema.weak_text_field == "text"
    assert list(cfg.fl_client_split_materialization.view_schema.strong_text_fields) == [
        "aug_0",
        "aug_1",
    ]


def test_fl_client_split_materialization_supports_labeled_policy_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/materialize_fl_client_split",
            overrides=[
                "fl_client_split_materialization.labeled_policy.mode=count_per_class",
                "fl_client_split_materialization.labeled_policy.count_per_class=256",
            ],
        )

    assert cfg.fl_client_split_materialization.labeled_policy.mode == "count_per_class"
    assert cfg.fl_client_split_materialization.labeled_policy.count_per_class == 256


def test_federated_simulation_config_keeps_fl_semantic_axes_separate() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert cfg.ssl_method.client_step.uses_local_update_profile is True
    assert cfg.ssl_method.client_step.custom_method_runtime_required is False
    assert cfg.ssl_method.server_step.custom_round_policy_required is False
    assert cfg.ssl_method.round_state_exchange.custom_exchange_required is False
    assert cfg.fl_method.composition_mode == "manual"
    assert "training_algorithm_profile" not in cfg
    assert "adapter_family_name" not in cfg.local_update_profile
    assert "aggregation_backend_name" not in cfg.local_update_profile
    assert "round_runtime_profile" not in cfg
    assert "fl_profile" not in cfg
    assert (
        cfg.training_task.objective.algorithm_profile_name
        == cfg.local_update_profile.algorithm_profile_name
    )
    assert cfg.round_runtime.adapter_family_name == "lora_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.report.labeled_ratio == cfg.client_pool_split.labeled_ratio
    assert cfg.report.unlabeled_ratio == cfg.client_pool_split.unlabeled_ratio
    assert len(cfg.seed_sweep.seeds) == cfg.report.seed_count
    assert cfg.report.seed_count == 3


@pytest.mark.parametrize(
    "descriptor",
    list_federated_ssl_method_descriptors(),
    ids=lambda descriptor: descriptor.name,
)
def test_federated_simulation_method_recipe_axes_are_composable(
    descriptor: FederatedSslMethodDescriptor,
) -> None:
    assert descriptor.recipe is not None

    for local_profile_name in descriptor.recipe.supported_local_update_profile_names:
        with initialize_config_module(version_base=None, config_module="conf"):
            local_cfg = compose(
                config_name="entrypoints/fl_ssl/run_federated_simulation",
                overrides=[
                    f"strategy_axes/fl/method_descriptor={descriptor.name}",
                    f"strategy_axes/fl/local_update_profile={local_profile_name}",
                ],
            )
        local_adapter_kind = resolve_federated_training_backend_adapter_kind(
            objective_config=TrainingObjectiveConfig.from_mapping(
                _plain_dict(local_cfg.training_task.objective)
            )
        )
        runtime_pair = next(
            (
                pair
                for pair in descriptor.recipe.supported_runtime_pairs
                if pair.adapter_family_name == local_adapter_kind
            ),
            None,
        )
        assert runtime_pair is not None

        with initialize_config_module(version_base=None, config_module="conf"):
            cfg = compose(
                config_name="entrypoints/fl_ssl/run_federated_simulation",
                overrides=[
                    f"strategy_axes/fl/method_descriptor={descriptor.name}",
                    (f"strategy_axes/fl/local_update_profile={local_profile_name}"),
                    (
                        "round_runtime.adapter_family_name="
                        f"{runtime_pair.adapter_family_name}"
                    ),
                    "round_runtime.aggregation_backend_name="
                    f"{runtime_pair.aggregation_backend_name}",
                ],
            )

        _assert_fl_ssl_method_config_matches_descriptor(cfg)
        _assert_composed_fl_runtime_is_compatible(cfg)


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


def test_federated_simulation_supports_diagonal_scale_profiles() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/local_update_profile=prototype_pseudo_label_v1",
                "round_runtime.adapter_family_name=diagonal_scale",
                "round_runtime.aggregation_backend_name=fedavg",
            ],
        )

    assert cfg.ssl_method.name == "fedavg_pseudo_label"
    assert (
        cfg.local_update_profile.algorithm_profile_name == "prototype_pseudo_label_v1"
    )
    assert cfg.round_runtime.adapter_family_name == "diagonal_scale"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.training_task.objective.training_backend_name == (
        "diagonal_scale_heuristic"
    )
    assert cfg.training_task.objective.privacy_guard_name == "diagonal_scale_clip_only"
    assert cfg.training_task.objective["lora_classifier.backbone_model_id"] == (
        "mixedbread-ai/mxbai-embed-large-v1"
    )
    assert cfg.training_task.objective["lora_classifier.rank"] == 8
    assert cfg.training_task.objective["lora_classifier.alpha"] == 16
    assert cfg.training_task.objective["lora_classifier.delta_format"] == (
        "inline_delta"
    )
    assert "adapter_family_name" not in cfg.local_update_profile
    assert "aggregation_backend_name" not in cfg.local_update_profile


def test_federated_simulation_ssl_method_config_matches_methods_spec() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    _assert_fl_ssl_method_config_matches_descriptor(cfg)
    _assert_composed_fl_runtime_is_compatible(cfg)


def test_federated_simulation_supports_short_preset_and_leaf_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "run_controls/fl_ssl/budget=main",
                "federated_run_budget.rounds=3",
                "federated_run_budget.client_count=8",
                "strategy_axes/prototype/build_strategy=kmeans",
                "prototype_builder.candidate_ks=[2]",
            ],
        )

    assert cfg.federated_run_budget.name == "main"
    assert cfg.federated_run_budget.rounds == 3
    assert cfg.federated_run_budget.client_count == 8
    assert cfg.federated_run_budget.output_dir == "runs/federated_simulation"
    assert cfg.prototype_builder.name == "kmeans"
    assert list(cfg.prototype_builder.candidate_ks) == [2]


def test_federated_simulation_main_budget_fixes_main_comparison_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["run_controls/fl_ssl/budget=main"],
        )

    assert cfg.federated_run_budget.client_count == 10
    assert cfg.federated_run_budget.rounds == 50
    assert cfg.training_task.local_epochs == 1
    assert cfg.training_task.max_steps == 50


def test_federated_simulation_supports_detail_strategy_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/local_update_profile=prototype_top1_confidence_v1",
                "round_runtime.adapter_family_name=diagonal_scale",
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


def test_federated_simulation_supports_manual_fl_method_plan() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "fl_method.composition_mode=manual",
            ],
        )

    assert cfg.fl_method.composition_mode == "manual"
    plan = build_federated_ssl_execution_plan(
        fl_method=_with_inferred_manual_axes(
            cfg=cfg,
            fl_method=_plain_dict(cfg.fl_method),
        ),
        security_policy=_plain_dict(cfg.security_policy),
        method_descriptor=resolve_federated_ssl_method_descriptor(
            str(cfg.ssl_method.name)
        ),
    )
    assert plan.method_name == "manual"
    assert plan.descriptor_name == "fedavg_pseudo_label"
    assert plan.manual_axes.client_ssl_objective == "fixmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "lora_classifier"


def test_federated_simulation_manual_plan_supports_direct_runtime_leaf_overrides() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "fl_method.composition_mode=manual",
                "strategy_axes/fl/local_update_profile=lora_pseudo_label_v1",
                "round_runtime.adapter_family_name=lora_classifier",
                "round_runtime.aggregation_backend_name=fedavg",
            ],
        )

    plan = build_federated_ssl_execution_plan(
        fl_method=_with_inferred_manual_axes(
            cfg=cfg,
            fl_method=_plain_dict(cfg.fl_method),
        ),
        security_policy=_plain_dict(cfg.security_policy),
        method_descriptor=resolve_federated_ssl_method_descriptor(
            str(cfg.ssl_method.name)
        ),
    )

    assert cfg.local_update_profile.algorithm_profile_name == "lora_pseudo_label_v1"
    assert cfg.round_runtime.adapter_family_name == "lora_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert plan.method_name == "manual"
    assert plan.manual_axes.client_ssl_objective == "fixmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "lora_classifier"


def test_federated_simulation_manual_plan_switches_ssl_algorithm_by_hydra_name() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/ssl/consistency_method=flexmatch_usb_v1",
                "training_task.local_epochs=2",
                "training_task.batch_size=8",
                "training_task.max_steps=7",
            ],
        )

    plan = build_federated_ssl_execution_plan(
        fl_method=_with_inferred_manual_axes(
            cfg=cfg,
            fl_method=_plain_dict(cfg.fl_method),
        ),
        security_policy=_plain_dict(cfg.security_policy),
        method_descriptor=resolve_federated_ssl_method_descriptor(
            str(cfg.ssl_method.name)
        ),
    )

    assert cfg.query_ssl_method.name == "flexmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "flexmatch"
    assert cfg.training_task.local_epochs == 2
    assert cfg.training_task.batch_size == 8
    assert cfg.training_task.max_steps == 7
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.training_task.objective["query_ssl.method_name"] == ("flexmatch_usb_v1")
    assert cfg.training_task.objective["query_ssl.algorithm_name"] == "flexmatch"
    assert plan.manual_axes.client_ssl_objective == "flexmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "lora_classifier"


def test_train_lora_supervised_classifier_defaults_to_gpu_online_scaffold() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_supervised_classifier"
        )

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.selection_set == "validation"
    assert cfg.output_dir == "runs/train_lora_supervised_classifier"


def test_train_lora_ssl_classifier_defaults_to_fixmatch_precomputed_views() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier"
        )

    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is True
    assert cfg.ssl_input_mode == "consistency"
    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "fixmatch"
    assert (
        cfg.query_source.name == "labeled_ourafla_reddit_unlabeled_ourafla_reddit_"
        "validation_ourafla_reddit_test_ourafla_reddit"
    )
    assert cfg.query_data_selection.labeled == "ourafla_reddit"
    assert cfg.query_data_selection.unlabeled == "ourafla_reddit"
    assert cfg.query_data_selection.validation == "ourafla_reddit"
    assert cfg.query_data_selection.test == "ourafla_reddit"
    assert cfg.train_jsonl.endswith(
        "data/datasets/ourafla_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
    )
    assert cfg.unlabeled_jsonl.endswith(
        "data/datasets/ourafla_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/unlabeled_pool.with_views.jsonl"
    )
    assert cfg.query_ssl_augmenter.name == "precomputed_usb_candidates_v1"
    assert cfg.query_ssl_strong_view_policy == "first_aug"
    assert cfg.train_batch_size == 12
    assert cfg.eval_batch_size == 32
    assert cfg.query_ssl_method.unlabeled_batch_size == 12
    assert cfg.epochs == 5
    assert cfg.max_train_steps == 3000
    assert cfg.query_adaptation_initial_checkpoint.name == "none"
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""
    assert cfg.output_dir == (
        "runs/train_lora_ssl_classifier/consistency/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit"
    )


def test_train_lora_ssl_classifier_switches_method_by_hydra_name() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/consistency_method=pseudolabel_usb_v1",
                "output_dir=runs/train_lora_ssl_classifier_pseudolabel",
            ],
        )

    assert cfg.query_ssl_method.name == "pseudolabel_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "pseudolabel"
    assert cfg.query_ssl_method.require_multiview is False
    assert (
        cfg.query_source.name == "labeled_ourafla_reddit_unlabeled_ourafla_reddit_"
        "validation_ourafla_reddit_test_ourafla_reddit"
    )
    assert cfg.output_dir == "runs/train_lora_ssl_classifier_pseudolabel"


def test_train_lora_ssl_classifier_supports_general_labeled_reddit_pool() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "query_data_selection.unlabeled=ourafla_reddit",
                "query_data_selection.validation=ourafla_reddit",
                "query_data_selection.test=ourafla_reddit",
            ],
        )

    assert (
        cfg.query_source.name == "labeled_szegeelim_general4_unlabeled_ourafla_reddit_"
        "validation_ourafla_reddit_test_ourafla_reddit"
    )
    assert cfg.train_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
    )
    assert cfg.unlabeled_jsonl.endswith(
        "data/datasets/ourafla_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/unlabeled_pool.with_views.jsonl"
    )
    assert cfg.eval_sets.validation.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/validation.jsonl"
    )
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test.jsonl"
    )
    assert cfg.output_dir.endswith(
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit"
    )


def test_train_lora_ssl_classifier_supports_general_pool_with_reddit_eval() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "query_data_selection.unlabeled=szegeelim_general4",
                "query_data_selection.validation=ourafla_reddit",
                "query_data_selection.test=ourafla_reddit",
            ],
        )

    assert (
        cfg.query_source.name
        == "labeled_szegeelim_general4_unlabeled_szegeelim_general4_"
        "validation_ourafla_reddit_test_ourafla_reddit"
    )
    assert cfg.train_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
    )
    assert cfg.unlabeled_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/unlabeled_pool.with_views.jsonl"
    )
    assert cfg.eval_sets.validation.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/validation.jsonl"
    )
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test.jsonl"
    )
    assert cfg.output_dir.endswith(
        "labeled-szegeelim_general4_unlabeled-szegeelim_general4_"
        "validation-ourafla_reddit_test-ourafla_reddit"
    )


def test_train_lora_ssl_classifier_can_select_validation_and_test_independently() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
            overrides=[
                "query_data_selection.validation=szegeelim_general4",
                "query_data_selection.test=ourafla_reddit",
            ],
        )

    assert cfg.eval_sets.validation.endswith(
        "data/datasets/szegeelim_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/validation.jsonl"
    )
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test.jsonl"
    )


def test_dataset_pipeline_defaults_to_ourafla_and_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/dataset_pipeline/run_dataset_pipeline")

    assert cfg.dataset.name == "ourafla"
    assert cfg.dataset.test_jsonl == cfg.dataset.test_labeled_jsonl
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is False
    assert cfg.prototype_builder.name == "single"
