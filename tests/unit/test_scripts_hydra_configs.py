"""scripts Hydra config group tests."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.execution_plan import build_federated_ssl_execution_plan
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_ORIGINAL_COMMIT,
    FEDMATCH_ORIGINAL_REPOSITORY,
    fedmatch_original_parameter_mapping,
)
from methods.federated_ssl.local_update_profile import (
    LocalUpdateProfile,
    require_training_objective_matches_local_update_profile,
)
from methods.federated_ssl.registry import (
    list_federated_ssl_method_descriptors,
)
from scripts.experiments.fl_ssl.federated_simulation.config_request import (
    _build_capability_plan,
    _build_execution_plan,
    _build_ssl_method_config,
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


def _assert_manual_fl_runtime_is_compatible(cfg: DictConfig) -> None:
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
    assert resolve_federated_training_backend_adapter_kind(
        objective_config=objective_config
    ) == str(cfg.round_runtime.adapter_family_name)


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
        "entrypoints/central_ssl_control/train_peft_supervised_classifier",
        "entrypoints/central_ssl_control/train_peft_ssl_classifier",
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


def test_train_peft_supervised_classifier_supports_auto_local_runtime_override() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_supervised_classifier",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_peft_supervised_classifier_supports_source_and_budget_overrides() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_supervised_classifier",
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
    assert cfg.central_ssl_budget.output_root == "runs/_smoke"
    assert cfg.output_dir == "runs/_smoke/train_peft_supervised_classifier"
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


def test_train_peft_ssl_classifier_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_train_peft_ssl_classifier_supports_source_budget_and_leaf_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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
    assert cfg.central_ssl_budget.output_root == "runs/_smoke"
    assert cfg.output_dir.startswith("runs/_smoke/train_peft_ssl_classifier/")
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


def test_train_peft_ssl_classifier_supports_query_ssl_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
            overrides=[
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.hard_label is True


def test_train_peft_ssl_classifier_supports_pseudolabel_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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


def test_train_peft_ssl_classifier_supports_flexmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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


def test_train_peft_ssl_classifier_supports_freematch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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


def test_train_peft_ssl_classifier_supports_adamatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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


def test_train_peft_ssl_classifier_uses_precomputed_query_views() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/augmentation_source=precomputed_usb_candidates_v1",
                "query_ssl_strong_view_policy=first_aug",
            ],
        )

    assert cfg.query_ssl_augmenter.name == "precomputed_usb_candidates_v1"
    assert cfg.query_ssl_augmenter.augmenter_type == "precomputed_usb_candidates"
    assert cfg.query_ssl_augmenter.cache_dir == "data/cache/query_ssl_augmentations"
    assert cfg.query_ssl_strong_view_policy == "first_aug"


def test_train_peft_ssl_classifier_supports_pseudo_label_replay_mode() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
            overrides=[
                "ssl_input_mode=pseudo_label_replay",
                "pseudo_label_jsonl=data/artifacts/query_peft_pseudo_label/run/pseudo_label_train.jsonl",
                "include_seed_train_rows=true",
            ],
        )

    assert cfg.ssl_input_mode == "pseudo_label_replay"
    assert cfg.pseudo_label_jsonl.endswith("pseudo_label_train.jsonl")
    assert cfg.include_seed_train_rows is True
    assert cfg.pseudo_label_export_root == "data/artifacts/query_peft_pseudo_label"
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
    assert cfg.local_update_profile.algorithm_profile_name == "peft_pseudo_label_v1"
    assert "fl_profile" not in cfg
    assert "round_runtime_profile" not in cfg
    assert cfg.round_runtime.update_family_name == "peft_text_classifier"
    assert cfg.round_runtime.runtime_payload_key == "peft_text_classifier"
    assert cfg.round_runtime.composition_slug_builder == (
        "methods.adaptation.peft_text_classifier.runtime_family."
        "build_peft_text_classifier_composition_slug"
    )
    assert cfg.round_runtime.round_runtime_payload_builder == (
        "scripts.runtime_adapters.federated_server.peft_encoder_round_runtime."
        "build_peft_encoder_round_runtime_payloads"
    )
    assert list(cfg.round_runtime.local_objective_executors) == [
        "scripts.runtime_adapters.federated_agent."
        "peft_encoder_method_owned_client_round."
        "run_peft_encoder_method_owned_client_round_if_supported",
        "scripts.runtime_adapters.federated_agent."
        "peft_encoder_query_ssl_client_round."
        "run_peft_encoder_query_ssl_client_round_if_supported",
    ]
    assert cfg.round_runtime.initial_state_builder == (
        "methods.adaptation.peft_text_classifier.runtime_family."
        "build_initial_peft_encoder_state"
    )
    assert cfg.round_runtime.validation_evaluator == (
        "methods.adaptation.peft_text_classifier.evaluation."
        "evaluate_peft_encoder_simulation_validation_payload"
    )
    assert cfg.round_runtime.final_projection_builder == (
        "scripts.runtime_adapters.federated_server.peft_encoder_final_projection."
        "build_peft_encoder_final_projection_artifacts"
    )
    assert cfg.round_runtime.transient_resource_cleaner == (
        "methods.adaptation.peft_text_classifier.resource_cache."
        "clear_peft_encoder_transient_resource_cache"
    )
    assert cfg.round_runtime.adapter_family_name == "peft_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.round_runtime.classifier_head_bootstrap_logit_scale == 8.0
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.lora.name == "default"
    assert cfg.training_task.objective.algorithm_profile_name == (
        "peft_pseudo_label_v1"
    )
    assert cfg.training_task.objective.training_backend_name == (
        "peft_classifier_trainer"
    )
    assert cfg.training_task.objective.example_generation_backend_name == (
        "peft_classifier_raw_rows"
    )
    assert cfg.training_task.objective.evidence_backend_name == (
        "peft_classifier_logits"
    )
    assert cfg.training_task.objective.scorer_backend_name == "peft_classifier_logits"
    assert cfg.training_task.objective.privacy_guard_name == "noop"
    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "fixmatch"
    assert cfg.training_task.batch_size == 12
    assert cfg.train_batch_size == 12
    assert cfg.query_ssl_method.unlabeled_batch_size == cfg.training_task.batch_size
    assert cfg.query_ssl_strong_view_policy == "first_aug"
    assert cfg.training_task.objective.query_ssl.method_name == "fixmatch_usb_v1"
    assert cfg.training_task.objective.query_ssl.algorithm_name == "fixmatch"
    assert cfg.training_task.objective.query_ssl.strong_view_policy == "first_aug"
    assert cfg.local_update_profile.validation_scorer_backend_name == (
        "peft_classifier_eval"
    )
    assert cfg.validation.scorer_backend_name == "peft_classifier_eval"
    assert cfg.validation.score_policy_name is None
    assert cfg.validation.score_top_k is None
    assert cfg.validation.confidence_threshold == 0.6
    assert cfg.validation.margin_threshold == 0.02
    assert cfg.federated_run_budget.output_dir == "runs/_smoke/fl_ssl"
    assert cfg.federated_run_budget.client_count == 4
    assert cfg.federated_run_budget.rounds == 3
    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.local_files_only is True
    assert cfg.fl_data.source_mode == "runtime_split_from_train"
    assert cfg.fl_data.split_manifest is None
    assert cfg.seed_sweep.output_dir == "runs/_smoke/fl_ssl"
    assert list(cfg.seed_sweep.seeds) == [42, 43, 44]
    assert cfg.client_count_sweep.output_dir == "runs/_smoke/fl_ssl"
    assert list(cfg.client_count_sweep.client_counts) == list(range(1, 11))
    assert cfg.client_count_sweep.split_manifest_by_client_count is None
    assert cfg.run_safety.max_total_rounds_without_ack == 30
    assert cfg.run_safety.allow_long_run is False
    assert cfg.run_safety.long_run_ack is None
    assert cfg.run_safety.required_long_run_ack == "ALLOW_FL_SSL_LONG_RUN"
    assert cfg.shard_policy.name == "label_dominant"
    assert cfg.shard_policy.dominant_ratio == 0.75
    assert "ssl_method" not in cfg
    assert cfg.fl_method.composition_mode == "manual"
    assert "manual_axes" not in cfg.fl_method
    assert cfg.security_policy.name == "plaintext"
    assert cfg.security_policy.update_payload_visibility == "per_client_plaintext"
    assert cfg.security_policy.client_metric_visibility == "per_client_plaintext"
    assert cfg.report.track == "fl_ssl_main_comparison"
    assert cfg.report.table_role == "main_comparison"
    assert cfg.client_pool_split.labeled_ratio == 0.1
    assert cfg.client_pool_split.unlabeled_ratio == 0.9
    assert cfg.artifact_persistence.persist_agent_local_updates is False
    assert cfg.diagnostic_view.enabled is True
    assert cfg.diagnostic_view.selection_policy == "deterministic_random"
    assert cfg.diagnostic_view.max_rows == 512
    assert cfg.diagnostic_view.seed_offset == 1309
    assert cfg.report.labeled_ratio == 0.1
    assert cfg.report.unlabeled_ratio == 0.9
    assert cfg.labeled_exposure_policy.name == "shared_client_seed"
    assert cfg.client_participation_policy.name == "all_clients"
    assert cfg.local_supervision_regime.name == "client_labeled_and_unlabeled"
    assert cfg.server_step_policy.name == "none"
    assert cfg.peer_context_policy.name == "none"
    assert cfg.update_partition_policy.name == "unified"
    assert cfg.aggregation_weight_policy.name == "example_count"
    assert cfg.query_multiview_source.name == "materialized_rows"


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
    assert cfg.fl_client_split_materialization.split_id.endswith(
        "_dirichlet_label_skew_dominantNone_alpha0.3_clients8_seed42"
    )
    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.fl_client_split_materialization.labeled_policy.mode == "all"
    assert cfg.fl_client_split_materialization.labeled_policy.count_per_class is None
    assert cfg.fl_client_split_materialization.labeled_policy.fraction is None
    assert cfg.labeled_exposure_policy.name == "shared_client_seed"
    assert cfg.labeled_exposure_policy.split_id_component == "shared_client_seed_"
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


def test_fl_client_split_materialization_supports_labeled_exposure_policy_axis() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/materialize_fl_client_split",
            overrides=[
                "strategy_axes/fl/labeled_exposure_policy=shared_client_seed",
            ],
        )

    assert cfg.labeled_exposure_policy.name == "shared_client_seed"
    assert cfg.labeled_exposure_policy.split_id_component == "shared_client_seed_"
    assert "_shared_client_seed_" in cfg.fl_client_split_materialization.split_id


def test_fl_client_split_materialization_shared_seed_main_split_id_matches_docs() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/materialize_fl_client_split",
            overrides=[
                "run_controls/fl_ssl/budget=main",
                "query_data_selection.labeled=ourafla_reddit",
                "query_data_selection.unlabeled=ourafla_reddit",
                "query_data_selection.validation=ourafla_reddit",
                "query_data_selection.test=ourafla_reddit",
                "strategy_axes/fl/shard_policy=dirichlet_alpha03",
                "strategy_axes/fl/labeled_exposure_policy=shared_client_seed",
            ],
        )

    assert cfg.federated_run_budget.client_count == 10
    assert cfg.fl_client_split_materialization.split_id == (
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit_"
        "shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_"
        "clients10_seed42"
    )


def test_federated_simulation_config_keeps_fl_semantic_axes_separate() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    assert cfg.fl_data.source_mode == "runtime_split_from_train"
    assert cfg.fl_data.split_manifest is None
    assert "ssl_method" not in cfg
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
    assert cfg.round_runtime.update_family_name == "peft_text_classifier"
    assert cfg.round_runtime.runtime_payload_key == "peft_text_classifier"
    assert cfg.round_runtime.composition_slug_builder == (
        "methods.adaptation.peft_text_classifier.runtime_family."
        "build_peft_text_classifier_composition_slug"
    )
    assert cfg.round_runtime.round_runtime_payload_builder == (
        "scripts.runtime_adapters.federated_server.peft_encoder_round_runtime."
        "build_peft_encoder_round_runtime_payloads"
    )
    assert list(cfg.round_runtime.local_objective_executors) == [
        "scripts.runtime_adapters.federated_agent."
        "peft_encoder_method_owned_client_round."
        "run_peft_encoder_method_owned_client_round_if_supported",
        "scripts.runtime_adapters.federated_agent."
        "peft_encoder_query_ssl_client_round."
        "run_peft_encoder_query_ssl_client_round_if_supported",
    ]
    assert cfg.round_runtime.initial_state_builder == (
        "methods.adaptation.peft_text_classifier.runtime_family."
        "build_initial_peft_encoder_state"
    )
    assert cfg.round_runtime.validation_evaluator == (
        "methods.adaptation.peft_text_classifier.evaluation."
        "evaluate_peft_encoder_simulation_validation_payload"
    )
    assert cfg.round_runtime.final_projection_builder == (
        "scripts.runtime_adapters.federated_server.peft_encoder_final_projection."
        "build_peft_encoder_final_projection_artifacts"
    )
    assert cfg.round_runtime.transient_resource_cleaner == (
        "methods.adaptation.peft_text_classifier.resource_cache."
        "clear_peft_encoder_transient_resource_cache"
    )
    assert cfg.round_runtime.adapter_family_name == "peft_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.report.labeled_ratio == cfg.client_pool_split.labeled_ratio
    assert cfg.report.unlabeled_ratio == cfg.client_pool_split.unlabeled_ratio
    assert len(cfg.seed_sweep.seeds) == cfg.report.seed_count
    assert cfg.report.seed_count == 3


def test_federated_simulation_materialized_split_axis_selects_manifest() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/materialized_split=shared_general_reddit_pc100_alpha03_clients10",
            ],
        )

    assert cfg.fl_data.source_mode == "materialized_client_split"
    assert cfg.query_data_selection.labeled == "szegeelim_general4"
    assert cfg.query_data_selection.unlabeled == "ourafla_reddit"
    assert cfg.query_data_selection.validation == "ourafla_reddit"
    assert cfg.query_data_selection.test == "ourafla_reddit"
    assert cfg.federated_run_budget.client_count == 10
    assert cfg.federated_run_budget.bootstrap_ratio == 0.2
    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.labeled_exposure_policy.name == "shared_client_seed"
    assert cfg.fl_data.split_manifest == (
        "data/datasets/fl_client_splits/shared_client_labeled/"
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit_labels_pc100_"
        "shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_"
        "clients10_seed42/manifest.json"
    )


@pytest.mark.parametrize(
    ("selector", "labeled", "budget"),
    [
        ("shared_reddit_reddit_pc25_alpha03_clients10", "ourafla_reddit", 25),
        ("shared_reddit_reddit_pc100_alpha03_clients10", "ourafla_reddit", 100),
        ("shared_reddit_reddit_pc400_alpha03_clients10", "ourafla_reddit", 400),
        ("shared_reddit_reddit_pc1024_alpha03_clients10", "ourafla_reddit", 1024),
        ("shared_general_reddit_pc25_alpha03_clients10", "szegeelim_general4", 25),
        ("shared_general_reddit_pc100_alpha03_clients10", "szegeelim_general4", 100),
        ("shared_general_reddit_pc400_alpha03_clients10", "szegeelim_general4", 400),
        (
            "shared_general_reddit_pc1024_alpha03_clients10",
            "szegeelim_general4",
            1024,
        ),
    ],
)
def test_federated_simulation_materialized_split_axis_covers_shared_budgets(
    selector: str,
    labeled: str,
    budget: int,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[f"strategy_axes/fl/materialized_split={selector}"],
        )

    assert cfg.fl_data.source_mode == "materialized_client_split"
    assert cfg.query_data_selection.labeled == labeled
    assert cfg.query_data_selection.unlabeled == "ourafla_reddit"
    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.labeled_exposure_policy.name == "shared_client_seed"
    assert f"labels_pc{budget}_" in cfg.fl_data.split_manifest
    assert cfg.fl_data.split_manifest.endswith("clients10_seed42/manifest.json")


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

        _assert_manual_fl_runtime_is_compatible(cfg)


def test_fedmatch_method_config_injects_original_parameter_snapshot() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/method_descriptor=fedmatch",
                "fl_method.composition_mode=method_owned",
                "strategy_axes/fl/update_partition_policy=partitioned",
                "strategy_axes/fl/aggregation_weight_policy=uniform",
            ],
        )

    expected = fedmatch_original_parameter_mapping()
    ssl_method_config = _build_ssl_method_config(
        cfg,
        execution_plan=_build_execution_plan(cfg),
    )
    assert ssl_method_config is not None

    assert (
        cfg.ssl_method.implementation_status == "partitioned_trainable_state_slice_v1"
    )
    assert cfg.ssl_method.local_budget_policy == "iteration_capped"
    assert cfg.training_task.max_steps == 20
    assert cfg.ssl_method.original_source.repository == FEDMATCH_ORIGINAL_REPOSITORY
    assert cfg.ssl_method.original_source.commit == FEDMATCH_ORIGINAL_COMMIT
    assert "original_parameters" not in cfg.ssl_method
    assert cfg.ssl_method.scenario == "labels-at-client"
    assert cfg.ssl_method.use_original_parameters is True
    assert dict(cfg.ssl_method.parameter_overrides) == {}
    assert ssl_method_config.original_parameters == expected
    assert ssl_method_config.effective_parameters == expected
    assert ssl_method_config.parameter_overrides == {}
    assert ssl_method_config.parameter_override_status == "original"
    assert cfg.ssl_method.trace_mapping.supervised_partition == "sigma"
    assert cfg.ssl_method.trace_mapping.unsupervised_partition == "psi"
    assert (
        cfg.ssl_method.trace_mapping.parameter_decomposition
        == "peft_classifier_sigma_psi"
    )
    assert cfg.ssl_method.trace_mapping.aggregation_weight_policy == "uniform"
    assert cfg.ssl_method.trace_mapping.update_partition_policy == "partitioned"
    assert cfg.ssl_method.trace_mapping.partition_scheme == "sigma_psi"
    assert list(cfg.ssl_method.server_step.labels_at_client.aggregated_partitions) == [
        "merged_peft_classifier_delta"
    ]


def test_federated_simulation_local_ssl_policy_defaults_to_query_ssl_algorithm() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/ssl/consistency_method=flexmatch_usb_v1",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_plain_dict(cfg.labeled_exposure_policy),
    )

    assert cfg.local_ssl_policy.name == cfg.query_ssl_method.algorithm_name
    assert capability_plan.local_ssl_policy_name == "flexmatch"
    assert capability_plan.server_update_policy_name == "fedavg_merged_delta"


def test_federated_simulation_can_express_fedmatch_server_with_fixmatch() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/method_descriptor=fedmatch",
                "fl_method.composition_mode=method_owned",
                "strategy_axes/fl/server_update_policy=fedmatch_partitioned",
                "strategy_axes/fl/update_partition_policy=partitioned",
                "strategy_axes/fl/aggregation_weight_policy=uniform",
                "strategy_axes/fl/peer_context_policy=fixed_probe_output_knn",
                "strategy_axes/ssl/consistency_method=fixmatch_usb_v1",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_plain_dict(cfg.labeled_exposure_policy),
    )

    assert cfg.local_ssl_policy.name == "fixmatch"
    assert cfg.server_update_policy.name == "fedmatch_partitioned"
    assert capability_plan.local_ssl_policy_name == "fixmatch"
    assert capability_plan.server_update_policy_name == "fedmatch_partitioned"
    assert capability_plan.update_partition_policy_name == "partitioned"
    assert capability_plan.peer_context_policy_name == "fixed_probe_output_knn"


def test_federated_simulation_server_step_policy_declares_executor() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/server_step_policy=supervised_seed_step",
            ],
        )

    assert cfg.server_step_policy.name == "supervised_seed_step"
    assert (
        cfg.server_step_policy.executor
        == "scripts.runtime_adapters.federated_server.peft_encoder_server_step."
        "run_peft_encoder_supervised_seed_step"
    )


def test_federated_simulation_can_express_fedmatch_physical_faithful_shape() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "run_controls/fl_ssl/budget=reduced",
                "strategy_axes/fl/method_descriptor=fedmatch",
                "fl_method.composition_mode=method_owned",
                "strategy_axes/fl/server_update_policy=fedmatch_partitioned",
                "strategy_axes/fl/update_partition_policy=partitioned",
                "strategy_axes/fl/aggregation_weight_policy=uniform",
                "strategy_axes/fl/peer_context_policy=fixed_probe_output_knn",
                "strategy_axes/fl/local_ssl_policy=fedmatch_agreement",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_plain_dict(cfg.labeled_exposure_policy),
    )

    assert cfg.federated_run_budget.name == "reduced"
    assert cfg.federated_run_budget.rounds == 5
    assert cfg.local_ssl_policy.name == "fedmatch_agreement"
    assert cfg.server_update_policy.name == "fedmatch_partitioned"
    assert capability_plan.local_ssl_policy_name == "fedmatch_agreement"
    assert capability_plan.server_update_policy_name == "fedmatch_partitioned"
    assert capability_plan.update_partition_policy_name == "partitioned"
    assert capability_plan.aggregation_weight_policy.name == "uniform"
    assert capability_plan.peer_context_policy_name == "fixed_probe_output_knn"


def test_federated_simulation_legacy_peer_context_override_uses_canonical_name() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/peer_context_policy=prediction_similarity_topk",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_plain_dict(cfg.labeled_exposure_policy),
    )

    assert cfg.peer_context_policy.name == "fixed_probe_output_knn"
    assert cfg.peer_context_policy.legacy_alias_name == "prediction_similarity_topk"
    assert capability_plan.peer_context_policy_name == "fixed_probe_output_knn"


def test_fedmatch_method_config_records_parameter_overrides_as_ablation() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/method_descriptor=fedmatch",
                "fl_method.composition_mode=method_owned",
                "strategy_axes/fl/update_partition_policy=partitioned",
                "strategy_axes/fl/aggregation_weight_policy=uniform",
                "+ssl_method.parameter_overrides.confidence_threshold=0.85",
                "+ssl_method.parameter_overrides.num_helpers=4",
            ],
        )

    ssl_method_config = _build_ssl_method_config(
        cfg,
        execution_plan=_build_execution_plan(cfg),
    )
    assert ssl_method_config is not None

    assert ssl_method_config.original_parameters["confidence_threshold"] == (
        pytest.approx(0.75)
    )
    assert ssl_method_config.original_parameters["num_helpers"] == 2
    assert ssl_method_config.parameter_overrides == {
        "confidence_threshold": 0.85,
        "num_helpers": 4,
    }
    assert ssl_method_config.effective_parameters["confidence_threshold"] == (
        pytest.approx(0.85)
    )
    assert ssl_method_config.effective_parameters["num_helpers"] == 4
    assert ssl_method_config.parameter_override_status == "ablation"


def test_fedmatch_local_budget_policy_can_select_original_method() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl/method_descriptor=fedmatch",
                "fl_method.composition_mode=method_owned",
                "strategy_axes/fl/update_partition_policy=partitioned",
                "strategy_axes/fl/aggregation_weight_policy=uniform",
                "ssl_method.local_budget_policy=original_method",
            ],
        )

    ssl_method_config = _build_ssl_method_config(
        cfg,
        execution_plan=_build_execution_plan(cfg),
    )
    assert ssl_method_config is not None

    assert ssl_method_config.local_budget_policy == "original_method"
    assert ssl_method_config.parameter_override_status == "original"


@pytest.mark.parametrize(
    "profile_name",
    [
        "peft_pseudo_label_v1",
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
    assert (
        cfg.validation.scorer_backend_name
        == local_update_profile.validation_scorer_backend_name
    )
    assert (
        cfg.validation.score_policy_name
        == local_update_profile.validation_score_policy_name
    )
    assert cfg.validation.score_top_k == local_update_profile.validation_score_top_k
    require_training_objective_matches_local_update_profile(
        objective_config=objective_config,
        local_update_profile=local_update_profile,
    )


def test_federated_simulation_manual_runtime_axes_are_compatible() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    _assert_manual_fl_runtime_is_compatible(cfg)


def test_federated_simulation_supports_short_preset_and_leaf_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "run_controls/fl_ssl/budget=main",
                "federated_run_budget.rounds=3",
                "federated_run_budget.client_count=8",
            ],
        )

    assert cfg.federated_run_budget.name == "main"
    assert cfg.federated_run_budget.rounds == 3
    assert cfg.federated_run_budget.client_count == 8
    assert cfg.federated_run_budget.output_dir == "runs/fl_ssl"


def test_federated_simulation_main_budget_fixes_main_comparison_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["run_controls/fl_ssl/budget=main"],
        )

    assert cfg.federated_run_budget.client_count == 10
    assert cfg.federated_run_budget.rounds == 30
    assert cfg.federated_run_budget.output_dir == "runs/fl_ssl"
    assert cfg.seed_sweep.output_dir == "runs/fl_ssl"
    assert cfg.client_count_sweep.output_dir == "runs/fl_ssl"
    assert cfg.training_task.local_epochs == 1
    assert cfg.training_task.batch_size == 12
    assert cfg.train_batch_size == 12
    assert cfg.training_task.max_steps == 20


def test_federated_simulation_reduced_budget_uses_5_rounds() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["run_controls/fl_ssl/budget=reduced"],
        )

    assert cfg.federated_run_budget.name == "reduced"
    assert cfg.federated_run_budget.client_count == 10
    assert cfg.federated_run_budget.rounds == 5
    assert cfg.federated_run_budget.output_dir == "runs/fl_ssl"
    assert cfg.seed_sweep.output_dir == "runs/fl_ssl"
    assert cfg.client_count_sweep.output_dir == "runs/fl_ssl"
    assert cfg.training_task.batch_size == 12
    assert cfg.train_batch_size == 12


def test_federated_simulation_shared_seed_flexmatch_reduced_command_shape() -> None:
    split_manifest = (
        "data/datasets/fl_client_splits/"
        "shared_client_labeled/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit_"
        "shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_"
        "clients10_seed42/manifest.json"
    )
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "run_controls/fl_ssl/budget=reduced",
                "fl_method.composition_mode=manual",
                "strategy_axes/fl/shard_policy=dirichlet_alpha03",
                "strategy_axes/ssl/consistency_method=flexmatch_usb_v1",
                "round_runtime.adapter_family_name=peft_classifier",
                "round_runtime.aggregation_backend_name=fedavg",
                "fl_data.source_mode=materialized_client_split",
                f"fl_data.split_manifest={split_manifest}",
                "training_task.batch_size=12",
                "training_task.max_steps=20",
            ],
        )

    assert cfg.federated_run_budget.name == "reduced"
    assert cfg.federated_run_budget.client_count == 10
    assert cfg.federated_run_budget.rounds == 5
    assert cfg.federated_run_budget.output_dir == "runs/fl_ssl"
    assert cfg.fl_method.composition_mode == "manual"
    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.query_ssl_method.name == "flexmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "flexmatch"
    assert cfg.round_runtime.adapter_family_name == "peft_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.fl_data.source_mode == "materialized_client_split"
    assert cfg.fl_data.split_manifest == split_manifest
    assert cfg.training_task.batch_size == 12
    assert cfg.train_batch_size == 12
    assert cfg.query_ssl_method.unlabeled_batch_size == 12
    assert cfg.training_task.max_steps == 20


def test_federated_simulation_supports_detail_strategy_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "shard_policy.dominant_ratio=0.6",
                "training_task.objective.confidence_threshold=0.7",
                "training_task.objective.margin_threshold=0.1",
                "diagnostics.dump_dir_name=custom_dumps",
            ],
        )

    assert cfg.shard_policy.name == "label_dominant"
    assert cfg.shard_policy.dominant_ratio == 0.6
    assert cfg.training_task.objective.algorithm_profile_name == (
        "peft_pseudo_label_v1"
    )
    assert cfg.training_task.objective.confidence_threshold == 0.7
    assert cfg.training_task.objective.margin_threshold == 0.1
    assert cfg.validation.scorer_backend_name == "peft_classifier_eval"
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


def test_federated_simulation_rejects_removed_manual_baseline_descriptor_override() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        with pytest.raises(Exception, match="method_descriptor"):
            compose(
                config_name="entrypoints/fl_ssl/run_federated_simulation",
                overrides=["strategy_axes/fl/method_descriptor=fedavg_pseudo_label"],
            )


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
        method_descriptor=None,
    )
    assert plan.method_name == "manual"
    assert plan.descriptor_name is None
    assert plan.execution_role == "manual_baseline"
    assert plan.manual_axes.client_ssl_objective == "fixmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "peft_text_classifier"


def test_federated_simulation_manual_plan_supports_direct_runtime_leaf_overrides() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "fl_method.composition_mode=manual",
                "round_runtime.adapter_family_name=peft_classifier",
                "round_runtime.aggregation_backend_name=fedavg",
            ],
        )

    plan = build_federated_ssl_execution_plan(
        fl_method=_with_inferred_manual_axes(
            cfg=cfg,
            fl_method=_plain_dict(cfg.fl_method),
        ),
        security_policy=_plain_dict(cfg.security_policy),
        method_descriptor=None,
    )

    assert cfg.local_update_profile.algorithm_profile_name == "peft_pseudo_label_v1"
    assert cfg.round_runtime.adapter_family_name == "peft_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert plan.method_name == "manual"
    assert plan.execution_role == "manual_baseline"
    assert plan.manual_axes.client_ssl_objective == "fixmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "peft_text_classifier"


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
        method_descriptor=None,
    )

    assert cfg.query_ssl_method.name == "flexmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "flexmatch"
    assert cfg.training_task.local_epochs == 2
    assert cfg.training_task.batch_size == 8
    assert cfg.training_task.max_steps == 7
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.training_task.objective.query_ssl.method_name == "flexmatch_usb_v1"
    assert cfg.training_task.objective.query_ssl.algorithm_name == "flexmatch"
    assert plan.manual_axes.client_ssl_objective == "flexmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "peft_text_classifier"


def test_train_peft_supervised_classifier_defaults_to_gpu_online_scaffold() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_supervised_classifier"
        )

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.selection_set == "validation"
    assert cfg.output_dir == "runs/train_peft_supervised_classifier"
    assert cfg.central_ssl_budget.output_root == "runs"


def test_train_peft_ssl_classifier_defaults_to_fixmatch_precomputed_views() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier"
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
        "runs/train_peft_ssl_classifier/consistency/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit"
    )
    assert cfg.central_ssl_budget.output_root == "runs"


def test_train_peft_ssl_classifier_switches_method_by_hydra_name() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
            overrides=[
                "strategy_axes/ssl/consistency_method=pseudolabel_usb_v1",
                "output_dir=runs/train_peft_ssl_classifier_pseudolabel",
            ],
        )

    assert cfg.query_ssl_method.name == "pseudolabel_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "pseudolabel"
    assert cfg.query_ssl_method.require_multiview is False
    assert (
        cfg.query_source.name == "labeled_ourafla_reddit_unlabeled_ourafla_reddit_"
        "validation_ourafla_reddit_test_ourafla_reddit"
    )
    assert cfg.output_dir == "runs/train_peft_ssl_classifier_pseudolabel"


def test_train_peft_ssl_classifier_supports_general_labeled_reddit_pool() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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


def test_train_peft_ssl_classifier_supports_general_pool_with_reddit_eval() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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


def test_train_peft_ssl_classifier_can_select_validation_and_test_independently() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
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
