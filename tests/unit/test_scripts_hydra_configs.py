"""scripts Hydra config group tests."""

from __future__ import annotations

from pathlib import Path

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
    _resolve_labeled_exposure_policy_mapping,
    _resolve_local_update_profile,
    _with_inferred_manual_axes,
    build_simulation_request_from_config,
)
from scripts.runtime_adapters.federated_agent.backend_resolver import (
    resolve_federated_training_backend_adapter_kind,
)
from scripts.support.configured_callable import load_configured_callable
from shared.src.contracts.training_contracts import TrainingObjectiveConfig
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
PC100_QUERY_LABEL_BUDGET = "labeled100_per_class_seed42_nllb_views_v1"
PC100_SZEGEELIM_LABEL_WITH_VIEWS = (
    "data/datasets/szegeelim_mental_health/views/"
    "labeled100_per_class_seed42_v1/"
    "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
)
PC100_OURAFLA_LABEL_WITH_VIEWS = (
    "data/datasets/ourafla_mental_health/views/"
    "labeled100_per_class_seed42_v1/"
    "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
)


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
    ) == str(cfg.round_runtime.payload_adapter_kind)


def _central_supervised_contract_snapshot(cfg: DictConfig) -> dict[str, object]:
    return {
        "output_dir": str(cfg.output_dir),
        "runtime": {
            "name": str(cfg.runtime.name),
            "device": str(cfg.runtime.device),
            "local_files_only": bool(cfg.runtime.local_files_only),
        },
        "train_jsonl": str(cfg.train_jsonl),
        "selection_set": str(cfg.selection_set),
        "budget": {
            "name": str(cfg.central_ssl_budget.name),
            "train_batch_size": int(cfg.train_batch_size),
            "eval_batch_size": int(cfg.eval_batch_size),
            "epochs": int(cfg.epochs),
            "max_train_steps": int(cfg.max_train_steps),
        },
        "trainable_surface": {
            "name": str(cfg.trainable_surface.name),
            "trainable_state": str(cfg.trainable_surface.trainable_state),
        },
        "initial_checkpoint": str(cfg.query_adaptation_initial_checkpoint.name),
    }


def _central_query_ssl_contract_snapshot(cfg: DictConfig) -> dict[str, object]:
    return {
        "output_dir": str(cfg.output_dir),
        "runtime": {
            "name": str(cfg.runtime.name),
            "device": str(cfg.runtime.device),
            "local_files_only": bool(cfg.runtime.local_files_only),
        },
        "data": {
            "train_jsonl": str(cfg.train_jsonl),
            "unlabeled_jsonl": str(cfg.unlabeled_jsonl),
            "selection_set": str(cfg.selection_set),
        },
        "budget": {
            "name": str(cfg.central_ssl_budget.name),
            "train_batch_size": int(cfg.train_batch_size),
            "eval_batch_size": int(cfg.eval_batch_size),
            "epochs": int(cfg.epochs),
            "max_train_steps": int(cfg.max_train_steps),
            "drop_last_train_batches": bool(cfg.drop_last_train_batches),
            "drop_last_unlabeled_batches": bool(cfg.drop_last_unlabeled_batches),
        },
        "query_ssl_method": {
            "name": str(cfg.query_ssl_method.name),
            "algorithm_name": str(cfg.query_ssl_method.algorithm_name),
            "unlabeled_batch_size": int(cfg.query_ssl_method.unlabeled_batch_size),
            "p_cutoff": float(cfg.query_ssl_method.p_cutoff),
        },
        "initial_checkpoint": str(cfg.query_adaptation_initial_checkpoint.name),
    }


def _federated_ssl_request_contract_snapshot(
    cfg: DictConfig,
    *,
    output_dir: Path,
) -> dict[str, object]:
    request = build_simulation_request_from_config(cfg, output_dir=output_dir)
    query_ssl_config = request.query_ssl_objective_config
    if query_ssl_config is None:
        query_ssl_snapshot = None
    else:
        query_ssl_snapshot = {
            "method_name": query_ssl_config.method_name,
            "algorithm_name": query_ssl_config.algorithm_name,
            "strong_view_policy": query_ssl_config.strong_view_policy,
            "unlabeled_batch_size": query_ssl_config.unlabeled_batch_size,
        }
    shard_count = (
        None
        if request.materialized_dataset_split is None
        else len(request.materialized_dataset_split.client_shards)
    )

    return {
        "run": {
            "budget_name": request.run_budget_name,
            "run_output_dir": request.run_output_dir,
            "client_count": request.client_count,
            "rounds": request.rounds,
            "bootstrap_ratio": request.bootstrap_ratio,
            "seed": request.seed,
        },
        "rows": {
            "train": len(request.train_rows),
            "validation": len(request.validation_rows),
            "test": len(request.test_rows),
            "client_shards": shard_count,
        },
        "training_task": {
            "task_type": str(request.training_task_config.task_type),
            "local_epochs": request.training_task_config.local_epochs,
            "batch_size": request.training_task_config.batch_size,
            "learning_rate": request.training_task_config.learning_rate,
            "max_steps": request.training_task_config.max_steps,
            "min_required_examples": (
                request.training_task_config.min_required_examples
            ),
        },
        "round_runtime": {
            "payload_adapter_kind": (request.round_runtime_config.payload_adapter_kind),
            "aggregation_backend_name": (
                request.round_runtime_config.aggregation_backend_name
            ),
            "update_family_name": request.round_runtime_config.update_family_name,
            "runtime_payload_key": request.round_runtime_config.runtime_payload_key,
            "release_transient_model_cache_after_client": (
                request.round_runtime_config.release_transient_model_cache_after_client
            ),
            "transient_resource_cleaner": (
                request.round_runtime_config.transient_resource_cleaner
            ),
        },
        "capability_plan": (
            None
            if request.capability_plan is None
            else request.capability_plan.to_payload()
        ),
        "query_ssl_objective": query_ssl_snapshot,
        "data_source": {
            "source_mode": request.data_source_config.source_mode,
            "split_manifest_path": request.data_source_config.split_manifest_path,
        },
        "diagnostic_view": {
            "enabled": request.diagnostic_view_config.enabled,
            "selection_policy": request.diagnostic_view_config.selection_policy,
            "max_rows": request.diagnostic_view_config.max_rows,
        },
        "final_projection": {
            "enabled": request.final_projection_config.enabled,
            "dataset_names": list(request.final_projection_config.dataset_names),
        },
    }


def test_trainable_state_update_family_leafs_are_executable_surfaces() -> None:
    update_family_dir = (
        REPO_ROOT / "conf" / "strategy_axes" / "model_architecture" / "update_family"
    )
    leaf_paths = sorted(
        path for path in update_family_dir.glob("*.yaml") if path.name != "__init__.py"
    )

    assert leaf_paths
    for path in leaf_paths:
        cfg = OmegaConf.load(path)
        assert cfg.get("update_family_name"), path
        assert cfg.get("payload_adapter_kind"), path
        assert cfg.get("initial_state_builder"), path


def test_central_ssl_trainable_surface_leafs_declare_runtime_callables() -> None:
    trainable_surface_dir = (
        REPO_ROOT
        / "conf"
        / "strategy_axes"
        / "model_architecture"
        / "trainable_surface"
    )
    leaf_paths = sorted(
        path
        for path in trainable_surface_dir.glob("*.yaml")
        if path.name != "__init__.py"
    )

    central_ssl_leaf_count = 0
    for path in leaf_paths:
        cfg = OmegaConf.load(path)
        central_ssl = cfg.get("central_ssl")
        if central_ssl is None:
            continue
        central_ssl_leaf_count += 1
        assert cfg.get("name"), path
        assert central_ssl.get("trainer_version_prefix"), path
        for field_name in ("model_builder", "local_session_runner", "artifact_writer"):
            callable_path = central_ssl.get(field_name)
            assert callable_path, path
            load_configured_callable(
                str(callable_path),
                field_name=f"trainable_surface.central_ssl.{field_name}",
            )

    assert central_ssl_leaf_count >= 1


@pytest.mark.parametrize(
    "config_name",
    [
        "entrypoints/dataset_pipeline/run_dataset_pipeline",
        "entrypoints/dataset_pipeline/materialize_query_ssl_split",
        "entrypoints/dataset_pipeline/materialize_query_ssl_views",
        "entrypoints/fl_ssl/materialize_fl_client_split",
        "entrypoints/fl_ssl/run_federated_simulation",
        "entrypoints/central/ssl_control/run_peft_supervised_control",
        "entrypoints/central/ssl_control/run_full_text_encoder_supervised_control",
        "entrypoints/central/ssl_control/run_query_ssl_control",
        "entrypoints/central/fixed_feature_control/run_fixed_feature_baseline",
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


def test_run_peft_supervised_control_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_peft_supervised_control",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_run_peft_supervised_control_supports_source_and_budget_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_peft_supervised_control",
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
    assert "validation" not in cfg.eval_sets
    assert cfg.eval_sets.test == cfg.query_source.test_jsonl
    assert cfg.selection_set == "test"
    assert cfg.central_ssl_budget.name == "smoke"
    assert cfg.central_ssl_budget.output_root == "runs/_smoke"
    assert cfg.output_dir == "runs/_smoke/central/supervised/peft_classifier"
    assert cfg.train_batch_size == 8
    assert cfg.eval_batch_size == 32
    assert cfg.epochs == 1
    assert cfg.max_train_steps == 100
    assert cfg.learning_rate == 0.0002
    assert cfg.classifier_learning_rate == 0.0002
    assert cfg.weight_decay == 0.01
    assert cfg.log_every_steps == 100
    assert list(cfg.fixed_categories) == list(cfg.dataset.label_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == "none"
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""
    assert cfg.epoch_artifact_kind == "peft_adapter_classifier"
    assert cfg.epoch_artifact_every_epochs == 1


def test_run_full_text_encoder_supervised_control_supports_transfer_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name=(
                "entrypoints/central/ssl_control/"
                "run_full_text_encoder_supervised_control"
            ),
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "query_data_selection.validation=ourafla_reddit",
                "query_data_selection.test=ourafla_reddit",
                "run_controls/central_ssl/budget=smoke",
            ],
        )

    assert cfg.trainable_surface.name == "full_text_encoder"
    assert cfg.trainable_surface.trainable_state == "full_encoder_and_classifier_head"
    assert cfg.trainable_surface.requires_peft_adapter is False
    assert "peft_adapter" not in cfg
    assert cfg.query_data_selection.labeled == "szegeelim_general4"
    assert cfg.query_data_selection.validation == "ourafla_reddit"
    assert cfg.query_data_selection.test == "ourafla_reddit"
    assert cfg.train_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
    )
    assert "validation" not in cfg.eval_sets
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
    )
    assert cfg.selection_set == "test"
    assert cfg.output_dir == "runs/_smoke/central/supervised/full_text_encoder"
    assert "model_output_dir" not in cfg
    assert "classifier_output_dir" not in cfg
    assert cfg.learning_rate == 0.00002
    assert cfg.classifier_learning_rate == 0.0002


def test_central_supervised_controls_support_pc100_labeled_budget() -> None:
    overrides = [
        f"execution_context/query_labeled_budget={PC100_QUERY_LABEL_BUDGET}",
        "run_controls/central_ssl/budget=smoke",
        "central_ssl_budget.max_train_steps=2000",
    ]

    with initialize_config_module(version_base=None, config_module="conf"):
        peft_cfg = compose(
            config_name="entrypoints/central/ssl_control/run_peft_supervised_control",
            overrides=overrides,
        )
        full_cfg = compose(
            config_name=(
                "entrypoints/central/ssl_control/"
                "run_full_text_encoder_supervised_control"
            ),
            overrides=overrides,
        )

    for cfg in (peft_cfg, full_cfg):
        assert cfg.query_labeled_budget.name == PC100_QUERY_LABEL_BUDGET
        assert cfg.query_labeled_budget.count_per_class == 100
        assert cfg.query_data_selection.labeled == "szegeelim_general4"
        assert cfg.query_data_selection.unlabeled == "ourafla_reddit"
        assert cfg.query_data_selection.test == "ourafla_reddit"
        assert cfg.query_source.name == (
            "labeled_szegeelim_general4_unlabeled_ourafla_reddit_test_ourafla_reddit"
        )
        assert cfg.query_data_selection_slug == (
            "labeled-szegeelim_general4_labels-pc100_unlabeled-"
            "ourafla_reddit_test-ourafla_reddit"
        )
        assert cfg.train_jsonl == PC100_SZEGEELIM_LABEL_WITH_VIEWS
        assert cfg.eval_sets.test.endswith(
            "data/datasets/ourafla_mental_health/query_ssl/"
            "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
        )
        assert cfg.max_train_steps == 2000

    with initialize_config_module(version_base=None, config_module="conf"):
        reddit_cfg = compose(
            config_name="entrypoints/central/ssl_control/run_peft_supervised_control",
            overrides=[
                "query_data_selection.labeled=ourafla_reddit",
                *overrides,
            ],
        )

    assert reddit_cfg.query_data_selection.labeled == "ourafla_reddit"
    assert reddit_cfg.train_jsonl == PC100_OURAFLA_LABEL_WITH_VIEWS


def test_run_query_ssl_control_supports_auto_local_runtime_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=["execution_context/runtime_env=auto_local"],
        )

    assert cfg.runtime.name == "auto_local"
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.local_files_only is True


def test_run_fixed_feature_baseline_defaults_to_tfidf_logistic_regression() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name=(
                "entrypoints/central/fixed_feature_control/run_fixed_feature_baseline"
            )
        )

    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.device == "cuda"
    assert cfg.fixed_feature_space.name == "tfidf_word"
    assert cfg.fixed_feature_space.ngram_min == 1
    assert cfg.fixed_feature_space.ngram_max == 2
    assert cfg.fixed_feature_estimator.name == "logistic_regression"
    assert cfg.fixed_feature_estimator.class_weight == "balanced"
    assert cfg.query_labeled_budget.count_per_class == 1024
    assert cfg.selection_set == "test"
    assert cfg.output_dir == (
        "runs/central/supervised/fixed_feature/tfidf_word/"
        "logistic_regression/"
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_test-ourafla_reddit"
    )


@pytest.mark.parametrize(
    "estimator_name",
    ["logistic_regression", "multinomial_nb", "decision_tree", "linear_svc"],
)
def test_run_fixed_feature_baseline_supports_estimator_override(
    estimator_name: str,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name=(
                "entrypoints/central/fixed_feature_control/run_fixed_feature_baseline"
            ),
            overrides=[f"strategy_axes/classification/estimator={estimator_name}"],
        )

    assert cfg.fixed_feature_estimator.name == estimator_name
    assert cfg.output_dir == (
        "runs/central/supervised/fixed_feature/tfidf_word/"
        f"{estimator_name}/"
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_test-ourafla_reddit"
    )


def test_run_fixed_feature_baseline_supports_pc100_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name=(
                "entrypoints/central/fixed_feature_control/run_fixed_feature_baseline"
            ),
            overrides=[
                f"execution_context/query_labeled_budget={PC100_QUERY_LABEL_BUDGET}",
                "strategy_axes/classification/estimator=linear_svc",
            ],
        )

    assert cfg.query_labeled_budget.name == PC100_QUERY_LABEL_BUDGET
    assert cfg.query_labeled_budget.count_per_class == 100
    assert cfg.train_jsonl == PC100_SZEGEELIM_LABEL_WITH_VIEWS
    assert cfg.fixed_feature_estimator.name == "linear_svc"
    assert cfg.output_dir == (
        "runs/central/supervised/fixed_feature/tfidf_word/linear_svc/"
        "labeled-szegeelim_general4_labels-pc100_unlabeled-"
        "ourafla_reddit_test-ourafla_reddit"
    )


def test_run_query_ssl_control_supports_source_budget_and_leaf_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
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
    assert "validation" not in cfg.eval_sets
    assert cfg.eval_sets.test == cfg.query_source.test_jsonl
    assert cfg.selection_set == "test"
    assert cfg.central_ssl_budget.name == "smoke"
    assert cfg.central_ssl_budget.output_root == "runs/_smoke"
    assert cfg.output_dir.startswith("runs/_smoke/central/ssl/peft_text_encoder/")
    assert cfg.train_batch_size == 8
    assert cfg.eval_batch_size == 32
    assert cfg.epochs == 1
    assert cfg.max_train_steps == 100
    assert cfg.learning_rate == 0.0002
    assert cfg.classifier_learning_rate == 0.0002
    assert cfg.weight_decay == 0.01
    assert cfg.log_every_steps == 20
    assert list(cfg.fixed_categories) == list(cfg.dataset.label_categories)
    assert cfg.query_adaptation_initial_checkpoint.name == "none"
    assert cfg.initial_adapter_dir == ""
    assert cfg.initial_classifier_path == ""


def test_run_query_ssl_control_supports_query_ssl_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.hard_label is True


def test_run_query_ssl_control_supports_pseudolabel_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=pseudolabel_usb_v1",
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


def test_run_query_ssl_control_supports_flexmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=flexmatch_usb_v1",
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


def test_run_query_ssl_control_supports_refixmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=refixmatch_usb_v1",
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "refixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "refixmatch"
    assert cfg.query_ssl_method.temperature == 0.5
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.hard_label is True
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_supports_freematch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=freematch_usb_v1",
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


def test_run_query_ssl_control_supports_adamatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=adamatch_usb_v1",
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


def test_run_query_ssl_control_supports_dash_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=dash_usb_v1",
                "query_ssl_method.gamma=1.5",
                "query_ssl_method.C=1.1",
                "query_ssl_method.rho_min=0.1",
                "query_ssl_method.num_wu_iter=128",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "dash_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "dash"
    assert cfg.query_ssl_method.gamma == 1.5
    assert cfg.query_ssl_method.C == 1.1
    assert cfg.query_ssl_method.rho_min == 0.1
    assert cfg.query_ssl_method.num_wu_iter == 128
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_supports_uda_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=uda_usb_v1",
                "query_ssl_method.p_cutoff=0.9",
                "query_ssl_method.tsa_schedule=linear",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "uda_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "uda"
    assert cfg.query_ssl_method.T == 0.4
    assert cfg.query_ssl_method.p_cutoff == 0.9
    assert cfg.query_ssl_method.tsa_schedule == "linear"
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_supports_pimodel_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=pimodel_usb_v1",
                "query_ssl_method.unsup_warm_up=0.2",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "pimodel_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "pimodel"
    assert cfg.query_ssl_method.unsup_warm_up == 0.2
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_supports_meanteacher_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=meanteacher_usb_v1",
                "query_ssl_method.ema_m=0.9",
                "query_ssl_method.unsup_warm_up=0.2",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "meanteacher_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "meanteacher"
    assert cfg.query_ssl_method.ema_m == 0.9
    assert cfg.query_ssl_method.unsup_warm_up == 0.2
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_supports_comatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=comatch_usb_v1",
                "query_ssl_method.queue_batch=4",
                "query_ssl_method.proj_size=2",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "comatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "comatch"
    assert cfg.query_ssl_method.queue_batch == 4
    assert cfg.query_ssl_method.proj_size == 2
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True
    assert cfg.query_ssl_method.require_weak_strong_pair is True


def test_run_query_ssl_control_supports_simmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=simmatch_usb_v1",
                "query_ssl_method.proj_size=4",
                "query_ssl_method.da_len=8",
                "query_ssl_method.in_loss_ratio=0.5",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "simmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "simmatch"
    assert cfg.query_ssl_method.proj_size == 4
    assert cfg.query_ssl_method.da_len == 8
    assert cfg.query_ssl_method.in_loss_ratio == 0.5
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_supports_mixmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=mixmatch_usb_v1",
                "query_ssl_method.unsup_warm_up=0.2",
                "query_ssl_method.mixup_alpha=0.7",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "mixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "mixmatch"
    assert cfg.query_ssl_method.T == 0.5
    assert cfg.query_ssl_method.unsup_warm_up == 0.2
    assert cfg.query_ssl_method.mixup_alpha == 0.7
    assert cfg.query_ssl_method.mixup_manifold is True
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_supports_remixmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=remixmatch_usb_v1",
                "query_ssl_method.mixup_alpha=0.7",
                "query_ssl_method.kl_loss_ratio=0.25",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "remixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "remixmatch"
    assert cfg.query_ssl_method.T == 0.5
    assert cfg.query_ssl_method.mixup_alpha == 0.7
    assert cfg.query_ssl_method.kl_loss_ratio == 0.25
    assert cfg.query_ssl_method.rot_loss_ratio == 0.0
    assert cfg.query_ssl_method.mixup_manifold is True
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True
    assert cfg.query_ssl_method.require_weak_strong_pair is True


def test_run_query_ssl_control_supports_softmatch_method_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=softmatch_usb_v1",
                "query_ssl_method.ema_p=0.9",
                "query_ssl_method.n_sigma=3.0",
                "query_ssl_method.per_class=true",
                "query_ssl_method.unlabeled_batch_size=8",
            ],
        )

    assert cfg.query_ssl_method.name == "softmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "softmatch"
    assert cfg.query_ssl_method.ema_p == 0.9
    assert cfg.query_ssl_method.n_sigma == 3.0
    assert cfg.query_ssl_method.per_class is True
    assert cfg.query_ssl_method.unlabeled_batch_size == 8
    assert cfg.query_ssl_method.require_multiview is True


def test_run_query_ssl_control_uses_materialized_views_without_augmenter() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control"
        )

    assert "query_ssl_augmenter" not in cfg
    assert "query_ssl_strong_view_policy" not in cfg


def test_central_supervised_smoke_orchestration_contract_snapshot() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_peft_supervised_control",
            overrides=["run_controls/central_ssl/budget=smoke"],
        )

    assert _central_supervised_contract_snapshot(cfg) == {
        "output_dir": "runs/_smoke/central/supervised/peft_classifier",
        "runtime": {
            "name": "gpu_online",
            "device": "cuda",
            "local_files_only": False,
        },
        "train_jsonl": (
            "data/datasets/szegeelim_mental_health/views/"
            "labeled1024_per_class_seed42_v1/"
            "backtranslation_nllb_en_de_fr_usb_v1/labeled_train.with_views.jsonl"
        ),
        "selection_set": "test",
        "budget": {
            "name": "smoke",
            "train_batch_size": 8,
            "eval_batch_size": 32,
            "epochs": 1,
            "max_train_steps": 100,
        },
        "trainable_surface": {
            "name": "peft_text_encoder",
            "trainable_state": "peft_adapter_and_classifier_head",
        },
        "initial_checkpoint": "none",
    }


def test_central_query_ssl_smoke_orchestration_contract_snapshot() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=["run_controls/central_ssl/budget=smoke"],
        )

    assert _central_query_ssl_contract_snapshot(cfg) == {
        "output_dir": (
            "runs/_smoke/central/ssl/peft_text_encoder/"
            "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
            "test-ourafla_reddit"
        ),
        "runtime": {
            "name": "gpu_local",
            "device": "cuda",
            "local_files_only": True,
        },
        "data": {
            "train_jsonl": (
                "data/datasets/szegeelim_mental_health/views/"
                "labeled1024_per_class_seed42_v1/"
                "backtranslation_nllb_en_de_fr_usb_v1/"
                "labeled_train.with_views.jsonl"
            ),
            "unlabeled_jsonl": (
                "data/datasets/ourafla_mental_health/views/"
                "labeled1024_per_class_seed42_v1/"
                "backtranslation_nllb_en_de_fr_usb_v1/"
                "unlabeled_pool.with_views.jsonl"
            ),
            "selection_set": "test",
        },
        "budget": {
            "name": "smoke",
            "train_batch_size": 8,
            "eval_batch_size": 32,
            "epochs": 1,
            "max_train_steps": 100,
            "drop_last_train_batches": True,
            "drop_last_unlabeled_batches": True,
        },
        "query_ssl_method": {
            "name": "fixmatch_usb_v1",
            "algorithm_name": "fixmatch",
            "unlabeled_batch_size": 8,
            "p_cutoff": 0.95,
        },
        "initial_checkpoint": "none",
    }


def test_federated_ssl_smoke_orchestration_contract_snapshot(
    tmp_path: Path,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["run_controls/fl_ssl/budget=smoke"],
        )

    assert _federated_ssl_request_contract_snapshot(
        cfg,
        output_dir=tmp_path,
    ) == {
        "run": {
            "budget_name": "smoke",
            "run_output_dir": "runs/_smoke/fl_ssl",
            "client_count": 10,
            "rounds": 3,
            "bootstrap_ratio": 0.2,
            "seed": 42,
        },
        "rows": {
            "train": 82335,
            "validation": 4961,
            "test": 992,
            "client_shards": 10,
        },
        "training_task": {
            "task_type": "pseudo_label_self_training",
            "local_epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "max_steps": 50,
            "min_required_examples": 4,
        },
        "round_runtime": {
            "payload_adapter_kind": "peft_classifier",
            "aggregation_backend_name": "fedavg",
            "update_family_name": "peft_text_encoder",
            "runtime_payload_key": "peft_text_encoder",
            "release_transient_model_cache_after_client": True,
            "transient_resource_cleaner": (
                "methods.adaptation.peft_text_encoder.resource_cache."
                "clear_peft_encoder_transient_resource_cache"
            ),
        },
        "capability_plan": {
            "client_participation_policy": {
                "name": "all_clients",
                "fraction": None,
                "count": None,
                "min_clients": 1,
            },
            "labeled_exposure_policy": {"name": "shared_client_seed"},
            "local_supervision_regime": {"name": "client_labeled_and_unlabeled"},
            "server_step_policy": {"name": "none"},
            "server_update_policy": {"name": "fedavg_merged_delta"},
            "peer_context_policy": {"name": "none"},
            "update_partition_policy": {"name": "unified"},
            "local_ssl_policy": {"name": "fixmatch"},
            "aggregation_weight_policy": {"name": "uniform"},
            "query_multiview_source": {"name": "materialized_rows"},
        },
        "query_ssl_objective": {
            "method_name": "fixmatch_usb_v1",
            "algorithm_name": "fixmatch",
            "strong_view_policy": "first_aug",
            "unlabeled_batch_size": 8,
        },
        "data_source": {
            "source_mode": "materialized_client_split",
            "split_manifest_path": (
                "data/datasets/fl_client_splits/shared_client_labeled/"
                "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
                "validation-ourafla_reddit_test-ourafla_reddit_labels_pc1024_"
                "shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_"
                "clients10_seed42/manifest.json"
            ),
        },
        "diagnostic_view": {
            "enabled": True,
            "selection_policy": "deterministic_random",
            "max_rows": 512,
        },
        "final_projection": {
            "enabled": True,
            "dataset_names": ["validation", "test"],
        },
    }


def test_federated_simulation_uses_smoke_preset_by_default(tmp_path: Path) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    assert cfg.federated_run_budget.name == "smoke"
    assert cfg.local_update_profile.algorithm_profile_name == (
        "peft_classifier_update_v1"
    )
    assert "fl_profile" not in cfg
    assert "round_runtime_profile" not in cfg
    assert cfg.round_runtime.update_family_name == "peft_text_encoder"
    assert cfg.round_runtime.runtime_payload_key == "peft_text_encoder"
    assert cfg.round_runtime.release_transient_model_cache_after_client is True
    assert "peft_classifier" not in cfg.round_runtime
    assert "peft_text_encoder" in cfg.round_runtime.runtime_payloads
    assert cfg.round_runtime.composition_slug_builder == (
        "methods.adaptation.peft_text_encoder.update_family_runtime."
        "build_peft_text_encoder_composition_slug"
    )
    assert cfg.round_runtime.round_runtime_payload_builder == (
        "methods.adaptation.peft_text_encoder.simulation_runtime.round_runtime."
        "build_peft_encoder_round_runtime_payloads"
    )
    assert list(cfg.round_runtime.local_objective_executors) == [
        "scripts.runtime_adapters.federated_agent."
        "generic_client_runtime_bridge."
        "run_method_owned_client_round_if_supported",
        "scripts.runtime_adapters.federated_agent."
        "generic_client_runtime_bridge."
        "run_query_ssl_client_round_if_supported",
    ]
    assert cfg.round_runtime.client_round_runtime.base_state_materializer == (
        "scripts.runtime_adapters.federated_agent.base_state_materialization."
        "load_peft_encoder_base_parameters_with_timing"
    )
    assert cfg.round_runtime.client_round_runtime.query_ssl_training_runner == (
        "scripts.runtime_adapters.federated_agent.query_ssl_training_request."
        "run_query_ssl_peft_encoder_local_training"
    )
    assert (
        cfg.round_runtime.server_round_runtime.final_projection_artifacts_builder
        == (
            "methods.adaptation.peft_text_encoder.simulation_runtime."
            "final_projection.build_peft_encoder_final_projection_artifacts_from_state"
        )
    )
    assert cfg.round_runtime.initial_state_builder == (
        "methods.adaptation.peft_text_encoder.update_family_runtime."
        "build_initial_peft_encoder_state"
    )
    assert cfg.round_runtime.validation_evaluator == (
        "methods.adaptation.peft_text_encoder.evaluation."
        "evaluate_peft_encoder_simulation_validation_payload"
    )
    assert cfg.round_runtime.final_projection_builder == (
        "scripts.runtime_adapters.federated_server.generic_server_runtime_bridge."
        "build_final_projection_artifacts"
    )
    assert cfg.round_runtime.transient_resource_cleaner == (
        "methods.adaptation.peft_text_encoder.resource_cache."
        "clear_peft_encoder_transient_resource_cache"
    )
    assert cfg.round_runtime.payload_adapter_kind == "peft_classifier"
    assert "adapter_family_name" not in cfg.round_runtime
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.peft_adapter.name == "default"
    assert cfg.training_task.objective.algorithm_profile_name == (
        "peft_classifier_update_v1"
    )
    assert cfg.training_task.objective.training_backend_name == (
        "peft_classifier_trainer"
    )
    assert "example_generation_backend_name" not in cfg.training_task.objective
    assert "evidence_backend_name" not in cfg.training_task.objective
    assert "scorer_backend_name" not in cfg.training_task.objective
    assert "score_policy_name" not in cfg.training_task.objective
    assert cfg.training_task.objective.privacy_guard_name == "noop"
    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "fixmatch"
    assert cfg.training_task.batch_size == 8
    assert cfg.train_batch_size == 8
    assert cfg.train_jsonl == cfg.query_source.train_jsonl
    assert cfg.train_jsonl != cfg.dataset.train_jsonl
    assert cfg.query_ssl_method.unlabeled_batch_size == cfg.training_task.batch_size
    assert "query_ssl_strong_view_policy" not in cfg
    assert "query_ssl" not in cfg.training_task.objective
    request = build_simulation_request_from_config(cfg, output_dir=tmp_path)
    objective = request.training_task_config.objective_config.to_mapping()
    assert objective["query_ssl.method_name"] == "fixmatch_usb_v1"
    assert objective["query_ssl.algorithm_name"] == "fixmatch"
    assert "query_ssl.strong_view_policy" not in objective
    assert cfg.local_update_profile.validation_scorer_backend_name == (
        "peft_classifier_eval"
    )
    assert cfg.validation.scorer_backend_name == "peft_classifier_eval"
    assert cfg.validation.score_policy_name is None
    assert cfg.validation.score_top_k is None
    assert "selection" not in cfg.training_task.objective
    assert "pseudo_label_algorithm_name" not in cfg.training_task.objective
    assert "acceptance_policy_name" not in cfg.training_task.objective


def test_federated_simulation_composes_named_initial_checkpoint_preset(
    tmp_path: Path,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/model_architecture/initial_checkpoint="
                "supervised_20260612_step2000",
            ],
        )

    request = build_simulation_request_from_config(cfg, output_dir=tmp_path)

    assert cfg.query_adaptation_initial_checkpoint.name == (
        "supervised_20260612_step2000"
    )
    assert cfg.query_adaptation_initial_checkpoint.mode == "required"
    assert cfg.query_adaptation_initial_checkpoint.manifest_path == (
        "runs/central/supervised/peft_classifier/"
        "peft_clf_2026_06_12_154038/checkpoints/"
        "epoch_0001_step_002000/manifest.json"
    )
    assert request.initial_checkpoint_config.name == ("supervised_20260612_step2000")
    assert request.initial_checkpoint_config.mode == "required"
    assert request.initial_checkpoint_config.manifest_path == (
        "runs/central/supervised/peft_classifier/"
        "peft_clf_2026_06_12_154038/checkpoints/"
        "epoch_0001_step_002000/manifest.json"
    )
    assert cfg.federated_run_budget.output_dir == "runs/_smoke/fl_ssl"
    assert cfg.federated_run_budget.client_count == 10
    assert cfg.federated_run_budget.rounds == 3
    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.local_files_only is True
    assert cfg.fl_data.source_mode == "materialized_client_split"
    assert cfg.fl_data.split_manifest == (
        "data/datasets/fl_client_splits/shared_client_labeled/"
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_"
        "validation-ourafla_reddit_test-ourafla_reddit_labels_pc1024_"
        "shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_"
        "clients10_seed42/manifest.json"
    )
    assert cfg.sweep.axis == "none"
    assert cfg.sweep.output_dir == "runs/_smoke/fl_ssl"
    assert list(cfg.sweep.seed.members) == [42, 43, 44]
    assert list(cfg.sweep.client_count.members) == list(range(1, 11))
    assert cfg.sweep.client_count.split_manifest_by_client_count is None
    assert cfg.run_safety.max_total_rounds_without_ack == 30
    assert cfg.run_safety.allow_long_run is False
    assert cfg.run_safety.long_run_ack is None
    assert cfg.run_safety.required_long_run_ack == "ALLOW_FL_SSL_LONG_RUN"
    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.shard_policy.dominant_ratio is None
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
    assert "server_step_policy" not in cfg
    assert "peer_context_policy" not in cfg
    assert "update_partition_policy" not in cfg
    assert "aggregation_weight_policy" not in cfg
    assert "query_multiview_source" not in cfg


def test_fl_client_split_materialization_uses_query_data_source_and_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/materialize_fl_client_split",
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "query_data_selection.unlabeled=ourafla_reddit",
                "run_controls/fl_ssl/budget=main",
                "federated_run_budget.client_count=8",
                "strategy_axes/fl_topology/shard_policy=dirichlet_alpha03",
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
        "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
    )
    assert cfg.fl_client_split_materialization.source_test_jsonl.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
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
                "strategy_axes/fl_topology/labeled_exposure=shared_client_seed",
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
                "strategy_axes/fl_topology/shard_policy=dirichlet_alpha03",
                "strategy_axes/fl_topology/labeled_exposure=shared_client_seed",
            ],
        )

    assert cfg.federated_run_budget.client_count == 10
    assert cfg.fl_client_split_materialization.split_id == (
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "test-ourafla_reddit_"
        "shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_"
        "clients10_seed42"
    )


def test_federated_simulation_config_keeps_fl_semantic_axes_separate() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/fl_ssl/run_federated_simulation")

    assert cfg.fl_data.source_mode == "materialized_client_split"
    assert cfg.fl_data.split_manifest.endswith(
        "labels_pc1024_shared_client_seed_dirichlet_label_skew_"
        "dominantNone_alpha0.3_clients10_seed42/manifest.json"
    )
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
    assert cfg.round_runtime.update_family_name == "peft_text_encoder"
    assert cfg.round_runtime.runtime_payload_key == "peft_text_encoder"
    assert "peft_classifier" not in cfg.round_runtime
    assert "peft_text_encoder" in cfg.round_runtime.runtime_payloads
    assert cfg.round_runtime.composition_slug_builder == (
        "methods.adaptation.peft_text_encoder.update_family_runtime."
        "build_peft_text_encoder_composition_slug"
    )
    assert cfg.round_runtime.round_runtime_payload_builder == (
        "methods.adaptation.peft_text_encoder.simulation_runtime.round_runtime."
        "build_peft_encoder_round_runtime_payloads"
    )
    assert list(cfg.round_runtime.local_objective_executors) == [
        "scripts.runtime_adapters.federated_agent."
        "generic_client_runtime_bridge."
        "run_method_owned_client_round_if_supported",
        "scripts.runtime_adapters.federated_agent."
        "generic_client_runtime_bridge."
        "run_query_ssl_client_round_if_supported",
    ]
    assert cfg.round_runtime.client_round_runtime.base_state_materializer == (
        "scripts.runtime_adapters.federated_agent.base_state_materialization."
        "load_peft_encoder_base_parameters_with_timing"
    )
    assert cfg.round_runtime.release_transient_model_cache_after_client is True
    assert cfg.round_runtime.client_round_runtime.query_ssl_training_runner == (
        "scripts.runtime_adapters.federated_agent.query_ssl_training_request."
        "run_query_ssl_peft_encoder_local_training"
    )
    assert (
        cfg.round_runtime.server_round_runtime.final_projection_artifacts_builder
        == (
            "methods.adaptation.peft_text_encoder.simulation_runtime."
            "final_projection.build_peft_encoder_final_projection_artifacts_from_state"
        )
    )
    assert cfg.round_runtime.initial_state_builder == (
        "methods.adaptation.peft_text_encoder.update_family_runtime."
        "build_initial_peft_encoder_state"
    )
    assert cfg.round_runtime.validation_evaluator == (
        "methods.adaptation.peft_text_encoder.evaluation."
        "evaluate_peft_encoder_simulation_validation_payload"
    )
    assert cfg.round_runtime.final_projection_builder == (
        "scripts.runtime_adapters.federated_server.generic_server_runtime_bridge."
        "build_final_projection_artifacts"
    )
    assert cfg.round_runtime.transient_resource_cleaner == (
        "methods.adaptation.peft_text_encoder.resource_cache."
        "clear_peft_encoder_transient_resource_cache"
    )
    assert cfg.round_runtime.payload_adapter_kind == "peft_classifier"
    assert "adapter_family_name" not in cfg.round_runtime
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert cfg.report.labeled_ratio == cfg.client_pool_split.labeled_ratio
    assert cfg.report.unlabeled_ratio == cfg.client_pool_split.unlabeled_ratio
    assert len(cfg.sweep.seed.members) == cfg.report.seed_count
    assert cfg.report.seed_count == 3


def test_federated_simulation_fl_client_split_context_selects_manifest() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "execution_context/fl_client_split=shared_general_reddit_pc100_alpha03_clients10",
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
def test_federated_simulation_fl_client_split_context_covers_shared_budgets(
    selector: str,
    labeled: str,
    budget: int,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[f"execution_context/fl_client_split={selector}"],
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
            compose(
                config_name="entrypoints/fl_ssl/run_federated_simulation",
                overrides=[
                    f"strategy_axes/fssl_method={descriptor.name}",
                    f"strategy_axes/ssl_objective/local_update_profile={local_profile_name}",
                ],
            )
        for runtime_pair in descriptor.recipe.supported_runtime_pairs:
            with initialize_config_module(version_base=None, config_module="conf"):
                cfg = compose(
                    config_name="entrypoints/fl_ssl/run_federated_simulation",
                    overrides=[
                        f"strategy_axes/fssl_method={descriptor.name}",
                        (
                            "strategy_axes/ssl_objective/local_update_profile="
                            f"{local_profile_name}"
                        ),
                        (
                            "strategy_axes/model_architecture/update_family="
                            f"{runtime_pair.update_family_name}"
                        ),
                        "round_runtime.aggregation_backend_name="
                        f"{runtime_pair.aggregation_backend_name}",
                    ],
                )

            _assert_manual_fl_runtime_is_compatible(cfg)
            assert cfg.round_runtime.update_family_name == (
                runtime_pair.update_family_name
            )


def test_method_owned_fedmatch_keeps_single_local_profile() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
            ],
        )

    execution_plan = _build_execution_plan(cfg)
    local_update_profile = _resolve_local_update_profile(
        cfg=cfg,
        execution_plan=execution_plan,
    )

    assert local_update_profile.algorithm_profile_name == "peft_classifier_update_v1"


def test_fedmatch_method_config_injects_original_parameter_snapshot() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
            ],
        )

    expected = fedmatch_original_parameter_mapping()
    ssl_method_config = _build_ssl_method_config(
        cfg,
        execution_plan=_build_execution_plan(cfg),
    )
    assert ssl_method_config is not None

    assert cfg.ssl_method.local_budget_policy == "iteration_capped"
    assert cfg.training_task.max_steps == 50
    assert "original_parameters" not in cfg.ssl_method
    assert "original_source" not in cfg.ssl_method
    assert "trace_mapping" not in cfg.ssl_method
    assert "client_step" not in cfg.ssl_method
    assert "server_step" not in cfg.ssl_method
    assert "round_state_exchange" not in cfg.ssl_method
    assert cfg.ssl_method.scenario == "labels-at-client"
    assert cfg.ssl_method.use_original_parameters is True
    assert dict(cfg.ssl_method.parameter_overrides) == {}
    assert ssl_method_config.implementation_status == (
        "partitioned_trainable_state_slice_v1"
    )
    assert ssl_method_config.original_source["repository"] == (
        FEDMATCH_ORIGINAL_REPOSITORY
    )
    assert ssl_method_config.original_source["commit"] == FEDMATCH_ORIGINAL_COMMIT
    assert ssl_method_config.original_parameters == expected
    assert ssl_method_config.effective_parameters == expected
    assert ssl_method_config.parameter_overrides == {}
    assert ssl_method_config.parameter_override_status == "original"
    assert ssl_method_config.trace_mapping["supervised_partition"] == "sigma"
    assert ssl_method_config.trace_mapping["unsupervised_partition"] == "psi"
    assert (
        ssl_method_config.trace_mapping["parameter_decomposition"]
        == "peft_text_encoder_sigma_psi"
    )
    assert ssl_method_config.trace_mapping["aggregation_weight_policy"] == "uniform"
    assert ssl_method_config.trace_mapping["update_partition_policy"] == "partitioned"
    assert ssl_method_config.trace_mapping["partition_scheme"] == "sigma_psi"
    assert ssl_method_config.client_step["task_type"] == (
        "federated_ssl_method_local_step"
    )


def test_federated_simulation_local_ssl_policy_defaults_to_query_ssl_algorithm() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=flexmatch_usb_v1",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_resolve_labeled_exposure_policy_mapping(
            cfg=cfg,
            execution_plan=_build_execution_plan(cfg),
        ),
    )

    assert cfg.local_ssl_policy.name == cfg.query_ssl_method.algorithm_name
    assert capability_plan.local_ssl_policy_name == "flexmatch"
    assert capability_plan.server_update_policy_name == "fedavg_merged_delta"


def test_method_owned_fedmatch_labels_at_client_scenario_derives_capabilities() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
                "strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_resolve_labeled_exposure_policy_mapping(
            cfg=cfg,
            execution_plan=_build_execution_plan(cfg),
        ),
    )

    assert cfg.local_ssl_policy.name == cfg.query_ssl_method.algorithm_name
    assert capability_plan.local_ssl_policy_name == "fedmatch_agreement"
    assert capability_plan.server_update_policy_name == "fedmatch_partitioned"
    assert capability_plan.update_partition_policy_name == "partitioned"
    assert capability_plan.peer_context_policy_name == "fixed_probe_output_knn"
    assert capability_plan.server_step_policy_name == "none"


def test_method_owned_fedmatch_ignores_query_ssl_lower_axis_objective_payload(
    tmp_path: Path,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
                "strategy_axes/ssl_objective/consistency_method=flexmatch_usb_v1",
            ],
        )

    request = build_simulation_request_from_config(cfg, output_dir=tmp_path)

    assert request.query_ssl_objective_config is None
    assert request.capability_plan is not None
    assert request.capability_plan.local_ssl_policy_name == "fedmatch_agreement"
    objective = request.training_task_config.objective_config.to_mapping()
    assert "query_ssl.algorithm_name" not in objective
    assert "query_ssl.method_name" not in objective
    assert objective["algorithm_profile_name"] == "peft_classifier_update_v1"


def test_federated_simulation_main_default_gives_each_client_unlabeled_rows(
    tmp_path: Path,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["run_controls/fl_ssl/budget=main"],
        )

    request = build_simulation_request_from_config(cfg, output_dir=tmp_path)

    assert request.client_count == 10
    assert request.shard_policy.name == "dirichlet_label_skew"
    assert request.shard_policy.alpha == 0.3
    assert request.materialized_dataset_split is not None
    assert all(
        shard.unlabeled_rows
        for shard in request.materialized_dataset_split.client_shards
    )


def test_method_owned_fedmatch_labels_at_server_derives_method_capabilities() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fssl_method=fedmatch",
                "ssl_method.scenario=labels-at-server",
                "fl_method.composition_mode=method_owned",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_resolve_labeled_exposure_policy_mapping(
            cfg=cfg,
            execution_plan=_build_execution_plan(cfg),
        ),
    )

    assert capability_plan.local_ssl_policy_name == "fedmatch_agreement"
    assert capability_plan.server_update_policy_name == "fedmatch_partitioned"
    assert capability_plan.update_partition_policy_name == "partitioned"
    assert capability_plan.peer_context_policy_name == "fixed_probe_output_knn"
    assert capability_plan.server_step_policy_name == "supervised_seed_step"
    assert capability_plan.labeled_exposure_policy_name == "server_only_seed"
    assert capability_plan.local_supervision_regime_name == "client_unlabeled_only"


def test_federated_simulation_update_family_declares_server_step_executor() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fl_topology/server_step=supervised_seed_step",
            ],
        )

    assert cfg.server_step_policy.name == "supervised_seed_step"
    assert (
        cfg.round_runtime.server_step_executors.supervised_seed_step
        == "scripts.runtime_adapters.federated_server.generic_server_runtime_bridge."
        "run_supervised_seed_step"
    )
    assert "executor" not in cfg.server_step_policy


def test_federated_simulation_can_express_fedmatch_physical_faithful_shape() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "run_controls/fl_ssl/budget=reduced",
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
            ],
        )

    capability_plan = _build_capability_plan(
        cfg=cfg,
        labeled_exposure_policy=_resolve_labeled_exposure_policy_mapping(
            cfg=cfg,
            execution_plan=_build_execution_plan(cfg),
        ),
    )

    assert cfg.federated_run_budget.name == "reduced"
    assert cfg.federated_run_budget.rounds == 5
    assert cfg.local_ssl_policy.name == cfg.query_ssl_method.algorithm_name
    assert capability_plan.local_ssl_policy_name == "fedmatch_agreement"
    assert capability_plan.server_update_policy_name == "fedmatch_partitioned"
    assert capability_plan.update_partition_policy_name == "partitioned"
    assert capability_plan.aggregation_weight_policy.name == "uniform"
    assert capability_plan.peer_context_policy_name == "fixed_probe_output_knn"


def test_federated_simulation_rejects_legacy_peer_context_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        with pytest.raises(Exception, match="prediction_similarity_topk"):
            compose(
                config_name="entrypoints/fl_ssl/run_federated_simulation",
                overrides=[
                    "strategy_axes/fl_topology/peer_context=prediction_similarity_topk",
                ],
            )


def test_fedmatch_method_config_records_parameter_overrides_as_ablation() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
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
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
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


def test_fedmatch_leaf_is_public_method_identity_with_scenario_axis() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/fssl_method=fedmatch",
                "fl_method.composition_mode=method_owned",
                "ssl_method.scenario=labels-at-server",
            ],
        )

    execution_plan = _build_execution_plan(cfg)
    ssl_method_config = _build_ssl_method_config(cfg, execution_plan=execution_plan)

    assert execution_plan.descriptor_name == "fedmatch"
    assert ssl_method_config is not None
    assert ssl_method_config.scenario == "labels-at-server"


@pytest.mark.parametrize(
    "profile_name",
    [
        "peft_classifier_update_v1",
    ],
)
def test_federated_simulation_local_update_profile_is_hydra_source_of_truth(
    profile_name: str,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                f"strategy_axes/ssl_objective/local_update_profile={profile_name}",
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
    assert cfg.sweep.output_dir == "runs/fl_ssl"
    assert cfg.training_task.local_epochs == 1
    assert cfg.training_task.batch_size == 8
    assert cfg.train_batch_size == 8
    assert cfg.training_task.max_steps == 50


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
    assert cfg.sweep.output_dir == "runs/fl_ssl"
    assert cfg.training_task.batch_size == 8
    assert cfg.train_batch_size == 8


def test_federated_simulation_shared_seed_flexmatch_reduced_command_shape() -> None:
    split_manifest = (
        "data/datasets/fl_client_splits/"
        "shared_client_labeled/"
        "labeled-ourafla_reddit_unlabeled-ourafla_reddit_"
        "test-ourafla_reddit_"
        "shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_"
        "clients10_seed42/manifest.json"
    )
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "run_controls/fl_ssl/budget=reduced",
                "fl_method.composition_mode=manual",
                "strategy_axes/fl_topology/shard_policy=dirichlet_alpha03",
                "strategy_axes/ssl_objective/consistency_method=flexmatch_usb_v1",
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
    assert cfg.round_runtime.payload_adapter_kind == "peft_classifier"
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
                "~execution_context/fl_client_split",
                "strategy_axes/fl_topology/shard_policy=label_dominant",
                "shard_policy.dominant_ratio=0.6",
            ],
        )

    assert cfg.shard_policy.name == "label_dominant"
    assert cfg.shard_policy.dominant_ratio == 0.6
    assert cfg.training_task.objective.algorithm_profile_name == (
        "peft_classifier_update_v1"
    )
    assert cfg.validation.scorer_backend_name == "peft_classifier_eval"


def test_federated_simulation_supports_dirichlet_shard_policy_override() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=["strategy_axes/fl_topology/shard_policy=dirichlet_alpha03"],
        )

    assert cfg.shard_policy.name == "dirichlet_label_skew"
    assert cfg.shard_policy.alpha == 0.3
    assert cfg.shard_policy.dominant_ratio is None


def test_federated_simulation_rejects_removed_manual_baseline_descriptor_override() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        with pytest.raises(Exception, match="fssl_method"):
            compose(
                config_name="entrypoints/fl_ssl/run_federated_simulation",
                overrides=["strategy_axes/fssl_method=fedavg_pseudo_label"],
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
    assert plan.manual_axes.update_family == "peft_text_encoder"


def test_federated_simulation_manual_plan_supports_direct_runtime_leaf_overrides() -> (
    None
):
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "fl_method.composition_mode=manual",
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

    assert cfg.local_update_profile.algorithm_profile_name == (
        "peft_classifier_update_v1"
    )
    assert cfg.round_runtime.payload_adapter_kind == "peft_classifier"
    assert cfg.round_runtime.aggregation_backend_name == "fedavg"
    assert plan.method_name == "manual"
    assert plan.execution_role == "manual_baseline"
    assert plan.manual_axes.client_ssl_objective == "fixmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "peft_text_encoder"


def test_federated_simulation_manual_plan_switches_ssl_algorithm_by_hydra_name(
    tmp_path: Path,
) -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=flexmatch_usb_v1",
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
    assert "query_ssl" not in cfg.training_task.objective
    request = build_simulation_request_from_config(cfg, output_dir=tmp_path)
    objective = request.training_task_config.objective_config.to_mapping()
    assert objective["query_ssl.method_name"] == "flexmatch_usb_v1"
    assert objective["query_ssl.algorithm_name"] == "flexmatch"
    assert plan.manual_axes.client_ssl_objective == "flexmatch"
    assert plan.manual_axes.server_aggregation == "fedavg"
    assert plan.manual_axes.update_family == "peft_text_encoder"


def test_run_peft_supervised_control_defaults_to_gpu_online_scaffold() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_peft_supervised_control"
        )

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.paper_backbone.model_id == "mixedbread-ai/mxbai-embed-large-v1"
    assert cfg.trainable_surface.name == "peft_text_encoder"
    assert cfg.trainable_surface.trainable_state == "peft_adapter_and_classifier_head"
    assert cfg.peft_adapter.target_modules == "all-linear"
    assert cfg.selection_set == "test"
    assert cfg.output_dir == "runs/central/supervised/peft_classifier"
    assert cfg.central_ssl_budget.output_root == "runs"
    assert cfg.epoch_artifact_kind == "peft_adapter_classifier"
    assert cfg.epoch_artifact_every_epochs == 1


def test_run_full_text_encoder_supervised_control_defaults_to_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name=(
                "entrypoints/central/ssl_control/"
                "run_full_text_encoder_supervised_control"
            )
        )

    assert cfg.dataset.name == "ourafla"
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.paper_backbone.name == "mxbai_encoder"
    assert cfg.trainable_surface.name == "full_text_encoder"
    assert cfg.trainable_surface.trainable_state == "full_encoder_and_classifier_head"
    assert cfg.trainable_surface.supports_initial_adapter is False
    assert cfg.selection_set == "test"
    assert cfg.output_dir == "runs/central/supervised/full_text_encoder"
    assert cfg.central_ssl_budget.output_root == "runs"


def test_run_query_ssl_control_defaults_to_fixmatch_precomputed_views() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control"
        )

    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is True
    assert "teacher_provider" not in cfg
    assert "pseudo_label_algorithm" not in cfg
    assert "ssl_input_mode" not in cfg
    assert cfg.trainable_surface.name == "peft_text_encoder"
    assert cfg.group_by_query_ssl_method is True
    assert cfg.query_ssl_method.name == "fixmatch_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "fixmatch"
    assert (
        cfg.query_source.name == "labeled_szegeelim_general4_unlabeled_ourafla_reddit_"
        "test_ourafla_reddit"
    )
    assert cfg.query_data_selection.labeled == "szegeelim_general4"
    assert cfg.query_data_selection.unlabeled == "ourafla_reddit"
    assert cfg.query_data_selection.validation == "ourafla_reddit"
    assert cfg.query_data_selection.test == "ourafla_reddit"
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
    assert "query_ssl_augmenter" not in cfg
    assert "query_ssl_strong_view_policy" not in cfg
    assert cfg.train_batch_size == 8
    assert cfg.eval_batch_size == 32
    assert cfg.drop_last_train_batches is True
    assert cfg.drop_last_unlabeled_batches is True
    assert cfg.resume_checkpoint_every_epochs == 0


def test_run_query_ssl_control_switches_method_by_hydra_name() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/ssl_objective/consistency_method=pseudolabel_usb_v1",
                "output_dir=runs/run_query_ssl_control_pseudolabel",
            ],
        )

    assert cfg.query_ssl_method.name == "pseudolabel_usb_v1"
    assert cfg.query_ssl_method.algorithm_name == "pseudolabel"
    assert cfg.query_ssl_method.require_multiview is False
    assert (
        cfg.query_source.name == "labeled_szegeelim_general4_unlabeled_ourafla_reddit_"
        "test_ourafla_reddit"
    )
    assert cfg.output_dir == "runs/run_query_ssl_control_pseudolabel"


def test_run_query_ssl_control_supports_general_labeled_reddit_pool() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "query_data_selection.labeled=szegeelim_general4",
                "query_data_selection.unlabeled=ourafla_reddit",
                "query_data_selection.validation=ourafla_reddit",
                "query_data_selection.test=ourafla_reddit",
            ],
        )

    assert (
        cfg.query_source.name == "labeled_szegeelim_general4_unlabeled_ourafla_reddit_"
        "test_ourafla_reddit"
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
    assert "validation" not in cfg.eval_sets
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
    )
    assert cfg.output_dir.endswith(
        "labeled-szegeelim_general4_unlabeled-ourafla_reddit_test-ourafla_reddit"
    )


def test_run_query_ssl_control_supports_pc100_labeled_budget() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                f"execution_context/query_labeled_budget={PC100_QUERY_LABEL_BUDGET}",
                "run_controls/central_ssl/budget=smoke",
                "central_ssl_budget.max_train_steps=2000",
            ],
        )

    assert cfg.query_labeled_budget.name == PC100_QUERY_LABEL_BUDGET
    assert cfg.query_labeled_budget.count_per_class == 100
    assert cfg.query_data_selection.labeled == "szegeelim_general4"
    assert cfg.query_data_selection.unlabeled == "ourafla_reddit"
    assert cfg.query_data_selection.test == "ourafla_reddit"
    assert cfg.train_jsonl == PC100_SZEGEELIM_LABEL_WITH_VIEWS
    assert cfg.unlabeled_jsonl.endswith(
        "data/datasets/ourafla_mental_health/views/"
        "labeled1024_per_class_seed42_v1/"
        "backtranslation_nllb_en_de_fr_usb_v1/unlabeled_pool.with_views.jsonl"
    )
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
    )
    assert cfg.query_data_selection_slug == (
        "labeled-szegeelim_general4_labels-pc100_unlabeled-"
        "ourafla_reddit_test-ourafla_reddit"
    )
    assert cfg.output_dir.endswith(
        "labeled-szegeelim_general4_labels-pc100_unlabeled-"
        "ourafla_reddit_test-ourafla_reddit"
    )
    assert cfg.max_train_steps == 2000


def test_run_query_ssl_control_supports_full_text_encoder_flexmatch_pc100() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "strategy_axes/model_architecture/trainable_surface=full_text_encoder",
                "strategy_axes/ssl_objective/consistency_method=flexmatch_usb_v1",
                f"execution_context/query_labeled_budget={PC100_QUERY_LABEL_BUDGET}",
                "run_controls/central_ssl/budget=smoke",
            ],
        )

    assert cfg.trainable_surface.name == "full_text_encoder"
    assert cfg.trainable_surface.requires_peft_adapter is False
    assert cfg.trainable_surface.central_ssl.local_session_runner.endswith(
        "run_query_ssl_full_text_encoder_local_session"
    )
    assert cfg.trainable_surface.central_ssl.artifact_writer.endswith(
        "write_full_text_encoder_run_artifacts"
    )
    assert cfg.query_ssl_method.name == "flexmatch_usb_v1"
    assert cfg.query_labeled_budget.name == PC100_QUERY_LABEL_BUDGET
    assert cfg.output_dir.startswith("runs/_smoke/central/ssl/full_text_encoder/")
    assert cfg.learning_rate == 0.00002


def test_run_query_ssl_control_supports_general_pool_with_reddit_eval() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
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
        "test_ourafla_reddit"
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
    assert "validation" not in cfg.eval_sets
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
    )
    assert cfg.output_dir.endswith(
        "labeled-szegeelim_general4_unlabeled-szegeelim_general4_test-ourafla_reddit"
    )


def test_run_query_ssl_control_uses_test_only_eval_set() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/central/ssl_control/run_query_ssl_control",
            overrides=[
                "query_data_selection.validation=szegeelim_general4",
                "query_data_selection.test=ourafla_reddit",
            ],
        )

    assert "validation" not in cfg.eval_sets
    assert cfg.eval_sets.test.endswith(
        "data/datasets/ourafla_mental_health/query_ssl/"
        "labeled1024_per_class_seed42_v1/test_balanced_validation_test_seed42.jsonl"
    )
    assert cfg.selection_set == "test"


def test_dataset_pipeline_defaults_to_ourafla_and_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name="entrypoints/dataset_pipeline/run_dataset_pipeline")

    assert cfg.dataset.name == "ourafla"
    assert cfg.dataset.test_jsonl == cfg.dataset.test_labeled_jsonl
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is False
