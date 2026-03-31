"""scripts Hydra config group tests."""

from __future__ import annotations

from hydra import compose, initialize_config_module


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
            ],
        )

    assert cfg.runtime.name == "gpu_local"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is True
    assert cfg.embedding.backend == "hash_debug"
    assert cfg.embedding.model_id == "hash_debug"
    assert cfg.strategy.name == "kmeans"
    assert list(cfg.strategy.kmeans_candidate_ks) == [2]


def test_threshold_sweep_supports_short_leaf_override() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(
            config_name="experiments/prototype_threshold_sweep",
            overrides=["strategy.name=single"],
        )

    assert cfg.strategy.name == "single"
    assert cfg.runtime.name == "gpu_online"


def test_federated_simulation_uses_smoke_preset_by_default() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="experiments/run_federated_simulation")

    assert cfg.federated_run_preset.name == "smoke"
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


def test_dataset_pipeline_defaults_to_ourafla_and_gpu_online() -> None:
    with initialize_config_module(version_base=None, config_module="scripts.conf"):
        cfg = compose(config_name="datasets/run_dataset_pipeline")

    assert cfg.dataset.name == "ourafla"
    assert cfg.dataset.test_jsonl == cfg.dataset.test_labeled_jsonl
    assert cfg.runtime.name == "gpu_online"
    assert cfg.runtime.device == "cuda"
    assert cfg.runtime.local_files_only is False
    assert cfg.prototype_builder.name == "single"
