"""run_dataset_pipeline 스크립트 unit tests."""

from __future__ import annotations

from hydra import compose, initialize_config_module

from scripts.datasets.run_dataset_pipeline import (
    resolve_pipeline_output_dir,
    supported_dataset_aliases,
)


def test_supported_dataset_aliases_include_ourafla_and_cssrs() -> None:
    aliases = supported_dataset_aliases()

    assert "ourafla" in aliases
    assert "cssrs" in aliases
    assert "mental_health_kaggle" in aliases


def test_hydra_dataset_group_contains_pipeline_metadata() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/data_pipeline/run_dataset_pipeline",
            overrides=["execution_context/dataset_asset=ourafla"],
        )

    assert cfg.dataset.stages == ["download", "map", "split", "prototype"]
    assert cfg.dataset.split.source == "train"
    assert cfg.dataset.prototype.source == "split_train"
    assert cfg.dataset.sources.train.data_file == "mental_heath_unbanlanced.csv"


def test_hydra_dataset_group_contains_kaggle_mental_health_metadata() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/data_pipeline/run_dataset_pipeline",
            overrides=["execution_context/dataset_asset=mental_health_kaggle"],
        )

    assert cfg.dataset.stages == ["download", "map", "split"]
    assert cfg.dataset.output_root == "data/datasets/szegeelim_mental_health"
    assert cfg.dataset.output_paths.raw_dir.endswith(
        "data/datasets/szegeelim_mental_health/raw"
    )
    assert cfg.dataset.train_labeled_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/mapped/"
        "szegeelim_mental_health_4cat.v1.jsonl"
    )
    assert cfg.dataset.train_jsonl.endswith(
        "data/datasets/szegeelim_mental_health/splits/train_split.v1.train.jsonl"
    )
    assert cfg.dataset.split.source == "train"
    assert cfg.dataset.split.validation_ratio == 0.1
    assert cfg.dataset.sources.train.kind == "kaggle"
    assert cfg.dataset.sources.train.dataset_ref == "szegeelim/mental-health"
    assert cfg.dataset.sources.train.data_file == "Combined Data.csv"
    assert (
        cfg.dataset.sources.train.mapping_config
        == "data/mappings/szegeelim_mental_health_to_4cat.v1.toml"
    )


def test_kaggle_mental_health_dataset_overrides_pipeline_output_dirs() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/data_pipeline/run_dataset_pipeline",
            overrides=["execution_context/dataset_asset=mental_health_kaggle"],
        )

    raw_dir = resolve_pipeline_output_dir(
        cfg=cfg,
        dataset_cfg=cfg.dataset,
        path_key="raw_dir",
    )
    split_dir = resolve_pipeline_output_dir(
        cfg=cfg,
        dataset_cfg=cfg.dataset,
        path_key="split_dir",
    )

    assert str(raw_dir).endswith("data/datasets/szegeelim_mental_health/raw")
    assert str(split_dir).endswith("data/datasets/szegeelim_mental_health/splits")
