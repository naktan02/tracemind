"""run_dataset_pipeline 스크립트 unit tests."""

from __future__ import annotations

from hydra import compose, initialize_config_module

from scripts.datasets.run_dataset_pipeline import supported_dataset_aliases


def test_supported_dataset_aliases_include_ourafla_and_cssrs() -> None:
    aliases = supported_dataset_aliases()

    assert "ourafla" in aliases
    assert "cssrs" in aliases


def test_hydra_dataset_group_contains_pipeline_metadata() -> None:
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="jobs/datasets/run_dataset_pipeline",
            overrides=["dataset=ourafla"],
        )

    assert cfg.dataset.stages == ["download", "map", "split", "prototype"]
    assert cfg.dataset.split.source == "train"
    assert cfg.dataset.prototype.source == "split_train"
    assert cfg.dataset.sources.train.data_file == "mental_heath_unbanlanced.csv"
