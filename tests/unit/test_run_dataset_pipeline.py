"""run_dataset_pipeline 스크립트 unit tests."""

from __future__ import annotations

from pathlib import Path

from scripts.datasets.run_dataset_pipeline import load_registry


def test_load_registry_reads_ourafla_and_cssrs_configs() -> None:
    registry = load_registry(Path("data/datasets/registry.yaml"))

    assert registry.schema_version == "dataset_registry.v1"
    assert "ourafla" in registry.datasets
    assert "cssrs" in registry.datasets

    ourafla = registry.datasets["ourafla"]
    assert ourafla.stages == ("download", "map", "split", "prototype")
    assert ourafla.split is not None
    assert ourafla.prototype is not None
    assert ourafla.prototype.source == "split_train"
    assert ourafla.sources["train"].data_file == "mental_heath_unbanlanced.csv"
    assert ourafla.sources["test"].mapping_config is not None
    assert ourafla.sources["test"].mapping_config.is_absolute()

    cssrs = registry.datasets["cssrs"]
    assert cssrs.stages == ("download",)
    assert cssrs.prototype is None
    assert cssrs.sources["train"].dataset_id == "av9ash/CSSR-S_labelled_suicidewatch_posts_reddit"
