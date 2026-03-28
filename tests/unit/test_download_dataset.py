"""download_dataset 스크립트 unit tests."""

from __future__ import annotations

from pathlib import Path

from scripts.download_dataset import build_default_output_path


def test_build_default_output_path_includes_split_name() -> None:
    output_path = build_default_output_path(
        dataset_id="ourafla/Mental-Health_Text-Classification_Dataset",
        split="test",
        output_dir=Path("data/raw"),
    )

    assert output_path == Path(
        "data/raw/ourafla_Mental-Health_Text-Classification_Dataset__test.csv"
    )
