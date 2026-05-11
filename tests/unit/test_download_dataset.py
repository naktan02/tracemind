"""download_dataset 스크립트 unit tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from scripts.datasets.lib.download import (
    build_default_output_path,
    download_kaggle_dataset_file_to_csv,
)


def test_build_default_output_path_includes_split_name() -> None:
    output_path = build_default_output_path(
        dataset_id="ourafla/Mental-Health_Text-Classification_Dataset",
        split="test",
        output_dir=Path("data/raw"),
    )

    assert output_path == Path(
        "data/raw/ourafla_Mental-Health_Text-Classification_Dataset__test.csv"
    )


def test_download_kaggle_dataset_file_extracts_configured_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w") as archive:
        archive.writestr("nested/Combined Data.csv", "statement,status\nhello,Normal\n")

    def _fake_download(*_args, **_kwargs):
        return archive_buffer.getvalue()

    monkeypatch.setattr(
        "scripts.datasets.lib.download._download_url_bytes",
        _fake_download,
    )

    output_path = download_kaggle_dataset_file_to_csv(
        dataset_ref="szegeelim/mental-health",
        data_file="Combined Data.csv",
        output_path=tmp_path / "mental_health.csv",
        dataset_version_number=1,
    )

    assert output_path.read_text(encoding="utf-8") == (
        "statement,status\nhello,Normal\n"
    )
