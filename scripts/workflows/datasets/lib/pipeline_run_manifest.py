"""Dataset pipeline run manifest writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig


def write_pipeline_run_manifest(
    *,
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    dataset_config_group: str,
    dataset_config_path: Path,
    selected_stages: list[str],
    raw_outputs: dict[str, str],
    mapped_outputs: dict[str, dict[str, Path]],
    split_output: dict[str, Path] | None,
    pipeline_run_dir: Path,
) -> Path:
    """pipeline 실행 요약 manifest를 기록한다."""

    pipeline_run_dir.mkdir(parents=True, exist_ok=True)
    executed_at = datetime.now(timezone.utc)
    run_manifest_path = (
        pipeline_run_dir
        / f"{dataset_cfg.name}__{executed_at.strftime('%Y%m%dT%H%M%SZ')}.manifest.json"
    )
    run_manifest = {
        "schema_version": "dataset_pipeline_run.v2",
        "dataset_name": dataset_cfg.name,
        "description": str(dataset_cfg.get("description", "")),
        "dataset_config_group": f"{dataset_config_group}={dataset_cfg.name}",
        "dataset_config_path": str(dataset_config_path),
        "executed_at": executed_at.isoformat(),
        "executed_stages": selected_stages,
        "runtime": {
            "name": str(cfg.runtime.name),
            "device": str(cfg.runtime.device),
            "local_files_only": bool(cfg.runtime.local_files_only),
        },
        "raw_outputs": raw_outputs,
        "mapped_outputs": {
            source_name: {
                "jsonl": str(paths["jsonl"]),
                "manifest": str(paths["manifest"]),
            }
            for source_name, paths in mapped_outputs.items()
        },
        "split_output": None
        if split_output is None
        else {name: str(path) for name, path in split_output.items()},
        "sources": {
            source_name: {
                "kind": str(source_cfg.get("kind")),
                "dataset_id": str(source_cfg.get("dataset_id")),
                "split": str(source_cfg.get("split")),
                "data_file": source_cfg.get("data_file"),
                "reference_urls": [
                    str(url) for url in source_cfg.get("reference_urls", [])
                ],
                "mapping_config": source_cfg.get("mapping_config"),
            }
            for source_name, source_cfg in dataset_cfg.sources.items()
        },
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return run_manifest_path
