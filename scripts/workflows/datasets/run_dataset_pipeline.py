"""Hydra dataset config 기반 데이터셋 파이프라인 러너."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.support.configured_callable import load_configured_callable
from scripts.workflows.datasets.lib.download import build_default_output_path
from scripts.workflows.datasets.lib.label_mapping import (
    build_labeled_query_set,
    load_mapping_config,
)
from scripts.workflows.datasets.lib.pipeline_run_manifest import (
    write_pipeline_run_manifest,
)
from scripts.workflows.datasets.lib.split import (
    build_split_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STAGE_ORDER = ("download", "map", "split")
DATASET_CONFIG_DIR = PROJECT_ROOT / "conf/execution_context/dataset_asset"
DATASET_CONFIG_GROUP = "execution_context/dataset_asset"


def supported_dataset_aliases() -> tuple[str, ...]:
    """Hydra dataset group에 등록된 dataset alias 목록을 반환한다."""
    return tuple(sorted(path.stem for path in DATASET_CONFIG_DIR.glob("*.yaml")))


def load_dataset_group(alias: str) -> DictConfig:
    """dataset group YAML 하나를 직접 읽는다."""
    dataset_path = DATASET_CONFIG_DIR / f"{alias}.yaml"
    if not dataset_path.exists():
        raise ValueError(
            f"Unknown dataset alias: '{alias}'. supported={supported_dataset_aliases()}"
        )
    return OmegaConf.load(dataset_path)


def _resolve_project_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_pipeline_output_dir(
    *,
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    path_key: str,
) -> Path:
    """dataset별 output override가 있으면 우선하고 없으면 entrypoint 기본값을 쓴다."""

    dataset_output_paths = dataset_cfg.get("output_paths")
    if dataset_output_paths is not None:
        dataset_path = dataset_output_paths.get(path_key)
        if dataset_path is not None:
            return _resolve_project_path(str(dataset_path)) or PROJECT_ROOT

    entrypoint_path = cfg.paths.get(path_key)
    if entrypoint_path is None:
        raise ValueError(f"Missing output path: paths.{path_key}")
    return _resolve_project_path(str(entrypoint_path)) or PROJECT_ROOT


def _require_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {field_name}")
    return value


def resolve_dataset_output_path(*, source_cfg: DictConfig, raw_dir: Path) -> Path:
    """source 설정으로부터 raw CSV 출력 경로를 결정한다."""
    output_filename = source_cfg.get("output_filename")
    if isinstance(output_filename, str) and output_filename.strip():
        return raw_dir / output_filename
    return build_default_output_path(
        dataset_id=_require_string(
            source_cfg.get("dataset_id"),
            field_name="dataset_id",
        ),
        split=_require_string(source_cfg.get("split"), field_name="split"),
        output_dir=raw_dir,
    )


def resolve_mapped_output_paths(
    *,
    source_cfg: DictConfig,
    labeled_query_set_dir: Path,
) -> dict[str, Path]:
    """mapping config로부터 labeled JSONL/manifest 경로를 계산한다."""
    mapping_config_path = _resolve_project_path(source_cfg.get("mapping_config"))
    if mapping_config_path is None:
        raise ValueError("mapping_config is required for map stage.")
    mapping_config = load_mapping_config(mapping_config_path)
    dataset_id = _require_string(
        mapping_config.get("dataset_id"),
        field_name="dataset_id",
    )
    return {
        "jsonl": labeled_query_set_dir / f"{dataset_id}.jsonl",
        "manifest": labeled_query_set_dir / f"{dataset_id}.manifest.json",
    }


def resolve_split_output_paths(
    *,
    split_cfg: DictConfig,
    split_dir: Path,
) -> dict[str, Path]:
    """split 설정으로부터 train/validation/manifest 경로를 계산한다."""
    split_name = _require_string(
        split_cfg.get("split_name"),
        field_name="split.split_name",
    )
    return {
        "train_jsonl": split_dir / f"{split_name}.train.jsonl",
        "validation_jsonl": split_dir / f"{split_name}.validation.jsonl",
        "manifest": split_dir / f"{split_name}.manifest.json",
    }


def _selected_stages(dataset_cfg: DictConfig, only_stages: set[str]) -> list[str]:
    dataset_stages = [str(stage) for stage in dataset_cfg.stages]
    invalid_stages = [
        stage for stage in dataset_stages if stage not in PIPELINE_STAGE_ORDER
    ]
    if invalid_stages:
        raise ValueError(
            f"Dataset '{dataset_cfg.name}' has unsupported stages: {invalid_stages}"
        )

    if only_stages:
        unsupported_requested = sorted(only_stages.difference(dataset_stages))
        if unsupported_requested:
            raise ValueError(
                f"Dataset '{dataset_cfg.name}' does not configure stage(s): "
                f"{unsupported_requested}"
            )
    return [
        stage for stage in dataset_stages if not only_stages or stage in only_stages
    ]


def _run_download_stage(
    *,
    dataset_cfg: DictConfig,
    raw_dir: Path,
    cache_dir: Path,
) -> dict[str, str]:
    """download stage를 실행하고 source별 raw CSV 경로를 반환한다."""
    raw_outputs: dict[str, str] = {}
    for source_name, source_cfg in dataset_cfg.sources.items():
        download_cfg = source_cfg.get("download")
        if download_cfg is None:
            raise ValueError(
                f"Dataset '{dataset_cfg.name}' source '{source_name}' has "
                "no download config."
            )
        callable_path = _require_string(
            download_cfg.get("callable_path"),
            field_name=f"sources.{source_name}.download.callable_path",
        )
        download_source = load_configured_callable(
            callable_path,
            field_name=f"sources.{source_name}.download.callable_path",
        )
        output_path = resolve_dataset_output_path(
            source_cfg=source_cfg,
            raw_dir=raw_dir,
        )
        raw_outputs[source_name] = str(
            download_source(
                source_name=source_name,
                source_cfg=source_cfg,
                raw_dir=raw_dir,
                cache_dir=cache_dir,
                output_path=output_path,
            )
        )
    return raw_outputs


def _run_map_stage(
    *,
    dataset_cfg: DictConfig,
    raw_outputs: dict[str, str],
    raw_dir: Path,
    labeled_query_set_dir: Path,
) -> dict[str, dict[str, Path]]:
    """map stage를 실행하고 source별 labeled JSONL 산출물을 반환한다."""
    mapped_outputs: dict[str, dict[str, Path]] = {}
    for source_name, source_cfg in dataset_cfg.sources.items():
        mapping_config = source_cfg.get("mapping_config")
        if mapping_config is None:
            continue

        raw_csv_path = Path(
            raw_outputs.get(
                source_name,
                resolve_dataset_output_path(
                    source_cfg=source_cfg,
                    raw_dir=raw_dir,
                ),
            )
        )
        jsonl_path, manifest_path = build_labeled_query_set(
            raw_csv_path=raw_csv_path,
            mapping_config_path=_resolve_project_path(str(mapping_config)),
            output_dir=labeled_query_set_dir,
        )
        mapped_outputs[source_name] = {
            "jsonl": jsonl_path,
            "manifest": manifest_path,
        }
    return mapped_outputs


def _run_split_stage(
    *,
    dataset_cfg: DictConfig,
    mapped_outputs: dict[str, dict[str, Path]],
    labeled_query_set_dir: Path,
    split_dir: Path,
) -> dict[str, Path]:
    """split stage를 실행하고 train/validation 경로를 반환한다."""
    split_cfg = dataset_cfg.get("split")
    if split_cfg is None:
        raise ValueError(f"Dataset '{dataset_cfg.name}' does not define split config.")

    split_source = _require_string(
        split_cfg.get("source"),
        field_name="split.source",
    )
    mapped_output = mapped_outputs.get(split_source)
    if mapped_output is None:
        source_cfg = dataset_cfg.sources.get(split_source)
        if source_cfg is None:
            raise ValueError(
                f"Dataset '{dataset_cfg.name}' split source '{split_source}' "
                "is unknown."
            )
        mapped_output = resolve_mapped_output_paths(
            source_cfg=source_cfg,
            labeled_query_set_dir=labeled_query_set_dir,
        )

    train_path, validation_path, manifest_path = build_split_artifacts(
        input_jsonl=mapped_output["jsonl"],
        split_name=_require_string(
            split_cfg.get("split_name"),
            field_name="split.split_name",
        ),
        validation_ratio=float(split_cfg.get("validation_ratio", 0.1)),
        seed=int(split_cfg.get("seed", 42)),
        output_dir=split_dir,
    )
    return {
        "train_jsonl": train_path,
        "validation_jsonl": validation_path,
        "manifest": manifest_path,
    }


def _resolve_declared_split_output(
    *,
    dataset_cfg: DictConfig,
    split_dir: Path,
) -> dict[str, Path] | None:
    """split stage를 건너뛴 경우 선언된 산출물 경로를 계산한다."""
    split_cfg = dataset_cfg.get("split")
    if split_cfg is None:
        return None
    return resolve_split_output_paths(
        split_cfg=split_cfg,
        split_dir=split_dir,
    )


def run_dataset(
    *,
    cfg: DictConfig,
) -> Path:
    """dataset group 하나에 대해 pipeline을 실행한다."""
    dataset_cfg = cfg.dataset
    only_stages = {str(stage) for stage in cfg.only_stages}
    selected_stages = _selected_stages(dataset_cfg, only_stages)

    raw_dir = resolve_pipeline_output_dir(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        path_key="raw_dir",
    )
    labeled_query_set_dir = resolve_pipeline_output_dir(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        path_key="labeled_query_set_dir",
    )
    split_dir = resolve_pipeline_output_dir(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        path_key="split_dir",
    )
    pipeline_run_dir = resolve_pipeline_output_dir(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        path_key="pipeline_run_dir",
    )
    cache_dir = resolve_pipeline_output_dir(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        path_key="cache_dir",
    )

    raw_outputs: dict[str, str] = {}
    mapped_outputs: dict[str, dict[str, Path]] = {}
    split_output: dict[str, Path] | None = None

    if "download" in selected_stages:
        raw_outputs = _run_download_stage(
            dataset_cfg=dataset_cfg,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
        )

    if "map" in selected_stages:
        mapped_outputs = _run_map_stage(
            dataset_cfg=dataset_cfg,
            raw_outputs=raw_outputs,
            raw_dir=raw_dir,
            labeled_query_set_dir=labeled_query_set_dir,
        )

    if "split" in selected_stages:
        split_output = _run_split_stage(
            dataset_cfg=dataset_cfg,
            mapped_outputs=mapped_outputs,
            labeled_query_set_dir=labeled_query_set_dir,
            split_dir=split_dir,
        )
    else:
        split_output = _resolve_declared_split_output(
            dataset_cfg=dataset_cfg,
            split_dir=split_dir,
        )

    return write_pipeline_run_manifest(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        dataset_config_group=DATASET_CONFIG_GROUP,
        dataset_config_path=DATASET_CONFIG_DIR / f"{dataset_cfg.name}.yaml",
        selected_stages=selected_stages,
        raw_outputs=raw_outputs,
        mapped_outputs=mapped_outputs,
        split_output=split_output,
        pipeline_run_dir=pipeline_run_dir,
    )


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/dataset_pipeline/run_dataset_pipeline",
)
def main(cfg: DictConfig) -> None:
    if bool(cfg.list_datasets):
        for alias in supported_dataset_aliases():
            dataset_cfg = load_dataset_group(alias)
            stages = [str(stage) for stage in dataset_cfg.get("stages", [])]
            print(f"{alias}\t{','.join(stages)}")
        return

    run_manifest_path = run_dataset(cfg=cfg)
    print(f"dataset={cfg.dataset.name} run_manifest={run_manifest_path}")


if __name__ == "__main__":
    main()
