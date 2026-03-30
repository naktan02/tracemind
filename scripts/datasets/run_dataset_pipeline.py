"""Hydra dataset config 기반 데이터셋 파이프라인 러너."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.build_labeled_query_set import (  # noqa: E402
    build_labeled_query_set,
    load_mapping_config,
)
from scripts.datasets.download_dataset import (  # noqa: E402
    build_default_output_path,
    download_huggingface_dataset_to_csv,
)
from scripts.datasets.split_labeled_query_set import (  # noqa: E402
    build_split_artifacts,
)
from scripts.prototypes.seed_prototypes import seed_prototype_pack  # noqa: E402

PIPELINE_STAGE_ORDER = ("download", "map", "split", "prototype")
DATASET_CONFIG_DIR = PROJECT_ROOT / "scripts/conf/dataset"


def supported_dataset_aliases() -> tuple[str, ...]:
    """Hydra dataset group에 등록된 dataset alias 목록을 반환한다."""
    return tuple(sorted(path.stem for path in DATASET_CONFIG_DIR.glob("*.yaml")))


def load_dataset_group(alias: str) -> DictConfig:
    """dataset group YAML 하나를 직접 읽는다."""
    dataset_path = DATASET_CONFIG_DIR / f"{alias}.yaml"
    if not dataset_path.exists():
        raise ValueError(
            f"Unknown dataset alias: '{alias}'. "
            f"supported={supported_dataset_aliases()}"
        )
    return OmegaConf.load(dataset_path)


def _resolve_project_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
    mapping_config_path = _resolve_project_path(
        source_cfg.get("mapping_config")
    )
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


def resolve_prototype_input_jsonl(
    *,
    dataset_cfg: DictConfig,
    mapped_outputs: dict[str, dict[str, Path]],
    split_output: dict[str, Path] | None,
    split_dir: Path,
) -> Path:
    """prototype stage 입력 JSONL 경로를 결정한다."""
    prototype_cfg = dataset_cfg.get("prototype")
    if prototype_cfg is None:
        raise ValueError(f"Dataset '{dataset_cfg.name}' has no prototype config.")

    prototype_source = _require_string(
        prototype_cfg.get("source"),
        field_name="prototype.source",
    )
    if prototype_source == "split_train":
        if split_output is not None:
            return split_output["train_jsonl"]
        split_cfg = dataset_cfg.get("split")
        if split_cfg is None:
            raise ValueError(
                f"Dataset '{dataset_cfg.name}' prototype.source=split_train "
                "requires split config."
            )
        split_name = _require_string(
            split_cfg.get("split_name"),
            field_name="split.split_name",
        )
        return split_dir / f"{split_name}.train.jsonl"

    if prototype_source.startswith("mapped:"):
        mapped_name = prototype_source.removeprefix("mapped:")
        mapped_output = mapped_outputs.get(mapped_name)
        if mapped_output is not None:
            return mapped_output["jsonl"]

        source_cfg = dataset_cfg.sources.get(mapped_name)
        if source_cfg is None:
            raise ValueError(
                f"Dataset '{dataset_cfg.name}' prototype source "
                f"'{prototype_source}' is unknown."
            )
        return resolve_mapped_output_paths(
            source_cfg=source_cfg,
            labeled_query_set_dir=_resolve_project_path(
                dataset_cfg.train_labeled_jsonl
            ).parent,
        )["jsonl"]

    raise ValueError(
        f"Dataset '{dataset_cfg.name}' has unsupported "
        f"prototype.source='{prototype_source}'."
    )


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


def run_dataset(
    *,
    cfg: DictConfig,
) -> Path:
    """dataset group 하나에 대해 pipeline을 실행한다."""
    dataset_cfg = cfg.dataset
    only_stages = {str(stage) for stage in cfg.only_stages}
    selected_stages = _selected_stages(dataset_cfg, only_stages)

    raw_dir = _resolve_project_path(str(cfg.paths.raw_dir)) or PROJECT_ROOT
    labeled_query_set_dir = (
        _resolve_project_path(str(cfg.paths.labeled_query_set_dir)) or PROJECT_ROOT
    )
    split_dir = _resolve_project_path(str(cfg.paths.split_dir)) or PROJECT_ROOT
    prototype_pack_dir = (
        _resolve_project_path(str(cfg.paths.prototype_pack_dir)) or PROJECT_ROOT
    )
    prototype_build_state_dir = (
        _resolve_project_path(str(cfg.paths.prototype_build_state_dir)) or PROJECT_ROOT
    )
    pipeline_run_dir = (
        _resolve_project_path(str(cfg.paths.pipeline_run_dir)) or PROJECT_ROOT
    )
    cache_dir = _resolve_project_path(str(cfg.paths.cache_dir)) or PROJECT_ROOT

    raw_outputs: dict[str, str] = {}
    mapped_outputs: dict[str, dict[str, Path]] = {}
    split_output: dict[str, Path] | None = None
    prototype_output: dict[str, str] | None = None

    if "download" in selected_stages:
        for source_name, source_cfg in dataset_cfg.sources.items():
            source_kind = _require_string(source_cfg.get("kind"), field_name="kind")
            if source_kind != "huggingface":
                raise ValueError(
                    f"Dataset '{dataset_cfg.name}' source '{source_name}' has "
                    f"unsupported kind '{source_kind}'."
                )

            output_path = resolve_dataset_output_path(
                source_cfg=source_cfg,
                raw_dir=raw_dir,
            )
            download_huggingface_dataset_to_csv(
                dataset_id=_require_string(
                    source_cfg.get("dataset_id"),
                    field_name=f"sources.{source_name}.dataset_id",
                ),
                split=_require_string(
                    source_cfg.get("split"),
                    field_name=f"sources.{source_name}.split",
                ),
                output_dir=raw_dir,
                cache_dir=cache_dir,
                data_file=source_cfg.get("data_file"),
                output_path=output_path,
                revision=source_cfg.get("revision"),
            )
            raw_outputs[source_name] = str(output_path)

    if "map" in selected_stages:
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

    if "split" in selected_stages:
        split_cfg = dataset_cfg.get("split")
        if split_cfg is None:
            raise ValueError(
                f"Dataset '{dataset_cfg.name}' does not define split config."
            )

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
        split_output = {
            "train_jsonl": train_path,
            "validation_jsonl": validation_path,
            "manifest": manifest_path,
        }
    elif dataset_cfg.get("split") is not None:
        split_output = resolve_split_output_paths(
            split_cfg=dataset_cfg.split,
            split_dir=split_dir,
        )

    if "prototype" in selected_stages:
        prototype_cfg = dataset_cfg.get("prototype")
        if prototype_cfg is None:
            raise ValueError(
                f"Dataset '{dataset_cfg.name}' does not define prototype config."
            )

        prototype_input_jsonl = resolve_prototype_input_jsonl(
            dataset_cfg=dataset_cfg,
            mapped_outputs=mapped_outputs,
            split_output=split_output,
            split_dir=split_dir,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prototype_version = (
            f"{prototype_cfg.prototype_version_prefix}_{timestamp}"
        )
        (
            pack_path,
            build_state_path,
            manifest_path,
            main_server_pack_path,
            main_server_build_state_path,
        ) = seed_prototype_pack(
            input_jsonl=prototype_input_jsonl,
            output_dir=prototype_pack_dir,
            build_state_output_dir=prototype_build_state_dir,
            prototype_version=prototype_version,
            backend=_require_string(
                prototype_cfg.get("backend"),
                field_name="prototype.backend",
            ),
            embedding_model_id=_require_string(
                prototype_cfg.get("embedding_model_id"),
                field_name="prototype.embedding_model_id",
            ),
            embedding_model_revision=str(
                prototype_cfg.get("embedding_model_revision", "main")
            ),
            translation_model_id=prototype_cfg.get("translation_model_id"),
            translation_model_revision=prototype_cfg.get(
                "translation_model_revision"
            ),
            translation_direction=prototype_cfg.get("translation_direction"),
            batch_size=int(prototype_cfg.get("batch_size", 16)),
            cache_dir=cache_dir,
            device=str(cfg.runtime.device),
            task_prefix=str(prototype_cfg.get("task_prefix", "")),
            local_files_only=bool(cfg.runtime.local_files_only),
            expected_categories=[
                str(value)
                for value in prototype_cfg.get("expected_categories", [])
            ],
            hash_dim=int(prototype_cfg.get("hash_dim", 256)),
            build_strategy=instantiate(cfg.prototype_builder),
        )
        prototype_output = {
            "prototype_build_state": None
            if build_state_path is None
            else str(build_state_path),
            "prototype_pack": str(pack_path),
            "manifest": str(manifest_path),
            "main_server_build_state": None
            if main_server_build_state_path is None
            else str(main_server_build_state_path),
            "main_server_pack": str(main_server_pack_path),
            "input_jsonl": str(prototype_input_jsonl),
        }

    pipeline_run_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_manifest_path = (
        pipeline_run_dir / f"{dataset_cfg.name}__{run_timestamp}.manifest.json"
    )
    run_manifest = {
        "schema_version": "dataset_pipeline_run.v2",
        "dataset_name": dataset_cfg.name,
        "description": str(dataset_cfg.get("description", "")),
        "dataset_config_group": f"dataset={dataset_cfg.name}",
        "dataset_config_path": str(DATASET_CONFIG_DIR / f"{dataset_cfg.name}.yaml"),
        "executed_at": datetime.now(timezone.utc).isoformat(),
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
        "prototype_output": prototype_output,
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


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="datasets/run_dataset_pipeline",
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
