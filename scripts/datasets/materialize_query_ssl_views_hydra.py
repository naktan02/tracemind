"""Hydra config 기반 중앙 Query SSL weak/strong view materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.datasets.materialize_query_ssl_views import (
    run_query_ssl_view_materialization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _string_pair(value: object, *, field_name: str) -> tuple[str, str]:
    raw_value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(raw_value, list | tuple) or len(raw_value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values.")
    return (str(raw_value[0]), str(raw_value[1]))


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text == "" else text


def run_query_ssl_view_materialization_from_hydra(
    *,
    cfg: DictConfig,
) -> None:
    """Hydra preset을 Query SSL view materializer 실행 파라미터로 변환한다."""

    view_cfg = cfg.query_view_materialization
    artifacts = run_query_ssl_view_materialization(
        split_dir=_resolve_project_path(str(view_cfg.split_dir)),
        split_name=str(view_cfg.split_name),
        augmenter_name=str(view_cfg.augmenter_name),
        source_lang=str(view_cfg.source_lang),
        pivot_languages=_string_pair(
            view_cfg.pivot_languages,
            field_name="query_view_materialization.pivot_languages",
        ),
        model_id=str(view_cfg.model_id),
        revision=str(view_cfg.revision),
        device=str(view_cfg.device),
        batch_size=int(view_cfg.batch_size),
        max_new_tokens=int(view_cfg.max_new_tokens),
        torch_dtype=str(view_cfg.torch_dtype),
        cache_dir=_optional_string(view_cfg.cache_dir),
        local_files_only=bool(view_cfg.local_files_only),
        chunk_size=int(view_cfg.chunk_size),
        output_root=_resolve_project_path(str(view_cfg.output_root)),
        resume=bool(view_cfg.resume),
        overwrite=bool(view_cfg.overwrite),
    )

    print(f"labeled_train_with_views_jsonl={artifacts.labeled_train_with_views_jsonl}")
    print(
        f"unlabeled_pool_with_views_jsonl={artifacts.unlabeled_pool_with_views_jsonl}"
    )
    print(f"manifest_json={artifacts.manifest_json}")
    print(f"summary_json={artifacts.summary_json}")
    print(f"progress_json={artifacts.progress_json}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="entrypoints/data_pipeline/materialize_query_ssl_views",
)
def main(cfg: DictConfig) -> None:
    run_query_ssl_view_materialization_from_hydra(cfg=cfg)


if __name__ == "__main__":
    main()
