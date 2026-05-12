"""Hydra config 기반 중앙 Query SSL labeled/unlabeled split materialization."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts.datasets.lib.query_ssl_split import (
    materialize_class_balanced_query_ssl_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_query_ssl_split_materialization_from_hydra(
    *,
    cfg: DictConfig,
) -> None:
    """Hydra 설정을 Query SSL split materializer 실행 파라미터로 변환한다."""

    split_cfg = cfg.query_ssl_split_materialization
    artifacts = materialize_class_balanced_query_ssl_split(
        source_train_jsonl=_resolve_project_path(str(split_cfg.source_train_jsonl)),
        source_validation_jsonl=_resolve_project_path(
            str(split_cfg.source_validation_jsonl)
        ),
        source_test_jsonl=_resolve_project_path(str(split_cfg.source_test_jsonl)),
        split_name=str(split_cfg.name),
        labeled_count_per_class=int(split_cfg.labeled_count_per_class),
        seed=int(split_cfg.seed),
        output_root=_resolve_project_path(str(split_cfg.output_root)),
    )

    print(f"labeled_train_jsonl={artifacts.labeled_train_jsonl}")
    print(f"unlabeled_pool_jsonl={artifacts.unlabeled_pool_jsonl}")
    print(f"validation_jsonl={artifacts.validation_jsonl}")
    print(f"test_jsonl={artifacts.test_jsonl}")
    print(f"manifest_json={artifacts.manifest_json}")
    print(f"summary_json={artifacts.summary_json}")


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="entrypoints/dataset_pipeline/materialize_query_ssl_split",
)
def main(cfg: DictConfig) -> None:
    run_query_ssl_split_materialization_from_hydra(cfg=cfg)


if __name__ == "__main__":
    main()
