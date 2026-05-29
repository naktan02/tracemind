"""Fixed embedding classifier 실행 준비 helper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


@dataclass(slots=True)
class FixedClassifierRunContext:
    """fixed embedding classifier 실행에 필요한 정규화된 입력."""

    cfg: DictConfig
    train_rows: list[LabeledQueryRow]
    eval_rows_by_name: dict[str, list[LabeledQueryRow]]
    eval_set_map: dict[str, Path]
    effective_selection_set: str
    effective_train_jsonl_ref: str
    output_dir_root: str
    model_output_dir: str
    classifier_version: str
    created_at: datetime
    embedding_spec: Any


def prepare_fixed_classifier_run_context(
    *,
    cfg: DictConfig,
    train_rows: list[LabeledQueryRow] | None = None,
    eval_rows_by_name: dict[str, list[LabeledQueryRow]] | None = None,
    train_jsonl_ref: str | None = None,
    output_dir_root: str | None = None,
    model_output_dir: str | None = None,
    classifier_version: str | None = None,
) -> FixedClassifierRunContext:
    """Hydra config와 override를 실행 컨텍스트로 정규화한다."""

    eval_set_map = {name: Path(str(path)) for name, path in cfg.eval_sets.items()}
    effective_selection_set = str(cfg.selection_set)
    if effective_selection_set not in (
        eval_set_map if eval_rows_by_name is None else eval_rows_by_name
    ):
        raise ValueError(
            f"selection_set '{effective_selection_set}' is not included in eval_sets."
        )

    effective_train_rows = (
        load_labeled_query_rows(Path(str(cfg.train_jsonl)))
        if train_rows is None
        else list(train_rows)
    )
    effective_eval_rows = (
        {name: load_labeled_query_rows(path) for name, path in eval_set_map.items()}
        if eval_rows_by_name is None
        else {name: list(rows) for name, rows in eval_rows_by_name.items()}
    )
    created_at = datetime.now(timezone.utc)
    effective_classifier_version = classifier_version or (
        cfg.classifier_version or created_at.strftime("clf_%Y_%m_%d_%H%M%S")
    )

    return FixedClassifierRunContext(
        cfg=cfg,
        train_rows=effective_train_rows,
        eval_rows_by_name=effective_eval_rows,
        eval_set_map=eval_set_map,
        effective_selection_set=effective_selection_set,
        effective_train_jsonl_ref=str(train_jsonl_ref or cfg.train_jsonl),
        output_dir_root=str(output_dir_root or cfg.output_dir),
        model_output_dir=str(model_output_dir or cfg.model_output_dir),
        classifier_version=str(effective_classifier_version),
        created_at=created_at,
        embedding_spec=instantiate(cfg.embedding.spec),
    )
