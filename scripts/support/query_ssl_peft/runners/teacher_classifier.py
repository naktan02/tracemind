"""Teacher fixed-classifier 준비 helper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from scripts.experiments.central.fixed_classifier_seed.artifacts import (
    load_fixed_classifier_artifacts,
    write_fixed_classifier_artifacts,
)
from scripts.experiments.central.fixed_classifier_seed.runner import (
    train_fixed_embedding_classifier,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


@dataclass(slots=True)
class ResolvedTeacherClassifier:
    """bootstrap runner가 사용할 teacher classifier와 artifact 출력."""

    trained: Any
    outputs: dict[str, str]


def resolve_teacher_classifier(
    *,
    cfg: DictConfig,
    run_id: str,
    generated_at: datetime,
    seed_rows: list[LabeledQueryRow],
    seed_jsonl_ref: str,
) -> ResolvedTeacherClassifier:
    """teacher artifact를 재사용하거나 fixed classifier를 학습해 준비한다."""

    teacher_reuse_manifest_path = str(
        getattr(cfg, "teacher_reuse_manifest_path", "") or ""
    ).strip()
    if teacher_reuse_manifest_path:
        trained_teacher, teacher_outputs = load_fixed_classifier_artifacts(
            manifest_path=teacher_reuse_manifest_path,
            device=str(cfg.runtime.device),
            batch_size=int(cfg.teacher_eval_batch_size),
            cache_dir=str(getattr(cfg.embedding, "cache_dir", "") or "") or None,
            local_files_only=bool(getattr(cfg.runtime, "local_files_only", False)),
        )
        return ResolvedTeacherClassifier(
            trained=trained_teacher,
            outputs={
                **teacher_outputs,
                "reused_teacher_manifest": teacher_reuse_manifest_path,
            },
        )

    eval_set_map = {name: Path(str(path)) for name, path in cfg.eval_sets.items()}
    eval_rows_by_name = {
        name: load_labeled_query_rows(path) for name, path in eval_set_map.items()
    }
    embedding_spec = instantiate(cfg.embedding.spec)
    teacher_classifier_version = (
        str(getattr(cfg, "teacher_classifier_version", "") or "").strip()
        or f"{run_id}_teacher"
    )

    trained_teacher = train_fixed_embedding_classifier(
        train_rows=seed_rows,
        eval_rows_by_name=eval_rows_by_name,
        selection_set_name=str(cfg.selection_set),
        embedding_spec=embedding_spec,
        embed_chunk_size=int(cfg.teacher_embed_chunk_size),
        train_batch_size=int(cfg.teacher_train_batch_size),
        eval_batch_size=int(cfg.teacher_eval_batch_size),
        epochs=int(cfg.teacher_epochs),
        learning_rate=float(cfg.teacher_learning_rate),
        weight_decay=float(cfg.teacher_weight_decay),
    )
    teacher_outputs = write_fixed_classifier_artifacts(
        classifier_version=teacher_classifier_version,
        created_at=generated_at,
        train_jsonl_ref=seed_jsonl_ref,
        eval_set_map={name: str(path) for name, path in eval_set_map.items()},
        selection_set_name=str(cfg.selection_set),
        output_dir_root=str(cfg.teacher_output_dir),
        model_output_dir=str(cfg.teacher_model_output_dir),
        epochs=int(cfg.teacher_epochs),
        train_batch_size=int(cfg.teacher_train_batch_size),
        learning_rate=float(cfg.teacher_learning_rate),
        weight_decay=float(cfg.teacher_weight_decay),
        trained=trained_teacher,
    )
    return ResolvedTeacherClassifier(trained=trained_teacher, outputs=teacher_outputs)
