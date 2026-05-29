"""중앙 SSL teacher_bootstrap source adapter."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from methods.ssl.hooks.teacher import (
    PreparedTeacher,
    TeacherPrediction,
    TeacherPreparationContext,
    TeacherSource,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)


class FixedEmbeddingClassifierTeacherSource:
    """teacher_bootstrap input mode의 checkpoint artifact source."""

    source_kind = "checkpoint_artifact"
    artifact_kind = "fixed_embedding_classifier"

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def prepare(self, context: TeacherPreparationContext) -> PreparedTeacher:
        cfg = self._cfg
        teacher_reuse_manifest_path = str(
            getattr(cfg, "teacher_reuse_manifest_path", "") or ""
        ).strip()
        if teacher_reuse_manifest_path:
            trained_teacher, teacher_outputs = _load_fixed_classifier_artifacts(
                manifest_path=teacher_reuse_manifest_path,
                device=str(cfg.runtime.device),
                batch_size=int(cfg.teacher_eval_batch_size),
                cache_dir=str(getattr(cfg.embedding, "cache_dir", "") or "") or None,
                local_files_only=bool(getattr(cfg.runtime, "local_files_only", False)),
            )
            return PreparedTeacher(
                source_kind=self.source_kind,
                model=trained_teacher,
                categories=tuple(
                    str(category) for category in trained_teacher.categories
                ),
                outputs={
                    **teacher_outputs,
                    "teacher_bootstrap_source_kind": self.source_kind,
                    "teacher_artifact_kind": self.artifact_kind,
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
            or f"{context.run_id}_teacher"
        )

        trained_teacher = _train_fixed_embedding_classifier(
            train_rows=list(context.seed_rows),
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
        teacher_outputs = _write_fixed_classifier_artifacts(
            classifier_version=teacher_classifier_version,
            created_at=context.generated_at,
            train_jsonl_ref=context.seed_jsonl_ref,
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
        return PreparedTeacher(
            source_kind=self.source_kind,
            model=trained_teacher,
            categories=tuple(str(category) for category in trained_teacher.categories),
            outputs={
                **teacher_outputs,
                "teacher_bootstrap_source_kind": self.source_kind,
                "teacher_artifact_kind": self.artifact_kind,
            },
        )

    def predict_rows(
        self,
        *,
        teacher: PreparedTeacher,
        rows: Sequence[LabeledQueryRow],
    ) -> list[TeacherPrediction]:
        cfg = self._cfg
        return list(
            _predict_fixed_classifier_rows(
                trained=teacher.model,
                rows=rows,
                embed_chunk_size=int(cfg.teacher_embed_chunk_size),
                eval_batch_size=int(cfg.teacher_eval_batch_size),
            )
        )


def resolve_teacher_bootstrap_source(cfg: DictConfig) -> TeacherSource:
    """teacher_bootstrap input mode의 teacher source를 해석한다."""

    return FixedEmbeddingClassifierTeacherSource(cfg)


def _load_fixed_classifier_artifacts(**kwargs: Any) -> tuple[Any, dict[str, str]]:
    module = import_module(
        "scripts.support.query_ssl_peft.teacher_providers."
        "fixed_embedding_classifier.artifacts"
    )
    return module.load_fixed_classifier_artifacts(**kwargs)


def _write_fixed_classifier_artifacts(**kwargs: Any) -> dict[str, str]:
    module = import_module(
        "scripts.support.query_ssl_peft.teacher_providers."
        "fixed_embedding_classifier.artifacts"
    )
    return module.write_fixed_classifier_artifacts(**kwargs)


def _train_fixed_embedding_classifier(**kwargs: Any) -> Any:
    module = import_module(
        "scripts.support.query_ssl_peft.teacher_providers."
        "fixed_embedding_classifier.runner"
    )
    return module.train_fixed_embedding_classifier(**kwargs)


def _predict_fixed_classifier_rows(**kwargs: Any) -> Sequence[TeacherPrediction]:
    module = import_module(
        "scripts.support.query_ssl_peft.teacher_providers."
        "fixed_embedding_classifier.prediction"
    )
    return module.predict_fixed_classifier_rows(**kwargs)
