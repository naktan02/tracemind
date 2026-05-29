"""Teacher bootstrap pseudo-label materialization runner."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig

from methods.ssl.hooks.teacher import TeacherPreparationContext
from scripts.support.query_ssl_peft.config.pseudo_label_algorithm import (
    resolve_pseudo_label_algorithm,
)
from scripts.support.query_ssl_peft.io.teacher_pseudo_label_artifact_writer import (
    TeacherPseudoLabelArtifactWriter,
)
from scripts.support.query_ssl_peft.io.teacher_pseudo_label_builder import (
    TeacherPseudoLabelBuilder,
)
from scripts.support.query_ssl_peft.runners.pseudo_label import (
    run_pseudo_label_self_training,
)
from scripts.support.query_ssl_peft.runners.teacher_source import (
    resolve_teacher_bootstrap_source,
)
from scripts.support.query_ssl_peft.runners.teacher_split import (
    resolve_teacher_and_unlabeled_rows,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

BOOTSTRAP_SUMMARY_SCHEMA_VERSION = "teacher_bootstrap.v1"


def run_teacher_bootstrap_peft_student(
    *,
    cfg: DictConfig,
    teacher_seed_rows: Sequence[LabeledQueryRow] | None = None,
    teacher_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    export_root: str | Path | None = None,
    generated_at: datetime | None = None,
) -> dict[str, str]:
    """초기 teacher source와 PEFT text encoder student를 연결한다."""

    effective_generated_at = generated_at or datetime.now(tz=timezone.utc)
    run_id = _resolve_run_id(cfg=cfg, generated_at=effective_generated_at)
    resolved_export_root = (
        Path(str(cfg.bootstrap_export_root))
        if export_root is None
        else Path(str(export_root))
    )
    export_dir = resolved_export_root / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    teacher_rows = resolve_teacher_and_unlabeled_rows(
        cfg=cfg,
        run_id=run_id,
        export_dir=export_dir,
        teacher_seed_rows=teacher_seed_rows,
        teacher_unlabeled_rows=teacher_unlabeled_rows,
    )

    teacher_source = resolve_teacher_bootstrap_source(cfg)
    teacher = teacher_source.prepare(
        TeacherPreparationContext(
            run_id=run_id,
            generated_at=effective_generated_at,
            seed_rows=tuple(teacher_rows.seed_rows),
            seed_jsonl_ref=teacher_rows.seed_jsonl_ref,
        )
    )

    predictions = teacher_source.predict_rows(
        teacher=teacher,
        rows=teacher_rows.unlabeled_rows,
    )
    teacher_export = TeacherPseudoLabelBuilder().build_export(
        rows=teacher_rows.unlabeled_rows,
        predictions=predictions,
        pseudo_label_algorithm=resolve_pseudo_label_algorithm(cfg),
        generated_at=effective_generated_at,
        run_id=run_id,
    )
    pseudo_label_rows = teacher_export.pseudo_label_rows
    if not pseudo_label_rows:
        raise ValueError(
            "Teacher bootstrap did not accept any pseudo-label rows. "
            "Lower the threshold or provide a different unlabeled pool."
        )

    teacher_artifact_writer = TeacherPseudoLabelArtifactWriter()
    prediction_artifacts = teacher_artifact_writer.write_prediction_artifacts(
        export_dir=export_dir,
        prediction_trace_rows=teacher_export.prediction_trace_rows,
        prediction_summary=teacher_export.prediction_summary,
    )

    student_outputs = _run_student_peft_bootstrap(
        cfg=cfg,
        run_id=run_id,
        teacher_seed_jsonl_ref=teacher_rows.seed_jsonl_ref,
        seed_train_rows=teacher_rows.seed_rows,
        pseudo_label_rows=pseudo_label_rows,
        generated_at=effective_generated_at,
        categories_override=teacher.categories,
    )

    bootstrap_summary = {
        "schema_version": BOOTSTRAP_SUMMARY_SCHEMA_VERSION,
        "bootstrap_version": run_id,
        "generated_at": effective_generated_at.isoformat(),
        "teacher_train_jsonl": teacher_rows.seed_jsonl_ref,
        "teacher_unlabeled_jsonl": teacher_rows.unlabeled_jsonl_ref,
        "teacher_seed_row_count": len(teacher_rows.seed_rows),
        "teacher_unlabeled_row_count": len(teacher_rows.unlabeled_rows),
        "teacher_bootstrap_source_kind": teacher.source_kind,
        "teacher_artifact_kind": teacher.outputs.get(
            "teacher_artifact_kind",
            "fixed_embedding_classifier",
        ),
        "teacher_outputs": teacher.outputs,
        "prediction_trace_jsonl": str(prediction_artifacts.prediction_trace_jsonl),
        "prediction_summary_json": str(prediction_artifacts.prediction_summary_json),
        "student_outputs": student_outputs,
        "split_manifest": (
            None
            if teacher_rows.split_artifacts is None
            else str(teacher_rows.split_artifacts.manifest_path)
        ),
    }
    bootstrap_summary_path = teacher_artifact_writer.write_bootstrap_summary(
        export_dir=export_dir,
        bootstrap_summary=bootstrap_summary,
    )

    outputs = {
        "bootstrap_summary": str(bootstrap_summary_path),
        "prediction_trace_jsonl": str(prediction_artifacts.prediction_trace_jsonl),
        "prediction_summary_json": str(prediction_artifacts.prediction_summary_json),
        **teacher.outputs,
        **student_outputs,
    }
    if teacher_rows.split_artifacts is not None:
        outputs.update(
            {
                "teacher_seed_jsonl": str(
                    teacher_rows.split_artifacts.seed_train_jsonl
                ),
                "teacher_unlabeled_jsonl": str(
                    teacher_rows.split_artifacts.unlabeled_jsonl
                ),
                "teacher_split_manifest": str(
                    teacher_rows.split_artifacts.manifest_path
                ),
            }
        )
    return outputs


def _resolve_run_id(
    *,
    cfg: DictConfig,
    generated_at: datetime,
) -> str:
    bootstrap_version = str(getattr(cfg, "bootstrap_version", "") or "").strip()
    if bootstrap_version:
        return bootstrap_version
    return generated_at.strftime("peft_bootstrap_%Y_%m_%d_%H%M%S")


def _run_student_peft_bootstrap(
    *,
    cfg: DictConfig,
    run_id: str,
    teacher_seed_jsonl_ref: str,
    seed_train_rows: Sequence[LabeledQueryRow],
    pseudo_label_rows: Sequence[LabeledQueryRow],
    generated_at: datetime,
    categories_override: Sequence[str],
) -> dict[str, str]:
    return run_pseudo_label_self_training(
        cfg=cfg,
        seed_train_rows=seed_train_rows,
        pseudo_label_rows=pseudo_label_rows,
        include_seed_train_rows=bool(
            getattr(cfg, "student_include_seed_train_rows", False)
        ),
        train_jsonl_ref=teacher_seed_jsonl_ref,
        trainer_version_override=_resolve_student_trainer_version(cfg, run_id),
        export_root=str(cfg.pseudo_label_export_root),
        generated_at=generated_at,
        categories_override=categories_override,
    )


def _resolve_student_trainer_version(cfg: DictConfig, run_id: str) -> str:
    trainer_version = str(getattr(cfg, "trainer_version", "") or "").strip()
    if trainer_version:
        return trainer_version
    return f"{run_id}_student"
