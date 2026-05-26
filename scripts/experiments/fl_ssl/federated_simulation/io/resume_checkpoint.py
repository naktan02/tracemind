"""FL SSL simulation round-level resume checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientRoundSummary,
    SimulationEvaluation,
    SimulationRoundSummary,
)

CHECKPOINT_SCHEMA_VERSION = "fl_ssl_round_resume.v1"
CHECKPOINT_PATH = Path("checkpoints") / "fl_ssl_round_resume.json"


@dataclass(frozen=True, slots=True)
class SimulationResumeCheckpoint:
    """완료 round까지만 재개 기준으로 삼는 checkpoint."""

    initial_model_revision: str
    initial_validation: SimulationEvaluation
    rounds: tuple[SimulationRoundSummary, ...]

    @property
    def completed_round_count(self) -> int:
        return len(self.rounds)


def resume_checkpoint_path(output_dir: Path) -> Path:
    """run output root 아래 resume checkpoint 경로를 반환한다."""

    return output_dir / CHECKPOINT_PATH


def write_resume_checkpoint(
    *,
    output_dir: Path,
    initial_model_revision: str,
    initial_validation: SimulationEvaluation,
    rounds: tuple[SimulationRoundSummary, ...],
) -> Path:
    """완료된 round summary를 atomic하게 저장한다."""

    path = resume_checkpoint_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "initial_model_revision": initial_model_revision,
        "initial_validation": asdict(initial_validation),
        "completed_round_count": len(rounds),
        "rounds": [asdict(round_summary) for round_summary in rounds],
    }
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path


def load_resume_checkpoint(output_dir: Path) -> SimulationResumeCheckpoint:
    """저장된 round-level checkpoint를 domain summary로 복원한다."""

    path = resume_checkpoint_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"FL SSL resume checkpoint not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported FL SSL resume checkpoint schema_version: {schema_version!r}."
        )
    rounds = tuple(
        _round_summary_from_payload(round_payload)
        for round_payload in payload.get("rounds", [])
    )
    completed_round_count = int(payload.get("completed_round_count", len(rounds)))
    if completed_round_count != len(rounds):
        raise ValueError(
            "FL SSL resume checkpoint completed_round_count does not match rounds."
        )
    return SimulationResumeCheckpoint(
        initial_model_revision=str(payload["initial_model_revision"]),
        initial_validation=_evaluation_from_payload(payload["initial_validation"]),
        rounds=rounds,
    )


def _round_summary_from_payload(payload: dict[str, object]) -> SimulationRoundSummary:
    clients = tuple(
        _client_summary_from_payload(client_payload)
        for client_payload in payload.get("clients", [])
    )
    return SimulationRoundSummary(
        round_id=str(payload["round_id"]),
        model_revision=str(payload["model_revision"]),
        update_count=int(payload["update_count"]),
        validation=_evaluation_from_payload(payload["validation"]),
        clients=clients,
        round_time_seconds=_optional_float(payload.get("round_time_seconds")),
        round_timing_breakdown={
            str(key): float(value)
            for key, value in dict(payload.get("round_timing_breakdown", {})).items()
        },
        aggregation_metrics={
            str(key): float(value)
            for key, value in dict(payload.get("aggregation_metrics", {})).items()
        },
        total_payload_bytes=_optional_int(payload.get("total_payload_bytes")),
        gpu_memory_peak_mb=_optional_float(payload.get("gpu_memory_peak_mb")),
        total_client_count=int(payload.get("total_client_count", 0)),
        selected_client_count=int(payload.get("selected_client_count", 0)),
        skipped_client_count=int(payload.get("skipped_client_count", 0)),
        skipped_client_ids=tuple(
            str(client_id) for client_id in payload.get("skipped_client_ids", [])
        ),
    )


def _evaluation_from_payload(payload: dict[str, object]) -> SimulationEvaluation:
    return SimulationEvaluation(
        row_count=int(payload["row_count"]),
        top1_accuracy=float(payload["top1_accuracy"]),
        accepted_ratio=float(payload["accepted_ratio"]),
        loss=float(payload.get("loss", 0.0)),
        loss_kind=str(payload.get("loss_kind", "not_computed")),
        accuracy_top_1=float(payload.get("accuracy_top_1", 0.0)),
        correct_top_1=int(payload.get("correct_top_1", 0)),
        macro_f1=float(payload.get("macro_f1", 0.0)),
        macro_precision=float(payload.get("macro_precision", 0.0)),
        macro_recall=float(payload.get("macro_recall", 0.0)),
        weighted_f1=float(payload.get("weighted_f1", 0.0)),
        balanced_accuracy=float(payload.get("balanced_accuracy", 0.0)),
        worst_category_f1=_optional_str(payload.get("worst_category_f1")),
        worst_category_f1_value=_optional_float(payload.get("worst_category_f1_value")),
        worst_category_recall=_optional_float(payload.get("worst_category_recall")),
        worst_category_precision=_optional_float(
            payload.get("worst_category_precision")
        ),
        expected_calibration_error=float(
            payload.get("expected_calibration_error", 0.0)
        ),
        max_calibration_error=float(payload.get("max_calibration_error", 0.0)),
        overconfidence_gap=float(payload.get("overconfidence_gap", 0.0)),
        mean_true_label_probability=float(
            payload.get("mean_true_label_probability", 0.0)
        ),
        mean_top_1_probability=float(payload.get("mean_top_1_probability", 0.0)),
        mean_margin_top1_top2=float(payload.get("mean_margin_top1_top2", 0.0)),
        mean_correct_top_1_probability=float(
            payload.get("mean_correct_top_1_probability", 0.0)
        ),
        mean_incorrect_top_1_probability=float(
            payload.get("mean_incorrect_top_1_probability", 0.0)
        ),
        score_distribution_kind=str(
            payload.get("score_distribution_kind", "not_computed")
        ),
        selection_confidence_kind=_optional_str(
            payload.get("selection_confidence_kind")
        ),
        mean_selection_confidence=float(payload.get("mean_selection_confidence", 0.0)),
        mean_selection_margin=float(payload.get("mean_selection_margin", 0.0)),
        per_label=dict(payload.get("per_label", {})),
        confusion_matrix=dict(payload.get("confusion_matrix", {})),
        classification_report=dict(payload.get("classification_report", {})),
    )


def _client_summary_from_payload(payload: dict[str, object]) -> ClientRoundSummary:
    return ClientRoundSummary(
        client_id=str(payload["client_id"]),
        candidate_count=int(payload["candidate_count"]),
        accepted_count=int(payload["accepted_count"]),
        update_generated=bool(payload["update_generated"]),
        diagnostic_candidate_count=int(payload.get("diagnostic_candidate_count", 0)),
        delta_l2_norm=_optional_float(payload.get("delta_l2_norm")),
        aggregation_example_count=_optional_int(
            payload.get("aggregation_example_count")
        ),
        client_train_time_seconds=_optional_float(
            payload.get("client_train_time_seconds")
        ),
        client_payload_bytes=_optional_int(payload.get("client_payload_bytes")),
        client_artifact_bytes=_optional_int(payload.get("client_artifact_bytes")),
        pseudo_label_confidence_mean=_optional_float(
            payload.get("pseudo_label_confidence_mean")
        ),
        pseudo_label_margin_mean=_optional_float(
            payload.get("pseudo_label_margin_mean")
        ),
        pseudo_label_correct_count=int(payload.get("pseudo_label_correct_count", 0)),
        pseudo_label_evaluated_count=int(
            payload.get("pseudo_label_evaluated_count", 0)
        ),
        accepted_label_distribution={
            str(key): int(value)
            for key, value in dict(
                payload.get("accepted_label_distribution", {})
            ).items()
        },
        rejected_label_distribution={
            str(key): int(value)
            for key, value in dict(
                payload.get("rejected_label_distribution", {})
            ).items()
        },
        fedmatch_helper_count=_optional_float(payload.get("fedmatch_helper_count")),
        fedmatch_peer_context_helper_count=_optional_float(
            payload.get("fedmatch_peer_context_helper_count")
        ),
        fedmatch_peer_context_refreshed=_optional_float(
            payload.get("fedmatch_peer_context_refreshed")
        ),
        fedmatch_c2s_sparse_upload_value_count=_optional_float(
            payload.get("fedmatch_c2s_sparse_upload_value_count")
        ),
        fedmatch_s2c_sparse_download_value_count=_optional_float(
            payload.get("fedmatch_s2c_sparse_download_value_count")
        ),
        timing_breakdown={
            str(key): float(value)
            for key, value in dict(payload.get("timing_breakdown", {})).items()
        },
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
