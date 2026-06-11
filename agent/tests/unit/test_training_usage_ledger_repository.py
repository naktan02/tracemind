"""Training usage ledger 저장소 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.src.infrastructure.repositories.local_agent_database import (
    DEFAULT_AGENT_LOCAL_DB_PATH,
)
from agent.src.infrastructure.repositories.training_usage_ledger_repository import (
    TRAINING_USAGE_ROLE_LABELED_ANCHOR,
    TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
    TRAINING_USAGE_STATUS_UPLOADED,
    TrainingUsageLedgerRepository,
    TrainingUsageRowRecord,
    TrainingUsageRunRecord,
)


def test_training_usage_ledger_round_trips_run_and_rows(tmp_path) -> None:
    repository = TrainingUsageLedgerRepository(db_path=tmp_path / "usage.db")
    recorded_at = datetime(2026, 6, 8, 1, 0, tzinfo=timezone.utc)

    repository.save_run(
        TrainingUsageRunRecord(
            update_id="update_001",
            round_id="round_001",
            task_id="task_001",
            recorded_at=recorded_at,
            agent_id="agent_01",
            model_id="model",
            model_revision="rev_001",
            objective_method_name="fixmatch_usb_v1",
            objective_algorithm_name="fixmatch",
            status=TRAINING_USAGE_STATUS_UPLOADED,
            candidate_count=3,
            accepted_count=2,
            metadata={"max_steps": 10},
        ),
        rows=(
            TrainingUsageRowRecord(
                update_id="update_001",
                source_id="query_001",
                role=TRAINING_USAGE_ROLE_LABELED_ANCHOR,
                round_id="round_001",
                task_id="task_001",
                recorded_at=recorded_at,
                source_kind="analysis_event",
                stage=TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
                label="anxiety",
                metadata={"raw_label_scheme": "agent_local_pseudo_label"},
            ),
        ),
    )

    run = repository.get_run("update_001")
    rows = repository.get_rows_for_update("update_001")
    source_rows = repository.get_rows_for_source(
        source_kind="analysis_event",
        source_id="query_001",
    )

    assert run is not None
    assert run.round_id == "round_001"
    assert run.candidate_count == 3
    assert run.accepted_count == 2
    assert run.objective_algorithm_name == "fixmatch"
    assert len(rows) == 1
    assert rows[0].source_id == "query_001"
    assert rows[0].label == "anxiety"
    assert source_rows == rows
    assert repository.count_rows() == 1


def test_training_usage_ledger_default_uses_agent_local_db() -> None:
    repository = TrainingUsageLedgerRepository()

    assert repository.db_path == DEFAULT_AGENT_LOCAL_DB_PATH
