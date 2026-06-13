"""Query SSL update upload flow."""

from __future__ import annotations

from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.training_runtime.query_ssl_peft.local_training_service import (
    QuerySslPeftEncoderClientTrainingResult,
)
from shared.src.contracts.training_contracts import (
    TrainingTask,
    make_training_update_submission,
)


def upload_query_ssl_update(
    *,
    round_client: RoundClient,
    training_task: TrainingTask,
    local_result: QuerySslPeftEncoderClientTrainingResult,
) -> None:
    """local update result를 server update submission으로 업로드한다."""

    round_client.upload_update(
        training_task.round_id,
        make_training_update_submission(
            envelope=local_result.update_envelope,
            update_payload=local_result.update_payload,
        ),
    )
