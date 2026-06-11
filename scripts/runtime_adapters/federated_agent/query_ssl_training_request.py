"""FL simulation Query SSL training request bridge."""

from __future__ import annotations

from agent.src.services.training_runtime.query_ssl_peft.local_training_service import (
    QuerySslLocalTrainingService,
    QuerySslPeftEncoderLocalTrainingRequest,
)
from agent.src.services.training_runtime.query_ssl_peft.local_training_service import (
    run_query_ssl_peft_encoder_local_training as run_agent_query_ssl_training,
)


def build_query_ssl_peft_encoder_local_training_request(
    **kwargs: object,
) -> QuerySslPeftEncoderLocalTrainingRequest:
    """simulation client 입력을 agent-local training request로 변환한다."""

    return QuerySslPeftEncoderLocalTrainingRequest(**kwargs)


def run_query_ssl_peft_encoder_local_training(
    *,
    local_training_service: QuerySslLocalTrainingService,
    request: QuerySslPeftEncoderLocalTrainingRequest,
):
    """simulation bridge가 agent local training service를 실행하는 entrypoint."""

    return run_agent_query_ssl_training(
        local_training_service=local_training_service,
        request=request,
    )
