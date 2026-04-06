"""FL round 관련 agent↔server 간 교환 계약.

이 모듈은 agent가 서버 FL round API를 소비할 때 필요한 최소 payload를 정의한다.
서버 내부 구현(main_server.payloads)을 직접 참조하면 경계 위반이므로,
agent가 필요로 하는 필드만 여기에 선언한다.

agent가 round API에서 읽는 것:
- status: round가 open인지 확인
- training_task: 이번 라운드 학습 지시문
- round_id: update 업로드 경로에 사용
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from shared.src.contracts.training_contracts import TrainingTaskPayload


class ActiveRoundPayload(BaseModel):
    """agent가 서버 /fl/rounds/current 응답에서 필요한 필드.

    서버가 반환하는 전체 RoundRecordPayload 중 agent가 실제로 사용하는 것만 선언한다.
    extra 필드는 무시(ignore)해서 서버 응답에 새 필드가 추가돼도 깨지지 않는다.
    """

    model_config = ConfigDict(extra="ignore")

    round_id: str = Field(description="FL round 고유 식별자.")
    status: str = Field(description="round 상태. 'open' 또는 'finalized'.")
    training_task: TrainingTaskPayload = Field(
        description="이번 라운드에서 agent가 수행할 학습 지시문."
    )
    created_at: datetime = Field(description="round 생성 시각.")
    updated_at: datetime = Field(description="round 마지막 업데이트 시각.")

    @property
    def is_open(self) -> bool:
        """round가 현재 참여 가능한 상태인지 확인한다."""
        return self.status == "open"
