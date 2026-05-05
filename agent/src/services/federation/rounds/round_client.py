"""서버 FL round API와 통신한다."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

import httpx

from shared.src.contracts.adapter_contracts import CurrentSharedAdapterStatePayload
from shared.src.contracts.fl_round_contracts import ActiveRoundPayload
from shared.src.contracts.training_contracts import (
    TrainingTaskPayload,
    TrainingUpdateSubmissionPayload,
)

RoundHttpClientFactory = Callable[[], AbstractContextManager[httpx.Client]]


@dataclass(slots=True)
class RoundClient:
    """중앙 서버 FL round API 클라이언트.

    현재 round/task를 가져오고 update를 업로드한다.
    인증이 없는 단순 HTTP 클라이언트다.

    httpx를 사용하므로 테스트에서 ASGITransport로 in-process 교체가 가능하다.
    """

    server_base_url: str
    timeout: float = 10.0
    # 외부에서 transport를 주입하면 ASGITransport 등으로 교체 가능하다.
    _transport: httpx.BaseTransport | None = field(default=None, repr=False)
    # 테스트에서 기존 client를 빌려 쓰고 싶을 때 context factory를 주입한다.
    _client_factory: RoundHttpClientFactory | None = field(
        default=None,
        repr=False,
    )

    def _client(self) -> AbstractContextManager[httpx.Client]:
        if self._client_factory is not None:
            return self._client_factory()
        kwargs: dict = {"base_url": self.server_base_url, "timeout": self.timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def fetch_current_round(self) -> ActiveRoundPayload | None:
        """현재 active round를 가져온다. round가 없으면 None을 반환한다."""
        with self._client() as client:
            response = client.get("/api/v1/fl/rounds/current")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return ActiveRoundPayload.model_validate(response.json())

    def fetch_current_task(self) -> TrainingTaskPayload | None:
        """현재 active round의 training task를 가져온다."""
        record = self.fetch_current_round()
        if record is None or not record.is_open:
            return None
        return record.training_task

    def fetch_current_shared_adapter_state(
        self,
    ) -> CurrentSharedAdapterStatePayload | None:
        """서버 current shared adapter state를 가져온다."""
        with self._client() as client:
            response = client.get("/api/v1/fl/rounds/active-state/current")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return CurrentSharedAdapterStatePayload.model_validate(response.json())

    def upload_update(
        self,
        round_id: str,
        submission: TrainingUpdateSubmissionPayload,
    ) -> dict:
        """update submission을 서버에 업로드하고 수락 응답을 반환한다."""
        with self._client() as client:
            response = client.post(
                f"/api/v1/fl/rounds/{round_id}/updates",
                json=submission.model_dump(mode="json"),
            )
            response.raise_for_status()
            return response.json()
