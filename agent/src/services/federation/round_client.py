"""서버 FL round API와 통신한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from shared.src.contracts.fl_round_contracts import ActiveRoundPayload
from shared.src.contracts.training_contracts import (
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)


@dataclass(slots=True)
class RoundClient:
    """중앙 서버 FL round API 클라이언트.

    현재 round/task를 가져오고 update를 업로드한다.
    인증이 없는 단순 HTTP 클라이언트다.
    """

    server_base_url: str

    def _url(self, path: str) -> str:
        return urljoin(self.server_base_url.rstrip("/") + "/", path.lstrip("/"))

    def _get_json(self, path: str) -> dict:
        url = self._url(path)
        with urlopen(url) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, body: dict) -> dict:
        url = self._url(path)
        data = json.dumps(body).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_current_round(self) -> ActiveRoundPayload | None:
        """현재 active round를 가져온다. round가 없으면 None을 반환한다."""
        try:
            raw = self._get_json("api/v1/fl/rounds/current")
            return ActiveRoundPayload.model_validate(raw)
        except Exception as error:  # noqa: BLE001
            # 404(active round 없음) 포함 모든 실패를 None으로 처리한다.
            if _is_http_not_found(error):
                return None
            raise

    def fetch_current_task(self) -> TrainingTaskPayload | None:
        """현재 active round의 training task를 가져온다."""
        record = self.fetch_current_round()
        if record is None or not record.is_open:
            return None
        return record.training_task

    def upload_update(
        self,
        round_id: str,
        envelope: TrainingUpdateEnvelopePayload,
    ) -> dict:
        """update envelope을 서버에 업로드하고 수락 응답을 반환한다."""
        path = f"api/v1/fl/rounds/{round_id}/updates"
        return self._post_json(path, envelope.model_dump(mode="json"))


def _is_http_not_found(error: Exception) -> bool:
    """urllib HTTPError 404 여부를 확인한다."""
    from urllib.error import HTTPError

    return isinstance(error, HTTPError) and error.code == 404
