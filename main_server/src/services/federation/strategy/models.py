"""main_server 소유 active FL strategy 모델."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ACTIVE_STRATEGY_CONFIG_V1 = "active_strategy_config.v1"
DEFAULT_SSL_METHOD = "fixmatch_usb_v1"
DEFAULT_AGGREGATION_BACKEND = "fedavg"


@dataclass(frozen=True, slots=True)
class ActiveStrategyConfig:
    """다음 round open 시 적용할 서버 운영 strategy 포인터."""

    schema_version: str
    ssl_method: str | None
    aggregation_backend: str
    activated_at: datetime
    fssl_method: str | None = None
    notes: str | None = None

    def to_json(self) -> str:
        data = {
            "schema_version": self.schema_version,
            "ssl_method": self.ssl_method,
            "fssl_method": self.fssl_method,
            "aggregation_backend": self.aggregation_backend,
            "activated_at": self.activated_at.isoformat(),
            "notes": self.notes,
        }
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> "ActiveStrategyConfig":
        data = json.loads(raw)
        return cls(
            schema_version=data["schema_version"],
            ssl_method=data["ssl_method"],
            aggregation_backend=data["aggregation_backend"],
            activated_at=datetime.fromisoformat(data["activated_at"]),
            fssl_method=data.get("fssl_method"),
            notes=data.get("notes"),
        )


def make_default_active_strategy_config() -> ActiveStrategyConfig:
    """기본 active strategy config를 만든다."""
    return ActiveStrategyConfig(
        schema_version=ACTIVE_STRATEGY_CONFIG_V1,
        ssl_method=DEFAULT_SSL_METHOD,
        aggregation_backend=DEFAULT_AGGREGATION_BACKEND,
        activated_at=datetime.now(tz=timezone.utc),
    )


def dump_active_strategy_config(path: Path, config: ActiveStrategyConfig) -> None:
    """ActiveStrategyConfig를 JSON 파일로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_json(), encoding="utf-8")


def load_active_strategy_config(path: Path) -> ActiveStrategyConfig:
    """JSON 파일에서 ActiveStrategyConfig를 읽는다."""
    return ActiveStrategyConfig.from_json(path.read_text(encoding="utf-8"))
