"""active FL strategy config 저장소.

ModelManifestRepository와 동일한 패턴으로 파일 기반 저장을 제공한다.

디렉터리 구조:
    state/active_strategy/
    ├── active.json          ← 현재 active strategy
    └── history/
        └── <iso-timestamp>.json  ← 전환 이력
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from main_server.src.services.federation.strategy.models import (
    ActiveStrategyConfig,
    dump_active_strategy_config,
    load_active_strategy_config,
    make_default_active_strategy_config,
)

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class ActiveStrategyRepository:
    """active strategy config와 전환 이력을 파일로 관리한다."""

    state_root: Path = field(
        default_factory=lambda: MAIN_SERVER_ROOT / "state" / "active_strategy"
    )

    @property
    def active_path(self) -> Path:
        return self.state_root / "active.json"

    @property
    def history_dir(self) -> Path:
        return self.state_root / "history"

    def _history_path(self, activated_at: datetime) -> Path:
        # 파일명에 콜론이 들어가지 않도록 ISO 형식을 정규화한다
        ts = activated_at.isoformat().replace(":", "-").replace("+", "Z")
        return self.history_dir / f"{ts}.json"

    def load_active(self) -> ActiveStrategyConfig:
        """현재 active strategy를 반환한다. 파일이 없으면 기본값을 반환한다."""
        if not self.active_path.exists():
            return make_default_active_strategy_config()
        return load_active_strategy_config(self.active_path)

    def save_active(self, config: ActiveStrategyConfig) -> None:
        """strategy config를 active.json과 history에 저장한다."""
        dump_active_strategy_config(self.active_path, config)
        history_path = self._history_path(config.activated_at)
        dump_active_strategy_config(history_path, config)

    def load_history(self) -> tuple[ActiveStrategyConfig, ...]:
        """전환 이력을 시간 역순으로 반환한다."""
        if not self.history_dir.exists():
            return ()
        paths = sorted(self.history_dir.glob("*.json"), reverse=True)
        configs = []
        for path in paths:
            try:
                configs.append(load_active_strategy_config(path))
            except Exception:
                # 손상된 이력 파일은 건너뛴다
                continue
        return tuple(configs)
