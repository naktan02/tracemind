"""active FL strategy 전환 서비스.

ActiveModelManifestService와 동일한 패턴으로 방법론 전환을 관리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from main_server.src.infrastructure.repositories.active_strategy_repository import (
    ActiveStrategyRepository,
)
from main_server.src.services.federation.strategy.models import (
    ACTIVE_STRATEGY_CONFIG_V1,
    ActiveStrategyConfig,
)
from shared.src.domain.services.clock import Clock, SystemUtcClock


class StrategyValidationError(ValueError):
    """strategy 전환 요청이 현재 live runtime에서 지원되지 않을 때 발생한다."""


def _build_active_strategy_repository() -> ActiveStrategyRepository:
    return ActiveStrategyRepository()


@dataclass(slots=True)
class ActiveStrategyService:
    """운영 중 FL strategy 전환을 관리한다.

    Phase 1에서는 live composed SSL method 전환만 소유한다.
    """

    repository: ActiveStrategyRepository = field(
        default_factory=_build_active_strategy_repository
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def get_active_strategy(self) -> ActiveStrategyConfig:
        """현재 active strategy를 반환한다. 설정 파일이 없으면 기본값을 반환한다."""
        return self.repository.load_active()

    def switch(
        self,
        *,
        ssl_method: str | None = None,
        aggregation_backend: str | None = None,
        notes: str | None = None,
    ) -> ActiveStrategyConfig:
        """strategy를 전환하고 이력을 기록한다.

        ssl_method는 runtime_fallbacks에 등록된 live SSL method여야 한다.
        """
        current = self.repository.load_active()

        effective_ssl_method = ssl_method or current.ssl_method
        effective_backend = aggregation_backend or current.aggregation_backend

        self._validate_ssl_method(effective_ssl_method)
        self._validate_aggregation_backend(effective_backend)

        new_config = ActiveStrategyConfig(
            schema_version=ACTIVE_STRATEGY_CONFIG_V1,
            ssl_method=effective_ssl_method,
            aggregation_backend=effective_backend,
            activated_at=self.clock.now(),
            notes=notes,
        )
        self.repository.save_active(new_config)
        return new_config

    def get_history(self) -> tuple[ActiveStrategyConfig, ...]:
        """전환 이력을 시간 역순으로 반환한다."""
        return self.repository.load_history()

    def _validate_ssl_method(self, ssl_method: str) -> None:
        """ssl_method가 runtime_fallbacks에 등록된 method인지 확인한다."""
        from methods.federated_ssl.runtime_fallbacks import (
            QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS,
        )

        if ssl_method not in QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS:
            supported = list(QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS.keys())
            raise StrategyValidationError(
                f"지원되지 않는 ssl_method입니다: {ssl_method!r}. "
                f"지원 목록: {supported}"
            )

    def _validate_aggregation_backend(self, aggregation_backend: str) -> None:
        """Phase 1 live strategy가 지원하는 aggregation backend인지 확인한다."""
        from methods.federated_ssl.runtime_fallbacks import (
            FEDAVG_AGGREGATION_BACKEND_NAME,
        )

        if aggregation_backend != FEDAVG_AGGREGATION_BACKEND_NAME:
            raise StrategyValidationError(
                "지원되지 않는 aggregation_backend입니다: "
                f"{aggregation_backend!r}. 지원 목록: "
                f"{[FEDAVG_AGGREGATION_BACKEND_NAME]}"
            )
