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
    DEFAULT_SSL_METHOD,
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

    composed SSL은 ssl_method를, method-owned FL SSL은 fssl_method를 active
    pointer로 저장한다. fssl_method가 설정된 동안 ssl_method는 사용자 선택값으로
    저장하지 않는다.
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
        fssl_method: str | None = None,
        aggregation_backend: str | None = None,
        notes: str | None = None,
    ) -> ActiveStrategyConfig:
        """strategy를 전환하고 이력을 기록한다.

        ssl_method는 composed mode에서만 runtime_fallbacks에 등록된 live SSL
        method여야 한다.
        """
        current = self.repository.load_active()

        effective_fssl_method = (
            fssl_method if fssl_method is not None else current.fssl_method
        )
        if fssl_method == "":
            effective_fssl_method = None
        if effective_fssl_method is not None and ssl_method is not None:
            raise StrategyValidationError(
                "fssl_method 모드에서는 ssl_method를 함께 지정하지 않습니다. "
                "composed SSL로 돌아가려면 fssl_method=''와 ssl_method를 보내세요."
            )
        effective_ssl_method = (
            None
            if effective_fssl_method is not None
            else (ssl_method or current.ssl_method or DEFAULT_SSL_METHOD)
        )
        effective_backend = aggregation_backend or current.aggregation_backend

        if effective_ssl_method is not None:
            self._validate_ssl_method(effective_ssl_method)
        if effective_fssl_method is not None:
            self._validate_fssl_method(effective_fssl_method)
        self._validate_aggregation_backend(effective_backend)

        new_config = ActiveStrategyConfig(
            schema_version=ACTIVE_STRATEGY_CONFIG_V1,
            ssl_method=effective_ssl_method,
            aggregation_backend=effective_backend,
            activated_at=self.clock.now(),
            fssl_method=effective_fssl_method,
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

    def _validate_fssl_method(self, fssl_method: str) -> None:
        """fssl_method descriptor가 live server/agent를 지원하는지 확인한다."""
        from methods.federated_ssl.registry import (
            resolve_federated_ssl_method_descriptor,
        )

        try:
            descriptor = resolve_federated_ssl_method_descriptor(fssl_method)
        except NotImplementedError as error:
            raise StrategyValidationError(
                f"알 수 없는 fssl_method입니다: {fssl_method!r}. "
                "등록된 method 목록은 GET /api/v1/admin/methods 에서 확인하세요."
            ) from error
        if not descriptor.runtime_capabilities.live_server_supported:
            raise StrategyValidationError(
                f"fssl_method={fssl_method!r}는 아직 live server를 지원하지 않습니다."
            )
        if not descriptor.runtime_capabilities.live_agent_supported:
            raise StrategyValidationError(
                f"fssl_method={fssl_method!r}는 아직 live agent를 지원하지 않습니다."
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
