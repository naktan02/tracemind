"""method descriptor가 요구하는 round state exchange를 실행한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundValidationError,
)
from main_server.src.services.federation.rounds.boundary.models import RoundRecord
from methods.federated_ssl.base import (
    ROUND_STATE_EXCHANGE_CLIENT_METRIC_SUMMARY,
    ROUND_STATE_EXCHANGE_NONE,
    FederatedSslMethodDescriptor,
    FederatedSslRoundStateExchangeSpec,
)
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope

NO_ROUND_STATE_EXCHANGE_NAME = ROUND_STATE_EXCHANGE_NONE
CLIENT_METRIC_SUMMARY_EXCHANGE_NAME = ROUND_STATE_EXCHANGE_CLIENT_METRIC_SUMMARY
SUPPORTED_DEFAULT_ROUND_STATE_EXCHANGES = frozenset(
    {NO_ROUND_STATE_EXCHANGE_NAME, CLIENT_METRIC_SUMMARY_EXCHANGE_NAME}
)


@dataclass(frozen=True, slots=True)
class RoundStateExchangeResult:
    """round state exchange 실행 결과."""

    exchange_name: str
    summary_metrics: dict[str, float] = field(default_factory=dict)


class RoundStateExchangeExecutor(Protocol):
    """main_server가 제공하는 method-agnostic round state exchange capability."""

    def summarize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        record: RoundRecord,
    ) -> RoundStateExchangeResult:
        """finalize 전에 client metric/state summary를 만든다."""


@dataclass(frozen=True, slots=True)
class DefaultRoundStateExchangeExecutor:
    """기본 live runtime이 제공하는 round state exchange 구현."""

    def summarize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        record: RoundRecord,
    ) -> RoundStateExchangeResult:
        spec = method_descriptor.round_state_exchange
        if spec is None or spec.exchange_name == NO_ROUND_STATE_EXCHANGE_NAME:
            return RoundStateExchangeResult(exchange_name=NO_ROUND_STATE_EXCHANGE_NAME)
        _validate_default_round_state_exchange_spec(
            method_name=method_descriptor.name,
            spec=spec,
            error_type=RoundValidationError,
        )
        return RoundStateExchangeResult(
            exchange_name=spec.exchange_name,
            summary_metrics=_summarize_client_metric_exchange(
                spec=spec,
                updates=record.updates,
            ),
        )


def validate_default_round_state_exchange_descriptor(
    method_descriptor: FederatedSslMethodDescriptor,
) -> None:
    """runtime bootstrap 단계에서 default exchange 지원 여부를 검증한다."""

    spec = method_descriptor.round_state_exchange
    if spec is None or spec.exchange_name == NO_ROUND_STATE_EXCHANGE_NAME:
        return
    _validate_default_round_state_exchange_spec(
        method_name=method_descriptor.name,
        spec=spec,
        error_type=ValueError,
    )


def _validate_default_round_state_exchange_spec(
    *,
    method_name: str,
    spec: FederatedSslRoundStateExchangeSpec,
    error_type: type[Exception],
) -> None:
    if spec.requires_custom_exchange:
        raise error_type(
            "Configured FL SSL method requires a custom round state exchange "
            "capability, but only the default live server exchange is wired: "
            f"{method_name}."
        )
    if spec.exchange_name not in SUPPORTED_DEFAULT_ROUND_STATE_EXCHANGES:
        raise error_type(
            "Unsupported round state exchange for default live runtime: "
            f"{spec.exchange_name}."
        )


def _summarize_client_metric_exchange(
    *,
    spec: FederatedSslRoundStateExchangeSpec,
    updates: tuple[TrainingUpdateEnvelope, ...],
) -> dict[str, float]:
    if not updates:
        return {}
    summary = {
        f"{spec.summary_metric_prefix}.update_count": float(len(updates)),
        f"{spec.summary_metric_prefix}.example_count": float(
            sum(update.example_count for update in updates)
        ),
    }
    for metric_key in spec.required_client_metric_keys:
        _require_metric_in_all_updates(metric_key=metric_key, updates=updates)
        summary[f"{spec.summary_metric_prefix}.{metric_key}.mean"] = (
            _example_weighted_metric_mean(metric_key=metric_key, updates=updates)
        )
    return summary


def _require_metric_in_all_updates(
    *,
    metric_key: str,
    updates: tuple[TrainingUpdateEnvelope, ...],
) -> None:
    missing_update_ids = tuple(
        update.update_id
        for update in updates
        if metric_key not in update.client_metrics
    )
    if missing_update_ids:
        raise RoundValidationError(
            "Round state exchange requires client metric missing from updates: "
            f"metric_key={metric_key}, update_ids={missing_update_ids}."
        )


def _example_weighted_metric_mean(
    *,
    metric_key: str,
    updates: tuple[TrainingUpdateEnvelope, ...],
) -> float:
    total_examples = sum(update.example_count for update in updates)
    if total_examples <= 0:
        return sum(update.client_metrics[metric_key] for update in updates) / len(
            updates
        )
    return (
        sum(
            update.client_metrics[metric_key] * update.example_count
            for update in updates
        )
        / total_examples
    )
