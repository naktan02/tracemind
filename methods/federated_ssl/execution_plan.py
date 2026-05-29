"""FL SSL 실행 계획 해석과 bootstrap 전 검증."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from methods.common.config_reading import (
    read_str,
    read_str_tuple,
    validate_allowed_keys,
)
from methods.federated_ssl.base import (
    ROUND_STATE_EXCHANGE_NONE,
    FederatedSslMethodDescriptor,
)
from methods.federated_ssl.method_config_surface import (
    is_public_method_owned_canonical,
    recommended_method_owned_variants,
)

COMPOSITION_MODE_METHOD_OWNED = "method_owned"
COMPOSITION_MODE_MANUAL = "manual"
EXECUTION_ROLE_METHOD_OWNED = "method_owned"
EXECUTION_ROLE_MANUAL_BASELINE = "manual_baseline"
SUPPORTED_COMPOSITION_MODES = frozenset(
    {
        COMPOSITION_MODE_METHOD_OWNED,
        COMPOSITION_MODE_MANUAL,
    }
)

SECURITY_POLICY_PLAINTEXT = "plaintext"
SUPPORTED_SECURITY_POLICIES = frozenset({SECURITY_POLICY_PLAINTEXT})


@dataclass(frozen=True, slots=True)
class FederatedSslManualAxes:
    """manual composition에서 사람이 직접 고르는 하위 mechanism 축."""

    client_ssl_objective: str | None = None
    server_aggregation: str | None = None
    update_family: str | None = None

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "FederatedSslManualAxes":
        """Hydra fl_method.manual_axes mapping을 typed 값으로 해석한다."""

        if source is None:
            return cls()
        validate_allowed_keys(
            source,
            allowed_keys=_MANUAL_AXIS_KEYS,
            config_name="fl_method.manual_axes",
        )
        return cls(
            client_ssl_objective=_read_optional_str(
                source,
                "client_ssl_objective",
            ),
            server_aggregation=_read_optional_str(
                source,
                "server_aggregation",
            ),
            update_family=_read_optional_str(
                source,
                "update_family",
            ),
        )

    @property
    def is_configured(self) -> bool:
        """manual 하위 축이 하나라도 명시됐는지 반환한다."""

        return any(
            value is not None
            for value in (
                self.client_ssl_objective,
                self.server_aggregation,
                self.update_family,
            )
        )

    def to_mapping(self) -> dict[str, str | None]:
        """report/diagnostics용 plain payload를 만든다."""

        return {
            "client_ssl_objective": self.client_ssl_objective,
            "server_aggregation": self.server_aggregation,
            "update_family": self.update_family,
        }


@dataclass(frozen=True, slots=True)
class FederatedSslSecurityPolicy:
    """FL update와 client metric을 어떤 보안 조건으로 다룰지 나타내는 실행 축."""

    name: str = SECURITY_POLICY_PLAINTEXT
    update_payload_visibility: str = "per_client_plaintext"
    client_metric_visibility: str = "per_client_plaintext"

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object] | None,
    ) -> "FederatedSslSecurityPolicy":
        """Hydra security_policy mapping을 typed 값으로 해석한다."""

        if source is None:
            return cls()
        validate_allowed_keys(
            source,
            allowed_keys=_SECURITY_POLICY_KEYS,
            config_name="security_policy",
        )
        return cls(
            name=read_str(
                source,
                "name",
                SECURITY_POLICY_PLAINTEXT,
                field_prefix="security_policy",
            ),
            update_payload_visibility=read_str(
                source,
                "update_payload_visibility",
                "per_client_plaintext",
                field_prefix="security_policy",
            ),
            client_metric_visibility=read_str(
                source,
                "client_metric_visibility",
                "per_client_plaintext",
                field_prefix="security_policy",
            ),
        )

    def require_supported(self) -> None:
        """현재 runtime이 지원하는 security policy인지 확인한다."""

        if self.name not in SUPPORTED_SECURITY_POLICIES:
            raise ValueError(
                "Unsupported security_policy.name for FL SSL simulation: "
                f"{self.name!r}. Supported values: "
                f"{sorted(SUPPORTED_SECURITY_POLICIES)}."
            )

    def to_mapping(self) -> dict[str, str]:
        """report/diagnostics용 plain payload를 만든다."""

        return {
            "name": self.name,
            "update_payload_visibility": self.update_payload_visibility,
            "client_metric_visibility": self.client_metric_visibility,
        }


@dataclass(frozen=True, slots=True)
class FederatedSslExecutionPlan:
    """method-first FL SSL 실행 계획의 canonical 해석 결과."""

    method_name: str
    descriptor_name: str | None
    composition_mode: str
    manual_axes: FederatedSslManualAxes
    round_state_exchange_name: str | None
    required_client_metric_keys: tuple[str, ...]
    security_policy: FederatedSslSecurityPolicy

    @classmethod
    def from_mapping(
        cls,
        *,
        fl_method: Mapping[str, object] | None,
        security_policy: Mapping[str, object] | None,
        method_descriptor: FederatedSslMethodDescriptor | None,
    ) -> "FederatedSslExecutionPlan":
        """Hydra FL method/security config와 descriptor를 실행 계획으로 해석한다."""

        source = fl_method or {}
        validate_allowed_keys(
            source,
            allowed_keys=_FL_METHOD_KEYS,
            config_name="fl_method",
        )
        composition_mode = read_str(
            source,
            "composition_mode",
            COMPOSITION_MODE_METHOD_OWNED,
            field_prefix="fl_method",
        )
        round_state_exchange = (
            None
            if method_descriptor is None
            else method_descriptor.round_state_exchange
        )
        default_exchange_name = (
            ROUND_STATE_EXCHANGE_NONE
            if method_descriptor is None
            else (
                None
                if round_state_exchange is None
                else round_state_exchange.exchange_name
            )
        )
        default_metric_keys = (
            ()
            if round_state_exchange is None
            else round_state_exchange.required_client_metric_keys
        )
        plan = cls(
            method_name=_read_method_name(
                source,
                composition_mode=composition_mode,
                method_descriptor=method_descriptor,
            ),
            descriptor_name=_read_descriptor_name(
                source,
                method_descriptor=method_descriptor,
            ),
            composition_mode=composition_mode,
            manual_axes=FederatedSslManualAxes.from_mapping(
                _read_optional_mapping(source, "manual_axes")
            ),
            round_state_exchange_name=default_exchange_name,
            required_client_metric_keys=read_str_tuple(
                source,
                "required_client_metric_keys",
                default_metric_keys,
            ),
            security_policy=FederatedSslSecurityPolicy.from_mapping(security_policy),
        )
        if method_descriptor is None:
            plan.require_manual_plan_without_descriptor()
        else:
            plan.require_matches_descriptor(method_descriptor)
        return plan

    def require_manual_plan_without_descriptor(self) -> None:
        """manual baseline 실행 계획이 method descriptor를 참조하지 않는지 검증한다."""

        if self.composition_mode not in SUPPORTED_COMPOSITION_MODES:
            raise ValueError(
                "fl_method.composition_mode must be one of "
                f"{sorted(SUPPORTED_COMPOSITION_MODES)}: {self.composition_mode!r}."
            )
        if self.composition_mode != COMPOSITION_MODE_MANUAL:
            raise ValueError(
                "method-owned fl_method requires a selected method descriptor."
            )
        if self.descriptor_name is not None:
            raise ValueError(
                "manual fl_method.descriptor_name must be omitted; manual_baseline "
                "is an execution role, not a method descriptor."
            )
        if self.round_state_exchange_name != ROUND_STATE_EXCHANGE_NONE:
            raise ValueError(
                "manual fl_method round_state_exchange must be 'none' unless a "
                "method-owned descriptor defines a custom exchange."
            )
        if self.required_client_metric_keys:
            raise ValueError(
                "manual fl_method.required_client_metric_keys must be empty unless "
                "a method-owned descriptor defines them."
            )
        self._require_manual_axes_are_explicit()
        self.security_policy.require_supported()

    def require_matches_descriptor(
        self,
        method_descriptor: FederatedSslMethodDescriptor,
    ) -> None:
        """계획이 descriptor와 지원 capability에서 drift되지 않았는지 검증한다."""

        if self.composition_mode == COMPOSITION_MODE_MANUAL:
            raise ValueError(
                "manual fl_method must not be validated with a method descriptor; "
                "use descriptor=None so report metadata does not look method-owned."
            )
        if self.descriptor_name != method_descriptor.name:
            raise ValueError(
                "fl_method.descriptor_name must match the selected method descriptor: "
                f"{self.descriptor_name!r} != {method_descriptor.name!r}."
            )
        if self.composition_mode not in SUPPORTED_COMPOSITION_MODES:
            raise ValueError(
                "fl_method.composition_mode must be one of "
                f"{sorted(SUPPORTED_COMPOSITION_MODES)}: {self.composition_mode!r}."
            )
        if self.composition_mode == COMPOSITION_MODE_METHOD_OWNED:
            self._require_method_owned_plan_matches_descriptor(method_descriptor)

        descriptor_exchange = method_descriptor.round_state_exchange
        expected_exchange_name = (
            None if descriptor_exchange is None else descriptor_exchange.exchange_name
        )
        expected_metric_keys = (
            ()
            if descriptor_exchange is None
            else descriptor_exchange.required_client_metric_keys
        )
        if self.round_state_exchange_name != expected_exchange_name:
            raise ValueError(
                "fl_method round_state_exchange must match the selected method "
                f"descriptor: {self.round_state_exchange_name!r} != "
                f"{expected_exchange_name!r}."
            )
        if self.required_client_metric_keys != expected_metric_keys:
            raise ValueError(
                "fl_method.required_client_metric_keys must match the selected "
                "method descriptor: "
                f"{list(self.required_client_metric_keys)!r} != "
                f"{list(expected_metric_keys)!r}."
            )
        self.security_policy.require_supported()

    def to_mapping(self) -> dict[str, object]:
        """report/diagnostics용 plain payload를 만든다."""

        return {
            "name": self.method_name,
            "descriptor_name": self.descriptor_name,
            "composition_mode": self.composition_mode,
            "execution_role": self.execution_role,
            "manual_axes": self.manual_axes.to_mapping(),
            "round_state_exchange": {
                "exchange_name": self.round_state_exchange_name,
                "required_client_metric_keys": list(self.required_client_metric_keys),
            },
            "security_policy": self.security_policy.to_mapping(),
        }

    @property
    def execution_role(self) -> str:
        """report에서 manual baseline과 method-owned 실행을 분리하는 역할값."""

        if self.composition_mode == COMPOSITION_MODE_MANUAL:
            return EXECUTION_ROLE_MANUAL_BASELINE
        return EXECUTION_ROLE_METHOD_OWNED

    def _require_method_owned_plan_matches_descriptor(
        self,
        method_descriptor: FederatedSslMethodDescriptor,
    ) -> None:
        if self.method_name != method_descriptor.name:
            raise ValueError(
                "method-owned fl_method.name must match the selected method "
                f"descriptor: {self.method_name!r} != {method_descriptor.name!r}."
            )
        if self.manual_axes.is_configured:
            raise ValueError(
                "fl_method.manual_axes must stay empty when composition_mode is "
                f"{COMPOSITION_MODE_METHOD_OWNED!r}."
            )
        if not is_public_method_owned_canonical(method_descriptor):
            recommended_variants = recommended_method_owned_variants(method_descriptor)
            recommendation_text = (
                ""
                if not recommended_variants
                else " Use one of "
                + ", ".join(repr(name) for name in recommended_variants)
                + " instead."
            )
            raise ValueError(
                "method-owned public surface no longer accepts generic method "
                f"{method_descriptor.name!r}.{recommendation_text}"
            )

    def _require_manual_axes_are_explicit(self) -> None:
        if self.method_name != COMPOSITION_MODE_MANUAL:
            raise ValueError(
                "manual fl_method must omit name or use name='manual' so reports do "
                "not look like a paper method run."
            )
        missing_axes = [
            axis_name
            for axis_name, value in self.manual_axes.to_mapping().items()
            if value is None
        ]
        if missing_axes:
            raise ValueError(
                f"manual fl_method requires explicit lower axes: {missing_axes}."
            )


def build_federated_ssl_execution_plan(
    *,
    fl_method: Mapping[str, object] | None,
    security_policy: Mapping[str, object] | None,
    method_descriptor: FederatedSslMethodDescriptor | None,
) -> FederatedSslExecutionPlan:
    """FL SSL 실행 계획을 만들고 bootstrap 전 검증을 수행한다."""

    return FederatedSslExecutionPlan.from_mapping(
        fl_method=fl_method,
        security_policy=security_policy,
        method_descriptor=method_descriptor,
    )


def _read_optional_mapping(
    source: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _read_optional_str(
    source: Mapping[str, object],
    key: str,
) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _read_method_name(
    source: Mapping[str, object],
    *,
    composition_mode: str,
    method_descriptor: FederatedSslMethodDescriptor | None,
) -> str:
    if composition_mode == COMPOSITION_MODE_MANUAL and source.get("name") is None:
        return COMPOSITION_MODE_MANUAL
    default_name = None if method_descriptor is None else method_descriptor.name
    if default_name is None and source.get("name") is None:
        raise ValueError("method-owned fl_method requires name or method_descriptor.")
    return read_str(
        source,
        "name",
        default_name,
        field_prefix="fl_method",
    )


def _read_descriptor_name(
    source: Mapping[str, object],
    *,
    method_descriptor: FederatedSslMethodDescriptor | None,
) -> str | None:
    if method_descriptor is None:
        return _read_optional_str(source, "descriptor_name")
    return read_str(
        source,
        "descriptor_name",
        method_descriptor.name,
        field_prefix="fl_method",
    )


_MANUAL_AXIS_KEYS = frozenset(
    {
        "client_ssl_objective",
        "server_aggregation",
        "update_family",
    }
)

_SECURITY_POLICY_KEYS = frozenset(
    {
        "name",
        "update_payload_visibility",
        "client_metric_visibility",
    }
)

_FL_METHOD_KEYS = frozenset(
    {
        "name",
        "descriptor_name",
        "composition_mode",
        "manual_axes",
        "required_client_metric_keys",
    }
)
