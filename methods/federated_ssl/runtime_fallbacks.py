"""FL SSL runtime fallback profile.

이 모듈은 API/runtime 요청이 명시 training 값을 주지 않았을 때만 쓰는
compatibility fallback을 소유한다. live 기본값은 FL SSL 실험 기본 조합인
FixMatch local objective + PEFT text encoder update + FedAvg server aggregation을
가리키되, 실제 실험 실행값의 source of truth는 Hydra `conf/strategy_axes/`에
남긴다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from methods.adaptation.peft_text_encoder.config import (
    PEFT_ENCODER_DELTA_FORMAT_INLINE,
    PEFT_ENCODER_PAYLOAD_ADAPTER_KIND,
    PEFT_ENCODER_TRAINING_BACKEND_NAME,
)
from methods.adaptation.privacy_guards.noop import NOOP_PRIVACY_GUARD_NAME
from methods.common.config_reading import freeze_mapping
from shared.src.contracts.training_contracts import (
    SecureAggregationConfig,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
)
from shared.src.contracts.training_example_backends import (
    WEAK_STRONG_PAIR_EXAMPLE_BACKEND,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERY_SSL_METHOD_CONFIG_DIR = (
    REPO_ROOT / "conf" / "strategy_axes" / "ssl_objective" / "consistency_method"
)
_NON_OBJECTIVE_QUERY_SSL_CONFIG_KEYS = frozenset(
    {
        "name",
        "algorithm_name",
        "require_multiview",
        # Hydra config may express this as ${train_batch_size}; live runtime resolves
        # it from the round task batch size below.
        "unlabeled_batch_size",
    }
)


def _merged_mapping(
    source: Mapping[str, TrainingConfigScalar],
    overrides: Mapping[str, TrainingConfigScalar] | None,
) -> dict[str, TrainingConfigScalar]:
    merged = dict(source)
    if overrides is not None:
        merged.update(dict(overrides))
    return merged


def _load_query_ssl_method_objective_defaults(
    method_name: str,
) -> Mapping[str, TrainingConfigScalar]:
    """Hydra consistency_method YAML에서 live task objective 기본값을 읽는다."""

    path = QUERY_SSL_METHOD_CONFIG_DIR / f"{method_name}.yaml"
    if not path.exists():
        raise ValueError(f"Query SSL method config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Query SSL method config must be a mapping: {path}")
    declared_name = str(payload.get("name", "")).strip()
    if declared_name != method_name:
        raise ValueError(
            f"Query SSL method config name mismatch: {declared_name!r} != "
            f"{method_name!r}."
        )
    algorithm_name = str(payload.get("algorithm_name", "")).strip()
    if not algorithm_name:
        raise ValueError(f"Query SSL method config requires algorithm_name: {path}")
    defaults: dict[str, TrainingConfigScalar] = {
        "method_name": method_name,
        "algorithm_name": algorithm_name,
    }
    for key, value in payload.items():
        normalized_key = str(key).strip()
        if normalized_key in _NON_OBJECTIVE_QUERY_SSL_CONFIG_KEYS:
            continue
        defaults[normalized_key] = _normalize_query_ssl_config_scalar(
            value,
            key=normalized_key,
            path=path,
        )
    return freeze_mapping(defaults)


def _normalize_query_ssl_config_scalar(
    value: object,
    *,
    key: str,
    path: Path,
) -> TrainingConfigScalar:
    if isinstance(value, str):
        if value.strip().startswith("${"):
            raise ValueError(
                f"Query SSL runtime fallback cannot use unresolved interpolation "
                f"for {key!r}: {path}"
            )
        return value
    if isinstance(value, bool | int | float):
        return value
    raise ValueError(
        f"Query SSL runtime fallback only supports scalar config values for "
        f"{key!r}: {path}"
    )


@dataclass(frozen=True, slots=True)
class RuntimeTrainingTaskDefaults:
    """round open/task 생성에 쓰는 top-level runtime fallback 값."""

    local_epochs: int
    batch_size: int
    learning_rate: float
    max_steps: int
    min_required_examples: int | None = None
    gradient_clip_norm: float | None = None


@dataclass(frozen=True, slots=True)
class RuntimeFallbackTrainingProfile:
    """명시 config가 없는 legacy/runtime 요청을 위한 fallback profile."""

    profile_name: str
    objective_mapping: Mapping[str, TrainingConfigScalar]
    selection_mapping: Mapping[str, TrainingConfigScalar]
    secure_aggregation_mapping: Mapping[str, TrainingConfigScalar]
    task_runtime_defaults: RuntimeTrainingTaskDefaults
    default_acceptance_policy_name: str = "top1_ranked"
    default_pseudo_label_algorithm_name: str = "top1_ranked"
    default_evidence_backend_name: str = "analysis_score_evidence"
    default_score_policy_name: str = "max_cosine"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_mapping",
            freeze_mapping(self.objective_mapping),
        )
        object.__setattr__(
            self,
            "selection_mapping",
            freeze_mapping(self.selection_mapping),
        )
        object.__setattr__(
            self,
            "secure_aggregation_mapping",
            freeze_mapping(self.secure_aggregation_mapping),
        )

    @property
    def acceptance_policy_name(self) -> str:
        return self._objective_str(
            "acceptance_policy_name",
            default=self.default_acceptance_policy_name,
        )

    @property
    def pseudo_label_algorithm_name(self) -> str:
        return self._objective_str(
            "pseudo_label_algorithm_name",
            default=self.default_pseudo_label_algorithm_name,
        )

    @property
    def training_backend_name(self) -> str:
        return self._objective_str("training_backend_name")

    @property
    def algorithm_profile_name(self) -> str:
        return self._objective_str("algorithm_profile_name")

    @property
    def example_generation_backend_name(self) -> str:
        return self._objective_str("example_generation_backend_name")

    @property
    def evidence_backend_name(self) -> str:
        return self._objective_str(
            "evidence_backend_name",
            default=self.default_evidence_backend_name,
        )

    @property
    def score_policy_name(self) -> str:
        return self._objective_str(
            "score_policy_name",
            default=self.default_score_policy_name,
        )

    @property
    def privacy_guard_name(self) -> str:
        return self._objective_str("privacy_guard_name")

    @property
    def max_examples(self) -> int | None:
        value = self.selection_mapping.get("max_examples")
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Runtime fallback max_examples must not be bool.")
        return int(value)

    @property
    def local_epochs(self) -> int:
        return self.task_runtime_defaults.local_epochs

    @property
    def batch_size(self) -> int:
        return self.task_runtime_defaults.batch_size

    @property
    def learning_rate(self) -> float:
        return self.task_runtime_defaults.learning_rate

    @property
    def max_steps(self) -> int:
        return self.task_runtime_defaults.max_steps

    @property
    def min_required_examples(self) -> int | None:
        return self.task_runtime_defaults.min_required_examples

    @property
    def gradient_clip_norm(self) -> float | None:
        return self.task_runtime_defaults.gradient_clip_norm

    def build_objective_config(
        self,
        *,
        overrides: Mapping[str, TrainingConfigScalar] | None = None,
    ) -> TrainingObjectiveConfig:
        return TrainingObjectiveConfig.from_mapping(
            _merged_mapping(self.objective_mapping, overrides)
        )

    def build_selection_policy(
        self,
        *,
        overrides: Mapping[str, TrainingConfigScalar] | None = None,
    ) -> TrainingSelectionPolicy:
        return TrainingSelectionPolicy.from_mapping(
            _merged_mapping(self.selection_mapping, overrides)
        )

    def build_secure_aggregation_config(
        self,
        *,
        overrides: Mapping[str, TrainingConfigScalar] | None = None,
    ) -> SecureAggregationConfig:
        return SecureAggregationConfig.from_mapping(
            _merged_mapping(self.secure_aggregation_mapping, overrides)
        )

    def _objective_str(self, key: str, *, default: str | None = None) -> str:
        if key not in self.objective_mapping and default is not None:
            return default
        return self._require_str(self.objective_mapping, key)

    def _objective_float(self, key: str) -> float:
        return self._require_float(self.objective_mapping, key)

    @staticmethod
    def _require_str(
        source: Mapping[str, TrainingConfigScalar],
        key: str,
    ) -> str:
        value = source.get(key)
        if value is None:
            raise ValueError(f"Runtime fallback training profile is missing: {key}")
        return str(value)

    @staticmethod
    def _require_float(
        source: Mapping[str, TrainingConfigScalar],
        key: str,
    ) -> float:
        value = source.get(key)
        if value is None:
            raise ValueError(f"Runtime fallback training profile is missing: {key}")
        if isinstance(value, bool):
            raise ValueError(f"Runtime fallback key must not be bool: {key}")
        return float(value)


@dataclass(frozen=True, slots=True)
class RuntimeFallbackServerRoundProfile:
    """명시 round runtime config가 없는 live/API 요청용 fallback profile."""

    profile_name: str
    payload_adapter_kind: str
    update_family_name: str
    aggregation_backend_name: str
    method_descriptor_name: str | None = None
    aggregation_backend_overrides: Mapping[str, TrainingConfigScalar] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aggregation_backend_overrides",
            freeze_mapping(self.aggregation_backend_overrides),
        )


FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK_NAME = "fixmatch_fedavg.v1"
FIXMATCH_QUERY_SSL_METHOD_NAME = "fixmatch_usb_v1"
FIXMATCH_QUERY_SSL_ALGORITHM_NAME = "fixmatch"
FIXMATCH_QUERY_SSL_STRONG_VIEW_POLICY = "first_aug"
FLEXMATCH_QUERY_SSL_METHOD_NAME = "flexmatch_usb_v1"
FLEXMATCH_QUERY_SSL_ALGORITHM_NAME = "flexmatch"
PEFT_CLASSIFIER_UPDATE_PROFILE_NAME = "peft_classifier_update_v1"
PEFT_TEXT_ENCODER_FEDAVG_SERVER_ROUND_RUNTIME_FALLBACK_NAME = (
    "default_peft_text_encoder_fedavg.v1"
)
PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME = "peft_text_encoder"
FEDAVG_AGGREGATION_BACKEND_NAME = "fedavg"
FEDAVG_MERGED_DELTA_SERVER_UPDATE_POLICY_NAME = "fedavg_merged_delta"

QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS = freeze_mapping(
    {
        FIXMATCH_QUERY_SSL_METHOD_NAME: _load_query_ssl_method_objective_defaults(
            FIXMATCH_QUERY_SSL_METHOD_NAME
        ),
        FLEXMATCH_QUERY_SSL_METHOD_NAME: _load_query_ssl_method_objective_defaults(
            FLEXMATCH_QUERY_SSL_METHOD_NAME
        ),
    }
)

RUNTIME_FALLBACK_TRAINING_OBJECTIVE_MAPPING = freeze_mapping(
    {
        "algorithm_profile_name": PEFT_CLASSIFIER_UPDATE_PROFILE_NAME,
        "training_backend_name": PEFT_ENCODER_TRAINING_BACKEND_NAME,
        "example_generation_backend_name": WEAK_STRONG_PAIR_EXAMPLE_BACKEND,
        "privacy_guard_name": NOOP_PRIVACY_GUARD_NAME,
        "query_ssl.method_name": FIXMATCH_QUERY_SSL_METHOD_NAME,
        "query_ssl.algorithm_name": FIXMATCH_QUERY_SSL_ALGORITHM_NAME,
        "query_ssl.strong_view_policy": FIXMATCH_QUERY_SSL_STRONG_VIEW_POLICY,
        "query_ssl.unlabeled_batch_size": 8,
        **{
            f"query_ssl.{key}": value
            for key, value in QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS[
                FIXMATCH_QUERY_SSL_METHOD_NAME
            ].items()
            if key not in {"method_name", "algorithm_name"}
        },
        "peft_classifier.delta_format": PEFT_ENCODER_DELTA_FORMAT_INLINE,
    }
)

RUNTIME_FALLBACK_TRAINING_SELECTION_MAPPING = freeze_mapping({"max_examples": 128})

RUNTIME_FALLBACK_SECURE_AGGREGATION_MAPPING = freeze_mapping({"required": False})

RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS = RuntimeTrainingTaskDefaults(
    local_epochs=1,
    batch_size=8,
    learning_rate=1e-4,
    max_steps=50,
    min_required_examples=None,
    gradient_clip_norm=None,
)

FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK = RuntimeFallbackTrainingProfile(
    profile_name=FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK_NAME,
    objective_mapping=RUNTIME_FALLBACK_TRAINING_OBJECTIVE_MAPPING,
    selection_mapping=RUNTIME_FALLBACK_TRAINING_SELECTION_MAPPING,
    secure_aggregation_mapping=RUNTIME_FALLBACK_SECURE_AGGREGATION_MAPPING,
    task_runtime_defaults=RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS,
)

RUNTIME_FALLBACK_TRAINING_PROFILE = FIXMATCH_FEDAVG_V1_RUNTIME_FALLBACK
RUNTIME_FALLBACK_SERVER_ROUND_PROFILE = RuntimeFallbackServerRoundProfile(
    profile_name=PEFT_TEXT_ENCODER_FEDAVG_SERVER_ROUND_RUNTIME_FALLBACK_NAME,
    payload_adapter_kind=PEFT_ENCODER_PAYLOAD_ADAPTER_KIND,
    update_family_name=PEFT_TEXT_ENCODER_UPDATE_FAMILY_NAME,
    aggregation_backend_name=FEDAVG_AGGREGATION_BACKEND_NAME,
)


def build_runtime_fallback_training_objective_config(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> TrainingObjectiveConfig:
    """명시 objective가 없는 runtime 요청용 fallback config를 조립한다."""

    return RUNTIME_FALLBACK_TRAINING_PROFILE.build_objective_config(overrides=overrides)


def build_runtime_strategy_training_objective_config(
    *,
    local_update_profile_name: str | None = None,
    strategy_mode: str = "composed",
    ssl_method_name: str | None = None,
    fssl_method_name: str | None = None,
    server_update_policy_name: str | None = None,
    aggregation_backend_name: str | None = None,
    strong_view_policy: str = FIXMATCH_QUERY_SSL_STRONG_VIEW_POLICY,
    unlabeled_batch_size: int = RUNTIME_FALLBACK_TRAINING_TASK_DEFAULTS.batch_size,
    parameter_overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> TrainingObjectiveConfig:
    """운영 round strategy 입력을 canonical TrainingObjectiveConfig로 조립한다."""

    normalized_mode = _optional_name(strategy_mode) or "composed"
    if normalized_mode not in {"composed", "method_owned"}:
        raise ValueError(f"Unsupported live strategy mode: {normalized_mode!r}.")
    normalized_fssl_method = _optional_name(fssl_method_name)
    if normalized_mode == "method_owned" and normalized_fssl_method is None:
        raise ValueError("method_owned live strategy requires fssl_method.")
    if normalized_mode == "composed" and normalized_fssl_method is not None:
        raise ValueError("composed live strategy must not provide fssl_method.")
    normalized_profile = (
        _optional_name(local_update_profile_name) or PEFT_CLASSIFIER_UPDATE_PROFILE_NAME
    )
    if normalized_profile != PEFT_CLASSIFIER_UPDATE_PROFILE_NAME:
        raise ValueError(
            "Unsupported live local_update_profile: "
            f"{normalized_profile!r}. Supported: "
            f"{PEFT_CLASSIFIER_UPDATE_PROFILE_NAME!r}."
        )
    normalized_server_update = (
        _optional_name(server_update_policy_name)
        or FEDAVG_MERGED_DELTA_SERVER_UPDATE_POLICY_NAME
    )
    if normalized_server_update != FEDAVG_MERGED_DELTA_SERVER_UPDATE_POLICY_NAME:
        raise ValueError(
            f"Unsupported live server_update_policy: {normalized_server_update!r}."
        )
    normalized_aggregation = _optional_name(aggregation_backend_name)
    if (
        normalized_aggregation is not None
        and normalized_aggregation != FEDAVG_AGGREGATION_BACKEND_NAME
    ):
        raise ValueError(
            f"Unsupported live aggregation_backend: {normalized_aggregation!r}."
        )

    query_ssl_method = _optional_name(ssl_method_name) or FIXMATCH_QUERY_SSL_METHOD_NAME
    query_ssl_defaults = QUERY_SSL_METHOD_OBJECTIVE_DEFAULTS.get(query_ssl_method)
    if query_ssl_defaults is None:
        raise ValueError(f"Unsupported live ssl_method: {query_ssl_method!r}.")
    if unlabeled_batch_size <= 0:
        raise ValueError("unlabeled_batch_size must be positive.")

    objective = dict(RUNTIME_FALLBACK_TRAINING_OBJECTIVE_MAPPING)
    objective["algorithm_profile_name"] = normalized_profile
    for key, value in query_ssl_defaults.items():
        objective[f"query_ssl.{key}"] = value
    objective["query_ssl.strong_view_policy"] = strong_view_policy
    objective["query_ssl.unlabeled_batch_size"] = unlabeled_batch_size
    if parameter_overrides:
        objective.update(
            _normalize_runtime_strategy_parameter_overrides(parameter_overrides)
        )
    return TrainingObjectiveConfig.from_mapping(objective)


def _normalize_runtime_strategy_parameter_overrides(
    source: Mapping[str, TrainingConfigScalar],
) -> dict[str, TrainingConfigScalar]:
    """운영 strategy override를 method parameter scope로 정규화한다."""

    result: dict[str, TrainingConfigScalar] = {}
    for key, value in source.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("strategy parameter override keys must not be empty.")
        if "." not in normalized_key:
            normalized_key = f"query_ssl.{normalized_key}"
        result[normalized_key] = value
    return result


def _optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def resolve_runtime_example_generation_backend_name(
    objective_config: object | None,
) -> str:
    """명시 objective 값이 없으면 runtime fallback backend 이름을 반환한다."""

    if objective_config is None:
        return RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    raw_value = getattr(objective_config, "example_generation_backend_name", None)
    if raw_value is None:
        return RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    normalized = str(raw_value).strip()
    return (
        normalized or RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )


def build_runtime_fallback_training_selection_policy(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> TrainingSelectionPolicy:
    """명시 selection policy가 없는 runtime 요청용 fallback을 조립한다."""

    return RUNTIME_FALLBACK_TRAINING_PROFILE.build_selection_policy(overrides=overrides)


def build_runtime_fallback_secure_aggregation_config(
    *,
    overrides: Mapping[str, TrainingConfigScalar] | None = None,
) -> SecureAggregationConfig:
    """명시 secure aggregation config가 없는 runtime 요청용 fallback을 조립한다."""

    return RUNTIME_FALLBACK_TRAINING_PROFILE.build_secure_aggregation_config(
        overrides=overrides
    )
