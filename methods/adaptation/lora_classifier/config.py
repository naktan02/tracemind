"""LoRA-classifier training backend config parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shared.src.contracts.training_contracts import (
    TrainingConfigScalar,
    TrainingObjectiveConfig,
)

LORA_CLASSIFIER_TRAINING_BACKEND_NAME = "lora_classifier_trainer"
LORA_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE = "lora_classifier_trainer"
LORA_CLASSIFIER_FAMILY_EXTRA_SCOPE = "lora_classifier"

_STRING_CONFIG_KEYS = (
    "backbone_model_id",
    "backbone_revision",
    "tokenizer_model_id",
    "tokenizer_revision",
    "pooling",
    "peft_adapter_name",
    "bias",
    "target_modules",
    "delta_format",
    "artifact_ref_prefix",
)
_ALLOW_EMPTY_STRING_CONFIG_KEYS = ("task_prefix",)
_POSITIVE_INT_CONFIG_KEYS = ("max_length", "rank", "alpha")
_FLOAT_CONFIG_KEYS = ("dropout",)
_BOOL_CONFIG_KEYS = ("use_rslora",)
_STRING_TUPLE_CONFIG_KEYS = ("text_metadata_keys", "label_schema")
_CONFIG_EXTRA_KEYS = frozenset(
    {
        *_STRING_CONFIG_KEYS,
        *_ALLOW_EMPTY_STRING_CONFIG_KEYS,
        *_POSITIVE_INT_CONFIG_KEYS,
        *_FLOAT_CONFIG_KEYS,
        *_BOOL_CONFIG_KEYS,
        *_STRING_TUPLE_CONFIG_KEYS,
    }
)


@dataclass(frozen=True, slots=True)
class LoraClassifierTrainingBackendConfig:
    """Agent가 LoRA-classifier update payload에 기록할 trainer snapshot."""

    backbone_model_id: str = "mixedbread-ai/mxbai-embed-large-v1"
    backbone_revision: str = "main"
    tokenizer_model_id: str = "mixedbread-ai/mxbai-embed-large-v1"
    tokenizer_revision: str = "main"
    pooling: str = "mean"
    max_length: int = 256
    task_prefix: str = ""
    peft_adapter_name: str = "lora"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    bias: str = "none"
    target_modules: str = "all-linear"
    use_rslora: bool = False
    delta_format: str = "agent_local_artifact_ref"
    artifact_ref_prefix: str = "agent-local://lora_classifier"
    text_metadata_keys: tuple[str, ...] = (
        "strong_text",
        "training_text",
        "raw_text",
        "text",
        "weak_text",
    )
    label_schema: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _set_normalized_str(self, "backbone_model_id", self.backbone_model_id)
        _set_normalized_str(self, "backbone_revision", self.backbone_revision)
        _set_normalized_str(self, "tokenizer_model_id", self.tokenizer_model_id)
        _set_normalized_str(self, "tokenizer_revision", self.tokenizer_revision)
        _set_normalized_str(self, "pooling", self.pooling)
        _set_normalized_str(
            self,
            "task_prefix",
            self.task_prefix,
            allow_empty=True,
        )
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        _set_normalized_str(self, "peft_adapter_name", self.peft_adapter_name)
        if self.rank <= 0:
            raise ValueError("rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")
        _set_normalized_str(self, "bias", self.bias)
        _set_normalized_str(self, "target_modules", self.target_modules)
        _set_normalized_str(self, "delta_format", self.delta_format)
        _set_normalized_str(self, "artifact_ref_prefix", self.artifact_ref_prefix)
        text_keys = tuple(
            str(value).strip()
            for value in self.text_metadata_keys
            if str(value).strip()
        )
        if not text_keys:
            raise ValueError("text_metadata_keys must not be empty.")
        object.__setattr__(self, "text_metadata_keys", text_keys)
        object.__setattr__(
            self,
            "label_schema",
            tuple(
                str(value).strip() for value in self.label_schema if str(value).strip()
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, TrainingConfigScalar],
    ) -> "LoraClassifierTrainingBackendConfig":
        """Scoped objective extras에서 backend config를 만든다."""

        if not source:
            return cls()

        unknown_keys = sorted(set(source) - _CONFIG_EXTRA_KEYS)
        if unknown_keys:
            raise ValueError(
                "Unsupported LoRA-classifier training backend config key(s): "
                f"{unknown_keys}."
            )

        defaults = cls()
        values: dict[str, object] = {}
        for key in _STRING_CONFIG_KEYS:
            values[key] = _str_value(source, key, getattr(defaults, key))
        for key in _ALLOW_EMPTY_STRING_CONFIG_KEYS:
            values[key] = _str_value(
                source,
                key,
                getattr(defaults, key),
                allow_empty=True,
            )
        for key in _POSITIVE_INT_CONFIG_KEYS:
            values[key] = _positive_int_value(source, key, getattr(defaults, key))
        for key in _FLOAT_CONFIG_KEYS:
            values[key] = _float_value(source, key, getattr(defaults, key))
        for key in _BOOL_CONFIG_KEYS:
            values[key] = _bool_value(source, key, getattr(defaults, key))
        for key in _STRING_TUPLE_CONFIG_KEYS:
            values[key] = _str_tuple_value(source, key, getattr(defaults, key))
        return cls(**values)

    def to_backbone_payload(self) -> dict[str, str | int]:
        """Shared payload에 들어갈 backbone/tokenizer snapshot."""

        return {
            "backbone_model_id": self.backbone_model_id,
            "backbone_revision": self.backbone_revision,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_revision": self.tokenizer_revision,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "task_prefix": self.task_prefix,
        }

    def to_lora_config_payload(self) -> dict[str, str | int | float | bool]:
        """Shared payload에 들어갈 LoRA config snapshot."""

        return {
            "peft_adapter_name": self.peft_adapter_name,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
        }


def build_lora_classifier_training_backend_config(
    objective_config: TrainingObjectiveConfig | None,
) -> LoraClassifierTrainingBackendConfig:
    """objective config에서 LoRA-classifier trainer 설정을 읽는다."""

    if objective_config is None:
        return LoraClassifierTrainingBackendConfig()

    extras = {
        **objective_config.get_component_extras(LORA_CLASSIFIER_FAMILY_EXTRA_SCOPE),
        **objective_config.get_component_extras(
            LORA_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE
        ),
    }
    return LoraClassifierTrainingBackendConfig.from_mapping(extras)


def _set_normalized_str(
    instance: LoraClassifierTrainingBackendConfig,
    field_name: str,
    value: str,
    *,
    allow_empty: bool = False,
) -> None:
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must not be empty.")
    object.__setattr__(instance, field_name, normalized)


def _str_value(
    source: Mapping[str, TrainingConfigScalar],
    key: str,
    default: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = source.get(key, default)
    normalized = str(value).strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{key} must not be empty.")
    return normalized


def _positive_int_value(
    source: Mapping[str, TrainingConfigScalar],
    key: str,
    default: int,
) -> int:
    value = source.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must not be bool.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be positive.")
    return parsed


def _float_value(
    source: Mapping[str, TrainingConfigScalar],
    key: str,
    default: float,
) -> float:
    value = source.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must not be bool.")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1.")
    return parsed


def _bool_value(
    source: Mapping[str, TrainingConfigScalar],
    key: str,
    default: bool,
) -> bool:
    value = source.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{key} must be bool.")


def _str_tuple_value(
    source: Mapping[str, TrainingConfigScalar],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = source.get(key)
    if value is None:
        return default
    return tuple(part.strip() for part in str(value).split(",") if part.strip())
