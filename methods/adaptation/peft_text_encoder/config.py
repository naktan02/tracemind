"""PEFT text encoder training backend config parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from methods.common.config_reading import (
    read_bool,
    read_float,
    read_positive_int,
    read_str,
    read_str_tuple,
    read_unit_interval_float,
    set_normalized_str,
    validate_allowed_keys,
)
from shared.src.contracts.training_contracts import (
    TrainingConfigScalar,
    TrainingObjectiveConfig,
)

PEFT_ENCODER_TRAINING_BACKEND_NAME = "peft_classifier_trainer"
PEFT_ENCODER_TRAINING_BACKEND_EXTRA_SCOPE = "peft_classifier_trainer"
PEFT_ENCODER_PAYLOAD_EXTRA_SCOPE = "peft_classifier"
PEFT_ENCODER_PAYLOAD_ADAPTER_KIND = "peft_classifier"
PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL = "agent_local_artifact_ref"
PEFT_ENCODER_DELTA_FORMAT_INLINE = "inline_delta"
PEFT_ENCODER_DELTA_FORMAT_SERVER_UPLOADED = "server_uploaded_artifact_ref"

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
_NON_NEGATIVE_FLOAT_CONFIG_KEYS = ("proximal_mu",)
_BOOL_CONFIG_KEYS = ("use_rslora",)
_STRING_TUPLE_CONFIG_KEYS = ("text_metadata_keys", "label_schema")
_CONFIG_EXTRA_KEYS = frozenset(
    {
        *_STRING_CONFIG_KEYS,
        *_ALLOW_EMPTY_STRING_CONFIG_KEYS,
        *_POSITIVE_INT_CONFIG_KEYS,
        *_FLOAT_CONFIG_KEYS,
        *_NON_NEGATIVE_FLOAT_CONFIG_KEYS,
        *_BOOL_CONFIG_KEYS,
        *_STRING_TUPLE_CONFIG_KEYS,
    }
)


@dataclass(frozen=True, slots=True)
class PeftEncoderTrainingBackendConfig:
    """PEFT text encoder update payload에 기록할 trainer snapshot."""

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
    proximal_mu: float = 0.0
    bias: str = "none"
    target_modules: str = "all-linear"
    use_rslora: bool = False
    delta_format: str = PEFT_ENCODER_DELTA_FORMAT_AGENT_LOCAL
    artifact_ref_prefix: str = "agent-local://peft_classifier"
    text_metadata_keys: tuple[str, ...] = (
        "strong_text",
        "training_text",
        "raw_text",
        "text",
        "weak_text",
    )
    label_schema: tuple[str, ...] = ()
    payload_adapter_kind: str = PEFT_ENCODER_PAYLOAD_ADAPTER_KIND

    def __post_init__(self) -> None:
        set_normalized_str(self, "backbone_model_id", self.backbone_model_id)
        set_normalized_str(self, "backbone_revision", self.backbone_revision)
        set_normalized_str(self, "tokenizer_model_id", self.tokenizer_model_id)
        set_normalized_str(self, "tokenizer_revision", self.tokenizer_revision)
        set_normalized_str(self, "pooling", self.pooling)
        set_normalized_str(
            self,
            "task_prefix",
            self.task_prefix,
            allow_empty=True,
        )
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        set_normalized_str(self, "peft_adapter_name", self.peft_adapter_name)
        if self.rank <= 0:
            raise ValueError("rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")
        if self.proximal_mu < 0.0:
            raise ValueError("proximal_mu must be non-negative.")
        set_normalized_str(self, "bias", self.bias)
        set_normalized_str(self, "target_modules", self.target_modules)
        set_normalized_str(self, "delta_format", self.delta_format)
        set_normalized_str(self, "artifact_ref_prefix", self.artifact_ref_prefix)
        set_normalized_str(self, "payload_adapter_kind", self.payload_adapter_kind)
        if self.payload_adapter_kind != PEFT_ENCODER_PAYLOAD_ADAPTER_KIND:
            raise ValueError("payload_adapter_kind must be peft_classifier.")
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
    ) -> "PeftEncoderTrainingBackendConfig":
        """Scoped objective extras에서 backend config를 만든다."""

        if not source:
            return cls()

        validate_allowed_keys(
            source,
            allowed_keys=_CONFIG_EXTRA_KEYS,
            config_name="PEFT encoder training backend config",
        )

        defaults = cls()
        values: dict[str, object] = {}
        for key in _STRING_CONFIG_KEYS:
            values[key] = read_str(source, key, getattr(defaults, key))
        for key in _ALLOW_EMPTY_STRING_CONFIG_KEYS:
            values[key] = read_str(
                source,
                key,
                getattr(defaults, key),
                allow_empty=True,
            )
        for key in _POSITIVE_INT_CONFIG_KEYS:
            values[key] = read_positive_int(source, key, getattr(defaults, key))
        for key in _FLOAT_CONFIG_KEYS:
            values[key] = read_unit_interval_float(source, key, getattr(defaults, key))
        for key in _NON_NEGATIVE_FLOAT_CONFIG_KEYS:
            values[key] = read_float(source, key, getattr(defaults, key))
        for key in _BOOL_CONFIG_KEYS:
            values[key] = read_bool(source, key, getattr(defaults, key))
        for key in _STRING_TUPLE_CONFIG_KEYS:
            values[key] = read_str_tuple(source, key, getattr(defaults, key))
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
        """LoRA-compatible shared payload config snapshot."""

        return {
            "peft_adapter_name": self.peft_adapter_name,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
        }

    def to_peft_adapter_config_payload(self) -> dict[str, object]:
        """v2 PEFT payload에 들어갈 mechanism-neutral config snapshot."""

        return {
            "peft_adapter_name": self.peft_adapter_name,
            "parameters": self.to_lora_config_payload(),
        }


def build_peft_encoder_training_backend_config(
    objective_config: TrainingObjectiveConfig | None,
) -> PeftEncoderTrainingBackendConfig:
    """objective config에서 PEFT text encoder trainer 설정을 읽는다."""

    if objective_config is None:
        config = PeftEncoderTrainingBackendConfig()
    else:
        extras = {
            **objective_config.get_component_extras(PEFT_ENCODER_PAYLOAD_EXTRA_SCOPE),
            **objective_config.get_component_extras(
                PEFT_ENCODER_TRAINING_BACKEND_EXTRA_SCOPE
            ),
        }
        config = PeftEncoderTrainingBackendConfig.from_mapping(extras)
    return replace(
        config,
        payload_adapter_kind=PEFT_ENCODER_PAYLOAD_ADAPTER_KIND,
    )
