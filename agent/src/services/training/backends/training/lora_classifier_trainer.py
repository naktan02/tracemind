"""LoRA-classifier local training backend scaffold."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.src.config.adapter_family_metadata import LORA_CLASSIFIER_FAMILY_METADATA
from shared.src.contracts.adapter_contracts import (
    LoraClassifierDelta,
    SharedAdapterUpdatePayload,
    make_lora_classifier_delta_payload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingTask,
)
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)

from .base import AcceptedTrainingExample

LORA_CLASSIFIER_TRAINING_BACKEND_NAME = "lora_classifier_trainer"
LORA_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE = "lora_classifier_trainer"
LORA_CLASSIFIER_FAMILY_EXTRA_SCOPE = "lora_classifier"


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
        return cls(
            backbone_model_id=_str_value(
                source,
                "backbone_model_id",
                defaults.backbone_model_id,
            ),
            backbone_revision=_str_value(
                source,
                "backbone_revision",
                defaults.backbone_revision,
            ),
            tokenizer_model_id=_str_value(
                source,
                "tokenizer_model_id",
                defaults.tokenizer_model_id,
            ),
            tokenizer_revision=_str_value(
                source,
                "tokenizer_revision",
                defaults.tokenizer_revision,
            ),
            pooling=_str_value(source, "pooling", defaults.pooling),
            max_length=_positive_int_value(
                source,
                "max_length",
                defaults.max_length,
            ),
            task_prefix=_str_value(
                source,
                "task_prefix",
                defaults.task_prefix,
                allow_empty=True,
            ),
            peft_adapter_name=_str_value(
                source,
                "peft_adapter_name",
                defaults.peft_adapter_name,
            ),
            rank=_positive_int_value(source, "rank", defaults.rank),
            alpha=_positive_int_value(source, "alpha", defaults.alpha),
            dropout=_float_value(source, "dropout", defaults.dropout),
            bias=_str_value(source, "bias", defaults.bias),
            target_modules=_str_value(
                source,
                "target_modules",
                defaults.target_modules,
            ),
            use_rslora=_bool_value(source, "use_rslora", defaults.use_rslora),
            delta_format=_str_value(
                source,
                "delta_format",
                defaults.delta_format,
            ),
            artifact_ref_prefix=_str_value(
                source,
                "artifact_ref_prefix",
                defaults.artifact_ref_prefix,
            ),
            text_metadata_keys=_str_tuple_value(
                source,
                "text_metadata_keys",
                defaults.text_metadata_keys,
            ),
            label_schema=_str_tuple_value(
                source,
                "label_schema",
                defaults.label_schema,
            ),
        )

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


@dataclass(frozen=True, slots=True)
class _LoraClassifierTrainingRow:
    """Payload 생성 전에만 쓰는 agent-local raw-text 학습 row."""

    text: str
    label: str
    confidence: float
    margin: float


@dataclass(slots=True)
class LoraClassifierTrainingBackend:
    """raw text accepted example을 LoRA-classifier update payload로 바꾼다.

    이 backend는 raw text를 shared payload에 넣지 않는다. 현재 4단계에서는
    실제 LoRA weight 파일을 생성하는 executor를 붙이기 전 계약-compatible
    artifact ref를 남기며, 이후 PEFT 실행기는 같은 backend 경계 뒤에 연결한다.
    """

    backend_name: str = LORA_CLASSIFIER_TRAINING_BACKEND_NAME
    payload_format: str = (
        LORA_CLASSIFIER_FAMILY_METADATA.canonical_update_payload_format
    )
    adapter_kind: str = LORA_CLASSIFIER_FAMILY_METADATA.adapter_kind
    config: LoraClassifierTrainingBackendConfig = field(
        default_factory=LoraClassifierTrainingBackendConfig
    )

    @classmethod
    def from_objective_config(
        cls,
        objective_config: TrainingObjectiveConfig | None,
    ) -> "LoraClassifierTrainingBackend":
        return cls(
            config=build_lora_classifier_training_backend_config(objective_config)
        )

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple[AcceptedTrainingExample, ...],
        created_at: datetime,
    ) -> LoraClassifierDelta:
        rows = tuple(
            _build_training_row(example=example, config=self.config)
            for example in accepted_examples
        )
        if not rows:
            raise ValueError("accepted_examples must not be empty.")

        label_schema = _resolve_label_schema(
            accepted_examples=accepted_examples,
            rows=rows,
            configured_labels=self.config.label_schema,
        )
        label_counts = Counter(row.label for row in rows)
        base_artifact_ref = _build_base_artifact_ref(
            prefix=self.config.artifact_ref_prefix,
            training_task=training_task,
            created_at=created_at,
        )

        return make_lora_classifier_delta_payload(
            model_id=model_manifest.model_id,
            base_model_revision=model_manifest.model_revision,
            training_scope=training_task.training_scope,
            backbone=self.config.to_backbone_payload(),
            lora_config=self.config.to_lora_config_payload(),
            label_schema=label_schema,
            example_count=len(rows),
            lora_delta_artifact_ref=f"{base_artifact_ref}/lora_delta",
            classifier_head_delta_artifact_ref=(
                f"{base_artifact_ref}/classifier_head_delta"
            ),
            delta_format=self.config.delta_format,
            mean_confidence=sum(row.confidence for row in rows) / len(rows),
            mean_margin=sum(row.margin for row in rows) / len(rows),
            label_counts=dict(sorted(label_counts.items())),
            delta_l2_norm=0.0,
            created_at=created_at,
        )

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload:
        if not isinstance(update, LoraClassifierDelta):
            raise TypeError(
                "LoraClassifierTrainingBackend expects LoraClassifierDelta "
                f"for payload conversion, got {type(update)!r}."
            )
        return update

    def build_client_metrics(self, update: SharedAdapterUpdate) -> dict[str, float]:
        if not isinstance(update, LoraClassifierDelta):
            raise TypeError(
                "LoraClassifierTrainingBackend expects LoraClassifierDelta "
                f"for metric extraction, got {type(update)!r}."
            )
        return {
            ClientMetricKeys.MEAN_CONFIDENCE: update.mean_confidence or 0.0,
            ClientMetricKeys.MEAN_MARGIN: update.mean_margin or 0.0,
            ClientMetricKeys.DELTA_L2_NORM: update.l2_norm(),
            "lora_training_rows": float(update.example_count),
            "lora_label_schema_size": float(len(update.label_schema)),
        }

    def matches_objective_config(
        self,
        objective_config: TrainingObjectiveConfig | None,
    ) -> bool:
        return self.config == build_lora_classifier_training_backend_config(
            objective_config
        )


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


def _build_training_row(
    *,
    example: AcceptedTrainingExample,
    config: LoraClassifierTrainingBackendConfig,
) -> _LoraClassifierTrainingRow:
    candidate = example.candidate
    if candidate is None:
        raise ValueError("Accepted example must carry a pseudo-label candidate.")
    text = _extract_training_text(example=example, config=config)
    label = candidate.label.strip()
    if not label:
        raise ValueError("Pseudo-label candidate label must not be empty.")
    return _LoraClassifierTrainingRow(
        text=text,
        label=label,
        confidence=float(candidate.confidence),
        margin=float(candidate.margin),
    )


def _extract_training_text(
    *,
    example: AcceptedTrainingExample,
    config: LoraClassifierTrainingBackendConfig,
) -> str:
    metadata = getattr(example, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in config.text_metadata_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    translated_text = getattr(example.update_scored_event, "translated_text", None)
    if isinstance(translated_text, str) and translated_text.strip():
        return translated_text.strip()

    raise ValueError(
        "lora_classifier_trainer requires raw text or translated text on accepted "
        "examples. The fixed-embedding-only training path cannot produce LoRA "
        "classifier updates."
    )


def _resolve_label_schema(
    *,
    accepted_examples: Sequence[AcceptedTrainingExample],
    rows: Sequence[_LoraClassifierTrainingRow],
    configured_labels: Sequence[str],
) -> tuple[str, ...]:
    labels = tuple(configured_labels) or tuple(
        sorted(
            {
                label
                for example in accepted_examples
                for label in example.update_scored_event.category_scores
                if str(label).strip()
            }
        )
    )
    if not labels:
        labels = tuple(sorted({row.label for row in rows}))

    missing_labels = sorted({row.label for row in rows} - set(labels))
    if missing_labels:
        raise ValueError(
            "LoRA-classifier label_schema must include accepted labels: "
            f"{missing_labels}."
        )
    return labels


def _build_base_artifact_ref(
    *,
    prefix: str,
    training_task: TrainingTask,
    created_at: datetime,
) -> str:
    timestamp = _utc_timestamp(created_at)
    return "/".join(
        (
            prefix.rstrip("/"),
            _slug_ref_part(training_task.round_id),
            _slug_ref_part(training_task.task_id),
            timestamp,
        )
    )


def _utc_timestamp(value: datetime) -> str:
    effective = (
        value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    )
    return effective.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _slug_ref_part(value: str) -> str:
    normalized = value.strip().replace("/", "_")
    if not normalized:
        raise ValueError("artifact ref path parts must not be empty.")
    return normalized


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


_CONFIG_EXTRA_KEYS = frozenset(
    {
        "backbone_model_id",
        "backbone_revision",
        "tokenizer_model_id",
        "tokenizer_revision",
        "pooling",
        "max_length",
        "task_prefix",
        "peft_adapter_name",
        "rank",
        "alpha",
        "dropout",
        "bias",
        "target_modules",
        "use_rslora",
        "delta_format",
        "artifact_ref_prefix",
        "text_metadata_keys",
        "label_schema",
    }
)
