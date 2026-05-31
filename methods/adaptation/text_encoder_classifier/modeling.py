"""Text encoder + linear classifier 공통 모델 helper."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
from torch import nn


class TextEncoderWithLinearHead(nn.Module):
    """Transformer text encoder와 linear classifier head 묶음."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int,
        classifier_dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
    ):
        pooled = self.extract_pooled_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self.dropout(pooled).to(self.classifier.weight.dtype)
        return self.classifier(pooled)

    def extract_pooled_features(
        self,
        *,
        input_ids,
        attention_mask,
    ):
        """분포 시각화용 dropout 전 pooled backbone representation을 반환한다."""

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.to(self.classifier.weight.dtype)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """총 파라미터와 학습 가능한 파라미터 수를 반환한다."""

    total = 0
    trainable = 0
    for parameter in model.parameters():
        size = parameter.numel()
        total += size
        if parameter.requires_grad:
            trainable += size
    return {"total": total, "trainable": trainable}


def build_trainable_surface_manifest(
    cfg: Any,
    *,
    supported_surface_names: Collection[str],
) -> dict[str, object]:
    """Hydra trainable_surface 축을 trainer manifest로 정규화한다."""

    surface_cfg = getattr(cfg, "trainable_surface", None)
    if surface_cfg is None:
        raise ValueError(
            "trainable_surface config is required. Select "
            "strategy_axes/model_architecture/trainable_surface=<surface>."
        )
    surface_name = _required_str(
        getattr(surface_cfg, "name", ""),
        field_name="trainable_surface.name",
    )
    if surface_name not in supported_surface_names:
        raise ValueError(
            "Unsupported trainable_surface.name for text encoder classifier "
            f"trainer: {surface_name!r}. Supported: {sorted(supported_surface_names)}."
        )
    return {
        "name": surface_name,
        "model_artifact_kind": _required_str(
            getattr(surface_cfg, "model_artifact_kind", ""),
            field_name="trainable_surface.model_artifact_kind",
        ),
        "trainable_state": _required_str(
            getattr(surface_cfg, "trainable_state", ""),
            field_name="trainable_surface.trainable_state",
        ),
        "requires_peft_adapter": bool(
            getattr(surface_cfg, "requires_peft_adapter", False)
        ),
        "supports_initial_adapter": bool(
            getattr(surface_cfg, "supports_initial_adapter", False)
        ),
    }


def set_module_trainable(module: nn.Module, *, trainable: bool) -> None:
    """module 아래 모든 parameter의 requires_grad 값을 맞춘다."""

    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def require_transformer_auto_stack() -> tuple[Any, Any]:
    """실험 extra가 설치돼 있을 때만 transformers stack을 연다."""

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "Text encoder classifier experiments require the experiments extra with "
            "transformers installed. Example: uv sync --extra experiments"
        ) from exc
    return AutoModel, AutoTokenizer


def load_transformer_tokenizer(
    *,
    tokenizer_cls: Any,
    model_id: str,
    revision: str,
    cache_dir: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    """공통 tokenizer loading 규칙을 적용한다."""

    tokenizer = tokenizer_cls.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_transformer_backbone(
    *,
    model_cls: Any,
    model_id: str,
    revision: str,
    cache_dir: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> nn.Module:
    """공통 transformer backbone loading 규칙을 적용한다."""

    return model_cls.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def load_classifier_head_state_if_configured(
    *,
    model: TextEncoderWithLinearHead,
    categories: list[str],
    classifier_path: str | Path,
) -> None:
    """classifier head checkpoint가 지정된 경우 label order를 검증하고 적재한다."""

    normalized_path = str(classifier_path or "").strip()
    if not normalized_path:
        return
    classifier_bundle = torch.load(normalized_path, map_location="cpu")
    checkpoint_categories = [
        str(category) for category in classifier_bundle["categories"]
    ]
    if checkpoint_categories != categories:
        raise ValueError(
            "initial classifier categories do not match current categories: "
            f"{checkpoint_categories} != {categories}"
        )
    model.classifier.load_state_dict(classifier_bundle["classifier_state_dict"])


def _required_str(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized
