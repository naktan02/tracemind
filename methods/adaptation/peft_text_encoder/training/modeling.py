"""PEFT text encoder/head scaffold 모델 빌더."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Protocol

from torch import nn

from methods.adaptation.peft_adapters.base import PeftAdapterBuildContext
from methods.adaptation.peft_adapters.registry import (
    build_peft_adapter_builder,
    peft_adapter_config_from_cfg,
    resolve_peft_adapter_name,
)
from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.resource_cache import (
    peft_encoder_resource_cache_key,
)
from methods.common.runtime_resources import RuntimeResourceCache

_SUPPORTED_TRAINABLE_SURFACES = frozenset({"peft_text_encoder"})


def require_transformer_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    """실험 extra가 설치돼 있을 때만 transformer/peft stack을 연다."""

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "PEFT text encoder/head experiments require the experiments extra with "
            "transformers and peft installed. Example: uv sync --extra experiments"
        ) from exc
    return AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model, PeftModel


class PeftEncoderModelRuntimeConfig(Protocol):
    """PEFT text encoder/head 모델 생성에 필요한 runtime surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


class PeftTextEncoderWithLinearHead(nn.Module):
    """Frozen backbone + PEFT adapter + linear head 묶음."""

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


def build_trainable_surface_manifest(cfg: Any) -> dict[str, object]:
    """Hydra trainable_surface 축을 중앙 trainer manifest로 정규화한다."""

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
    if surface_name not in _SUPPORTED_TRAINABLE_SURFACES:
        raise ValueError(
            "Unsupported trainable_surface.name for PEFT text encoder trainer: "
            f"{surface_name!r}. Supported: {sorted(_SUPPORTED_TRAINABLE_SURFACES)}."
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


def build_peft_text_encoder_with_linear_head_from_config(
    *,
    labels: list[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: PeftEncoderModelRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None = None,
) -> tuple[PeftTextEncoderWithLinearHead, Any]:
    """PEFT encoder config snapshot에서 평가/학습용 모델을 조립한다."""

    AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model, _PeftModel = (
        require_transformer_stack()
    )
    tokenizer = _load_tokenizer(
        tokenizer_cls=AutoTokenizer,
        peft_config=peft_config,
        runtime_config=runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )

    backbone_base = _load_backbone_base(
        model_cls=AutoModel,
        peft_config=peft_config,
        runtime_config=runtime_config,
        runtime_resource_cache=runtime_resource_cache,
    )
    peft_adapter_builder = build_peft_adapter_builder(peft_config.peft_adapter_name)
    backbone = peft_adapter_builder.build_backbone(
        backbone_base=backbone_base,
        context=PeftAdapterBuildContext(
            cfg=_peft_adapter_cfg_from_training_config(peft_config),
            lora_config_cls=LoraConfig,
            task_type=TaskType,
            get_peft_model=get_peft_model,
        ),
    )
    model = PeftTextEncoderWithLinearHead(
        backbone=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(labels),
        classifier_dropout=float(runtime_config.classifier_dropout),
    ).to(runtime_config.device)
    return model, tokenizer


def _load_tokenizer(
    *,
    tokenizer_cls: Any,
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: PeftEncoderModelRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
) -> Any:
    key = _runtime_resource_key(
        kind="tokenizer",
        values={
            "model_id": peft_config.tokenizer_model_id,
            "revision": peft_config.tokenizer_revision,
            "cache_dir": runtime_config.cache_dir,
            "local_files_only": runtime_config.local_files_only,
            "trust_remote_code": runtime_config.trust_remote_code,
        },
    )
    if runtime_resource_cache is not None:
        cached = runtime_resource_cache.get_resource(key)
        if cached is not None:
            return cached
    tokenizer = tokenizer_cls.from_pretrained(
        peft_config.tokenizer_model_id,
        revision=peft_config.tokenizer_revision,
        cache_dir=runtime_config.cache_dir,
        local_files_only=runtime_config.local_files_only,
        trust_remote_code=runtime_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if runtime_resource_cache is not None:
        runtime_resource_cache.set_resource(key, tokenizer)
    return tokenizer


def _load_backbone_base(
    *,
    model_cls: Any,
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: PeftEncoderModelRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None,
) -> nn.Module:
    key = _runtime_resource_key(
        kind="backbone_base",
        values={
            "model_id": peft_config.backbone_model_id,
            "revision": peft_config.backbone_revision,
            "cache_dir": runtime_config.cache_dir,
            "local_files_only": runtime_config.local_files_only,
            "trust_remote_code": runtime_config.trust_remote_code,
        },
    )
    if runtime_resource_cache is not None:
        cached = runtime_resource_cache.get_resource(key)
        if cached is not None:
            if not isinstance(cached, nn.Module):
                raise TypeError(
                    "Cached PEFT text encoder/head backbone must be nn.Module."
                )
            return copy.deepcopy(cached)
    backbone_base = model_cls.from_pretrained(
        peft_config.backbone_model_id,
        revision=peft_config.backbone_revision,
        cache_dir=runtime_config.cache_dir,
        local_files_only=runtime_config.local_files_only,
        trust_remote_code=runtime_config.trust_remote_code,
    )
    if runtime_resource_cache is not None:
        runtime_resource_cache.set_resource(key, backbone_base)
        return copy.deepcopy(backbone_base)
    return backbone_base


def _runtime_resource_key(
    *,
    kind: str,
    values: dict[str, object],
) -> str:
    return peft_encoder_resource_cache_key(kind=kind, values=values)


def _peft_adapter_cfg_from_training_config(
    peft_config: PeftEncoderTrainingBackendConfig,
) -> SimpleNamespace:
    """PEFT adapter builder가 기대하는 config surface로 trainer config를 맞춘다."""

    return SimpleNamespace(
        peft_adapter=SimpleNamespace(
            peft_adapter_name=peft_config.peft_adapter_name,
            rank=peft_config.rank,
            alpha=peft_config.alpha,
            dropout=peft_config.dropout,
            target_modules=peft_config.target_modules,
            bias=peft_config.bias,
            use_rslora=peft_config.use_rslora,
        )
    )


def build_model(
    *,
    cfg,
    categories: list[str],
    device: str,
) -> tuple[PeftTextEncoderWithLinearHead, Any, dict[str, Any]]:
    """Hydra config 기준으로 PEFT text encoder/head scaffold를 조립한다."""

    trainable_surface_manifest = build_trainable_surface_manifest(cfg)
    AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model, PeftModel = (
        require_transformer_stack()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.paper_backbone.tokenizer_model_id),
        revision=str(cfg.paper_backbone.tokenizer_revision),
        cache_dir=str(cfg.paper_backbone.cache_dir),
        local_files_only=bool(cfg.runtime.local_files_only),
        trust_remote_code=bool(cfg.paper_backbone.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    initial_adapter_dir = str(getattr(cfg, "initial_adapter_dir", "") or "").strip()
    initial_classifier_path = str(
        getattr(cfg, "initial_classifier_path", "") or ""
    ).strip()

    backbone_base = AutoModel.from_pretrained(
        str(cfg.paper_backbone.model_id),
        revision=str(cfg.paper_backbone.revision),
        cache_dir=str(cfg.paper_backbone.cache_dir),
        local_files_only=bool(cfg.runtime.local_files_only),
        trust_remote_code=bool(cfg.paper_backbone.trust_remote_code),
    )

    if initial_adapter_dir:
        backbone = PeftModel.from_pretrained(
            backbone_base,
            initial_adapter_dir,
            is_trainable=True,
        )
        peft_adapter_builder = None
    else:
        peft_adapter_name = resolve_peft_adapter_name(cfg)
        peft_adapter_builder = build_peft_adapter_builder(peft_adapter_name)
        backbone = peft_adapter_builder.build_backbone(
            backbone_base=backbone_base,
            context=PeftAdapterBuildContext(
                cfg=cfg,
                lora_config_cls=LoraConfig,
                task_type=TaskType,
                get_peft_model=get_peft_model,
            ),
        )
    hidden_size = int(backbone.config.hidden_size)
    model = PeftTextEncoderWithLinearHead(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=len(categories),
        classifier_dropout=float(cfg.paper_backbone.classifier_dropout),
    ).to(device)
    if initial_classifier_path:
        import torch

        classifier_bundle = torch.load(initial_classifier_path, map_location="cpu")
        checkpoint_categories = [
            str(category) for category in classifier_bundle["categories"]
        ]
        if checkpoint_categories != categories:
            raise ValueError(
                "initial classifier categories do not match current categories: "
                f"{checkpoint_categories} != {categories}"
            )
        model.classifier.load_state_dict(classifier_bundle["classifier_state_dict"])

    summary = {
        "backbone_model_id": str(cfg.paper_backbone.model_id),
        "backbone_revision": str(cfg.paper_backbone.revision),
        "tokenizer_model_id": str(cfg.paper_backbone.tokenizer_model_id),
        "tokenizer_revision": str(cfg.paper_backbone.tokenizer_revision),
        "pooling": str(cfg.paper_backbone.pooling),
        "max_length": int(cfg.paper_backbone.max_length),
        "task_prefix": str(cfg.paper_backbone.task_prefix),
        "trainable_surface": trainable_surface_manifest,
        "peft_adapter_config": {
            **(
                peft_adapter_builder.build_summary(cfg=cfg)
                if peft_adapter_builder is not None
                else _loaded_checkpoint_peft_adapter_summary(cfg)
            ),
        },
        "initial_checkpoint": {
            "adapter_dir": initial_adapter_dir or None,
            "classifier_path": initial_classifier_path or None,
        },
        "parameter_counts": count_parameters(model),
    }
    return model, tokenizer, summary


def _required_str(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _loaded_checkpoint_peft_adapter_summary(cfg: Any) -> dict[str, object]:
    peft_adapter = peft_adapter_config_from_cfg(cfg)
    return {
        "adapter_name": "loaded_from_checkpoint",
        "rank": int(peft_adapter.rank),
        "alpha": int(peft_adapter.alpha),
        "dropout": float(peft_adapter.dropout),
        "bias": str(peft_adapter.bias),
        "target_modules": str(peft_adapter.target_modules),
        "use_rslora": bool(peft_adapter.use_rslora),
    }
