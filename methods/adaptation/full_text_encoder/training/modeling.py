"""Full text encoder + linear head supervised 모델 빌더."""

from __future__ import annotations

from typing import Any

from methods.adaptation.text_encoder_classifier.modeling import (
    TextEncoderWithLinearHead,
    count_parameters,
    load_classifier_head_state_if_configured,
    load_transformer_backbone,
    load_transformer_tokenizer,
    require_transformer_auto_stack,
    set_module_trainable,
)
from methods.adaptation.text_encoder_classifier.modeling import (
    build_trainable_surface_manifest as build_text_encoder_trainable_surface_manifest,
)

_SUPPORTED_TRAINABLE_SURFACES = frozenset({"full_text_encoder"})


def build_trainable_surface_manifest(cfg: Any) -> dict[str, object]:
    """Hydra trainable_surface 축을 full text encoder manifest로 정규화한다."""

    return build_text_encoder_trainable_surface_manifest(
        cfg,
        supported_surface_names=_SUPPORTED_TRAINABLE_SURFACES,
    )


def build_model(
    *,
    cfg,
    categories: list[str],
    device: str,
) -> tuple[TextEncoderWithLinearHead, Any, dict[str, Any]]:
    """Hydra config 기준으로 full text encoder/head scaffold를 조립한다."""

    trainable_surface_manifest = build_trainable_surface_manifest(cfg)
    initial_adapter_dir = str(getattr(cfg, "initial_adapter_dir", "") or "").strip()
    if initial_adapter_dir:
        raise ValueError(
            "full_text_encoder supervised control does not accept initial_adapter_dir. "
            "Use the PEFT supervised control for adapter warm-start ablations."
        )
    initial_classifier_path = str(
        getattr(cfg, "initial_classifier_path", "") or ""
    ).strip()

    AutoModel, AutoTokenizer = require_transformer_auto_stack()
    tokenizer = load_transformer_tokenizer(
        tokenizer_cls=AutoTokenizer,
        model_id=str(cfg.paper_backbone.tokenizer_model_id),
        revision=str(cfg.paper_backbone.tokenizer_revision),
        cache_dir=str(cfg.paper_backbone.cache_dir),
        local_files_only=bool(cfg.runtime.local_files_only),
        trust_remote_code=bool(cfg.paper_backbone.trust_remote_code),
    )
    backbone = load_transformer_backbone(
        model_cls=AutoModel,
        model_id=str(cfg.paper_backbone.model_id),
        revision=str(cfg.paper_backbone.revision),
        cache_dir=str(cfg.paper_backbone.cache_dir),
        local_files_only=bool(cfg.runtime.local_files_only),
        trust_remote_code=bool(cfg.paper_backbone.trust_remote_code),
    )
    set_module_trainable(backbone, trainable=True)
    model = TextEncoderWithLinearHead(
        backbone=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(categories),
        classifier_dropout=float(cfg.paper_backbone.classifier_dropout),
    ).to(device)
    load_classifier_head_state_if_configured(
        model=model,
        categories=categories,
        classifier_path=initial_classifier_path,
    )

    summary = {
        "backbone_model_id": str(cfg.paper_backbone.model_id),
        "backbone_revision": str(cfg.paper_backbone.revision),
        "tokenizer_model_id": str(cfg.paper_backbone.tokenizer_model_id),
        "tokenizer_revision": str(cfg.paper_backbone.tokenizer_revision),
        "pooling": str(cfg.paper_backbone.pooling),
        "max_length": int(cfg.paper_backbone.max_length),
        "task_prefix": str(cfg.paper_backbone.task_prefix),
        "trainable_surface": trainable_surface_manifest,
        "full_text_encoder_config": {
            "encoder_trainable": True,
            "classifier_head_trainable": True,
        },
        "initial_checkpoint": {
            "classifier_path": initial_classifier_path or None,
        },
        "parameter_counts": count_parameters(model),
    }
    return model, tokenizer, summary
