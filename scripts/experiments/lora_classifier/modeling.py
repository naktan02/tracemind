"""LoRA classifier 실험용 모델 빌더."""

from __future__ import annotations

from typing import Any

from torch import nn


def require_transformer_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "train_lora_classifier requires the experiments extra with "
            "transformers and peft installed. Example: uv sync --extra experiments"
        ) from exc
    return AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model


class LoraTextClassifier(nn.Module):
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
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled).to(self.classifier.weight.dtype)
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        size = parameter.numel()
        total += size
        if parameter.requires_grad:
            trainable += size
    return {"total": total, "trainable": trainable}


def _resolve_target_modules(raw_value: Any) -> str | list[str]:
    if isinstance(raw_value, str):
        return raw_value
    return [str(value) for value in raw_value]


def build_model(
    *,
    cfg,
    categories: list[str],
    device: str,
) -> tuple[LoraTextClassifier, Any, dict[str, Any]]:
    AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model = (
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

    backbone = AutoModel.from_pretrained(
        str(cfg.paper_backbone.model_id),
        revision=str(cfg.paper_backbone.revision),
        cache_dir=str(cfg.paper_backbone.cache_dir),
        local_files_only=bool(cfg.runtime.local_files_only),
        trust_remote_code=bool(cfg.paper_backbone.trust_remote_code),
    )

    lora_config = LoraConfig(
        r=int(cfg.lora.rank),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=_resolve_target_modules(cfg.lora.target_modules),
        bias=str(cfg.lora.bias),
        use_rslora=bool(cfg.lora.use_rslora),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone, lora_config)
    hidden_size = int(backbone.config.hidden_size)
    model = LoraTextClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=len(categories),
        classifier_dropout=float(cfg.paper_backbone.classifier_dropout),
    ).to(device)

    summary = {
        "backbone_model_id": str(cfg.paper_backbone.model_id),
        "backbone_revision": str(cfg.paper_backbone.revision),
        "tokenizer_model_id": str(cfg.paper_backbone.tokenizer_model_id),
        "tokenizer_revision": str(cfg.paper_backbone.tokenizer_revision),
        "pooling": str(cfg.paper_backbone.pooling),
        "max_length": int(cfg.paper_backbone.max_length),
        "task_prefix": str(cfg.paper_backbone.task_prefix),
        "lora": {
            "rank": int(cfg.lora.rank),
            "alpha": int(cfg.lora.alpha),
            "dropout": float(cfg.lora.dropout),
            "bias": str(cfg.lora.bias),
            "target_modules": _resolve_target_modules(cfg.lora.target_modules),
            "use_rslora": bool(cfg.lora.use_rslora),
        },
        "parameter_counts": count_parameters(model),
    }
    return model, tokenizer, summary
