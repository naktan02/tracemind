"""LoRA + classifier scaffold 모델 빌더."""

from __future__ import annotations

from typing import Any, Protocol

from torch import nn

from methods.adaptation.lora.lora_adapter import resolve_target_modules
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.peft.base import PeftAdapterBuildContext
from methods.adaptation.peft.registry import (
    build_peft_adapter_builder,
    resolve_peft_adapter_name,
)


def require_transformer_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    """실험 extra가 설치돼 있을 때만 transformer/peft stack을 연다."""

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "LoRA classifier experiments require the experiments extra with "
            "transformers and peft installed. Example: uv sync --extra experiments"
        ) from exc
    return AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model, PeftModel


class LoraClassifierModelRuntimeConfig(Protocol):
    """LoRA-classifier 모델 생성에 필요한 runtime surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


class LoraTextClassifier(nn.Module):
    """Frozen backbone + LoRA + classifier head 묶음."""

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


def build_lora_text_classifier_from_config(
    *,
    labels: list[str],
    lora_config: LoraClassifierTrainingBackendConfig,
    runtime_config: LoraClassifierModelRuntimeConfig,
) -> tuple[LoraTextClassifier, Any]:
    """LoRA-classifier config snapshot에서 평가/학습용 모델을 조립한다."""

    AutoModel, AutoTokenizer, LoraConfig, TaskType, get_peft_model, _PeftModel = (
        require_transformer_stack()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.tokenizer_model_id,
        revision=lora_config.tokenizer_revision,
        cache_dir=runtime_config.cache_dir,
        local_files_only=runtime_config.local_files_only,
        trust_remote_code=runtime_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    backbone_base = AutoModel.from_pretrained(
        lora_config.backbone_model_id,
        revision=lora_config.backbone_revision,
        cache_dir=runtime_config.cache_dir,
        local_files_only=runtime_config.local_files_only,
        trust_remote_code=runtime_config.trust_remote_code,
    )
    peft_config = LoraConfig(
        r=int(lora_config.rank),
        lora_alpha=int(lora_config.alpha),
        lora_dropout=float(lora_config.dropout),
        target_modules=resolve_target_modules(lora_config.target_modules),
        bias=lora_config.bias,
        use_rslora=bool(lora_config.use_rslora),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone_base, peft_config)
    model = LoraTextClassifier(
        backbone=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(labels),
        classifier_dropout=float(runtime_config.classifier_dropout),
    ).to(runtime_config.device)
    return model, tokenizer


def build_model(
    *,
    cfg,
    categories: list[str],
    device: str,
) -> tuple[LoraTextClassifier, Any, dict[str, Any]]:
    """Hydra config 기준으로 LoRA + classifier scaffold를 조립한다."""

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
    model = LoraTextClassifier(
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
        "lora": {
            **(
                peft_adapter_builder.build_summary(cfg=cfg)
                if peft_adapter_builder is not None
                else {
                    "adapter_name": "loaded_from_checkpoint",
                    "rank": int(cfg.lora.rank),
                    "alpha": int(cfg.lora.alpha),
                    "dropout": float(cfg.lora.dropout),
                    "bias": str(cfg.lora.bias),
                    "target_modules": str(cfg.lora.target_modules),
                    "use_rslora": bool(cfg.lora.use_rslora),
                }
            ),
        },
        "initial_checkpoint": {
            "adapter_dir": initial_adapter_dir or None,
            "classifier_path": initial_classifier_path or None,
        },
        "parameter_counts": count_parameters(model),
    }
    return model, tokenizer, summary
