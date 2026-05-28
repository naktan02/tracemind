"""PEFT encoder classifier pseudo-label 품질 진단 helper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.training.loops import (
    move_tensor_batch_to_device,
)
from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftEncoderTextClassifier,
)
from methods.adaptation.query_text_views.data import build_weak_dataloader
from methods.adaptation.query_text_views.tokenization import (
    TextTokenizationCache,
)
from methods.evaluation.pseudo_label_quality import (
    PseudoLabelCandidateRecord,
    PseudoLabelQualitySummary,
    build_pseudo_label_quality_summary,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


class PeftEncoderDiagnosticsRuntimeConfig(Protocol):
    """pseudo-label diagnostics가 필요한 runtime surface."""

    device: str


@dataclass(frozen=True, slots=True)
class PseudoLabelDiagnosticThreshold:
    """최종 snapshot pseudo-label 진단에 쓰는 fixed threshold metadata."""

    threshold: float | None

    @property
    def uses_fixed_threshold(self) -> bool:
        return self.threshold is not None

    def to_client_metrics(self) -> dict[str, float]:
        """client metric에는 숫자형 fixed-threshold 진단 여부만 남긴다."""

        metrics = {
            "query_ssl_diagnostic_uses_fixed_threshold": float(
                self.uses_fixed_threshold
            )
        }
        if self.threshold is not None:
            metrics["query_ssl_diagnostic_acceptance_threshold"] = float(self.threshold)
        return metrics


def resolve_fixed_pseudo_label_diagnostic_threshold(
    parameters: Mapping[str, object],
) -> PseudoLabelDiagnosticThreshold:
    """학습 후 pseudo-label 품질 진단용 fixed threshold를 해석한다.

    adaptive/classwise threshold method에서는 이 값이 실제 학습 mask와 다를 수 있다.
    그래서 여기서는 `p_cutoff`가 있는 경우에만 fixed-threshold snapshot diagnostic으로
    기록한다.
    """

    raw_value = parameters.get("p_cutoff")
    if raw_value is None:
        return PseudoLabelDiagnosticThreshold(threshold=None)
    return PseudoLabelDiagnosticThreshold(threshold=float(raw_value))


def build_final_snapshot_pseudo_label_quality(
    *,
    model: PeftEncoderTextClassifier,
    tokenizer: Any,
    rows: Sequence[LabeledQueryRow],
    labels: Sequence[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    acceptance_threshold: float | None,
    trainer_runtime_config: PeftEncoderDiagnosticsRuntimeConfig,
    unlabeled_batch_size: int,
    tokenization_cache: TextTokenizationCache | None,
    tokenization_cache_namespace: str,
) -> PseudoLabelQualitySummary:
    """round 종료 시점 local classifier snapshot의 pseudo-label 품질을 계산한다."""

    effective_rows = list(rows)
    if not effective_rows:
        return PseudoLabelQualitySummary.empty()

    loader = build_weak_dataloader(
        rows=effective_rows,
        tokenizer=tokenizer,
        batch_size=int(unlabeled_batch_size),
        max_length=peft_config.max_length,
        task_prefix=peft_config.task_prefix,
        shuffle=False,
        tokenization_cache=tokenization_cache,
        tokenization_cache_namespace=tokenization_cache_namespace,
    )
    candidates: list[PseudoLabelCandidateRecord] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            device_batch = move_tensor_batch_to_device(
                batch=batch,
                device=trainer_runtime_config.device,
            )
            probabilities = torch.softmax(
                model(
                    input_ids=device_batch["weak_input_ids"],
                    attention_mask=device_batch["weak_attention_mask"],
                ),
                dim=-1,
            )
            top_k = min(2, probabilities.shape[-1])
            top_values, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
            query_ids = [str(query_id) for query_id in batch["query_ids"]]
            for row_index, query_id in enumerate(query_ids):
                top1_index = int(top_indices[row_index, 0].detach().cpu().item())
                top1_score = float(top_values[row_index, 0].detach().cpu().item())
                top2_score = (
                    float(top_values[row_index, 1].detach().cpu().item())
                    if top_k > 1
                    else 0.0
                )
                candidates.append(
                    PseudoLabelCandidateRecord(
                        source_event_ref=query_id,
                        label=str(labels[top1_index]),
                        confidence=top1_score,
                        margin=top1_score - top2_score,
                        accepted=(
                            acceptance_threshold is not None
                            and top1_score >= acceptance_threshold
                        ),
                    )
                )

    return build_pseudo_label_quality_summary(
        candidates=tuple(candidates),
        rows_with_simulation_labels=effective_rows,
    )


def tokenization_cache_namespace(
    peft_config: PeftEncoderTrainingBackendConfig,
) -> str:
    """LoRA classifier tokenizer 설정을 cache namespace로 정규화한다."""

    return (
        f"tokenizer={peft_config.tokenizer_model_id}"
        f"|revision={peft_config.tokenizer_revision}"
    )
