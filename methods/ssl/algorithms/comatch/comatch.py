"""USB CoMatch core를 TraceMind Query SSL method로 옮긴 구현."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from methods.adaptation.query_text_views.view_rows import (
    USB_WEAK_STRONG_PAIR_BUILDER_NAME,
)

from ...base import (
    QUERY_SSL_ALGORITHM_STATE_FEATURE_QUEUE,
    QUERY_SSL_ALGORITHM_STATE_PROBABILITY_QUEUE,
    QUERY_SSL_BATCH_SURFACE_WEAK_STRONG_PAIR,
    QUERY_SSL_MODEL_OUTPUT_LOGITS,
    QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
    QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
    QuerySslRequiredViews,
    QuerySslRuntimeRequirements,
    QuerySslStepContext,
    QuerySslStepResult,
    TextBatchClassifier,
)
from ...common import compute_prob
from ...hooks.consistency import CrossEntropyConsistencyLossHook
from ...hooks.distribution_alignment import QueueDistributionAlignmentHook
from ...hooks.masking import FixedThresholdMaskingHook
from ...model_capabilities import (
    FeatureReturningTextBatchClassifier,
    require_pooled_feature_classifier,
)
from ...registry import register_query_ssl_algorithm
from ...state import (
    build_query_ssl_algorithm_state,
    require_matching_int_state_value,
    require_query_ssl_algorithm_state,
)
from ..usb_consistency import validate_usb_consistency_loaders
from .memory_bank import CoMatchMemoryBank

COMATCH_REQUIRED_VIEWS = QuerySslRequiredViews(
    view_names=("text", "aug_0", "aug_1"),
    view_builder_name=USB_WEAK_STRONG_PAIR_BUILDER_NAME,
)


class CoMatchProjectionHead(nn.Module):
    """CoMatch contrastive graph regularization용 projection head."""

    def __init__(self, *, input_dim: int, proj_size: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if proj_size <= 0:
            raise ValueError("proj_size must be positive.")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=False),
            nn.Linear(input_dim, proj_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.layers(features), dim=1)


class CoMatchAlgorithm:
    """CoMatch를 공통 Query SSL trainer seam에 맞춘 algorithm adapter."""

    algorithm_name: str = "comatch"

    def __init__(
        self,
        *,
        temperature: float,
        p_cutoff: float,
        contrast_p_cutoff: float,
        queue_batch: int,
        smoothing_alpha: float,
        da_len: int,
        proj_size: int,
        lambda_u: float = 1.0,
        lambda_c: float = 1.0,
        supervised_loss_weight: float = 1.0,
    ) -> None:
        self.temperature = _require_positive_float(temperature, "temperature")
        self.p_cutoff = _require_probability_cutoff(p_cutoff, "p_cutoff")
        self.contrast_p_cutoff = _require_probability_cutoff(
            contrast_p_cutoff,
            "contrast_p_cutoff",
        )
        self.queue_batch = _require_positive_int(queue_batch, "queue_batch")
        self.smoothing_alpha = _require_unit_interval(
            smoothing_alpha,
            "smoothing_alpha",
        )
        self.da_len = _require_positive_int(da_len, "da_len")
        self.proj_size = _require_positive_int(proj_size, "proj_size")
        self.lambda_u = float(lambda_u)
        self.lambda_c = float(lambda_c)
        self.supervised_loss_weight = float(supervised_loss_weight)
        self.num_classes: int | None = None
        self.projection_head: nn.Module | None = None
        self.dist_align_hook: QueueDistributionAlignmentHook | None = None
        self.memory_bank: CoMatchMemoryBank | None = None
        self._pending_memory_bank_state: Mapping[str, Any] | None = None
        self.labeled_batch_size = 0
        self.unlabeled_batch_size = 1

    @property
    def uses_labeled_batches(self) -> bool:
        return self.supervised_loss_weight > 0

    def configure_batching(
        self,
        *,
        labeled_batch_size: int,
        unlabeled_batch_size: int,
    ) -> None:
        """USB `queue_batch` 의미에 맞춰 실제 memory row capacity 입력을 받는다."""

        if labeled_batch_size < 0:
            raise ValueError("labeled_batch_size must not be negative.")
        if unlabeled_batch_size <= 0:
            raise ValueError("unlabeled_batch_size must be positive.")
        self.labeled_batch_size = (
            int(labeled_batch_size) if self.uses_labeled_batches else 0
        )
        self.unlabeled_batch_size = int(unlabeled_batch_size)
        if (
            self.memory_bank is not None
            and self.memory_bank.queue_size != self._memory_bank_queue_size()
        ):
            raise RuntimeError(
                "CoMatch batching must be configured before memory bank creation."
            )

    def configure_dataset(
        self,
        *,
        num_classes: int,
        unlabeled_row_count: int,
    ) -> None:
        del unlabeled_row_count
        self.num_classes = _require_positive_int(num_classes, "num_classes")
        self.dist_align_hook = QueueDistributionAlignmentHook(
            num_classes=self.num_classes,
            queue_length=self.da_len,
            p_target_type="uniform",
        )

    def build_auxiliary_modules(
        self,
        *,
        model: TextBatchClassifier,
    ) -> dict[str, nn.Module]:
        classifier = require_pooled_feature_classifier(model)
        input_dim = int(getattr(classifier.classifier, "in_features", 0))
        if input_dim <= 0:
            raise ValueError("CoMatch requires classifier.in_features.")
        if self.projection_head is None:
            self.projection_head = CoMatchProjectionHead(
                input_dim=input_dim,
                proj_size=self.proj_size,
            )
        self._ensure_memory_bank(device=next(self.projection_head.parameters()).device)
        return {"projection_head": self.projection_head}

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        validate_usb_consistency_loaders(
            algorithm_name="CoMatch",
            train_loader_length=train_loader_length,
            unlabeled_loader_length=unlabeled_loader_length,
            supervised_loss_weight=self.supervised_loss_weight,
        )

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
    ) -> QuerySslStepResult:
        return self.compute_step_with_context(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            step_context=None,
        )

    def compute_step_with_context(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Tensor],
        step_context: QuerySslStepContext | None,
    ) -> QuerySslStepResult:
        if self.dist_align_hook is None:
            raise RuntimeError("CoMatch must be configured with dataset metadata.")
        if self.projection_head is None:
            raise RuntimeError("CoMatch projection head was not initialized.")
        memory_bank = self._ensure_memory_bank(
            device=next(self.projection_head.parameters()).device
        )
        return compute_comatch_step(
            model=require_pooled_feature_classifier(model),
            projection_head=self.projection_head,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            dist_align_hook=self.dist_align_hook,
            memory_bank=memory_bank,
            temperature=self.temperature,
            p_cutoff=self.p_cutoff,
            contrast_p_cutoff=self.contrast_p_cutoff,
            smoothing_alpha=self.smoothing_alpha,
            apply_memory_smoothing=self._should_apply_memory_smoothing(step_context),
            lambda_u=self.lambda_u,
            lambda_c=self.lambda_c,
            supervised_loss_weight=self.supervised_loss_weight,
        )

    def export_state(self) -> Mapping[str, Any]:
        memory_bank_state = (
            None if self.memory_bank is None else self.memory_bank.export_state()
        )
        return build_query_ssl_algorithm_state(
            algorithm_name=self.algorithm_name,
            configured=self.num_classes is not None,
            metadata={
                "num_classes": self.num_classes,
                "queue_batch": self.queue_batch,
                "queue_size": None
                if self.memory_bank is None
                else self.memory_bank.queue_size,
                "labeled_batch_size": self.labeled_batch_size,
                "unlabeled_batch_size": self.unlabeled_batch_size,
                "proj_size": self.proj_size,
                "da_len": self.da_len,
                "dist_align": None
                if self.dist_align_hook is None
                else self.dist_align_hook.export_state(),
                "memory_bank": None
                if memory_bank_state is None
                else {
                    "feature_queue": memory_bank_state.feature_queue,
                    "probability_queue": memory_bank_state.probability_queue,
                    "queue_ptr": memory_bank_state.queue_ptr,
                    "filled_size": memory_bank_state.filled_size,
                },
            },
        )

    def load_state(self, state: Mapping[str, Any]) -> None:
        effective_state = require_query_ssl_algorithm_state(
            state=state,
            algorithm_name=self.algorithm_name,
        )
        if self.num_classes is not None:
            require_matching_int_state_value(
                state=effective_state,
                field_name="num_classes",
                expected=self.num_classes,
                algorithm_name=self.algorithm_name,
            )
        require_matching_int_state_value(
            state=effective_state,
            field_name="queue_batch",
            expected=self.queue_batch,
            algorithm_name=self.algorithm_name,
        )
        require_matching_int_state_value(
            state=effective_state,
            field_name="queue_size",
            expected=self._memory_bank_queue_size(),
            algorithm_name=self.algorithm_name,
        )
        require_matching_int_state_value(
            state=effective_state,
            field_name="proj_size",
            expected=self.proj_size,
            algorithm_name=self.algorithm_name,
        )
        require_matching_int_state_value(
            state=effective_state,
            field_name="da_len",
            expected=self.da_len,
            algorithm_name=self.algorithm_name,
        )
        dist_align_state = effective_state.get("dist_align")
        if dist_align_state is not None:
            if self.dist_align_hook is None:
                raise RuntimeError("CoMatch dist_align_hook is not configured.")
            if not isinstance(dist_align_state, Mapping):
                raise ValueError("CoMatch dist_align state must be a mapping.")
            self.dist_align_hook.load_state(
                dist_align_state,
                device=torch.device("cpu"),
            )
        memory_bank_state = effective_state.get("memory_bank")
        if memory_bank_state is not None:
            if not isinstance(memory_bank_state, Mapping):
                raise ValueError("CoMatch memory_bank state must be a mapping.")
            if self.memory_bank is None:
                self._pending_memory_bank_state = memory_bank_state
            else:
                self.memory_bank.load_state(memory_bank_state)

    def _ensure_memory_bank(self, *, device: torch.device) -> CoMatchMemoryBank:
        if self.num_classes is None:
            raise RuntimeError("CoMatch requires configure_dataset before training.")
        if self.memory_bank is None:
            self.memory_bank = CoMatchMemoryBank(
                queue_size=self._memory_bank_queue_size(),
                feature_dim=self.proj_size,
                num_classes=self.num_classes,
                device=device,
            )
            if self._pending_memory_bank_state is not None:
                self.memory_bank.load_state(self._pending_memory_bank_state)
                self._pending_memory_bank_state = None
        self.memory_bank.to(device)
        return self.memory_bank

    def _memory_bank_queue_size(self) -> int:
        per_step_rows = self.unlabeled_batch_size + self.labeled_batch_size
        return self.queue_batch * max(1, per_step_rows)

    def _should_apply_memory_smoothing(
        self,
        step_context: QuerySslStepContext | None,
    ) -> bool:
        return step_context is not None and step_context.global_step > self.queue_batch


def compute_comatch_step(
    *,
    model: FeatureReturningTextBatchClassifier,
    projection_head: nn.Module,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Tensor],
    dist_align_hook: QueueDistributionAlignmentHook,
    memory_bank: CoMatchMemoryBank,
    temperature: float,
    p_cutoff: float,
    contrast_p_cutoff: float,
    smoothing_alpha: float,
    apply_memory_smoothing: bool = True,
    lambda_u: float = 1.0,
    lambda_c: float = 1.0,
    supervised_loss_weight: float = 1.0,
) -> QuerySslStepResult:
    """USB CoMatch `train_step`의 tensor-level 핵심 경로."""

    logits_x_ulb_s_0, feats_x_ulb_s_0 = _forward_logits_and_projected_features(
        model=model,
        projection_head=projection_head,
        input_ids=unlabeled_batch["strong_0_input_ids"],
        attention_mask=unlabeled_batch["strong_0_attention_mask"],
    )
    _, feats_x_ulb_s_1 = _forward_logits_and_projected_features(
        model=model,
        projection_head=projection_head,
        input_ids=unlabeled_batch["strong_1_input_ids"],
        attention_mask=unlabeled_batch["strong_1_attention_mask"],
    )
    with torch.no_grad():
        logits_x_ulb_w, feats_x_ulb_w = _forward_logits_and_projected_features(
            model=model,
            projection_head=projection_head,
            input_ids=unlabeled_batch["weak_input_ids"],
            attention_mask=unlabeled_batch["weak_attention_mask"],
        )

    sup_loss = logits_x_ulb_s_0.new_zeros(())
    labeled_features: Tensor | None = None
    labeled_probabilities: Tensor | None = None
    if labeled_batch is not None and supervised_loss_weight > 0:
        logits_x_lb, feats_x_lb = _forward_logits_and_projected_features(
            model=model,
            projection_head=projection_head,
            input_ids=labeled_batch["input_ids"],
            attention_mask=labeled_batch["attention_mask"],
        )
        labels = labeled_batch["labels"].long()
        sup_loss = F.cross_entropy(logits_x_lb, labels, reduction="mean")
        labeled_features = feats_x_lb.detach()
        labeled_probabilities = F.one_hot(
            labels,
            num_classes=logits_x_lb.shape[1],
        ).to(dtype=logits_x_lb.dtype)

    probs = compute_prob(logits_x_ulb_w.detach())
    probs = dist_align_hook.dist_align(probs_x_ulb=probs.detach())
    probs_orig = probs.detach().clone()
    memory_smoothing_applied = bool(
        apply_memory_smoothing and memory_bank.filled_size > 0
    )
    if memory_smoothing_applied:
        probs = memory_bank.smooth_probabilities(
            features=feats_x_ulb_w.detach(),
            probabilities=probs,
            temperature=temperature,
            smoothing_alpha=smoothing_alpha,
        )

    mask = FixedThresholdMaskingHook().build_mask(
        probs_x_ulb_w=probs,
        p_cutoff=p_cutoff,
    )
    _update_comatch_memory_bank(
        memory_bank=memory_bank,
        weak_features=feats_x_ulb_w.detach(),
        weak_probabilities=probs_orig,
        labeled_features=labeled_features,
        labeled_probabilities=labeled_probabilities,
    )
    unsup_loss = CrossEntropyConsistencyLossHook().compute_loss(
        logits=logits_x_ulb_s_0,
        targets=probs.detach(),
        mask=mask,
    )
    graph_targets = compute_comatch_graph_targets(
        probabilities=probs.detach(),
        contrast_p_cutoff=contrast_p_cutoff,
    )
    contrast_loss = comatch_contrastive_loss(
        feats_x_ulb_s_0=feats_x_ulb_s_0,
        feats_x_ulb_s_1=feats_x_ulb_s_1,
        graph_targets=graph_targets,
        temperature=temperature,
    )
    total_loss = (
        supervised_loss_weight * sup_loss
        + float(lambda_u) * unsup_loss
        + float(lambda_c) * contrast_loss
    )
    return QuerySslStepResult(
        total_loss=total_loss,
        loss_components={
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "contrast_loss": contrast_loss,
        },
        metrics={
            "util_ratio": mask.float().mean(),
            "memory_bank_filled_size": logits_x_ulb_s_0.new_tensor(
                float(memory_bank.filled_size)
            ),
            "memory_bank_queue_size": logits_x_ulb_s_0.new_tensor(
                float(memory_bank.queue_size)
            ),
            "memory_smoothing_applied": logits_x_ulb_s_0.new_tensor(
                float(memory_smoothing_applied)
            ),
        },
        debug_tensors={
            "mask": mask,
            "graph_targets": graph_targets,
        },
    )


def compute_comatch_graph_targets(
    *,
    probabilities: Tensor,
    contrast_p_cutoff: float,
) -> Tensor:
    """CoMatch pseudo-label graph target Q를 만든다."""

    cutoff = _require_probability_cutoff(contrast_p_cutoff, "contrast_p_cutoff")
    graph_targets = probabilities @ probabilities.t()
    graph_targets.fill_diagonal_(1.0)
    graph_targets = graph_targets * graph_targets.ge(cutoff).to(graph_targets.dtype)
    row_sums = graph_targets.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return graph_targets / row_sums


def comatch_contrastive_loss(
    *,
    feats_x_ulb_s_0: Tensor,
    feats_x_ulb_s_1: Tensor,
    graph_targets: Tensor,
    temperature: float,
) -> Tensor:
    """CoMatch graph contrastive loss."""

    temperature = _require_positive_float(temperature, "temperature")
    sim = torch.exp((feats_x_ulb_s_0 @ feats_x_ulb_s_1.t()) / temperature)
    sim_probs = sim / sim.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return -(torch.log(sim_probs + 1e-7) * graph_targets).sum(dim=1).mean()


def _forward_logits_and_projected_features(
    *,
    model: FeatureReturningTextBatchClassifier,
    projection_head: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_features = model.extract_pooled_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    projected_features = projection_head(pooled_features)
    if projected_features.ndim != 2:
        raise ValueError("CoMatch projection head must return [batch, proj_size].")
    return logits, F.normalize(projected_features, dim=1)


def _update_comatch_memory_bank(
    *,
    memory_bank: CoMatchMemoryBank,
    weak_features: Tensor,
    weak_probabilities: Tensor,
    labeled_features: Tensor | None,
    labeled_probabilities: Tensor | None,
) -> None:
    features = weak_features
    probabilities = weak_probabilities
    if labeled_features is not None and labeled_probabilities is not None:
        features = torch.cat([weak_features, labeled_features], dim=0)
        probabilities = torch.cat([weak_probabilities, labeled_probabilities], dim=0)
    memory_bank.update(features=features, probabilities=probabilities)


def _require_positive_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _require_positive_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _require_probability_cutoff(value: float, field_name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return normalized


def _require_unit_interval(value: float, field_name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return normalized


@register_query_ssl_algorithm(
    "comatch",
    display_name="CoMatch",
    required_views=COMATCH_REQUIRED_VIEWS,
    default_uses_labeled_batches=True,
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface=QUERY_SSL_BATCH_SURFACE_WEAK_STRONG_PAIR,
        model_outputs=frozenset(
            {
                QUERY_SSL_MODEL_OUTPUT_LOGITS,
                QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES,
            }
        ),
        algorithm_state_surface=frozenset(
            {
                QUERY_SSL_ALGORITHM_STATE_FEATURE_QUEUE,
                QUERY_SSL_ALGORITHM_STATE_PROBABILITY_QUEUE,
            }
        ),
        optimizer_lifecycle=frozenset(
            {
                QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP,
                QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE,
            }
        ),
        step_context_required=True,
    ),
)
def build_comatch_algorithm(parameters: Mapping[str, Any]) -> CoMatchAlgorithm:
    """Hydra method parameter mapping으로 CoMatch algorithm을 만든다."""

    return CoMatchAlgorithm(
        temperature=float(parameters["temperature"]),
        p_cutoff=float(parameters["p_cutoff"]),
        contrast_p_cutoff=float(parameters["contrast_p_cutoff"]),
        queue_batch=int(parameters["queue_batch"]),
        smoothing_alpha=float(parameters["smoothing_alpha"]),
        da_len=int(parameters["da_len"]),
        proj_size=int(parameters["proj_size"]),
        lambda_u=float(parameters.get("lambda_u", 1.0)),
        lambda_c=float(
            parameters.get("lambda_c", parameters.get("contrast_loss_ratio", 1.0))
        ),
        supervised_loss_weight=float(parameters.get("supervised_loss_weight", 1.0)),
    )
