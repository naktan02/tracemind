"""SSL objective hook bundle."""

from __future__ import annotations

from dataclasses import dataclass

from methods.ssl.hooks.consistency import ConsistencyLossHook
from methods.ssl.hooks.masking import MaskingHook
from methods.ssl.hooks.pseudo_labeling import PseudoLabelingHook


@dataclass(frozen=True, slots=True)
class SslObjectiveHooks:
    """algorithm이 교체 가능한 tensor-level SSL hook 묶음."""

    pseudo_labeling: PseudoLabelingHook
    masking: MaskingHook
    consistency_loss: ConsistencyLossHook
