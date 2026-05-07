"""FedAvg pseudo-label local objective seam."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FedAvgPseudoLabelLocalObjective:
    """기존 agent local training service를 쓰는 pseudo-label self-training objective."""

    objective_name: str = "pseudo_label_self_training"
    trainer_hint: str = "local_training_service"
    pseudo_labeler_hint: str = "ssl_pseudo_label_selection_hook"


FEDAVG_PSEUDO_LABEL_LOCAL_OBJECTIVE = FedAvgPseudoLabelLocalObjective()
