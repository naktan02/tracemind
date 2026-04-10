"""Training domain entities."""

from shared.src.contracts.adapter_contracts import (
    VectorAdapterDelta,
    VectorAdapterState,
)
from shared.src.contracts.training_contracts import (
    DecisionFeedbackSignal,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
    TrainingUpdateEnvelope,
)

from .pseudo_label_candidate import PseudoLabelCandidate
from .pseudo_label_evidence import PSEUDO_LABEL_EVIDENCE_V1, PseudoLabelEvidence
from .shared_adapter_state import IdentitySharedAdapterState, SharedAdapterState
from .shared_adapter_update import SharedAdapterUpdate

__all__ = [
    "DecisionFeedbackSignal",
    "IdentitySharedAdapterState",
    "PseudoLabelCandidate",
    "PSEUDO_LABEL_EVIDENCE_V1",
    "PseudoLabelEvidence",
    "SharedAdapterState",
    "SharedAdapterUpdate",
    "TrainingConfigScalar",
    "TrainingObjectiveConfig",
    "TrainingSelectionPolicy",
    "TrainingTask",
    "TrainingUpdateEnvelope",
    "VectorAdapterDelta",
    "VectorAdapterState",
]
