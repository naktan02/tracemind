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
from .shared_adapter_state import IdentitySharedAdapterState, SharedAdapterState
from .shared_adapter_update import SharedAdapterUpdate

__all__ = [
    "DecisionFeedbackSignal",
    "IdentitySharedAdapterState",
    "PseudoLabelCandidate",
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
