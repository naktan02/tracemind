"""Agent training service package."""

from .acceptance_policies import (
    AcceptanceDecision,
    PseudoLabelAcceptancePolicy,
    Top1ConfidenceOnlyAcceptancePolicy,
    Top1MarginThresholdAcceptancePolicy,
    build_pseudo_label_acceptance_policy,
)
from .evidence_backends import (
    FixMatchWeakViewEvidenceBackend,
    PrototypeSimilarityEvidenceBackend,
    PseudoLabelEvidenceBackend,
    build_pseudo_label_evidence_backend,
)
from .evidence_service import PseudoLabelEvidenceService
from .local_training_service import (
    LocalTrainingResult,
    LocalTrainingService,
)
from .privacy_guard_service import (
    DiagonalScaleClipOnlyPrivacyGuard,
    NoOpSharedAdapterPrivacyGuard,
    PrivacyProtectedUpdate,
    SharedAdapterPrivacyGuard,
    build_shared_adapter_privacy_guard,
)
from .pseudo_label_service import (
    PseudoLabelSelectionResult,
    PseudoLabelSelectionService,
)
from .runtime_compatibility import (
    LocalTrainingRuntimeCompatibility,
    validate_local_training_runtime,
)
from .training_backends import (
    ClassifierHeadFixMatchConsistencyTrainingBackend,
    DiagonalScaleHeuristicTrainingBackend,
    SharedAdapterTrainingBackend,
    SyntheticVectorAdapterTrainingBackend,
    TrainingBackend,
    build_shared_adapter_training_backend,
)
from .training_example_models import EmbeddedTrainingExample

__all__ = [
    "AcceptanceDecision",
    "ClassifierHeadFixMatchConsistencyTrainingBackend",
    "DiagonalScaleHeuristicTrainingBackend",
    "EmbeddedTrainingExample",
    "FixMatchWeakViewEvidenceBackend",
    "LocalTrainingResult",
    "LocalTrainingService",
    "DiagonalScaleClipOnlyPrivacyGuard",
    "LocalTrainingRuntimeCompatibility",
    "NoOpSharedAdapterPrivacyGuard",
    "PrivacyProtectedUpdate",
    "PseudoLabelAcceptancePolicy",
    "PseudoLabelEvidenceBackend",
    "PseudoLabelEvidenceService",
    "PseudoLabelSelectionResult",
    "PseudoLabelSelectionService",
    "PrototypeSimilarityEvidenceBackend",
    "SharedAdapterPrivacyGuard",
    "SharedAdapterTrainingBackend",
    "SyntheticVectorAdapterTrainingBackend",
    "Top1ConfidenceOnlyAcceptancePolicy",
    "Top1MarginThresholdAcceptancePolicy",
    "TrainingBackend",
    "build_pseudo_label_acceptance_policy",
    "build_pseudo_label_evidence_backend",
    "build_shared_adapter_privacy_guard",
    "build_shared_adapter_training_backend",
    "validate_local_training_runtime",
]
