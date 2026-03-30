"""공용 contract 모듈."""

from .adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
    VectorAdapterDeltaPayload,
    VectorAdapterStatePayload,
)
from .model_contracts import ModelManifestPayload
from .personalization_contracts import PersonalizationStatePayload
from .prototype_contracts import (
    CategoryPrototypePayload,
    PrototypePackPayload,
    extract_category_centroids,
    extract_category_prototypes,
)
from .training_contracts import (
    DecisionFeedbackSignalPayload,
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    TrainingUpdateEnvelopePayload,
)

__all__ = [
    "DecisionFeedbackSignalPayload",
    "CategoryPrototypePayload",
    "DiagonalScaleAdapterStatePayload",
    "DiagonalScaleAdapterUpdatePayload",
    "extract_category_centroids",
    "extract_category_prototypes",
    "ModelManifestPayload",
    "PersonalizationStatePayload",
    "PrototypePackPayload",
    "SharedAdapterStatePayload",
    "SharedAdapterUpdatePayload",
    "TrainingObjectiveConfigPayload",
    "TrainingSelectionPolicyPayload",
    "TrainingTaskPayload",
    "TrainingUpdateEnvelopePayload",
    "VectorAdapterDeltaPayload",
    "VectorAdapterStatePayload",
]
