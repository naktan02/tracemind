"""공용 contract 모듈."""

from .adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
    VectorAdapterDeltaPayload,
    VectorAdapterStatePayload,
    make_diagonal_delta_payload,
    make_identity_state_payload,
)
from .fl_round_contracts import ActiveRoundPayload
from .model_contracts import ModelManifestPayload, make_embedding_manifest
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
    make_training_update_envelope,
)

__all__ = [
    "ActiveRoundPayload",
    "DecisionFeedbackSignalPayload",
    "CategoryPrototypePayload",
    "DiagonalScaleAdapterStatePayload",
    "DiagonalScaleAdapterUpdatePayload",
    "extract_category_centroids",
    "extract_category_prototypes",
    "make_diagonal_delta_payload",
    "make_embedding_manifest",
    "make_identity_state_payload",
    "make_training_update_envelope",
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
