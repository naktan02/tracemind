"""LoRA-classifier training backend compatibility facade."""

from __future__ import annotations

from agent.src.services.training.backends.training.lora_classifier import (
    backend as _backend,
)
from agent.src.services.training.backends.training.lora_classifier import (
    config as _config,
)
from methods.adaptation.lora_classifier import (
    local_update as _local_update,
)

LoraClassifierTrainingBackend = _backend.LoraClassifierTrainingBackend
LoraClassifierTrainingBackendConfig = _config.LoraClassifierTrainingBackendConfig
LORA_CLASSIFIER_TRAINING_BACKEND_NAME = _config.LORA_CLASSIFIER_TRAINING_BACKEND_NAME
LORA_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE = (
    _config.LORA_CLASSIFIER_TRAINING_BACKEND_EXTRA_SCOPE
)
LORA_CLASSIFIER_FAMILY_EXTRA_SCOPE = _config.LORA_CLASSIFIER_FAMILY_EXTRA_SCOPE
build_lora_classifier_training_backend_config = (
    _config.build_lora_classifier_training_backend_config
)
LoraClassifierTrainArtifacts = _local_update.LoraClassifierTrainArtifacts
LoraClassifierTrainExecutor = _local_update.LoraClassifierTrainExecutor
NotImplementedLoraClassifierTrainExecutor = (
    _local_update.NotImplementedLoraClassifierTrainExecutor
)
