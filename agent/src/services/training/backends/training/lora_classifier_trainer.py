"""LoRA-classifier training backend compatibility facade."""

from __future__ import annotations

from agent.src.services.training.backends.training.lora_classifier import (
    backend as _backend,
)
from agent.src.services.training.backends.training.lora_classifier import (
    config as _config,
)
from agent.src.services.training.backends.training.lora_classifier import (
    train_executor as _train_executor,
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
LoraClassifierTrainArtifacts = _train_executor.LoraClassifierTrainArtifacts
LoraClassifierTrainExecutor = _train_executor.LoraClassifierTrainExecutor
NotImplementedLoraClassifierTrainExecutor = (
    _train_executor.NotImplementedLoraClassifierTrainExecutor
)
