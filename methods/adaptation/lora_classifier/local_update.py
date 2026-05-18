"""Compatibility shim for LoRA-classifier update core imports."""

from __future__ import annotations

from methods.adaptation.lora_classifier.update import local_update as _core

LoraClassifierTrainArtifacts = _core.LoraClassifierTrainArtifacts
LoraClassifierTrainExecutor = _core.LoraClassifierTrainExecutor
LoraClassifierTrainingRow = _core.LoraClassifierTrainingRow
LoraClassifierUpdateConfig = _core.LoraClassifierUpdateConfig
NotImplementedLoraClassifierTrainExecutor = (
    _core.NotImplementedLoraClassifierTrainExecutor
)
build_lora_classifier_delta_from_rows = _core.build_lora_classifier_delta_from_rows
resolve_lora_classifier_label_schema = _core.resolve_lora_classifier_label_schema
