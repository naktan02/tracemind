"""Compatibility shim for LoRA-classifier delta extraction imports."""

from __future__ import annotations

from methods.adaptation.lora_classifier.training import delta_extraction as _core

extract_classifier_head_deltas = _core.extract_classifier_head_deltas
extract_lora_classifier_parameter_deltas = (
    _core.extract_lora_classifier_parameter_deltas
)
extract_lora_parameter_deltas = _core.extract_lora_parameter_deltas
finite_float_or_none = _core.finite_float_or_none
is_number = _core.is_number
load_lora_classifier_base_parameters_into_model = (
    _core.load_lora_classifier_base_parameters_into_model
)
lora_classifier_delta_l2_norm = _core.lora_classifier_delta_l2_norm
