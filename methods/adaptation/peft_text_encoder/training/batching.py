"""PEFT text encoder/head training batch iteration compatibility surface."""

from __future__ import annotations

from methods.adaptation.common import batching as _common_batching

next_cycling_batch = _common_batching.next_cycling_batch
move_tensor_batch_to_device = _common_batching.move_tensor_batch_to_device
_is_cuda_device = _common_batching._is_cuda_device
