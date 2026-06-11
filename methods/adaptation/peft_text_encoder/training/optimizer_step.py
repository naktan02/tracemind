"""PEFT text encoder/head optimizer step compatibility surface."""

from __future__ import annotations

from methods.adaptation.common import optimizer_step as _common_optimizer_step

run_optimizer_loss_step = _common_optimizer_step.run_optimizer_loss_step
