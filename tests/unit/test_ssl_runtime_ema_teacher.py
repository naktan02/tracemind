"""Query SSL EMA teacher runtime tests."""

from __future__ import annotations

import torch
from torch import nn

from methods.ssl.runtime.ema import EmaTrainableParameterTeacher
from methods.ssl.runtime.schedules import compute_linear_warmup


def test_ema_teacher_swaps_and_restores_trainable_parameters() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    teacher = EmaTrainableParameterTeacher(model=model, momentum=0.5)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[4.0, 6.0]]))

    with teacher.use_shadow_weights(model):
        assert torch.allclose(model.weight, torch.tensor([[1.0, 2.0]]))

    assert torch.allclose(model.weight, torch.tensor([[4.0, 6.0]]))


def test_ema_teacher_updates_shadow_with_momentum() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    teacher = EmaTrainableParameterTeacher(model=model, momentum=0.5)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[5.0, 8.0]]))

    teacher.update(model)

    state = teacher.state_dict()
    assert torch.allclose(state.shadow["weight"], torch.tensor([[3.0, 5.0]]))


def test_ema_teacher_state_roundtrips() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    teacher = EmaTrainableParameterTeacher(model=model, momentum=0.5)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[5.0, 8.0]]))
    teacher.update(model)
    exported = teacher.state_dict()

    restored = EmaTrainableParameterTeacher(model=model, momentum=0.5)
    restored.load_state_dict(
        shadow=exported.shadow,
        parameter_names=exported.parameter_names,
    )

    assert torch.allclose(
        restored.state_dict().shadow["weight"],
        torch.tensor([[3.0, 5.0]]),
    )


def test_compute_linear_warmup_matches_usb_clip_formula() -> None:
    assert (
        compute_linear_warmup(
            iteration=2,
            warm_up_ratio=0.4,
            num_train_iter=10,
        )
        == 0.5
    )
    assert (
        compute_linear_warmup(
            iteration=10,
            warm_up_ratio=0.4,
            num_train_iter=10,
        )
        == 1.0
    )
