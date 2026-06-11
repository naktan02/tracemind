"""Query SSL runtime schedule helper."""

from __future__ import annotations


def compute_linear_warmup(
    *,
    iteration: int,
    warm_up_ratio: float,
    num_train_iter: int,
) -> float:
    """USB `np.clip(it / (warm_up_ratio * num_train_iter), 0, 1)` 수식."""

    if iteration < 0:
        raise ValueError("iteration must not be negative.")
    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")
    if warm_up_ratio <= 0:
        return 1.0
    denominator = float(warm_up_ratio) * float(num_train_iter)
    return max(0.0, min(1.0, float(iteration) / denominator))
