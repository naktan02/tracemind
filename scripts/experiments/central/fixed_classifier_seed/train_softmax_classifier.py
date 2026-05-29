"""고정 임베딩 위에 linear classifier head + softmax를 학습하고 평가한다."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.central.fixed_classifier_seed.runner import (
    run_fixed_embedding_classifier,
)


@hydra.main(
    version_base=None,
    config_path="../../../../conf",
    config_name="entrypoints/central/fixed_classifier_seed/train_softmax_classifier",
)
def main(cfg: DictConfig) -> None:
    run_fixed_embedding_classifier(cfg=cfg)


if __name__ == "__main__":
    main()
