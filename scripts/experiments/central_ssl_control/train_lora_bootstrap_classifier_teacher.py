"""Fixed classifier teacher -> LoRA student bootstrap entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.query_lora_ssl.bootstrap_runner import (
    run_fixed_classifier_teacher_lora_student_bootstrap,
)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/central_ssl_control/train_lora_bootstrap_classifier_teacher",
)
def main(cfg: DictConfig) -> None:
    run_fixed_classifier_teacher_lora_student_bootstrap(cfg=cfg)


if __name__ == "__main__":
    main()
