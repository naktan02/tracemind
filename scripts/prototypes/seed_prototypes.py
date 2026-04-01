"""prototype pack 생성 Hydra entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from scripts.prototypes.seeding import seed_prototype_pack  # noqa: E402


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="prototypes/seed_prototypes",
)
def main(cfg: DictConfig) -> None:
    built_at = datetime.now(timezone.utc)
    prototype_version = cfg.prototype_version or built_at.strftime(
        "proto_%Y_%m_%d_%H%M%S"
    )
    embedding_spec = instantiate(cfg.embedding.spec)
    build_strategy = instantiate(cfg.prototype_builder)
    (
        pack_path,
        build_state_path,
        manifest_path,
        main_server_pack_path,
        main_server_build_state_path,
    ) = seed_prototype_pack(
        input_jsonl=Path(cfg.input_jsonl),
        output_dir=Path(cfg.output_dir),
        build_state_output_dir=Path(cfg.build_state_output_dir),
        prototype_version=prototype_version,
        backend=embedding_spec.backend,
        embedding_model_id=embedding_spec.model_id,
        embedding_model_revision=embedding_spec.revision,
        translation_model_id=cfg.translation_model_id,
        translation_model_revision=cfg.translation_model_revision,
        translation_direction=cfg.translation_direction,
        batch_size=embedding_spec.batch_size,
        cache_dir=Path(embedding_spec.cache_dir or "hf_cache"),
        device=embedding_spec.device,
        task_prefix=embedding_spec.task_prefix,
        local_files_only=embedding_spec.local_files_only,
        expected_categories=list(cfg.expected_categories),
        hash_dim=embedding_spec.hash_dim,
        build_strategy=build_strategy,
    )
    print(f"prototype_build_state={build_state_path}")
    print(f"prototype_pack={pack_path}")
    print(f"manifest={manifest_path}")
    print(f"main_server_build_state={main_server_build_state_path}")
    print(f"main_server_pack={main_server_pack_path}")


if __name__ == "__main__":
    main()
