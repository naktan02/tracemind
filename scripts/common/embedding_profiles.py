"""스크립트용 임베딩 실행 프로필과 공통 CLI helper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)


@dataclass(slots=True, frozen=True)
class EmbeddingProfilePreset:
    """스크립트 실행용 임베딩 preset."""

    name: str
    backend: str
    model_id: str
    revision: str
    batch_size: int = 16
    cache_dir: Path = Path("hf_cache")
    device: str = "auto"
    task_prefix: str = ""
    local_files_only: bool = False
    hash_dim: int = 256


@dataclass(slots=True, frozen=True)
class ResolvedEmbeddingSettings:
    """profile과 override를 병합한 최종 임베딩 설정."""

    profile: str
    backend: str
    model_id: str
    revision: str
    batch_size: int
    cache_dir: Path
    device: str
    task_prefix: str
    local_files_only: bool
    hash_dim: int

    def to_spec(self) -> EmbeddingAdapterSpec:
        return EmbeddingAdapterSpec(
            backend=self.backend,
            model_id=self.model_id,
            revision=self.revision,
            device=self.device,
            batch_size=self.batch_size,
            cache_dir=str(self.cache_dir),
            task_prefix=self.task_prefix,
            hash_dim=self.hash_dim,
            local_files_only=self.local_files_only,
        )


EMBEDDING_PROFILES: dict[str, EmbeddingProfilePreset] = {
    "mxbai": EmbeddingProfilePreset(
        name="mxbai",
        backend="transformers_mxbai",
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        revision="main",
    ),
    "hash_debug": EmbeddingProfilePreset(
        name="hash_debug",
        backend="hash_debug",
        model_id="hash_debug",
        revision="debug",
        hash_dim=256,
    ),
}


def supported_embedding_profiles() -> tuple[str, ...]:
    return tuple(sorted(EMBEDDING_PROFILES))


def infer_embedding_profile(
    *,
    backend: str | None = None,
    model_id: str | None = None,
    fallback: str = "mxbai",
) -> str:
    if backend is not None:
        for preset in EMBEDDING_PROFILES.values():
            if preset.backend == backend and (
                model_id is None or preset.model_id == model_id
            ):
                return preset.name
        for preset in EMBEDDING_PROFILES.values():
            if preset.backend == backend:
                return preset.name

    if model_id is not None:
        for preset in EMBEDDING_PROFILES.values():
            if preset.model_id == model_id:
                return preset.name

    return fallback


def resolve_embedding_settings(
    *,
    profile: str,
    backend: str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    batch_size: int | None = None,
    cache_dir: Path | str | None = None,
    device: str | None = None,
    task_prefix: str | None = None,
    local_files_only: bool | None = None,
    hash_dim: int | None = None,
) -> ResolvedEmbeddingSettings:
    if profile not in EMBEDDING_PROFILES:
        raise ValueError(
            f"지원되지 않는 embedding profile: '{profile}'. "
            f"지원 목록: {supported_embedding_profiles()}"
        )

    preset = EMBEDDING_PROFILES[profile]
    resolved_cache_dir = (
        Path(cache_dir) if cache_dir is not None else Path(preset.cache_dir)
    )
    resolved_settings = ResolvedEmbeddingSettings(
        profile=profile,
        backend=backend if backend is not None else preset.backend,
        model_id=model_id if model_id is not None else preset.model_id,
        revision=revision if revision is not None else preset.revision,
        batch_size=batch_size if batch_size is not None else preset.batch_size,
        cache_dir=resolved_cache_dir,
        device=device if device is not None else preset.device,
        task_prefix=task_prefix if task_prefix is not None else preset.task_prefix,
        local_files_only=(
            local_files_only
            if local_files_only is not None
            else preset.local_files_only
        ),
        hash_dim=hash_dim if hash_dim is not None else preset.hash_dim,
    )

    if resolved_settings.backend not in EmbeddingAdapterFactory.supported_backends():
        raise ValueError(
            f"지원되지 않는 embedding backend: '{resolved_settings.backend}'. "
            f"지원 목록: {EmbeddingAdapterFactory.supported_backends()}"
        )
    return resolved_settings


def add_embedding_profile_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_profile: str,
    default_device: str | None = None,
) -> None:
    parser.add_argument(
        "--embedding-profile",
        choices=supported_embedding_profiles(),
        default=default_profile,
        help=(
            "Embedding preset 이름. "
            "backend/model/revision 기본값을 함께 채운다."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=EmbeddingAdapterFactory.supported_backends(),
        default=None,
        help="Embedding profile의 backend를 덮어쓴다.",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=None,
        help="Embedding profile의 model_id를 덮어쓴다.",
    )
    parser.add_argument(
        "--embedding-model-revision",
        default=None,
        help="Embedding profile의 revision을 덮어쓴다.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding profile의 batch_size를 덮어쓴다.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Embedding profile의 cache_dir를 덮어쓴다.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help=(
            "Embedding profile의 device를 덮어쓴다 "
            "(auto/cpu/cuda/cuda:0/mps)."
        ),
    )
    parser.add_argument(
        "--task-prefix",
        default=None,
        help="Embedding profile의 task_prefix를 덮어쓴다.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=None,
        help="Embedding profile의 local_files_only를 true로 덮어쓴다.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=None,
        help="Embedding profile의 hash_dim을 덮어쓴다.",
    )


def resolve_embedding_settings_from_args(
    args: argparse.Namespace,
    *,
    model_id_override: str | None = None,
    revision_override: str | None = None,
) -> ResolvedEmbeddingSettings:
    return resolve_embedding_settings(
        profile=args.embedding_profile,
        backend=args.backend,
        model_id=(
            model_id_override
            if model_id_override is not None
            else args.embedding_model_id
        ),
        revision=(
            revision_override
            if revision_override is not None
            else args.embedding_model_revision
        ),
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        device=args.device,
        task_prefix=args.task_prefix,
        local_files_only=args.local_files_only,
        hash_dim=args.hash_dim,
    )
