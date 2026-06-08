"""Agent 기본 inference pipeline 조립."""

from __future__ import annotations

import os
from collections.abc import Mapping

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from agent.src.services.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.services.inference.embedding_service import EmbeddingService
from agent.src.services.inference.pipeline_service import InferencePipelineService
from agent.src.services.inference.scoring_service import ScoringService
from agent.src.services.language.translation_service import TranslationService
from shared.src.contracts.scoring_contracts import ScoringConfigPayload
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec

AGENT_EMBEDDING_BACKEND_ENV = "TRACEMIND_AGENT_EMBEDDING_BACKEND"
AGENT_EMBEDDING_MODEL_ID_ENV = "TRACEMIND_AGENT_EMBEDDING_MODEL_ID"
AGENT_EMBEDDING_REVISION_ENV = "TRACEMIND_AGENT_EMBEDDING_REVISION"
AGENT_EMBEDDING_DEVICE_ENV = "TRACEMIND_AGENT_EMBEDDING_DEVICE"
AGENT_EMBEDDING_BATCH_SIZE_ENV = "TRACEMIND_AGENT_EMBEDDING_BATCH_SIZE"
AGENT_EMBEDDING_CACHE_DIR_ENV = "TRACEMIND_AGENT_EMBEDDING_CACHE_DIR"
AGENT_EMBEDDING_LOCAL_FILES_ONLY_ENV = "TRACEMIND_AGENT_EMBEDDING_LOCAL_FILES_ONLY"
AGENT_EMBEDDING_HASH_DIM_ENV = "TRACEMIND_AGENT_EMBEDDING_HASH_DIM"
AGENT_SCORING_BACKEND_ENV = "TRACEMIND_AGENT_SCORING_BACKEND"


def build_default_pipeline_service(
    *,
    analysis_event_repository: AnalysisEventRepository,
    shared_adapter_runtime_service: SharedAdapterRuntimeService,
    translation_service: TranslationService | None,
) -> InferencePipelineService:
    """agent runtime 기본 inference pipeline을 조립한다."""

    embedding_spec = load_agent_embedding_spec_from_env()
    scoring_backend_name = _required_env_value(os.environ, AGENT_SCORING_BACKEND_ENV)
    return InferencePipelineService(
        embedding_service=EmbeddingService(
            adapter=EmbeddingAdapterFactory.create(embedding_spec)
        ),
        scoring_service=ScoringService.from_scoring_config(
            ScoringConfigPayload(
                scorer_backend_name=scoring_backend_name,
            )
        ),
        event_repository=analysis_event_repository,
        shared_adapter_provider=shared_adapter_runtime_service,
        translation_service=translation_service,
        embedding_model_id=embedding_spec.model_id,
        model_revision="agent_local_runtime",
    )


def load_agent_embedding_spec_from_env(
    environ: Mapping[str, str] | None = None,
) -> EmbeddingAdapterSpec:
    """환경변수에서 agent inference embedding adapter 설정을 읽는다."""

    effective_environ = os.environ if environ is None else environ
    return EmbeddingAdapterSpec(
        backend=_env_value(
            effective_environ,
            AGENT_EMBEDDING_BACKEND_ENV,
            "transformers_mxbai",
        ),
        model_id=_env_value(
            effective_environ,
            AGENT_EMBEDDING_MODEL_ID_ENV,
            "mixedbread-ai/mxbai-embed-large-v1",
        ),
        revision=_env_value(effective_environ, AGENT_EMBEDDING_REVISION_ENV, "main"),
        device=_env_value(effective_environ, AGENT_EMBEDDING_DEVICE_ENV, "auto"),
        batch_size=_env_int(effective_environ, AGENT_EMBEDDING_BATCH_SIZE_ENV, 16),
        cache_dir=_env_optional_value(effective_environ, AGENT_EMBEDDING_CACHE_DIR_ENV),
        hash_dim=_env_int(effective_environ, AGENT_EMBEDDING_HASH_DIM_ENV, 256),
        local_files_only=_env_bool(
            effective_environ,
            AGENT_EMBEDDING_LOCAL_FILES_ONLY_ENV,
            False,
        ),
    )


def _env_value(environ: Mapping[str, str], key: str, default: str) -> str:
    value = environ.get(key, "").strip()
    return value or default


def _env_optional_value(environ: Mapping[str, str], key: str) -> str | None:
    value = environ.get(key, "").strip()
    return value or None


def _required_env_value(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required.")
    return value


def _env_int(environ: Mapping[str, str], key: str, default: int) -> int:
    value = environ.get(key, "").strip()
    return int(value) if value else default


def _env_bool(environ: Mapping[str, str], key: str, default: bool) -> bool:
    value = environ.get(key, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be boolean-like.")
