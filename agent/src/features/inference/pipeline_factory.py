"""Agent 기본 inference pipeline 조립."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import httpx

from agent.src.features.assets.shared_adapters.runtime_service import (
    SharedAdapterRuntimeService,
)
from agent.src.features.federation.rounds.artifact_client import RoundArtifactClient
from agent.src.features.inference.embedding_service import EmbeddingService
from agent.src.features.inference.pipeline_service import InferencePipelineService
from agent.src.features.inference.scoring_service import ScoringService
from agent.src.features.language.translation_service import TranslationService
from agent.src.features.runtime_profile.repository import RuntimeProfileRepository
from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from methods.adaptation.peft_text_encoder.scoring import (
    PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME,
    build_peft_classifier_head_scoring_assets,
)
from methods.adaptation.text_encoder_classifier.classifier_head_tensor_artifact import (
    parse_classifier_head_state_tensor_artifact,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierAdapterStatePayload,
)
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
    runtime_profile_repository: RuntimeProfileRepository | None = None,
) -> InferencePipelineService:
    """agent runtime 기본 inference pipeline을 조립한다."""

    embedding_spec = load_agent_embedding_spec_from_env()
    scoring_backend_name = _required_env_value(os.environ, AGENT_SCORING_BACKEND_ENV)
    return _build_pipeline_service(
        analysis_event_repository=analysis_event_repository,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        translation_service=translation_service,
        runtime_profile_repository=runtime_profile_repository,
        embedding_spec=embedding_spec,
        scoring_backend_name=scoring_backend_name,
        model_revision="agent_local_runtime",
    )


def build_pipeline_service_from_runtime_profile(
    *,
    runtime_profile_repository: RuntimeProfileRepository,
    analysis_event_repository: AnalysisEventRepository,
    shared_adapter_runtime_service: SharedAdapterRuntimeService,
    translation_service: TranslationService | None,
    server_base_url: str | None = None,
) -> InferencePipelineService | None:
    """active runtime profile이 있으면 profile 기준 pipeline을 조립한다."""

    record = runtime_profile_repository.load_active()
    if record is None:
        return None
    profile = record.profile
    manifest = shared_adapter_runtime_service.get_active_manifest()
    state = shared_adapter_runtime_service.get_active_state()
    if manifest.model_revision != profile.model_revision:
        raise ValueError(
            "active runtime profile model_revision does not match shared manifest: "
            f"{profile.model_revision!r} != {manifest.model_revision!r}."
        )
    if profile.required_state_kind is not None:
        state_schema_version = str(getattr(state, "schema_version", ""))
        if state_schema_version != profile.required_state_kind:
            raise ValueError(
                "active runtime profile required_state_kind does not match "
                f"shared state: {profile.required_state_kind!r} != "
                f"{state_schema_version!r}."
            )
    env_spec = load_agent_embedding_spec_from_env()
    embedding_spec = EmbeddingAdapterSpec(
        backend=profile.embedding_backend,
        model_id=profile.embedding_model_id,
        revision=env_spec.revision,
        device=env_spec.device,
        batch_size=env_spec.batch_size,
        cache_dir=env_spec.cache_dir,
        task_prefix=env_spec.task_prefix,
        normalize_embeddings=env_spec.normalize_embeddings,
        hash_dim=env_spec.hash_dim,
        local_files_only=env_spec.local_files_only,
    )
    return _build_pipeline_service(
        analysis_event_repository=analysis_event_repository,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        translation_service=translation_service,
        runtime_profile_repository=runtime_profile_repository,
        embedding_spec=embedding_spec,
        scoring_backend_name=profile.scorer_backend_name,
        model_revision=profile.model_revision,
        scoring_asset_provider=_build_profile_scoring_asset_provider(
            scoring_backend_name=profile.scorer_backend_name,
            state=state,
            server_base_url=server_base_url,
        ),
    )


def _build_pipeline_service(
    *,
    analysis_event_repository: AnalysisEventRepository,
    shared_adapter_runtime_service: SharedAdapterRuntimeService,
    translation_service: TranslationService | None,
    runtime_profile_repository: RuntimeProfileRepository | None,
    embedding_spec: EmbeddingAdapterSpec,
    scoring_backend_name: str,
    model_revision: str,
    scoring_asset_provider: _StaticScoringAssetProvider | None = None,
) -> InferencePipelineService:
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
        scoring_asset_provider=scoring_asset_provider,
        shared_adapter_provider=shared_adapter_runtime_service,
        runtime_profile_repository=runtime_profile_repository,
        translation_service=translation_service,
        embedding_model_id=embedding_spec.model_id,
        model_revision=model_revision,
    )


@dataclass(slots=True)
class _StaticScoringAssetProvider:
    assets: Mapping[str, Sequence[float] | Sequence[Sequence[float]]]

    def get_scoring_assets(
        self,
    ) -> Mapping[str, Sequence[float] | Sequence[Sequence[float]]]:
        return self.assets


def _build_profile_scoring_asset_provider(
    *,
    scoring_backend_name: str,
    state: object,
    server_base_url: str | None,
) -> _StaticScoringAssetProvider | None:
    if scoring_backend_name != PEFT_CLASSIFIER_HEAD_LOGITS_BACKEND_NAME:
        return None
    if not isinstance(state, PeftClassifierAdapterStatePayload):
        raise ValueError(
            "peft_classifier_head_logits requires active peft_classifier state."
        )
    if state.classifier_head_artifact_ref is None:
        raise ValueError(
            "active peft_classifier state does not provide "
            "classifier_head_artifact_ref."
        )
    if server_base_url is None:
        raise ValueError(
            "server_base_url is required to materialize PEFT classifier head artifacts."
        )
    artifact = _load_peft_classifier_head_artifact(
        server_base_url=server_base_url,
        artifact_ref=state.classifier_head_artifact_ref,
    )
    return _StaticScoringAssetProvider(
        build_peft_classifier_head_scoring_assets(
            classifier_head_artifact=artifact,
            label_schema=state.labels,
        )
    )


def _load_peft_classifier_head_artifact(
    *,
    server_base_url: str,
    artifact_ref: str,
) -> Mapping[str, object]:
    client = RoundArtifactClient(server_base_url=server_base_url)
    try:
        tensors, metadata = client.load_safetensors_artifact(artifact_ref=artifact_ref)
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 404:
            raise
        return client.load_json_artifact(artifact_ref=artifact_ref)
    head_state = parse_classifier_head_state_tensor_artifact(
        tensors=tensors,
        metadata=metadata,
    )
    return {
        "classifier_head_weights": head_state.classifier_head_weights,
        "classifier_head_biases": head_state.classifier_head_biases,
    }


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
