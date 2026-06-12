from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.simulation_runtime.round_runtime import (
    FederatedPeftEncoderRuntimeConfig,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.peft_text_encoder.update.merged_tensor_artifact import (
    parse_peft_adapter_state_tensor_artifact,
)
from methods.adaptation.text_encoder_classifier.classifier_head_tensor_artifact import (
    parse_classifier_head_state_tensor_artifact,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedInitialCheckpointConfig,
    FederatedLocalTrainerRuntimeConfig,
    FederatedRoundRuntimeConfig,
)
from scripts.runtime_adapters.federated_server.initial_checkpoint_artifacts import (
    ResolvedInitialCheckpointSource,
    publish_initial_checkpoint_artifacts_for_request,
)


def test_publish_initial_checkpoint_artifacts_writes_server_owned_tensors(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "central" / "adapter"
    adapter_dir.mkdir(parents=True)
    classifier_path = tmp_path / "central" / "classifier_head.safetensors"
    classifier_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "central" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "central_peft_supervised_epoch_checkpoint.v1",
                "trainer_version": "central_step_2000",
                "adapter_dir": str(adapter_dir),
                "classifier_path": str(classifier_path),
            }
        ),
        encoding="utf-8",
    )
    runtime_payload = FederatedPeftEncoderRuntimeConfig(
        training_backend_config=PeftEncoderTrainingBackendConfig()
    )
    request = SimpleNamespace(
        output_dir=tmp_path / "run",
        initial_checkpoint_config=FederatedInitialCheckpointConfig(
            name="required",
            mode="required",
            manifest_path=str(manifest_path),
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(device="cpu"),
        round_runtime_config=FederatedRoundRuntimeConfig(
            aggregation_backend_name="fedavg",
            update_family_name="peft_text_encoder",
            payload_adapter_kind="peft_classifier",
            runtime_payload_key="peft_text_encoder",
            runtime_payloads={"peft_text_encoder": runtime_payload},
        ),
    )

    def fake_builder(
        source: ResolvedInitialCheckpointSource,
        _runtime_payload: FederatedPeftEncoderRuntimeConfig,
        _runtime_config: FederatedLocalTrainerRuntimeConfig,
        labels: tuple[str, ...],
    ) -> PeftEncoderMaterializedState:
        assert source.adapter_dir == adapter_dir
        assert source.classifier_path == classifier_path
        assert source.reference_id == "central_step_2000"
        assert labels == ("anxiety", "normal")
        return PeftEncoderMaterializedState(
            peft_parameters={"base_model.lora_A": [0.1, 0.2]},
            classifier_head_weights={
                "anxiety": [1.0, 2.0],
                "normal": [3.0, 4.0],
            },
            classifier_head_biases={"anxiety": 0.5, "normal": -0.25},
        )

    publish_initial_checkpoint_artifacts_for_request(
        request=request,
        labels=("anxiety", "normal"),
        materialized_checkpoint_builder=fake_builder,
    )

    assert runtime_payload.peft_adapter_artifact_ref == (
        "server-aggregate://initial-checkpoint/central_step_2000/peft-adapter-state"
    )
    assert runtime_payload.classifier_head_artifact_ref == (
        "server-aggregate://initial-checkpoint/central_step_2000/classifier-head-state"
    )
    assert request.initial_checkpoint_config.resolved_kind == (
        "central_peft_classifier_checkpoint"
    )

    store = AggregationArtifactStore(
        state_root=request.output_dir / "main_server" / "aggregation_artifacts"
    )
    peft_tensors, peft_metadata = store.load_safetensors_artifact(
        artifact_ref=runtime_payload.peft_adapter_artifact_ref
    )
    head_tensors, head_metadata = store.load_safetensors_artifact(
        artifact_ref=runtime_payload.classifier_head_artifact_ref
    )

    assert parse_peft_adapter_state_tensor_artifact(
        tensors=peft_tensors,
        metadata=peft_metadata,
    ) == {"base_model.lora_A": pytest.approx([0.1, 0.2])}
    head_state = parse_classifier_head_state_tensor_artifact(
        tensors=head_tensors,
        metadata=head_metadata,
    )
    assert head_state.label_schema == ("anxiety", "normal")
    assert head_state.classifier_head_weights["normal"] == pytest.approx([3.0, 4.0])
    assert head_state.classifier_head_biases["anxiety"] == pytest.approx(0.5)


def test_publish_initial_checkpoint_manifest_accepts_repo_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint_dir = (
        Path("runs")
        / "central"
        / "supervised"
        / "peft_classifier"
        / "peft_clf_unit"
        / "checkpoints"
        / "epoch_0001_step_002000"
    )
    adapter_dir = checkpoint_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    classifier_path = checkpoint_dir / "classifier_head.safetensors"
    classifier_path.write_bytes(b"placeholder")
    manifest_path = checkpoint_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "central_peft_supervised_epoch_checkpoint.v1",
                "trainer_version": "central_step_2000",
                "adapter_dir": str(adapter_dir),
                "classifier_path": str(classifier_path),
            }
        ),
        encoding="utf-8",
    )
    request = _request_for_checkpoint(
        tmp_path=tmp_path,
        adapter_dir=None,
        classifier_path=None,
    )
    request.initial_checkpoint_config.manifest_path = str(manifest_path)

    def fake_builder(
        source: ResolvedInitialCheckpointSource,
        _runtime_payload: FederatedPeftEncoderRuntimeConfig,
        _runtime_config: FederatedLocalTrainerRuntimeConfig,
        _labels: tuple[str, ...],
    ) -> PeftEncoderMaterializedState:
        assert source.adapter_dir == adapter_dir
        assert source.classifier_path == classifier_path
        return PeftEncoderMaterializedState(
            peft_parameters={"base_model.lora_A": [0.1]},
            classifier_head_weights={"anxiety": [1.0]},
            classifier_head_biases={"anxiety": 0.0},
        )

    publish_initial_checkpoint_artifacts_for_request(
        request=request,
        labels=("anxiety",),
        materialized_checkpoint_builder=fake_builder,
    )


def test_publish_initial_checkpoint_requires_canonical_safetensors_head(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    classifier_path = tmp_path / "classifier.pt"
    classifier_path.write_bytes(b"legacy")
    request = _request_for_checkpoint(
        tmp_path=tmp_path,
        adapter_dir=adapter_dir,
        classifier_path=classifier_path,
    )

    with pytest.raises(ValueError, match="classifier_head.safetensors"):
        publish_initial_checkpoint_artifacts_for_request(
            request=request,
            labels=("anxiety",),
            materialized_checkpoint_builder=_unused_builder,
        )


def test_publish_initial_checkpoint_required_requires_source(tmp_path: Path) -> None:
    request = _request_for_checkpoint(
        tmp_path=tmp_path,
        adapter_dir=None,
        classifier_path=None,
    )

    with pytest.raises(ValueError, match="query_adaptation_initial_checkpoint"):
        publish_initial_checkpoint_artifacts_for_request(
            request=request,
            labels=("anxiety",),
            materialized_checkpoint_builder=_unused_builder,
        )


def _request_for_checkpoint(
    *,
    tmp_path: Path,
    adapter_dir: Path | None,
    classifier_path: Path | None,
) -> SimpleNamespace:
    runtime_payload = FederatedPeftEncoderRuntimeConfig(
        training_backend_config=PeftEncoderTrainingBackendConfig()
    )
    return SimpleNamespace(
        output_dir=tmp_path / "run",
        initial_checkpoint_config=FederatedInitialCheckpointConfig(
            name="required",
            mode="required",
            adapter_dir=None if adapter_dir is None else str(adapter_dir),
            classifier_path=None if classifier_path is None else str(classifier_path),
        ),
        local_trainer_runtime_config=FederatedLocalTrainerRuntimeConfig(device="cpu"),
        round_runtime_config=FederatedRoundRuntimeConfig(
            aggregation_backend_name="fedavg",
            update_family_name="peft_text_encoder",
            payload_adapter_kind="peft_classifier",
            runtime_payload_key="peft_text_encoder",
            runtime_payloads={"peft_text_encoder": runtime_payload},
        ),
    )


def _unused_builder(
    _source: ResolvedInitialCheckpointSource,
    _runtime_payload: FederatedPeftEncoderRuntimeConfig,
    _runtime_config: FederatedLocalTrainerRuntimeConfig,
    _labels: tuple[str, ...],
) -> PeftEncoderMaterializedState:
    raise AssertionError("materialized checkpoint builder should not run.")
