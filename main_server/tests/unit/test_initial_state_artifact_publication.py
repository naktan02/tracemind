from __future__ import annotations

import torch

from main_server.src.services.federation.rounds import (
    initial_state_artifact_publication as initial_artifact_publication,
)
from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)


def test_initial_state_artifact_publication_saves_server_owned_tensors(
    tmp_path,
) -> None:
    artifact_store = AggregationArtifactStore(state_root=tmp_path / "artifacts")
    service = initial_artifact_publication.InitialStateArtifactPublicationService(
        artifact_store=artifact_store
    )

    publication = service.publish_tensor_artifacts(
        initial_artifact_publication.InitialStateArtifactPublicationRequest(
            publication_id="central step 2000",
            artifact_ref_prefix="server-aggregate://initial-checkpoint",
            artifact_slots=(
                initial_artifact_publication.ServerTensorArtifactSlot(
                    artifact_name="peft adapter state",
                    tensors={"weights": torch.tensor([1.0, 2.0])},
                    metadata={"schema": "test.v1"},
                ),
            ),
        )
    )

    artifact_ref = publication.artifact_refs["peft adapter state"]
    assert artifact_ref == (
        "server-aggregate://initial-checkpoint/central-step-2000/peft-adapter-state"
    )
    tensors, metadata = artifact_store.load_safetensors_artifact(
        artifact_ref=artifact_ref
    )
    assert tensors["weights"].tolist() == [1.0, 2.0]
    assert metadata == {"schema": "test.v1"}
