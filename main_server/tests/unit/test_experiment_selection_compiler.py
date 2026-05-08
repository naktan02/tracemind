"""Experiment workspace selection compiler tests."""

from __future__ import annotations

import pytest

from main_server.src.services.experiment_workspace.compiler.selection_compiler import (
    compile_workspace_selections,
)
from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
    CatalogTrackPayload,
)
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceSelectionPayload,
)


def _track_with_item(item: CatalogItemPayload) -> CatalogTrackPayload:
    return CatalogTrackPayload(
        track_name="central_ssl_control",
        display_name="Central SSL",
        sections=(
            CatalogSectionPayload(
                section_name="peft_adapter",
                display_name="PEFT Adapter",
                item_kind="peft_adapter",
                source_of_truth="conf/strategy_axes/adaptation/peft_adapter",
                source_kind="hydra_config",
                items=(item,),
            ),
        ),
    )


def _preset_item(**overrides: object) -> CatalogItemPayload:
    defaults = {
        "item_name": "lora_r8",
        "display_name": "LoRA r8",
        "item_kind": "peft_adapter",
        "family_name": "lora",
        "core_method_name": "lora_adapter",
        "variant_profile_name": "lora_r8",
        "compiled_selector_name": "lora",
        "preset_group": "lora",
        "source_of_truth": "conf/strategy_axes/adaptation/peft_adapter/lora.yaml",
        "source_kind": "hydra_config",
        "compile_support": "preset_selector",
        "default_override_patch": {"rank": 8, "use_rslora": False},
        "metadata": {"selector_group": "strategy_axes/adaptation/peft_adapter"},
    }
    defaults.update(overrides)
    return CatalogItemPayload(**defaults)


def test_compile_workspace_selections_builds_selectors_overrides_and_payloads() -> None:
    track = _track_with_item(_preset_item())

    compiled = compile_workspace_selections(
        track=track,
        selections=(
            WorkspaceSelectionPayload(
                slot_name="peft_adapter",
                section_name="peft_adapter",
                variant_profile_name="lora_r8",
                family_name="lora",
                core_method_name="lora_adapter",
                override_patch={"rank": 16, "dropout": 0.1},
            ),
        ),
    )

    assert compiled.selection_default_groups == (
        "strategy_axes/adaptation/peft_adapter=lora",
    )
    assert compiled.hydra_overrides == (
        "lora.rank=16",
        "lora.use_rslora=false",
        "lora.dropout=0.1",
    )
    resolved = compiled.resolved_selections[0]
    assert resolved.slot_name == "peft_adapter"
    assert resolved.compiled_selector == "strategy_axes/adaptation/peft_adapter=lora"
    assert resolved.compiled_overrides == compiled.hydra_overrides


def test_compile_workspace_selections_rejects_duplicate_slots() -> None:
    track = _track_with_item(_preset_item())
    selection = WorkspaceSelectionPayload(
        slot_name="peft_adapter",
        section_name="peft_adapter",
        variant_profile_name="lora_r8",
    )

    with pytest.raises(ValueError, match="Duplicate workspace slot"):
        compile_workspace_selections(
            track=track,
            selections=(selection, selection),
        )


def test_compile_workspace_selections_rejects_family_mismatch() -> None:
    track = _track_with_item(_preset_item())

    with pytest.raises(ValueError, match="family mismatch"):
        compile_workspace_selections(
            track=track,
            selections=(
                WorkspaceSelectionPayload(
                    slot_name="peft_adapter",
                    section_name="peft_adapter",
                    variant_profile_name="lora_r8",
                    family_name="ia3",
                ),
            ),
        )


def test_compile_workspace_selections_rejects_metadata_only_item() -> None:
    item = _preset_item(
        compile_support="metadata_only",
        preset_group=None,
        compile_blocker_reason="preview only",
    )
    track = _track_with_item(item)

    with pytest.raises(ValueError, match="not compileable yet"):
        compile_workspace_selections(
            track=track,
            selections=(
                WorkspaceSelectionPayload(
                    slot_name="peft_adapter",
                    section_name="peft_adapter",
                    variant_profile_name="lora_r8",
                ),
            ),
        )
