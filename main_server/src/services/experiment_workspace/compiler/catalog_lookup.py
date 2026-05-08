"""Experiment catalog lookup helpers for compiler."""

from __future__ import annotations

from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
    CatalogSectionPayload,
    CatalogTrackPayload,
)


def find_track(
    tracks: tuple[CatalogTrackPayload, ...],
    track_name: str,
) -> CatalogTrackPayload:
    for track in tracks:
        if track.track_name == track_name:
            return track
    raise ValueError(f"Unsupported workspace track: {track_name}.")


def find_section(
    sections: tuple[CatalogSectionPayload, ...],
    section_name: str,
) -> CatalogSectionPayload:
    for section in sections:
        if section.section_name == section_name:
            return section
    raise ValueError(f"Unsupported catalog section: {section_name}.")


def find_variant_item(
    items: tuple[CatalogItemPayload, ...],
    variant_profile_name: str,
) -> CatalogItemPayload:
    for item in items:
        if item.variant_profile_name == variant_profile_name:
            return item
        if item.item_name == variant_profile_name:
            return item
    raise ValueError(f"Unsupported catalog item: {variant_profile_name}.")
