import { useEffect, useState } from "react";

import { loadExperimentCatalog } from "../api";
import { asErrorMessage } from "../lib/formatters";
import { getEntrypointSection } from "../lib/workspaceDraft";
import type {
  CatalogItemPayload,
  CatalogSectionPayload,
  CatalogTrackPayload,
  ExperimentCatalogPayload,
} from "../types";

export interface ExperimentCatalogState {
  catalog: ExperimentCatalogPayload | null;
  catalogError: string | null;
  isCatalogLoading: boolean;
  selectedTrackName: string | null;
  selectedEntrypointName: string | null;
  activeTrack: CatalogTrackPayload | null;
  entrypointItem: CatalogItemPayload | null;
  nonEntrypointSections: CatalogSectionPayload[];
  setInitialSelection: (trackName: string, entrypointName: string) => void;
  handleTrackChange: (track: CatalogTrackPayload) => void;
  handleEntrypointChange: (item: CatalogItemPayload) => void;
}

export function useExperimentCatalog(
  apiBaseUrl: string,
): ExperimentCatalogState {
  const [catalog, setCatalog] = useState<ExperimentCatalogPayload | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [isCatalogLoading, setIsCatalogLoading] = useState(true);
  const [selectedTrackName, setSelectedTrackName] = useState<string | null>(null);
  const [selectedEntrypointName, setSelectedEntrypointName] = useState<string | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setIsCatalogLoading(true);
      setCatalogError(null);

      try {
        const payload = await loadExperimentCatalog(apiBaseUrl);
        if (cancelled) {
          return;
        }
        setCatalog(payload);
        const firstTrack = payload.tracks[0] ?? null;
        const firstEntrypoint = firstTrack
          ? getEntrypointSection(firstTrack)?.items[0] ?? null
          : null;
        setSelectedTrackName(firstTrack?.track_name ?? null);
        setSelectedEntrypointName(firstEntrypoint?.item_name ?? null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setCatalogError(asErrorMessage(error));
      } finally {
        if (!cancelled) {
          setIsCatalogLoading(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const activeTrack =
    catalog?.tracks.find((track) => track.track_name === selectedTrackName) ?? null;
  const entrypointSection = activeTrack ? getEntrypointSection(activeTrack) : null;
  const entrypointItem =
    entrypointSection?.items.find((item) => item.item_name === selectedEntrypointName) ??
    entrypointSection?.items[0] ??
    null;
  const nonEntrypointSections =
    activeTrack?.sections.filter(
      (section) => section.section_name !== entrypointSection?.section_name,
    ) ?? [];

  function setInitialSelection(trackName: string, entrypointName: string) {
    setSelectedTrackName(trackName);
    setSelectedEntrypointName(entrypointName);
  }

  function handleTrackChange(track: CatalogTrackPayload) {
    const nextEntrypoint = getEntrypointSection(track)?.items[0] ?? null;
    setSelectedTrackName(track.track_name);
    setSelectedEntrypointName(nextEntrypoint?.item_name ?? null);
  }

  function handleEntrypointChange(item: CatalogItemPayload) {
    setSelectedEntrypointName(item.item_name);
  }

  return {
    catalog,
    catalogError,
    isCatalogLoading,
    selectedTrackName,
    selectedEntrypointName,
    activeTrack,
    entrypointItem,
    nonEntrypointSections,
    setInitialSelection,
    handleTrackChange,
    handleEntrypointChange,
  };
}
