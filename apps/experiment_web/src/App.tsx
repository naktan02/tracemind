import { useEffect, useState } from "react";

import {
  compileExperimentWorkspace,
  loadExperimentCatalog,
  resolveApiBaseUrl,
} from "./api";
import type {
  CatalogItemPayload,
  CatalogSectionPayload,
  CatalogTrackPayload,
  ExperimentCatalogPayload,
  ResolvedExperimentPlanPayload,
  WorkspaceConfigScalar,
  WorkspaceManifestPayload,
  WorkspaceSelectionPayload,
} from "./types";

const EMPTY_OVERRIDE_JSON = "{}";

interface ObjectParseResult {
  value: Record<string, WorkspaceConfigScalar>;
  error: string | null;
}

function App() {
  const apiBaseUrl = resolveApiBaseUrl();
  const [catalog, setCatalog] = useState<ExperimentCatalogPayload | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [isCatalogLoading, setIsCatalogLoading] = useState(true);

  const [selectedTrackName, setSelectedTrackName] = useState<string | null>(null);
  const [selectedEntrypointName, setSelectedEntrypointName] = useState<string | null>(
    null,
  );
  const [selectedItemNameBySection, setSelectedItemNameBySection] = useState<
    Record<string, string | null>
  >({});
  const [overrideTextBySection, setOverrideTextBySection] = useState<
    Record<string, string>
  >({});
  const [globalOverrideText, setGlobalOverrideText] =
    useState<string>(EMPTY_OVERRIDE_JSON);

  const [compilePlan, setCompilePlan] = useState<ResolvedExperimentPlanPayload | null>(
    null,
  );
  const [compileError, setCompileError] = useState<string | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);

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

  const activeTrack = catalog?.tracks.find(
    (track) => track.track_name === selectedTrackName,
  );
  const entrypointSection = activeTrack ? getEntrypointSection(activeTrack) : null;
  const entrypointItem =
    entrypointSection?.items.find((item) => item.item_name === selectedEntrypointName) ??
    entrypointSection?.items[0] ??
    null;
  const nonEntrypointSections =
    activeTrack?.sections.filter(
      (section) => section.section_name !== entrypointSection?.section_name,
    ) ??
    [];

  const globalOverrideParse = parseOverrideObject(globalOverrideText);
  const sectionOverrideErrors = buildSectionOverrideErrors(
    nonEntrypointSections,
    selectedItemNameBySection,
    overrideTextBySection,
  );
  const localParseErrors = [
    globalOverrideParse.error
      ? `global_override_patch: ${globalOverrideParse.error}`
      : null,
    ...sectionOverrideErrors,
  ].filter(Boolean) as string[];

  async function handleCompilePreview() {
    if (!activeTrack || !entrypointItem) {
      setCompileError("먼저 track과 entrypoint를 선택하세요.");
      setCompilePlan(null);
      return;
    }
    if (localParseErrors.length > 0) {
      setCompileError(localParseErrors[0]);
      setCompilePlan(null);
      return;
    }

    const selections = buildWorkspaceSelections(
      nonEntrypointSections,
      selectedItemNameBySection,
      overrideTextBySection,
    );
    const manifest: WorkspaceManifestPayload = {
      schema_version: "workspace_manifest.v1",
      manifest_id: buildManifestId(activeTrack.track_name),
      track_name: activeTrack.track_name,
      entrypoint_name: entrypointItem.item_name,
      selections,
      global_override_patch: globalOverrideParse.value,
      notes: null,
    };

    setIsCompiling(true);
    setCompileError(null);
    try {
      const plan = await compileExperimentWorkspace(apiBaseUrl, manifest);
      setCompilePlan(plan);
    } catch (error) {
      setCompilePlan(null);
      setCompileError(asErrorMessage(error));
    } finally {
      setIsCompiling(false);
    }
  }

  function handleTrackChange(track: CatalogTrackPayload) {
    const nextEntrypoint = getEntrypointSection(track)?.items[0] ?? null;
    setSelectedTrackName(track.track_name);
    setSelectedEntrypointName(nextEntrypoint?.item_name ?? null);
    setSelectedItemNameBySection({});
    setOverrideTextBySection({});
    setGlobalOverrideText(EMPTY_OVERRIDE_JSON);
    setCompilePlan(null);
    setCompileError(null);
  }

  function handleEntrypointChange(item: CatalogItemPayload) {
    setSelectedEntrypointName(item.item_name);
    setCompilePlan(null);
    setCompileError(null);
  }

  function handleSectionItemToggle(
    section: CatalogSectionPayload,
    item: CatalogItemPayload,
  ) {
    const nextValue =
      selectedItemNameBySection[section.section_name] === item.item_name
        ? null
        : item.item_name;
    setSelectedItemNameBySection((current) => ({
      ...current,
      [section.section_name]: nextValue,
    }));
    setCompilePlan(null);
    setCompileError(null);
  }

  function handleResetLane() {
    if (!activeTrack) {
      return;
    }
    const firstEntrypoint = getEntrypointSection(activeTrack)?.items[0] ?? null;
    setSelectedEntrypointName(firstEntrypoint?.item_name ?? null);
    setSelectedItemNameBySection({});
    setOverrideTextBySection({});
    setGlobalOverrideText(EMPTY_OVERRIDE_JSON);
    setCompilePlan(null);
    setCompileError(null);
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">TraceMind Developer Workspace</p>
          <h1>Experiment lanes before runtime</h1>
          <p className="hero-text">
            Track, entrypoint, preset, readiness, compile error, and Hydra preview를
            UI에서 먼저 확인하는 Phase 3 MVP입니다.
          </p>
        </div>
        <div className="hero-meta">
          <div className="meta-card">
            <span className="meta-label">API Base</span>
            <span className="meta-value">{apiBaseUrl}</span>
          </div>
          <div className="meta-card">
            <span className="meta-label">Catalog</span>
            <span className="meta-value">
              {catalog ? `${catalog.tracks.length} tracks` : "loading"}
            </span>
          </div>
        </div>
      </header>

      {isCatalogLoading ? (
        <main className="status-panel">
          <h2>Loading catalog</h2>
          <p>현재 experiment catalog와 compile surface를 읽는 중입니다.</p>
        </main>
      ) : null}

      {catalogError ? (
        <main className="status-panel status-panel--error">
          <h2>Catalog request failed</h2>
          <p>{catalogError}</p>
        </main>
      ) : null}

      {catalog && activeTrack ? (
        <main className="workspace-grid">
          <section className="panel lane-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Lane</p>
                <h2>Track and entrypoint</h2>
              </div>
            </div>

            <div className="track-tabs" role="tablist" aria-label="Experiment tracks">
              {catalog.tracks.map((track) => (
                <button
                  key={track.track_name}
                  type="button"
                  className={
                    track.track_name === activeTrack.track_name
                      ? "track-tab track-tab--active"
                      : "track-tab"
                  }
                  onClick={() => handleTrackChange(track)}
                >
                  <span>{track.display_name}</span>
                  <small>{track.track_name}</small>
                </button>
              ))}
            </div>

            <div className="track-summary">
              <p>{activeTrack.description}</p>
              <div className="pill-row">
                {activeTrack.supported_runtime_paths.map((runtimePath) => (
                  <span className="pill" key={runtimePath}>
                    {runtimePath}
                  </span>
                ))}
              </div>
            </div>

            <div className="entrypoint-list">
              {entrypointSection?.items.map((item) => (
                <button
                  key={item.item_name}
                  type="button"
                  className={
                    entrypointItem?.item_name === item.item_name
                      ? "entrypoint-card entrypoint-card--active"
                      : "entrypoint-card"
                  }
                  onClick={() => handleEntrypointChange(item)}
                >
                  <strong>{item.display_name}</strong>
                  <span>{item.script_path}</span>
                  <code>{item.source_of_truth}</code>
                </button>
              ))}
            </div>
          </section>

          <section className="panel catalog-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Palette</p>
                <h2>Catalog sections</h2>
              </div>
              <button type="button" className="ghost-button" onClick={handleResetLane}>
                Reset lane
              </button>
            </div>

            <div className="section-list">
              {nonEntrypointSections.map((section) => {
                const selectedItemName =
                  selectedItemNameBySection[section.section_name] ?? null;
                const selectedItem =
                  section.items.find((item) => item.item_name === selectedItemName) ??
                  null;
                const selectedOverrideText =
                  overrideTextBySection[section.section_name] ?? EMPTY_OVERRIDE_JSON;

                return (
                  <article className="section-card" key={section.section_name}>
                    <div className="section-card__header">
                      <div>
                        <h3>{section.display_name}</h3>
                        <p>{section.description}</p>
                      </div>
                      <span className="section-origin">{section.source_of_truth}</span>
                    </div>

                    <div className="item-grid">
                      {section.items.map((item) => {
                        const isPresetSelectable =
                          item.compile_support === "preset_selector";
                        const isSelected = selectedItemName === item.item_name;

                        return (
                          <button
                            key={item.item_name}
                            type="button"
                            className={
                              isSelected
                                ? "item-card item-card--selected"
                                : "item-card"
                            }
                            onClick={() =>
                              isPresetSelectable
                                ? handleSectionItemToggle(section, item)
                                : undefined
                            }
                            disabled={!isPresetSelectable}
                          >
                            <div className="item-card__topline">
                              <strong>{item.display_name}</strong>
                              <span
                                className={`support-badge support-badge--${item.compile_support}`}
                              >
                                {item.compile_support}
                              </span>
                            </div>

                            <div className="item-card__meta">
                              {item.core_method_name ? (
                                <span>core: {item.core_method_name}</span>
                              ) : null}
                              {item.family_name ? (
                                <span>family: {item.family_name}</span>
                              ) : null}
                              {item.preset_group ? (
                                <span>group: {item.preset_group}</span>
                              ) : null}
                            </div>

                            {item.compile_blocker_reason ? (
                              <p className="item-card__blocker">
                                {item.compile_blocker_reason}
                              </p>
                            ) : null}

                            {Object.keys(item.metadata).length > 0 ? (
                              <dl className="metadata-list">
                                {Object.entries(item.metadata).map(([key, value]) => (
                                  <div key={key}>
                                    <dt>{key}</dt>
                                    <dd>{formatMetadataValue(value)}</dd>
                                  </div>
                                ))}
                              </dl>
                            ) : null}
                          </button>
                        );
                      })}
                    </div>

                    {selectedItem ? (
                      <div className="override-editor">
                        <label htmlFor={`override-${section.section_name}`}>
                          {section.display_name} override patch
                        </label>
                        <textarea
                          id={`override-${section.section_name}`}
                          value={selectedOverrideText}
                          onChange={(event) =>
                            setOverrideTextBySection((current) => ({
                              ...current,
                              [section.section_name]: event.target.value,
                            }))
                          }
                          spellCheck={false}
                        />
                        <p className="hint-text">
                          scalar value만 허용합니다. 예: {`{"temperature": 0.7}`}
                        </p>
                        <p className="hint-text">
                          Hydra 파일 본문을 수정하는 대신, 선택한 preset 위에
                          override patch만 덧씌웁니다.
                        </p>
                        {selectedItem.declared_fields.length > 0 ? (
                          <div className="field-chip-row">
                            {selectedItem.declared_fields.map((fieldName) => (
                              <span className="field-chip" key={fieldName}>
                                {fieldName}
                              </span>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          </section>

          <section className="panel preview-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Preview</p>
                <h2>Workspace manifest and compile result</h2>
              </div>
              <button
                type="button"
                className="primary-button"
                onClick={() => void handleCompilePreview()}
                disabled={isCompiling}
              >
                {isCompiling ? "Compiling..." : "Compile preview"}
              </button>
            </div>

            <div className="preview-block">
              <label htmlFor="global-override">Global override patch</label>
              <textarea
                id="global-override"
                value={globalOverrideText}
                onChange={(event) => setGlobalOverrideText(event.target.value)}
                spellCheck={false}
              />
              <p className="hint-text">
                top-level Hydra override만 넣습니다. 예:{" "}
                {`{"train_batch_size": 32, "training_task.local_epochs": 2}`}
              </p>
            </div>

            <div className="preview-block">
              <h3>Workspace manifest</h3>
              <pre>
                {JSON.stringify(
                  buildWorkspaceManifestPreview(
                    activeTrack.track_name,
                    entrypointItem?.item_name ?? null,
                    nonEntrypointSections,
                    selectedItemNameBySection,
                    overrideTextBySection,
                    globalOverrideParse.value,
                  ),
                  null,
                  2,
                )}
              </pre>
            </div>

            {localParseErrors.length > 0 ? (
              <div className="message-block message-block--error">
                <h3>Local parse error</h3>
                <ul>
                  {localParseErrors.map((error) => (
                    <li key={error}>{error}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            {compileError ? (
              <div className="message-block message-block--error">
                <h3>Compile error</h3>
                <p>{compileError}</p>
              </div>
            ) : null}

            {compilePlan ? (
              <div className="compile-result">
                <div className="message-block message-block--ok">
                  <h3>Compile result</h3>
                  <p>
                    {compilePlan.track_name} / {compilePlan.entrypoint_name}
                  </p>
                </div>

                {compilePlan.warnings.length > 0 ? (
                  <div className="message-block message-block--warning">
                    <h3>Warnings</h3>
                    <ul>
                      {compilePlan.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                <div className="result-stack">
                  <ResultBlock
                    title="Script"
                    lines={[compilePlan.script_path, compilePlan.job_config_path]}
                  />
                  <ResultBlock
                    title="Default groups"
                    lines={[
                      ...compilePlan.base_default_groups,
                      ...compilePlan.selection_default_groups,
                    ]}
                  />
                  <ResultBlock
                    title="Hydra overrides"
                    lines={compilePlan.hydra_overrides}
                  />
                  <ResultBlock
                    title="Command args"
                    lines={compilePlan.command_args}
                  />
                </div>
              </div>
            ) : null}
          </section>
        </main>
      ) : null}
    </div>
  );
}

function ResultBlock(props: { title: string; lines: string[] }) {
  return (
    <section className="result-block">
      <h4>{props.title}</h4>
      {props.lines.length > 0 ? (
        <pre>{props.lines.join("\n")}</pre>
      ) : (
        <p className="hint-text">No values</p>
      )}
    </section>
  );
}

function getEntrypointSection(
  track: CatalogTrackPayload,
): CatalogSectionPayload | undefined {
  if (track.entrypoint_section_name) {
    return track.sections.find(
      (section) => section.section_name === track.entrypoint_section_name,
    );
  }
  return track.sections.find(
    (section) => section.item_kind === "experiment_entrypoint",
  );
}

function buildManifestId(trackName: string) {
  return `${trackName}_${Date.now()}`;
}

function buildWorkspaceManifestPreview(
  trackName: string,
  entrypointName: string | null,
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overrideTextBySection: Record<string, string>,
  globalOverridePatch: Record<string, WorkspaceConfigScalar>,
) {
  return {
    schema_version: "workspace_manifest.v1",
    manifest_id: buildManifestId(trackName),
    track_name: trackName,
    entrypoint_name: entrypointName,
    selections: buildWorkspaceSelections(
      sections,
      selectedItemNameBySection,
      overrideTextBySection,
    ),
    global_override_patch: globalOverridePatch,
    notes: null,
  };
}

function buildWorkspaceSelections(
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overrideTextBySection: Record<string, string>,
): WorkspaceSelectionPayload[] {
  return sections.flatMap((section) => {
    const selectedItemName = selectedItemNameBySection[section.section_name];
    if (!selectedItemName) {
      return [];
    }
    const item = section.items.find((candidate) => candidate.item_name === selectedItemName);
    if (!item || item.compile_support !== "preset_selector") {
      return [];
    }

    return [
      {
        slot_name: section.default_slot_name ?? section.section_name,
        section_name: section.section_name,
        variant_profile_name: item.variant_profile_name ?? item.item_name,
        core_method_name: item.core_method_name ?? null,
        family_name: item.family_name ?? null,
        override_patch: parseOverrideObject(
          overrideTextBySection[section.section_name] ?? EMPTY_OVERRIDE_JSON,
        ).value,
      },
    ];
  });
}

function buildSectionOverrideErrors(
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overrideTextBySection: Record<string, string>,
): string[] {
  const errors: string[] = [];

  for (const section of sections) {
    if (!selectedItemNameBySection[section.section_name]) {
      continue;
    }
    const parseResult = parseOverrideObject(
      overrideTextBySection[section.section_name] ?? EMPTY_OVERRIDE_JSON,
    );
    if (parseResult.error) {
      errors.push(`${section.display_name}: ${parseResult.error}`);
    }
  }

  return errors;
}

function parseOverrideObject(input: string): ObjectParseResult {
  const normalized = input.trim();
  if (!normalized) {
    return { value: {}, error: null };
  }

  try {
    const parsed = JSON.parse(normalized) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {
        value: {},
        error: "JSON object여야 합니다.",
      };
    }

    const validated: Record<string, WorkspaceConfigScalar> = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (!isScalarOverrideValue(value)) {
        return {
          value: {},
          error: `${key} 값은 string/number/boolean/null만 허용합니다.`,
        };
      }
      validated[key] = value;
    }
    return { value: validated, error: null };
  } catch (error) {
    return {
      value: {},
      error: asErrorMessage(error),
    };
  }
}

function isScalarOverrideValue(value: unknown): value is WorkspaceConfigScalar {
  return (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function formatMetadataValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }
  return JSON.stringify(value);
}

function asErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export default App;
