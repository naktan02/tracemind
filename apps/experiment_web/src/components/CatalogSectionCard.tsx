import { OverrideFieldEditor } from "./OverrideFieldEditor";
import {
  formatCompileSupport,
  formatMetadataValue,
} from "../lib/formatters";
import type { WorkspaceSectionPresentation } from "../lib/workspaceSections";
import type {
  CatalogItemPayload,
  CatalogOverrideFieldPayload,
  CatalogSectionPayload,
  WorkspaceConfigScalar,
} from "../types";

export function CatalogSectionCard(props: {
  section: CatalogSectionPayload;
  presentation?: WorkspaceSectionPresentation;
  selectedItemName: string | null;
  selectedOverrideText: string;
  selectedOverridePatch: Record<string, WorkspaceConfigScalar>;
  onItemToggle: (section: CatalogSectionPayload, item: CatalogItemPayload) => void;
  onOverrideTextChange: (sectionName: string, nextText: string) => void;
  onOverrideFieldChange: (
    sectionName: string,
    field: CatalogOverrideFieldPayload,
    nextValue: string | number | boolean | undefined,
  ) => void;
}) {
  const { section } = props;
  const presentation = props.presentation ?? "cards";
  const isListPresentation = presentation === "list";

  return (
    <article className="section-card">
      <div className="section-card__header">
        <div>
          <h3>{section.display_name}</h3>
          <p>{section.description}</p>
        </div>
        <details className="section-tech-details">
          <summary>기술 정보</summary>
          <span className="section-origin">{section.source_of_truth}</span>
        </details>
      </div>

      <div className={isListPresentation ? "item-list" : "item-grid"}>
        {section.items.map((item) => {
          const isPresetSelectable = item.compile_support === "preset_selector";
          const isSelected = props.selectedItemName === item.item_name;

          return (
            <article
              key={item.item_name}
              className={
                isSelected
                  ? `item-card item-card--selected ${
                      isListPresentation ? "item-card--list" : ""
                    }`
                  : `item-card ${isListPresentation ? "item-card--list" : ""}`
              }
            >
              <button
                type="button"
                className={
                  isListPresentation
                    ? "item-card__button item-card__button--list"
                    : "item-card__button"
                }
                onClick={() =>
                  isPresetSelectable ? props.onItemToggle(section, item) : undefined
                }
                disabled={!isPresetSelectable}
              >
                <div className="item-card__topline">
                  <div className="item-card__title-block">
                    <strong>{item.display_name}</strong>
                    {isListPresentation ? (
                      <span className="hint-text">
                        {item.description ?? "이 항목을 선택합니다."}
                      </span>
                    ) : null}
                  </div>
                  <span
                    className={`support-badge support-badge--${item.compile_support}`}
                  >
                    {formatCompileSupport(item.compile_support)}
                  </span>
                </div>

                {isListPresentation ? (
                  <div className="item-card__list-meta">
                    {item.tags.includes("generated_artifact") ? (
                      <span>기존 생성 기록</span>
                    ) : (
                      <span>기본 preset</span>
                    )}
                    {item.default_override_patch &&
                    Object.keys(item.default_override_patch).length > 0 ? (
                      <span>기본 경로 자동 연결</span>
                    ) : null}
                  </div>
                ) : (
                  <div className="item-card__meta">
                    {item.core_method_name ? (
                      <span>핵심 방식: {item.core_method_name}</span>
                    ) : null}
                    {item.family_name ? <span>패밀리: {item.family_name}</span> : null}
                    {item.preset_group ? <span>그룹: {item.preset_group}</span> : null}
                  </div>
                )}

                {item.description && !isListPresentation ? (
                  <p className="item-card__description">{item.description}</p>
                ) : null}

                {item.tags.includes("generated_artifact") && !isListPresentation ? (
                  <p className="hint-text">
                    기존에 만들어 둔 결과물에서 자동으로 읽어온 항목입니다.
                  </p>
                ) : null}

                {Object.keys(item.metadata).length > 0 && !isListPresentation ? (
                  <dl className="metadata-list metadata-list--compact">
                    {Object.entries(item.metadata)
                      .slice(0, 3)
                      .map(([key, value]) => (
                        <div key={key}>
                          <dt>{key}</dt>
                          <dd>{formatMetadataValue(value)}</dd>
                        </div>
                      ))}
                  </dl>
                ) : null}

                {item.compile_blocker_reason ? (
                  <p className="item-card__blocker">{item.compile_blocker_reason}</p>
                ) : null}
              </button>

              {isSelected ? (
                <div className="item-card__editor">
                  {item.override_fields.length > 0 ? (
                    <div className="override-field-editor">
                      <p className="override-editor__title">
                        이 카드에서 바로 조정할 수 있는 항목
                      </p>
                      {item.override_fields.map((field) => (
                        <OverrideFieldEditor
                          key={field.field_name}
                          sectionName={section.section_name}
                          field={field}
                          overridePatch={props.selectedOverridePatch}
                          onChange={props.onOverrideFieldChange}
                        />
                      ))}
                    </div>
                  ) : (
                    <p className="hint-text">
                      이 preset은 빠르게 조정할 scalar 항목이 없습니다.
                    </p>
                  )}

                  {Object.keys(item.default_override_patch).length > 0 ? (
                    <p className="hint-text">
                      이 항목을 고르면 기존 결과물 경로가 기본값으로 함께 연결됩니다.
                    </p>
                  ) : null}

                  <details className="advanced-panel advanced-panel--inline">
                    <summary>고급 JSON 패치</summary>
                    <label htmlFor={`override-${section.section_name}`}>
                      세부 override
                    </label>
                    <textarea
                      id={`override-${section.section_name}`}
                      value={props.selectedOverrideText}
                      onChange={(event) =>
                        props.onOverrideTextChange(
                          section.section_name,
                          event.target.value,
                        )
                      }
                      spellCheck={false}
                    />
                    <p className="hint-text">
                      scalar value만 허용합니다. 예: {`{"temperature": 0.7}`}
                    </p>
                    <p className="hint-text">
                      Hydra 파일 본문을 수정하는 대신, 선택한 preset 위에 override
                      patch만 덧씌웁니다.
                    </p>
                    {item.declared_fields.length > 0 ? (
                      <div className="field-chip-row">
                        {item.declared_fields.map((fieldName) => (
                          <span className="field-chip" key={fieldName}>
                            {fieldName}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </details>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </article>
  );
}
