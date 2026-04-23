import { OverrideFieldEditor } from "./OverrideFieldEditor";
import {
  formatCompileSupport,
  formatMetadataValue,
} from "../lib/formatters";
import type {
  CatalogItemPayload,
  CatalogOverrideFieldPayload,
  CatalogSectionPayload,
  WorkspaceConfigScalar,
} from "../types";

export function CatalogSectionCard(props: {
  section: CatalogSectionPayload;
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
  const selectedItem =
    section.items.find((item) => item.item_name === props.selectedItemName) ?? null;

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

      <div className="item-grid">
        {section.items.map((item) => {
          const isPresetSelectable = item.compile_support === "preset_selector";
          const isSelected = props.selectedItemName === item.item_name;

          return (
            <button
              key={item.item_name}
              type="button"
              className={isSelected ? "item-card item-card--selected" : "item-card"}
              onClick={() =>
                isPresetSelectable ? props.onItemToggle(section, item) : undefined
              }
              disabled={!isPresetSelectable}
            >
              <div className="item-card__topline">
                <strong>{item.display_name}</strong>
                <span
                  className={`support-badge support-badge--${item.compile_support}`}
                >
                  {formatCompileSupport(item.compile_support)}
                </span>
              </div>

              <div className="item-card__meta">
                {item.core_method_name ? (
                  <span>핵심 방식: {item.core_method_name}</span>
                ) : null}
                {item.family_name ? <span>패밀리: {item.family_name}</span> : null}
                {item.preset_group ? <span>그룹: {item.preset_group}</span> : null}
              </div>

              {item.description ? <p className="item-card__description">{item.description}</p> : null}

              {item.compile_blocker_reason ? (
                <p className="item-card__blocker">{item.compile_blocker_reason}</p>
              ) : null}

              {Object.keys(item.metadata).length > 0 ? (
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
            </button>
          );
        })}
      </div>

      {selectedItem ? (
        <div className="override-editor">
          {selectedItem.override_fields.length > 0 ? (
            <div className="override-field-editor">
              <p className="override-editor__title">
                {section.display_name} 빠른 조정 항목
              </p>
              {selectedItem.override_fields.map((field) => (
                <OverrideFieldEditor
                  key={field.field_name}
                  sectionName={section.section_name}
                  field={field}
                  overridePatch={props.selectedOverridePatch}
                  onChange={props.onOverrideFieldChange}
                />
              ))}
            </div>
          ) : null}
          <label htmlFor={`override-${section.section_name}`}>
            고급 JSON 패치
          </label>
          <textarea
            id={`override-${section.section_name}`}
            value={props.selectedOverrideText}
            onChange={(event) =>
              props.onOverrideTextChange(section.section_name, event.target.value)
            }
            spellCheck={false}
          />
          <p className="hint-text">
            scalar value만 허용합니다. 예: {`{"temperature": 0.7}`}
          </p>
          <p className="hint-text">
            Hydra 파일 본문을 수정하는 대신, 선택한 preset 위에 override patch만
            덧씌웁니다.
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
}
