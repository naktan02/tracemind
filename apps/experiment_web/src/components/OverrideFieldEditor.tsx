import { formatScalarValue } from "../lib/formatters";
import type {
  CatalogOverrideFieldPayload,
  WorkspaceConfigScalar,
} from "../types";

export function OverrideFieldEditor(props: {
  sectionName: string;
  field: CatalogOverrideFieldPayload;
  overridePatch: Record<string, WorkspaceConfigScalar>;
  onChange: (
    sectionName: string,
    field: CatalogOverrideFieldPayload,
    nextValue: string | number | boolean | undefined,
  ) => void;
}) {
  const { field, overridePatch, sectionName, onChange } = props;
  const isOverridden = Object.prototype.hasOwnProperty.call(
    overridePatch,
    field.field_name,
  );
  const effectiveValue = overridePatch[field.field_name] ?? field.default_value;

  return (
    <div className="override-field-row">
      <div className="override-field-meta">
        <label htmlFor={`${sectionName}-${field.field_name}`}>
          {field.field_name}
        </label>
        <span>default: {formatScalarValue(field.default_value)}</span>
      </div>

      {field.value_kind === "boolean" ? (
        <select
          id={`${sectionName}-${field.field_name}`}
          value={String(effectiveValue)}
          onChange={(event) =>
            onChange(sectionName, field, event.target.value === "true")
          }
        >
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      ) : field.value_kind === "integer" || field.value_kind === "number" ? (
        <input
          id={`${sectionName}-${field.field_name}`}
          type="number"
          step={field.value_kind === "integer" ? "1" : "any"}
          value={String(effectiveValue)}
          onChange={(event) => {
            const nextRaw = event.target.value;
            if (!nextRaw) {
              onChange(sectionName, field, undefined);
              return;
            }
            const nextValue =
              field.value_kind === "integer"
                ? Number.parseInt(nextRaw, 10)
                : Number(nextRaw);
            if (Number.isNaN(nextValue)) {
              return;
            }
            onChange(sectionName, field, nextValue);
          }}
        />
      ) : (
        <input
          id={`${sectionName}-${field.field_name}`}
          type="text"
          value={String(effectiveValue)}
          onChange={(event) => onChange(sectionName, field, event.target.value)}
        />
      )}

      <button
        type="button"
        className="ghost-button ghost-button--small"
        disabled={!isOverridden}
        onClick={() => onChange(sectionName, field, undefined)}
      >
        Reset
      </button>
    </div>
  );
}
