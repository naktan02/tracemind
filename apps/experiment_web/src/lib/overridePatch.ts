import type {
  CatalogSectionPayload,
  WorkspaceConfigScalar,
} from "../types";

export const EMPTY_OVERRIDE_JSON = "{}";

export interface ObjectParseResult {
  value: Record<string, WorkspaceConfigScalar>;
  error: string | null;
}

export function formatOverridePatch(
  patch: Record<string, WorkspaceConfigScalar>,
): string {
  const normalized = Object.fromEntries(
    Object.entries(patch).sort(([left], [right]) => left.localeCompare(right)),
  );
  return JSON.stringify(normalized, null, 2);
}

export function parseOverrideObject(input: string): ObjectParseResult {
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
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function buildSectionOverrideParseBySection(
  sections: CatalogSectionPayload[],
  overrideTextBySection: Record<string, string>,
): Record<string, ObjectParseResult> {
  const parseBySection: Record<string, ObjectParseResult> = {};
  for (const section of sections) {
    parseBySection[section.section_name] = parseOverrideObject(
      overrideTextBySection[section.section_name] ?? EMPTY_OVERRIDE_JSON,
    );
  }
  return parseBySection;
}

export function buildSectionOverrideValueBySection(
  overrideParseBySection: Record<string, ObjectParseResult>,
): Record<string, Record<string, WorkspaceConfigScalar>> {
  return Object.fromEntries(
    Object.entries(overrideParseBySection).map(([sectionName, parseResult]) => [
      sectionName,
      parseResult.value,
    ]),
  );
}

export function buildSectionOverrideErrors(
  sections: CatalogSectionPayload[],
  selectedItemNameBySection: Record<string, string | null>,
  overrideParseBySection: Record<string, ObjectParseResult>,
): string[] {
  const errors: string[] = [];

  for (const section of sections) {
    if (!selectedItemNameBySection[section.section_name]) {
      continue;
    }
    const parseResult =
      overrideParseBySection[section.section_name] ?? parseOverrideObject("{}");
    if (parseResult.error) {
      errors.push(`${section.display_name}: ${parseResult.error}`);
    }
  }

  return errors;
}

function isScalarOverrideValue(value: unknown): value is WorkspaceConfigScalar {
  return (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}
