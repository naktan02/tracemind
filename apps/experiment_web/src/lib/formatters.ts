export function formatMetadataValue(value: unknown): string {
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

export function formatScalarValue(value: string | number | boolean): string {
  return String(value);
}

export function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function asErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
