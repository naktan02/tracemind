export function normalizeEditorSnapshotText(text: string): string {
  return text.replace(/[\u200b\ufeff]/g, "");
}
