export type TextDiff = {
  inserted: string;
  deleted: string;
};

export type TextDiffRange = TextDiff & {
  prefixLength: number;
  suffixLength: number;
};

export function diffText(previous: string, current: string): TextDiff {
  const diff = diffTextRange(previous, current);
  return { inserted: diff.inserted, deleted: diff.deleted };
}

export function diffTextRange(
  previous: string,
  current: string,
): TextDiffRange {
  if (previous === current) {
    return { inserted: "", deleted: "", prefixLength: 0, suffixLength: 0 };
  }

  let prefixLength = 0;
  const maxPrefixLength = Math.min(previous.length, current.length);
  while (
    prefixLength < maxPrefixLength &&
    previous[prefixLength] === current[prefixLength]
  ) {
    prefixLength += 1;
  }

  let suffixLength = 0;
  while (
    suffixLength < previous.length - prefixLength &&
    suffixLength < current.length - prefixLength &&
    previous[previous.length - 1 - suffixLength] ===
      current[current.length - 1 - suffixLength]
  ) {
    suffixLength += 1;
  }

  return {
    deleted: previous.slice(prefixLength, previous.length - suffixLength),
    inserted: current.slice(prefixLength, current.length - suffixLength),
    prefixLength,
    suffixLength,
  };
}
