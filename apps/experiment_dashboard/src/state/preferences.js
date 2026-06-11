const SERIES_COLOR_STORAGE_KEYS = {
  central_compare: "tracemind_dashboard.central_compare_run_colors.v1",
  fl_round: "tracemind_dashboard.fl_round_run_colors.v1",
};

const RUN_ALIAS_STORAGE_KEYS = {
  central_overview: "tracemind_dashboard.central_overview_run_aliases.v1",
  central_compare: "tracemind_dashboard.central_compare_run_aliases.v1",
  fl_runs: "tracemind_dashboard.fl_run_aliases.v1",
  fl_round: "tracemind_dashboard.fl_round_run_aliases.v1",
};

export function loadStoredSeriesColors(scope) {
  return loadObject(SERIES_COLOR_STORAGE_KEYS[scope], isHexColor);
}

export function storeSeriesColors(scope, colors) {
  storeObject(SERIES_COLOR_STORAGE_KEYS[scope], colors);
}

export function loadStoredRunAliases(scope) {
  return loadObject(RUN_ALIAS_STORAGE_KEYS[scope], (value) => String(value).trim());
}

export function storeRunAliases(scope, aliases) {
  storeObject(RUN_ALIAS_STORAGE_KEYS[scope], aliases);
}

function loadObject(storageKey, valueFilter) {
  if (!storageKey) return {};
  try {
    const parsed = JSON.parse(window.localStorage.getItem(storageKey) ?? "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return Object.fromEntries(
      Object.entries(parsed)
        .map(([key, value]) => [key, String(value).trim()])
        .filter(([_key, value]) => value && valueFilter(value)),
    );
  } catch (_error) {
    return {};
  }
}

function storeObject(storageKey, value) {
  if (!storageKey) return;
  try {
    window.localStorage.setItem(storageKey, JSON.stringify(value));
  } catch (_error) {
    // localStorage가 막힌 환경에서는 현재 화면 상태만 유지한다.
  }
}

function isHexColor(value) {
  return /^#[0-9a-fA-F]{6}$/.test(value);
}
