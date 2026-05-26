const SERIES_COLOR_STORAGE_KEYS = {
  central_compare: "tracemind_dashboard.central_compare_run_colors.v1",
  fl_round: "tracemind_dashboard.fl_round_run_colors.v1",
};

const RUN_ALIAS_STORAGE_KEYS = {
  central_overview: "tracemind_dashboard.central_overview_run_aliases.v1",
  fl_runs: "tracemind_dashboard.fl_run_aliases.v1",
  fl_round: "tracemind_dashboard.fl_round_run_aliases.v1",
};

export function loadStoredSeriesColors(scope) {
  const storageKey = SERIES_COLOR_STORAGE_KEYS[scope];
  if (!storageKey) {
    return {};
  }
  try {
    const rawValue = window.localStorage.getItem(storageKey);
    const parsed = rawValue ? JSON.parse(rawValue) : {};
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).filter(([_key, value]) =>
        isHexColor(String(value)),
      ),
    );
  } catch (_error) {
    return {};
  }
}

export function loadStoredRunAliases(scope) {
  const storageKey = RUN_ALIAS_STORAGE_KEYS[scope];
  if (!storageKey) {
    return {};
  }
  try {
    const rawValue = window.localStorage.getItem(storageKey);
    const parsed = rawValue ? JSON.parse(rawValue) : {};
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed)
        .map(([key, value]) => [key, String(value).trim()])
        .filter(([_key, value]) => value.length > 0),
    );
  } catch (_error) {
    return {};
  }
}

export function storeRunAliases(scope, aliases) {
  const storageKey = RUN_ALIAS_STORAGE_KEYS[scope];
  if (!storageKey) {
    return;
  }
  try {
    window.localStorage.setItem(storageKey, JSON.stringify(aliases));
  } catch (_error) {
    // localStorage가 막힌 환경에서는 현재 화면 상태만 유지한다.
  }
}

export function storeSeriesColors(scope, colors) {
  const storageKey = SERIES_COLOR_STORAGE_KEYS[scope];
  if (!storageKey) {
    return;
  }
  try {
    window.localStorage.setItem(storageKey, JSON.stringify(colors));
  } catch (_error) {
    // localStorage가 막힌 환경에서는 현재 화면 상태만 유지한다.
  }
}

function isHexColor(value) {
  return /^#[0-9a-fA-F]{6}$/.test(value);
}
