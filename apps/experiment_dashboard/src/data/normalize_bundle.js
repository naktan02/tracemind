const LEGACY_TRACK_NAMES = {
  central_lora_ssl: "central_peft_ssl",
  central_lora_initial_eval: "central_peft_initial_eval",
};

const LEGACY_FIELD_RENAMES = {
  lora_rank: "peft_adapter_rank",
  lora_alpha: "peft_adapter_alpha",
  lora_dropout: "peft_adapter_dropout",
  lora_bias: "peft_adapter_bias",
  lora_target_modules: "peft_adapter_target_modules",
  lora_use_rslora: "peft_adapter_use_rslora",
  lora_use_dora: "peft_adapter_use_dora",
};

export function normalizeDashboardBundle(bundle) {
  const runs = (bundle.runs ?? []).map(normalizeRunRecord);
  const flSslRuns = (bundle.fl_ssl_runs ?? []).map(normalizeRunRecord);
  return {
    ...bundle,
    filters: normalizeDashboardFilters(bundle.filters ?? {}),
    runs,
    fl_ssl_runs: flSslRuns,
  };
}

function normalizeRunRecord(row) {
  const normalized = {
    ...row,
    track: LEGACY_TRACK_NAMES[row.track] ?? row.track,
  };
  if (
    normalized.payload_adapter_kind === undefined &&
    normalized.adapter_family_name !== undefined
  ) {
    normalized.payload_adapter_kind = normalized.adapter_family_name;
  }
  for (const [legacyKey, currentKey] of Object.entries(LEGACY_FIELD_RENAMES)) {
    if (normalized[currentKey] === undefined && normalized[legacyKey] !== undefined) {
      normalized[currentKey] = normalized[legacyKey];
    }
  }
  return normalized;
}

function normalizeDashboardFilters(filters) {
  return {
    ...filters,
    initial_checkpoints: filters.initial_checkpoints ?? [],
    backbone_model_ids: filters.backbone_model_ids ?? [],
    created_dates: filters.created_dates ?? [],
    label_budgets: filters.label_budgets ?? [],
    label_budget_counts_per_class: filters.label_budget_counts_per_class ?? [],
    train_batch_sizes: filters.train_batch_sizes ?? [],
    labeled_batch_sizes: filters.labeled_batch_sizes ?? [],
    unlabeled_batch_sizes: filters.unlabeled_batch_sizes ?? [],
    eval_batch_sizes: filters.eval_batch_sizes ?? [],
    peft_adapter_ranks: filters.peft_adapter_ranks ?? filters.lora_ranks ?? [],
    peft_adapter_alphas: filters.peft_adapter_alphas ?? filters.lora_alphas ?? [],
    peft_adapter_use_rslora_values:
      filters.peft_adapter_use_rslora_values ??
      filters.lora_use_rslora_values ??
      [],
    peft_adapter_use_dora_values:
      filters.peft_adapter_use_dora_values ?? filters.lora_use_dora_values ?? [],
  };
}
