import { formatMetric } from "../../shared/formatting/numbers.js";
import { shortRun } from "../../shared/formatting/text.js";

export function runId(row) {
  return row.run_id ?? row.id ?? "-";
}

export function algorithmName(row) {
  if (
    row.fl_composition_mode === "method_owned" ||
    row.fl_execution_role === "method_owned" ||
    row.method_family === "fedmatch"
  ) {
    return row.fl_descriptor_name ?? row.algorithm_name ?? row.method_family ?? row.method_name ?? "-";
  }
  return row.algorithm_name ?? row.method_name ?? row.ssl_method_name ?? "-";
}

export function localRegularizerLabel(row) {
  const name = row.local_regularizer_name ?? inferRegularizerFromRunId(runId(row));
  if (!name || name === "none") return "none";
  const mu = row.local_regularizer_mu ?? inferFedProxMuFromRunId(runId(row));
  return mu === null || mu === undefined ? name : `${name}_mu${mu}`;
}

export function adapterKind(row) {
  return displayAdapterKind(
    row.payload_adapter_kind ??
      row.protocol?.round_runtime?.payload_adapter_kind ??
      row.peft_adapter_name ??
      "-",
  );
}

export function adapterConfigLabel(row) {
  return [
    `adapter=${row.peft_adapter_name ?? "-"}`,
    `r=${row.peft_adapter_rank ?? "-"}`,
    `alpha=${row.peft_adapter_alpha ?? "-"}`,
    `dropout=${row.peft_adapter_dropout ?? "-"}`,
  ].join(" · ");
}

export function dataSourceLabel(row) {
  const labeled = row.labeled_dataset_name ?? extractRunIdPart(row, "labeled");
  const unlabeled = row.unlabeled_dataset_name ?? extractRunIdPart(row, "unlabeled");
  return `L:${labeled ?? "?"} U:${unlabeled ?? "?"}`;
}

export function labelBudgetLabel(row) {
  const match = String(row.selection_slug ?? runId(row)).match(/labels_pc(\d+)/);
  return match ? `pc${match[1]}` : "pc?";
}

export function runDescriptor(row) {
  const cost = row.communication_cost;
  const costValue = typeof cost === "object" && cost !== null ? cost.value : cost;
  return [
    dataSourceLabel(row),
    labelBudgetLabel(row),
    adapterConfigLabel(row),
    `payload=${adapterKind(row)}`,
    `agg=${row.aggregation_backend_name ?? "-"}`,
    `regularizer=${localRegularizerLabel(row)}`,
    `clients=${row.client_count ?? "-"}`,
    `rounds=${row.completed_rounds ?? "-"}/${row.round_budget ?? "-"}`,
    `updates=${costValue ?? "-"}`,
    `seed=${row.seed ?? "-"}`,
  ].join(" · ");
}

function displayAdapterKind(value) {
  const raw = String(value ?? "-");
  if (raw === "peft_classifier") return "classifier";
  if (raw === "peft_text_encoder_lora") return "lora text encoder";
  return raw.replace(/^peft_/, "");
}

export function compactRunLabel(row) {
  return [
    algorithmName(row),
    localRegularizerLabel(row),
    `r${row.completed_rounds ?? "?"}`,
    `seed${row.seed ?? "?"}`,
  ].join(" · ");
}

export function runDisplayLabel(row, aliases) {
  return aliases[runId(row)] || compactRunLabel(row);
}

export function compactRunSubLabel(row) {
  return [
    labelBudgetLabel(row),
    `clients=${row.client_count ?? "-"}`,
    `rank=${row.peft_adapter_rank ?? "-"}`,
    runSuffix(row),
  ].join(" · ");
}

export function roundLegendLabel(row, aliases) {
  return aliases[runId(row)] || algorithmName(row);
}

export function runDetailLabel(row) {
  return [
    algorithmName(row),
    localRegularizerLabel(row),
    `clients=${row.client_count ?? "-"}`,
    `rounds=${row.completed_rounds ?? "-"}/${row.round_budget ?? "-"}`,
    `alpha=${formatMetric(row.shard_alpha)}`,
    `seed=${row.seed ?? "-"}`,
    runSuffix(row),
  ].join(" · ");
}

export function runSuffix(row) {
  const id = runId(row);
  const timestampMatch = String(id).match(/(\d{8}T\d{6}Z)$/);
  if (timestampMatch) return timestampMatch[1];
  const parts = String(id).split("__").filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : shortRun(id);
}

function inferRegularizerFromRunId(id) {
  return String(id ?? "").includes("fedprox") ? "fedprox" : "none";
}

function inferFedProxMuFromRunId(id) {
  const match = String(id ?? "").match(/fedprox_mu([0-9.]+)/);
  return match ? match[1].replace(/\.$/, "") : null;
}

function extractRunIdPart(row, prefix) {
  const source = String(row.selection_slug ?? runId(row));
  const next = prefix === "labeled" ? "unlabeled" : "labels_pc";
  const match = source.match(new RegExp(`${prefix}-(.+?)_${next}`));
  return match ? match[1] : null;
}
