import { formatMetric } from "../../../shared/formatting/numbers.js";
import {
  compactDate,
  compactDateTime,
  shortRun,
} from "../../../shared/formatting/text.js";

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
    row.payload_adapter_kind ?? row.protocol?.round_runtime?.payload_adapter_kind,
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
  if (row.label_budget_name) return row.label_budget_name;
  if (row.label_budget_count_per_class) return `pc${row.label_budget_count_per_class}`;
  const source = [
    row.selection_slug,
    row.run_id,
    row.report_path,
    row.labeled_dataset_name,
  ].join(" ");
  const labelsPcMatch = source.match(/labels_pc(\d+)/);
  if (labelsPcMatch) return `pc${labelsPcMatch[1]}`;
  const perClassMatch = source.match(/labeled(\d+)_per_class/);
  if (perClassMatch) return `pc${perClassMatch[1]}`;
  const inferred = inferLabelBudgetFromUniqueRows(row);
  return inferred ? `pc${inferred}` : "pc?";
}

export function initialCheckpointLabel(row) {
  return checkpointDisplayValue(row.initial_checkpoint_name);
}

export function runCreatedDateLabel(row) {
  return compactDate(row.created_at);
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
    `checkpoint=${initialCheckpointLabel(row)}`,
    `created=${compactDateTime(row.created_at)}`,
    `clients=${row.client_count ?? "-"}`,
    `rounds=${row.completed_rounds ?? "-"}/${row.round_budget ?? "-"}`,
    `updates=${costValue ?? "-"}`,
    `seed=${row.seed ?? "-"}`,
  ].join(" · ");
}

export function runHoverDetail(row) {
  return [
    algorithmName(row),
    runDescriptor(row),
    `run_id=${runId(row)}`,
  ].join(" · ");
}

function displayAdapterKind(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return "unrecorded";
  if (raw === "peft_classifier") return "peft classifier";
  if (raw === "peft_text_encoder_lora") return "lora text encoder";
  return raw.replace(/^peft_/, "");
}

function checkpointDisplayValue(value) {
  const raw = String(value ?? "").trim();
  return raw || "unrecorded";
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
  const created = compactDateTime(row.created_at);
  return [
    labelBudgetLabel(row),
    `clients=${row.client_count ?? "-"}`,
    `rank=${row.peft_adapter_rank ?? "-"}`,
    `ckpt=${initialCheckpointLabel(row)}`,
    created !== "-" ? created : runSuffix(row),
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
    `checkpoint=${initialCheckpointLabel(row)}`,
    `created=${compactDateTime(row.created_at)}`,
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

function inferLabelBudgetFromUniqueRows(row) {
  const uniqueLabeledRows = Number(row.unique_labeled_row_count);
  if (!Number.isFinite(uniqueLabeledRows) || uniqueLabeledRows <= 0) return null;
  const labelCount = labelCountFromSchema(row);
  if (!labelCount || uniqueLabeledRows % labelCount !== 0) return null;
  return Math.round(uniqueLabeledRows / labelCount);
}

function labelCountFromSchema(row) {
  const rawParameters = row.peft_adapter_parameters_json;
  if (typeof rawParameters !== "string" || !rawParameters.trim()) return null;
  try {
    const parameters = JSON.parse(rawParameters);
    const labels = String(parameters.label_schema ?? "")
      .split(",")
      .map((label) => label.trim())
      .filter(Boolean);
    return labels.length > 0 ? labels.length : null;
  } catch (_error) {
    return null;
  }
}
