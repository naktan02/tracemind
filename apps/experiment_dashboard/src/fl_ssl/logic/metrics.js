import { numberOrNull, formatBytes, formatCount, formatMetric } from "../../shared/formatting/numbers.js";
import { FL_RUN_METRICS } from "./constants.js";

export function flRunMetricKeys(rows) {
  const discovered = new Set();
  for (const row of rows) {
    for (const metric of FL_RUN_METRICS) {
      if (flRunMetricValue(row, metric) !== null) {
        discovered.add(metric);
      }
    }
  }
  return FL_RUN_METRICS.filter((metric) => discovered.has(metric));
}

export function flMetric(row, metric) {
  if (row[metric] !== undefined) return row[metric];
  if (row.metrics?.primary?.[metric] !== undefined) return row.metrics.primary[metric];
  if (row.metrics?.secondary?.[metric] !== undefined) return row.metrics.secondary[metric];
  if (metric === "expected_calibration_error") {
    return row.metrics?.final_validation?.expected_calibration_error;
  }
  if (metric === "macro_f1") {
    return row.metrics?.final_validation?.macro_f1;
  }
  return null;
}

export function flRunMetricValue(row, metric) {
  if (metric === "accuracy_top_1") return row.final_accuracy_top_1 ?? flMetric(row, metric);
  if (metric === "communication_cost") return communicationCostValue(row);
  if (metric === "c2s_total_bytes") return communicationEstimateBytes(row, "c2s_total_bytes");
  if (metric === "s2c_total_bytes_estimated") {
    return communicationEstimateBytes(row, "s2c_total_bytes_estimated");
  }
  return flMetric(row, metric);
}

export function formatFlRunMetric(row, metric) {
  const value = flRunMetricValue(row, metric);
  if (metric === "c2s_total_bytes" || metric === "s2c_total_bytes_estimated") {
    return formatBytes(value);
  }
  if (
    metric === "communication_cost" ||
    metric.endsWith("_count") ||
    metric.endsWith("_bytes") ||
    metric.includes("row_exposure")
  ) {
    return formatCount(value);
  }
  return formatMetric(value);
}

export function roundPointValue(row, metric) {
  return numberOrNull(row[metric]);
}

function communicationCostValue(row) {
  const cost = flMetric(row, "communication_cost");
  return typeof cost === "object" && cost !== null ? cost.value : cost;
}

function communicationEstimateBytes(row, key) {
  const cost = flMetric(row, "communication_cost");
  if (typeof cost !== "object" || cost === null) return null;
  const estimates = cost.artifact_byte_estimates ?? cost.posthoc_byte_estimates;
  return typeof estimates === "object" && estimates !== null ? estimates[key] ?? null : null;
}
