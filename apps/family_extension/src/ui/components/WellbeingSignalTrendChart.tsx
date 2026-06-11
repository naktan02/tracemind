import type {
  WellbeingSignalRange,
  WellbeingSignalTimeseriesPayload,
} from "../../contracts/generated";
import { formatComputedAtLabel, formatRangeLabel } from "../lib/formatters";

type WellbeingSignalTrendChartProps = {
  activeRange: WellbeingSignalRange;
  onRangeChange: (nextRange: WellbeingSignalRange) => void;
  timeseries: WellbeingSignalTimeseriesPayload;
};

const SUPPORTED_RANGES: WellbeingSignalRange[] = ["7d", "14d", "30d"];

function buildPolylinePoints(values: number[]): string {
  if (values.length === 0) {
    return "";
  }

  const width = 100;
  const height = 100;

  return values
    .map((value, index) => {
      const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
      const clampedValue = Math.min(Math.max(value, 0), 100);
      const y = height - clampedValue;
      return `${x},${y}`;
    })
    .join(" ");
}

function buildPointMarkers(values: number[]) {
  return values.map((value, index) => {
    const x = values.length === 1 ? 50 : (index / (values.length - 1)) * 100;
    const clampedValue = Math.min(Math.max(value, 0), 100);
    const y = 100 - clampedValue;
    return { x, y, key: `${index}-${Math.round(value)}` };
  });
}

export function WellbeingSignalTrendChart({
  activeRange,
  onRangeChange,
  timeseries,
}: WellbeingSignalTrendChartProps) {
  const points = timeseries.points.map((point) => point.signal_score);
  const polylinePoints = buildPolylinePoints(points);
  const pointMarkers = buildPointMarkers(points);
  const latestPoint =
    timeseries.points.length === 0
      ? null
      : timeseries.points[timeseries.points.length - 1];

  return (
    <section className="surface-card trend-card">
      <div className="trend-card-header">
        <div>
          <p className="card-label">최근 변화</p>
          <p className="section-copy">
            최근 상태가 어떻게 변했는지 본인이 확인할 수 있는 그래프입니다.
          </p>
        </div>
        <div className="range-toggle" role="tablist" aria-label="timeseries range">
          {SUPPORTED_RANGES.map((range) => (
            <button
              key={range}
              className={
                range === activeRange ? "range-pill active" : "range-pill"
              }
              type="button"
              onClick={() => onRangeChange(range)}
            >
              {formatRangeLabel(range)}
            </button>
          ))}
        </div>
      </div>

      <div className="trend-chart-shell">
        <div className="trend-axis-labels" aria-hidden="true">
          <span>0</span>
          <span>25</span>
          <span>50</span>
          <span>75</span>
          <span>100</span>
        </div>
        <svg
          className="trend-chart"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          role="img"
          aria-label={`${formatRangeLabel(timeseries.range)} 위험도 변화 추이`}
        >
          <line x1="0" y1="0" x2="100" y2="0" className="trend-grid-line" />
          <line x1="0" y1="25" x2="100" y2="25" className="trend-grid-line" />
          <line x1="0" y1="50" x2="100" y2="50" className="trend-grid-line" />
          <line x1="0" y1="75" x2="100" y2="75" className="trend-grid-line" />
          <line x1="0" y1="100" x2="100" y2="100" className="trend-grid-line" />
          <polyline
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            points={polylinePoints}
          />
          {pointMarkers.map((point) => (
            <circle
              key={point.key}
              className="trend-point"
              cx={point.x}
              cy={point.y}
              r="1.4"
            />
          ))}
        </svg>
      </div>

      <div className="trend-footer">
        <span>
          기준 시각 {formatComputedAtLabel(timeseries.computed_at)}
        </span>
        {latestPoint != null && (
          <span>
            최근 위험도 {Math.round(latestPoint.signal_score)} /{" "}
            {formatComputedAtLabel(latestPoint.ts)}
          </span>
        )}
      </div>
    </section>
  );
}
