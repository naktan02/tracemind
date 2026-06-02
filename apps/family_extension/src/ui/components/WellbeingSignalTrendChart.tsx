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
  const maxValue = Math.max(...values, 100);
  const minValue = Math.min(...values, 0);
  const valueSpan = Math.max(maxValue - minValue, 1);

  return values
    .map((value, index) => {
      const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
      const normalized = (value - minValue) / valueSpan;
      const y = height - normalized * height;
      return `${x},${y}`;
    })
    .join(" ");
}

export function WellbeingSignalTrendChart({
  activeRange,
  onRangeChange,
  timeseries,
}: WellbeingSignalTrendChartProps) {
  const points = timeseries.points.map((point) => point.signal_score);
  const polylinePoints = buildPolylinePoints(points);
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
            전체 wellbeing signal만 보여주고, 내부 카테고리별 score는 숨깁니다.
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
        <div className="trend-band-grid" aria-hidden="true">
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
          aria-label={`${formatRangeLabel(timeseries.range)} wellbeing signal 추이`}
        >
          <rect
            x="0"
            y="0"
            width="100"
            height="25"
            className="trend-band very-high"
          />
          <rect x="0" y="25" width="100" height="25" className="trend-band high" />
          <rect
            x="0"
            y="50"
            width="100"
            height="25"
            className="trend-band moderate"
          />
          <rect
            x="0"
            y="75"
            width="100"
            height="25"
            className="trend-band low"
          />
          <polyline
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            points={polylinePoints}
          />
        </svg>
      </div>

      <div className="trend-footer">
        <span>
          기준 시각 {formatComputedAtLabel(timeseries.computed_at)}
        </span>
        {latestPoint != null && (
          <span>
            최근 값 {Math.round(latestPoint.signal_score)} /{" "}
            {formatComputedAtLabel(latestPoint.ts)}
          </span>
        )}
      </div>
    </section>
  );
}
