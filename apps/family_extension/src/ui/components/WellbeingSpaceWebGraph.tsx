import type {
  WellbeingSignalLevel,
  WellbeingSpaceWebNodePayload,
  WellbeingSpaceWebPayload,
} from "../../contracts/generated";
import {
  formatComputedAtLabel,
  formatSignalLevelLabel,
  formatTrendLabel,
} from "../lib/formatters";

type WellbeingSpaceWebGraphProps = {
  spaceWeb: WellbeingSpaceWebPayload;
};

type PositionedNode = {
  node: WellbeingSpaceWebNodePayload;
  x: number;
  y: number;
  index: number;
};

function buildNodePositions(
  nodes: WellbeingSpaceWebNodePayload[],
): PositionedNode[] {
  if (nodes.length === 0) {
    return [];
  }
  if (nodes.length === 1) {
    return [{ node: nodes[0], x: 50, y: 50, index: 1 }];
  }

  const radius = 33;
  return nodes.map((node, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / nodes.length;
    return {
      node,
      x: 50 + radius * Math.cos(angle),
      y: 50 + radius * Math.sin(angle),
      index: index + 1,
    };
  });
}

function nodeRadius(intensity: number): number {
  return 5.2 + Math.min(Math.max(intensity, 0), 100) / 14;
}

function edgeWidth(weight: number): number {
  return 0.8 + Math.min(Math.max(weight, 0), 100) / 34;
}

function edgeOpacity(weight: number): number {
  return 0.16 + Math.min(Math.max(weight, 0), 100) / 140;
}

function levelClass(level: WellbeingSignalLevel): string {
  return `level-${level.replace("_", "-")}`;
}

export function WellbeingSpaceWebGraph({
  spaceWeb,
}: WellbeingSpaceWebGraphProps) {
  const positionedNodes = buildNodePositions(spaceWeb.nodes);
  const nodePositions = new Map(
    positionedNodes.map((positioned) => [positioned.node.id, positioned]),
  );

  return (
    <section className="surface-card space-web-card">
      <div className="trend-card-header">
        <div>
          <p className="card-label">내 공간웹</p>
          <p className="section-copy">최근 함께 움직인 상태 축</p>
        </div>
      </div>

      {spaceWeb.nodes.length === 0 ? (
        <div className="trend-chart-loading">공간웹 데이터가 아직 없습니다.</div>
      ) : (
        <div className="space-web-layout">
          <div className="space-web-chart-shell">
            <svg
              className="space-web-chart"
              viewBox="0 0 100 100"
              role="img"
              aria-label="카테고리 공간웹"
            >
              {spaceWeb.edges.map((edge) => {
                const source = nodePositions.get(edge.source);
                const target = nodePositions.get(edge.target);
                if (source == null || target == null) {
                  return null;
                }
                return (
                  <line
                    key={`${edge.source}-${edge.target}`}
                    className="space-web-edge"
                    x1={source.x}
                    y1={source.y}
                    x2={target.x}
                    y2={target.y}
                    strokeWidth={edgeWidth(edge.weight)}
                    opacity={edgeOpacity(edge.weight)}
                  />
                );
              })}
              {positionedNodes.map(({ node, x, y, index }) => (
                <g key={node.id}>
                  <circle
                    className={`space-web-node ${levelClass(node.level)}`}
                    cx={x}
                    cy={y}
                    r={nodeRadius(node.intensity)}
                  />
                  <text className="space-web-node-index" x={x} y={y + 1.5}>
                    {index}
                  </text>
                </g>
              ))}
            </svg>
          </div>

          <div className="space-web-legend">
            {positionedNodes.map(({ node, index }) => (
              <div className="space-web-legend-row" key={node.id}>
                <span className={`space-web-legend-index ${levelClass(node.level)}`}>
                  {index}
                </span>
                <span className="space-web-legend-main">
                  <strong>{node.label}</strong>
                  <span>
                    {formatSignalLevelLabel(node.level)} ·{" "}
                    {formatTrendLabel(node.trend)}
                  </span>
                </span>
                <span className="space-web-score">
                  {Math.round(node.intensity)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="trend-footer">
        <span>기준 시각 {formatComputedAtLabel(spaceWeb.computed_at)}</span>
        {spaceWeb.low_data && <span>데이터가 아직 적습니다</span>}
      </div>
    </section>
  );
}
