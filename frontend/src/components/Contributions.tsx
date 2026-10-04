import type { Contribution } from "../api";

/** Diverging SHAP bars for one prediction: right = pushes the model toward failure. */
export function Contributions({ items, limit = 9, outputSpace }: { items: Contribution[]; limit?: number; outputSpace: string }) {
  const rows = items.slice(0, limit);
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.shap)), 1e-9);
  const W = 760;
  const LABEL = 300;
  const VAL = 70;
  const ROW = 30;
  const mid = LABEL + (W - LABEL - VAL) / 2;
  const half = (W - LABEL - VAL) / 2 - 6;
  const H = rows.length * ROW + 26;
  const unit = outputSpace === "probability" ? "probability points" : "log-odds";
  const fmt = (v: number) => (outputSpace === "probability" ? `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(1)}` : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)}`);
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" style={{ maxWidth: W }} viewBox={`0 0 ${W} ${H}`} role="img" data-testid="contributions"
        aria-label="Feature contributions to this prediction">
        <text x={mid - 8} y={12} textAnchor="end">← lowers predicted risk</text>
        <text x={mid + 8} y={12}>raises predicted risk →</text>
        <line className="baseline" x1={mid} x2={mid} y1={18} y2={H} />
        {rows.map((r, i) => {
          const y = 22 + i * ROW;
          const len = (Math.abs(r.shap) / maxAbs) * half;
          const up = r.shap > 0;
          return (
            <g key={r.feature}>
              <title>{`${r.label} = ${r.display_value}: ${fmt(r.shap)} ${unit}${r.formula ? ` (engineered: ${r.formula})` : ""}`}</title>
              <rect className="hit" x={0} y={y} width={W} height={ROW} />
              <text x={0} y={y + 18} style={{ fill: "var(--ink)", fontSize: 12.5 }}>
                {r.label}
              </text>
              <text x={LABEL - 14} y={y + 18} textAnchor="end" style={{ fill: "var(--ink-2)", fontSize: 12 }}>
                {r.display_value}
              </text>
              <rect className="mark" x={up ? mid + 1 : mid - len - 1} y={y + 7} width={Math.max(len, 1.5)} height={ROW - 14}
                rx={2} fill={up ? "var(--up)" : "var(--down)"} />
              <text x={up ? mid + len + 6 : mid - len - 6} y={y + 18} textAnchor={up ? "start" : "end"} className="value-label">
                {fmt(r.shap)}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="note">
        SHAP values in {unit}, relative to the average training prediction. Bars show how much each input moved
        this prediction in the model; they describe the model's reasoning, not physical cause.
      </figcaption>
    </figure>
  );
}
