import type { ReactNode } from "react";
import { niceTicks, scale } from "../lib/format";

export const SERIES = ["var(--series-1, #2a78d6)", "#eb6834", "#1baf7a", "#eda100"];

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <ul className="legend">
      {items.map((i) => (
        <li key={i.label}>
          <span className="swatch" style={{ background: i.color, opacity: i.dashed ? 0.6 : 1 }} />
          {i.label}
        </li>
      ))}
    </ul>
  );
}

/* --------------------------------------------------------------------- horizontal bars */
export function HBars({
  rows, format, testId, color = "var(--normal)",
}: {
  rows: { label: string; value: number; err?: number; note?: string }[];
  format: (v: number) => string;
  testId?: string;
  color?: string;
}) {
  const W = 500;
  const LABEL = 190;
  const ROW = 28;
  const max = Math.max(...rows.map((r) => r.value + (r.err ?? 0)), 1e-9);
  const x = scale(0, max, 0, W - LABEL - 70);
  const H = rows.length * ROW + 4;
  return (
    <svg className="chart" style={{ maxWidth: W }} viewBox={`0 0 ${W} ${H}`} role="img" data-testid={testId}>
      {rows.map((r, i) => {
        const y = i * ROW;
        return (
          <g key={r.label}>
            <title>{`${r.label}: ${format(r.value)}${r.err ? ` ± ${format(r.err)}` : ""}${r.note ? ` — ${r.note}` : ""}`}</title>
            <rect className="hit" x={0} y={y} width={W} height={ROW} />
            <text x={0} y={y + 18} style={{ fill: "var(--ink)", fontSize: 12.5 }}>{r.label}</text>
            <rect className="mark" x={LABEL} y={y + 7} width={Math.max(x(Math.max(r.value, 0)), 1.5)} height={ROW - 14} rx={2} fill={color} />
            {r.err !== undefined && r.err > 0 && (
              <line x1={LABEL + x(Math.max(r.value - r.err, 0))} x2={LABEL + x(r.value + r.err)} y1={y + ROW / 2} y2={y + ROW / 2}
                stroke="var(--ink-2)" strokeWidth={1} />
            )}
            <text x={LABEL + x(Math.max(r.value, 0) + (r.err ?? 0)) + 6} y={y + 18} className="value-label">{format(r.value)}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* --------------------------------------------------------------------- grouped bars */
export function GroupedBars({
  groups, series, values, highlight, testId,
}: {
  groups: string[];
  series: string[];
  values: number[][]; // [series][group]
  highlight?: number;
  testId?: string;
}) {
  const W = 680;
  const H = 240;
  const M = { l: 36, r: 8, t: 10, b: 28 };
  const y = scale(0, 1, H - M.b, M.t);
  const gw = (W - M.l - M.r) / groups.length;
  const bw = Math.min(22, (gw - 16) / series.length);
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" style={{ maxWidth: W }} viewBox={`0 0 ${W} ${H}`} role="img" data-testid={testId}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
          </g>
        ))}
        {groups.map((g, gi) => {
          const gx = M.l + gi * gw + (gw - bw * series.length - 2 * (series.length - 1)) / 2;
          return (
            <g key={g}>
              {series.map((s, si) => {
                const v = values[si]?.[gi] ?? 0;
                const bx = gx + si * (bw + 2);
                return (
                  <g key={s}>
                    <title>{`${s} — ${g}: ${v.toFixed(3)}`}</title>
                    <rect className="mark" x={bx} y={y(v)} width={bw} height={Math.max(y(0) - y(v), 1)} rx={2}
                      fill={SERIES[si]} opacity={highlight === undefined || highlight === si ? 1 : 0.45} />
                  </g>
                );
              })}
              <text x={M.l + gi * gw + gw / 2} y={H - 8} textAnchor="middle" className="axis-label">{g}</text>
            </g>
          );
        })}
        <line className="baseline" x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} />
      </svg>
      <Legend items={series.map((s, i) => ({ label: s, color: SERIES[i] ?? "#999" }))} />
    </figure>
  );
}

/* --------------------------------------------------------------------- lines */
export interface LineSeries {
  name: string;
  color: string;
  points: [number, number][];
  marker?: boolean;
}

export function LineChart({
  series, xLabel, yLabel, xDomain, yDomain, diagonal, refY, markers, height = 260, xFormat, yFormat, testId, legend = true,
}: {
  series: LineSeries[];
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  diagonal?: boolean;
  refY?: { y: number; label: string };
  markers?: { x: number; y: number; label: string }[];
  height?: number;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  testId?: string;
  legend?: boolean;
}) {
  const W = 520;
  const H = height;
  const M = { l: 46, r: 12, t: 12, b: 40 };
  const x = scale(xDomain[0], xDomain[1], M.l, W - M.r);
  const y = scale(yDomain[0], yDomain[1], H - M.b, M.t);
  const xf = xFormat ?? ((v: number) => String(v));
  const yf = yFormat ?? ((v: number) => String(v));
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" style={{ maxWidth: W }} viewBox={`0 0 ${W} ${H}`} role="img" data-testid={testId} aria-label={`${yLabel} vs ${xLabel}`}>
        {niceTicks(yDomain[0], yDomain[1], 4).map((t) => (
          <g key={`y${t}`}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end">{yf(t)}</text>
          </g>
        ))}
        {niceTicks(xDomain[0], xDomain[1], 5).map((t) => (
          <text key={`x${t}`} x={x(t)} y={H - M.b + 16} textAnchor="middle">{xf(t)}</text>
        ))}
        <line className="baseline" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        {diagonal && <line x1={x(xDomain[0])} y1={y(yDomain[0])} x2={x(xDomain[1])} y2={y(yDomain[1])} stroke="var(--rule)" strokeWidth={1} />}
        {refY && (
          <g>
            <line x1={M.l} x2={W - M.r} y1={y(refY.y)} y2={y(refY.y)} stroke="var(--muted)" strokeWidth={1} />
            <text x={W - M.r} y={y(refY.y) - 5} textAnchor="end">{refY.label}</text>
          </g>
        )}
        {series.map((s) => (
          <g key={s.name}>
            <path d={s.points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")}
              fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
            {s.marker && s.points.map((p, i) => (
              <g key={i}>
                <title>{`${s.name}: ${xLabel} ${xf(p[0])}, ${yLabel} ${yf(p[1])}`}</title>
                <circle cx={x(p[0])} cy={y(p[1])} r={10} className="hit" />
                <circle cx={x(p[0])} cy={y(p[1])} r={4} fill={s.color} stroke="var(--panel)" strokeWidth={2} />
              </g>
            ))}
          </g>
        ))}
        {markers?.map((m) => (
          <g key={m.label}>
            <circle cx={x(m.x)} cy={y(m.y)} r={6} fill="var(--failed)" stroke="var(--panel)" strokeWidth={2} />
            <text x={x(m.x) - 10} y={y(m.y) + 20} textAnchor="end" className="value-label">{m.label}</text>
          </g>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="axis-label">{xLabel}</text>
        <text transform={`translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{yLabel}</text>
      </svg>
      {legend && series.length > 1 && <Legend items={series.map((s) => ({ label: s.name, color: s.color }))} />}
    </figure>
  );
}

/* --------------------------------------------------------------------- overlaid histogram */
export function ClassHistogram({ bins, normal, failed, label, testId }: { bins: number[]; normal: number[]; failed: number[]; label: string; testId?: string }) {
  const W = 300;
  const H = 150;
  const M = { l: 8, r: 8, t: 8, b: 32 };
  const nTot = normal.reduce((a, b) => a + b, 0) || 1;
  const fTot = failed.reduce((a, b) => a + b, 0) || 1;
  const nd = normal.map((c) => c / nTot);
  const fd = failed.map((c) => c / fTot);
  const lo = bins[0] ?? 0;
  const hi = bins[bins.length - 1] ?? 1;
  const x = scale(lo, hi, M.l, W - M.r);
  const y = scale(0, Math.max(...nd, ...fd, 1e-9), H - M.b, M.t);
  const step = (b: number[], d: number[]) =>
    d.map((v, i) => `${i ? "L" : "M"}${x(b[i] ?? 0)},${y(v)}L${x(b[i + 1] ?? 0)},${y(v)}`).join("");
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" data-testid={testId} aria-label={`Distribution of ${label} for normal vs failed`}>
      <path d={`${step(bins, nd)}L${x(hi)},${y(0)}L${x(lo)},${y(0)}Z`} fill="var(--normal)" opacity={0.22} />
      <path d={step(bins, nd)} fill="none" stroke="var(--normal)" strokeWidth={1.5} />
      <path d={`${step(bins, fd)}L${x(hi)},${y(0)}L${x(lo)},${y(0)}Z`} fill="var(--failed)" opacity={0.22} />
      <path d={step(bins, fd)} fill="none" stroke="var(--failed)" strokeWidth={2} />
      <line className="baseline" x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} />
      {niceTicks(lo, hi, 4).map((t) => (
        <text key={t} x={x(t)} y={H - M.b + 14} textAnchor="middle">{Math.abs(t) >= 1000 ? `${(t / 1000).toFixed(1)}k` : t}</text>
      ))}
      <text x={W / 2} y={H - 3} textAnchor="middle" className="axis-label">{label}</text>
    </svg>
  );
}

/* --------------------------------------------------------------------- scatter */
export function Scatter({
  points, xLabel, yLabel, testId,
}: {
  points: { x: number; y: number; failed: boolean }[];
  xLabel: string;
  yLabel: string;
  testId?: string;
}) {
  const W = 560;
  const H = 320;
  const M = { l: 46, r: 12, t: 10, b: 40 };
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xd: [number, number] = [Math.min(...xs), Math.max(...xs)];
  const yd: [number, number] = [0, Math.max(...ys) * 1.05];
  const x = scale(xd[0], xd[1], M.l, W - M.r);
  const y = scale(yd[0], yd[1], H - M.b, M.t);
  const ordered = [...points].sort((a, b) => Number(a.failed) - Number(b.failed));
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" style={{ maxWidth: W }} viewBox={`0 0 ${W} ${H}`} role="img" data-testid={testId}>
        {niceTicks(yd[0], yd[1], 4).map((t) => (
          <g key={t}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end">{t}</text>
          </g>
        ))}
        {niceTicks(xd[0], xd[1], 6).map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 16} textAnchor="middle">{t}</text>
        ))}
        {ordered.map((p, i) => (
          <circle key={i} cx={x(p.x)} cy={y(p.y)} r={p.failed ? 3.6 : 2.2}
            fill={p.failed ? "var(--failed)" : "var(--steel)"} opacity={p.failed ? 0.95 : 0.35}
            stroke={p.failed ? "var(--panel)" : "none"} strokeWidth={p.failed ? 1 : 0} />
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="axis-label">{xLabel}</text>
        <text transform={`translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{yLabel}</text>
      </svg>
      <Legend items={[{ label: "Normal (sample)", color: "var(--steel)" }, { label: "Failed (all)", color: "var(--failed)" }]} />
    </figure>
  );
}

/* --------------------------------------------------------------------- confusion matrix */
export function ConfusionMatrix({ cm, caption }: { cm: { tn: number; fp: number; fn: number; tp: number }; caption: ReactNode }) {
  const cells: [string, number, string][][] = [
    [["True negative", cm.tn, "Normal, predicted normal"], ["False alarm", cm.fp, "Normal, predicted failure"]],
    [["Missed failure", cm.fn, "Failed, predicted normal"], ["Caught failure", cm.tp, "Failed, predicted failure"]],
  ];
  return (
    <figure style={{ margin: 0 }} data-testid="confusion-matrix">
      <table>
        <thead>
          <tr><th></th><th className="n">Predicted normal</th><th className="n">Predicted failure</th></tr>
        </thead>
        <tbody>
          {cells.map((row, i) => (
            <tr key={i}>
              <th>{i === 0 ? "Actual normal" : "Actual failure"}</th>
              {row.map(([name, v, desc]) => (
                <td key={name} className="n" title={desc}>
                  <div style={{ fontFamily: "var(--font-gauge)", fontSize: 26, fontWeight: 600, color: "var(--ink)" }}>{v.toLocaleString()}</div>
                  <div className="note">{name}</div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <figcaption className="note" style={{ marginTop: 8 }}>{caption}</figcaption>
    </figure>
  );
}

/* --------------------------------------------------------------------- SHAP summary (beeswarm) */
export function ShapSummary({
  features, labels, sample, order, outputSpace,
}: {
  features: string[];
  labels: Record<string, string>;
  sample: { shap: number[]; value: number[] }[];
  order: string[];
  outputSpace: string;
}) {
  const W = 520;
  const LABEL = 175;
  const ROW = 30;
  const H = order.length * ROW + 36;
  const all = sample.flatMap((s) => s.shap);
  const lim = Math.max(Math.abs(Math.min(...all)), Math.abs(Math.max(...all)), 1e-9);
  const x = scale(-lim, lim, LABEL, W - 12);
  const ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"];
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" style={{ maxWidth: W }} viewBox={`0 0 ${W} ${H}`} role="img" data-testid="shap-summary">
        <line className="baseline" x1={x(0)} x2={x(0)} y1={0} y2={H - 30} />
        {order.map((f, r) => {
          const j = features.indexOf(f);
          const vals = sample.map((s) => s.value[j] ?? 0);
          const vmin = Math.min(...vals);
          const vmax = Math.max(...vals);
          const yC = r * ROW + ROW / 2;
          return (
            <g key={f}>
              <text x={0} y={yC + 4} style={{ fill: "var(--ink)", fontSize: 12.5 }}>{labels[f] ?? f}</text>
              {sample.map((s, i) => {
                const v = s.shap[j] ?? 0;
                const t = vmax > vmin ? ((s.value[j] ?? 0) - vmin) / (vmax - vmin) : 0.5;
                const jitter = (((i * 7919) % 97) / 97 - 0.5) * (ROW - 10);
                return <circle key={i} cx={x(v)} cy={yC + jitter} r={2.2} fill={ramp[Math.min(4, Math.floor(t * 5))]} opacity={0.85} />;
              })}
            </g>
          );
        })}
        {[-lim, 0, lim].map((t) => (
          <text key={t} x={x(t)} y={H - 16} textAnchor="middle">
            {outputSpace === "probability" ? `${t > 0 ? "+" : ""}${(t * 100).toFixed(0)} pts` : t.toFixed(1)}
          </text>
        ))}
        <text x={(LABEL + W) / 2} y={H - 2} textAnchor="middle" className="axis-label">
          SHAP value (← lowers risk · raises risk →)
        </text>
      </svg>
      <div className="row note" style={{ marginTop: 6 }}>
        Feature value: low
        <span style={{ display: "inline-flex" }}>
          {ramp.map((c) => <span key={c} className="swatch" style={{ background: c, borderRadius: 0, width: 18 }} />)}
        </span>
        high
      </div>
    </figure>
  );
}
