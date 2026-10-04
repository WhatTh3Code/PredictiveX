import { useEffect, useRef, useState } from "react";
import { pct } from "../lib/format";

interface Props {
  probability: number;
  medium: number;
  high: number;
  compareTo?: number; // what-if: previous probability, drawn as a hollow marker
}

const H = 92;
const PAD = 12;
const TRACK_Y = 34;
const TRACK_H = 16;

/** The PredictiveX instrument: a linear 0-100% probability rule with the risk zones drawn at
 *  the thresholds the training pipeline derived. The needle is the model output. */
export function RiskScale({ probability, medium, high, compareTo }: Props) {
  // Draw at the container's real pixel width so labels stay legible on phones.
  const ref = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setW(Math.max(260, Math.round(el.getBoundingClientRect().width)));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const x = (p: number) => PAD + Math.min(Math.max(p, 0), 1) * (W - 2 * PAD);
  const zones = [
    { from: 0, to: medium, color: "var(--good)", label: "LOW" },
    { from: medium, to: high, color: "var(--warning)", label: "MEDIUM" },
    { from: high, to: 1, color: "var(--critical)", label: "HIGH" },
  ];
  const needle = x(probability);
  return (
    <div ref={ref}>
    <svg className="scale-svg" viewBox={`0 0 ${W} ${H}`} role="img" data-testid="risk-scale"
      aria-label={`Failure probability ${pct(probability)} on a scale with MEDIUM from ${pct(medium)} and HIGH from ${pct(high)}`}>
      {zones.map((z) => (
        <g key={z.label}>
          <rect x={x(z.from)} y={TRACK_Y} width={Math.max(x(z.to) - x(z.from) - 2, 1)} height={TRACK_H}
            fill={z.color} opacity={0.28} rx={2} />
          <rect x={x(z.from)} y={TRACK_Y + TRACK_H - 3} width={Math.max(x(z.to) - x(z.from) - 2, 1)} height={3}
            fill={z.color} rx={1} />
          {x(z.to) - x(z.from) > 64 && (
            <text x={x(z.from) + 8} y={TRACK_Y + 11.5} style={{ font: "600 10.5px var(--font-ui)", fill: "var(--ink-2)" }}>
              {z.label}
            </text>
          )}
        </g>
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={TRACK_Y + TRACK_H + 2} y2={TRACK_Y + TRACK_H + 7} stroke="var(--rule)" />
          <text x={x(t)} y={TRACK_Y + TRACK_H + 20} textAnchor="middle" style={{ font: "11px var(--font-gauge)", fill: "var(--muted)" }}>
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      {[medium, high].map((t, i) => (
        <g key={i}>
          <line x1={x(t)} x2={x(t)} y1={TRACK_Y - 5} y2={TRACK_Y + TRACK_H} stroke="var(--ink-2)" strokeWidth={1} />
          <text x={x(t)} y={TRACK_Y - 9} textAnchor={i === 0 ? "start" : "middle"} dx={i === 0 ? -2 : 0}
            style={{ font: "11px var(--font-gauge)", fill: "var(--ink-2)" }}>
            {pct(t)}
          </text>
        </g>
      ))}
      {compareTo !== undefined && (
        <g aria-hidden="true">
          <line x1={x(compareTo)} x2={x(compareTo)} y1={TRACK_Y - 2} y2={TRACK_Y + TRACK_H + 2} stroke="var(--steel)" strokeWidth={2} strokeDasharray="3 2" />
        </g>
      )}
      <g data-testid="risk-needle">
        <line x1={needle} x2={needle} y1={TRACK_Y - 14} y2={TRACK_Y + TRACK_H + 4} stroke="var(--ink)" strokeWidth={2.5} />
        <path d={`M${needle - 6},${TRACK_Y - 20} L${needle + 6},${TRACK_Y - 20} L${needle},${TRACK_Y - 12} Z`} fill="var(--ink)" />
      </g>
    </svg>
    </div>
  );
}
