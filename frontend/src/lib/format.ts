// Pure helpers (unit-tested in tests/format.test.ts).

import type { MachineInput, RiskLevel } from "../api";

export const FIELD_META: Record<
  Exclude<keyof MachineInput, "type">,
  { label: string; unit: string; step: number; min: number; max: number }
> = {
  air_temp_k: { label: "Air temperature", unit: "K", step: 0.1, min: 250, max: 350 },
  process_temp_k: { label: "Process temperature", unit: "K", step: 0.1, min: 250, max: 400 },
  rotational_speed_rpm: { label: "Rotational speed", unit: "rpm", step: 1, min: 1, max: 10000 },
  torque_nm: { label: "Torque", unit: "Nm", step: 0.1, min: 0, max: 500 },
  tool_wear_min: { label: "Tool wear", unit: "min", step: 1, min: 0, max: 1000 },
};

export const NUMERIC_FIELDS = Object.keys(FIELD_META) as (keyof typeof FIELD_META)[];

export const TYPE_LABELS: Record<string, string> = { L: "L — low quality", M: "M — medium quality", H: "H — high quality" };

export const MODE_NAMES: Record<string, string> = {
  twf: "Tool wear failure",
  hdf: "Heat dissipation failure",
  pwf: "Power failure",
  osf: "Overstrain failure",
  rnf: "Random failure",
};

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function int(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return Math.round(value).toLocaleString("en-US");
}

export function signedPoints(delta: number): string {
  const pts = delta * 100;
  const sign = pts > 0 ? "+" : pts < 0 ? "−" : "±";
  return `${sign}${Math.abs(pts).toFixed(1)} pts`;
}

export function riskDescription(level: RiskLevel): string {
  switch (level) {
    case "HIGH":
      return "At or above the decision threshold: the model flags this machine as likely to fail.";
    case "MEDIUM":
      return "Watch list: above the threshold that catches 95% of failures, below the decision threshold.";
    case "LOW":
      return "Below the watch threshold.";
  }
}

export interface FieldError {
  field: keyof MachineInput;
  message: string;
}

/** Client-side mirror of the API's hard limits, so the form can explain problems instantly.
 *  The server re-validates every request; this is a convenience, not the gatekeeper. */
export function validateMachine(draft: Record<string, string>): { machine: MachineInput | null; errors: FieldError[] } {
  const errors: FieldError[] = [];
  const type = draft.type;
  if (type !== "L" && type !== "M" && type !== "H") {
    errors.push({ field: "type", message: "Choose L, M or H." });
  }
  const values: Partial<Record<keyof typeof FIELD_META, number>> = {};
  for (const f of NUMERIC_FIELDS) {
    const meta = FIELD_META[f];
    const raw = (draft[f] ?? "").trim();
    if (raw === "") {
      errors.push({ field: f, message: `Enter ${meta.label.toLowerCase()}.` });
      continue;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) {
      errors.push({ field: f, message: `${meta.label} must be a number.` });
    } else if (v < meta.min || v > meta.max) {
      errors.push({ field: f, message: `${meta.label} must be between ${meta.min} and ${meta.max} ${meta.unit}.` });
    } else {
      values[f] = v;
    }
  }
  if (errors.length) return { machine: null, errors };
  return { machine: { type: type as MachineInput["type"], ...(values as Omit<MachineInput, "type">) }, errors };
}

export function machineToDraft(m: object): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(m)) out[k] = v === null || v === undefined ? "" : String(v);
  return out;
}

/** Linear scale helper for SVG charts. */
export function scale(d0: number, d1: number, r0: number, r1: number): (v: number) => number {
  const span = d1 - d0 || 1;
  return (v: number) => r0 + ((v - d0) / span) * (r1 - r0);
}

export function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min;
  if (span <= 0) return [min];
  const rough = span / count;
  const mag = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? mag * 10;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}
