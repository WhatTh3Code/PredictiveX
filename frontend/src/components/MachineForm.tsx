import type { FormEvent } from "react";
import type { DemoCase, ModelInfo } from "../api";
import { FIELD_META, NUMERIC_FIELDS, type FieldError } from "../lib/format";

interface Props {
  draft: Record<string, string>;
  onChange: (d: Record<string, string>) => void;
  onSubmit?: () => void;
  errors: FieldError[];
  ranges: ModelInfo["training_ranges"] | null;
  submitLabel?: string;
  busy?: boolean;
  idPrefix?: string;
}

export function MachineForm({ draft, onChange, onSubmit, errors, ranges, submitLabel = "Predict", busy, idPrefix = "m" }: Props) {
  const errFor = (f: string) => errors.find((e) => e.field === f)?.message;
  const set = (k: string, v: string) => onChange({ ...draft, [k]: v });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit?.();
  };
  return (
    <form onSubmit={submit} noValidate data-testid="machine-form">
      <div className="field" style={{ gridTemplateColumns: "1fr auto" }}>
        <label id={`${idPrefix}-type-label`}>Product type</label>
        <div className="seg" role="group" aria-labelledby={`${idPrefix}-type-label`}>
          {(["L", "M", "H"] as const).map((t) => (
            <button type="button" key={t} aria-pressed={draft.type === t} onClick={() => set("type", t)}
              title={{ L: "Low quality (60% of products)", M: "Medium quality (30%)", H: "High quality (10%)" }[t]}>
              {t}
            </button>
          ))}
        </div>
        {errFor("type") && <div className="field-error" role="alert">{errFor("type")}</div>}
      </div>
      {NUMERIC_FIELDS.map((f) => {
        const meta = FIELD_META[f];
        const r = ranges?.[f];
        const err = errFor(f);
        const v = Number(draft[f]);
        return (
          <div className={`field${err ? " has-error" : ""}`} key={f}>
            <label htmlFor={`${idPrefix}-${f}`}>
              <b>{meta.label}</b> ({meta.unit})
            </label>
            <input id={`${idPrefix}-${f}`} name={f} type="text" inputMode="decimal" value={draft[f] ?? ""}
              aria-invalid={Boolean(err)} aria-describedby={err ? `${idPrefix}-${f}-err` : undefined}
              onChange={(e) => set(f, e.target.value)} />
            {r && (
              <>
                <input type="range" aria-label={`${meta.label} slider`} min={r.min} max={r.max} step={meta.step}
                  value={Number.isFinite(v) ? Math.min(Math.max(v, r.min), r.max) : r.min}
                  onChange={(e) => set(f, e.target.value)} />
                <div className="range-ends"><span>{r.min}</span><span>training range</span><span>{r.max}</span></div>
              </>
            )}
            {err && <div className="field-error" id={`${idPrefix}-${f}-err`} role="alert">{err}</div>}
          </div>
        );
      })}
      {onSubmit && (
        <button className="btn" type="submit" disabled={busy} style={{ width: "100%", marginTop: 4 }} data-testid="predict-button">
          {busy ? "Predicting…" : submitLabel}
        </button>
      )}
    </form>
  );
}

export function DemoPicker({ cases, active, onPick }: { cases: DemoCase[]; active: string | null; onPick: (c: DemoCase) => void }) {
  return (
    <div className="demo-list" role="group" aria-label="Demo machines" data-testid="demo-picker">
      {cases.map((c) => (
        <button key={c.id} type="button" className="chip" aria-pressed={active === c.id} onClick={() => onPick(c)}
          title={`${c.selection_rule}. ${c.source}`} data-case={c.id}>
          {c.title}
        </button>
      ))}
    </div>
  );
}
