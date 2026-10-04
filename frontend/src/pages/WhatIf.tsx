import { useEffect, useRef, useState } from "react";
import { api, ApiError, type MachineInput, type Simulation } from "../api";
import { DemoPicker, MachineForm } from "../components/MachineForm";
import { RiskScale } from "../components/RiskScale";
import { EmptyState, ErrorState, PageHead, RiskBadge, useAsync } from "../components/ui";
import { FIELD_META, machineToDraft, pct, signedPoints, validateMachine, type FieldError } from "../lib/format";
import { useSession } from "../state";

export function WhatIf() {
  const s = useSession();
  const info = useAsync(() => api.modelInfo(), []);
  const demos = useAsync(() => api.demoCases(), []);
  const base: MachineInput | null = s.prediction?.input ?? null;
  const [draft, setDraft] = useState<Record<string, string>>(() => (base ? machineToDraft(base) : {}));
  const [errors, setErrors] = useState<FieldError[]>([]);
  const [sim, setSim] = useState<Simulation | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const seq = useRef(0);

  // When the base machine changes, reset the modified copy.
  useEffect(() => {
    if (base) setDraft(machineToDraft(base));
    setSim(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(base)]);

  const run = async (d: Record<string, string>) => {
    if (!base) return;
    const { machine, errors: errs } = validateMachine(d);
    setErrors(errs);
    if (!machine) return;
    const changes: Partial<MachineInput> = {};
    (Object.keys(machine) as (keyof MachineInput)[]).forEach((k) => {
      if (machine[k] !== base[k]) (changes as Record<string, unknown>)[k] = machine[k];
    });
    if (Object.keys(changes).length === 0) {
      setSim(null);
      return;
    }
    const id = ++seq.current;
    setBusy(true);
    try {
      const r = await api.simulate(base, changes);
      if (id === seq.current) {
        setSim(r);
        setError(null);
      }
    } catch (e) {
      if (id === seq.current) setError(e instanceof ApiError ? e : new ApiError(0, "Unexpected error"));
    } finally {
      if (id === seq.current) setBusy(false);
    }
  };

  // Debounced re-run while the user drags sliders.
  useEffect(() => {
    if (!base) return;
    const t = setTimeout(() => run(draft), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(draft)]);

  return (
    <>
      <PageHead title="What-if simulator">
        Change one or more parameters of the current machine and the model re-scores it. This is a model simulation:
        it shows how the model's output responds, not a guaranteed real-world outcome.
      </PageHead>
      {!base ? (
        <div className="panel">
          <EmptyState title="Choose a machine to simulate" testId="whatif-empty">
            <p>Start from a demo machine, or predict your own machine first.</p>
            {demos.data && (
              <DemoPicker cases={demos.data.cases.filter((c) => c.id !== "invalid")} active={null}
                onPick={(c) => s.loadCase(c.id, c.input)} />
            )}
          </EmptyState>
        </div>
      ) : (
        <div className="grid-predict">
          <section className="panel">
            <div className="chart-tools">
              <h2>Modified parameters</h2>
              <button className="btn btn-quiet btn-small" onClick={() => setDraft(machineToDraft(base))}>Reset to original</button>
            </div>
            <MachineForm draft={draft} onChange={setDraft} errors={errors} ranges={info.data?.training_ranges ?? null} idPrefix="w" />
            <p className="note">The model re-runs automatically as you edit.</p>
          </section>
          <section aria-live="polite" data-testid="whatif-result">
            {error ? (
              <div className="panel"><ErrorState error={error} onRetry={() => run(draft)} /></div>
            ) : errors.length ? (
              <div className="panel"><div className="state" role="alert"><h3>Fix the highlighted fields to simulate</h3></div></div>
            ) : sim ? (
              <SimulationView sim={sim} busy={busy} />
            ) : (
              <div className="panel">
                <EmptyState title="Change a parameter to see its effect" testId="whatif-idle">
                  <p>Original estimate: <b>{pct(s.prediction?.failure_probability)}</b>{" "}
                    {s.prediction && <RiskBadge level={s.prediction.risk_level} />}</p>
                  <p className="note">Try lowering tool wear or torque on the high-risk demo machine.</p>
                </EmptyState>
              </div>
            )}
          </section>
        </div>
      )}
    </>
  );
}

function SimulationView({ sim, busy }: { sim: Simulation; busy: boolean }) {
  const d = sim.probability_change;
  const unit = (k: string) => (k in FIELD_META ? FIELD_META[k as keyof typeof FIELD_META].unit : "");
  return (
    <div className={`stack${busy ? " busy" : ""}`} data-testid="simulation">
      <div className="panel">
        <span className="sim-label">Model simulation</span>
        <div className="compare" style={{ marginTop: 12 }}>
          <div className="compare-side">
            <div className="note">Before</div>
            <div className="big-prob" style={{ fontSize: 48 }} data-testid="sim-before">{(sim.before.failure_probability * 100).toFixed(1)}<small>%</small></div>
            <RiskBadge level={sim.before.risk_level} />
          </div>
          <div className="compare-arrow" aria-hidden="true">→</div>
          <div className="compare-side">
            <div className="note">After</div>
            <div className="big-prob" style={{ fontSize: 48 }} data-testid="sim-after">{(sim.after.failure_probability * 100).toFixed(1)}<small>%</small></div>
            <RiskBadge level={sim.after.risk_level} />
          </div>
        </div>
        <p style={{ margin: "12px 0 0" }} data-testid="sim-delta">
          <b className={d > 0 ? "delta-up" : d < 0 ? "delta-down" : undefined}>{signedPoints(d)}</b>{" "}
          <span className="note">
            {sim.before.risk_level === sim.after.risk_level ? `risk level unchanged (${sim.after.risk_level})` : `risk level ${sim.risk_change}`}
            {sim.prediction_changed ? "; the failure prediction flips" : ""}
          </span>
        </p>
        <RiskScale probability={sim.after.failure_probability} compareTo={sim.before.failure_probability}
          medium={sim.after.thresholds.medium} high={sim.after.thresholds.high} />
        <p className="note">Solid needle: after. Dashed line: before.</p>
      </div>
      {sim.after.warnings.length > 0 && (
        <div className="panel">{sim.after.warnings.map((w) => <p key={w} className="callout warn" style={{ margin: "4px 0" }}>{w}</p>)}</div>
      )}
      <div className="panel">
        <h2>What changed</h2>
        <table data-testid="sim-changes">
          <thead><tr><th>Parameter</th><th className="n">Before</th><th className="n">After</th></tr></thead>
          <tbody>
            {sim.changed_parameters.map((c) => (
              <tr key={c.parameter}><td>{c.label}</td><td className="n">{String(c.before)} {unit(c.parameter)}</td><td className="n">{String(c.after)} {unit(c.parameter)}</td></tr>
            ))}
          </tbody>
        </table>
        <h3 style={{ marginTop: 20 }}>Largest shifts in the model's reasoning</h3>
        <table>
          <thead><tr><th>Feature</th><th className="n">Contribution before</th><th className="n">After</th><th className="n">Shift</th></tr></thead>
          <tbody>
            {sim.contribution_shift.slice(0, 5).map((c) => (
              <tr key={c.feature}>
                <td>{c.label}</td>
                <td className="n">{signedPoints(c.before)}</td>
                <td className="n">{signedPoints(c.after)}</td>
                <td className="n"><b className={c.change > 0 ? "delta-up" : c.change < 0 ? "delta-down" : undefined}>{signedPoints(c.change)}</b></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">Contributions are SHAP values in probability points. Engineered features (power, strain, temperature gap) shift when their inputs change.</p>
      </div>
    </div>
  );
}
