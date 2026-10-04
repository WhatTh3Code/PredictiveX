import { api, type Prediction } from "../api";
import { DemoPicker, MachineForm } from "../components/MachineForm";
import { RiskScale } from "../components/RiskScale";
import { EmptyState, ErrorState, PageHead, RiskBadge, useAsync } from "../components/ui";
import { MODE_NAMES, pct, riskDescription } from "../lib/format";
import { navigate, useSession } from "../state";

export function Predict() {
  const s = useSession();
  const info = useAsync(() => api.modelInfo(), []);
  const demos = useAsync(() => api.demoCases(), []);
  const activeDemo = demos.data?.cases.find((c) => c.id === s.activeCase) ?? null;

  return (
    <>
      <PageHead title="Predict a machine">
        Enter one machine's operating parameters. The trained model returns a failure probability, a risk level and
        the inputs that drove the estimate.
      </PageHead>
      <div className="grid-predict">
        <section className="panel" aria-labelledby="inputs-h">
          <h2 id="inputs-h">Machine parameters</h2>
          {demos.data && (
            <div style={{ marginBottom: 16 }}>
              <p className="note" style={{ margin: "0 0 6px" }}>Demo machines from the unseen test split</p>
              <DemoPicker cases={demos.data.cases} active={s.activeCase} onPick={(c) => s.loadCase(c.id, c.input)} />
            </div>
          )}
          <MachineForm
            draft={s.draft}
            onChange={(d) => { s.setDraft(d); s.setActiveCase(null); }}
            onSubmit={() => s.runPrediction()}
            errors={s.fieldErrors}
            ranges={info.data?.training_ranges ?? null}
            busy={s.predicting}
          />
        </section>

        <section aria-live="polite" aria-busy={s.predicting} data-testid="result-area">
          {s.fieldErrors.length > 0 ? (
            <div className="panel">
              <div className="state" role="alert" data-testid="validation-errors">
                <h3>Input rejected before reaching the model</h3>
                <p>Fix {s.fieldErrors.length === 1 ? "this field" : `these ${s.fieldErrors.length} fields`} and predict again:</p>
                <ul>{s.fieldErrors.map((e) => <li key={e.field}>{e.message}</li>)}</ul>
                <p className="note">The API applies the same limits and returns HTTP 422 for invalid records.</p>
              </div>
            </div>
          ) : s.requestError ? (
            <div className="panel"><ErrorState error={s.requestError} onRetry={() => s.runPrediction()} /></div>
          ) : s.prediction ? (
            <div className={s.predicting ? "busy" : undefined}>
              <PredictionResult p={s.prediction} actual={activeDemo && activeDemo.actual_failure !== null
                ? { failed: activeDemo.actual_failure, modes: activeDemo.actual_failure_modes, source: activeDemo.source } : null} />
            </div>
          ) : s.predicting ? (
            <div className="panel"><div className="state" role="status" data-testid="loading">Running the model…</div></div>
          ) : (
            <div className="panel">
              <EmptyState title="No prediction yet">
                <p>Pick a demo machine or enter parameters, then choose <b>Predict</b>.</p>
              </EmptyState>
            </div>
          )}
        </section>
      </div>
    </>
  );
}

export function PredictionResult({ p, actual }: { p: Prediction; actual: { failed: boolean; modes: string[]; source: string } | null }) {
  const ev = p.risk_band_evidence;
  const top = p.contributions.filter((c) => c.direction === "increases").slice(0, 3);
  return (
    <div className="stack" data-testid="prediction-result">
      <div className="panel instrument">
        <div className="instrument-top">
          <div>
            <div className="big-prob" data-testid="probability">
              {(p.failure_probability * 100).toFixed(1)}<small>%</small>
            </div>
            <div className="prob-caption">Model-estimated failure probability</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <RiskBadge level={p.risk_level} size={18} />
            <div className="prob-caption" data-testid="prediction-verdict">
              {p.predicted_failure ? "Prediction: failure likely" : "Prediction: no failure flagged"}
            </div>
          </div>
        </div>
        <RiskScale probability={p.failure_probability} medium={p.thresholds.medium} high={p.thresholds.high} />
        <p className="note" style={{ marginBottom: 0 }}>{riskDescription(p.risk_level)}</p>
        {ev && ev.observed_failure_rate !== null && (
          <p className="note" style={{ margin: "6px 0 0" }} data-testid="band-evidence">
            On the held-out test set, {ev.test_failures_in_band} of {ev.test_records_in_band} machines the model placed
            in {p.risk_level} actually failed ({pct(ev.observed_failure_rate)}; 95% interval {pct(ev.wilson_95ci?.[0])}–{pct(ev.wilson_95ci?.[1])}).
            That observed rate, not the probability above, is the evidence for how much to trust this band.
          </p>
        )}
        {actual && (
          <p className="note" style={{ margin: "6px 0 0" }} data-testid="actual-outcome">
            Ground truth for this demo record ({actual.source}):{" "}
            <b>{actual.failed ? `failed — ${actual.modes.map((m) => MODE_NAMES[m] ?? m).join(", ") || "no mode recorded"}` : "did not fail"}</b>.
          </p>
        )}
      </div>

      {p.warnings.length > 0 && (
        <div className="panel" data-testid="warnings">
          {p.warnings.map((w) => <p className="callout warn" key={w} style={{ margin: "4px 0" }}>{w}</p>)}
        </div>
      )}

      <div className="panel">
        <h2>Why this estimate</h2>
        <p className="explanation" data-testid="explanation">{p.explanation}</p>
        {top.length > 0 && (
          <ol style={{ margin: "8px 0 0", paddingLeft: 20 }}>
            {top.map((c) => <li key={c.feature}>{c.label} <span className="note">({c.display_value})</span></li>)}
          </ol>
        )}
        <div className="row" style={{ marginTop: 16 }}>
          <button className="btn" onClick={() => navigate("/explain")} data-testid="go-explain">See full explanation</button>
          <button className="btn btn-quiet" onClick={() => navigate("/what-if")} data-testid="go-whatif">Try a what-if change</button>
        </div>
      </div>

      {p.failure_mode && (
        <div className="panel" data-testid="failure-mode">
          <h2>If it fails, which mechanism?</h2>
          <p className="note" style={{ marginTop: 0 }}>
            Conditional estimate from a separate failure-mode model: it assumes a failure occurs and ranks the likely
            mechanism. It is not an additional failure probability.
          </p>
          <table>
            <thead><tr><th>Failure mode</th><th className="n">Conditional share</th><th className="n">Test recall of this class</th></tr></thead>
            <tbody>
              {p.failure_mode.probabilities.map((m) => (
                <tr key={m.mode} className={m.mode === p.failure_mode?.most_likely ? "selected" : undefined}>
                  <td>{m.name}</td>
                  <td className="n">{pct(m.probability, 0)}</td>
                  <td className="n">{pct(m.test_recall, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">Random failures (RNF) are excluded: in this dataset they occur independently of the inputs.</p>
        </div>
      )}
    </div>
  );
}
