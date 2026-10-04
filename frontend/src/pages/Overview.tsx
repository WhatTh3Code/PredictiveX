import { api, type RiskLevel } from "../api";
import { Async, PageHead, Readouts, RiskBadge, useAsync } from "../components/ui";
import { int, num, pct } from "../lib/format";
import { navigate, useSession } from "../state";

export function Overview() {
  const s = useSession();
  const metrics = useAsync(() => api.metrics(), []);
  const dq = useAsync(() => api.datasetSummary(), []);
  const demos = useAsync(() => api.demoCases(), []);

  const openCase = async (id: string) => {
    const c = demos.data?.cases.find((x) => x.id === id);
    if (!c) return;
    navigate("/predict");
    await s.loadCase(c.id, c.input);
  };

  return (
    <>
      <PageHead title="Spot machines heading for failure">
        PredictiveX scores a machine's operating state for failure risk, shows which readings drive the score, and
        lets you test how changes would move it. Trained and validated on the AI4I 2020 predictive-maintenance dataset.
      </PageHead>

      <Async state={metrics} label="Loading results">
        {(m) => (
          <Readouts items={[
            { value: dq.data ? int(dq.data.quality.n_rows) : "…", label: "Machine records analysed", sub: dq.data ? `${pct(dq.data.quality.failure_rate, 2)} failed` : undefined },
            { value: int(m.test.n), label: "Unseen test machines scored", sub: `${m.test.positives} actual failures` },
            { value: `${m.test.confusion_matrix.tp} / ${m.test.positives}`, label: "Test failures flagged HIGH", sub: `${m.test.confusion_matrix.fp} false alarms` },
            { value: num(m.test.pr_auc), label: "Test PR-AUC", sub: `ROC-AUC ${num(m.test.roc_auc)}` },
          ]} />
        )}
      </Async>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Try it on a real machine</h2>
        <p className="lede" style={{ marginTop: 0 }}>These records come from the test split, which the model never saw during training. Each opens the prediction straight away.</p>
        <Async state={demos} label="Loading demo machines">
          {(d) => (
            <div className="row" data-testid="overview-demos">
              {d.cases.filter((c) => ["low", "high", "borderline"].includes(c.id)).map((c) => (
                <button key={c.id} className={c.id === "high" ? "btn" : "btn btn-quiet"} onClick={() => openCase(c.id)} data-case={c.id}>
                  {`Predict the ${c.title.toLowerCase()}`}
                </button>
              ))}
              <a href="#/predict" style={{ marginLeft: 4 }}>or enter your own parameters</a>
            </div>
          )}
        </Async>
      </section>

      <Async state={metrics} label="Loading risk bands">
        {(m) => (
          <div className="grid-2" style={{ marginTop: 16 }}>
            <section className="panel">
              <h2>Fleet health on the test split</h2>
              <p className="note" style={{ marginTop: 0 }}>How the model distributes the {int(m.test.n)} unseen machines, and how many in each band really failed.</p>
              <table data-testid="fleet-table">
                <thead><tr><th>Band</th><th className="n">Machines</th><th className="n">Actually failed</th><th className="n">Observed rate</th></tr></thead>
                <tbody>
                  {(["HIGH", "MEDIUM", "LOW"] as RiskLevel[]).map((b) => {
                    const r = m.test.risk_bands[b];
                    return (
                      <tr key={b}><td><RiskBadge level={b} /></td><td className="n">{int(r.records)}</td><td className="n">{r.actual_failures}</td><td className="n">{pct(r.observed_failure_rate)}</td></tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="note">Band thresholds ({pct(m.thresholds.policy.medium_threshold)} and {pct(m.thresholds.policy.high_threshold)}) were derived from training data only.</p>
            </section>
            <section className="panel">
              <h2>How a prediction is made</h2>
              <ol style={{ margin: 0, paddingLeft: 20, color: "var(--ink-2)" }}>
                <li>Six readings are validated against physical limits.</li>
                <li>Three engineered features are computed: temperature gap, mechanical power, tool strain.</li>
                <li>A {m.model.model_display_name.toLowerCase()} (selected from four tuned families) estimates failure probability.</li>
                <li>The probability is placed in a LOW, MEDIUM or HIGH band.</li>
                <li>Exact SHAP values explain which readings moved the estimate.</li>
              </ol>
              <p className="note">Nothing is retrained when you predict; the saved model answers in well under a second.</p>
              <div className="row"><a href="#/performance">See the validation evidence</a><a href="#/about">Read the methodology</a></div>
            </section>
          </div>
        )}
      </Async>
    </>
  );
}
