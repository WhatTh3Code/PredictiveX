import { api, type RiskLevel } from "../api";
import { ConfusionMatrix, GroupedBars, LineChart } from "../components/charts";
import { Async, PageHead, Readouts, RiskBadge, useAsync } from "../components/ui";
import { MODE_NAMES, int, num, pct } from "../lib/format";

const FEATURE_SET_LABELS: Record<string, string> = {
  raw_only: "Raw inputs only",
  "raw+temp_diff_k": "+ temperature gap",
  "raw+power_w": "+ power",
  "raw+strain_nm_min": "+ strain",
  "raw+all_engineered": "+ all three",
};

export function Performance() {
  const metrics = useAsync(() => api.metrics(), []);
  const comp = useAsync(() => api.modelComparison(), []);
  const fm = useAsync(() => api.failureMode(), []);

  return (
    <>
      <PageHead title="Model performance">
        Four model families were tuned with cross-validation on the training split, compared on the validation split,
        and the winner was evaluated once on a held-out test split it had never seen.
      </PageHead>

      <Async state={metrics} label="Loading metrics">
        {(m) => (
          <>
            <Readouts items={[
              { value: num(m.test.pr_auc), label: "Test PR-AUC", sub: `95% CI ${num(m.test.bootstrap_95ci.pr_auc[0])}–${num(m.test.bootstrap_95ci.pr_auc[1])} · no-skill ${num(m.test.prevalence)}` },
              { value: num(m.test.roc_auc), label: "Test ROC-AUC", sub: `95% CI ${num(m.test.bootstrap_95ci.roc_auc[0])}–${num(m.test.bootstrap_95ci.roc_auc[1])}` },
              { value: pct(m.test.recall), label: "Failures caught (recall)", sub: `${m.test.confusion_matrix.tp} of ${m.test.positives} test failures` },
              { value: pct(m.test.precision), label: "Alarms that were real (precision)", sub: `${m.test.confusion_matrix.fp} false alarms in ${int(m.test.n)} machines` },
            ]} />

            <section className="panel" style={{ marginTop: 16 }}>
              <h2>Validation methodology</h2>
              <div className="flow" data-testid="methodology-flow">
                <b>10,000 records</b> → stratified split (seed {m.model.random_seed}) →
                <b>train 6,000</b> (5-fold CV: feature ablation + tuning, thresholds from out-of-fold predictions) →
                <b>validation 2,000</b> (model selection) →
                <b>test 2,000</b> (evaluated once)
              </div>
              <p className="note">Selected model: <b>{m.model.model_display_name}</b> with {m.model.features.length} features. Hyper-parameters: {Object.entries(m.model.hyperparameters).map(([k, v]) => `${k}=${String(v)}`).join(", ")}.</p>
            </section>
          </>
        )}
      </Async>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Model comparison</h2>
        <Async state={comp} label="Loading comparison">
          {(c) => {
            const rows = c.comparison.models;
            const sel = rows.findIndex((r) => r.model === c.comparison.selected);
            return (
              <>
                <div className="table-wrap">
                  <table data-testid="model-table">
                    <thead>
                      <tr><th>Model</th><th className="n">CV PR-AUC (train)</th><th className="n">Val PR-AUC</th><th className="n">Val ROC-AUC</th>
                        <th className="n">Val precision</th><th className="n">Val recall</th><th className="n">Val F1</th><th className="n">Threshold</th></tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={r.model} className={r.model === c.comparison.selected ? "selected" : undefined}>
                          <td>{r.display_name}{r.model === c.comparison.selected && <span className="note"> — selected</span>}</td>
                          <td className="n">{num(r.tuning.cv_pr_auc_mean)} ± {num(r.tuning.cv_pr_auc_std)}</td>
                          <td className="n">{num(r.validation.pr_auc)}</td>
                          <td className="n">{num(r.validation.roc_auc)}</td>
                          <td className="n">{num(r.validation.precision)}</td>
                          <td className="n">{num(r.validation.recall)}</td>
                          <td className="n">{num(r.validation.f1)}</td>
                          <td className="n">{num(r.oof_threshold)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="note">Selection rule: highest {c.comparison.selection_metric}. PR-AUC is used because only 3.4% of records fail; accuracy would be 96.6% for a model that never predicts failure.</p>
                <GroupedBars testId="comparison-chart" groups={["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]}
                  series={rows.map((r) => r.display_name)} highlight={sel}
                  values={rows.map((r) => [r.validation.pr_auc, r.validation.roc_auc, r.validation.precision, r.validation.recall, r.validation.f1])} />
              </>
            );
          }}
        </Async>
      </section>

      <Async state={metrics} label="Loading test results">
        {(m) => (
          <>
            <div className="grid-2" style={{ marginTop: 16 }}>
              <section className="panel">
                <h2>Confusion matrix (test)</h2>
                <ConfusionMatrix cm={m.test.confusion_matrix}
                  caption={`Decision threshold ${num(m.test.threshold)} (F2-optimal on out-of-fold training predictions). Accuracy ${pct(m.test.accuracy)} is shown for completeness only.`} />
                <h3 style={{ marginTop: 16 }}>Missed failures by mode</h3>
                <table>
                  <thead><tr><th>Mode</th><th className="n">Test failures</th><th className="n">Missed</th></tr></thead>
                  <tbody>
                    {Object.entries(m.test.failures_by_mode).map(([k, v]) => (
                      <tr key={k}><td>{MODE_NAMES[k]}</td><td className="n">{v}</td><td className="n">{m.test.missed_failures_by_mode[k]}</td></tr>
                    ))}
                  </tbody>
                </table>
                <p className="note">Modes overlap (one record can carry several). Random failures are unpredictable by design.</p>
              </section>
              <section className="panel">
                <h2>Precision-recall curve (test)</h2>
                <LineChart testId="pr-curve" series={[{ name: "Random forest", color: "var(--normal)", points: m.test.curves.pr }]}
                  xLabel="Recall" yLabel="Precision" xDomain={[0, 1]} yDomain={[0, 1]}
                  refY={{ y: m.test.prevalence, label: `no-skill ${num(m.test.prevalence)}` }}
                  markers={[{ x: m.test.recall, y: m.test.precision, label: "operating point" }]} />
                <h3 style={{ marginTop: 12 }}>Calibration (test)</h3>
                <LineChart testId="calibration" series={[{ name: "Observed", color: "var(--normal)", points: m.test.calibration.map((r) => [r.mean_predicted, r.observed_rate]), marker: true }]}
                  xLabel="Mean predicted probability" yLabel="Observed failure rate" xDomain={[0, 1]} yDomain={[0, 1]} diagonal height={220}
                  xFormat={(v) => v.toFixed(2)} yFormat={(v) => v.toFixed(2)} />
                <p className="note">Brier score {num(m.test.brier, 4)}. Fixed probability bins ({m.test.calibration.map((r) => r.n).join(" / ")} machines); most machines sit in the lowest bin, so the upper bins rest on few records.</p>
              </section>
            </div>

            <section className="panel" style={{ marginTop: 16 }}>
              <h2>How the risk levels are set, and how they performed</h2>
              <p className="lede">{m.thresholds.policy.method}. MEDIUM starts at {pct(m.thresholds.policy.medium_threshold)}, HIGH at {pct(m.thresholds.policy.high_threshold)}. The table shows what actually happened to held-out test machines in each band.</p>
              <table data-testid="risk-bands">
                <thead><tr><th>Band</th><th className="n">Test machines</th><th className="n">Actually failed</th><th className="n">Observed failure rate</th><th className="n">95% interval</th><th className="n">Share of all failures</th></tr></thead>
                <tbody>
                  {(["HIGH", "MEDIUM", "LOW"] as RiskLevel[]).map((b) => {
                    const r = m.test.risk_bands[b];
                    return (
                      <tr key={b}>
                        <td><RiskBadge level={b} /></td>
                        <td className="n">{int(r.records)}</td>
                        <td className="n">{r.actual_failures}</td>
                        <td className="n">{pct(r.observed_failure_rate)}</td>
                        <td className="n">{r.wilson_95ci ? `${pct(r.wilson_95ci[0])}–${pct(r.wilson_95ci[1])}` : "—"}</td>
                        <td className="n">{pct(r.share_of_all_failures)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <h3 style={{ marginTop: 20 }}>Precision / recall trade-off on validation</h3>
              <div className="table-wrap">
                <table>
                  <thead><tr><th className="n">Threshold</th><th className="n">Precision</th><th className="n">Recall</th><th className="n">F1</th><th className="n">False alarms</th><th className="n">Missed failures</th></tr></thead>
                  <tbody>
                    {m.thresholds.validation_tradeoff.map((r) => (
                      <tr key={r.threshold}>
                        <td className="n">{r.threshold.toFixed(2)}</td><td className="n">{num(r.precision)}</td><td className="n">{num(r.recall)}</td>
                        <td className="n">{num(r.f1)}</td><td className="n">{r.confusion_matrix.fp}</td><td className="n">{r.confusion_matrix.fn}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note">Why F2: a missed failure is assumed to cost more than an unnecessary inspection, so recall is weighted above precision. The F1-optimal threshold would be {num(m.thresholds.f1_optimal_threshold_oof)}; both can be overridden via the API without retraining.</p>
            </section>

            <section className="panel" style={{ marginTop: 16 }}>
              <h2>Robustness across random splits</h2>
              <p className="lede">The selected recipe (same features and hyper-parameters) was re-trained and re-tested on five different random splits.</p>
              <table data-testid="robustness">
                <thead><tr><th>Seed</th><th className="n">PR-AUC</th><th className="n">ROC-AUC</th><th className="n">Precision</th><th className="n">Recall</th><th className="n">F1</th></tr></thead>
                <tbody>
                  {m.robustness.per_seed.map((r) => (
                    <tr key={r.seed}><td>{r.seed}</td><td className="n">{num(r.pr_auc)}</td><td className="n">{num(r.roc_auc)}</td><td className="n">{num(r.precision)}</td><td className="n">{num(r.recall)}</td><td className="n">{num(r.f1)}</td></tr>
                  ))}
                  <tr><th>Mean ± sd</th>
                    {(["pr_auc", "roc_auc", "precision", "recall", "f1"] as const).map((k) => (
                      <th className="n" key={k}>{num(m.robustness.mean[k])} ± {num(m.robustness.std[k])}</th>
                    ))}
                  </tr>
                </tbody>
              </table>
            </section>
          </>
        )}
      </Async>

      <div className="grid-2" style={{ marginTop: 16 }}>
        <section className="panel">
          <h2>Feature engineering ablation</h2>
          <Async state={comp} label="Loading ablation">
            {(c) => {
              const fams = Object.keys(c.feature_ablation.results);
              const sets = Object.keys(c.feature_ablation.results[fams[0] ?? ""] ?? {});
              const names = Object.fromEntries(c.comparison.models.map((r) => [r.model, r.display_name]));
              return (
                <div className="table-wrap">
                  <table data-testid="ablation-table">
                    <thead><tr><th>Feature set</th>{fams.map((f) => <th className="n" key={f}>{names[f] ?? f}</th>)}</tr></thead>
                    <tbody>
                      {sets.map((s) => (
                        <tr key={s}><td>{FEATURE_SET_LABELS[s] ?? s}</td>
                          {fams.map((f) => <td className="n" key={f}>{num(c.feature_ablation.results[f]?.[s]?.cv_pr_auc_mean)}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="note">{c.feature_ablation.protocol}; metric is mean PR-AUC. Every tree model improves with each engineered feature, and most with all three.</p>
                </div>
              );
            }}
          </Async>
        </section>
        <section className="panel">
          <h2>Failure-mode model</h2>
          <Async state={fm} label="Loading failure-mode results">
            {(f) => (
              <>
                <p className="lede" style={{ marginTop: 0 }}>{f.scope}</p>
                <table data-testid="mode-table">
                  <thead><tr><th>Mode</th><th className="n">Train records</th><th className="n">Test precision</th><th className="n">Test recall</th><th className="n">Test support</th></tr></thead>
                  <tbody>
                    {Object.entries(f.test.per_class).map(([k, v]) => (
                      <tr key={k}><td>{MODE_NAMES[k]}</td><td className="n">{f.training_class_counts[k]}</td><td className="n">{num(v.precision, 2)}</td><td className="n">{num(v.recall, 2)}</td><td className="n">{v.support}</td></tr>
                    ))}
                  </tbody>
                </table>
                <p className="note">Selected by 5-fold CV macro-F1 ({Object.entries(f.cv_macro_f1).map(([k, v]) => `${k.replace("_", " ")} ${num(v)}`).join(", ")}). Test macro-F1 {num(f.test.macro_f1)} on {f.test.single_mode_records} single-mode failures; small supports mean wide uncertainty.</p>
              </>
            )}
          </Async>
        </section>
      </div>
    </>
  );
}
