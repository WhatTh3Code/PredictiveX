import { api } from "../api";
import { HBars, ShapSummary } from "../components/charts";
import { Contributions } from "../components/Contributions";
import { Async, EmptyState, PageHead, RiskBadge, useAsync } from "../components/ui";
import { num, pct } from "../lib/format";
import { navigate, useSession } from "../state";

export function Explain() {
  const s = useSession();
  const imp = useAsync(() => api.featureImportance(), []);
  const abl = useAsync(() => api.modelComparison(), []);
  const p = s.prediction;

  return (
    <>
      <PageHead title="Explain the prediction">
        SHAP attributes each prediction to the model's inputs. Values are exact TreeSHAP for the deployed random
        forest; they describe how the model reached its estimate, not what physically caused a failure.
      </PageHead>

      <section className="panel" aria-labelledby="local-h">
        <div className="chart-tools">
          <h2 id="local-h">This machine</h2>
          {p && <span><RiskBadge level={p.risk_level} /> <span className="note">{pct(p.failure_probability)} estimated</span></span>}
        </div>
        {p ? (
          <>
            <p className="explanation">{p.explanation}</p>
            <Contributions items={p.contributions} outputSpace={p.shap_output_space} />
            <p className="note">
              Baseline (average prediction on training data): {p.shap_output_space === "probability" ? pct(p.shap_base_value) : num(p.shap_base_value)}.
              Baseline plus all bars equals this machine's estimate.
            </p>
          </>
        ) : (
          <EmptyState title="Nothing to explain yet">
            <p>Run a prediction first; its explanation appears here.</p>
            <button className="btn" onClick={() => navigate("/predict")}>Go to Predict</button>
          </EmptyState>
        )}
      </section>

      <div className="grid-2" style={{ marginTop: 16 }}>
        <section className="panel">
          <h2>What the model relies on overall</h2>
          <Async state={imp} label="Loading importance">
            {(d) => (
              <>
                <h3>Mean |SHAP| on the test set</h3>
                <HBars testId="global-shap" rows={d.shap_global_importance_test.map((r) => ({ label: r.label, value: r.mean_abs_shap }))}
                  format={(v) => (d.output_space === "probability" ? `${(v * 100).toFixed(2)} pts` : v.toFixed(3))} />
                <p className="note">{d.global_sample}. Local accuracy check: max |baseline + Σ SHAP − prediction| = {d.shap_additivity_max_abs_error.toExponential(1)}.</p>
                <h3 style={{ marginTop: 20 }}>Permutation importance on validation</h3>
                <HBars testId="perm-importance" rows={d.permutation_importance_validation.map((r) => ({ label: r.label, value: Math.max(r.mean, 0), err: r.std }))}
                  format={(v) => v.toFixed(3)} color="var(--steel)" />
                <p className="note">Drop in validation PR-AUC when one feature is shuffled (20 repeats). Correlated features share credit, so a low value does not mean a feature is irrelevant.</p>
              </>
            )}
          </Async>
        </section>
        <section className="panel">
          <h2>Direction of each effect</h2>
          <Async state={imp} label="Loading SHAP sample">
            {(d) => {
              const labels = Object.fromEntries(d.shap_global_importance_test.map((r) => [r.feature, r.label]));
              return (
                <>
                  <ShapSummary features={d.shap_beeswarm_test_sample.features} labels={labels}
                    sample={d.shap_beeswarm_test_sample.sample} order={d.shap_global_importance_test.map((r) => r.feature)}
                    outputSpace={d.output_space} />
                  <p className="note">Each dot is one test machine. Dark dots to the right mean high values of that feature push the model toward failure.</p>
                </>
              );
            }}
          </Async>
        </section>
      </div>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Engineered features</h2>
        <Async state={abl} label="Loading feature definitions">
          {(d) => {
            const m = d.feature_ablation.selected_model_marginal;
            return (
              <div className="table-wrap">
                <table data-testid="feature-table">
                  <thead><tr><th>Feature</th><th>Formula</th><th>Why it was created</th><th className="n">CV PR-AUC without it</th></tr></thead>
                  <tbody>
                    {Object.entries(d.feature_ablation.definitions).map(([f, def]) => (
                      <tr key={f}>
                        <td><code>{f}</code></td>
                        <td><code>{def.formula}</code></td>
                        <td>{def.reason}</td>
                        <td className="n">{num(m[`without_${f}`]?.cv_pr_auc_mean)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="note">
                  With all three: CV PR-AUC {num(m.selected_set?.cv_pr_auc_mean)}. Raw inputs only: {num(m.raw_only?.cv_pr_auc_mean)}.
                  Full ablation on the Model performance page.
                </p>
              </div>
            );
          }}
        </Async>
      </section>
    </>
  );
}
