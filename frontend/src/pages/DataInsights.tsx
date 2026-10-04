import { useState } from "react";
import { api } from "../api";
import { ClassHistogram, HBars, Legend, LineChart, Scatter } from "../components/charts";
import { Async, PageHead, Readouts, useAsync } from "../components/ui";
import { MODE_NAMES, int, num, pct } from "../lib/format";

const LABELS: Record<string, string> = {
  air_temp_k: "Air temperature [K]",
  process_temp_k: "Process temperature [K]",
  rotational_speed_rpm: "Rotational speed [rpm]",
  torque_nm: "Torque [Nm]",
  tool_wear_min: "Tool wear [min]",
  temp_diff_k: "Process-air gap [K]",
  power_w: "Power [W]",
  strain_nm_min: "Strain [Nm·min]",
};

export function DataInsights() {
  const dq = useAsync(() => api.datasetSummary(), []);
  const eda = useAsync(() => api.eda(), []);
  const [decileFeature, setDecileFeature] = useState("torque_nm");

  return (
    <>
      <PageHead title="Data insights">
        The AI4I 2020 dataset is synthetic: it was generated to mirror real milling-machine telemetry. Exploratory
        analysis uses the training split only, so nothing here was learned from test records.
      </PageHead>

      <Async state={dq} label="Loading dataset summary">
        {(d) => {
          const q = d.quality;
          return (
            <>
              <Readouts items={[
                { value: int(q.n_rows), label: "Records", sub: `${q.n_columns} columns, 6 used as model inputs` },
                { value: pct(q.failure_rate, 2), label: "Failure rate", sub: `${q.target_counts["1"]} failures, ${q.imbalance_ratio}:1 imbalance` },
                { value: q.missing_values_total, label: "Missing values", sub: `${q.duplicate_rows} duplicate rows` },
                { value: Object.values(q.failure_mode_counts).reduce((a, b) => a + b, 0), label: "Failure-mode labels", sub: "5 modes; some records carry several" },
              ]} />
              <div className="grid-2" style={{ marginTop: 16 }}>
                <section className="panel">
                  <h2>Data quality checks</h2>
                  <table data-testid="quality-table">
                    <tbody>
                      <tr><td>Schema and types</td><td>Validated (14 columns, binary labels, types L/M/H)</td></tr>
                      <tr><td>Missing values</td><td>{q.missing_values_total}</td></tr>
                      <tr><td>Duplicate rows / IDs</td><td>{q.duplicate_rows}</td></tr>
                      <tr><td>Failure with no mode flag</td><td>{q.label_consistency.failure_without_mode} records (kept; target is the official label)</td></tr>
                      <tr><td>Mode flag with no failure</td><td>{q.label_consistency.mode_without_failure} records, all RNF (kept as non-failures)</td></tr>
                      <tr><td>Failures with several modes</td><td>{q.label_consistency.failures_with_multiple_modes}</td></tr>
                      <tr><td>IQR outliers</td><td>{Object.entries(q.outliers_iqr).filter(([, v]) => v.n_outliers > 0).map(([k, v]) => `${LABELS[k]}: ${v.n_outliers}`).join("; ")}</td></tr>
                      <tr><td>Raw file checksum</td><td><code style={{ wordBreak: "break-all" }}>{d.sha256.slice(0, 16)}…</code></td></tr>
                    </tbody>
                  </table>
                  <p className="note">Outliers were investigated, not removed: see the torque finding below.</p>
                </section>
                <section className="panel">
                  <h2>Target and failure modes</h2>
                  <HBars testId="mode-counts" rows={Object.entries(q.failure_mode_counts).map(([k, v]) => ({ label: MODE_NAMES[k] ?? k, value: v }))}
                    format={(v) => String(v)} color="var(--failed)" />
                  <h3 style={{ marginTop: 16 }}>Splits</h3>
                  <table>
                    <thead><tr><th>Split</th><th className="n">Records</th><th className="n">Failures</th><th className="n">Rate</th></tr></thead>
                    <tbody>
                      {(["train", "validation", "test"] as const).map((s) => (
                        <tr key={s}><td>{s}</td><td className="n">{int(d.splits[s].rows)}</td><td className="n">{d.splits[s].failures}</td><td className="n">{pct(d.splits[s].failure_rate, 2)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="note">{d.split_strategy}; seed {d.random_seed}.</p>
                </section>
              </div>
            </>
          );
        }}
      </Async>

      <Async state={eda} label="Loading exploratory analysis">
        {(e) => {
          const torque = e.outlier_check.torque_nm as { n_outliers: number; failure_rate_outliers: number | null; failure_rate_rest: number };
          return (
            <>
              <section className="panel" style={{ marginTop: 16 }}>
                <h2>Which operating conditions differ between normal and failed machines?</h2>
                <Legend items={[{ label: "Normal", color: "var(--normal)" }, { label: "Failed", color: "var(--failed)" }]} />
                <div className="grid-3" style={{ marginTop: 8 }} data-testid="distributions">
                  {Object.entries(e.distributions).map(([k, v]) => (
                    <div key={k}>
                      <ClassHistogram bins={v.bins} normal={v.normal} failed={v.failed} label={LABELS[k] ?? k} />
                      <div className="note">Median {num(v.normal_median, 1)} normal vs {num(v.failed_median, 1)} failed · standardised difference {v.standardized_mean_difference > 0 ? "+" : ""}{num(v.standardized_mean_difference, 2)}</div>
                    </div>
                  ))}
                </div>
                <p className="note">Densities (each class sums to 1), training split. Strain and torque separate the classes most; the engineered features separate better than most raw inputs.</p>
              </section>

              <div className="grid-2" style={{ marginTop: 16 }}>
                <section className="panel">
                  <h2>Where do failures sit in speed-torque space?</h2>
                  <Scatter testId="scatter" points={e.scatter_sample.map((p) => ({ x: p.rpm, y: p.torque, failed: p.failed === 1 }))}
                    xLabel="Rotational speed [rpm]" yLabel="Torque [Nm]" />
                  <p className="note">Speed and torque are strongly negatively related (Spearman ρ = {String(e.outlier_check.rpm_torque_spearman)}). Failures cluster at both ends of the curve: low speed with high torque, and very high speed with low torque — the regions where mechanical power is extreme.</p>
                </section>
                <section className="panel">
                  <h2>How does failure rate change across a variable?</h2>
                  <div className="demo-list" role="group" aria-label="Variable">
                    {Object.keys(e.rate_by_decile).map((k) => (
                      <button key={k} className="chip" aria-pressed={decileFeature === k} onClick={() => setDecileFeature(k)}>{(LABELS[k] ?? k).split(" [")[0]}</button>
                    ))}
                  </div>
                  <div style={{ marginTop: 10 }}>
                    <LineChart testId="decile-chart" legend={false}
                      series={[{ name: "Failure rate", color: "var(--failed)", marker: true, points: (e.rate_by_decile[decileFeature] ?? []).map((r) => [(r.low + r.high) / 2, r.rate]) }]}
                      xLabel={LABELS[decileFeature] ?? decileFeature} yLabel="Failure rate in decile"
                      xDomain={[Math.min(...(e.rate_by_decile[decileFeature] ?? []).map((r) => (r.low + r.high) / 2)), Math.max(...(e.rate_by_decile[decileFeature] ?? []).map((r) => (r.low + r.high) / 2))]}
                      yDomain={[0, Math.max(0.05, ...(e.rate_by_decile[decileFeature] ?? []).map((r) => r.rate)) * 1.1]}
                      yFormat={(v) => `${Math.round(v * 100)}%`} xFormat={(v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v))}
                      refY={{ y: e.n_rows ? Object.values(e.failure_rate_by_type).reduce((a, t) => a + t.failures, 0) / e.n_rows : 0, label: "overall" }} />
                  </div>
                  <p className="note">Each point is one decile of the training split. Several relationships are U-shaped or threshold-like, which is why tree models beat logistic regression here.</p>
                </section>
              </div>

              <div className="grid-2" style={{ marginTop: 16 }}>
                <section className="panel">
                  <h2>Are the torque outliers errors?</h2>
                  <p className="lede" data-testid="outlier-finding">
                    No. The {torque.n_outliers} training records outside the IQR fences for torque failed at a rate of{" "}
                    <b>{pct(torque.failure_rate_outliers)}</b>, against {pct(torque.failure_rate_rest)} for the rest. They are a real high-stress
                    regime, so they were kept. Removing them would delete most power and overstrain failures.
                  </p>
                </section>
                <section className="panel">
                  <h2>Does product quality matter?</h2>
                  <table data-testid="type-table">
                    <thead><tr><th>Type</th><th className="n">Records</th><th className="n">Failure rate</th><th className="n">95% interval</th></tr></thead>
                    <tbody>
                      {Object.entries(e.failure_rate_by_type).map(([t, v]) => (
                        <tr key={t}><td>{t}</td><td className="n">{int(v.n)}</td><td className="n">{pct(v.rate, 2)}</td><td className="n">{pct(v.ci_low)}–{pct(v.ci_high)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="note">Lower-quality (L) products fail somewhat more often, but intervals overlap; type is a weak signal compared with the operating conditions.</p>
                </section>
              </div>
            </>
          );
        }}
      </Async>
    </>
  );
}
