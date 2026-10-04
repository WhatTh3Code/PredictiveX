import { api } from "../api";
import { PageHead, useAsync } from "../components/ui";
import { num, pct } from "../lib/format";

export function About() {
  const m = useAsync(() => api.metrics(), []);
  const d = m.data;
  return (
    <>
      <PageHead title="Methodology">
        What PredictiveX does, how it was built and validated, and where its limits are.
      </PageHead>
      <article className="panel prose" data-testid="methodology">
        <h2 style={{ marginTop: 0 }}>Problem</h2>
        <p>
          ALGOTHON'26 problem ALG-DATA-02, "Predict what happens next", asks for a complete predictive pipeline:
          exploration and cleaning, feature engineering, model selection, training and validation, metrics, and
          predictions on unseen data. PredictiveX applies this to predictive maintenance: given a machine's current
          operating readings, will it fail, and why does the model think so?
        </p>

        <h2>Dataset</h2>
        <p>
          AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository, id 601; Matzka, 2020). 10,000 records with
          product type, air and process temperature, rotational speed, torque and tool wear, plus a machine-failure
          label and five failure-mode flags. <b>The dataset is synthetic.</b> It was generated to reflect real
          industrial predictive-maintenance data, and its failure modes follow documented rules. Results here show
          what the method achieves on that data; they are not a measurement of any real factory.
        </p>

        <h2>Pipeline</h2>
        <ol>
          <li><b>Data checks.</b> Checksum, schema, types, missing values, duplicates, IQR outliers and label consistency.</li>
          <li><b>Leakage control.</b> Identifiers (UDI, product ID) and the five failure-mode flags are excluded from model inputs: each flag implies the target.</li>
          <li><b>Split.</b> Stratified by failure mode, 60/20/20 train/validation/test, fixed seed 42. Exploratory analysis uses training data only.</li>
          <li><b>Features.</b> Temperature gap, mechanical power and tool strain, each kept only after a cross-validated ablation showed it helps.</li>
          <li><b>Models.</b> Logistic regression, random forest, histogram gradient boosting and XGBoost, each tuned with 5-fold CV on the training split (PR-AUC).</li>
          <li><b>Selection.</b> Highest validation PR-AUC: {d ? d.model.model_display_name : "…"}.</li>
          <li><b>Thresholds.</b> Derived from out-of-fold training predictions: HIGH maximises F2 (missed failures weighted above false alarms); MEDIUM keeps 95% recall.</li>
          <li><b>Final test.</b> One evaluation on the untouched test split{d ? `: PR-AUC ${num(d.test.pr_auc)}, ROC-AUC ${num(d.test.roc_auc)}, precision ${pct(d.test.precision)}, recall ${pct(d.test.recall)}` : ""}, with bootstrap intervals and a five-split robustness check.</li>
          <li><b>Explainability.</b> Exact TreeSHAP per prediction, permutation importance on validation.</li>
        </ol>

        <h2>Class imbalance</h2>
        <p>
          Only 3.4% of records fail. Accuracy is therefore not used for any decision. Class weighting was included in
          every model's tuning grid and selected only where cross-validation favoured it; no synthetic oversampling
          was used, because threshold tuning addresses the decision trade-off directly without inventing records.
        </p>

        <h2>Limitations</h2>
        <ul>
          <li>Synthetic data with rule-based failure mechanisms; real equipment will be noisier, and the model must be re-validated on real telemetry before use.</li>
          <li>Each record is an independent snapshot. The model has no time history, so it predicts the current failure state rather than a forecast horizon.</li>
          <li>Random failures (RNF) are unpredictable by construction, and a few failures have no recorded mode.</li>
          <li>SHAP values and the what-if simulator describe the model, not physics. Changing an input in the simulator does not guarantee the same change in a real machine.</li>
          <li>Probabilities come from a random forest without post-hoc calibration; the observed failure rate per risk band on the test split is the better guide to reliability.</li>
          <li>Inputs outside the training range are accepted with a warning, because tree models cannot extrapolate.</li>
        </ul>

        <h2>Future scope</h2>
        <ul>
          <li>Time-series sensor streams and remaining-useful-life prediction instead of point-in-time snapshots.</li>
          <li>Validation on real plant data, with drift monitoring and periodic retraining.</li>
          <li>Cost-sensitive thresholds set from actual maintenance and downtime costs.</li>
          <li>Probability calibration and conformal prediction intervals.</li>
        </ul>

        <h2>Transparency</h2>
        <p>
          Every number in this dashboard is read from reports generated by the training pipeline, and an automated
          test recomputes the headline test metrics from the saved model. Predictions are computed live; none are
          hard-coded. This project was built with AI assistance (Claude, by Anthropic); the code, experiments and
          results were generated and verified in the repository and can be reproduced with the commands in the README.
        </p>
      </article>
    </>
  );
}
