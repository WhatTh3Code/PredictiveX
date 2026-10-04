// Typed client for the PredictiveX FastAPI backend. Every number on screen comes from here.

export type ProductType = "L" | "M" | "H";
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface MachineInput {
  type: ProductType;
  air_temp_k: number;
  process_temp_k: number;
  rotational_speed_rpm: number;
  torque_nm: number;
  tool_wear_min: number;
}

export interface Contribution {
  feature: string;
  label: string;
  value: number;
  display_value: string;
  shap: number;
  direction: "increases" | "decreases" | "neutral";
  formula: string | null;
}

export interface FailureModeResult {
  note: string;
  most_likely: string;
  most_likely_name: string;
  probabilities: { mode: string; name: string; probability: number; test_recall: number | null }[];
}

export interface BandEvidence {
  description: string;
  test_records_in_band: number;
  test_failures_in_band: number;
  observed_failure_rate: number | null;
  wilson_95ci: [number, number] | null;
}

export interface Prediction {
  input?: MachineInput;
  failure_probability: number;
  risk_level: RiskLevel;
  predicted_failure: boolean;
  thresholds: { medium: number; high: number };
  risk_band_evidence: BandEvidence | null;
  failure_mode: FailureModeResult | null;
  warnings: string[];
  engineered_features: Record<string, number>;
  contributions: Contribution[];
  explanation: string;
  shap_base_value: number;
  shap_output_space: string;
}

export interface Simulation {
  label: string;
  before: Prediction;
  after: Prediction;
  after_input: MachineInput;
  changed_parameters: { parameter: keyof MachineInput; label: string; before: number | string; after: number | string }[];
  probability_change: number;
  risk_change: string;
  prediction_changed: boolean;
  contribution_shift: { feature: string; label: string; before: number; after: number; change: number }[];
}

export interface ThresholdMetrics {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  f2: number;
  accuracy: number;
  specificity: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
}

export interface FullMetrics extends ThresholdMetrics {
  pr_auc: number;
  roc_auc: number;
  brier: number;
  positives: number;
  n: number;
  prevalence: number;
}

export interface BandOutcome {
  records: number;
  actual_failures: number;
  observed_failure_rate: number | null;
  wilson_95ci: [number, number] | null;
  share_of_all_failures: number | null;
}

export interface ModelInfo {
  trained_at: string;
  model_family: string;
  model_display_name: string;
  hyperparameters: Record<string, unknown>;
  features: string[];
  engineered_features: string[];
  training_rows: number;
  training_failures: number;
  random_seed: number;
  versions: Record<string, string | null>;
  risk_policy: { medium_threshold: number; high_threshold: number; method: string };
  training_ranges: Record<string, { min: number; max: number }>;
  explainer: { method: string; output_space: string };
  failure_mode_model: { classes: string[]; test_macro_f1: number; test_recall_per_class: Record<string, number> } | null;
}

export interface MetricsReport {
  model: Omit<ModelInfo, "risk_policy" | "training_ranges" | "explainer" | "failure_mode_model">;
  thresholds: {
    policy: { medium_threshold: number; high_threshold: number; method: string };
    f1_optimal_threshold_oof: number;
    oof_train: FullMetrics;
    validation_at_high: ThresholdMetrics;
    validation_at_medium: ThresholdMetrics;
    validation_at_0_5: ThresholdMetrics;
    validation_risk_bands: Record<RiskLevel, BandOutcome>;
    validation_tradeoff: ThresholdMetrics[];
  };
  test: FullMetrics & {
    note: string;
    at_medium_threshold: ThresholdMetrics;
    at_0_5: ThresholdMetrics;
    bootstrap_95ci: Record<"pr_auc" | "roc_auc" | "precision" | "recall" | "f1", [number, number]>;
    risk_bands: Record<RiskLevel, BandOutcome>;
    calibration: { mean_predicted: number; observed_rate: number; n: number }[];
    curves: { pr: [number, number][]; roc: [number, number][] };
    missed_failures_by_mode: Record<string, number>;
    failures_by_mode: Record<string, number>;
  };
  robustness: {
    seeds: number[];
    per_seed: Record<string, number>[];
    mean: Record<string, number>;
    std: Record<string, number>;
  };
}

export interface ComparisonRow {
  model: string;
  display_name: string;
  features: string[];
  engineered: string[];
  tuning: { search: string; candidates_evaluated: number; best_params: Record<string, unknown>; cv_pr_auc_mean: number; cv_pr_auc_std: number };
  oof_threshold: number;
  validation: FullMetrics;
  validation_at_0_5: ThresholdMetrics;
  test: FullMetrics;
  train_seconds: number;
}

export interface AblationEntry {
  cv_pr_auc_mean: number;
  cv_pr_auc_std: number;
  engineered: string[];
  validation_pr_auc?: number;
}

export interface ModelComparisonReport {
  comparison: { selection_metric: string; selected: string; models: ComparisonRow[] };
  feature_ablation: {
    protocol: string;
    results: Record<string, Record<string, AblationEntry>>;
    chosen_per_model: Record<string, string[]>;
    selected_model_marginal: Record<string, AblationEntry>;
    definitions: Record<string, { formula: string; reason: string }>;
  };
}

export interface ExplainabilityReport {
  method: string;
  output_space: string;
  expected_value: number;
  global_sample: string;
  shap_additivity_max_abs_error: number;
  shap_global_importance_test: { feature: string; label: string; mean_abs_shap: number }[];
  permutation_importance_validation: { feature: string; label: string; mean: number; std: number }[];
  shap_beeswarm_test_sample: { features: string[]; sample: { shap: number[]; value: number[]; failed: number }[] };
}

export interface FailureModeReport {
  scope: string;
  training_records: number;
  training_class_counts: Record<string, number>;
  cv_macro_f1: Record<string, number>;
  selected: string;
  validation: ModeEval;
  test: ModeEval;
}

export interface ModeEval {
  records_scored: number;
  single_mode_records: number;
  multi_mode_records: number;
  macro_f1: number;
  accuracy_single_mode: number;
  any_match_accuracy_all: number;
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix: { labels: string[]; matrix: number[][] };
}

export interface DataQualityReport {
  source_file: string;
  sha256: string;
  random_seed: number;
  split_strategy: string;
  quality: {
    n_rows: number;
    n_columns: number;
    missing_values_total: number;
    duplicate_rows: number;
    failure_rate: number;
    imbalance_ratio: number;
    target_counts: Record<string, number>;
    failure_mode_counts: Record<string, number>;
    type_counts: Record<string, number>;
    label_consistency: {
      failure_without_mode: number;
      mode_without_failure: number;
      failures_with_multiple_modes: number;
    };
    outliers_iqr: Record<string, { n_outliers: number; pct_outliers: number }>;
    feature_ranges: Record<string, { min: number; max: number; mean: number; std: number }>;
  };
  splits: Record<"train" | "validation" | "test", { rows: number; failures: number; failure_rate: number }>;
}

export interface EdaReport {
  computed_on: string;
  n_rows: number;
  distributions: Record<string, {
    bins: number[]; normal: number[]; failed: number[];
    normal_median: number; failed_median: number; standardized_mean_difference: number;
  }>;
  rate_by_decile: Record<string, { low: number; high: number; rate: number; n: number; failures: number }[]>;
  failure_rate_by_type: Record<string, { rate: number; n: number; failures: number; ci_low: number; ci_high: number }>;
  correlation: { columns: string[]; matrix: number[][] };
  outlier_check: Record<string, { n_outliers: number; failure_rate_outliers: number | null; failure_rate_rest: number } | number>;
  scatter_sample: { rpm: number; torque: number; wear: number; temp_diff: number; failed: number }[];
}

export interface DemoCase {
  id: string;
  title: string;
  selection_rule: string;
  source: string;
  actual_failure: boolean | null;
  actual_failure_modes: string[];
  input: Record<string, unknown>;
}

export interface ValidationIssue {
  field: string;
  message: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly issues: ValidationIssue[];

  constructor(status: number, message: string, issues: ValidationIssue[] = []) {
    super(message);
    this.status = status;
    this.issues = issues;
  }
}

const BASE = (globalThis as { PREDICTIVEX_API?: string }).PREDICTIVEX_API ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  } catch {
    throw new ApiError(0, "Can't reach the PredictiveX server. Check that the backend is running on this address.");
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const b = (body ?? {}) as { message?: string; detail?: ValidationIssue[] | string; error?: string };
    if (res.status === 422 && Array.isArray(b.detail)) {
      throw new ApiError(422, "Some inputs are not valid.", b.detail);
    }
    if (res.status === 503) {
      throw new ApiError(503, b.message ?? "The model is not loaded. Train it with scripts/train_models.py.");
    }
    throw new ApiError(res.status, b.message ?? (typeof b.detail === "string" ? b.detail : `Request failed (${res.status})`));
  }
  return body as T;
}

const post = <T,>(path: string, payload: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(payload) });

export const api = {
  health: () => request<{ status: string; model_loaded: boolean }>("/health"),
  modelInfo: () => request<ModelInfo>("/model/info"),
  metrics: () => request<MetricsReport>("/metrics"),
  modelComparison: () => request<ModelComparisonReport>("/model-comparison"),
  featureImportance: () => request<ExplainabilityReport>("/feature-importance"),
  failureMode: () => request<FailureModeReport>("/failure-mode"),
  datasetSummary: () => request<DataQualityReport>("/dataset/summary"),
  eda: () => request<EdaReport>("/dataset/eda"),
  demoCases: () => request<{ note: string; cases: DemoCase[] }>("/demo-cases"),
  predict: (machine: unknown) => post<Prediction & { input: MachineInput }>("/predict", { machine }),
  simulate: (base: MachineInput, changes: Partial<MachineInput>) => post<Simulation>("/simulate", { base, changes }),
};
