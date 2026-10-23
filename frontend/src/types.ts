export type ClassName = 'normal' | 'osteopenia' | 'osteoporosis';

export interface HealthResponse {
  status: string;
  device: string;
  model_loaded: boolean;
}

export interface Probabilities {
  normal: number;
  osteopenia: number;
  osteoporosis: number;
}

export interface PredictResponse {
  predicted_class: ClassName;
  probabilities: Probabilities;
  gradcam_overlay_png_base64: string;
  original_size: [number, number];
  inference_ms: number;
}

export interface MetricsResponse {
  accuracy: number;
  bootstrap_95_ci: [number, number];
  wald_95_ci: [number, number];
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_f1: number;
  per_class_precision: number[];
  per_class_recall: number[];
  per_class_f1: number[];
  roc_auc_normal: number;
  roc_auc_osteopenia: number;
  roc_auc_osteoporosis: number;
  roc_auc_macro: number;
  confusion_matrix: number[][];
  class_names: string[];
}

export type ActiveTab = 'analyze' | 'evaluation';
