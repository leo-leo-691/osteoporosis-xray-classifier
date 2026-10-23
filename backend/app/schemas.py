from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    model_loaded: bool


class Probabilities(BaseModel):
    normal: float
    osteopenia: float
    osteoporosis: float


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: Probabilities
    gradcam_overlay_png_base64: str
    original_size: Tuple[int, int]  # (width, height)
    inference_ms: float


class MetricsResponse(BaseModel):
    accuracy: float
    bootstrap_95_ci: List[float]
    wald_95_ci: List[float]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    roc_auc_normal: float
    roc_auc_osteopenia: float
    roc_auc_osteoporosis: float
    roc_auc_macro: float
    confusion_matrix: List[List[int]]
    class_names: List[str]
