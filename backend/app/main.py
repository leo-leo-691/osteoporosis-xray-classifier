import io
import json
import os
import time
import sys
from pathlib import Path
import threading
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

# Ensure project root and backend directory are in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torch
from PIL import Image, UnidentifiedImageError

try:
    from backend.app.schemas import HealthResponse, PredictResponse, Probabilities, MetricsResponse
    from backend.app.model import (
        load_locked_model,
        preprocess,
        CLASS_NAMES,
        CLASS_TO_INDEX,
        DEVICE,
        METRICS_FILE,
        MODEL_FILE
    )
    from backend.app.gradcam import GradCAM, overlay_heatmap, overlay_to_base64_png
except ImportError:
    from app.schemas import HealthResponse, PredictResponse, Probabilities, MetricsResponse
    from app.model import (
        load_locked_model,
        preprocess,
        CLASS_NAMES,
        CLASS_TO_INDEX,
        DEVICE,
        METRICS_FILE,
        MODEL_FILE
    )
    from app.gradcam import GradCAM, overlay_heatmap, overlay_to_base64_png

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
MAX_IMAGE_DIMENSION = 4096              # Reject images > 4096 px on either axis
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# Global model state protected by threading.Lock
model_lock = threading.Lock()
loaded_model = None
grad_cam_instance: Optional[GradCAM] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the locked model and initialize Grad-CAM once at startup."""
    global loaded_model, grad_cam_instance
    try:
        print(f"Loading locked model from {MODEL_FILE} onto {DEVICE}...")
        loaded_model = load_locked_model()
        target_layer = loaded_model.layer4[-1]
        grad_cam_instance = GradCAM(loaded_model, target_layer)
        print("Model and Grad-CAM initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load locked model: {e}")
        loaded_model = None
        grad_cam_instance = None
    yield
    # Cleanup hooks on shutdown if initialized
    if grad_cam_instance is not None:
        grad_cam_instance.remove_hooks()


app = FastAPI(
    title="OsteoKnee AI",
    description="Knee X-ray bone density classifier & Grad-CAM visualizer — research demo",
    version="1.1.0",
    lifespan=lifespan
)

# Configure CORS: use ALLOWED_ORIGINS env var if present; no wildcard in production
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint reporting device and model readiness."""
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        model_loaded=(loaded_model is not None and grad_cam_instance is not None)
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict_xray(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    target_class: Optional[str] = Query(None, description="Optional class to compute Grad-CAM for ('normal', 'osteopenia', 'osteoporosis' or 0, 1, 2)")
):
    """
    Run osteoporosis prediction and Grad-CAM heatmap generation on an uploaded knee X-ray.
    Accepts multipart form file under either 'file' or 'image'.
    Does not log or store uploaded images on disk.
    """
    upload = file or image
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please provide an image file."
        )

    # 1. Validate content type if supplied
    if upload.content_type and upload.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {upload.content_type}. Only JPEG and PNG are allowed."
        )

    # 2. Read bytes in memory and validate file size (max 10MB)
    try:
        content = await upload.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size limit of 10 MB (received {len(content) / (1024 * 1024):.2f} MB)."
        )

    # 3. Decode image with PIL purely in memory
    try:
        pil_img = Image.open(io.BytesIO(content))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or invalid image file: {str(e)}"
        )

    orig_width, orig_height = pil_img.size

    # Validate image dimensions (reject > 4096 px)
    if orig_width > MAX_IMAGE_DIMENSION or orig_height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions exceed maximum allowed {MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION} px (received {orig_width}×{orig_height} px)."
        )

    # 4. Resolve target_class if requested
    target_idx: Optional[int] = None
    if target_class is not None and target_class.strip() != "":
        norm_target = target_class.strip().lower()
        if norm_target in CLASS_TO_INDEX:
            target_idx = CLASS_TO_INDEX[norm_target]
        elif norm_target.isdigit() and int(norm_target) in range(len(CLASS_NAMES)):
            target_idx = int(norm_target)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid target_class '{target_class}'. Allowed values: {CLASS_NAMES} or 0, 1, 2."
            )

    # 5. Check model loaded
    if loaded_model is None or grad_cam_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or failed to initialize."
        )

    # 6. Preprocess image
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    # 7. Thread-safe inference & Grad-CAM pass (NO torch.no_grad())
    start_time = time.perf_counter()
    with model_lock:
        heatmap, explained_class_idx, logits = grad_cam_instance.generate_heatmap(
            input_tensor, class_idx=target_idx
        )
        # Probabilities calculated via softmax
        probs_tensor = torch.softmax(logits, dim=1).detach().cpu().squeeze(0)
        predicted_idx = int(probs_tensor.argmax().item())

    inference_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # 8. Overlay heatmap on original image and encode to base64
    overlay = overlay_heatmap(pil_img, heatmap, alpha=0.45)
    overlay_b64 = overlay_to_base64_png(overlay)

    probabilities = Probabilities(
        normal=float(probs_tensor[0].item()),
        osteopenia=float(probs_tensor[1].item()),
        osteoporosis=float(probs_tensor[2].item())
    )

    return PredictResponse(
        predicted_class=CLASS_NAMES[predicted_idx],
        probabilities=probabilities,
        gradcam_overlay_png_base64=overlay_b64,
        original_size=(orig_width, orig_height),
        inference_ms=inference_ms
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics():
    """Serve locked test evaluation metrics from results/final_test/final_test_metrics.json read-only."""
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics file not found at: {METRICS_FILE}"
        )
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["class_names"] = CLASS_NAMES
        return MetricsResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading metrics: {str(e)}"
        )


# Serve built frontend if dist exists (Production / Hugging Face Spaces single-server mode)
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", str(PROJECT_ROOT / "frontend" / "dist")))
if not FRONTEND_DIST.exists() and (BACKEND_DIR / "dist").exists():
    FRONTEND_DIST = BACKEND_DIR / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend_spa(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        index_html = FRONTEND_DIST / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        raise HTTPException(status_code=404, detail="Page not found")
