import io
import math
import base64
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.model import CLASS_NAMES


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan context to ensure model is loaded."""
    with TestClient(app) as test_client:
        yield test_client


def create_synthetic_image_bytes(format="JPEG", size=(256, 256), color=(128, 128, 128)) -> bytes:
    """Helper to generate an in-memory image for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_health_endpoint(client):
    """GET /api/health should return ok and model_loaded=True."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "device" in data


def test_predict_dummy_image(client):
    """POST /api/predict on a generated dummy image returns probabilities summing to ~1 and valid Grad-CAM."""
    img_bytes = create_synthetic_image_bytes(format="JPEG", size=(300, 300))
    files = {"file": ("dummy_knee.jpg", img_bytes, "image/jpeg")}

    response = client.post("/api/predict", files=files)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["predicted_class"] in CLASS_NAMES
    probs = data["probabilities"]
    assert set(probs.keys()) == set(CLASS_NAMES)

    total_prob = probs["normal"] + probs["osteopenia"] + probs["osteoporosis"]
    assert math.isclose(total_prob, 1.0, rel_tol=1e-3)

    assert data["original_size"] == [300, 300]
    assert data["inference_ms"] > 0

    # Verify Grad-CAM overlay is valid non-empty base64 PNG
    overlay_b64 = data["gradcam_overlay_png_base64"]
    assert len(overlay_b64) > 100
    raw_overlay = base64.b64decode(overlay_b64)
    # Valid PNG starts with 8-byte signature: \x89PNG\r\n\x1a\n
    assert raw_overlay.startswith(b"\x89PNG\r\n\x1a\n")


def test_predict_with_target_class(client):
    """POST /api/predict with target_class produces heatmap for that specific class."""
    img_bytes = create_synthetic_image_bytes(format="PNG", size=(224, 224))
    files = {"file": ("dummy_knee.png", img_bytes, "image/png")}

    response = client.post("/api/predict?target_class=osteopenia", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in CLASS_NAMES
    assert len(data["gradcam_overlay_png_base64"]) > 0


def test_predict_invalid_target_class(client):
    """POST /api/predict with invalid target_class returns 400 Bad Request."""
    img_bytes = create_synthetic_image_bytes()
    files = {"file": ("knee.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/predict?target_class=invalid_class", files=files)
    assert response.status_code == 400
    assert "Invalid target_class" in response.json()["detail"]


def test_predict_invalid_file_type(client):
    """Uploading a non-image file (.txt) returns 400 Bad Request."""
    fake_text = b"This is a text file, not a radiography image."
    files = {"file": ("report.txt", fake_text, "text/plain")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400


def test_predict_corrupt_image(client):
    """Uploading corrupt image bytes returns 400 Bad Request."""
    corrupt_bytes = b"\xff\xd8\xff\xe0corruptdatawithoutproperheader"
    files = {"file": ("broken.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400


def test_predict_oversized_dimensions(client):
    """Uploading an image with dimensions > 4096 px returns 400 Bad Request."""
    # Create 4097 x 100 image
    img_bytes = create_synthetic_image_bytes(format="JPEG", size=(4097, 100))
    files = {"file": ("too_wide.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert "dimensions exceed maximum allowed" in response.json()["detail"].lower()


def test_metrics_endpoint(client):
    """GET /api/metrics serves locked test results without modification."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()

    # Exact locked metrics checks
    assert math.isclose(data["accuracy"], 0.8682170542635659, rel_tol=1e-5)
    assert len(data["bootstrap_95_ci"]) == 2
    assert math.isclose(data["macro_f1"], 0.8708694813957972, rel_tol=1e-5)
    assert math.isclose(data["roc_auc_macro"], 0.9345674663778111, rel_tol=1e-5)
    assert len(data["confusion_matrix"]) == 3
    assert len(data["confusion_matrix"][0]) == 3
    assert data["class_names"] == CLASS_NAMES
