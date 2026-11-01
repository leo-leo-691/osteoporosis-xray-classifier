# Osteoporosis Knee X-Ray Radiography ML Research Demo

This repository contains the locked ResNet-50 research model, locked test evaluation results, and a local interactive web application for osteoporosis classification and Grad-CAM interpretability from knee radiographs.

> **Research Prototype**: This web application is a research demonstration prototype and is **not** intended for clinical diagnosis or clinical decision-making.

---

## Quick Start (Windows)

Double-click `run_app.bat` in the repository root to launch both the FastAPI backend and Vite frontend development server concurrently.

Alternatively, you can launch each service manually from separate terminals:

### 1. Backend Service (FastAPI + Uvicorn)

The backend utilizes the existing virtual environment containing PyTorch, Torchvision, NumPy, Matplotlib, and Pillow:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API Base: `http://localhost:8000`
- Interactive Swagger Documentation: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/health`
- Locked Metrics Endpoint: `http://localhost:8000/api/metrics`

To run the backend test suite (from the repository root):
```bash
pytest backend/tests -v
```

### 2. Frontend Application (React + Vite + TypeScript + Tailwind)

```bash
cd frontend
npm install
npm run dev
```

- Web App: `http://localhost:5173`
- The Vite dev server automatically proxies `/api` requests to `http://localhost:8000`.

### 3. Verified Smoke-Test Image

The web application was smoke-tested using a verified training split image from `data/processed/train_final.csv`:
- **Image**: `data/raw/primary/Osteoporosis Knee X-ray/normal/N26.jpg`
- **Model Output**: Predicted class `osteopenia`
- **Output Probabilities**: Normal: 7.94%, Osteopenia: 87.56%, Osteoporosis: 4.50%
- **Inference Latency**: ~60 ms (CUDA)

Note: N26.jpg is a training image labelled normal and was misclassified as osteopenia. Single-image demos are not evaluations and must not be read as model accuracy.

### 4. Fidelity Check (Read-Only)

Batched inference (batch size 16, as in evaluate_final_test.py) reproduces final_test_predictions.csv exactly. Single-image inference through the API differs by up to ~6e-4 in probability because cuDNN selects different convolution algorithms (and TF32 paths) for different batch sizes. The predicted class is unchanged.

---

## Deployment

To deploy the app to Hugging Face Spaces with Docker, see [DEPLOY.md](DEPLOY.md).

---

## App Features

1. **Analyze Page**:
   - Drag-and-drop or file upload for knee radiographs (JPEG/PNG up to 10 MB).
   - Instant inference with predicted class badge, model output probabilities (Softmax outputs; not calibrated), and latency counter.
   - Dual-mode Grad-CAM visualizer:
     - **Side-by-Side View**: Compares original radiograph with an adjustable-opacity Grad-CAM overlay.
     - **Interactive Split Slider**: Draggable curtain wipe divider between the raw image and the saliency heatmap.
   - **Explain Class Selector**: Statelessly re-computes Grad-CAM activations targeting specific non-predicted classes (Normal, Osteopenia, Osteoporosis).
   - Clinical disclaimer note.

2. **Model & Evaluation Page**:
   - Architecture summary: ResNet-50, ImageNet-pretrained, layer3/layer4 fine-tuned, 224×224 input, weighted cross-entropy, AdamW, 25 epochs, seed 42.
   - Locked test set metrics dynamically loaded from `results/final_test/final_test_metrics.json`:
     - Test Accuracy with $n = 129$ (112 correct) and Bootstrap 95% Confidence Interval.
     - Macro F1-Score and Macro ROC-AUC.
     - Per-class classification table (Precision, Recall, F1, ROC-AUC).
     - 3×3 Confusion Matrix heatmap with explicit row/column orientation (`rows = true class, columns = predicted class`).
   - Verbatim study limitations panel detailing image-level vs. patient-level validation gaps.
