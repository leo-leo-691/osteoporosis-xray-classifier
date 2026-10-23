# ==============================================================================
# Stage 1: Build Frontend (Vite + React + TypeScript + Tailwind)
# ==============================================================================
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Runtime Container (Python 3.11 Slim with CPU-only PyTorch)
# ==============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MODEL_PATH=/app/models/final_locked_model.pth \
    METRICS_PATH=/app/results/final_test/final_test_metrics.json \
    FRONTEND_DIST=/app/frontend/dist

WORKDIR /app

# Install minimal OS dependencies for PIL / Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch & Torchvision wheels from official index for fast image size
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0

# Install FastAPI and backend dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend code, locked model, and test metrics
COPY backend/ ./backend/
COPY models/final_locked_model.pth ./models/final_locked_model.pth
COPY results/final_test/final_test_metrics.json ./results/final_test/final_test_metrics.json

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Hugging Face Spaces runs as user 1000
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
