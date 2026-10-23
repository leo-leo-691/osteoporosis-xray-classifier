# Deploying OsteoKnee to Hugging Face Spaces

This guide outlines step-by-step instructions to deploy the OsteoKnee web application (FastAPI + React SPA) to **Hugging Face Spaces** using the **Docker SDK**.

---

## 1. Prerequisites

1. A Hugging Face account ([huggingface.co](https://huggingface.co/)).
2. [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/) installed locally.
   ```bash
   git lfs install
   ```
3. A Hugging Face User Access Token (with Write permissions) from your Settings → Access Tokens.

---

## 2. Create a New Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set a **Space name** (e.g., `osteoknee-radiology`).
3. Select **License** (e.g., `mit` or `apache-2.0`).
4. Select **Space SDK**: **Docker** (Blank template).
5. Hardware: **CPU Basic (2 vCPU · 16 GB RAM)** — Free tier is sufficient since the container uses optimized CPU PyTorch wheels.
6. Click **Create Space**.

---

## 3. Clone and Configure Git LFS

Clone your newly created Space repository locally:

```bash
git clone https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
cd <SPACE_NAME>
```

Initialize Git LFS to track large PyTorch model checkpoints:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

---

## 4. Copy Deployment Files into the Space

Copy the following files and directories from this repository into the cloned Space folder:

```text
├── Dockerfile
├── .dockerignore
├── README.md
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── gradcam.py
│   │   └── schemas.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── tsconfig.app.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── public/
│   │   └── favicon.svg
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── index.css
│       ├── config.ts
│       ├── types.ts
│       └── components/
│           ├── Banner.tsx
│           ├── Header.tsx
│           ├── AnalyzeView.tsx
│           ├── EvaluationView.tsx
│           └── ErrorBoundary.tsx
├── models/
│   └── final_locked_model.pth          <-- (Tracked via Git LFS)
└── results/
    └── final_test/
        └── final_test_metrics.json
```

---

## 5. Hugging Face Spaces Configuration

Hugging Face Spaces expects the container to listen on port `7860`. The included `Dockerfile` already configures:
- `EXPOSE 7860`
- `CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]`
- Non-root user with UID `1000` (`user:user`).

Add frontmatter to the Space's `README.md` so Hugging Face detects the configuration:

```yaml
---
title: OsteoKnee
emoji: 🦴
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---
```

---

## 6. Commit and Push

```bash
git add .
git commit -m "Deploy OsteoKnee DICOM Lightbox research demo"
git push origin main
```

Git LFS will upload `models/final_locked_model.pth` directly to the Hugging Face LFS store.

---

## 7. Monitor Build & Verification

1. In your Hugging Face Space page, navigate to the **Logs** tab.
2. Watch Stage 1 compile the Vite React bundle.
3. Watch Stage 2 install CPU PyTorch and launch Uvicorn on port `7860`.
4. Once the status turns to **Running**, the single container serves the full DICOM Lightbox UI and all `/api/*` endpoints.
