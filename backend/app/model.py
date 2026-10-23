from pathlib import Path
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import os

# Resolve paths relative to this file's directory (backend/app/model.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_FILE = Path(os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "final_locked_model.pth")))
METRICS_FILE = Path(os.environ.get("METRICS_PATH", str(PROJECT_ROOT / "results" / "final_test" / "final_test_metrics.json")))

IMAGE_SIZE = 224
CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# PyTorch ImageNet normalization constants as in ResNet50_Weights.DEFAULT
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
])


def create_model_architecture() -> nn.Module:
    """Instantiate torchvision resnet50 with custom 3-class classification head."""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    return model


def load_locked_model() -> nn.Module:
    """
    Load the locked model checkpoint read-only.
    Handles raw state_dict or dict containing 'model_state_dict'.
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Locked model file not found at: {MODEL_FILE}")

    # weights_only=True is safe with modern PyTorch if checkpoint allows, or fallback if needed
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)
    except Exception:
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)

    model = create_model_architecture()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    return model
