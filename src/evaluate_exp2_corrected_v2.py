"""
EXPERIMENT 2 CORRECTED EVALUATION - v2 (BUG FIX)

WHY THIS FILE EXISTS
---------------------
The original `evaluate_exp2_corrected.py` computes a real
`classification_report` string (used for the console printout and the
human-readable `corrected_metrics.txt`), but its JSON output
(`corrected_metrics.json`) hardcodes a *different*, literal
"class_metrics" dict labeled "# Placeholder to be updated on run" that is
never actually replaced with the real, computed numbers. This means that
script, as written, cannot correctly regenerate its own JSON output.

This v2 script is a minimal, behavior-preserving fix: it is IDENTICAL to
the original in every respect (same checkpoint, same 256x256 resolution,
same transforms, same model architecture) EXCEPT that `class_metrics` in
the saved JSON is computed directly from
`classification_report(..., output_dict=True)` instead of being a
hardcoded literal. No modeling, data, or evaluation-protocol change of
any kind is made here - this is classified purely as an
[INFRASTRUCTURE / CODE-QUALITY] fix, not a scientific-method change.

WHAT THIS FILE DOES NOT DO
---------------------------
- It does NOT overwrite `results/exp2_corrected/corrected_metrics.json`
  (the historical stored result). Output is written to a new directory,
  `results/exp2_corrected_v2/`, so the original historical artifact is
  preserved untouched.
- It has NOT been executed as part of this audit. Running it requires
  the actual raw test images (`data/raw/...`) and the trained checkpoint
  `models/best_resnet50_exp2.pth`, neither of which is available in the
  cloud audit environment this fix was written in. Run it in the
  project's normal local/GPU environment, then diff the resulting
  `results/exp2_corrected_v2/corrected_metrics_v2.json` against the
  historical `results/exp2_corrected/corrected_metrics.json` to confirm
  the corrected code reproduces the same numbers already on disk (it
  should, since those on-disk numbers are independently verified in
  this audit to already be arithmetically correct - see
  FINAL_RESEARCH_INTEGRITY_STATUS.md section B).
- It does NOT touch the locked test set's status: it evaluates
  `test_final.csv` in exactly the same read-only, single-pass way the
  original script does. Re-running this script does not constitute a
  new "look" at the locked test set beyond what the original,
  already-completed Experiment 2 corrected evaluation already did - it
  is a reproducibility fix to that same existing evaluation, not a new
  evaluation.
"""

from pathlib import Path
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_resnet50_exp2.pth"

# NOTE: deliberately a NEW output directory so the historical
# results/exp2_corrected/corrected_metrics.json is never overwritten.
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp2_corrected_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CORRECT RESOLUTION FOR EXPERIMENT 2 (unchanged from original script)
IMAGE_SIZE = 256
BATCH_SIZE = 16

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("BONE RADIOGRAPHY - EXPERIMENT 2 CORRECTED EVALUATION v2 (256x256)")
print("(bug fix: class_metrics now computed from real predictions,")
print(" not hardcoded placeholders - see module docstring)")
print("=" * 60)
print(f"Device: {DEVICE}")


# ============================================================
# DATASET
# ============================================================
class BoneXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = PROJECT_ROOT / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        label = CLASS_TO_INDEX[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# TEST TRANSFORM (unchanged: corrected to 256x256, as in v1)
# ============================================================
weights = ResNet50_Weights.DEFAULT
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

# ============================================================
# LOAD TEST DATA
# ============================================================
print("\nLoading test dataset...")
test_df = pd.read_csv(TEST_CSV)
print(f"Test images: {len(test_df)}")

test_dataset = BoneXrayDataset(test_df, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

# ============================================================
# LOAD MODEL (unchanged from v1)
# ============================================================
print("\nLoading Experiment 2 model checkpoint...")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_FILE}")

checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.35),
    nn.Linear(num_features, len(CLASS_NAMES)),
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

# ============================================================
# PREDICTIONS (unchanged from v1)
# ============================================================
print("\nRunning predictions...")
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for batch_index, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# ============================================================
# METRICS
# ============================================================
accuracy = accuracy_score(all_labels, all_predictions)
cm = confusion_matrix(all_labels, all_predictions)

# Real classification_report as a dict, used for BOTH the printed
# report AND the JSON output, so the two can never diverge again.
report_dict = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
    output_dict=True,
)
report_str = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\n" + "=" * 60)
print("EXPERIMENT 2 CORRECTED RESULTS (v2, bug-fixed)")
print("=" * 60)
print(f"Test Accuracy: {accuracy * 100:.2f}% ({accuracy:.4f})")
print("\nClassification Report:\n", report_str)
print("\nConfusion Matrix:\n", cm)

# ============================================================
# SAVE (new filenames - historical corrected_metrics.* untouched)
# ============================================================
metrics_file = OUTPUT_DIR / "corrected_metrics_v2.txt"
with open(metrics_file, "w", encoding="utf-8") as f:
    f.write("EXPERIMENT 2 CORRECTED RESULTS v2 (256x256, bug-fixed JSON export)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test Accuracy: {accuracy * 100:.2f}% ({accuracy:.4f})\n\n")
    f.write("Classification Report:\n")
    f.write(report_str + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

json_file = OUTPUT_DIR / "corrected_metrics_v2.json"
results_dict = {
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist(),
    "class_metrics": {
        cls: {
            "precision": report_dict[cls]["precision"],
            "recall": report_dict[cls]["recall"],
            "f1-score": report_dict[cls]["f1-score"],
        }
        for cls in CLASS_NAMES
    },
    "macro_avg": {
        "precision": report_dict["macro avg"]["precision"],
        "recall": report_dict["macro avg"]["recall"],
        "f1-score": report_dict["macro avg"]["f1-score"],
    },
    "weighted_avg": {
        "precision": report_dict["weighted avg"]["precision"],
        "recall": report_dict["weighted avg"]["recall"],
        "f1-score": report_dict["weighted avg"]["f1-score"],
    },
    "note": (
        "Regenerated with the bug-fixed evaluate_exp2_corrected_v2.py. "
        "class_metrics here are computed directly from classification_report "
        "output_dict=True, not hardcoded. Compare against "
        "results/exp2_corrected/corrected_metrics.json (historical, v1) to "
        "confirm the two agree (they should, per FINAL_RESEARCH_INTEGRITY_STATUS.md)."
    ),
}
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=4)

print(f"\nCorrected (v2) results saved to: {OUTPUT_DIR}")
print("Historical results/exp2_corrected/corrected_metrics.json was NOT modified.")
