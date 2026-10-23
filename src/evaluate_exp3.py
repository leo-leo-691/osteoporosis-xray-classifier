from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_final.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_resnet50_exp3.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = [
    "normal",
    "osteopenia",
    "osteoporosis",
]

CLASS_TO_INDEX = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("BONE RADIOGRAPHY - RESNET50 EXPERIMENT 3 EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


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

        image_path = (
            PROJECT_ROOT
            / row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = CLASS_TO_INDEX[
            row["label"]
        ]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TEST TRANSFORM
# ============================================================

weights = ResNet50_Weights.DEFAULT

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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

test_df = pd.read_csv(
    TEST_CSV
)

print(
    f"Test images: {len(test_df)}"
)

print("\nTest distribution:")

print(
    test_df["label"].value_counts()
)


test_dataset = BoneXrayDataset(
    test_df,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# LOAD EXPERIMENT 3 MODEL
# ============================================================

print("\nLoading Experiment 3 model...")

print(
    f"Checkpoint: {MODEL_FILE}"
)

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nExperiment 3 model not found:\n"
        f"{MODEL_FILE}\n\n"
        f"Make sure Experiment 3 training completed successfully."
    )


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

model = models.resnet50(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Sequential(

    nn.Dropout(
        p=0.35
    ),

    nn.Linear(
        num_features,
        len(CLASS_NAMES)
    ),
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

print("\nModel configuration:")

print("Architecture: ResNet-50")
print("Layer 1:      FROZEN")
print("Layer 2:      FROZEN")
print("Layer 3:      TRAINABLE")
print("Layer 4:      TRAINABLE")
print("FC:           TRAINABLE")
print("Dropout:      0.35")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_index, (
        images,
        labels
    ) in enumerate(test_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        processed = min(
            (batch_index + 1) * BATCH_SIZE,
            len(test_dataset)
        )

        if (
            processed % 25 == 0
            or processed == len(test_dataset)
        ):

            print(
                f"Processed "
                f"{processed}/{len(test_dataset)}"
            )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 3 RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:\n"
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)

print("\nClass order:")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{i}: {name}"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

BASELINE_ACCURACY = 0.7132
EXP1_ACCURACY = 0.8605
EXP2_ACCURACY = 0.8450

print(
    "\n" + "=" * 60
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 60
)

print(
    f"Baseline:       "
    f"{BASELINE_ACCURACY * 100:.2f}%"
)

print(
    f"Experiment 1:   "
    f"{EXP1_ACCURACY * 100:.2f}%"
)

print(
    f"Experiment 2:   "
    f"{EXP2_ACCURACY * 100:.2f}%"
)

print(
    f"Experiment 3:   "
    f"{accuracy * 100:.2f}%"
)

print(
    f"\nExp 3 vs Baseline: "
    f"{(accuracy - BASELINE_ACCURACY) * 100:+.2f} "
    f"percentage points"
)

print(
    f"Exp 3 vs Exp 1:   "
    f"{(accuracy - EXP1_ACCURACY) * 100:+.2f} "
    f"percentage points"
)

print(
    f"Exp 3 vs Exp 2:   "
    f"{(accuracy - EXP2_ACCURACY) * 100:+.2f} "
    f"percentage points"
)


# ============================================================
# BEST MODEL
# ============================================================

results = {
    "Baseline": BASELINE_ACCURACY,
    "Experiment 1": EXP1_ACCURACY,
    "Experiment 2": EXP2_ACCURACY,
    "Experiment 3": accuracy,
}

best_model = max(
    results,
    key=results.get
)

best_accuracy = results[best_model]

print(
    "\n" + "=" * 60
)

print(
    "CURRENT BEST MODEL"
)

print(
    "=" * 60
)

print(
    f"Best model:     {best_model}"
)

print(
    f"Best accuracy:  "
    f"{best_accuracy * 100:.2f}%"
)

if best_model == "Experiment 3":

    print(
        "\n✓ Experiment 3 is the NEW BEST MODEL."
    )

elif best_model == "Experiment 1":

    print(
        "\n→ Experiment 1 remains the BEST MODEL."
    )

elif best_model == "Experiment 2":

    print(
        "\n→ Experiment 2 is currently the BEST MODEL."
    )


# ============================================================
# TEST SET SAFETY
# ============================================================

print(
    "\nTest set was used only for final evaluation."
)

print(
    f"Test images evaluated: {len(test_df)}"
)

print(
    "\nExperiment 3 evaluation complete."
)

print("=" * 60)