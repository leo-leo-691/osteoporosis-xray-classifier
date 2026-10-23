from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_resnet50_exp1.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATASET & TRANSFORM
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
        return image, label, str(image_path.name)

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
# LOAD DATA & MODEL
# ============================================================
test_df = pd.read_csv(TEST_CSV)
test_dataset = BoneXrayDataset(test_df, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
# PREDICTIONS & PROBABILITIES
# ============================================================
records = []
with torch.no_grad():
    for images, labels, filenames in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        for i in range(len(labels)):
            actual = labels[i].item()
            pred = preds[i]
            prob = probs[i]
            
            records.append({
                "filename": filenames[i],
                "actual": actual,
                "predicted": pred,
                "actual_label": CLASS_NAMES[actual],
                "predicted_label": CLASS_NAMES[pred],
                "correct": actual == pred,
                "confidence": prob[pred],
                "prob_normal": prob[0],
                "prob_osteopenia": prob[1],
                "prob_osteoporosis": prob[2]
            })

results_df = pd.DataFrame(records)

# ============================================================
# ANALYSIS
# ============================================================
print("=" * 60)
print("EXPERIMENT 1 CONFIDENCE ANALYSIS")
print("=" * 60)

# 1. Correct vs. Incorrect Predictions
correct_mask = results_df["correct"] == True
incorrect_mask = results_df["correct"] == False

avg_conf_correct = results_df.loc[correct_mask, "confidence"].mean()
avg_conf_incorrect = results_df.loc[incorrect_mask, "confidence"].mean()

print(f"Average confidence for CORRECT predictions:   {avg_conf_correct:.4f} ({avg_conf_correct*100:.2f}%)")
print(f"Average confidence for INCORRECT predictions: {avg_conf_incorrect:.4f} ({avg_conf_incorrect*100:.2f}%)")

# 2. Normal <-> Osteoporosis Mistakes
print("\n" + "-" * 60)
print("CONFUSION FOCUS: NORMAL <-> OSTEOPOROSIS")
print("-" * 60)

# Normal misclassified as Osteoporosis
normal_as_osteo = results_df[(results_df["actual_label"] == "normal") & (results_df["predicted_label"] == "osteoporosis")]
print(f"\nNormal misclassified as Osteoporosis (Count: {len(normal_as_osteo)}):")
if len(normal_as_osteo) > 0:
    for _, row in normal_as_osteo.iterrows():
        print(f"  File: {row['filename']} | Confidence: {row['confidence']:.4f} (N: {row['prob_normal']:.2f}, Op: {row['prob_osteopenia']:.2f}, Os: {row['prob_osteoporosis']:.2f})")

# Osteoporosis misclassified as Normal
osteo_as_normal = results_df[(results_df["actual_label"] == "osteoporosis") & (results_df["predicted_label"] == "normal")]
print(f"\nOsteoporosis misclassified as Normal (Count: {len(osteo_as_normal)}):")
if len(osteo_as_normal) > 0:
    for _, row in osteo_as_normal.iterrows():
        print(f"  File: {row['filename']} | Confidence: {row['confidence']:.4f} (N: {row['prob_normal']:.2f}, Op: {row['prob_osteopenia']:.2f}, Os: {row['prob_osteoporosis']:.2f})")

# 3. High-confidence mistakes vs. Low-confidence ambiguous cases
print("\n" + "-" * 60)
print("ERROR SEVERITY AUDIT")
print("-" * 60)

high_conf_errors = results_df[incorrect_mask & (results_df["confidence"] >= 0.70)]
low_conf_errors = results_df[incorrect_mask & (results_df["confidence"] < 0.70)]

print(f"High-confidence errors (Confidence >= 70%): {len(high_conf_errors)} / {len(results_df[incorrect_mask])}")
print(f"Low-confidence/Ambiguous errors (Confidence < 70%): {len(low_conf_errors)} / {len(results_df[incorrect_mask])}")

if len(high_conf_errors) > 0:
    print("\nDetailed High-confidence Errors:")
    for _, row in high_conf_errors.sort_values("confidence", ascending=False).iterrows():
        print(f"  {row['filename']}: Actual={row['actual_label']:12s} | Predicted={row['predicted_label']:12s} | Confidence={row['confidence']:.4f}")
