import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_final.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "final_locked_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================================================
# DATASET
# ============================================================
class BoneXrayDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
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

# ============================================================
# TRANSFORM
# ============================================================
weights_init = ResNet50_Weights.DEFAULT
normalization_mean = weights_init.transforms().mean
normalization_std = weights_init.transforms().std

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalization_mean, std=normalization_std),
])

# ============================================================
# MAIN EVALUATION
# ============================================================
def main():
    print("=" * 60)
    print("FINAL LOCKED TEST EVALUATION")
    print("=" * 60)
    
    # 1. Load Test Data
    test_df = pd.read_csv(TEST_CSV)
    print(f"Test split images: {len(test_df)}")
    
    test_dataset = BoneXrayDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE)
    model.eval()
    
    # 3. Predict
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    
    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_filenames.extend(filenames)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    _, _, f_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    
    p_class, r_class, f_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # ROC-AUC OVR
    num_classes = len(CLASS_NAMES)
    labels_onehot = np.zeros((len(all_labels), num_classes))
    for i, l in enumerate(all_labels):
        labels_onehot[i, l] = 1
        
    roc_auc_per_class = []
    for c in range(num_classes):
        auc = roc_auc_score(labels_onehot[:, c], all_probs[:, c])
        roc_auc_per_class.append(auc)
        
    macro_auc = roc_auc_score(labels_onehot, all_probs, multi_class="ovr", average="macro")
    
    # 5. Bootstrap 95% Confidence Interval for Accuracy
    # Lock seed for bootstrap determinism
    np.random.seed(42)
    boot_accs = []
    correct_array = (all_labels == all_preds).astype(int)
    for _ in range(10000):
        sample = np.random.choice(correct_array, size=len(correct_array), replace=True)
        boot_accs.append(np.mean(sample))
        
    boot_ci = [np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)]
    
    # Binomial exact / Wald comparison
    # Wald: p +/- 1.96 * sqrt(p(1-p)/N)
    wald_std = np.sqrt(acc * (1 - acc) / len(all_labels))
    wald_ci = [acc - 1.96 * wald_std, acc + 1.96 * wald_std]
    
    # 6. Save Predictions and Probabilities CSV
    results_csv = OUTPUT_DIR / "final_test_predictions.csv"
    preds_df = pd.DataFrame({
        "filename": all_filenames,
        "actual": [CLASS_NAMES[l] for l in all_labels],
        "predicted": [CLASS_NAMES[p] for p in all_preds],
        "correct": (all_labels == all_preds).tolist(),
        "confidence": [all_probs[i, all_preds[i]] for i in range(len(all_preds))],
        "prob_normal": all_probs[:, 0],
        "prob_osteopenia": all_probs[:, 1],
        "prob_osteoporosis": all_probs[:, 2]
    })
    preds_df.to_csv(results_csv, index=False)
    
    # 7. Print Results
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    
    print("\n" + "=" * 60)
    print("FINAL LOCKED TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {acc*100:.2f}% ({acc:.4f})")
    print(f"95% Bootstrap CI: [{boot_ci[0]*100:.2f}%, {boot_ci[1]*100:.2f}%]")
    print(f"95% Wald CI:      [{wald_ci[0]*100:.2f}%, {wald_ci[1]*100:.2f}%]")
    print(f"Macro F1-Score:   {f_macro:.4f}")
    print(f"Weighted F1-Score:{f_weighted:.4f}")
    print(f"Macro ROC-AUC:    {macro_auc:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    # 8. Save Metrics JSON and TXT
    metrics_json = {
        "accuracy": acc,
        "bootstrap_95_ci": boot_ci,
        "wald_95_ci": wald_ci,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_f1": f_weighted,
        "per_class_precision": p_class.tolist(),
        "per_class_recall": r_class.tolist(),
        "per_class_f1": f_class.tolist(),
        "roc_auc_normal": roc_auc_per_class[0],
        "roc_auc_osteopenia": roc_auc_per_class[1],
        "roc_auc_osteoporosis": roc_auc_per_class[2],
        "roc_auc_macro": macro_auc,
        "confusion_matrix": cm.tolist()
    }
    
    with open(OUTPUT_DIR / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)
        
    with open(OUTPUT_DIR / "final_test_metrics.txt", "w", encoding="utf-8") as f:
        f.write("FINAL LOCKED TEST EVALUATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy: {acc*100:.2f}% ({acc:.4f})\n")
        f.write(f"95% Bootstrap CI: [{boot_ci[0]*100:.2f}%, {boot_ci[1]*100:.2f}%]\n")
        f.write(f"95% Wald CI:      [{wald_ci[0]*100:.2f}%, {wald_ci[1]*100:.2f}%]\n")
        f.write(f"Macro F1-Score:   {f_macro:.4f}\n")
        f.write(f"Weighted F1-Score:{f_weighted:.4f}\n")
        f.write(f"Macro ROC-AUC:    {macro_auc:.4f}\n\n")
        f.write("Per-Class ROC-AUC:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {name:12s}: {roc_auc_per_class[i]:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        
    print(f"\nFinal test evaluation outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
