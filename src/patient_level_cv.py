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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from PIL import Image

# ============================================================
# SEED SETUP FOR DETERMINISM
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_CSV = PROJECT_ROOT / "data" / "processed" / "primary_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "patient_level_cv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================================================
# DATASET USING PRE-LOADED IMAGES
# ============================================================
class BoneXrayDataset(Dataset):
    def __init__(self, df, pre_loaded_images, transform=None):
        self.df = df.reset_index(drop=True)
        self.pre_loaded_images = pre_loaded_images
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = self.pre_loaded_images[row["patient_id"]]
        label = CLASS_TO_INDEX[row["diagnosis"]]
        if self.transform:
            image = self.transform(image)
        return image, label, row["patient_id"]

# ============================================================
# TRANSFORMS
# ============================================================
weights_init = ResNet50_Weights.DEFAULT
normalization_mean = weights_init.transforms().mean
normalization_std = weights_init.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalization_mean, std=normalization_std),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalization_mean, std=normalization_std),
])

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(config_name, train_df, val_df, pre_loaded_images, fold_idx):
    set_seed(42 + fold_idx) # Unique seed per fold
    
    # Check patient structures
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    assert len(train_patients.intersection(val_patients)) == 0, f"Leakage detected in Fold {fold_idx}!"
    
    train_dataset = BoneXrayDataset(train_df, pre_loaded_images, transform=train_transform)
    val_dataset = BoneXrayDataset(val_df, pre_loaded_images, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate inverse frequency class weights dynamically on train fold to avoid leak
    class_counts = train_df["diagnosis"].value_counts()
    total_samples = len(train_df)
    weights_list = []
    for cls in CLASS_NAMES:
        weights_list.append(total_samples / (len(CLASS_NAMES) * class_counts.get(cls, 1)))
    class_weights = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize Model
    model = models.resnet50(weights=weights_init)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    
    # Layer freezing configurations
    if config_name == "baseline":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-4)
        epochs = 20
        
    elif config_name == "experiment1":
        for param in model.parameters():
            param.requires_grad = False
            
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW([
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-4}
        ])
        epochs = 25
        
    model = model.to(DEVICE)
    
    best_loss = float("inf")
    best_model_state = None
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Load best state for evaluation
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    model.eval()
    
    # Save checkpoint
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(best_model_state, checkpoint_dir / f"{config_name}_fold{fold_idx}.pth")
    
    # Collect predictions and probabilities
    all_preds = []
    all_labels = []
    all_probs = []
    all_patients = []
    
    with torch.no_grad():
        for images, labels, patients in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_patients.extend(patients)
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_patients

# ============================================================
# METRICS COMPILATION
# ============================================================
def calculate_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    p_class, r_class, f_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    
    num_classes = len(CLASS_NAMES)
    labels_onehot = np.zeros((len(labels), num_classes))
    for i, l in enumerate(labels):
        labels_onehot[i, l] = 1
        
    roc_auc_per_class = []
    for c in range(num_classes):
        if len(np.unique(labels_onehot[:, c])) > 1:
            auc = roc_auc_score(labels_onehot[:, c], probs[:, c])
        else:
            auc = 0.5
        roc_auc_per_class.append(auc)
        
    try:
        macro_auc = roc_auc_score(labels_onehot, probs, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = np.mean(roc_auc_per_class)
        
    return {
        "accuracy": acc,
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
        "roc_auc_macro": macro_auc
    }

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("RUNNING PATIENT-LEVEL 5-FOLD CROSS-VALIDATION (OPTIMIZED)")
    print("=" * 60)
    
    df = pd.read_csv(PRIMARY_CSV)
    print(f"Total primary records: {len(df)}")
    print(f"Unique patient count: {df['patient_id'].nunique()}")
    
    # Pre-load all primary images in memory resized to 256x256
    print("\nPre-loading and resizing images to RAM...")
    pre_loaded_images = {}
    for idx, row in df.iterrows():
        img_path = PROJECT_ROOT / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((256, 256), Image.Resampling.BILINEAR)
        pre_loaded_images[row["patient_id"]] = img_resized
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"Loaded {idx + 1}/{len(df)} images.")
            
    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for config_name in ["baseline", "experiment1"]:
        print(f"\nEvaluating Model Configuration: {config_name.upper()}")
        print("-" * 60)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["diagnosis"]), 1):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            print(f"Fold {fold_idx} | Train count: {len(train_df)} | Val count: {len(val_df)}")
            
            labels, preds, probs, patients = train_model(config_name, train_df, val_df, pre_loaded_images, fold_idx)
            
            metrics = calculate_metrics(labels, preds, probs)
            fold_metrics.append(metrics)
            
            print(f"  Fold {fold_idx} Accuracy: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")
            
            # Save fold predictions csv
            fold_preds = pd.DataFrame({
                "patient_id": patients,
                "actual": [CLASS_NAMES[l] for l in labels],
                "predicted": [CLASS_NAMES[p] for p in preds],
                "prob_normal": probs[:, 0],
                "prob_osteopenia": probs[:, 1],
                "prob_osteoporosis": probs[:, 2]
            })
            
            fold_dir = OUTPUT_DIR / config_name / f"fold{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_preds.to_csv(fold_dir / "predictions.csv", index=False)
            
            # Save confusion matrix and classification report
            cm = confusion_matrix(labels, preds)
            report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
            
            with open(fold_dir / "classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report + "\n")
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + "\n")
                
            with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
                
        # Aggregate statistics across folds
        summary = {}
        keys = fold_metrics[0].keys()
        
        for k in keys:
            if isinstance(fold_metrics[0][k], list):
                list_length = len(fold_metrics[0][k])
                means = []
                stds = []
                cis = []
                for idx in range(list_length):
                    values = [m[k][idx] for m in fold_metrics]
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    ci_val = 1.96 * (std_val / np.sqrt(5))
                    means.append(mean_val)
                    stds.append(std_val)
                    cis.append(ci_val)
                summary[k] = {"mean": means, "std": stds, "95%_ci": cis}
            else:
                values = [m[k] for m in fold_metrics]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                ci_val = 1.96 * (std_val / np.sqrt(5))
                summary[k] = {"mean": mean_val, "std": std_val, "95%_ci": ci_val}
                
        summary_file = OUTPUT_DIR / f"{config_name}_cv_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
            
        print(f"\n{config_name.upper()} CV Mean Accuracy: {summary['accuracy']['mean']*100:.2f}% (± {summary['accuracy']['std']*100:.2f}%)")
        print(f"{config_name.upper()} CV Mean Macro F1:  {summary['macro_f1']['mean']:.4f} (± {summary['macro_f1']['std']:.4f})")
        print(f"{config_name.upper()} CV Mean Macro AUC: {summary['roc_auc_macro']['mean']:.4f}")
        
    print("\nPatient-Level Cross-Validation completed successfully.")

if __name__ == "__main__":
    main()
