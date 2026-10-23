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
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train_final.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val_final.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "final_locked_model.pth"
HISTORY_FILE = PROJECT_ROOT / "models" / "final_locked_model_history.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================================================
# DATASET
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
        image = self.pre_loaded_images[row["image_path"]]
        label = CLASS_TO_INDEX[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label

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
# TRAINING PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("TRAINING FINAL LOCKED MODEL CONFIGURATION")
    print("=" * 60)
    set_seed(42)
    
    # Load splits
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"Train count: {len(train_df)} | Val count: {len(val_df)}")
    
    # Pre-load train/val images in memory
    print("\nPre-loading train and validation images...")
    pre_loaded_images = {}
    
    # Combine dataframes for loading convenience
    combined_paths = pd.concat([train_df["image_path"], val_df["image_path"]]).unique()
    for idx, path in enumerate(combined_paths, 1):
        img_path = PROJECT_ROOT / path
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((256, 256), Image.Resampling.BILINEAR)
        pre_loaded_images[path] = img_resized
        if idx % 100 == 0 or idx == len(combined_paths):
            print(f"Loaded {idx}/{len(combined_paths)} images.")
            
    train_dataset = BoneXrayDataset(train_df, pre_loaded_images, transform=train_transform)
    val_dataset = BoneXrayDataset(val_df, pre_loaded_images, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Setup model (Experiment 1 configuration)
    print("\nInitializing ResNet-50 model architecture...")
    model = models.resnet50(weights=weights_init)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    
    # Freeze layers 1 & 2, keep 3 & 4 trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    model = model.to(DEVICE)
    
    # Loss Function (Inverse-frequency weights from training split)
    # normal: 368, osteopenia: 200, osteoporosis: 32 (approximately, let's calculate dynamically on train_df)
    class_counts = train_df["label"].value_counts()
    total_samples = len(train_df)
    weights_list = []
    for cls in CLASS_NAMES:
        weights_list.append(total_samples / (len(CLASS_NAMES) * class_counts.get(cls, 1)))
    class_weights = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
    print("Class counts in train:", class_counts.to_dict())
    print("Dynamic Class weights:", weights_list)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW([
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ])
    
    best_loss = float("inf")
    best_model_state = None
    
    epochs = 25
    history = []
    
    # Training Loop
    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
        train_loss = train_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Save best checkpoint
    print(f"\nTraining completed. Saving best model checkpoint with Val Loss: {best_loss:.4f}...")
    torch.save(best_model_state, MODEL_FILE)
    
    # Save training history
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    print(f"Model saved to: {MODEL_FILE}")
    print(f"History saved to: {HISTORY_FILE}")

if __name__ == "__main__":
    main()
