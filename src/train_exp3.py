from pathlib import Path
import copy
import json

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train_final.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val_final.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "best_resnet50_exp3.pth"
HISTORY_FILE = MODEL_DIR / "training_history_exp3.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16

NUM_EPOCHS = 30

# Backbone learning rate
BACKBONE_LR = 5e-6

# Classification head learning rate
FC_LR = 5e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 7

NUM_WORKERS = 0

# Focal loss parameters
FOCAL_GAMMA = 2.0

# Small amount of label smoothing
LABEL_SMOOTHING = 0.05


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

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print("BONE RADIOGRAPHY - RESNET50 EXPERIMENT 3")
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

        image_path = PROJECT_ROOT / row["image_path"]

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
# TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT

imagenet_mean = weights.transforms().mean
imagenet_std = weights.transforms().std


# ------------------------------------------------------------
# Experiment 3 training augmentation
#
# Designed for radiographs:
# - small rotations
# - small translations
# - contrast changes
# - horizontal flip
#
# Avoid aggressive geometric distortion.
# ------------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.04, 0.04),
        scale=(0.96, 1.04)
    ),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)


print("\nDataset:")

print(
    f"Training images:   {len(train_df)}"
)

print(
    f"Validation images: {len(val_df)}"
)


print("\nTraining distribution:")

print(
    train_df["label"]
    .value_counts()
)


print("\nValidation distribution:")

print(
    val_df["label"]
    .value_counts()
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = BoneXrayDataset(
    train_df,
    transform=train_transform
)

val_dataset = BoneXrayDataset(
    val_df,
    transform=val_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=(DEVICE.type == "cuda")
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (

    train_df["label"]
    .value_counts()
    .reindex(CLASS_NAMES)
)


# Standard inverse-frequency weights
base_weights = (

    len(train_df)
    /
    (
        len(CLASS_NAMES)
        *
        class_counts
    )
)


# ------------------------------------------------------------
# Experiment 3 adjustment
#
# Experiment 1 showed:
#
# normal -> osteoporosis = major error source
#
# Therefore we slightly increase the importance of
# NORMAL and OSTEOPOROSIS while retaining support for
# OSTEOPENIA.
#
# This is intentionally conservative.
# ------------------------------------------------------------

class_weights = torch.tensor(

    [
        base_weights["normal"] * 1.15,

        base_weights["osteopenia"] * 0.90,

        base_weights["osteoporosis"] * 1.15,
    ],

    dtype=torch.float32
)


# Normalize weights so their average is approximately 1.

class_weights = (
    class_weights
    /
    class_weights.mean()
)


class_weights = class_weights.to(
    DEVICE
)


print("\nExperiment 3 class weights:")

for name, weight in zip(
    CLASS_NAMES,
    class_weights
):

    print(
        f"{name:15s}: {weight.item():.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading pretrained ResNet-50..."
)


model = models.resnet50(
    weights=weights
)


# ------------------------------------------------------------
# Freeze layer1 and layer2
# ------------------------------------------------------------

for parameter in model.parameters():

    parameter.requires_grad = False


# ------------------------------------------------------------
# Train layer3 and layer4
# ------------------------------------------------------------

for parameter in model.layer3.parameters():

    parameter.requires_grad = True


for parameter in model.layer4.parameters():

    parameter.requires_grad = True


# ------------------------------------------------------------
# Replace classifier
# ------------------------------------------------------------

num_features = model.fc.in_features


model.fc = nn.Sequential(

    nn.Dropout(
        p=0.35
    ),

    nn.Linear(
        num_features,
        len(CLASS_NAMES)
    )
)


# FC is trainable by default
for parameter in model.fc.parameters():

    parameter.requires_grad = True


model = model.to(
    DEVICE
)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\nModel parameters:")

print(
    f"Total parameters:     {total_parameters:,}"
)

print(
    f"Trainable parameters: {trainable_parameters:,}"
)


print("\nFine-tuning:")

print(
    "layer1: FROZEN"
)

print(
    "layer2: FROZEN"
)

print(
    "layer3: TRAINABLE"
)

print(
    "layer4: TRAINABLE"
)

print(
    "fc:     TRAINABLE"
)

print(
    "Dropout: 0.35"
)


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        weight=None,
        gamma=2.0,
        label_smoothing=0.0
    ):

        super().__init__()

        self.weight = weight

        self.gamma = gamma

        self.label_smoothing = label_smoothing


    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = F.cross_entropy(

            logits,

            targets,

            weight=self.weight,

            reduction="none",

            label_smoothing=self.label_smoothing
        )


        pt = torch.exp(
            -ce_loss
        )


        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss


        return focal_loss.mean()


criterion = FocalLoss(

    weight=class_weights,

    gamma=FOCAL_GAMMA,

    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    [

        {
            "params": model.layer3.parameters(),

            "lr": BACKBONE_LR
        },

        {
            "params": model.layer4.parameters(),

            "lr": BACKBONE_LR
        },

        {
            "params": model.fc.parameters(),

            "lr": FC_LR
        },

    ],

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3
)


# ============================================================
# TRAINING STATE
# ============================================================

best_val_loss = float(
    "inf"
)

best_model_weights = copy.deepcopy(
    model.state_dict()
)

epochs_without_improvement = 0

history = []


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    NUM_EPOCHS
):


    print(
        "\n" + "=" * 60
    )

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:


        images = images.to(

            DEVICE,

            non_blocking=True
        )


        labels = labels.to(

            DEVICE,

            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(

            outputs,

            labels
        )


        loss.backward()


        # Prevent extremely large gradients

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0
        )


        optimizer.step()


        running_loss += (

            loss.item()
            *
            images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (

            predictions == labels
        ).sum().item()


        total += labels.size(0)


    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():


        for images, labels in val_loader:


            images = images.to(

                DEVICE,

                non_blocking=True
            )


            labels = labels.to(

                DEVICE,

                non_blocking=True
            )


            outputs = model(
                images
            )


            loss = criterion(

                outputs,

                labels
            )


            val_running_loss += (

                loss.item()
                *
                images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            val_correct += (

                predictions == labels
            ).sum().item()


            val_total += labels.size(0)


    val_loss = (
        val_running_loss / val_total
    )


    val_accuracy = (
        val_correct / val_total
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_loss
    )


    # ========================================================
    # LOGGING
    # ========================================================

    print(

        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f}"
    )


    print(

        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:    {val_accuracy:.4f}"
    )


    print(

        f"LR layer3:  "
        f"{optimizer.param_groups[0]['lr']:.2e}"
    )


    print(

        f"LR layer4:  "
        f"{optimizer.param_groups[1]['lr']:.2e}"
    )


    print(

        f"LR fc:      "
        f"{optimizer.param_groups[2]['lr']:.2e}"
    )


    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "lr_layer3":
            optimizer.param_groups[0]["lr"],

        "lr_layer4":
            optimizer.param_groups[1]["lr"],

        "lr_fc":
            optimizer.param_groups[2]["lr"],
    })


    # ========================================================
    # BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "image_size":
                    IMAGE_SIZE,

                "model_name":
                    "resnet50",

                "experiment":
                    "experiment_3",

                "dropout":
                    0.35,

                "focal_gamma":
                    FOCAL_GAMMA,

                "label_smoothing":
                    LABEL_SMOOTHING,

                "backbone_lr":
                    BACKBONE_LR,

                "fc_lr":
                    FC_LR,

                "class_weights":
                    class_weights.cpu(),

            },

            MODEL_FILE
        )


        print(
            "✓ New best Experiment 3 model saved."
        )


        epochs_without_improvement = 0


    else:

        epochs_without_improvement += 1


        print(

            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping."
            )

            break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    HISTORY_FILE,
    "w"
) as f:

    json.dump(

        history,

        f,

        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "EXPERIMENT 3 TRAINING COMPLETE"
)

print(
    "=" * 60
)


print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)


print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)


print(
    "\nHistory saved to:"
)

print(
    HISTORY_FILE
)


print(
    "\nExperiment 3 complete."
)

print(
    "Test set was NOT used during training."
)