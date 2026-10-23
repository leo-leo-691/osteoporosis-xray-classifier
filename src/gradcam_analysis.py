import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "best_resnet50_exp1.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# GRAD-CAM CLASS
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1)
        
        # Apply ReLU to keep only positive influence features
        cam = torch.clamp(cam, min=0)
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.cpu().numpy()[0], class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

# ============================================================
# UTILITIES
# ============================================================
weights_init = ResNet50_Weights.DEFAULT
normalization_mean = weights_init.transforms().mean
normalization_std = weights_init.transforms().std

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalization_mean, std=normalization_std),
])

def overlay_heatmap(original_img, heatmap, alpha=0.45):
    # Resize heatmap to match original image size
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_img.size, Image.Resampling.BILINEAR)
    heatmap_np = np.array(heatmap_resized) / 255.0
    
    # Apply colormap (Jet colormap manually via matplotlib)
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_np)[:, :, :3] # Keep RGB, drop alpha
    
    original_np = np.array(original_img) / 255.0
    
    # Ensure original is RGB
    if len(original_np.shape) == 2:
        original_np = np.stack([original_np]*3, axis=-1)
        
    overlay = (1 - alpha) * original_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    
    return overlay

# ============================================================
# MAIN GRAD-CAM RUN
# ============================================================
def main():
    print("=" * 60)
    print("RUNNING GRAD-CAM EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    
    # 1. Load Model
    print("Loading Experiment 1 model...")
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
    
    # Target the last bottleneck block of the ResNet-50 backbone
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # 2. Images to Analyze (path, actual_label, prediction_label)
    # Selected based on confidence analysis and split CSVs
    images_to_analyze = [
        # Correctly Classified
        {
            "path": "data/raw/primary/Osteoporosis Knee X-ray/normal/N7.JPEG",
            "actual": "normal",
            "type": "Correct_Normal"
        },
        {
            "path": "data/raw/secondary/OS Collected Data/Osteopenia/Osteopenia 100.jpg",
            "actual": "osteopenia",
            "type": "Correct_Osteopenia"
        },
        {
            "path": "data/raw/secondary/OS Collected Data/Osteoporosis/Osteoporosis 155.JPEG",
            "actual": "osteoporosis",
            "type": "Correct_Osteoporosis"
        },
        # Incorrectly Classified (Critical Polar Mistakes)
        {
            "path": "data/raw/secondary/OS Collected Data/Normal/Normal 114.jpg",
            "actual": "normal",
            "type": "Error_Normal_as_Osteoporosis",
            "pred_idx": 2 # osteoporosis
        },
        {
            "path": "data/raw/secondary/OS Collected Data/Osteoporosis/Osteoporosis 28.png",
            "actual": "osteoporosis",
            "type": "Error_Osteoporosis_as_Normal",
            "pred_idx": 0 # normal
        }
    ]
    
    for idx, img_info in enumerate(images_to_analyze, 1):
        img_path = PROJECT_ROOT / img_info["path"]
        if not img_path.exists():
            print(f"Skipping {img_info['type']} (File not found: {img_path})")
            continue
            
        print(f"\nProcessing image {idx}: {img_info['type']}")
        
        # Load and preprocess
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        # Generate CAM
        pred_idx = img_info.get("pred_idx", None)
        heatmap, final_pred_idx = grad_cam.generate_heatmap(input_tensor, class_idx=pred_idx)
        
        # Save overlay visualization
        overlay = overlay_heatmap(img, heatmap)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"Original Image\nActual: {img_info['actual']}")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nPredicted: {CLASS_NAMES[final_pred_idx]}")
        axes[2].axis("off")
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"{img_info['type']}_gradcam.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        print(f"Saved visualization to: {save_path}")
        
    grad_cam.remove_hooks()
    print("\nGrad-CAM Explainability Analysis completed successfully.")

if __name__ == "__main__":
    main()
