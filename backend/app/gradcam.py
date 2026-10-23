import io
import base64
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM implementation matching src/gradcam_final.py.
    Computes class activation maps for a target convolutional layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
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

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, torch.Tensor]:
        """
        Generate Grad-CAM heatmap.
        Note: Model forward and backward pass must NOT be under torch.no_grad().
        Returns:
            heatmap (H x W numpy array in [0, 1])
            class_idx (the index for which Grad-CAM was computed)
            output (raw logits tensor from model forward pass)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
            
        score = output[0, class_idx]
        score.backward()
        
        # Global average pool the gradients across spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1)
        
        # Apply ReLU to keep only positive influence features
        cam = torch.clamp(cam, min=0)
        
        # Min-max normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.cpu().numpy()[0], class_idx, output

    def remove_hooks(self):
        if self.forward_hook:
            self.forward_hook.remove()
        if self.backward_hook:
            self.backward_hook.remove()


def overlay_heatmap(original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on the original image with alpha blending.
    Exact implementation from src/gradcam_final.py.
    """
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        original_img.size, Image.Resampling.BILINEAR
    )
    heatmap_np = np.array(heatmap_resized) / 255.0
    
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_np)[:, :, :3]
    
    original_np = np.array(original_img) / 255.0
    if len(original_np.shape) == 2:
        original_np = np.stack([original_np] * 3, axis=-1)
        
    overlay = (1 - alpha) * original_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    
    return overlay


def overlay_to_base64_png(overlay: np.ndarray) -> str:
    """Convert float [0.0, 1.0] overlay numpy array to base64 PNG string."""
    overlay_uint8 = (overlay * 255.0).astype(np.uint8)
    img = Image.fromarray(overlay_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
