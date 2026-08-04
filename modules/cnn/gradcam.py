"""
=========================================================
PhotoGuard AI
Grad-CAM Visual Explainability Module

Author : Vinay Kirithic

Description:
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
for PhotoGuard CNN predictions, visualizing high-impact pixel regions.
=========================================================
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from modules.cnn.config import DEVICE, IMAGE_SIZE, GRADCAM_DIR
from modules.cnn.model import build_model
from modules.paths import MODELS_DIR

BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
if not BEST_MODEL_PATH.exists():
    BEST_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "weights" / "best_model.pth"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        return cam


def generate_gradcam_overlay(image_path: Path, output_filename: str = "gradcam_output.jpg"):
    image_path = Path(image_path)
    model = build_model()
    if BEST_MODEL_PATH.exists():
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Get last convolutional layer of EfficientNet backbone
    target_layer = model.backbone.features[-1]
    gradcam = GradCAM(model, target_layer)

    orig_img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(orig_img).unsqueeze(0).to(DEVICE)

    heatmap = gradcam.generate_heatmap(tensor)

    # Overlay heatmap on original image
    np_img = np.array(orig_img)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np_img, 0.6, heatmap_colored, 0.4, 0)

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    save_path = GRADCAM_DIR / output_filename
    Image.fromarray(overlay).save(save_path)
    print(f"Grad-CAM overlay saved to: {save_path}")
    return str(save_path)


if __name__ == "__main__":
    print("Grad-CAM Module initialized.")
