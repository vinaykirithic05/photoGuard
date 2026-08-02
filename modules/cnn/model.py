"""
=========================================================
PhotoGuard AI
CNN Model Architecture Module

Author : Vinay Kirithic

Description:
Builds CNN models (EfficientNet-B0 default, ResNet50, or
ConvNeXt-Tiny options) for AI Image vs Real Image Detection.
Features flexible feature head unfreezing and custom classification head.
=========================================================
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

from modules.cnn.config import (
    DEVICE,
    NUM_CLASSES,
    DROPOUT,
    MODEL_NAME,
)

# =========================================================
# CNN MODEL CLASS
# =========================================================

class PhotoGuardCNN(nn.Module):

    def __init__(self, model_name: str = MODEL_NAME, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.model_name = model_name.lower()

        if "efficientnet" in self.model_name:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features

            if freeze_backbone:
                for param in self.backbone.features.parameters():
                    param.requires_grad = False

            # Custom Classification Head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=DROPOUT),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=DROPOUT / 2),
                nn.Linear(256, NUM_CLASSES)
            )

        elif "resnet" in self.model_name:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            in_features = self.backbone.fc.in_features

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=DROPOUT),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=DROPOUT / 2),
                nn.Linear(256, NUM_CLASSES)
            )

        elif "convnext" in self.model_name:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[2].in_features

            if freeze_backbone:
                for param in self.backbone.features.parameters():
                    param.requires_grad = False

            self.backbone.classifier[2] = nn.Sequential(
                nn.Dropout(p=DROPOUT),
                nn.Linear(in_features, NUM_CLASSES)
            )

        else:
            raise ValueError(f"Unsupported backbone model: {model_name}")

    def forward(self, x):
        return self.backbone(x)


# =========================================================
# BUILD MODEL HELPER
# =========================================================

def build_model(model_name: str = MODEL_NAME, freeze_backbone: bool = False):
    model = PhotoGuardCNN(model_name=model_name, freeze_backbone=freeze_backbone)
    model = model.to(DEVICE)
    return model


# =========================================================
# MODEL SUMMARY
# =========================================================

def model_summary(model):
    print("\n" + "=" * 70)
    print(f"PhotoGuard AI - CNN Architecture ({model.model_name.upper()})")
    print("=" * 70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")
    print(f"Device Target         : {DEVICE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    model = build_model()
    model_summary(model)