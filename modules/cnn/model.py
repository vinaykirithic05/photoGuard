"""
=========================================================
PhotoGuard AI
CNN Model

Author : Vinay

Description:
Builds EfficientNet-B0 model for
AI Image Detection

=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from modules.cnn.config import (
    DEVICE,
    NUM_CLASSES,
    DROPOUT,
)

# =========================================================
# CNN MODEL
# =========================================================

class PhotoGuardCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------
        # Load Pretrained EfficientNet-B0
        # --------------------------------------

        self.model = efficientnet_b0(

            weights=EfficientNet_B0_Weights.DEFAULT

        )

        # --------------------------------------
        # Get Input Features
        # --------------------------------------

        in_features = self.model.classifier[1].in_features

        # --------------------------------------
        # Replace Classifier
        # --------------------------------------

        self.model.classifier = nn.Sequential(

            nn.Dropout(

                p=DROPOUT

            ),

            nn.Linear(

                in_features,

                NUM_CLASSES

            )

        )

    # ------------------------------------------
    # Forward Pass
    # ------------------------------------------

    def forward(self, x):

        return self.model(x)


# =========================================================
# BUILD MODEL
# =========================================================

def build_model():

    model = PhotoGuardCNN()

    model = model.to(DEVICE)

    return model


# =========================================================
# MODEL INFORMATION
# =========================================================

def model_summary(model):

    print("\n")

    print("=" * 70)

    print("PhotoGuard AI - CNN Model")

    print("=" * 70)

    print()

    print(model)

    print()

    total_params = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print()

    print("=" * 70)

    print(f"Total Parameters      : {total_params:,}")

    print(f"Trainable Parameters  : {trainable_params:,}")

    print(f"Device                : {DEVICE}")

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

    model = build_model()

    model_summary(model)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()