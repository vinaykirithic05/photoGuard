"""
=========================================================
PhotoGuard AI
CNN Training Script

Author : Vinay

Description:
Main training script for EfficientNet-B0.

=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.cnn.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL,
    LAST_MODEL,
)

from modules.cnn.dataset import get_dataloaders

from modules.cnn.model import build_model

from modules.cnn.trainer import (
    train_one_epoch,
    validate_one_epoch,
)

from modules.cnn.utils import (
    set_seed,
    freeze_backbone,
    unfreeze_backbone,
    save_model,
    print_model_info,
)


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train():

    print("=" * 70)
    print("PhotoGuard AI")
    print("CNN Training")
    print("=" * 70)

    # ------------------------------------
    # Random Seed
    # ------------------------------------

    set_seed()

    # ------------------------------------
    # Dataset
    # ------------------------------------

    print("\nLoading Dataset...\n")

    train_loader, validation_loader, _ = get_dataloaders()

    print("Dataset Loaded Successfully")

    # ------------------------------------
    # Model
    # ------------------------------------

    print("\nBuilding Model...\n")

    model = build_model()

    print_model_info(model)

    # ------------------------------------
    # Freeze Backbone
    # ------------------------------------

    freeze_backbone(model)

    # ------------------------------------
    # Loss Function
    # ------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------
    # Optimizer
    # ------------------------------------

    optimizer = optim.Adam(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    # ------------------------------------
    # Scheduler
    # ------------------------------------

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2

    )

    # ------------------------------------
    # Variables
    # ------------------------------------

    best_accuracy = 0.0

    start_time = time.time()

    # ------------------------------------
    # Epoch Loop
    # ------------------------------------

    for epoch in range(EPOCHS):

        print("\n")

        print("=" * 70)

        print(f"Epoch {epoch + 1}/{EPOCHS}")

        print("=" * 70)

        # --------------------------------
        # Training
        # --------------------------------

        train_loss, train_accuracy = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer

        )

        # --------------------------------
        # Validation
        # --------------------------------

        validation_loss, validation_accuracy = validate_one_epoch(

            model,

            validation_loader,

            criterion

        )
                # --------------------------------
        # Update Scheduler
        # --------------------------------

        scheduler.step(validation_accuracy)

        # --------------------------------
        # Print Metrics
        # --------------------------------

        print("\nTraining Results")

        print(f"Train Loss         : {train_loss:.4f}")
        print(f"Train Accuracy     : {train_accuracy*100:.2f}%")

        print()

        print("Validation Results")

        print(f"Validation Loss    : {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy*100:.2f}%")

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning Rate      : {current_lr:.6f}")

        # --------------------------------
        # Save Best Model
        # --------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                accuracy=validation_accuracy,
                filepath=BEST_MODEL
            )

            print("\n✅ Best Model Updated")

        # --------------------------------
        # Save Last Model
        # --------------------------------

        save_model(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            accuracy=validation_accuracy,
            filepath=LAST_MODEL
        )

        # --------------------------------
        # Fine Tuning
        # --------------------------------

        if epoch == 4:

            print("\n")
            print("=" * 70)
            print("Fine Tuning Started")
            print("=" * 70)

            unfreeze_backbone(model)

            optimizer = optim.Adam(

                model.parameters(),

                lr=LEARNING_RATE / 10,

                weight_decay=WEIGHT_DECAY

            )

            scheduler = ReduceLROnPlateau(

                optimizer,

                mode="max",

                factor=0.5,

                patience=2

            )

    # =====================================================
    # Training Completed
    # =====================================================

    end_time = time.time()

    total_time = end_time - start_time

    hours = int(total_time // 3600)

    minutes = int((total_time % 3600) // 60)

    seconds = int(total_time % 60)

    print("\n")
    print("=" * 70)

    print("Training Completed Successfully")

    print("=" * 70)

    print(f"Best Validation Accuracy : {best_accuracy*100:.2f}%")

    print(
        f"Training Time            : "
        f"{hours}h {minutes}m {seconds}s"
    )

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

    train()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()