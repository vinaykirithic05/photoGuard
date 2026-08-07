"""
=========================================================
PhotoGuard AI
CNN Training Engine

Author : Vinay Kirithic

Description:
Main training script.

Responsibilities

1. Load Dataset
2. Build CNN
3. Configure Optimizer
4. Configure Scheduler
5. Resume Training
6. Save Best Model
7. Save Checkpoints

=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from modules.cnn.config import (
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)

from modules.cnn.utils import (
    set_seed,
    get_learning_rate
)


from modules.cnn.dataset import (
    get_dataloaders
)

from modules.cnn.model import (
    build_model
)

from modules.cnn.trainer import (
    train_one_epoch,
    validate_one_epoch
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Check if running on Google Colab
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/PhotoGuard")
if COLAB_DRIVE_ROOT.exists():
    WEIGHTS_DIR = COLAB_DRIVE_ROOT / "weights"
    CHECKPOINT_DIR = COLAB_DRIVE_ROOT / "checkpoints"
    HISTORY_DIR = COLAB_DRIVE_ROOT / "history"
else:
    WEIGHTS_DIR = BASE_DIR / "weights"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    HISTORY_DIR = BASE_DIR / "history"

BEST_MODEL = WEIGHTS_DIR / "best_model.pth"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last_checkpoint.pth"
HISTORY_FILE = HISTORY_DIR / "history.json"
# =========================================================
# CREATE PROJECT FOLDERS
# =========================================================

def create_directories():

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# CREATE MODEL
# =========================================================

def create_model():

    model = build_model()

    model = model.to(DEVICE)

    return model


# =========================================================
# LOSS FUNCTION
# =========================================================

def build_loss():

    return nn.CrossEntropyLoss()


# =========================================================
# OPTIMIZER
# =========================================================

def build_optimizer(model):

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    return optimizer


# =========================================================
# LR SCHEDULER
# =========================================================

def build_scheduler(optimizer):

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3

    )

    return scheduler


# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_checkpoint(model, optimizer):

    start_epoch = 1

    best_accuracy = 0.0

    if LAST_CHECKPOINT.exists():

        print("\nLoading Previous Checkpoint...\n")

        checkpoint = torch.load(

            LAST_CHECKPOINT,

            map_location=DEVICE

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        start_epoch = checkpoint["epoch"] + 1

        best_accuracy = checkpoint["best_accuracy"]

        print(

            f"Resuming From Epoch {start_epoch}"

        )

    return (

        model,

        optimizer,

        start_epoch,

        best_accuracy

    )
# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    create_directories()

    print("\n" + "=" * 70)
    print("             PhotoGuard AI - CNN Training")
    print("=" * 70)

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader, validation_loader, _ = get_dataloaders()

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = create_model()

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = build_loss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = build_optimizer(model)

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = build_scheduler(optimizer)

    # -----------------------------------------------------
    # Resume Checkpoint
    # -----------------------------------------------------

    model, optimizer, start_epoch, best_accuracy = load_checkpoint(

        model,
        optimizer

    )

    # -----------------------------------------------------
    # Training History
    # -----------------------------------------------------

    history = {

        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []

    }

    patience = 0

    EARLY_STOPPING = EARLY_STOPPING_PATIENCE

    # -----------------------------------------------------
    # Epoch Loop
    # -----------------------------------------------------

    for epoch in range(start_epoch, EPOCHS + 1):

        print("\n" + "=" * 70)

        print(f"Epoch {epoch}/{EPOCHS}")

        print("=" * 70)

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(

            model,
            train_loader,
            criterion,
            optimizer

        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        validation_loss, validation_accuracy = validate_one_epoch(

            model,
            validation_loader,
            criterion

        )

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        scheduler.step(

            validation_accuracy

        )

        # -------------------------------------------------
        # Save History
        # -------------------------------------------------

        history["train_loss"].append(

            train_loss

        )

        history["train_accuracy"].append(

            train_accuracy

        )

        history["validation_loss"].append(

            validation_loss

        )

        history["validation_accuracy"].append(

            validation_accuracy

        )

        # -------------------------------------------------
        # Print Results
        # -------------------------------------------------

        print()

        print(f"Train Loss        : {train_loss:.4f}")

        print(f"Train Accuracy    : {train_accuracy*100:.2f}%")

        print(f"Validation Loss   : {validation_loss:.4f}")

        print(f"Validation Acc    : {validation_accuracy*100:.2f}%")

        print(

            f"Learning Rate     : "

            f"{get_learning_rate(optimizer):.7f}"

        )

        # -------------------------------------------------
        # Save Best Model
        # -------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            torch.save(

                model.state_dict(),

                BEST_MODEL

            )

            print("\n✓ Best Model Saved")

            patience = 0

        else:

            patience += 1

        # -------------------------------------------------
        # Save Checkpoint
        # -------------------------------------------------

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "best_accuracy": best_accuracy

        }

        torch.save(

            checkpoint,

            LAST_CHECKPOINT

        )

        # -------------------------------------------------
        # Early Stopping
        # -------------------------------------------------

        if patience >= EARLY_STOPPING:

            print("\nEarly Stopping Triggered.")

            break

    # -----------------------------------------------------
    # Save History
    # -----------------------------------------------------

    with open(

        HISTORY_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            history,

            file,

            indent=4

        )

    print("\n" + "=" * 70)

    print("Training Completed Successfully")

    print(f"Best Validation Accuracy : {best_accuracy*100:.2f}%")

    print("=" * 70)

    # =========================================================
# PRINT TRAINING SUMMARY
# =========================================================

def print_training_summary():

    print("\n")
    print("=" * 70)
    print("              PhotoGuard AI - Training Summary")
    print("=" * 70)

    print(f"\nBest Model")

    print(f"{BEST_MODEL}")

    print(f"\nCheckpoint")

    print(f"{LAST_CHECKPOINT}")

    print(f"\nTraining History")

    print(f"{HISTORY_FILE}")

    print("\nModel Training Finished Successfully.")

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n")
    print("=" * 70)
    print("            PhotoGuard AI - CNN Training Engine")
    print("=" * 70)

    print(f"\nDevice : {DEVICE}")

    set_seed()

    train_model()

    print_training_summary()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()