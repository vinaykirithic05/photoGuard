"""
=========================================================
PhotoGuard AI
CNN Configuration File

Author : Vinay
Model  : EfficientNet-B0
=========================================================
"""

from pathlib import Path
import torch

# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# =========================================================
# DATASET PATHS
# =========================================================

DATASET_DIR = BASE_DIR / "datasets"

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

TEST_DIR = DATASET_DIR / "test"

# =========================================================
# MODEL PATHS
# =========================================================

MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "efficientnet_b0"

BEST_MODEL = MODELS_DIR / "best_model.pth"

LAST_MODEL = MODELS_DIR / "last_model.pth"

# =========================================================
# OUTPUT PATHS
# =========================================================

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

PREDICTION_DIR = OUTPUT_DIR / "predictions"

GRADCAM_DIR = OUTPUT_DIR / "gradcam"

LOG_DIR = OUTPUT_DIR / "logs"

METRICS_DIR = OUTPUT_DIR / "metrics"

PREDICTION_DIR.mkdir(exist_ok=True)

GRADCAM_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)

METRICS_DIR.mkdir(exist_ok=True)

# =========================================================
# IMAGE SETTINGS
# =========================================================

IMAGE_SIZE = 224

CHANNELS = 3

NUM_CLASSES = 2

CLASS_NAMES = [

    "AI",

    "Real"

]

# =========================================================
# TRAINING SETTINGS
# =========================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PIN_MEMORY = True

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

# =========================================================
# RANDOM SEED
# =========================================================

SEED = 42

# =========================================================
# EARLY STOPPING
# =========================================================

PATIENCE = 5

# =========================================================
# MODEL
# =========================================================

PRETRAINED = True

DROPOUT = 0.30

# =========================================================
# DATA AUGMENTATION
# =========================================================

HORIZONTAL_FLIP = True

RANDOM_ROTATION = 10

COLOR_JITTER = True

# =========================================================
# PRINT CONFIG
# =========================================================

def show_config():

    print("=" * 60)

    print("PhotoGuard AI")

    print("=" * 60)

    print()

    print("Dataset")

    print(TRAIN_DIR)

    print()

    print("Validation")

    print(VALIDATION_DIR)

    print()

    print("Test")

    print(TEST_DIR)

    print()

    print(f"Image Size      : {IMAGE_SIZE}")

    print(f"Batch Size      : {BATCH_SIZE}")

    print(f"Epochs          : {EPOCHS}")

    print(f"Learning Rate   : {LEARNING_RATE}")

    print(f"Device          : {DEVICE}")

    print("=" * 60)


if __name__ == "__main__":

    show_config()