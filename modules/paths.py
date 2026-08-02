"""
============================================================
PhotoGuard AI
Project Path Definitions & Directory Setup

Author : Vinay Kirithic

Description:
Centralizes all directory paths across local workspace and Google Colab environments.
Automatically creates required directory trees if they do not exist.
============================================================
"""

from pathlib import Path

# ==========================================================
# BASE DIRECTORY
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================================
# ENVIRONMENT & DATASET DIRECTORIES
# ==========================================================
COLAB_DATASET = Path("/content/drive/MyDrive/PhotoGuard/datasets")
LOCAL_DATASET = BASE_DIR / "datasets"

# Automatically resolve path depending on environment (Colab vs Local)
DATASET_DIR = COLAB_DATASET if COLAB_DATASET.exists() else LOCAL_DATASET

# Raw and preprocessed dataset folders
RAW_DIR = DATASET_DIR / "raw"
PREPROCESSED_DIR = DATASET_DIR / "preprocessed"
BALANCED_DIR = DATASET_DIR / "balanced"

# Train / Validation / Test dataset splits
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# ==========================================================
# PROJECT METADATA, MODELS, & OUTPUT DIRECTORIES
# ==========================================================
METADATA_DIR = BASE_DIR / "metadata"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Sub-directories inside outputs
PREDICTION_DIR = OUTPUT_DIR / "predictions"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
LOG_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"

# ==========================================================
# AUTOMATIC DIRECTORY CREATION
# ==========================================================
def ensure_directories_exist():
    """Ensures all essential project directories exist."""
    directories = [
        RAW_DIR,
        PREPROCESSED_DIR,
        BALANCED_DIR,
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR,
        METADATA_DIR,
        MODELS_DIR,
        OUTPUT_DIR,
        PREDICTION_DIR,
        GRADCAM_DIR,
        LOG_DIR,
        METRICS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Run directory check on import
ensure_directories_exist()

if __name__ == "__main__":
    print("=" * 60)
    print("PhotoGuard AI - Path Configuration")
    print("=" * 60)
    print(f"Base Directory       : {BASE_DIR}")
    print(f"Dataset Directory    : {DATASET_DIR}")
    print(f"Raw Images Directory : {RAW_DIR}")
    print(f"Preprocessed Dir     : {PREPROCESSED_DIR}")
    print(f"Metadata Directory   : {METADATA_DIR}")
    print(f"Models Directory     : {MODELS_DIR}")
    print(f"Outputs Directory    : {OUTPUT_DIR}")
    print("=" * 60)
    print("All project directories verified successfully!")