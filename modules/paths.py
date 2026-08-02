"""
============================================================
PhotoGuard AI
Project Path Definitions & Directory Setup

Author : Vinay Kirithic

Description:
Centralizes all directory paths across local workspace and Google Colab environments.
Supports fallback directory paths for Google Drive case-sensitivity.
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
# Possible Google Drive dataset paths (handling case sensitivity and capitalization)
COLAB_DATASET_PATHS = [
    Path("/content/drive/MyDrive/PhotoGuard/datasets"),
    Path("/content/drive/MyDrive/PhotoGuard/Datasets"),
    Path("/content/drive/MyDrive/photoguard/datasets"),
    Path("/content/drive/MyDrive/photoGuard/datasets"),
]

LOCAL_DATASET = BASE_DIR / "datasets"

# Automatically resolve path depending on environment (Colab vs Local)
DATASET_DIR = LOCAL_DATASET
for colab_path in COLAB_DATASET_PATHS:
    if colab_path.exists():
        DATASET_DIR = colab_path
        break

# Raw and preprocessed dataset folders
RAW_DIR = DATASET_DIR / "raw"
if not RAW_DIR.exists() and (DATASET_DIR / "Raw").exists():
    RAW_DIR = DATASET_DIR / "Raw"

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