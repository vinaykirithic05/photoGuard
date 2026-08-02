"""
============================================================
PhotoGuard AI
Project Path Definitions & Directory Setup

Author : Vinay Kirithic

Description:
Centralizes all directory paths across local workspace and Google Colab environments.
Dynamically resolves paths by checking actual filesystem contents.
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
COLAB_DRIVE_BASE = Path("/content/drive/MyDrive")
LOCAL_DATASET = BASE_DIR / "datasets"

DATASET_DIR = LOCAL_DATASET

# Search for PhotoGuard dataset folder dynamically inside Google Drive
if COLAB_DRIVE_BASE.exists():
    for item in COLAB_DRIVE_BASE.iterdir():
        if item.is_dir() and item.name.lower() == "photoguard":
            datasets_candidate = item / "datasets"
            if not datasets_candidate.exists():
                datasets_candidate = item / "Datasets"
            if datasets_candidate.exists():
                DATASET_DIR = datasets_candidate
                break
            elif item.exists():
                # Fallback to project root if datasets subfolder isn't explicit
                DATASET_DIR = item

# Resolve raw directory dynamically
RAW_DIR = DATASET_DIR / "raw"
if not RAW_DIR.exists():
    for child in DATASET_DIR.iterdir() if DATASET_DIR.exists() else []:
        if child.is_dir() and child.name.lower() == "raw":
            RAW_DIR = child
            break

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