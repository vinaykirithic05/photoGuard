from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import shutil
import random

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "datasets" / "preprocessed"

TRAIN_DIR = BASE_DIR / "datasets" / "train"
VAL_DIR = BASE_DIR / "datasets" / "validation"
TEST_DIR = BASE_DIR / "datasets" / "test"

METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(exist_ok=True)

SUMMARY_FILE = METADATA_DIR / "split_summary.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

RANDOM_STATE = 42

summary = []

# =====================================================
# COPY FUNCTION
# =====================================================

def copy_images(image_list, destination_root, label):

    destination = destination_root / label
    destination.mkdir(parents=True, exist_ok=True)

    for image in tqdm(image_list, desc=f"Copying {label} -> {destination_root.name}"):

        shutil.copy2(
            image,
            destination / image.name
        )

# =====================================================
# PROCESS LABEL
# =====================================================

def process_label(label):

    folder = INPUT_DIR / label

    images = []

    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))

    random.shuffle(images)

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    copy_images(train_imgs, TRAIN_DIR, label)
    copy_images(val_imgs, VAL_DIR, label)
    copy_images(test_imgs, TEST_DIR, label)

    summary.append({

        "label": label,

        "total": len(images),

        "train": len(train_imgs),

        "validation": len(val_imgs),

        "test": len(test_imgs)

    })

# =====================================================
# MAIN
# =====================================================

def main():

    print("\n")
    print("="*60)
    print("      PhotoGuard AI - Dataset Split")
    print("="*60)

    if not INPUT_DIR.exists():

        print("Preprocessed dataset not found.")
        return

    process_label("real")
    process_label("ai")

    df = pd.DataFrame(summary)

    df.to_csv(SUMMARY_FILE, index=False)

    print("\n")
    print("="*60)
    print("Dataset Split Completed")
    print("="*60)

    print(df)

    print("\nSummary saved to")

    print(SUMMARY_FILE)

if __name__ == "__main__":
    main()