"""
============================================================
PhotoGuard AI
Dataset Split Module

Author : Vinay Kirithic

Description:
Splits the preprocessed dataset into

1. Train
2. Validation
3. Test

Split Ratio

Train       : 70%
Validation  : 15%
Test        : 15%

============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import random
import shutil

from pathlib import Path

from tqdm import tqdm


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "datasets" / "preprocessed"

TRAIN_DIR = BASE_DIR / "datasets" / "train"

VALIDATION_DIR = BASE_DIR / "datasets" / "validation"

TEST_DIR = BASE_DIR / "datasets" / "test"


SUPPORTED_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"

}

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

def create_output_folders():

    # Remove old folders

    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)

    if VALIDATION_DIR.exists():
        shutil.rmtree(VALIDATION_DIR)

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    # Create fresh folders

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nOld dataset split removed.")

    print("New folders created successfully.\n")


# ==========================================================
# GET ALL CLASS FOLDERS
# ==========================================================

def get_class_folders():

    folders = []

    for folder in INPUT_DIR.rglob("*"):

        if folder.is_dir():

            images = [

                file

                for file in folder.iterdir()

                if file.suffix.lower() in SUPPORTED_EXTENSIONS

            ]

            if len(images) > 0:

                folders.append(folder)

    return folders


# ==========================================================
# COPY IMAGES
# ==========================================================

def copy_images(images, destination):

    destination.mkdir(

        parents=True,

        exist_ok=True

    )

    for image in images:

        shutil.copy2(

            image,

            destination / image.name

        )
# ==========================================================
# SPLIT DATASET
# ==========================================================

def split_dataset():

    create_output_folders()

    folders = get_class_folders()

    print("\n")
    print("=" * 60)
    print("PhotoGuard AI - Dataset Splitting")
    print("=" * 60)

    total_train = 0
    total_validation = 0
    total_test = 0

    for folder in folders:

        relative_folder = folder.relative_to(INPUT_DIR)

        images = [

            file

            for file in folder.iterdir()

            if file.suffix.lower() in SUPPORTED_EXTENSIONS

        ]

        random.shuffle(images)

        total_images = len(images)

        train_count = int(total_images * TRAIN_RATIO)

        validation_count = int(total_images * VALIDATION_RATIO)

        train_images = images[:train_count]

        validation_images = images[
            train_count:
            train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]

        copy_images(

            train_images,

            TRAIN_DIR / relative_folder

        )

        copy_images(

            validation_images,

            VALIDATION_DIR / relative_folder

        )

        copy_images(

            test_images,

            TEST_DIR / relative_folder

        )

        total_train += len(train_images)

        total_validation += len(validation_images)

        total_test += len(test_images)

        print()

        print(f"{relative_folder}")

        print(f"Total Images : {total_images}")

        print(f"Train        : {len(train_images)}")

        print(f"Validation   : {len(validation_images)}")

        print(f"Test         : {len(test_images)}")

    print("\n")
    print("=" * 60)

    print("Overall Summary")

    print("=" * 60)

    print(f"Train Images      : {total_train}")

    print(f"Validation Images : {total_validation}")

    print(f"Test Images       : {total_test}")

    print()

    print("Dataset Split Completed Successfully.")

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

def main():

    if not INPUT_DIR.exists():

        print()

        print("Preprocessed dataset not found.")

        print(INPUT_DIR)

        return

    split_dataset()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()