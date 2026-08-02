"""
============================================================
PhotoGuard AI
Image Preprocessing Module

Author : Vinay Kirithic

Description:
This module scans the RAW datasets and performs:

1. Image Validation
2. Corrupted Image Removal
3. RGB Conversion
4. Resize to CNN Input Size
5. JPEG Conversion
6. Metadata Generation
7. Logging

Output:
datasets/preprocessed/

============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import time

from pathlib import Path

import pandas as pd

from PIL import (
    Image,
    UnidentifiedImageError
)

from tqdm import tqdm


# ==========================================================
# CONFIGURATION
# ==========================================================

IMAGE_SIZE = (224, 224)

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
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "datasets" / "raw"

OUTPUT_DIR = BASE_DIR / "datasets" / "preprocessed"

METADATA_DIR = BASE_DIR / "metadata"

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)

METADATA_DIR.mkdir(

    parents=True,

    exist_ok=True

)

METADATA_FILE = METADATA_DIR / "metadata.csv"

LOG_FILE = METADATA_DIR / "preprocessing_log.txt"


# ==========================================================
# GLOBAL VARIABLES
# ==========================================================

metadata = []

processed_images = 0

corrupted_images = 0

skipped_images = 0

dataset_statistics = {}


# ==========================================================
# CREATE OUTPUT STRUCTURE
# ==========================================================

def create_output_structure():

    """
    Creates the required output folders.
    """

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    print()

    print("=" * 60)

    print("Output Folder Ready")

    print("=" * 60)


# ==========================================================
# GET ALL IMAGES
# ==========================================================

def get_all_images():

    images = [

        file

        for file in RAW_DIR.rglob("*")

        if file.suffix.lower() in SUPPORTED_EXTENSIONS

    ]

    return images
# ==========================================================
# PROCESS SINGLE IMAGE
# ==========================================================

def process_image(image_path: Path):

    global processed_images
    global corrupted_images
    global skipped_images

    try:

        with Image.open(image_path) as img:

            # ---------------------------------------------
            # Original Image Information
            # ---------------------------------------------

            original_width, original_height = img.size

            original_format = img.format

            # ---------------------------------------------
            # Convert to RGB
            # ---------------------------------------------

            img = img.convert("RGB")

            # ---------------------------------------------
            # Resize Image
            # ---------------------------------------------

            img = img.resize(

                IMAGE_SIZE,

                Image.Resampling.LANCZOS

            )

            # ---------------------------------------------
            # Preserve Folder Structure
            # ---------------------------------------------

            relative_path = image_path.relative_to(RAW_DIR)

            save_path = OUTPUT_DIR / relative_path

            save_path.parent.mkdir(

                parents=True,

                exist_ok=True

            )

            save_path = save_path.with_suffix(".jpg")

            # ---------------------------------------------
            # Save Image
            # ---------------------------------------------

            img.save(

                save_path,

                "JPEG",

                quality=95,

                optimize=True

            )

            # ---------------------------------------------
            # Folder Information
            # ---------------------------------------------

            parts = relative_path.parts

            label = parts[0]

            dataset = parts[1] if len(parts) > 1 else "Unknown"

            filename = image_path.name

            # ---------------------------------------------
            # Metadata
            # ---------------------------------------------

            metadata.append({

                "filename": filename,

                "label": label,

                "dataset": dataset,

                "width": original_width,

                "height": original_height,

                "original_format": original_format,

                "output_format": "JPEG",

                "output_width": IMAGE_SIZE[0],

                "output_height": IMAGE_SIZE[1],

                "path": str(save_path)

            })

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            dataset_statistics[dataset] = (

                dataset_statistics.get(dataset, 0) + 1

            )

            processed_images += 1

    except (

        UnidentifiedImageError,

        OSError,

        Exception

    ) as error:

        corrupted_images += 1

        skipped_images += 1

        with open(

            LOG_FILE,

            "a",

            encoding="utf-8"

        ) as log:

            log.write(

                f"{image_path}\n"

            )

            log.write(

                f"{str(error)}\n\n"

            )


# ==========================================================
# PREPROCESS DATASET
# ==========================================================

def preprocess_dataset():

    images = get_all_images()

    print()

    print("=" * 60)

    print(f"Found {len(images)} Images")

    print("=" * 60)

    if len(images) == 0:

        print("\nNo Images Found.\n")

        return

    for image in tqdm(

        images,

        desc="Preprocessing",

        colour="green"

    ):

        process_image(image)


# ==========================================================
# SAVE METADATA
# ==========================================================

def save_metadata():

    dataframe = pd.DataFrame(metadata)

    dataframe.to_csv(

        METADATA_FILE,

        index=False

    )
# ==========================================================
# PRINT SUMMARY
# ==========================================================

def print_summary(start_time):

    elapsed_time = time.time() - start_time

    print("\n")
    print("=" * 60)
    print("        PhotoGuard AI - Preprocessing Summary")
    print("=" * 60)

    print("\nDataset Statistics")
    print("-" * 60)

    for dataset, count in dataset_statistics.items():
        print(f"{dataset:<20} : {count}")

    print("\nOverall Statistics")
    print("-" * 60)
    print(f"Processed Images : {processed_images}")
    print(f"Skipped Images   : {skipped_images}")
    print(f"Corrupted Images : {corrupted_images}")

    print("\nOutput Information")
    print("-" * 60)
    print(f"Output Folder : {OUTPUT_DIR}")
    print(f"Metadata File : {METADATA_FILE}")
    print(f"Log File      : {LOG_FILE}")

    print(f"\nTotal Time : {elapsed_time:.2f} seconds")

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("          PhotoGuard AI - Image Preprocessing")
    print("=" * 60)

    if not RAW_DIR.exists():
        print("\nERROR : Raw dataset folder not found.\n")
        print(RAW_DIR)
        return

    start_time = time.time()

    create_output_structure()

    preprocess_dataset()

    save_metadata()

    print_summary(start_time)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()