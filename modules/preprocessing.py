from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import time

# ======================================================
# CONFIGURATION
# ======================================================

IMAGE_SIZE = (224, 224)

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "datasets" / "processed"
OUTPUT_DIR = BASE_DIR / "datasets" / "preprocessed"

METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(exist_ok=True)

METADATA_FILE = METADATA_DIR / "metadata.csv"
LOG_FILE = METADATA_DIR / "preprocessing_log.txt"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

# ======================================================
# GLOBAL VARIABLES
# ======================================================

metadata = []

processed_images = 0
corrupted_images = 0
skipped_images = 0

dataset_statistics = {}

# ======================================================
# FUNCTIONS
# ======================================================

def process_image(image_path: Path):

    global processed_images
    global corrupted_images

    try:

        with Image.open(image_path) as img:

            original_width, original_height = img.size

            img = img.convert("RGB")

            img = img.resize(IMAGE_SIZE)

            relative_path = image_path.relative_to(INPUT_DIR)

            save_path = OUTPUT_DIR / relative_path

            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_path = save_path.with_suffix(".jpg")

            img.save(save_path, quality=95)

            parts = relative_path.parts

            label = parts[0]

            dataset = parts[1] if len(parts) > 1 else "Unknown"

            metadata.append({

                "filename": save_path.name,

                "label": label,

                "dataset": dataset,

                "width": original_width,

                "height": original_height,

                "format": "JPEG",

                "path": str(save_path)

            })

            dataset_statistics[dataset] = dataset_statistics.get(dataset, 0) + 1

            processed_images += 1

    except (UnidentifiedImageError, OSError, Exception) as e:

        global skipped_images

        skipped_images += 1

        corrupted_images += 1

        with open(LOG_FILE, "a") as log:

            log.write(f"{image_path}\n")

            log.write(f"{str(e)}\n\n")


def preprocess_dataset():

    images = [

        file

        for file in INPUT_DIR.rglob("*")

        if file.suffix.lower() in SUPPORTED_EXTENSIONS

    ]

    print(f"\nFound {len(images)} images.\n")

    for image in tqdm(images, desc="Processing"):

        process_image(image)


def save_metadata():

    df = pd.DataFrame(metadata)

    df.to_csv(METADATA_FILE, index=False)


def print_summary(start_time):

    elapsed = time.time() - start_time

    print("\n")

    print("=" * 60)

    print("        PhotoGuard AI - Preprocessing Summary")

    print("=" * 60)

    print()

    print("Dataset Statistics")

    print("------------------")

    for dataset, count in dataset_statistics.items():

        print(f"{dataset:<20} {count}")

    print()

    print("-" * 60)

    print(f"Processed Images : {processed_images}")

    print(f"Skipped Images   : {skipped_images}")

    print(f"Corrupted Images : {corrupted_images}")

    print(f"Metadata File    : {METADATA_FILE}")

    print(f"Output Folder    : {OUTPUT_DIR}")

    print(f"Time Taken       : {elapsed:.2f} seconds")

    print("=" * 60)


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n")

    print("=" * 60)

    print("        PhotoGuard AI - Image Preprocessing")

    print("=" * 60)

    print()

    if not INPUT_DIR.exists():

        print("Processed dataset folder not found.")

        print(INPUT_DIR)

        return

    start = time.time()

    preprocess_dataset()

    save_metadata()

    print_summary(start)


if __name__ == "__main__":

    main()