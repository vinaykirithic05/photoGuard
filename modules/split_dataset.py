"""
============================================================
PhotoGuard AI
Dataset Splitting & Class Balancing Module

Author : Vinay Kirithic

Description:
Splits preprocessed images into Train (70%), Validation (15%), and Test (15%) sets.
Supports optional class balancing (e.g., capping majority Real class at 12,000 to match AI class).
============================================================
"""

import random
import shutil
from pathlib import Path
from tqdm import tqdm

from modules.paths import (
    PREPROCESSED_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
)

# ==========================================================
# CONFIGURATION & RATIOS
# ==========================================================
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Set MAX_SAMPLES_PER_CLASS to an integer (e.g., 12000) to balance dataset (12k AI vs 12k Real)
# Set to None if using all available images without capping
MAX_SAMPLES_PER_CLASS = 12000  

RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class DatasetSplitter:
    def __init__(
        self,
        input_dir: Path = PREPROCESSED_DIR,
        max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
        seed: int = RANDOM_SEED
    ):
        self.input_dir = input_dir
        self.max_samples_per_class = max_samples_per_class
        self.seed = seed
        random.seed(self.seed)

    def clean_output_folders(self):
        """Cleans existing train/val/test directories for a fresh split."""
        for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
        print("Previous dataset splits cleared. Fresh directories initialized.")

    def get_class_folders(self):
        """Retrieves subdirectories containing valid image files."""
        folders = []
        if not self.input_dir.exists():
            return folders
            
        for folder in self.input_dir.rglob("*"):
            if folder.is_dir():
                images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
                if images:
                    folders.append(folder)
        return folders

    def copy_files(self, files, destination_dir):
        """Copies list of image files to target directory."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, destination_dir / file.name)

    def split(self):
        """Executes stratified train/validation/test split with balancing."""
        if not self.input_dir.exists():
            print(f"Preprocessed dataset directory not found: {self.input_dir}")
            return

        self.clean_output_folders()
        class_folders = self.get_class_folders()

        print("\n" + "=" * 60)
        print("PhotoGuard AI - Dataset Splitting & Class Balancing")
        print("=" * 60)

        total_train, total_val, total_test = 0, 0, 0

        for folder in class_folders:
            relative_folder = folder.relative_to(self.input_dir)
            images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

            random.shuffle(images)

            # Class balancing cap
            if self.max_samples_per_class and len(images) > self.max_samples_per_class:
                print(f"Balancing Class [{relative_folder}]: Capped from {len(images)} -> {self.max_samples_per_class} images")
                images = images[:self.max_samples_per_class]

            total = len(images)
            train_cnt = int(total * TRAIN_RATIO)
            val_cnt = int(total * VALIDATION_RATIO)

            train_imgs = images[:train_cnt]
            val_imgs = images[train_cnt : train_cnt + val_cnt]
            test_imgs = images[train_cnt + val_cnt :]

            self.copy_files(train_imgs, TRAIN_DIR / relative_folder)
            self.copy_files(val_imgs, VALIDATION_DIR / relative_folder)
            self.copy_files(test_imgs, TEST_DIR / relative_folder)

            total_train += len(train_imgs)
            total_val += len(val_imgs)
            total_test += len(test_imgs)

            print(f"  Folder [{relative_folder}] -> Total: {total} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

        print("\n" + "=" * 60)
        print("Dataset Split Completed Successfully!")
        print("-" * 60)
        print(f"Total Train Images : {total_train}")
        print(f"Total Val Images   : {total_val}")
        print(f"Total Test Images  : {total_test}")
        print("=" * 60 + "\n")

def main():
    splitter = DatasetSplitter()
    splitter.split()

if __name__ == "__main__":
    main()