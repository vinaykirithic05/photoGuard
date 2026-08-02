"""
=========================================================
PhotoGuard AI
CNN Dataset Loader

Author : Vinay
Description:
Loads Train / Validation / Test datasets
Applies Data Augmentation
Creates reusable DataLoaders

=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.cnn.config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_ROTATION,
)

# =========================================================
# VERIFY DATASET PATHS
# =========================================================

def verify_dataset():

    folders = [
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR
    ]

    for folder in folders:

        if not folder.exists():

            raise FileNotFoundError(
                f"\nDataset folder not found:\n{folder}"
            )

    print("✓ Dataset folders verified.\n")


# =========================================================
# TRAIN TRANSFORMS
# =========================================================

def get_train_transforms():

    transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            RANDOM_ROTATION
        ),

        transforms.ColorJitter(

            brightness=0.20,

            contrast=0.20,

            saturation=0.20,

            hue=0.05

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]

        )

    ])

    return transform


# =========================================================
# VALIDATION / TEST TRANSFORMS
# =========================================================

def get_test_transforms():

    transform = transforms.Compose([

        transforms.Resize(

            (IMAGE_SIZE, IMAGE_SIZE)

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]

        )

    ])

    return transform


# =========================================================
# LOAD DATASETS
# =========================================================

def get_datasets():

    verify_dataset()

    train_dataset = datasets.ImageFolder(

        root=TRAIN_DIR,

        transform=get_train_transforms()

    )

    validation_dataset = datasets.ImageFolder(

        root=VALIDATION_DIR,

        transform=get_test_transforms()

    )

    test_dataset = datasets.ImageFolder(

        root=TEST_DIR,

        transform=get_test_transforms()

    )

    return (

        train_dataset,

        validation_dataset,

        test_dataset

    )
# =========================================================
# CREATE DATALOADERS
# =========================================================

def get_dataloaders():

    train_dataset, validation_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, validation_loader, test_loader


# =========================================================
# DATASET INFORMATION
# =========================================================

def show_dataset_info():

    train_dataset, validation_dataset, test_dataset = get_datasets()

    print("\n" + "=" * 70)
    print("              PhotoGuard AI - Dataset Information")
    print("=" * 70)

    print(f"\nTrain Directory      : {TRAIN_DIR}")
    print(f"Validation Directory : {VALIDATION_DIR}")
    print(f"Test Directory       : {TEST_DIR}")

    print("\nClasses")
    print("-" * 70)

    for class_name, index in train_dataset.class_to_idx.items():
        print(f"{index} -> {class_name}")

    print("\nDataset Statistics")
    print("-" * 70)

    print(f"Training Images      : {len(train_dataset)}")
    print(f"Validation Images    : {len(validation_dataset)}")
    print(f"Testing Images       : {len(test_dataset)}")

    total_images = (
        len(train_dataset)
        + len(validation_dataset)
        + len(test_dataset)
    )

    print(f"Total Images         : {total_images}")

    print("\nConfiguration")
    print("-" * 70)

    print(f"Image Size           : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch Size           : {BATCH_SIZE}")
    print(f"Workers              : {NUM_WORKERS}")
    print(f"Pin Memory           : {PIN_MEMORY}")

    train_loader, _, _ = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("\nSample Batch")
    print("-" * 70)

    print(f"Image Tensor Shape   : {images.shape}")
    print(f"Label Tensor Shape   : {labels.shape}")

    print("\nFirst Batch Labels")
    print(labels.tolist())

    print("\nDataset Loaded Successfully")
    print("=" * 70)


# =========================================================
# TEST DATALOADER
# =========================================================

def test_dataloader():

    train_loader, validation_loader, test_loader = get_dataloaders()

    print("\nTesting DataLoader...")

    train_batch = len(train_loader)
    validation_batch = len(validation_loader)
    test_batch = len(test_loader)

    print(f"Train Batches        : {train_batch}")
    print(f"Validation Batches   : {validation_batch}")
    print(f"Test Batches         : {test_batch}")

    print("\nDataLoader Working Successfully")


# =========================================================
# MAIN
# =========================================================

def main():

    show_dataset_info()

    test_dataloader()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()