from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "datasets" / "train"
VAL_DIR = BASE_DIR / "datasets" / "validation"
TEST_DIR = BASE_DIR / "datasets" / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

# ============================================
# TRANSFORMS
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================
# DATASETS
# ============================================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)

# ============================================
# DATALOADERS
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("PhotoGuard AI - DataLoader")
    print("=" * 50)

    print("\nClasses:")
    print(train_dataset.classes)

    print("\nClass Mapping:")
    print(train_dataset.class_to_idx)

    print("\nTraining Images :", len(train_dataset))
    print("Validation Images :", len(validation_dataset))
    print("Testing Images :", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("\nBatch Shape :", images.shape)
    print("Labels Shape:", labels.shape)

    print("\nDataLoader Ready!")