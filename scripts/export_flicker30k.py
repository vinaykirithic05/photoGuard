from datasets import load_dataset
from pathlib import Path

dataset = load_dataset("AminDehnavi/flickr30k")

SAVE_DIR = Path("datasets/raw/Flickr30k")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(dataset["train"]):
    image = sample["image"]
    image.save(SAVE_DIR / f"flickr_{i:05d}.jpg")

print("Done!")