"""
=========================================================
PhotoGuard AI
CNN Model Evaluation Module

Author : Vinay Kirithic

Description:
Evaluates trained PhotoGuard CNN model on the test dataset.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix,
and saves metrics summary to output directory.
=========================================================
"""

import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

from modules.cnn.config import (
    DEVICE,
    CLASS_NAMES,
    METRICS_DIR,
)
from modules.cnn.model import build_model
from modules.paths import MODELS_DIR, TEST_DIR
from modules.dataloader import create_dataloaders

# Target Model Weights
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
if not BEST_MODEL_PATH.exists():
    BEST_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "weights" / "best_model.pth"


def evaluate_model(model_path: Path = BEST_MODEL_PATH):
    print("\n" + "=" * 70)
    print("           PhotoGuard AI - Model Evaluation Engine")
    print("=" * 70)

    if not model_path.exists():
        print(f"Error: Model weights file not found at {model_path}")
        return

    # Load DataLoaders
    _, _, test_loader, class_map = create_dataloaders()
    print(f"Test Batches: {len(test_loader)} | Class Map: {class_map}")

    # Load Trained Model
    model = build_model()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    print("\nEvaluating on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for Class 'Real' or positive class

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("-" * 70)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc:.4f}")
    print("=" * 70)

    # Save metrics JSON
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "test_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("PhotoGuard AI - Confusion Matrix")
    plt.tight_layout()
    cm_path = METRICS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix plot saved to: {cm_path}\n")

if __name__ == "__main__":
    evaluate_model()
