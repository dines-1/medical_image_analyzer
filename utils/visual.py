import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def plot_training_history(
    history,
    title: str = "Training History",
    save_path: str = None,
) -> str:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # -- Accuracy --
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # -- Loss --
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = f"training_history_{int(time.time())}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Helper] Training plot saved to: {save_path}")
    return str(Path(save_path).resolve())


def plot_model_comparison(
    histories: Dict[str, object],
    title: str = "Model Comparison",
    save_path: str = None,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for name, history in histories.items():
        axes[0].plot(history.history["val_accuracy"], label=name, linewidth=2)
        axes[1].plot(history.history["val_loss"], label=name, linewidth=2)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = f"model_comparison_{int(time.time())}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Helper] Comparison plot saved to: {save_path}")
    return str(Path(save_path).resolve())


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> str:

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = f"confusion_matrix_{int(time.time())}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Helper] Confusion matrix saved to: {save_path}")
    return str(Path(save_path).resolve())


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> None:

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)



class Timer:
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        minutes, seconds = divmod(self.elapsed, 60)
        print(f"[Timer] Training completed in {int(minutes)}m {seconds:.1f}s")