import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import NAMES


def coco_instance_counts(coco_path):
    data = json.loads(Path(coco_path).read_text())
    names = {c["id"]: c["name"].strip() for c in data["categories"]}
    counts = {name: 0 for name in names.values()}
    for ann in data["annotations"]:
        counts[names[ann["category_id"]]] += 1
    return counts


def plot_instance_counts(coco_path, out_path):
    counts = coco_instance_counts(coco_path)
    labels, values = list(counts), list(counts.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#D55E00"][:len(labels)])
    ax.bar_label(bars, padding=3)
    ax.set_title("Annotated instances by category")
    ax.set_ylabel("Number of instances")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return counts


def plot_training_history(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, history["train_loss"], "-o", linewidth=1.5, markersize=3,
                     label="Train loss")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(alpha=.25)

    val_epochs = np.array([ep for ep, value in zip(epochs, history["val_miou"])
                           if value is not None])
    val_miou = np.array([value for value in history["val_miou"] if value is not None])
    if len(val_epochs):
        axes[0, 1].plot(val_epochs, val_miou, "-o", linewidth=1.5, markersize=4,
                        label="Validation mIoU")
    axes[0, 1].set_title("Validation mean IoU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=.25)

    for name in NAMES:
        values = history["val_iou"][name]
        points = [(ep, value) for ep, value in zip(epochs, values) if value is not None]
        if points:
            point_epochs, point_values = zip(*points)
            axes[1, 0].plot(point_epochs, point_values, "-o", linewidth=1.2, markersize=3,
                            label=name)
    axes[1, 0].set_title("Validation IoU by material")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=.25)

    axes[1, 1].plot(epochs, history["learning_rate"], "-o", linewidth=1.2, markersize=2)
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=160)
    plt.close(fig)
