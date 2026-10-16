"""Training metrics and plot persistence."""
import csv
import json
from pathlib import Path


def update_counts(counts, logits, target):
    prediction = logits[:, 0].sigmoid() >= 0.5
    truth = target == 1
    for key, value in {
        "tp": (prediction & truth).sum(),
        "tn": (~prediction & ~truth).sum(),
        "fp": (prediction & ~truth).sum(),
        "fn": (~prediction & truth).sum(),
    }.items():
        counts[key] += int(value)


def summarize(loss, counts, batches):
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    pixels = max(tp + tn + fp + fn, 1)
    return {
        "loss": loss / max(batches, 1),
        "accuracy": (tp + tn) / pixels,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }


def save(history, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training_metrics.json").write_text(json.dumps(history, indent=2))
    with (directory / "training_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    for filename, train_key, val_key, ylabel in (
        ("loss", "train_loss", "val_loss", "Loss"),
        ("iou", "train_iou", "val_iou", "Intersection over Union"),
        ("f1", "train_f1", "val_f1", "F1 score"),
    ):
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(epochs, [row[train_key] for row in history], marker="o", label="train")
        axis.plot(epochs, [row[val_key] for row in history], marker="o", label="validation")
        axis.set(xlabel="Epoch", ylabel=ylabel, title=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.savefig(directory / f"{filename}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, [row["learning_rate"] for row in history], marker="o")
    axis.set(xlabel="Epoch", ylabel="Learning rate", title="Learning rate")
    axis.grid(alpha=0.25)
    fig.savefig(directory / "learning_rate.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
