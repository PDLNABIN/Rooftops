#!/usr/bin/env python3
"""Fine-tune the pretrained HOTOSM DINOv3 building UPerNet decoder."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import CropDataset, footprint_loss, load_tiles, split


class Pyramid(nn.Module):
    """HOTOSM's learned four-level pyramid around DINOv3 token features."""

    def __init__(self, channels=384):
        super().__init__()
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, 2),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channels // 2, channels // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(channels, channels // 2, 2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.Sequential(nn.MaxPool2d(2, 2))

    @staticmethod
    def _tokens_to_image(tokens, height, width):
        grid_h, grid_w = height // 16, width // 16
        tokens = tokens[:, -grid_h * grid_w:]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, grid_h, grid_w)

    def forward(self, features, height, width):
        images = [self._tokens_to_image(f, height, width) for f in features]
        return [self.fpn1(images[0]), self.fpn2(images[1]), self.fpn3(images[2]), self.fpn4(images[3])]


class HOTOSMNet(nn.Module):
    """Exact trainable structure represented by hotosm/dinov3s-buildings/model.ckpt."""

    def __init__(self):
        super().__init__()
        from terratorch.models.decoders.upernet_decoder import UperNetDecoder

        self.net = nn.Module()
        self.net.backbone = torch.hub.load(
            "facebookresearch/dinov3", "dinov3_vits16", pretrained=False
        )
        self.net.pyramid = Pyramid()
        self.net.decoder = UperNetDecoder([96, 192, 384, 384], channels=512)
        self.net.head = nn.Conv2d(512, 3, 1)
        self.net.aux_head = nn.Sequential(
            nn.Conv2d(384, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 3, 1),
        )

    def freeze_batchnorm_stats(self):
        """Use pretrained BatchNorm statistics so batch-size-one training is valid."""
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def forward(self, x):
        height, width = x.shape[-2:]
        with torch.no_grad():
            features = self.net.backbone.get_intermediate_layers(
                x, n=(2, 5, 8, 11), return_class_token=False
            )
        pyramidal = self.net.pyramid(features, height, width)
        decoded = self.net.decoder(pyramidal)
        return F.interpolate(self.net.head(decoded), size=(height, width), mode="bilinear", align_corners=False)


def load_pretrained(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a Lightning checkpoint with a state_dict")
    model = HOTOSMNet()
    result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {result}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.net.pyramid, model.net.decoder, model.net.head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.net.backbone.eval()
    model.to(device)
    return model


def segmentation_metrics(loss, counts, batches):
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


def save_training_metrics(history, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training_metrics.json").write_text(json.dumps(history, indent=2))
    fields = list(history[0])
    with (directory / "training_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plots = (
        ("loss", "train_loss", "val_loss", "Loss"),
        ("iou", "train_iou", "val_iou", "Intersection over Union"),
        ("f1", "train_f1", "val_f1", "F1 score"),
    )
    for filename, train_key, val_key, ylabel in plots:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="hotosm/dinov3s-buildings model.ckpt")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--footprints", help="directory of per-tile footprint targets")
    parser.add_argument("--out", default="dinov3s-buildings-finetuned.ckpt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=1, help="reduce if CUDA runs out of memory")
    parser.add_argument("--n_per_tile", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=256, help="decoder resolution; 256 is GPU-friendly")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics_dir", help="directory for JSON, CSV, and PNG metrics")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tiles, validation_tiles = split(load_tiles(args.coco, args.images, args.footprints), args.val_every)
    if not train_tiles or not any(tile["target"].any() for tile in train_tiles):
        raise SystemExit("no footprint targets found; pass --footprints or provide roof polygons in COCO")

    model = load_pretrained(args.checkpoint, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    allowed = ("net.pyramid.", "net.decoder.", "net.head.")
    if not trainable or any(
        not name.startswith(allowed) for name, p in model.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("only the pretrained building decoder path may be trainable")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    loader = DataLoader(
        CropDataset(train_tiles, args.crop, args.n_per_tile),
        args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    validation_loader = (
        DataLoader(
            CropDataset(validation_tiles, args.crop, args.n_per_tile),
            args.bs,
            shuffle=False,
            num_workers=args.workers,
        )
        if validation_tiles
        else None
    )
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else Path(args.out).with_suffix("").parent / (
        f"{Path(args.out).stem}_training"
    )
    history = []

    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        model.net.backbone.eval()
        model.freeze_batchnorm_stats()
        train_loss = 0.0
        train_counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for x, target in progress:
            x, target = x.to(device), target.to(device)
            x = F.interpolate(x, size=args.input_size, mode="bilinear", align_corners=False)
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(x)
                target = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
                loss = footprint_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            update_counts(train_counts, logits.detach(), target)
            running = segmentation_metrics(train_loss, train_counts, progress.n)
            progress.set_postfix(loss=f"{loss.item():.4f}", iou=f"{running['iou']:.3f}")

        validation_loss = 0.0
        validation_counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
        if validation_loader:
            model.eval()
            with torch.no_grad():
                for x, target in validation_loader:
                    x, target = x.to(device), target.to(device)
                    x = F.interpolate(x, size=args.input_size, mode="bilinear", align_corners=False)
                    logits = model(x)
                    target = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
                    validation_loss += float(footprint_loss(logits, target))
                    update_counts(validation_counts, logits, target)
        train_summary = segmentation_metrics(train_loss, train_counts, len(loader))
        validation_summary = segmentation_metrics(
            validation_loss, validation_counts, len(validation_loader) if validation_loader else 1
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_summary["loss"],
            "val_loss": validation_summary["loss"],
            "train_accuracy": train_summary["accuracy"],
            "val_accuracy": validation_summary["accuracy"],
            "train_precision": train_summary["precision"],
            "val_precision": validation_summary["precision"],
            "train_recall": train_summary["recall"],
            "val_recall": validation_summary["recall"],
            "train_f1": train_summary["f1"],
            "val_f1": validation_summary["f1"],
            "train_iou": train_summary["iou"],
            "val_iou": validation_summary["iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
        history.append(row)
        save_training_metrics(history, metrics_dir)
        print(
            f"epoch {row['epoch']:3d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_iou={row['val_iou']:.4f} "
            f"metrics={metrics_dir}",
            flush=True,
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": {"seg_out_indices": (2, 5, 8, 11), "decoder_channels": 512},
        },
        args.out,
    )
    print("saved:", Path(args.out))


if __name__ == "__main__":
    main()
