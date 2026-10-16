"""Training loop for the HOTOSM building decoder."""
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import CropDataset, footprint_loss
from .metrics import save, summarize, update_counts
from .model import MaskedDINOClassifier, freeze_all


def train(model, train_tiles, validation_tiles, args, device):
    trainable = [p for p in model.parameters() if p.requires_grad]
    allowed = ("net.pyramid.", "net.decoder.", "net.head.")
    if not trainable or any(
        not name.startswith(allowed) for name, p in model.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("only the pretrained building decoder path may be trainable")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    loader = DataLoader(
        CropDataset(train_tiles, args.crop, args.n_per_tile),
        args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
    )
    validation_loader = (
        DataLoader(CropDataset(validation_tiles, args.crop, args.n_per_tile, augment=False),
                   args.bs, shuffle=False, num_workers=args.workers)
        if validation_tiles else None
    )
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else Path(args.out).with_suffix("").parent / (
        f"{Path(args.out).stem}_training"
    )
    history = []
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        model.net.backbone.eval()
        model.freeze_batchnorm_stats()
        train_loss = 0.0
        train_counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for x, target, _ in progress:
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
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 iou=f"{summarize(train_loss, train_counts, progress.n)['iou']:.3f}")

        validation_loss = 0.0
        validation_counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
        if validation_loader:
            model.eval()
            with torch.no_grad():
                for x, target, _ in validation_loader:
                    x, target = x.to(device), target.to(device)
                    x = F.interpolate(x, size=args.input_size, mode="bilinear", align_corners=False)
                    logits = model(x)
                    target = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
                    validation_loss += float(footprint_loss(logits, target))
                    update_counts(validation_counts, logits, target)
        train_summary = summarize(train_loss, train_counts, len(loader))
        validation_summary = summarize(validation_loss, validation_counts, len(validation_loader) if validation_loader else 1)
        row = {"epoch": epoch + 1, "train_loss": train_summary["loss"], "val_loss": validation_summary["loss"],
               **{f"train_{key}": train_summary[key] for key in ("accuracy", "precision", "recall", "f1", "iou")},
               **{f"val_{key}": validation_summary[key] for key in ("accuracy", "precision", "recall", "f1", "iou")},
               "learning_rate": optimizer.param_groups[0]["lr"],
               "duration_seconds": round(time.perf_counter() - started, 3)}
        history.append(row)
        save(history, metrics_dir)
        if validation_loader:
            scheduler.step(row["val_loss"])
            if row["val_loss"] < best_validation_loss:
                best_validation_loss = row["val_loss"]
                best_state = deepcopy(model.state_dict())
        else:
            scheduler.step(row["train_loss"])
        print(f"epoch {row['epoch']:3d} train_loss={row['train_loss']:.4f} "
              f"val_loss={row['val_loss']:.4f} val_iou={row['val_iou']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} metrics={metrics_dir}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_rooftop_classifier(model, train_tiles, validation_tiles, args, device, class_count):
    freeze_all(model)
    classifier = MaskedDINOClassifier(class_count).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.classifier_lr,
                                  weight_decay=args.classifier_weight_decay)
    classifier_train_tiles = [tile for tile in train_tiles if tile["class_mask"].any()]
    classifier_validation_tiles = [tile for tile in validation_tiles if tile["class_mask"].any()]
    if not classifier_train_tiles:
        raise ValueError("no COCO rooftop class polygons are available for classifier training")
    train_loader = DataLoader(
        CropDataset(classifier_train_tiles, args.crop, args.n_per_tile, class_sampling=True),
        args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
    )
    validation_loader = (
        DataLoader(CropDataset(classifier_validation_tiles, args.crop, args.n_per_tile,
                               augment=False, class_sampling=True),
                   args.bs, shuffle=False, num_workers=args.workers)
        if validation_tiles else None
    )
    history = []
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else Path(args.out).with_suffix("").parent / (
        f"{Path(args.out).stem}_training"
    )
    for epoch in range(args.classifier_epochs):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        progress = tqdm(train_loader, desc=f"Roof classifier {epoch + 1}/{args.classifier_epochs}", unit="batch")
        for x, target, class_target in progress:
            x, target, class_target = x.to(device), target.to(device), class_target.to(device)
            x = F.interpolate(x, size=args.input_size, mode="bilinear", align_corners=False)
            with torch.no_grad():
                features = model.backbone_features(x)
            logits = classifier(features, target)
            loss = F.cross_entropy(logits, class_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            train_correct += int((logits.argmax(1) == class_target).sum())
            train_total += class_target.numel()
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 accuracy=f"{train_correct / max(train_total, 1):.3f}")

        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0
        if validation_loader:
            classifier.eval()
            with torch.no_grad():
                for x, target, class_target in validation_loader:
                    x, target, class_target = x.to(device), target.to(device), class_target.to(device)
                    x = F.interpolate(x, size=args.input_size, mode="bilinear", align_corners=False)
                    logits = classifier(model.backbone_features(x), target)
                    validation_loss += float(F.cross_entropy(logits, class_target))
                    validation_correct += int((logits.argmax(1) == class_target).sum())
                    validation_total += class_target.numel()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val_loss": validation_loss / max(len(validation_loader) if validation_loader else 1, 1),
            "val_accuracy": validation_correct / max(validation_total, 1),
        })
        print(f"roof classifier epoch {epoch + 1:3d} train_loss={history[-1]['train_loss']:.4f} "
              f"val_loss={history[-1]['val_loss']:.4f} val_accuracy={history[-1]['val_accuracy']:.4f}",
              flush=True)
    return classifier, history
