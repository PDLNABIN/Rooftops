"""Training loop for the HOTOSM building decoder."""
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import CropDataset, footprint_loss
from .metrics import save, summarize, update_counts


def train(model, train_tiles, validation_tiles, args, device):
    trainable = [p for p in model.parameters() if p.requires_grad]
    allowed = ("net.pyramid.", "net.decoder.", "net.head.")
    if not trainable or any(
        not name.startswith(allowed) for name, p in model.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("only the pretrained building decoder path may be trainable")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    loader = DataLoader(
        CropDataset(train_tiles, args.crop, args.n_per_tile),
        args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
    )
    validation_loader = (
        DataLoader(CropDataset(validation_tiles, args.crop, args.n_per_tile),
                   args.bs, shuffle=False, num_workers=args.workers)
        if validation_tiles else None
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
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 iou=f"{summarize(train_loss, train_counts, progress.n)['iou']:.3f}")

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
        train_summary = summarize(train_loss, train_counts, len(loader))
        validation_summary = summarize(validation_loss, validation_counts, len(validation_loader) if validation_loader else 1)
        row = {"epoch": epoch + 1, "train_loss": train_summary["loss"], "val_loss": validation_summary["loss"],
               **{f"train_{key}": train_summary[key] for key in ("accuracy", "precision", "recall", "f1", "iou")},
               **{f"val_{key}": validation_summary[key] for key in ("accuracy", "precision", "recall", "f1", "iou")},
               "learning_rate": optimizer.param_groups[0]["lr"],
               "duration_seconds": round(time.perf_counter() - started, 3)}
        history.append(row)
        save(history, metrics_dir)
        print(f"epoch {row['epoch']:3d} train_loss={row['train_loss']:.4f} "
              f"val_loss={row['val_loss']:.4f} val_iou={row['val_iou']:.4f} metrics={metrics_dir}", flush=True)
    return history
