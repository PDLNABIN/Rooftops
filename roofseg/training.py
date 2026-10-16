import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import IGNORE, NAMES
from .data import CropDS, jitter_mask
from .io import load_tiles, split
from .inference import predict_probs
from .model import RoofSeg, dice_loss, model_forward
from .charts import plot_training_history


def confusion(pred, gt, n):
    k = gt != IGNORE
    return np.bincount(gt[k].astype(np.int64) * n + pred[k], minlength=n * n).reshape(n, n)


def evaluate_tiles(model, tiles, dev, cfg):
    """Full inference path (sliding window, footprint-gated) on whole tiles, scored on labelled roof pixels."""
    n = len(NAMES); cm = np.zeros((n, n), np.int64)
    for _, rgb, lab, foot in tiles:
        pred = predict_probs(model, rgb, foot, dev, cfg).argmax(0).numpy()
        cm += confusion(pred, lab, n)
    iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
    return iou, cm.sum(1) > 0, cm


def make_cfg(a):
    return {"backbone": a.backbone, "layers": [2, 5, 8, 11], "dim": 128, "unfreeze_last": a.unfreeze_last,
            "crop": a.crop, "input_size": a.input_size}


def cmd_train(a):
    if a.plot_every < 1:
        raise ValueError("--plot_every must be at least 1")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, cfg = len(NAMES), make_cfg(a)
    tr, va = split(load_tiles(a.coco, a.images), a.val_every)
    print(f"annotated tiles: train={len(tr)} val={len(va)}")
    cnt = np.bincount(np.concatenate([t[2].ravel() for t in tr]), minlength=256)[:n].astype(float)
    print("labelled pixels (train):", {NAMES[i]: int(cnt[i]) for i in range(n)})
    if va:
        vc = np.bincount(np.concatenate([t[2].ravel() for t in va]), minlength=256)[:n]
        print("labelled pixels (val)  :", {NAMES[i]: int(vc[i]) for i in range(n)})
    w = 1 / np.sqrt(cnt / cnt.sum() + 1e-6); w = np.clip(w / w.mean(), .2, 5)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), ignore_index=IGNORE)

    tl = DataLoader(CropDS(tr, a.crop, a.n_per_tile), a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True)
    model = RoofSeg(a.backbone, n, cfg["layers"], cfg["dim"], a.unfreeze_last).to(dev)
    dec = [p for k, p in model.named_parameters() if p.requires_grad and not k.startswith("bb.")]
    bbp = [p for k, p in model.named_parameters() if p.requires_grad and k.startswith("bb.")]
    groups = [{"params": dec, "lr": a.lr}] + ([{"params": bbp, "lr": a.bb_lr}] if bbp else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    keep = {k for k, p in model.named_parameters() if p.requires_grad or not k.startswith("bb.")}

    def save():
        sd = {k: v for k, v in model.state_dict().items() if k in keep or not k.startswith("bb.")}
        torch.save({"state": sd, "cfg": cfg}, a.out)

    best = -1.0
    history = {"train_loss": [], "val_miou": [], "val_iou": {name: [] for name in NAMES},
               "learning_rate": []}
    chart_dir = Path(a.out).parent / f"{Path(a.out).stem}_charts"

    def save_history():
        chart_dir.mkdir(parents=True, exist_ok=True)
        (chart_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        plot_training_history(history, chart_dir)

    for ep in range(a.epochs):
        model.train(); model.bb.eval()
        tot = 0.0
        for x, m, y in tl:
            x, m, y = x.to(dev), jitter_mask(m.to(dev)), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                lo = model_forward(model, x, m, a.input_size)
            lo = lo.float()
            loss = crit(lo, y) + a.dice * dice_loss(lo, y, n)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        msg = f"ep{ep + 1:3d} loss={tot / len(tl):.4f}"
        history["train_loss"].append(tot / len(tl))
        history["learning_rate"].append(opt.param_groups[0]["lr"])
        history["val_miou"].append(None)
        for name in NAMES:
            history["val_iou"][name].append(None)
        should_plot = (ep + 1) % a.plot_every == 0 or ep == a.epochs - 1
        if va and should_plot:
            iou, present, cm = evaluate_tiles(model, va, dev, cfg)
            miou = float(iou[present].mean())
            msg += f" | val mIoU={miou:.3f} " + " ".join(
                f"{NAMES[i]}={iou[i]:.2f}" + ("" if present[i] else "(absent)") for i in range(n))
            history["val_miou"][-1] = miou
            for i, name in enumerate(NAMES):
                history["val_iou"][name][-1] = float(iou[i]) if present[i] else None
            if miou > best:
                best, best_cm = miou, cm; save(); msg += "  *saved*"
        print(msg, flush=True)
        if should_plot:
            save_history()
    if not va:
        save(); print("no validation split: saved last epoch ->", a.out)
    else:
        print(f"best val mIoU {best:.3f} -> {a.out}\nconfusion (rows=true, cols=pred) {NAMES}\n{best_cm}")
        print("NOTE: validation uses label-exact footprints, so it is optimistic. Check on real predicted footprints.")
