import argparse
from pathlib import Path

import numpy as np
import torch

from .constants import IGNORE, IMG_EXT, NAMES
from .export import export
from .inference import clean_instances, find_footprint, load_instances, predict_probs
from .io import read_image
from .model import RoofSeg
from .training import cmd_train
from .charts import plot_instance_counts


def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    cfg, n = ck["cfg"], len(NAMES)
    model = RoofSeg(cfg["backbone"], n, cfg["layers"], cfg["dim"], cfg["unfreeze_last"]).to(dev)
    res = model.load_state_dict(ck["state"], strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("bb.")]
    assert not bad and not res.unexpected_keys, f"checkpoint mismatch: {bad[:5]} {res.unexpected_keys[:5]}"
    img = Path(a.image)
    paths = sorted(p for p in img.iterdir() if p.suffix.lower() in IMG_EXT) if img.is_dir() else [img]
    for p in paths:
        print("tile", p.name, flush=True)
        rgb, meta = read_image(p)
        inst = load_instances(find_footprint(a.footprints, p.stem), rgb.shape[:2], meta, a)
        foot = inst > 0
        if not foot.any():
            print("  no buildings"); continue
        lab = predict_probs(model, rgb, foot, dev, cfg, a.tta).argmax(0).numpy().astype(np.uint8)
        lab[~foot] = IGNORE
        clean_instances(lab, inst, n, a.min_frag_px, a.min_frag_frac)
        stats = export(a.out_dir, p.stem, rgb, lab, inst, meta)
        mixed = sum(s["n_materials"] > 1 for s in stats.values())
        print(f"  {len(stats)} buildings, {mixed} with mixed roof materials")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--coco", required=True); t.add_argument("--images", required=True)
    t.add_argument("--backbone", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    t.add_argument("--out", default="roof_seg.pt")
    t.add_argument("--epochs", type=int, default=80); t.add_argument("--bs", type=int, default=16)
    t.add_argument("--workers", type=int, default=0,
                   help="DataLoader worker processes (0 avoids /dev/shm limits)")
    t.add_argument("--n_per_tile", type=int, default=12, help="random crops per tile per epoch")
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--bb_lr", type=float, default=1e-5)
    t.add_argument("--unfreeze_last", type=int, default=0, help="fine-tune the last N backbone blocks")
    t.add_argument("--crop", type=int, default=256)
    t.add_argument("--input_size", type=int, default=512, help="crops are upsampled to this (finer token grid)")
    t.add_argument("--dice", type=float, default=.5)
    t.add_argument("--val_every", type=int, default=5, help="block-split period; 0 = train on all tiles, no val")
    t.add_argument("--plot_every", type=int, default=5,
                   help="evaluate validation and refresh training curves every N epochs")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    c = sp.add_parser("charts")
    c.add_argument("--coco", required=True)
    c.add_argument("--out_dir", default="charts")
    def cmd_charts(a):
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        counts = plot_instance_counts(a.coco, out / "instance_counts.png")
        print("instances:", counts)
        print("saved:", out / "instance_counts.png")
    c.set_defaults(fn=cmd_charts)
    p = sp.add_parser("predict")
    p.add_argument("--image", required=True, help="tile file or folder of tiles")
    p.add_argument("--footprints", required=True, help="GeoJSON | binary mask raster | folder of <stem>.{geojson,png,tif}")
    p.add_argument("--weights", required=True); p.add_argument("--out_dir", default="roofseg_out")
    p.add_argument("--footprints_crs", default="EPSG:4326", help="CRS of the GeoJSON footprints")
    p.add_argument("--fp_threshold", type=float, default=0, help="raster footprints: pixel > threshold = building")
    p.add_argument("--min_building_px", type=int, default=30)
    p.add_argument("--min_frag_px", type=int, default=25, help="min size of a material fragment inside a building")
    p.add_argument("--min_frag_frac", type=float, default=.02, help="...or this fraction of the building area")
    p.add_argument("--tta", action="store_true", help="4-rotation test-time augmentation (4x slower)")
    p.set_defaults(fn=cmd_predict)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
