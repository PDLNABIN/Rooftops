import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import IGNORE, IMG_EXT, MEAN, NAMES, STD
from .io import read_mask
from .model import model_forward


def predict_probs(model, rgb, foot, dev, cfg, tta=False):
    """Sliding window over an arbitrary-size image. Windows with no footprint are skipped. -> probs n,H,W (CPU)."""
    model.eval()
    crop, size, n = cfg["crop"], cfg["input_size"], len(NAMES)
    stride = max(crop * 3 // 4, 1)
    H0, W0 = foot.shape
    ph, pw = max(0, crop - H0), max(0, crop - W0)
    if ph or pw:
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge"); foot = np.pad(foot, ((0, ph), (0, pw)))
    H, W = foot.shape
    img = torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    fm = torch.from_numpy(foot.astype(np.float32))[None]
    acc, cnt = torch.zeros(n, H, W), torch.zeros(1, H, W)
    ys = sorted(set(range(0, H - crop + 1, stride)) | {H - crop})
    xs = sorted(set(range(0, W - crop + 1, stride)) | {W - crop})
    for y in ys:
        for x in xs:
            if not foot[y:y + crop, x:x + crop].any():
                continue
            xi, mi = img[None, :, y:y + crop, x:x + crop].to(dev), fm[None, :, y:y + crop, x:x + crop].to(dev)
            pr = 0
            for r in range(4 if tta else 1):  # optional 4-rotation test-time augmentation
                lo = model_forward(model, torch.rot90(xi, r, (2, 3)), torch.rot90(mi, r, (2, 3)), size)
                pr = pr + torch.rot90(lo.softmax(1), -r, (2, 3))
            acc[:, y:y + crop, x:x + crop] += (pr / (4 if tta else 1))[0].cpu()
            cnt[:, y:y + crop, x:x + crop] += 1
    return (acc / cnt.clamp(min=1))[:, :H0, :W0]


def clean_instances(lab, inst, n, min_px, min_frac):
    """Per building: merge tiny material fragments into the neighbouring material they touch most.
    Fragment = connected region of one material smaller than max(min_px, min_frac * building area).
    A building that is a single material, or a fragment with no in-building neighbour, is left untouched."""
    from scipy import ndimage as ndi
    for bid, sl in enumerate(ndi.find_objects(inst), 1):
        if sl is None:
            continue
        sub, mine = lab[sl], inst[sl] == bid  # `sub` is a view -> edits land in `lab`
        thr = max(min_px, min_frac * mine.sum())
        for _ in range(2):
            changed = False
            for c in range(n):
                comp, k = ndi.label((sub == c) & mine)
                for j in range(1, k + 1):
                    cm = comp == j
                    if cm.sum() >= thr:
                        continue
                    ring = ndi.binary_dilation(cm) & mine & ~cm
                    nb = sub[ring].astype(np.int64)
                    nb = nb[nb != c]
                    if nb.size:
                        sub[cm] = np.bincount(nb, minlength=n).argmax(); changed = True
            if not changed:
                break
    return lab


def load_instances(fp_path, shape, meta, a):
    """Footprints -> int32 instance raster (0 = no building, 1..N = building id)."""
    p, (H, W) = Path(fp_path), shape
    if p.suffix.lower() in {".geojson", ".json"}:
        from rasterio import features, warp
        from affine import Affine
        gj = json.load(open(p))
        feats = gj["features"] if "features" in gj else [gj]
        georef = meta["crs"] is not None
        tf = meta["transform"] if georef else Affine.identity()  # no CRS on tile -> polygons in pixel coords
        shp = []
        for i, f in enumerate(feats, 1):
            g = f.get("geometry")
            if g:
                shp.append((warp.transform_geom(a.footprints_crs, meta["crs"], g) if georef else g, i))
        return features.rasterize(shp, out_shape=(H, W), transform=tf, fill=0, dtype="int32")
    from scipy import ndimage as ndi
    m = read_mask(p) > a.fp_threshold
    assert m.shape == (H, W), f"footprint raster {m.shape} != image {(H, W)}"
    inst, _ = ndi.label(m)  # 4-connected: buildings that touch in the mask are ONE instance - prefer vector input
    sizes = np.bincount(inst.ravel())
    inst[np.isin(inst, [i for i in range(1, len(sizes)) if sizes[i] < a.min_building_px])] = 0
    return inst.astype(np.int32)


def find_footprint(fp, stem):
    fp = Path(fp)
    if fp.is_file():
        return fp
    for ext in (".geojson", ".json", ".png", ".tif", ".tiff"):
        if (fp / f"{stem}{ext}").exists():
            return fp / f"{stem}{ext}"
    raise FileNotFoundError(f"No footprint for tile '{stem}' in {fp}")
