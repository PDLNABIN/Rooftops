import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .constants import IGNORE, NAME2IDX


def read_image(path):
    """-> rgb uint8 HxWx3, meta{transform, crs}. Georeferencing kept when rasterio can read it."""
    meta = {"transform": None, "crs": None}
    try:
        import rasterio
        with rasterio.open(str(path)) as s:
            a = s.read(list(range(1, min(3, s.count) + 1))).transpose(1, 2, 0)
            if a.shape[2] == 1:
                a = np.repeat(a, 3, 2)
            if s.crs is not None:
                meta = {"transform": s.transform, "crs": s.crs}
    except ImportError:
        a = np.array(Image.open(str(path)).convert("RGB"))
    if a.dtype == np.uint16:
        a = a / 257
    return np.clip(a, 0, 255).astype(np.uint8), meta


def read_mask(path):
    if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
        a = np.array(Image.open(path))
    else:
        import rasterio
        with rasterio.open(str(path)) as s:
            a = s.read(1)
    return a[..., 0] if a.ndim == 3 else a


def load_tiles(coco_path, img_dir):
    """-> sorted list of (name, rgb, label, footprint). label = material idx on annotated roofs, 255 elsewhere."""
    d = json.load(open(coco_path))
    cat = {}
    for c in d["categories"]:
        k = c["name"].strip().lower()
        if k not in NAME2IDX:
            raise ValueError(f"Unknown category '{c['name']}'. Expected one of {list(NAME2IDX)}")
        cat[c["id"]] = NAME2IDX[k]
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    out = []
    for im in sorted(d["images"], key=lambda i: i["file_name"]):
        anns = by.get(im["id"], [])
        if not anns:
            continue  # no labels -> no training signal
        W, H = im["width"], im["height"]
        lab = Image.new("L", (W, H), IGNORE); foot = Image.new("L", (W, H), 0)
        dl, df = ImageDraw.Draw(lab), ImageDraw.Draw(foot)
        for a in sorted(anns, key=lambda a: -a.get("area", 0)):  # big first -> small patches stay on top
            assert isinstance(a["segmentation"], list), "RLE segmentation not supported; export polygons"
            for poly in a["segmentation"]:
                xy = list(zip(poly[0::2], poly[1::2]))
                dl.polygon(xy, fill=cat[a["category_id"]]); df.polygon(xy, fill=1)
        rgb, _ = read_image(Path(img_dir) / Path(im["file_name"]).name)
        assert rgb.shape[:2] == (H, W), f"{im['file_name']}: image {rgb.shape[:2]} != COCO {(H, W)}"
        out.append((im["file_name"], rgb, np.array(lab), np.array(foot)))
    return out


def split(tiles, val_every):
    """Contiguous blocks of 10 name-sorted tiles -> neighbouring tiles don't leak into validation."""
    if val_every <= 0:
        return tiles, []
    tr, va = [], []
    for i, t in enumerate(tiles):
        (va if (i // 10) % val_every == 0 else tr).append(t)
    return tr, va
