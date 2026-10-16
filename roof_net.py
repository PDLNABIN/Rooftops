#!/usr/bin/env python3
"""
roofnet.py - ONE frozen DINOv3 ViT-S/16 backbone, several independent heads (one backbone pass per window).

  fp  : building FOOTPRINT segmentation (binary)                         [semantic mask -> instances]
  mat : roof-MATERIAL segmentation inside a footprint (CGI/Concrete/Tarpaulin/Tile), conditioned on a footprint mask
  cls : building-LEVEL roof type: dominant material (4-way) + materials present (4 multi-label).
        It is a dense per-token head whose logits are mean-pooled over each building, so it works on windows.

Because the backbone is frozen, heads never interfere: train all at once, or add/retrain one later
(`--init roofnet.pt --train_heads cls`) and the other heads stay byte-identical.

Footprint supervision (fp head) comes from, in order of preference:
  --footprints DIR : <tile_stem>.{geojson,png,tif} per tile (e.g. Google Open Buildings polygons,
                     fAIr / hotosm-model output, hand digitised ...). Also used for tiles WITHOUT roof annotations.
  otherwise        : union of your roof polygons on annotated tiles (assumes every building on those tiles is annotated).

Install: pip install torch "transformers>=4.56" numpy pillow scipy rasterio shapely     (single file, nothing else needed)

  python roofnet.py train   --coco instances_default.json --images tiles/ [--footprints fp_dir/] --out roofnet.pt
  python roofnet.py predict --image tiles/ --weights roofnet.pt --out_dir out/                    # imagery only!
  python roofnet.py predict --image tiles/ --weights roofnet.pt --footprints polys.geojson --qa_fp # external polygons
  python roofnet.py inspect --ckpt model.ckpt        # list tensors in the hotosm Lightning checkpoint
"""
import argparse, csv, json, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from torch.utils.data import Dataset, DataLoader


# =============================================================== constants
IGNORE = 255
NAMES = ["CGI", "Concrete", "Tarpaulin", "Tile"]
NAME2IDX = {"cgi": 0, "concrete": 1, "tarpaulin": 2, "tile": 3, "mud": IGNORE}  # Mud: 1 example -> ignored
PALETTE = np.array([(70, 130, 180), (150, 150, 150), (255, 127, 0), (200, 50, 50)], np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# =============================================================== IO, losses and export helpers
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


def jitter_mask(m):
    """Random dilation/erosion (0-3 px) of the footprint channel: predicted footprints are never as tight as labels."""
    k = random.choice([0, 0, 1, 2, 3])
    if k == 0:
        return m
    if random.random() < .5:
        return F.max_pool2d(m, 2 * k + 1, 1, k)
    return 1 - F.max_pool2d(1 - m, 2 * k + 1, 1, k)


def dice_loss(logits, y, n):
    valid = y != IGNORE
    p = logits.softmax(1) * valid[:, None]
    t = F.one_hot(torch.where(valid, y, torch.zeros_like(y)), n).permute(0, 3, 1, 2) * valid[:, None]
    inter, den = (p * t).sum((0, 2, 3)), p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    return (1 - (2 * inter + 1) / (den + 1))[present].mean()


def confusion(pred, gt, n):
    k = gt != IGNORE
    return np.bincount(gt[k].astype(np.int64) * n + pred[k], minlength=n * n).reshape(n, n)


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
    """Footprints -> (int32 instance raster [0 = none, 1..N = building id], info).
    info[bid] = {"geom": original polygon (GeoJSON mapping, in the TILE CRS), "props": source properties}
    for vector input; {} for raster input. For GeoJSON, building id == 1-based feature order."""
    p, (H, W) = Path(fp_path), shape
    if p.suffix.lower() in {".geojson", ".json"}:
        from rasterio import features, warp
        from affine import Affine
        gj = json.load(open(p))
        feats = gj["features"] if "features" in gj else [gj]
        georef = meta["crs"] is not None
        tf = meta["transform"] if georef else Affine.identity()  # no CRS on tile -> polygons in pixel coords
        shp, info = [], {}
        for i, f in enumerate(feats, 1):
            g = f.get("geometry")
            if g:
                g = warp.transform_geom(a.footprints_crs, meta["crs"], g) if georef else g
                shp.append((g, i)); info[i] = {"geom": g, "props": f.get("properties") or {}}
        return features.rasterize(shp, out_shape=(H, W), transform=tf, fill=0, dtype="int32"), info
    from scipy import ndimage as ndi
    m = read_mask(p) > a.fp_threshold
    assert m.shape == (H, W), f"footprint raster {m.shape} != image {(H, W)}"
    inst, _ = ndi.label(m)  # 4-connected: buildings that touch in the mask are ONE instance - prefer vector input
    sizes = np.bincount(inst.ravel())
    inst[np.isin(inst, [i for i in range(1, len(sizes)) if sizes[i] < a.min_building_px])] = 0
    return inst.astype(np.int32), {}


def find_footprint(fp, stem):
    fp = Path(fp)
    if fp.is_file():
        return fp
    for ext in (".geojson", ".json", ".png", ".tif", ".tiff"):
        if (fp / f"{stem}{ext}").exists():
            return fp / f"{stem}{ext}"
    raise FileNotFoundError(f"No footprint for tile '{stem}' in {fp}")


def building_stats(lab, inst, n):
    from scipy import ndimage as ndi
    out = {}
    for bid, sl in enumerate(ndi.find_objects(inst), 1):
        if sl is None:
            continue
        cnt = np.bincount(lab[sl][inst[sl] == bid].astype(np.int64), minlength=n)[:n]
        tot = int(cnt.sum())
        out[bid] = {"area_px": tot,
                    "area_px_by_material": {NAMES[i]: int(cnt[i]) for i in range(n)},
                    "fractions": {NAMES[i]: round(float(cnt[i]) / tot, 4) for i in range(n)},
                    "dominant": NAMES[int(cnt.argmax())],
                    "n_materials": int((cnt / max(tot, 1) >= 0.02).sum())}
    return out


def _polys_only(g):
    from shapely.geometry import MultiPolygon
    if g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    ps = []
    for x in getattr(g, "geoms", []):
        if x.geom_type == "Polygon": ps.append(x)
        elif x.geom_type == "MultiPolygon": ps += list(x.geoms)
    return MultiPolygon(ps) if ps else None


def vectorize_building(lab, inst, bid, sl, tf, bgeom, simplify_px):
    """-> (building polygon, {material_idx: polygon(s)}) in tile CRS.
    Parts come from this building's pixel map, are simplified TOGETHER (shared borders stay shared, no gaps/overlaps
    between materials) and clipped to the building polygon (the original footprint when we have it)."""
    import shapely
    from affine import Affine
    from rasterio import features
    from shapely.geometry import shape
    from shapely.ops import unary_union
    sub_tf = tf * Affine.translation(sl[1].start, sl[0].start)
    mine = inst[sl] == bid
    by = {}
    for g, v in features.shapes(lab[sl], mask=mine, transform=sub_tf):
        by.setdefault(int(v), []).append(shape(g))
    mats = sorted(by)
    regions = [unary_union(by[m]) for m in mats]
    tol = simplify_px * abs(tf.a)
    if tol > 0 and regions:
        try:
            regions = list(shapely.coverage_simplify(regions, tol))
        except Exception:  # older shapely/GEOS: per-part simplification (tiny gaps <= tol)
            regions = [r.simplify(tol, preserve_topology=True) for r in regions]
    if bgeom is not None:
        bpoly = shapely.make_valid(shape(bgeom))
    else:  # raster footprints: outline = outer boundary of the instance mask
        bpoly = unary_union([shape(g) for g, _ in features.shapes(mine.astype(np.uint8), mask=mine, transform=sub_tf)])
        bpoly = bpoly.simplify(tol, preserve_topology=True) if tol > 0 else bpoly
    parts = {}
    for m, r in zip(mats, regions):
        r = _polys_only(shapely.make_valid(r).intersection(bpoly))
        if r is not None and r.area > 0:
            parts[m] = r
    return bpoly, parts


def new_coco():
    # ids match the original CVAT label set so predictions can be corrected in CVAT and fed back to `train`
    cats = [(1, "CGI"), (2, "Concrete"), (3, "Mud"), (4, "Tarpaulin"), (5, "Tile")]
    return {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in cats]}


def coco_add(coco, file_name, W, H, items):
    ids = {"CGI": 1, "Concrete": 2, "Tarpaulin": 4, "Tile": 5}
    iid = len(coco["images"]) + 1
    coco["images"].append({"id": iid, "file_name": file_name, "width": W, "height": H})
    for it in items:
        coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": iid,
                                    "category_id": ids[it["material"]], "segmentation": it["segmentation"],
                                    "area": it["area"], "bbox": it["bbox"], "iscrowd": 0,
                                    "attributes": {"building_id": it["building_id"]}})


def plot_prediction(out_dir, stem, rgb, lab, inst, stats):
    """Save an annotated four-panel prediction plot for one tile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy import ndimage as ndi

    out_dir = Path(out_dir)
    n = len(NAMES)
    colors = PALETTE / 255.0
    masked = np.ma.masked_where(inst == 0, lab)
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("Input image")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(rgb)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=n - 1, alpha=0.62)
    if (inst > 0).any():
        ax.contour(inst > 0, levels=[0.5], colors="white", linewidths=0.7)
        for bid in stats:
            y, x = ndi.center_of_mass(inst == bid)
            ax.text(x, y, str(bid), color="black", fontsize=7, weight="bold",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    ax.set_title(f"Predicted buildings and materials ({len(stats)} buildings)")
    ax.legend(handles=[Patch(facecolor=colors[i], label=name) for i, name in enumerate(NAMES)],
              loc="upper right", fontsize=8, framealpha=0.9)
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=n - 1)
    ax.set_title("Roof-material segmentation")
    ax.legend(handles=[Patch(facecolor=colors[i], label=name) for i, name in enumerate(NAMES)],
              loc="upper right", fontsize=8, framealpha=0.9)
    ax.axis("off")

    dominant = {name: 0 for name in NAMES}
    for item in stats.values():
        dominant[item["dominant"]] += 1
    ax = axes[1, 1]
    bars = ax.bar(NAMES, [dominant[name] for name in NAMES], color=colors)
    ax.set_title("Buildings by dominant roof material")
    ax.set_ylabel("Number of buildings")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(dominant.values(), default=0) * 1.2 + 1)

    fig.suptitle(f"RoofNet prediction: {stem}", fontsize=16, weight="bold")
    fig.savefig(out_dir / f"{stem}_prediction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_summary(out_dir, rows):
    """Save a summary chart across all predicted buildings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {name: 0 for name in NAMES}
    mixed = 0
    for row in rows:
        name = row["seg_dominant"]
        if name in counts:
            counts[name] += 1
        mixed += int(row["n_materials"] > 1)
    colors = PALETTE / 255.0
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(list(counts), list(counts.values()), color=colors)
    ax.set_title("RoofNet predicted roof materials")
    ax.set_ylabel("Number of buildings")
    ax.text(0.99, 0.96, f"Total buildings: {len(rows)}\nMixed-material buildings: {mixed}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(counts.values(), default=0) * 1.2 + 1)
    fig.savefig(Path(out_dir) / "material_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def export(out_dir, stem, rgb, lab, inst, meta, info=None, simplify_px=0.7, extra=None):
    """Writes: <stem>_materials.tif, <stem>_overlay.png, <stem>_buildings.geojson (one feature per building:
    original footprint + roof types + nested part polygons), <stem>_roof_parts.geojson (flat: one feature per
    building x material, for GIS styling). Returns (stats, coco_items)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n, info, extra = len(NAMES), info or {}, extra or {}  # extra[bid] = dict merged into the building properties
    try:
        import rasterio
        prof = dict(driver="GTiff", height=lab.shape[0], width=lab.shape[1], count=1, dtype="uint8",
                    nodata=IGNORE, compress="deflate")
        if meta["crs"] is not None:
            prof.update(crs=meta["crs"], transform=meta["transform"])
        with rasterio.open(out_dir / f"{stem}_materials.tif", "w", **prof) as d:
            d.write(lab, 1)
    except ImportError:
        Image.fromarray(lab).save(out_dir / f"{stem}_materials.png")
    stats = building_stats(lab, inst, n)
    plot_prediction(out_dir, stem, rgb, lab, inst, stats)
    ov = rgb.copy().astype(np.float32); sel = inst > 0
    ov[sel] = .45 * ov[sel] + .55 * PALETTE[lab[sel].clip(max=n - 1)]
    Image.fromarray(ov.astype(np.uint8)).save(out_dir / f"{stem}_overlay.png")

    coco_items = []
    try:
        from affine import Affine
        from rasterio import warp
        from scipy import ndimage as ndi
        from shapely.affinity import affine_transform
        from shapely.geometry import mapping
    except ImportError:
        print("  [warn] rasterio/shapely missing -> vector + COCO export skipped"); return stats, coco_items
    georef = meta["crs"] is not None
    tf = meta["transform"] if georef else Affine.identity()
    px_m2 = None
    if georef and meta["crs"].is_projected and str(meta["crs"].linear_units).lower().startswith("met"):
        px_m2 = abs(tf.a * tf.e)
    to_wgs = (lambda g: warp.transform_geom(meta["crs"], "EPSG:4326", mapping(g))) if georef else mapping
    inv = ~tf
    inv6 = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]  # shapely affine order

    b_feats, p_feats = [], []
    for bid, sl in enumerate(ndi.find_objects(inst), 1):
        if bid not in stats:
            continue
        st = stats[bid]
        bpoly, regs = vectorize_building(lab, inst, bid, sl, tf, (info.get(bid) or {}).get("geom"), simplify_px)
        roof_types = [k for k, v in sorted(st["fractions"].items(), key=lambda kv: -kv[1]) if v >= 0.02]
        parts = []
        for m, g in sorted(regs.items(), key=lambda kv: -st["area_px_by_material"][NAMES[kv[0]]]):
            name = NAMES[m]
            apx = st["area_px_by_material"][name]
            part = {"material": name, "fraction": st["fractions"][name], "area_px": apx}
            if px_m2:
                part["area_m2"] = round(apx * px_m2, 2)
            parts.append({**part, "geometry": to_wgs(g)})
            p_feats.append({"type": "Feature", "geometry": to_wgs(g),
                            "properties": {"building_id": bid, **part, "building_dominant": st["dominant"],
                                           "building_is_mixed": len(roof_types) > 1}})
            gp = affine_transform(g, inv6)  # pixel coordinates for COCO
            polys = list(gp.geoms) if gp.geom_type == "MultiPolygon" else [gp]
            seg = [[round(c, 2) for xy in list(p.exterior.coords)[:-1] for c in xy] for p in polys
                   if len(p.exterior.coords) >= 4]
            if seg:
                x0, y0, x1, y1 = gp.bounds
                coco_items.append({"material": name, "segmentation": seg, "area": round(float(gp.area), 2),
                                   "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                                   "building_id": bid})
        props = dict((info.get(bid) or {}).get("props") or {})  # keep source attrs (e.g. Open Buildings confidence)
        props.update(building_id=bid, roof_types=roof_types, dominant=st["dominant"],
                     is_mixed=len(roof_types) > 1, n_materials=len(roof_types),
                     fractions=st["fractions"], area_px=st["area_px"], parts=parts)
        if px_m2:
            props["area_m2"] = round(st["area_px"] * px_m2, 2)
        props.update(extra.get(bid, {}))
        b_feats.append({"type": "Feature", "geometry": to_wgs(bpoly), "properties": props})
    for name, fs in (("buildings", b_feats), ("roof_parts", p_feats)):
        json.dump({"type": "FeatureCollection", "features": fs}, open(out_dir / f"{stem}_{name}.geojson", "w"))
    return stats, coco_items


# =============================================================== config
N = len(NAMES)
HEADS = ("fp", "mat", "cls")
FRAC_PRESENT, MIN_LABELLED = 0.02, 0.9  # a material is "present" in a building at >=2% of its area


# =============================================================== data
def find_ext_footprint(fp_dir, stem):
    for ext in (".geojson", ".json", ".png", ".tif", ".tiff"):
        if (Path(fp_dir) / f"{stem}{ext}").exists():
            return Path(fp_dir) / f"{stem}{ext}"
    return None


def load_tiles(coco_path, img_dir, fp_dir=None):
    """-> sorted list of dicts: name, rgb, lab (material idx on annotated roofs, 255 elsewhere), foot (union of roof
    polygons), fpt (footprint target 0/1/255), has_mat, has_fp. Tiles with no supervision at all are dropped."""
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
    A = SimpleNamespace(footprints_crs="EPSG:4326", fp_threshold=0, min_building_px=30)
    out = []
    for im in sorted(d["images"], key=lambda i: i["file_name"]):
        W, H = im["width"], im["height"]
        lab = Image.new("L", (W, H), IGNORE); foot = Image.new("L", (W, H), 0)
        dl, df = ImageDraw.Draw(lab), ImageDraw.Draw(foot)
        for a in sorted(by.get(im["id"], []), key=lambda a: -a.get("area", 0)):
            assert isinstance(a["segmentation"], list), "RLE segmentation not supported; export polygons"
            for poly in a["segmentation"]:
                xy = list(zip(poly[0::2], poly[1::2]))
                dl.polygon(xy, fill=cat[a["category_id"]]); df.polygon(xy, fill=1)
        lab, foot = np.array(lab), np.array(foot)
        has_mat = bool((lab != IGNORE).any())
        fpt, has_fp = np.full((H, W), IGNORE, np.uint8), False
        ext = find_ext_footprint(fp_dir, Path(im["file_name"]).stem) if fp_dir else None
        rgb = None
        if ext is not None or has_mat or has_fp:
            rgb, meta = read_image(Path(img_dir) / Path(im["file_name"]).name)
            assert rgb.shape[:2] == (H, W), f"{im['file_name']}: image {rgb.shape[:2]} != COCO {(H, W)}"
        if ext is not None:
            inst, _ = load_instances(ext, (H, W), meta, A)
            fpt, has_fp = ((inst > 0) | (foot > 0)).astype(np.uint8), True  # external OR your exact roofs
        elif has_mat:
            fpt, has_fp = foot.copy(), True
        if has_mat or has_fp:
            out.append(dict(name=im["file_name"], rgb=rgb, lab=lab, foot=foot, fpt=fpt, has_mat=has_mat, has_fp=has_fp))
    return out


def split(tiles, val_every):
    """Contiguous blocks of 10 name-sorted tiles -> neighbouring tiles don't leak into validation."""
    if val_every <= 0:
        return tiles, []
    tr, va = [], []
    for i, t in enumerate(tiles):
        (va if (i // 10) % val_every == 0 else tr).append(t)
    return tr, va


class CropDS(Dataset):
    def __init__(self, tiles, crop, n_per_tile):
        self.t, self.c, self.n = tiles, crop, n_per_tile
        self.pm = [np.argwhere(t["lab"] != IGNORE)[::37] for t in tiles]
        self.pf = [np.argwhere(t["fpt"] == 1)[::37] for t in tiles]

    def __len__(self):
        return len(self.t) * self.n

    def __getitem__(self, k):
        i = k // self.n
        t, c = self.t[i], self.c
        H, W = t["lab"].shape
        pts = self.pm[i] if len(self.pm[i]) else self.pf[i]
        if len(pts):  # centre the crop (with jitter) on a labelled roof / footprint pixel
            py, px = pts[random.randrange(len(pts))]
        else:
            py, px = random.randrange(H), random.randrange(W)
        j = c // 4
        y = int(np.clip(py - c // 2 + random.randint(-j, j), 0, H - c))
        x = int(np.clip(px - c // 2 + random.randint(-j, j), 0, W - c))
        im = t["rgb"][y:y + c, x:x + c].astype(np.float32) / 255
        lab, foot, fpt = (t[q][y:y + c, x:x + c] for q in ("lab", "foot", "fpt"))
        if random.random() < .5: im, lab, foot, fpt = im[:, ::-1], lab[:, ::-1], foot[:, ::-1], fpt[:, ::-1]
        if random.random() < .5: im, lab, foot, fpt = im[::-1], lab[::-1], foot[::-1], fpt[::-1]
        r = random.randint(0, 3)
        im, lab, foot, fpt = (np.rot90(v, r) for v in (im, lab, foot, fpt))
        im = np.clip(im * random.uniform(.8, 1.2) + random.uniform(-.08, .08), 0, 1)
        im = (im - MEAN) / STD
        inst, _ = ndi.label(foot)  # buildings = connected roof regions (adjacent material patches stay together)
        tt = lambda v, dt: torch.from_numpy(np.ascontiguousarray(v).astype(dt))
        return (tt(im.transpose(2, 0, 1), np.float32), tt(foot, np.float32)[None],
                tt(lab, np.int64), tt(fpt, np.int64), tt(inst, np.int64))


# =============================================================== model
class DecoderHead(nn.Module):
    """ViT features (stride 16) + small full-resolution CNN stem (stride 4) -> logits at input resolution."""
    def __init__(self, h, n_feats, dim, in_ch, n_out):
        super().__init__()
        gn = lambda c: nn.GroupNorm(8, c)
        self.proj = nn.ModuleList(nn.Sequential(nn.Conv2d(h, dim, 1), gn(dim), nn.ReLU()) for _ in range(n_feats))
        self.fuse = nn.Sequential(nn.Conv2d(dim * n_feats, dim, 3, padding=1), gn(dim), nn.ReLU())
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 2, 1), gn(32), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, 2, 1), gn(64), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(dim + 64, dim, 3, padding=1), gn(dim), nn.ReLU(),
                                 nn.Conv2d(dim, dim, 3, padding=1), gn(dim), nn.ReLU())
        self.head = nn.Conv2d(dim, n_out, 1)

    def forward(self, feats, x, m=None):
        y = self.fuse(torch.cat([p(f) for p, f in zip(self.proj, feats)], 1))
        s = self.stem(x if m is None else torch.cat([x, m], 1))
        y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec(torch.cat([y, s], 1))
        return F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


class TokenHead(nn.Module):
    """Dense per-token classifier (B,C,gh,gw) -> (B,2N,gh,gw); logits are pooled per building outside."""
    def __init__(self, in_c, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_c, hid, 1), nn.GroupNorm(8, hid), nn.ReLU(), nn.Dropout2d(0.1),
                                 nn.Conv2d(hid, 2 * N, 1))

    def forward(self, f):
        return self.net(f)


class RoofNet(nn.Module):
    def __init__(self, backbone, layers=(2, 5, 8, 11), dim=128):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(backbone)
        for p in self.bb.parameters():
            p.requires_grad = False
        self.layers = tuple(layers)
        h, nf = self.bb.config.hidden_size, len(self.layers)
        self.fp = DecoderHead(h, nf, dim, 3, 1)
        self.mat = DecoderHead(h, nf, dim, 4, N)  # RGB + footprint mask
        self.cls = TokenHead(2 * h)

    def train(self, mode=True):
        super().train(mode); self.bb.eval(); return self  # backbone always eval

    @torch.no_grad()
    def features(self, x):
        B, _, H, W = x.shape
        gh, gw = H // 16, W // 16
        hs = self.bb(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[l + 1][:, -gh * gw:].transpose(1, 2).reshape(B, -1, gh, gw) for l in self.layers]  # drop CLS/regs

    def forward(self, x, m=None, heads=HEADS, fp_thr=0.5):
        """x: B,3,H,W (H,W multiples of 16) | m: B,1,H,W footprint (None -> predicted by the fp head)."""
        feats = self.features(x)
        out = {}
        if "fp" in heads or ("mat" in heads and m is None):
            out["fp"] = self.fp(feats, x)
        if "mat" in heads:
            mm = m if m is not None else (out["fp"].sigmoid() > fp_thr).float()
            out["mat"] = self.mat(feats, x, mm)
        if "cls" in heads:
            out["cls"] = self.cls(torch.cat([feats[-1], feats[-2]], 1))
        return out


# =============================================================== losses
def fp_loss(logit, t):
    valid = t != IGNORE
    if not valid.any():
        return logit.sum() * 0
    tt = t.clamp(max=1).float()
    bce = F.binary_cross_entropy_with_logits(logit[:, 0], tt, reduction="none")[valid].mean()
    p, ti = logit[:, 0].sigmoid() * valid, tt * valid
    return bce + 1 - (2 * (p * ti).sum() + 1) / (p.sum() + ti.sum() + 1)


def mat_loss(logit, y, crit, dice_w):
    if not (y != IGNORE).any():
        return logit.sum() * 0
    return crit(logit, y) + dice_w * dice_loss(logit, y, N)


def cls_loss(cls_map, lab, inst, w, min_px):
    """Building-level loss: logits of the dense token head are mean-pooled over each building (connected roof region).
    Targets come from the labelled pixels of that building inside the crop."""
    dense = F.interpolate(cls_map, size=lab.shape[-2:], mode="bilinear", align_corners=False)
    valid = lab != IGNORE
    oh = F.one_hot(torch.where(valid, lab, torch.zeros_like(lab)), N).float() * valid[..., None]
    L, D, P = [], [], []
    for b in range(lab.shape[0]):
        ids = inst[b].reshape(-1)
        K = int(ids.max()) + 1
        if K <= 1:
            continue
        cnt = torch.bincount(ids, minlength=K).float()
        sums = torch.zeros(K, 2 * N, device=ids.device).index_add_(0, ids, dense[b].reshape(2 * N, -1).T)
        lc = torch.zeros(K, N, device=ids.device).index_add_(0, ids, oh[b].reshape(-1, N))
        lp = lc.sum(1)
        keep = (cnt >= min_px) & (lp >= MIN_LABELLED * cnt); keep[0] = False
        if keep.any():
            frac = lc[keep] / lp[keep][:, None]
            L.append(sums[keep] / cnt[keep][:, None]); D.append(frac.argmax(1)); P.append((frac >= FRAC_PRESENT).float())
    if not L:
        return dense.sum() * 0
    L, D, P = torch.cat(L), torch.cat(D), torch.cat(P)
    return F.cross_entropy(L[:, :N], D, weight=w) + F.binary_cross_entropy_with_logits(L[:, N:], P)


# =============================================================== inference
def window_forward(model, x, m, size, thr, tta, heads):
    """One window, all requested heads, optional 4-rotation TTA. Returns crop-resolution CPU tensors."""
    c, k, acc = x.shape[-1], (4 if tta else 1), {}
    for r in range(k):
        xr = torch.rot90(x, r, (2, 3))
        mr = None if m is None else torch.rot90(m, r, (2, 3))
        xi = F.interpolate(xr, size=size, mode="bilinear", align_corners=False)
        mi = None if mr is None else F.interpolate(mr, size=size, mode="nearest")
        out = model(xi, mi, heads, thr)
        res = {}
        if "fp" in heads: res["fp"] = out["fp"].float().sigmoid()
        if "mat" in heads: res["mat"] = out["mat"].float().softmax(1)
        if "cls" in heads: res["cls"] = out["cls"].float()
        for key, v in res.items():
            v = torch.rot90(F.interpolate(v, size=(c, c), mode="bilinear", align_corners=False), -r, (2, 3))
            acc[key] = acc.get(key, 0) + v / k
    return {k_: v[0].cpu() for k_, v in acc.items()}


@torch.no_grad()
def predict_all(model, rgb, ext, dev, cfg, heads=HEADS, thr=0.5, tta=False):
    """Sliding window over an arbitrary-size tile. ext = bool footprint raster (None -> fp head predicts it per window;
    windows with no footprint are skipped when given). -> dict: fp (H,W) prob, mat (N,H,W) prob, cls (2N,H,W) logits."""
    model.eval()
    crop, size = cfg["crop"], cfg["input_size"]
    stride = max(crop * 3 // 4, 1)
    H0, W0 = rgb.shape[:2]
    ph, pw = max(0, crop - H0), max(0, crop - W0)
    if ph or pw:
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
        ext = None if ext is None else np.pad(ext, ((0, ph), (0, pw)))
    H, W = rgb.shape[:2]
    img = torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    em = None if ext is None else torch.from_numpy(ext.astype(np.float32))[None]
    ch = {"fp": 1, "mat": N, "cls": 2 * N}
    acc = {h: torch.zeros(ch[h], H, W) for h in heads}
    cnt = torch.zeros(1, H, W)
    ys = sorted(set(range(0, H - crop + 1, stride)) | {H - crop})
    xs = sorted(set(range(0, W - crop + 1, stride)) | {W - crop})
    for y in ys:
        for x in xs:
            if ext is not None and not ext[y:y + crop, x:x + crop].any():
                continue
            xi = img[None, :, y:y + crop, x:x + crop].to(dev)
            mi = None if em is None else em[None, :, y:y + crop, x:x + crop].to(dev)
            o = window_forward(model, xi, mi, size, thr, tta, heads)
            for h in heads:
                acc[h][:, y:y + crop, x:x + crop] += o[h]
            cnt[:, y:y + crop, x:x + crop] += 1
    res = {h: (acc[h] / cnt.clamp(min=1))[:, :H0, :W0] for h in heads}
    if "fp" in res:
        res["fp"] = res["fp"][0]
    return res


def pool_by_instance(dense, inst):
    """dense (C,H,W) float, inst (H,W) int ids -> per-id mean (K+1,C), pixel counts (K+1)."""
    flat = inst.ravel()
    K = int(flat.max())
    cnt = np.bincount(flat, minlength=K + 1).astype(np.float64)
    out = np.stack([np.bincount(flat, weights=dense[c].ravel(), minlength=K + 1) for c in range(dense.shape[0])], 1)
    return out / np.maximum(cnt[:, None], 1), cnt


# =============================================================== evaluation
@torch.no_grad()
def evaluate(model, tiles, dev, cfg, heads, have_fp, min_px):
    thrs = [.3, .4, .5, .6, .7]
    fps = np.zeros((len(thrs), 3), np.int64)  # tp, fp, fn per threshold
    cm_o = np.zeros((N, N), np.int64)
    rows, store = [], []
    for t in tiles:
        rgb, lab, foot, fpt = t["rgb"], t["lab"], t["foot"], t["fpt"]
        oh = tuple(h for h in ("mat", "cls") if h in heads)
        o = None
        if t["has_mat"] and oh:  # ORACLE footprint: isolates material / building-type quality
            o = predict_all(model, rgb, foot > 0, dev, cfg, oh)
            if "mat" in heads:
                cm_o += confusion(o["mat"].argmax(0).numpy(), lab, N)
        if t["has_mat"] and "cls" in heads:
            inst, _ = ndi.label(foot)
            flat, K = inst.ravel(), int(inst.max())
            if K:
                cnt = np.bincount(flat, minlength=K + 1)
                lc = np.stack([np.bincount(flat[lab.ravel() == c], minlength=K + 1) for c in range(N)], 1)
                lp = lc.sum(1)
                keep = (cnt >= min_px) & (lp >= MIN_LABELLED * cnt); keep[0] = False
                pooled, _ = pool_by_instance(o["cls"].numpy(), inst)
                sc = None
                if "mat" in heads:
                    sp = o["mat"].argmax(0).numpy().ravel()
                    sc = np.stack([np.bincount(flat[sp == c], minlength=K + 1) for c in range(N)], 1)
                for k in np.where(keep)[0]:
                    fr = lc[k] / lp[k]
                    rows.append(dict(gd=int(fr.argmax()), gp=fr >= FRAC_PRESENT, cd=int(pooled[k, :N].argmax()),
                                     cp=pooled[k, N:] > 0, sd=None if sc is None else int(sc[k].argmax())))
        if have_fp and t["has_fp"] and ("fp" in heads or "mat" in heads):  # END-TO-END: predicted footprint
            o2 = predict_all(model, rgb, None, dev, cfg, ("fp", "mat") if "mat" in heads else ("fp",))
            v = fpt != IGNORE
            for i, th in enumerate(thrs):
                pr = (o2["fp"].numpy() > th) & v; g = (fpt == 1) & v
                fps[i] += [(pr & g).sum(), (pr & ~g).sum(), (~pr & g).sum()]
            store.append((o2["fp"].numpy().astype(np.float16), o2["mat"].argmax(0).numpy().astype(np.uint8)
                          if "mat" in o2 and t["has_mat"] else None, lab))
    res, scores = {}, []
    if have_fp and fps.sum():
        iou = fps[:, 0] / np.maximum(fps.sum(1), 1)
        bi = int(iou.argmax()); res["fp_thr"], res["fp_iou"] = thrs[bi], float(iou[bi])
        res["fp_f1"] = float(2 * fps[bi, 0] / max(2 * fps[bi, 0] + fps[bi, 1] + fps[bi, 2], 1))
        if "fp" in heads:
            scores.append(res["fp_iou"])
    if "mat" in heads and cm_o.sum():
        io = np.diag(cm_o) / np.maximum(cm_o.sum(0) + cm_o.sum(1) - np.diag(cm_o), 1)
        pres = cm_o.sum(1) > 0
        res["mat_iou"] = io; res["mat_present"] = pres; res["mat_miou"] = float(io[pres].mean())
        scores.append(res["mat_miou"])
        if store and "fp_thr" in res:  # end-to-end: pixels outside the predicted footprint count as missed
            cm_e = np.zeros((N, N + 1), np.int64)
            for fpp, ma, lb in store:
                if ma is None: continue
                pred = np.where(fpp.astype(np.float32) > res["fp_thr"], ma, N)
                g = lb != IGNORE
                cm_e += np.bincount(lb[g].astype(np.int64) * (N + 1) + pred[g], minlength=N * (N + 1)).reshape(N, N + 1)
            d = np.diag(cm_e[:, :N]); ie = d / np.maximum(cm_e.sum(1) + cm_e[:, :N].sum(0) - d, 1)
            res["e2e_miou"] = float(ie[cm_e.sum(1) > 0].mean())
    if "cls" in heads and rows:
        gd = np.array([r["gd"] for r in rows]); cd = np.array([r["cd"] for r in rows])
        res["cls_acc"], res["n_buildings"] = float((gd == cd).mean()), len(rows)
        if rows[0]["sd"] is not None:
            res["seg_acc"] = float((gd == np.array([r["sd"] for r in rows])).mean())
        gp, cp = np.stack([r["gp"] for r in rows]), np.stack([r["cp"] for r in rows])
        f1 = [2 * (gp[:, c] & cp[:, c]).sum() / max(gp[:, c].sum() + cp[:, c].sum(), 1) for c in range(N) if gp[:, c].any()]
        res["present_f1"] = float(np.mean(f1)) if f1 else 0.0
        scores.append(res["cls_acc"])
    res["score"] = float(np.mean(scores)) if scores else -1.0
    return res


def fmt(res):
    s = []
    if "fp_iou" in res: s.append(f"fp IoU={res['fp_iou']:.3f} F1={res['fp_f1']:.3f} @thr={res['fp_thr']}")
    if "mat_miou" in res:
        s.append(f"mat mIoU(oracle fp)={res['mat_miou']:.3f} " + " ".join(
            f"{NAMES[i]}={res['mat_iou'][i]:.2f}" + ("" if res["mat_present"][i] else "(absent)") for i in range(N)))
    if "e2e_miou" in res: s.append(f"mat mIoU(predicted fp)={res['e2e_miou']:.3f}")
    if "cls_acc" in res:
        s.append(f"building cls acc={res['cls_acc']:.3f}"
                 + (f" (seg-vote acc={res['seg_acc']:.3f})" if "seg_acc" in res else "")
                 + f" present-F1={res['present_f1']:.3f} n={res['n_buildings']}")
    return " | ".join(s)


# =============================================================== commands
def cmd_train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    heads = tuple(h for h in a.train_heads.split(",") if h in HEADS)
    assert heads, f"--train_heads must contain some of {HEADS}"
    cfg = {"backbone": a.backbone, "layers": [2, 5, 8, 11], "dim": 128, "crop": a.crop,
           "input_size": a.input_size, "fp_thr": 0.5, "min_inst_px": a.min_inst_px}
    tr, va = split(load_tiles(a.coco, a.images, a.footprints), a.val_every)
    print(f"tiles: train={len(tr)} val={len(va)} | with roof labels: {sum(t['has_mat'] for t in tr)} "
          f"| with footprint targets: {sum(t['has_fp'] for t in tr)}")
    mat_tiles = [t for t in tr if t["has_mat"]]
    if ("mat" in heads or "cls" in heads) and not mat_tiles:
        raise SystemExit("no tiles with roof-material labels in the training split")
    cnt = np.bincount(np.concatenate([t["lab"].ravel() for t in mat_tiles]), minlength=256)[:N].astype(float) \
        if mat_tiles else np.ones(N)
    print("labelled roof pixels (train):", {NAMES[i]: int(cnt[i]) for i in range(N)})
    w = 1 / np.sqrt(cnt / cnt.sum() + 1e-6); w = np.clip(w / w.mean(), .2, 5)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=wt, ignore_index=IGNORE)

    model = RoofNet(a.backbone, cfg["layers"], cfg["dim"]).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        res = model.load_state_dict(ck["state"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        print(f"warm start from {a.init} | heads not in --train_heads stay frozen/unchanged")
    for h in HEADS:
        for p in getattr(model, h).parameters():
            p.requires_grad = h in heads
    have_fp = "fp" in heads or bool(a.init)

    tl = DataLoader(CropDS(tr, a.crop, a.n_per_tile), a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    lw = {"fp": a.w_fp, "mat": a.w_mat, "cls": a.w_cls}

    def save(extra=None):
        c = dict(cfg, **(extra or {}))
        torch.save({"state": {k: v for k, v in model.state_dict().items() if not k.startswith("bb.")}, "cfg": c}, a.out)

    best = -1e9
    for ep in range(a.epochs):
        model.train()
        tot = {h: 0.0 for h in heads}
        for x, m, y, fpt, inst in tl:
            x, m, y, fpt, inst = (v.to(dev) for v in (x, m, y, fpt, inst))
            m = jitter_mask(m)
            xi = F.interpolate(x, size=a.input_size, mode="bilinear", align_corners=False)
            mi = F.interpolate(m, size=a.input_size, mode="nearest")
            with torch.autocast(dev, enabled=dev == "cuda"):
                out = model(xi, mi, heads)  # ONE backbone pass feeds every head
            up = lambda v: F.interpolate(v.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            L = {}
            if "fp" in heads: L["fp"] = fp_loss(up(out["fp"]), fpt)
            if "mat" in heads: L["mat"] = mat_loss(up(out["mat"]), y, crit, a.dice)
            if "cls" in heads: L["cls"] = cls_loss(out["cls"].float(), y, inst, wt, a.min_inst_px)
            loss = sum(lw[h] * v for h, v in L.items())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            for h, v in L.items():
                tot[h] += float(v.detach())
        msg = f"ep{ep + 1:3d} " + " ".join(f"{h}={tot[h] / len(tl):.4f}" for h in heads)
        if va and ((ep + 1) % a.eval_every == 0 or ep == a.epochs - 1):
            r = evaluate(model, va, dev, cfg, heads, have_fp, a.min_inst_px)
            msg += " || " + fmt(r)
            if r["score"] > best:
                best = r["score"]; save({"fp_thr": r.get("fp_thr", 0.5)}); msg += "  *saved*"
        print(msg, flush=True)
    if not va:
        save(); print("no validation split: saved last epoch ->", a.out)
    else:
        print(f"best val score {best:.3f} -> {a.out} (stores the best footprint threshold)")
        print("NOTE: 'oracle fp' scores use label-exact footprints; 'predicted fp' is the realistic one.")


def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = RoofNet(cfg["backbone"], cfg["layers"], cfg["dim"]).to(dev)
    res = model.load_state_dict(ck["state"], strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("bb.")]
    assert not bad and not res.unexpected_keys, f"checkpoint mismatch: {bad[:5]} {res.unexpected_keys[:5]}"
    thr = a.fp_head_thr if a.fp_head_thr is not None else cfg.get("fp_thr", 0.5)
    img = Path(a.image)
    paths = sorted(p for p in img.iterdir() if p.suffix.lower() in IMG_EXT) if img.is_dir() else [img]
    rows, coco = [], new_coco()
    for p in paths:
        print("tile", p.name, flush=True)
        rgb, meta = read_image(p)
        H, W = rgb.shape[:2]
        if a.footprints:  # external polygons (Google Open Buildings, fAIr, ...)
            inst, info = load_instances(find_footprint(a.footprints, p.stem), (H, W), meta, a)
            if not (inst > 0).any():
                print("  no buildings"); continue
            o = predict_all(model, rgb, inst > 0, dev, cfg, HEADS if a.qa_fp else ("mat", "cls"), thr, a.tta)
        else:  # imagery only: footprint head -> threshold -> instances
            o = predict_all(model, rgb, None, dev, cfg, HEADS, thr, a.tta)
            cc, _ = ndi.label(o["fp"].numpy() > thr)
            sizes = np.bincount(cc.ravel())
            cc[np.isin(cc, [i for i in range(1, len(sizes)) if sizes[i] < a.min_building_px])] = 0
            inst, info = cc.astype(np.int32), {}
            if not (inst > 0).any():
                print("  no buildings"); continue
        foot = inst > 0
        lab = o["mat"].argmax(0).numpy().astype(np.uint8); lab[~foot] = IGNORE
        clean_instances(lab, inst, N, a.min_frag_px, a.min_frag_frac)
        seg = building_stats(lab, inst, N)
        pooled, _ = pool_by_instance(o["cls"].numpy(), inst)
        fpc = None
        if "fp" in o:
            fpc, _ = pool_by_instance(o["fp"].numpy()[None], inst)
        extra = {}
        for bid, st in seg.items():
            pd = np.exp(pooled[bid, :N] - pooled[bid, :N].max()); pd /= pd.sum()
            dom = int(pd.argmax())
            pres = [NAMES[i] for i in range(N) if 1 / (1 + np.exp(-pooled[bid, N + i])) >= 0.5]
            ex = {"cls_dominant": NAMES[dom], "cls_confidence": round(float(pd[dom]), 3), "cls_present": pres,
                  "cls_agrees_with_segmentation": NAMES[dom] == st["dominant"]}
            if fpc is not None:
                ex["fp_confidence"] = round(float(fpc[bid, 0]), 3)
            ex["needs_review"] = (not ex["cls_agrees_with_segmentation"]) or ex["cls_confidence"] < a.review_conf \
                or (fpc is not None and ex["fp_confidence"] < a.review_conf)
            extra[bid] = ex
        stats, items = export(a.out_dir, p.stem, rgb, lab, inst, meta, info, a.simplify_px, extra)
        coco_add(coco, p.name, W, H, items)
        rows += [dict(tile=p.name, building_id=b, area_px=s["area_px"], seg_dominant=s["dominant"],
                      n_materials=s["n_materials"], **{f"frac_{k}": v for k, v in s["fractions"].items()}, **extra[b])
                 for b, s in stats.items()]
        mixed = sum(s["n_materials"] > 1 for s in stats.values())
        review = sum(extra[b]["needs_review"] for b in stats)
        print(f"  {len(stats)} buildings | {mixed} mixed-roof | {review} flagged for review")
    out = Path(a.out_dir)
    if coco["images"]:
        json.dump(coco, open(out / "predictions_coco.json", "w"))
    if rows:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        plot_summary(out, rows)
        print(f"summary.csv + material_summary.png + predictions_coco.json -> {out} ({len(rows)} buildings)")


def cmd_inspect(a):
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    print("top-level keys:", list(ck.keys())[:20])
    sd = ck.get("state_dict", ck)
    pref = {}
    for k, v in sd.items():
        pref.setdefault(".".join(k.split(".")[:2]), []).append((k, tuple(v.shape)))
    for p, items in pref.items():
        print(f"{p:45s} {len(items):4d} tensors   e.g. {items[0][0]} {items[0][1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--coco", required=True); t.add_argument("--images", required=True)
    t.add_argument("--footprints", help="folder of <tile_stem>.{geojson,png,tif} footprint targets for the fp head")
    t.add_argument("--backbone", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    t.add_argument("--train_heads", default="fp,mat,cls", help="comma list; other heads stay frozen")
    t.add_argument("--init", help="warm start from a previous roofnet.pt (add/retrain individual heads)")
    t.add_argument("--out", default="roofnet.pt")
    t.add_argument("--epochs", type=int, default=80); t.add_argument("--bs", type=int, default=16)
    t.add_argument("--n_per_tile", type=int, default=12); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--w_fp", type=float, default=1.0); t.add_argument("--w_mat", type=float, default=1.0)
    t.add_argument("--w_cls", type=float, default=0.5); t.add_argument("--dice", type=float, default=.5)
    t.add_argument("--crop", type=int, default=256); t.add_argument("--input_size", type=int, default=512)
    t.add_argument("--min_inst_px", type=int, default=300, help="smallest building (px, inside a crop) used by cls head")
    t.add_argument("--val_every", type=int, default=5, help="block-split period; 0 = no validation")
    t.add_argument("--eval_every", type=int, default=5); t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    p = sp.add_parser("predict")
    p.add_argument("--image", required=True); p.add_argument("--weights", required=True)
    p.add_argument("--out_dir", default="roofnet_out")
    p.add_argument("--footprints", help="optional external polygons/mask/folder; omitted -> footprint head is used")
    p.add_argument("--qa_fp", action="store_true", help="with --footprints: also run the fp head -> fp_confidence per building")
    p.add_argument("--fp_head_thr", type=float, help="footprint-head threshold (default: best one found in validation)")
    p.add_argument("--footprints_crs", default="EPSG:4326"); p.add_argument("--fp_threshold", type=float, default=0)
    p.add_argument("--min_building_px", type=int, default=30)
    p.add_argument("--min_frag_px", type=int, default=25); p.add_argument("--min_frag_frac", type=float, default=.02)
    p.add_argument("--simplify_px", type=float, default=0.7)
    p.add_argument("--review_conf", type=float, default=0.6, help="below this a building is flagged needs_review")
    p.add_argument("--tta", action="store_true")
    p.set_defaults(fn=cmd_predict)
    i = sp.add_parser("inspect"); i.add_argument("--ckpt", required=True); i.set_defaults(fn=cmd_inspect)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()