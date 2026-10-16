import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .constants import IGNORE, NAMES, PALETTE


def building_stats(lab, inst, n):
    from scipy import ndimage as ndi
    out = {}
    for bid, sl in enumerate(ndi.find_objects(inst), 1):
        if sl is None:
            continue
        cnt = np.bincount(lab[sl][inst[sl] == bid].astype(np.int64), minlength=n)[:n]
        tot = int(cnt.sum())
        out[bid] = {"area_px": tot,
                    "fractions": {NAMES[i]: round(float(cnt[i]) / tot, 4) for i in range(n)},
                    "dominant": NAMES[int(cnt.argmax())],
                    "n_materials": int((cnt / max(tot, 1) >= 0.02).sum())}
    return out


def render_overlay(rgb, lab, inst):
    n = len(NAMES)
    mask = np.full((*lab.shape, 3), 35, dtype=np.uint8)
    sel = inst > 0
    mask[sel] = PALETTE[lab[sel].clip(max=n - 1)]
    original = Image.fromarray(rgb)
    material = Image.fromarray(mask)
    legend_h = 58
    gap = 16
    panel = Image.new("RGB", (rgb.shape[1] * 2 + gap, rgb.shape[0] + legend_h), "white")
    panel.paste(original, (0, 0))
    panel.paste(material, (rgb.shape[1] + gap, 0))
    draw = ImageDraw.Draw(panel)
    draw.text((12, rgb.shape[0] + 8), "Original image", fill="black")
    draw.text((rgb.shape[1] + gap + 12, rgb.shape[0] + 8), "Predicted material mask", fill="black")
    x = 12
    y = rgb.shape[0] + 31
    for name, color in zip(NAMES, PALETTE):
        rgb_color = tuple(int(v) for v in color)
        draw.rectangle((x, y, x + 18, y + 18), fill=rgb_color, outline="black")
        draw.text((x + 24, y + 2), name, fill="black")
        x += 24 + draw.textlength(name) + 28
    return np.asarray(panel)


def export(out_dir, stem, rgb, lab, inst, meta):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = len(NAMES)
    # 1) material raster (255 = outside buildings)
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
    # 2) side-by-side original and material mask with legend
    Image.fromarray(render_overlay(rgb, lab, inst)).save(out_dir / f"{stem}_overlay.png")
    # 3) vectors: per-building stats + per-material parts (GeoJSON)
    stats = building_stats(lab, inst, n)
    try:
        from rasterio import features, warp
        from affine import Affine
    except ImportError:
        print("  [warn] rasterio missing -> vector export skipped"); return stats
    georef = meta["crs"] is not None
    tf = meta["transform"] if georef else Affine.identity()
    px_m2 = None
    if georef and meta["crs"].is_projected and str(meta["crs"].linear_units).lower().startswith("met"):
        px_m2 = abs(tf.a * tf.e)
    to_wgs = (lambda g: warp.transform_geom(meta["crs"], "EPSG:4326", g)) if georef else (lambda g: g)

    b_feats = []
    for geom, v in features.shapes(inst, mask=inst > 0, transform=tf):
        pr = dict(stats[int(v)], building_id=int(v))
        if px_m2:
            pr["area_m2"] = round(pr["area_px"] * px_m2, 2)
        b_feats.append({"type": "Feature", "geometry": to_wgs(geom), "properties": pr})
    code = (inst.astype(np.int64) * 8 + np.where(inst > 0, lab, 0)).astype(np.int32)  # building id x material
    p_feats = []
    for geom, v in features.shapes(code, mask=inst > 0, transform=tf):
        bid, mat = int(v) // 8, int(v) % 8
        p_feats.append({"type": "Feature", "geometry": to_wgs(geom),
                        "properties": {"building_id": bid, "material": NAMES[mat]}})
    for name, fs in (("buildings", b_feats), ("roof_parts", p_feats)):
        json.dump({"type": "FeatureCollection", "features": fs}, open(out_dir / f"{stem}_{name}.geojson", "w"))
    return stats
