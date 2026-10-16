#!/usr/bin/env python3
import numpy as np

"""
roofseg.py - roof-MATERIAL segmentation inside building footprints (mixed roofs per building supported).

  Backbone : DINOv3 ViT-S/16 (HF transformers, same family as hotosm/dinov3s-buildings), frozen by default,
             optional unfreezing of the last N blocks (--unfreeze_last).
  Decoder  : multi-layer ViT features (stride 16) + a small full-resolution CNN stem that sees RGB + the
             FOOTPRINT MASK (stride 4) -> sharper material boundaries inside one roof.
  Classes  : 0 CGI | 1 Concrete | 2 Tarpaulin | 3 Tile   (no background; 'Mud' -> ignored)
  Training : loss ONLY on annotated roof pixels (union of your COCO polygons = footprint). The footprint
             mask is randomly dilated/eroded during training so the model tolerates noisy predicted footprints.
  Inference: each building (instance) gets its own per-pixel material map; tiny fragments are merged into the
             neighbouring material; outputs = material raster, per-building stats + per-part polygons (GeoJSON).

Install : pip install torch "transformers>=4.56" numpy pillow scipy rasterio
          (DINOv3 weights are gated on Hugging Face: accept the licence and `huggingface-cli login`)

Train   : python Roff.py train --coco instances_default.json --images tiles/ --out roof_seg.pt
Predict : python roofseg.py predict --image tiles/ --footprints footprints.geojson --weights roof_seg.pt --out_dir out/
          (--footprints: GeoJSON of building polygons, OR a binary mask raster, OR a folder with <tile_stem>.{geojson,png,tif})
"""
IGNORE = 255
NAMES = ["CGI", "Concrete", "Tarpaulin", "Tile"]
NAME2IDX = {"cgi": 0, "concrete": 1, "tarpaulin": 2, "tile": 3, "mud": IGNORE}  # Mud: 1 example -> ignored
# High-contrast, color-vision-friendly RGB colors for material overlays.
PALETTE = np.array([(0, 114, 178), (230, 159, 0), (204, 121, 167), (0, 158, 115)], np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
