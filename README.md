# Roof Material Segmentation

PyTorch tools for training and running roof-material segmentation models inside
building footprints. The project supports four roof-material classes:

- CGI
- Concrete
- Tarpaulin
- Tile

The segmentation pipeline uses a DINOv3 ViT backbone and can export material
rasters, per-building statistics, and GeoJSON roof-part polygons.

## Requirements

- Python 3.10 or newer
- A CUDA-capable GPU is recommended for training and inference
- DINOv3 model access through Hugging Face for training or first-time model use

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository intentionally excludes local imagery, annotations, checkpoints,
and generated outputs. Place those assets in your local working tree before
running the commands below.

## Training

Training expects COCO-format roof annotations and a directory of image tiles:

```bash
python Roff.py train \
  --coco path/to/instances_default.json \
  --images path/to/images \
  --out roof_seg.pt
```

Useful options include `--epochs`, `--batch-size` (exposed as `--bs`),
`--unfreeze_last`, `--crop`, and `--input_size`. Training charts and history
are written next to the checkpoint.

## Prediction

Prediction accepts a tile or image directory and building footprints supplied
as GeoJSON, a binary raster, or a directory of per-tile footprint files:

```bash
python Roff.py predict \
  --image path/to/images \
  --footprints path/to/footprints \
  --weights roof_seg.pt \
  --out_dir predictions
```

Use `--tta` to enable four-rotation test-time augmentation. Predictions
include material rasters, overlays, building statistics, and roof-part
polygons.

## Charts

To generate instance-count charts from COCO annotations:

```bash
python Roff.py charts \
  --coco path/to/instances_default.json \
  --out_dir charts
```

## Project layout

| Path | Purpose |
| --- | --- |
| `Roff.py` | Command-line entry point |
| `roofseg/` | Reusable data, model, training, inference, and export modules |
| `roof_net.py` | Standalone footprint/material training and inference pipeline |
| `requirements.txt` | Python dependencies |

## Two-stage training

`roof_net.py` keeps the DINOv3 backbone frozen. The footprint (`fp`) decoder
is trained independently from the material (`mat`) and building-level
classification (`cls`) heads. To first fine-tune building detection only,
train just the footprint UPerNet-style decoder:

```bash
python roof_net.py train \
  --coco path/to/instances_default.json \
  --images path/to/images \
  --footprints path/to/footprint_targets \
  --train_heads fp \
  --out roofnet_footprints.pt
```

The footprint targets can be a directory containing one
`<tile_stem>.{geojson,json,png,tif}` file per tile. If `--footprints` is
omitted, the union of roof polygons in the COCO annotations is used for
annotated tiles. Building-only training does not require material labels.
Only `RoofNet.fp` is passed to the optimizer; DINOv3 remains frozen and in
evaluation mode.

Afterwards, initialize a second stage from that checkpoint and train the
remaining heads as needed, for example:

```bash
python roof_net.py train \
  --coco path/to/instances_default.json \
  --images path/to/images \
  --init roofnet_footprints.pt \
  --train_heads mat,cls \
  --out roofnet.pt
```

## Notes

The DINOv3 backbone may require accepting the model license and authenticating
with Hugging Face before training:

```bash
huggingface-cli login
```

Model checkpoints and datasets are not included in this repository. Make sure
you have permission to use and redistribute any imagery, annotations, model
weights, or derived geospatial outputs before publishing them.
