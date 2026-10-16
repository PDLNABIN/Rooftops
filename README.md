# HOTOSM DINOv3 Building Fine-Tuning

Fine-tune the pretrained HOTOSM DINOv3 ViT-S/16 building-segmentation model.
The DINOv3 backbone stays frozen while the pretrained TerraTorch UPerNet
decoder and segmentation head are updated on local building labels.

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

## Project layout

| Path | Purpose |
| --- | --- |
| `hotosm_finetune.py` | HOTOSM checkpoint loading and fine-tuning entry point |
| `hotosm_data.py` | Building-only COCO/footprint data loading and loss |
| `requirements.txt` | Python dependencies |

## Fine-tuning

Fine-tune the pretrained UPerNet from
[`hotosm/dinov3s-buildings`](https://huggingface.co/hotosm/dinov3s-buildings).
Download its `model.ckpt` file separately; it is intentionally not included in
this repository. TerraTorch reconstructs the checkpoint's UPerNet decoder and
learned feature pyramid, while every DINOv3 parameter remains frozen:

```bash
python hotosm_finetune.py \
  --checkpoint path/to/model.ckpt \
  --coco path/to/instances_default.json \
  --images path/to/images \
  --footprints path/to/footprint_targets \
  --out dinov3s-buildings-finetuned.ckpt
```

The footprint targets can be a directory containing one
`<tile_stem>.{geojson,json,png,tif}` file per tile. If `--footprints` is
omitted, the union of roof polygons in the COCO annotations is used for
annotated tiles. Building-only fine-tuning does not require material labels.
The checkpoint is loaded strictly, so a mismatched architecture fails loudly
instead of silently starting a new decoder.

## Notes

Model checkpoints and datasets are not included in this repository. Make sure
you have permission to use and redistribute any imagery, annotations, model
weights, or derived geospatial outputs before publishing them.
