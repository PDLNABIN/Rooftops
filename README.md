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
| `src/rooftops/finetune.py` | CLI argument parsing and training orchestration |
| `src/rooftops/model.py` | DINOv3/UPerNet model and checkpoint loading |
| `src/rooftops/data.py` | Building-only COCO/footprint loading and augmentation |
| `src/rooftops/trainer.py` | Training and validation loop |
| `src/rooftops/metrics.py` | Metrics, CSV/JSON persistence, and plots |
| `src/rooftops/infer.py` | Building-mask inference and visualization |
| `src/rooftops/config.py` | YAML configuration loading |
| `config.yaml` | YAML controls for training and inference |
| `pyproject.toml` | Package metadata and console entry point |
| `requirements.txt` | Python dependencies |

## Fine-tuning

Fine-tune the pretrained UPerNet from
[`hotosm/dinov3s-buildings`](https://huggingface.co/hotosm/dinov3s-buildings).
Download its `model.ckpt` file separately; it is intentionally not included in
this repository. TerraTorch reconstructs the checkpoint's UPerNet decoder and
learned feature pyramid, while every DINOv3 parameter remains frozen:

```bash
python -m rooftops \
  --checkpoint path/to/model.ckpt \
  --coco path/to/instances_default.json \
  --images path/to/images \
  --footprints path/to/footprint_targets \
  --out dinov3s-buildings-finetuned.ckpt
```

After installing the project with `python -m pip install -e .`, the equivalent
console command is:

```bash
rooftops-finetune \
  --checkpoint path/to/model.ckpt \
  --coco path/to/instances_default.json \
  --images path/to/images \
  --out dinov3s-buildings-finetuned.ckpt
```

The memory-friendly defaults use one 256x256 crop per batch. The default
training horizon is 50 epochs. A validation-driven `ReduceLROnPlateau`
scheduler halves the learning rate after 3 epochs without improvement. These
values can be adjusted with `--lr_factor`, `--lr_patience`, and `--min_lr`.
The best validation checkpoint is retained in the output checkpoint. BatchNorm layers
use the pretrained running statistics so batch-size-one training is supported.
For an 11.6 GiB GPU, keep those defaults; if memory is still insufficient,
reduce the crop and decoder resolution together:

```bash
python -m rooftops \
  --checkpoint model.ckpt \
  --coco annot/instances_default.json \
  --images Images \
  --bs 1 \
  --crop 224 \
  --input_size 224 \
  --out dinov3s-buildings-finetuned.ckpt
```

Training displays a batch progress bar and writes metrics after every epoch.
Validation uses fixed, augmentation-free crops so validation metrics are
comparable between epochs; training crops remain randomized and augmented.
By default, artifacts are saved in
`dinov3s-buildings-finetuned_training/`; override this with `--metrics_dir`.
The directory contains:

- `training_metrics.json` and `training_metrics.csv`
- `loss.png`
- `iou.png`
- `f1.png`
- `learning_rate.png`

The footprint targets can be a directory containing one
`<tile_stem>.{geojson,json,png,tif}` file per tile. If `--footprints` is
omitted, the union of roof polygons in the COCO annotations is used for
annotated tiles. Building-only fine-tuning does not require material labels.
The checkpoint is loaded strictly, so a mismatched architecture fails loudly
instead of silently starting a new decoder.

After the building decoder is fine-tuned, the pipeline freezes the complete
DINOv3/UPerNet model and trains a separate rooftop-material classifier. The
classifier uses masked average pooling of frozen DINOv3 features inside each
COCO rooftop polygon. The included COCO categories are `CGI`, `Concrete`,
`Mud`, `Tarpaulin`, and `Tile`. Its settings are controlled by
`classifier_epochs`, `classifier_lr`, and `classifier_weight_decay`.

All training settings can be managed in [`config.yaml`](./config.yaml):

```bash
python -m rooftops --config config.yaml
```

This includes the checkpoint and dataset paths, batch size, crop and model
input sizes, epochs, optimizer settings, validation interval, workers, and
adaptive learning-rate settings. Any command-line option overrides the YAML
value, for example:

```bash
python -m rooftops --config config.yaml --bs 1 --epochs 100 --input_size 224
```

## Inference

Run inference on a test image after fine-tuning:

```bash
python -m rooftops.infer \
  --checkpoint dinov3s-buildings-finetuned.ckpt \
  --image test_images/example.tif \
  --out_dir inference
```

The inference output directory contains one labeled three-panel PNG for each
input image: the original image, binary building mask, and class-colored
building overlay. It also saves a NumPy probability map and a JSON list of
predicted rooftop classes for each connected building region. To process every supported
image in a test directory, pass the directory instead:

```bash
python -m rooftops.infer \
  --checkpoint dinov3s-buildings-finetuned.ckpt \
  --image test_images \
  --out_dir inference_all
```

Use `--threshold` to change the probability cutoff.

Inference can use the same YAML file:

```bash
python -m rooftops.infer --config config.yaml
```

The `inference` section controls the checkpoint, test image directory, output
directory, model input size, and mask threshold.

## Notes

Model checkpoints and datasets are not included in this repository. Make sure
you have permission to use and redistribute any imagery, annotations, model
weights, or derived geospatial outputs before publishing them.
