# HOTOSM DINOv3 Building Fine-Tuning

Fine-tune the pretrained HOTOSM DINOv3 ViT-S/16 building-segmentation model.
The DINOv3 backbone stays frozen while the pretrained TerraTorch UPerNet
decoder and segmentation head are updated on local building labels.

The project then trains a separate rooftop-material classification head. The
classification head uses frozen DINOv3 features pooled inside building
footprints during training. During inference, it uses the building mask
predicted by the frozen UPerNet model and classifies each predicted building
region.

The two tasks are trained sequentially. They are never jointly optimized.

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

## Expected project files

Before training, arrange the local files as follows:

```text
rooftops/
├── config.yaml
├── model.ckpt
├── Images/
│   ├── tile_000001.tif
│   └── ...
├── annot/
│   └── instances_default.json
└── test_images/
    ├── tile_000002.tif
    └── ...
```

`model.ckpt` must be the pretrained checkpoint downloaded from
[`hotosm/dinov3s-buildings`](https://huggingface.co/hotosm/dinov3s-buildings).
It is not included in this repository.

The image names in `Images/` must match the `file_name` entries in the COCO
JSON. The COCO image width and height must also match the actual image
dimensions.

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

## Sequential training stages

Running the training command performs two independent stages.

### Stage 1: building segmentation

The pretrained HOTOSM building model is loaded strictly. The DINOv3 backbone
is frozen, while the learned feature pyramid, TerraTorch UPerNet decoder, and
main building segmentation head are fine-tuned:

```text
DINOv3 backbone             frozen
learned feature pyramid     trainable
UPerNet decoder             trainable
building segmentation head  trainable
```

The training target is the building footprint. If `--footprints` is supplied,
the external footprint target is used and combined with the COCO polygon mask.
If it is omitted, the union of COCO polygons is used.

The loss combines binary cross-entropy with a Dice loss. Training reports
loss, accuracy, precision, recall, F1, IoU, and learning rate. Validation
uses deterministic, augmentation-free crops.

### Stage 2: rooftop classification

After stage 1, the complete DINOv3/UPerNet model is frozen:

```text
DINOv3 backbone             frozen
learned feature pyramid     frozen
UPerNet decoder             frozen
building segmentation head  frozen
rooftop classification head trainable
```

The classifier receives frozen DINOv3 features and pools them only inside the
provided building footprint. COCO rooftop polygons provide the class labels.
The default classes are:

```text
CGI
Concrete
Mud
Tarpaulin
Tile
```

The classification head is:

```text
LayerNorm(384)
Linear(384, 256)
GELU
Dropout(0.2)
Linear(256, number_of_classes)
```

Runtime checks ensure that the segmentation model is fully frozen before the
classification stage and that only the classifier head is optimized.

## Configuration

All normal training and inference settings are in
[`config.yaml`](./config.yaml). Command-line arguments override values from
the YAML file.

### Training settings

```yaml
training:
  checkpoint: model.ckpt
  coco: annot/instances_default.json
  images: Images
  footprints:
  out: dinov3s-buildings-finetuned.ckpt
  epochs: 50
  bs: 1
  n_per_tile: 12
  crop: 256
  input_size: 256
  lr: 0.0001
  lr_factor: 0.5
  lr_patience: 3
  min_lr: 0.000001
  workers: 0
  val_every: 5
  seed: 0
  metrics_dir:
  classifier_epochs: 20
  classifier_lr: 0.001
  classifier_weight_decay: 0.01
```

Important settings:

- `checkpoint`: pretrained HOTOSM checkpoint for the first run.
- `coco`: COCO annotation file containing rooftop polygons and categories.
- `images`: directory containing the COCO training images.
- `footprints`: optional directory of external building masks.
- `epochs`: number of building-segmentation epochs.
- `bs`: batch size. Use `1` for an 11.6 GiB GPU.
- `n_per_tile`: number of random crops sampled from each training tile.
- `crop`: crop size before model resizing.
- `input_size`: resolution sent to DINOv3/UPerNet.
- `lr`: initial segmentation-stage learning rate.
- `lr_factor`: multiplier applied when validation loss stalls.
- `lr_patience`: number of stagnant validation epochs before reducing LR.
- `min_lr`: lower bound for the adaptive learning rate.
- `val_every`: validation split period. Set `0` to disable validation.
- `classifier_epochs`: rooftop-classifier training epochs.
- `classifier_lr`: classifier learning rate.
- `classifier_weight_decay`: classifier AdamW weight decay.

The segmentation optimizer uses `ReduceLROnPlateau`. The best validation
segmentation weights are restored before the classifier stage begins.

### Inference settings

```yaml
inference:
  checkpoint: dinov3s-buildings-finetuned.ckpt
  image: test_images
  out_dir: inference_all
  input_size: 256
  threshold: 0.5
```

- `checkpoint`: checkpoint produced after both training stages.
- `image`: one image or a directory of supported images.
- `out_dir`: inference output directory.
- `input_size`: model input resolution.
- `threshold`: building-mask probability threshold.

After the building decoder is fine-tuned, the pipeline freezes the complete
DINOv3/UPerNet model and trains a separate rooftop-material classifier. The
classifier uses masked average pooling of frozen DINOv3 features inside each
provided building footprint during training, while class samples are selected
from labeled COCO rooftop polygons. The included COCO categories are `CGI`,
`Concrete`, `Mud`, `Tarpaulin`, and `Tile`. Its settings are controlled by
`classifier_epochs`, `classifier_lr`, and `classifier_weight_decay`.
Only one stage is trainable at a time: the building segmentation stage runs
first, then its complete model is frozen before the rooftop classifier stage
starts. The stages are not jointly optimized.

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

For every detected connected building region, inference also:

1. Pools frozen DINOv3 features inside that predicted building region.
2. Runs the rooftop classification head.
3. Colors the region according to its predicted rooftop material.

The JSON file records the predicted class for every detected building
component. Small predicted components with fewer than 16 pixels are ignored.

## Outputs

### Training outputs

The default metrics directory is:

```text
dinov3s-buildings-finetuned_training/
```

It contains:

```text
training_metrics.json
training_metrics.csv
loss.png
iou.png
f1.png
learning_rate.png
```

The final checkpoint contains:

```text
state_dict                 fine-tuned DINOv3/UPerNet model
classification_head        rooftop classifier weights
classification_history     classifier training metrics
class_names                COCO class ID to class name mapping
hparams                    model architecture metadata
```

### Inference outputs

For an input such as `tile_017798.tif`, inference writes:

```text
tile_017798_inference.png
tile_017798_building_probability.npy
tile_017798_roof_classes.json
```

The composite PNG contains:

1. The original image.
2. The binary building mask.
3. The building overlay colored by predicted rooftop class.

The NumPy file contains the floating-point building probability map. The JSON
file contains predicted rooftop classes for connected building regions.

## Complete command sequence

From the repository root:

```bash
cd /home/coder/rooftops
source .venv/bin/activate
python -m pip install -e .
python -m rooftops --config config.yaml
python -m rooftops.infer --config config.yaml
```

Equivalent installed console commands:

```bash
rooftops-finetune --config config.yaml
rooftops-infer --config config.yaml
```

Override settings without editing YAML:

```bash
python -m rooftops \
  --config config.yaml \
  --epochs 100 \
  --bs 1 \
  --crop 224 \
  --input_size 224 \
  --lr 0.00005
```

Run inference with a different threshold:

```bash
python -m rooftops.infer \
  --config config.yaml \
  --threshold 0.7 \
  --out_dir inference_threshold_07
```

## GPU memory guidance

For an 11.6 GiB GPU, begin with:

```yaml
bs: 1
crop: 256
input_size: 256
workers: 0
```

If CUDA runs out of memory, reduce both spatial settings together:

```yaml
crop: 224
input_size: 224
```

You can also set the allocator option before starting Python:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Close other GPU applications before training. The frozen DINOv3 backbone is
run without gradient tracking, but it still requires memory for feature
extraction.

If the error reports only a few megabytes free, another process is already
using the GPU. Check the active processes before starting training:

```bash
nvidia-smi
```

Stop only an unrelated process shown by `nvidia-smi`, or run on CPU when no
GPU is available:

```bash
CUDA_VISIBLE_DEVICES="" python -m rooftops --config config.yaml
```

The classifier and inference paths use `torch.inference_mode()` and CUDA
automatic mixed precision. The training command also releases the CUDA cache
between the segmentation and classifier stages.

The best segmentation weights are copied to CPU memory when a validation
improvement is found, so checkpoint selection does not duplicate the model on
the GPU at the end of an epoch.

## Troubleshooting

### `ModuleNotFoundError: No module named 'rooftops'`

Activate the virtual environment and install the project:

```bash
source .venv/bin/activate
python -m pip install -e .
```

### `FileNotFoundError` for the checkpoint or dataset

Check the paths in `config.yaml`, or provide absolute paths:

```bash
python -m rooftops \
  --checkpoint /path/to/model.ckpt \
  --coco /path/to/instances_default.json \
  --images /path/to/Images
```

### CUDA out-of-memory

Reduce `crop` and `input_size` to `224` or lower, and keep `bs: 1`.

### BatchNorm error with batch size one

The training loop automatically freezes BatchNorm running statistics. No
manual change is required.

### Classifier checkpoint error during inference

The checkpoint must be generated by the current two-stage pipeline. A
checkpoint created before the rooftop classifier was added does not contain
`classification_head` or `class_names`; retrain it with:

```bash
python -m rooftops --config config.yaml
```

### Unsupported COCO segmentation

The data loader currently expects polygon segmentations. COCO RLE masks must
be exported as polygons before training.

## Notes

Model checkpoints and datasets are not included in this repository. Make sure
you have permission to use and redistribute any imagery, annotations, model
weights, or derived geospatial outputs before publishing them.
