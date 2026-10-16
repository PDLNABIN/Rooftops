"""Command-line entry point for HOTOSM UPerNet fine-tuning."""
import argparse
from pathlib import Path

import torch

from .config import load_section
from .data import load_tiles, split
from .model import load_pretrained
from .trainer import train, train_rooftop_classifier


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--checkpoint", help="hotosm/dinov3s-buildings model.ckpt")
    parser.add_argument("--coco")
    parser.add_argument("--images")
    parser.add_argument("--footprints")
    parser.add_argument("--out")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--bs", type=int, help="reduce if CUDA runs out of memory")
    parser.add_argument("--n_per_tile", type=int)
    parser.add_argument("--crop", type=int)
    parser.add_argument("--input_size", type=int, help="decoder resolution; 256 is GPU-friendly")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_factor", type=float,
                        help="multiply the learning rate after validation stalls")
    parser.add_argument("--lr_patience", type=int,
                        help="epochs without validation improvement before reducing learning rate")
    parser.add_argument("--min_lr", type=float,
                        help="lower learning-rate bound")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--val_every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--metrics_dir")
    parser.add_argument("--classifier_epochs", type=int)
    parser.add_argument("--classifier_lr", type=float)
    parser.add_argument("--classifier_weight_decay", type=float)
    parsed = vars(parser.parse_args())
    config = load_section(parsed["config"] or config_args.config, "training")
    defaults = {
        "out": "dinov3s-buildings-finetuned.ckpt", "epochs": 50, "bs": 1,
        "n_per_tile": 12, "crop": 256, "input_size": 256, "lr": 1e-4,
        "lr_factor": 0.5, "lr_patience": 3, "min_lr": 1e-6, "workers": 0,
        "val_every": 5, "seed": 0,
        "classifier_epochs": 20, "classifier_lr": 1e-3,
        "classifier_weight_decay": 1e-2,
    }
    values = {**defaults, **config}
    values.update({key: value for key, value in parsed.items() if value is not None})
    for name in ("checkpoint", "coco", "images"):
        if not values.get(name):
            parser.error(f"--{name} is required either on the command line or in the YAML file")
    return argparse.Namespace(**values)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tiles, validation_tiles = split(load_tiles(args.coco, args.images, args.footprints), args.val_every)
    if not train_tiles or not any(tile["target"].any() for tile in train_tiles):
        raise SystemExit("no footprint targets found; pass --footprints or provide roof polygons in COCO")

    model = load_pretrained(args.checkpoint, device)
    train(model, train_tiles, validation_tiles, args, device)
    class_names = train_tiles[0].get("class_names", {})
    if not class_names:
        raise SystemExit("COCO categories are required to train the rooftop classifier")
    classifier, classifier_history = train_rooftop_classifier(
        model, train_tiles, validation_tiles, args, device, len(class_names)
    )
    torch.save(
        {"state_dict": model.state_dict(),
         "classification_head": classifier.state_dict(),
         "classification_history": classifier_history,
         "class_names": class_names,
         "hparams": {"seg_out_indices": (2, 5, 8, 11), "decoder_channels": 512}},
        args.out,
    )
    print("saved:", Path(args.out))
