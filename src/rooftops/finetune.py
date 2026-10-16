"""Command-line entry point for HOTOSM UPerNet fine-tuning."""
import argparse
from pathlib import Path

import torch

from .data import load_tiles, split
from .model import load_pretrained
from .trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="hotosm/dinov3s-buildings model.ckpt")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--footprints", help="directory of per-tile footprint targets")
    parser.add_argument("--out", default="dinov3s-buildings-finetuned.ckpt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=1, help="reduce if CUDA runs out of memory")
    parser.add_argument("--n_per_tile", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=256, help="decoder resolution; 256 is GPU-friendly")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics_dir", help="directory for JSON, CSV, and PNG metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tiles, validation_tiles = split(load_tiles(args.coco, args.images, args.footprints), args.val_every)
    if not train_tiles or not any(tile["target"].any() for tile in train_tiles):
        raise SystemExit("no footprint targets found; pass --footprints or provide roof polygons in COCO")

    model = load_pretrained(args.checkpoint, device)
    train(model, train_tiles, validation_tiles, args, device)
    torch.save(
        {"state_dict": model.state_dict(),
         "hparams": {"seg_out_indices": (2, 5, 8, 11), "decoder_channels": 512}},
        args.out,
    )
    print("saved:", Path(args.out))
