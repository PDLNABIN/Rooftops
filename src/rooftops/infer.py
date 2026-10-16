"""Run building-footprint inference with a fine-tuned HOTOSM model."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import MEAN, STD, read_image
from .model import load_pretrained


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="input image, including GeoTIFF")
    parser.add_argument("--out_dir", default="inference")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def infer(checkpoint, image_path, out_dir, input_size, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = read_image(Path(image_path))
    original_height, original_width = image.shape[:2]
    tensor = torch.from_numpy(
        np.ascontiguousarray((image.astype(np.float32) / 255 - MEAN) / STD)
    ).permute(2, 0, 1)[None]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
    ).to(device)
    model = load_pretrained(checkpoint, device)
    model.eval()
    with torch.no_grad():
        probability = model(tensor)[:, 0].sigmoid()
        probability = torch.nn.functional.interpolate(
            probability[:, None], size=(original_height, original_width),
            mode="bilinear", align_corners=False,
        )[0, 0].cpu().numpy()

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    mask = (probability >= threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(output / f"{stem}_building_mask.png")

    overlay = image.astype(np.float32)
    overlay[mask > 0] = overlay[mask > 0] * 0.45 + np.array([255, 40, 40]) * 0.55
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        output / f"{stem}_building_overlay.png"
    )
    np.save(output / f"{stem}_building_probability.npy", probability)
    return output


def main():
    args = parse_args()
    output = infer(args.checkpoint, args.image, args.out_dir, args.input_size, args.threshold)
    print("saved inference results:", output)


if __name__ == "__main__":
    main()
