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
    parser.add_argument("--image", required=True, help="input image or directory, including GeoTIFF")
    parser.add_argument("--out_dir", default="inference")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def infer(model, image_path, out_dir, input_size, threshold, device):
    image = read_image(Path(image_path))
    original_height, original_width = image.shape[:2]
    tensor = torch.from_numpy(
        np.ascontiguousarray((image.astype(np.float32) / 255 - MEAN) / STD)
    ).permute(2, 0, 1)[None]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
    ).to(device)
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
    overlay = image.astype(np.float32)
    overlay[mask > 0] = overlay[mask > 0] * 0.45 + np.array([255, 40, 40]) * 0.55
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Building mask (threshold={threshold:g})")
    axes[2].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[2].set_title("Building overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(stem)
    figure.tight_layout()
    figure.savefig(output / f"{stem}_inference.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    np.save(output / f"{stem}_building_probability.npy", probability)
    return output


def main():
    args = parse_args()
    image_path = Path(args.image)
    if image_path.is_dir():
        extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        image_paths = sorted(path for path in image_path.iterdir() if path.suffix.lower() in extensions)
    else:
        image_paths = [image_path]
    if not image_paths:
        raise SystemExit(f"no supported images found in {image_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained(args.checkpoint, device)
    for path in image_paths:
        output = infer(model, path, args.out_dir, args.input_size, args.threshold, device)
        print(f"saved inference results for {path.name}: {output}", flush=True)


if __name__ == "__main__":
    main()
