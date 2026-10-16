"""Run building-footprint inference with a fine-tuned HOTOSM model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .config import load_section
from .data import MEAN, STD, read_image
from .model import load_classification_head, load_pretrained


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--checkpoint")
    parser.add_argument("--image", help="input image or directory, including GeoTIFF")
    parser.add_argument("--out_dir")
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--threshold", type=float)
    parsed = vars(parser.parse_args())
    config = load_section(parsed["config"] or config_args.config, "inference")
    values = {"out_dir": "inference", "input_size": 256, "threshold": 0.5, **config}
    values.update({key: value for key, value in parsed.items() if value is not None})
    for name in ("checkpoint", "image"):
        if not values.get(name):
            parser.error(f"--{name} is required either on the command line or in the YAML file")
    return argparse.Namespace(**values)


def infer(model, classifier, class_names, image_path, out_dir, input_size, threshold, device):
    image = read_image(Path(image_path))
    original_height, original_width = image.shape[:2]
    tensor = torch.from_numpy(
        np.ascontiguousarray((image.astype(np.float32) / 255 - MEAN) / STD)
    ).permute(2, 0, 1)[None]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
    ).to(device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
        features = model.backbone_features(tensor)
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
    components, count = ndimage.label(mask > 0)
    colors = np.array([
        [230, 25, 75], [60, 180, 75], [255, 225, 25],
        [0, 130, 200], [245, 130, 48], [145, 30, 180],
    ])
    classifications = []
    for component_id in range(1, count + 1):
        component = components == component_id
        if component.sum() < 16:
            continue
        component_tensor = torch.from_numpy(component.astype(np.float32))[None].to(device)
        component_tensor = torch.nn.functional.interpolate(
            component_tensor[:, None], size=(input_size, input_size), mode="nearest"
        )[:, 0]
        class_id = int(classifier(features, component_tensor).argmax(1).item())
        class_name = class_names.get(class_id, str(class_id))
        color = colors[class_id % len(colors)]
        overlay[component] = overlay[component] * 0.35 + color * 0.65
        classifications.append({"component": component_id, "class": class_name})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Building mask (threshold={threshold:g})")
    axes[2].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    label_text = ", ".join(item["class"] for item in classifications[:5])
    suffix = "..." if len(classifications) > 5 else ""
    axes[2].set_title(f"Roof classes: {label_text}{suffix}" if label_text else "Building overlay")
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Building mask"),
    ]
    for class_id, class_name in sorted(class_names.items()):
        legend_handles.append(
            Patch(facecolor=colors[class_id % len(colors)] / 255, label=class_name)
        )
    figure.legend(handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 4),
                 frameon=True, bbox_to_anchor=(0.5, -0.04))
    for axis in axes:
        axis.axis("off")
    figure.suptitle(stem)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure.savefig(output / f"{stem}_inference.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    np.save(output / f"{stem}_building_probability.npy", probability)
    (output / f"{stem}_roof_classes.json").write_text(
        json.dumps(classifications, indent=2)
    )
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_names = checkpoint.get("class_names")
    if raw_names is None:
        raise SystemExit("checkpoint does not contain rooftop class names; retrain with the classifier head")
    class_names = {int(key): value for key, value in raw_names.items()}
    classifier = load_classification_head(args.checkpoint, device, len(class_names))
    for path in image_paths:
        output = infer(model, classifier, class_names, path, args.out_dir,
                       args.input_size, args.threshold, device)
        print(f"saved inference results for {path.name}: {output}", flush=True)


if __name__ == "__main__":
    main()
