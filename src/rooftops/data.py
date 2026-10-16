"""Minimal COCO/footprint data pipeline for HOTOSM building fine-tuning."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from torch.utils.data import Dataset

IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def read_image(path):
    try:
        import rasterio

        with rasterio.open(path) as source:
            image = source.read(list(range(1, min(3, source.count) + 1))).transpose(1, 2, 0)
    except ImportError:
        image = np.array(Image.open(path).convert("RGB"))
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.dtype == np.uint16:
        image = image / 257
    return np.clip(image, 0, 255).astype(np.uint8)


def read_mask(path):
    if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
        mask = np.array(Image.open(path))
    else:
        import rasterio

        with rasterio.open(path) as source:
            mask = source.read(1)
    return mask[..., 0] if mask.ndim == 3 else mask


def footprint_file(directory, stem):
    for extension in (".geojson", ".json", ".png", ".tif", ".tiff"):
        path = Path(directory) / f"{stem}{extension}"
        if path.exists():
            return path
    return None


def load_external_footprint(path, shape, threshold=0):
    if path.suffix.lower() in {".geojson", ".json"}:
        from rasterio import features
        from affine import Affine

        with path.open() as source:
            data = json.load(source)
        geometries = [
            feature["geometry"]
            for feature in data.get("features", [data])
            if feature.get("geometry")
        ]
        return features.rasterize(
            [(geometry, 1) for geometry in geometries],
            out_shape=shape,
            transform=Affine.identity(),
            fill=0,
            dtype="uint8",
        )
    mask = read_mask(path)
    if mask.shape != shape:
        raise ValueError(f"{path}: footprint {mask.shape} != image {shape}")
    return (mask > threshold).astype(np.uint8)


def load_tiles(coco_path, image_dir, footprint_dir=None):
    with Path(coco_path).open() as source:
        coco = json.load(source)
    category_ids = {
        category["id"]: index + 1
        for index, category in enumerate(sorted(coco.get("categories", []), key=lambda item: item["id"]))
    }
    category_names = {
        index: category["name"]
        for index, category in enumerate(sorted(coco.get("categories", []), key=lambda item: item["id"]))
    }
    annotations = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)

    tiles = []
    for image in sorted(coco["images"], key=lambda item: item["file_name"]):
        width, height = image["width"], image["height"]
        roof_mask = Image.new("L", (width, height), 0)
        class_mask = np.zeros((height, width), dtype=np.uint8)
        class_mask_image = Image.fromarray(class_mask)
        draw = ImageDraw.Draw(roof_mask)
        class_draw = ImageDraw.Draw(class_mask_image)
        for annotation in annotations.get(image["id"], []):
            if not isinstance(annotation["segmentation"], list):
                raise ValueError("RLE segmentation is not supported; export polygon annotations")
            for polygon in annotation["segmentation"]:
                points = list(zip(polygon[0::2], polygon[1::2]))
                draw.polygon(points, fill=1)
                class_draw.polygon(points, fill=category_ids.get(annotation["category_id"], 0))
        class_mask = np.array(class_mask_image, dtype=np.uint8)
        roof_mask = np.array(roof_mask, dtype=np.uint8)
        image_path = Path(image_dir) / Path(image["file_name"]).name
        rgb = read_image(image_path)
        if rgb.shape[:2] != (height, width):
            raise ValueError(f"{image_path}: image {rgb.shape[:2]} != COCO {(height, width)}")

        external = footprint_file(footprint_dir, Path(image["file_name"]).stem) if footprint_dir else None
        target = load_external_footprint(external, (height, width)) if external else roof_mask
        if external or roof_mask.any():
            tiles.append({
                "name": image["file_name"], "rgb": rgb,
                "target": np.maximum(target, roof_mask),
                "class_mask": class_mask,
                "class_names": category_names,
            })
    return tiles


def split(tiles, validation_period):
    if validation_period <= 0:
        return tiles, []
    train, validation = [], []
    for index, tile in enumerate(tiles):
        (validation if (index // 10) % validation_period == 0 else train).append(tile)
    return train, validation


class CropDataset(Dataset):
    def __init__(self, tiles, crop_size, crops_per_tile, augment=True):
        self.tiles = tiles
        self.crop_size = crop_size
        self.crops_per_tile = crops_per_tile
        self.augment = augment
        self.points = [np.argwhere(tile["target"] == 1)[::37] for tile in tiles]

    def __len__(self):
        return len(self.tiles) * self.crops_per_tile

    def __getitem__(self, index):
        tile = self.tiles[index // self.crops_per_tile]
        height, width = tile["target"].shape
        points = self.points[index // self.crops_per_tile]
        if len(points):
            point_index = (
                random.randrange(len(points))
                if self.augment
                else index % self.crops_per_tile % len(points)
            )
            py, px = points[point_index]
        else:
            if self.augment:
                py, px = random.randrange(height), random.randrange(width)
            else:
                py, px = height // 2, width // 2
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"crop size {self.crop_size} exceeds tile shape {(height, width)}")
        jitter = self.crop_size // 4 if self.augment else 0
        y = int(np.clip(py - self.crop_size // 2 + random.randint(-jitter, jitter), 0, height - self.crop_size))
        x = int(np.clip(px - self.crop_size // 2 + random.randint(-jitter, jitter), 0, width - self.crop_size))
        image = tile["rgb"][y:y + self.crop_size, x:x + self.crop_size].astype(np.float32) / 255
        target = tile["target"][y:y + self.crop_size, x:x + self.crop_size]
        if self.augment:
            if random.random() < 0.5:
                image, target = image[:, ::-1], target[:, ::-1]
            if random.random() < 0.5:
                image, target = image[::-1], target[::-1]
            rotation = random.randint(0, 3)
            image, target = np.rot90(image, rotation), np.rot90(target, rotation)
            image = np.clip(image * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
        image = (image - MEAN) / STD
        tensor = lambda value, dtype: torch.from_numpy(np.ascontiguousarray(value).astype(dtype))
        class_target = self.tile_class_target(tile, py, px)
        return (
            tensor(image.transpose(2, 0, 1), np.float32),
            tensor(target, np.int64),
            torch.tensor(class_target, dtype=torch.long),
        )

    @staticmethod
    def tile_class_target(tile, y, x):
        class_mask = tile["class_mask"]
        value = int(class_mask[y, x])
        if value:
            return value - 1
        nearby = class_mask[
            max(0, y - 3):y + 4,
            max(0, x - 3):x + 4,
        ]
        values, counts = np.unique(nearby[nearby > 0], return_counts=True)
        return int(values[np.argmax(counts)] - 1) if len(values) else 0


def footprint_loss(logits, target):
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0
    labels = target.clamp(max=1).float()
    bce = F.binary_cross_entropy_with_logits(logits[:, 0], labels, reduction="none")[valid].mean()
    probability = logits[:, 0].sigmoid() * valid
    truth = labels * valid
    dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    return bce + dice
