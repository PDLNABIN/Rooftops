"""Roof-material segmentation package."""

from .constants import IGNORE, IMG_EXT, MEAN, NAME2IDX, NAMES, PALETTE, STD
from .data import CropDS, jitter_mask
from .export import building_stats, export
from .inference import (
    clean_instances,
    find_footprint,
    load_instances,
    predict_probs,
)
from .io import load_tiles, read_image, read_mask, split
from .model import RoofSeg, dice_loss, model_forward

__all__ = [
    "CropDS",
    "IGNORE",
    "IMG_EXT",
    "MEAN",
    "NAME2IDX",
    "NAMES",
    "PALETTE",
    "RoofSeg",
    "STD",
    "building_stats",
    "clean_instances",
    "dice_loss",
    "export",
    "find_footprint",
    "jitter_mask",
    "load_instances",
    "load_tiles",
    "model_forward",
    "predict_probs",
    "read_image",
    "read_mask",
    "split",
]
