import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .constants import IGNORE, MEAN, STD


class CropDS(Dataset):
    def __init__(self, tiles, crop, n_per_tile):
        self.t, self.c, self.n = tiles, crop, n_per_tile
        self.pts = [np.argwhere(t[2] != IGNORE)[::37] for t in tiles]  # labelled pixel candidates

    def __len__(self):
        return len(self.t) * self.n

    def __getitem__(self, k):
        i = k // self.n
        _, im, lab, foot = self.t[i]
        c, (H, W) = self.c, lab.shape
        py, px = self.pts[i][random.randrange(len(self.pts[i]))]  # crop around a labelled pixel (+ jitter)
        j = c // 4
        y = int(np.clip(py - c // 2 + random.randint(-j, j), 0, H - c))
        x = int(np.clip(px - c // 2 + random.randint(-j, j), 0, W - c))
        im = im[y:y + c, x:x + c].astype(np.float32) / 255
        lab, foot = lab[y:y + c, x:x + c], foot[y:y + c, x:x + c]
        if random.random() < .5: im, lab, foot = im[:, ::-1], lab[:, ::-1], foot[:, ::-1]
        if random.random() < .5: im, lab, foot = im[::-1], lab[::-1], foot[::-1]
        r = random.randint(0, 3)
        im, lab, foot = np.rot90(im, r), np.rot90(lab, r), np.rot90(foot, r)
        im = np.clip(im * random.uniform(.8, 1.2) + random.uniform(-.08, .08), 0, 1)
        im = (im - MEAN) / STD
        return (torch.from_numpy(im.transpose(2, 0, 1).copy()).float(),
                torch.from_numpy(foot.astype(np.float32).copy())[None],
                torch.from_numpy(lab.astype(np.int64).copy()))


def jitter_mask(m):
    """Random dilation/erosion (0-3 px) of the footprint channel: predicted footprints are never as tight as labels."""
    k = random.choice([0, 0, 1, 2, 3])
    if k == 0:
        return m
    if random.random() < .5:
        return F.max_pool2d(m, 2 * k + 1, 1, k)
    return 1 - F.max_pool2d(1 - m, 2 * k + 1, 1, k)
