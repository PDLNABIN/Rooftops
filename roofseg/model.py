import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import IGNORE, NAMES


class RoofSeg(nn.Module):
    def __init__(self, backbone, n_cls=len(NAMES), layers=(2, 5, 8, 11), dim=128, unfreeze_last=0):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(backbone)
        self.layers = tuple(layers)
        for p in self.bb.parameters():
            p.requires_grad = False
        if unfreeze_last:
            blocks = getattr(self.bb, "layer", None)
            if blocks is None:
                blocks = getattr(getattr(self.bb, "encoder", None), "layer", None)
            if blocks is None:
                raise RuntimeError("Could not locate transformer blocks; run with --unfreeze_last 0")
            for b in blocks[-unfreeze_last:]:
                for p in b.parameters():
                    p.requires_grad = True
        self.unfrozen = unfreeze_last > 0
        h = self.bb.config.hidden_size
        gn = lambda c: nn.GroupNorm(8, c)
        self.proj = nn.ModuleList(nn.Sequential(nn.Conv2d(h, dim, 1), gn(dim), nn.ReLU()) for _ in self.layers)
        self.fuse = nn.Sequential(nn.Conv2d(dim * len(self.layers), dim, 3, padding=1), gn(dim), nn.ReLU())
        self.stem = nn.Sequential(nn.Conv2d(4, 32, 3, 2, 1), gn(32), nn.ReLU(),     # RGB + footprint, stride 2
                                  nn.Conv2d(32, 64, 3, 2, 1), gn(64), nn.ReLU())    # stride 4
        self.dec = nn.Sequential(nn.Conv2d(dim + 64, dim, 3, padding=1), gn(dim), nn.ReLU(),
                                 nn.Conv2d(dim, dim, 3, padding=1), gn(dim), nn.ReLU())
        self.head = nn.Conv2d(dim, n_cls, 1)

    def forward(self, x, m):
        """x: B,3,H,W normalised RGB (H,W multiples of 16) | m: B,1,H,W footprint in {0,1} -> logits B,n,H,W"""
        B, _, H, W = x.shape
        gh, gw = H // 16, W // 16
        with torch.set_grad_enabled(self.unfrozen and torch.is_grad_enabled()):
            hs = self.bb(pixel_values=x, output_hidden_states=True).hidden_states
        f = [p(hs[l + 1][:, -gh * gw:].transpose(1, 2).reshape(B, -1, gh, gw))  # drop CLS + register tokens
             for p, l in zip(self.proj, self.layers)]
        y = self.fuse(torch.cat(f, 1))
        s = self.stem(torch.cat([x, m], 1))
        y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec(torch.cat([y, s], 1))
        return F.interpolate(self.head(y), size=(H, W), mode="bilinear", align_corners=False)


def model_forward(model, x, m, size):
    """crop-resolution in / crop-resolution logits out; the network itself runs at `size` (finer token grid)."""
    c = x.shape[-1]
    xi = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    mi = F.interpolate(m, size=size, mode="nearest")
    return F.interpolate(model(xi, mi), size=(c, c), mode="bilinear", align_corners=False)


def dice_loss(logits, y, n):
    valid = y != IGNORE
    p = logits.softmax(1) * valid[:, None]
    t = F.one_hot(torch.where(valid, y, torch.zeros_like(y)), n).permute(0, 3, 1, 2) * valid[:, None]
    inter, den = (p * t).sum((0, 2, 3)), p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    return (1 - (2 * inter + 1) / (den + 1))[present].mean()
