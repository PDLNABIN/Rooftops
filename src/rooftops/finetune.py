#!/usr/bin/env python3
"""Fine-tune the pretrained HOTOSM DINOv3 building UPerNet decoder."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CropDataset, footprint_loss, load_tiles, split


class Pyramid(nn.Module):
    """HOTOSM's learned four-level pyramid around DINOv3 token features."""

    def __init__(self, channels=384):
        super().__init__()
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, 2),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channels // 2, channels // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(channels, channels // 2, 2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.Sequential(nn.MaxPool2d(2, 2))

    @staticmethod
    def _tokens_to_image(tokens, height, width):
        grid_h, grid_w = height // 16, width // 16
        tokens = tokens[:, -grid_h * grid_w:]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, grid_h, grid_w)

    def forward(self, features, height, width):
        images = [self._tokens_to_image(f, height, width) for f in features]
        return [self.fpn1(images[0]), self.fpn2(images[1]), self.fpn3(images[2]), self.fpn4(images[3])]


class HOTOSMNet(nn.Module):
    """Exact trainable structure represented by hotosm/dinov3s-buildings/model.ckpt."""

    def __init__(self):
        super().__init__()
        from terratorch.models.decoders.upernet_decoder import UperNetDecoder

        self.net = nn.Module()
        self.net.backbone = torch.hub.load(
            "facebookresearch/dinov3", "dinov3_vits16", pretrained=False
        )
        self.net.pyramid = Pyramid()
        self.net.decoder = UperNetDecoder([96, 192, 384, 384], channels=512)
        self.net.head = nn.Conv2d(512, 3, 1)
        self.net.aux_head = nn.Sequential(
            nn.Conv2d(384, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 3, 1),
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        features = self.net.backbone.get_intermediate_layers(
            x, n=(2, 5, 8, 11), return_class_token=False
        )
        pyramidal = self.net.pyramid(features, height, width)
        decoded = self.net.decoder(pyramidal)
        return F.interpolate(self.net.head(decoded), size=(height, width), mode="bilinear", align_corners=False)


def load_pretrained(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a Lightning checkpoint with a state_dict")
    model = HOTOSMNet()
    result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {result}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.net.pyramid, model.net.decoder, model.net.head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.net.backbone.eval()
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="hotosm/dinov3s-buildings model.ckpt")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--footprints", help="directory of per-tile footprint targets")
    parser.add_argument("--out", default="dinov3s-buildings-finetuned.ckpt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--n_per_tile", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tiles, _ = split(load_tiles(args.coco, args.images, args.footprints), args.val_every)
    if not train_tiles or not any(tile["target"].any() for tile in train_tiles):
        raise SystemExit("no footprint targets found; pass --footprints or provide roof polygons in COCO")

    model = load_pretrained(args.checkpoint, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    allowed = ("net.pyramid.", "net.decoder.", "net.head.")
    if not trainable or any(
        not name.startswith(allowed) for name, p in model.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("only the pretrained building decoder path may be trainable")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    loader = DataLoader(
        CropDataset(train_tiles, args.crop, args.n_per_tile),
        args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    for epoch in range(args.epochs):
        model.train()
        model.net.backbone.eval()
        total = 0.0
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            x = F.interpolate(x, size=args.input_size, mode="bilinear", align_corners=False)
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(x)
                target = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
                loss = footprint_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1:3d} loss={total / max(len(loader), 1):.4f}", flush=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": {"seg_out_indices": (2, 5, 8, 11), "decoder_channels": 512},
        },
        args.out,
    )
    print("saved:", Path(args.out))


if __name__ == "__main__":
    main()
