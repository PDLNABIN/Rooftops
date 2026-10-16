"""HOTOSM DINOv3 and UPerNet model definition."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pyramid(nn.Module):
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
    def tokens_to_image(tokens, height, width):
        grid_h, grid_w = height // 16, width // 16
        tokens = tokens[:, -grid_h * grid_w:]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, grid_h, grid_w)

    def forward(self, features, height, width):
        images = [self.tokens_to_image(f, height, width) for f in features]
        return [self.fpn1(images[0]), self.fpn2(images[1]), self.fpn3(images[2]), self.fpn4(images[3])]


class HOTOSMNet(nn.Module):
    """Exact trainable structure represented by the HOTOSM checkpoint."""

    def __init__(self):
        super().__init__()
        from terratorch.models.decoders.upernet_decoder import UperNetDecoder

        self.net = nn.Module()
        self.net.backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vits16", pretrained=False)
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

    def freeze_batchnorm_stats(self):
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def forward(self, x):
        decoded = self.decoder_features(x)
        return F.interpolate(self.net.head(decoded), size=x.shape[-2:], mode="bilinear", align_corners=False)

    def backbone_features(self, x):
        with torch.no_grad():
            features = self.net.backbone.get_intermediate_layers(
                x, n=(2, 5, 8, 11), return_class_token=False
            )
        tokens = features[-1]
        height, width = x.shape[-2:]
        grid_h, grid_w = height // 16, width // 16
        return tokens[:, -grid_h * grid_w:].transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)

    def decoder_features(self, x):
        height, width = x.shape[-2:]
        with torch.no_grad():
            features = self.net.backbone.get_intermediate_layers(
                x, n=(2, 5, 8, 11), return_class_token=False
            )
        pyramidal = self.net.pyramid(features, height, width)
        return self.net.decoder(pyramidal)


class MaskedDINOClassifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.head = nn.Linear(384, classes)

    def forward(self, feature_map, building_mask):
        mask = F.interpolate(
            building_mask[:, None].float(), size=feature_map.shape[-2:], mode="nearest"
        )
        pooled = (feature_map * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1)
        return self.head(pooled)


def freeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()


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
    return model.to(device)


def load_classification_head(path, device, class_count):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("classification_head")
    if state is None:
        raise ValueError(f"{path} does not contain a rooftop classification head")
    classifier = MaskedDINOClassifier(class_count)
    classifier.load_state_dict(state, strict=True)
    return classifier.to(device)
