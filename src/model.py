import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """A small convolutional network trained from scratch."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_efficientnet_b0(num_classes=10, pretrained=True):
    """EfficientNet-B0 with a fresh classification head for transfer learning."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes)
    return net


class CNNViTHybrid(nn.Module):
    """A hybrid that uses a small CNN to make patch tokens, then a Vision Transformer."""

    def __init__(self, num_classes=10, image_size=32, embed_dim=128, depth=4,
                 num_heads=4, patch=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=patch, stride=patch),
        )
        grid = image_size // patch
        self.num_tokens = grid * grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        b = x.size(0)
        x = self.stem(x)                       # (B, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)       # (B, num_tokens, embed_dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_model(config):
    """Construct the model selected in the config."""
    name = config["model"]["name"]
    num_classes = config["data"]["num_classes"]
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if name == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=config["model"].get("pretrained", True),
        )
    if name == "cnn_vit":
        return CNNViTHybrid(
            num_classes=num_classes,
            image_size=config["data"]["image_size"],
        )
    raise ValueError(f"Unknown model name: {name}")
