"""Stock timm ConvNeXt-v2 Small with the shared encoder interface."""

import timm
import torch
from torch import nn


class ConvNext(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, drop_path: float = 0.0):
        super().__init__()
        self.model = timm.create_model(
            "convnextv2_small",
            pretrained=False,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_path_rate=drop_path,
        )
        self.embed_dim = self.model.num_features
        self.provenance = {
            "architecture": "convnextv2_small",
            "library": "timm",
            "library_version": timm.__version__,
            "pretrained": False,
        }

    @property
    def head(self) -> nn.Linear:
        return self.model.head.fc

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_head(self.forward_features(x), pre_logits=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
