"""Microsoft CvT-13 with the repository encoder interface."""

import torch
import torch.nn as nn
import transformers
from transformers import CvtConfig, CvtForImageClassification

from .layers import DropPath


CVT_TAG = "microsoft/cvt-13"
CVT_REVISION = "84e365a5f6a5ca987486abb25f3d8e5265cdc44d"


class CvTBlock(nn.Module):
    """CvT residual branches with per-block stochastic depth."""

    def __init__(self, block: nn.Module, drop_path: float) -> None:
        super().__init__()
        self.attention = block.attention
        self.intermediate = block.intermediate
        self.output = block.output
        self.layernorm_before = block.layernorm_before
        self.layernorm_after = block.layernorm_after
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.layernorm_before(x), height, width))
        residual = self.intermediate(self.layernorm_after(x))
        residual = self.output.dropout(self.output.dense(residual))
        return x + self.drop_path(residual)


class CvT(nn.Module):
    """CvT-13 with optional pinned ImageNet-1K weights."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pretrained: bool = False,
        drop_path: float = 0.1,
        head_std: float = 0.02,
    ) -> None:
        super().__init__()
        if pretrained and in_channels != 3:
            raise ValueError("CvT ImageNet weights require three input channels")
        if pretrained:
            self.model, loading = CvtForImageClassification.from_pretrained(
                CVT_TAG,
                revision=CVT_REVISION,
                use_safetensors=True,
                output_loading_info=True,
            )
            if any(loading.values()):
                raise ValueError(f"incomplete CvT checkpoint: {loading}")
        else:
            self.model = CvtForImageClassification(
                CvtConfig(num_channels=in_channels, num_labels=num_classes)
            )
            # Microsoft leaves convolution kernels at PyTorch's default init.
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    module.reset_parameters()

        # Apply stochastic depth independently to both residual branches.
        for index, stage in enumerate(self.model.cvt.encoder.stages):
            rate = drop_path if index == 2 else 0.0
            rates = torch.linspace(0, rate, len(stage.layers)).tolist()
            stage.layers = nn.Sequential(
                *[
                    CvTBlock(block, probability)
                    for block, probability in zip(stage.layers, rates, strict=True)
                ]
            )
        self.model.config.drop_path_rate = [0.0, 0.0, drop_path]
        self.embed_dim = 384
        self.model.classifier = nn.Linear(self.embed_dim, num_classes)
        self.model.num_labels = num_classes
        self.model.config.num_labels = num_classes
        nn.init.trunc_normal_(self.head.weight, std=head_std)
        nn.init.zeros_(self.head.bias)
        self.provenance = {
            "architecture": "CvT-13",
            "source": "https://github.com/microsoft/CvT",
            "library": "transformers",
            "library_version": transformers.__version__,
            "model_tag": CVT_TAG,
            "weight_revision": CVT_REVISION,
            "weight_license": "apache-2.0",
            "pretrained": pretrained,
            "drop_path": "per-block rates on both residual branches",
        }

    @property
    def head(self) -> nn.Linear:
        return self.model.classifier

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.model.cvt.encoder.stages:
            x, cls_token = stage(x)
        return cls_token

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.layernorm(self.forward_features(x)).mean(dim=1)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.model.layernorm(x).mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
