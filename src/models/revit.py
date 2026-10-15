"""C4 REViT-v2 with centered downsampling and unpadded window attention."""

from collections.abc import Sequence

import escnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn.nn import FieldType, R2Conv, init

from .convnext_c4 import C4_ORDER, C4Readout
from .layers import DropPath


REVIT_REVISION = "255896277843043a727957d4cbe8a9dbf989d2e3"


class SteerableConv(nn.Module):
    """ESCNN's trainable basis expansion without GeometricTensor or eval caches."""

    def __init__(
        self, in_type: FieldType, out_type: FieldType, kernel_size: int
    ) -> None:
        super().__init__()
        conv = R2Conv(in_type, out_type, kernel_size, bias=False, initialize=False)
        with torch.no_grad():
            init.deltaorthonormal_init(conv.weights, conv.basisexpansion)
        # All fields in each type have the same representation. Keep one
        # sampled basis, independent of ESCNN's global module/device cache.
        (expansion,) = conv.basisexpansion.children()
        self.register_buffer("basis", expansion.sampled_basis.detach().clone())
        # Matrix axes include these convolution coefficients in weight decay.
        self.weight = nn.Parameter(
            conv.weights.detach().reshape(len(out_type), len(in_type), -1)
        )
        self.kernel_shape = (out_type.size, in_type.size, kernel_size, kernel_size)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = torch.einsum("doip,abd->aobip", self.basis, self.weight)
        kernel = kernel.reshape(self.kernel_shape)
        return F.conv2d(x, kernel, padding=self.padding)


class RegularNorm(nn.Module):
    """ESCNN InnerBatchNorm for contiguous regular C4 fields."""

    def __init__(self, fields: int) -> None:
        super().__init__()
        self.fields = fields
        self.norm = nn.BatchNorm3d(fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x = self.norm(x.reshape(batch, self.fields, C4_ORDER, height, width))
        return x.reshape(batch, channels, height, width)


class ReViTDownsample(nn.Sequential):
    """Center stride-two sampling between pixels, including on even grids."""

    def __init__(
        self, in_type: FieldType, out_type: FieldType, kernel_size: int
    ) -> None:
        super().__init__(
            SteerableConv(in_type, out_type, kernel_size),
            nn.AvgPool2d(2, stride=2),
            RegularNorm(len(out_type)),
            nn.ReLU(),
        )


class ReViTAttention(nn.Module):
    """Window attention with complete regular fields in each attention head."""

    def __init__(
        self,
        field: FieldType,
        heads: int,
        window_size: int,
        qkv_kernel: int,
    ) -> None:
        super().__init__()
        if heads < 1 or len(field) % heads:
            raise ValueError("attention heads must divide the regular-field count")
        self.heads = heads
        self.window_size = window_size
        self.head_dim = field.size // heads
        self.query = SteerableConv(field, field, qkv_kernel)
        self.key = SteerableConv(field, field, qkv_kernel)
        self.value = SteerableConv(field, field, qkv_kernel)
        self.project = SteerableConv(field, field, 1)

    def partition(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        size = self.window_size
        x = x.reshape(batch, channels, height // size, size, width // size, size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(-1, self.heads, self.head_dim, size * size).transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        size = self.window_size
        query = self.partition(self.query(x)).contiguous()
        key = self.partition(self.key(x)).contiguous()
        value = self.partition(self.value(x)).contiguous()
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(-1, -2).reshape(
            batch, height // size, width // size, channels, size, size
        )
        output = output.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, channels, height, width
        )
        return self.project(output)


class ReViTBlock(nn.Module):
    def __init__(
        self,
        field: FieldType,
        heads: int,
        window_size: int,
        qkv_kernel: int,
        mlp_ratio: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        hidden = FieldType(field.gspace, mlp_ratio * field.representations)
        self.norm1 = RegularNorm(len(field))
        self.norm2 = RegularNorm(len(field))
        self.attention = ReViTAttention(field, heads, window_size, qkv_kernel)
        self.mlp = nn.Sequential(
            SteerableConv(field, hidden, 1),
            RegularNorm(len(hidden)),
            nn.ReLU(),
            SteerableConv(hidden, field, 1),
            RegularNorm(len(field)),
        )
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.alpha1 * self.attention(self.norm1(x)))
        return x + self.drop_path(self.alpha2 * self.mlp(self.norm2(x)))


class ReViTC4(nn.Module):
    """REViT-v2 with exact C4 sampling and an equivariant C4 readout.

    ``dims`` counts regular fields, each with four orientation channels.
    The embedding is equivariant; the free classification head is not invariant.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dims: Sequence[int] = (12, 24, 48, 96),
        depths: Sequence[int] = (1, 2, 3, 1),
        heads: Sequence[int] = (1, 2, 4, 8),
        window_size: int = 7,
        qkv_kernel: int = 3,
        mlp_ratio: int = 4,
        drop_path: float = 0.1,
        head_std: float = 0.02,
        readout: str = "equivariant",
    ) -> None:
        super().__init__()
        if not dims or not len(dims) == len(depths) == len(heads):
            raise ValueError("dims, depths and heads must have equal nonzero length")
        if min(*dims, *depths, window_size, mlp_ratio, qkv_kernel) < 1:
            raise ValueError("widths, depths, windows and kernels must be positive")
        if qkv_kernel % 2 != 1:
            raise ValueError("qkv_kernel must be odd")
        space = gspaces.rot2dOnR2(N=C4_ORDER)
        fields = [FieldType(space, dim * [space.regular_repr]) for dim in dims]
        image_type = FieldType(space, in_channels * [space.trivial_repr])
        middle = FieldType(space, max(1, dims[0] // 2) * [space.regular_repr])
        self.stem = nn.Sequential(
            ReViTDownsample(image_type, middle, 5),
            ReViTDownsample(middle, fields[0], 3),
        )
        rates = iter(torch.linspace(0, drop_path, sum(depths)).tolist())
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for index, (field, depth, count) in enumerate(
            zip(fields, depths, heads, strict=True)
        ):
            self.stages.append(
                nn.Sequential(
                    *[
                        ReViTBlock(
                            field,
                            count,
                            window_size,
                            qkv_kernel,
                            mlp_ratio,
                            next(rates),
                        )
                        for _ in range(depth)
                    ]
                )
            )
            if index < len(fields) - 1:
                self.downsample.append(ReViTDownsample(field, fields[index + 1], 3))
        self.input_divisor = 4 * 2 ** (len(dims) - 1) * window_size
        self.fields = dims[-1]
        self.readout = C4Readout(self.fields, kind=readout)
        self.embed_dim = self.readout.out_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=head_std)
        nn.init.zeros_(self.head.bias)
        self.provenance = {
            "architecture": "REViT-v2 C4 with centered sampling and project readout",
            "source": "https://github.com/kc-ml2/revit",
            "source_revision": REVIT_REVISION,
            "source_license": "apache-2.0",
            "library": "escnn",
            "library_version": escnn.__version__,
            "pretrained": False,
            "sampling": "stride-one convolution then 2x2 average pooling",
            "windows": "aligned, without padding",
            "initialization": "escnn deltaorthonormal_init",
        }

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if height != width or height % self.input_divisor:
            raise ValueError(
                f"REViT needs square inputs divisible by {self.input_divisor}"
            )
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsample):
                x = self.downsample[index](x)
        # ESCNN stores [field, orientation] with a positive cyclic action.
        # Export [orientation, field] with the project's negative cyclic action.
        x = x.reshape(x.shape[0], self.fields, C4_ORDER, *x.shape[-2:])
        return x[:, :, [0, 3, 2, 1]].transpose(1, 2)

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.forward_features(x))

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.readout(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
