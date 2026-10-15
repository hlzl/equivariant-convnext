import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DropPath


C4_ORDER = 4


def expand_circulant(weights: torch.Tensor) -> torch.Tensor:
    """Expand four relative-orientation weights into a circulant C4 kernel."""
    return torch.stack(
        [torch.roll(weights, output, dims=0) for output in range(C4_ORDER)],
        dim=0,
    )


class C4Spatial(nn.Module):
    """Depthwise spatial convolution shared by exact quarter-turn rotations."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = "same",
        spatial_std: float = math.sqrt(0.02),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size, kernel_size))
        self.register_buffer("inference_weight", None, persistent=False)
        nn.init.trunc_normal_(self.weight, std=spatial_std)

    def rotated_weight(self) -> torch.Tensor:
        if self.inference_weight is not None and not torch.is_grad_enabled():
            return self.inference_weight
        return torch.stack(
            [
                torch.rot90(self.weight, -turns, dims=(-2, -1))
                for turns in range(C4_ORDER)
            ],
            dim=0,
        ).flatten(0, 1)

    @torch.no_grad()
    def prepare_inference(self, dtype: torch.dtype | None) -> None:
        self.inference_weight = None
        if dtype is not None:
            self.inference_weight = self.rotated_weight().to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, order, channels, height, width = x.shape
        if order != C4_ORDER or channels != self.dim:
            raise ValueError(
                f"expected [B, {C4_ORDER}, {self.dim}, H, W], got {tuple(x.shape)}"
            )

        x = x.reshape(batch, C4_ORDER * self.dim, height, width)
        x = F.conv2d(
            x,
            self.rotated_weight(),
            stride=self.stride,
            padding=self.padding,
            groups=C4_ORDER * self.dim,
        )
        return x.reshape(batch, C4_ORDER, self.dim, *x.shape[-2:])


class C4Map(nn.Module):
    """Full circulant pointwise map over orientation and channel dimensions."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_std: float = 0.02,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(C4_ORDER, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.register_buffer("inference_matrix", None, persistent=False)
        self.register_buffer("inference_bias", None, persistent=False)
        nn.init.trunc_normal_(self.weight, std=weight_std)

    def matrix(self, dtype: torch.dtype) -> torch.Tensor:
        if self.inference_matrix is not None and not torch.is_grad_enabled():
            return self.inference_matrix
        kernel = expand_circulant(self.weight).to(dtype)
        return kernel.permute(0, 2, 1, 3).reshape(
            C4_ORDER * self.out_dim,
            C4_ORDER * self.in_dim,
        )

    def expanded_bias(self) -> torch.Tensor | None:
        if self.inference_bias is not None and not torch.is_grad_enabled():
            return self.inference_bias
        return self.bias.repeat(C4_ORDER) if self.bias is not None else None

    @torch.no_grad()
    def prepare_inference(self, dtype: torch.dtype | None) -> None:
        self.inference_matrix = None
        self.inference_bias = None
        if dtype is not None:
            self.inference_matrix = self.matrix(dtype)
            if self.bias is not None:
                self.inference_bias = self.bias.repeat(C4_ORDER).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, order, channels, height, width = x.shape
        if order != C4_ORDER or channels != self.in_dim:
            raise ValueError(
                f"expected [B, {C4_ORDER}, {self.in_dim}, H, W], got {tuple(x.shape)}"
            )

        matrix = self.matrix(x.dtype)
        bias = self.expanded_bias()
        x = x.permute(0, 3, 4, 1, 2).reshape(
            batch,
            height,
            width,
            C4_ORDER * self.in_dim,
        )
        x = x.reshape(-1, C4_ORDER * self.in_dim)
        x = F.linear(x, matrix, bias)
        x = x.reshape(batch, height, width, C4_ORDER, self.out_dim)
        return x.permute(0, 3, 4, 1, 2)


class C4Conv(nn.Module):
    """Factorized spatial and orientation convolution for stems and downsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        spatial_std: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.spatial = C4Spatial(
            in_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            spatial_std=spatial_std,
        )
        self.orient = C4Map(
            in_dim,
            out_dim,
            weight_std=spatial_std,
            bias=bias,
        )
        self.register_buffer("inference_weight", None, persistent=False)

    def dense_weight(self) -> torch.Tensor:
        """Merge the depthwise and orientation maps into one dense kernel."""
        kernel = self.spatial.rotated_weight()
        return self.orient.matrix(kernel.dtype)[:, :, None, None] * kernel.transpose(
            0, 1
        )

    @torch.no_grad()
    def prepare_inference(self, dtype: torch.dtype | None) -> None:
        # Build the dense kernel from the parameters, not from stale child caches.
        self.inference_weight = None
        self.spatial.prepare_inference(None)
        self.orient.prepare_inference(None)
        if dtype is not None:
            self.inference_weight = self.dense_weight().to(dtype)
            self.spatial.prepare_inference(dtype)
            self.orient.prepare_inference(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_weight is None or torch.is_grad_enabled():
            return self.orient(self.spatial(x))
        batch, _, _, height, width = x.shape
        x = F.conv2d(
            x.reshape(batch, -1, height, width),
            self.inference_weight,
            self.orient.expanded_bias(),
            stride=self.spatial.stride,
            padding=self.spatial.padding,
        )
        return x.reshape(batch, C4_ORDER, -1, *x.shape[-2:])


class FieldNorm(nn.Module):
    """Normalize each orientation field over channel and spatial dimensions."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"rms", "layer"}:
            raise ValueError(f"unknown field normalization: {mode}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "rms":
            return F.rms_norm(x, x.shape[2:], eps=1e-6)
        return F.layer_norm(x, x.shape[2:], eps=1e-6)


class C4GRN(nn.Module):
    """ConvNeXt-v2 response normalization for [B, O, H, W, C]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x, dim=(2, 3), keepdim=True)
        response = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return x + self.gamma * x * response + self.beta


class C4Readout(nn.Module):
    """Embedding from a regular C4 feature field.

    `equivariant` preserves the orientation blocks after spatial pooling.
    `invariant` combines the orientation mean, log amplitude, energy, and
    circular correlations.
    """

    def __init__(
        self,
        dim: int,
        kind: str = "invariant",
    ) -> None:
        super().__init__()
        if kind not in {"invariant", "equivariant"}:
            raise ValueError(f"unknown C4 readout: {kind}")

        self.dim = dim
        self.kind = kind
        self.out_dim = C4_ORDER * dim

    def norm_block(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
        return x.permute(0, 3, 1, 2)

    def equivariant_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(-2, -1))
        return F.layer_norm(x, x.shape[1:], eps=1e-6).flatten(1)

    def invariant_embed(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x.mean(dim=(-2, -1), keepdim=True)

        rms = x.square().mean(dim=(1, 2), keepdim=True).add(1e-6).sqrt()
        x = x / rms
        linear = self.norm_block(x.mean(dim=1))
        energy = x.square().mean(dim=1)
        lag_one = (x * torch.roll(x, 1, dims=1)).mean(dim=1)
        lag_two = (x * torch.roll(x, 2, dims=1)).mean(dim=1)
        amplitude = rms.squeeze(1).log()
        quadratic = torch.cat(
            (amplitude, energy[:, 1:], lag_one, lag_two),
            dim=1,
        )
        quadratic = self.norm_block(quadratic)
        x = torch.cat((linear, quadratic), dim=1).mean(dim=(-2, -1))
        return x.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != C4_ORDER or x.shape[2] != self.dim:
            raise ValueError(
                f"expected [B, {C4_ORDER}, {self.dim}, H, W], got {tuple(x.shape)}"
            )
        if self.kind == "invariant":
            return self.invariant_embed(x)
        return self.equivariant_embed(x)


class C4Block(nn.Module):
    """C4 spatial convolution with orientation mixing in the MLP expansion."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        spatial_std: float,
        drop_path: float,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.spatial = C4Spatial(
            dim,
            kernel_size,
            spatial_std=spatial_std,
        )
        self.norm = FieldNorm("rms")
        self.expand = C4Map(dim, expansion * dim)
        self.act = nn.GELU()
        self.grn = C4GRN(expansion * dim)
        self.project = nn.Linear(expansion * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        nn.init.trunc_normal_(self.project.weight, std=0.02)
        nn.init.zeros_(self.project.bias)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.spatial(residual))
        x = self.expand(x)
        x = x.permute(0, 1, 3, 4, 2)
        x = self.grn(self.act(x))
        shape = x.shape
        x = self.project(x.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)
        x = self.drop_path(x)
        return residual + x.permute(0, 1, 4, 2, 3)


class ConvNextC4(nn.Module):
    """ConvNeXt with exact C4-equivariant feature fields."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depths: Sequence[int] = (2, 4, 2),
        widths: Sequence[int] = (48, 96, 192),
        stem_stride: int = 1,
        kernel_size: int = 3,
        readout: str = "invariant",
        drop_path: float = 0.0,
        cache_weights: bool = True,
    ) -> None:
        super().__init__()
        if len(depths) != len(widths):
            raise ValueError("depths and widths must have equal length")
        if any(width % C4_ORDER for width in widths):
            raise ValueError("total widths must be divisible by four")
        if stem_stride not in {1, 2, 4}:
            raise ValueError("stem_stride must be 1, 2, or 4")

        dims = [width // C4_ORDER for width in widths]
        spatial_std = math.sqrt(0.02)
        self.stem = C4Conv(
            in_channels,
            dims[0],
            kernel_size=stem_stride,
            stride=stem_stride,
            padding=0,
            spatial_std=spatial_std,
            bias=False,
        )

        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        rate_index = 0
        self.stages = nn.ModuleList()
        for depth, dim in zip(depths, dims, strict=True):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    C4Block(
                        dim,
                        kernel_size,
                        spatial_std,
                        rates[rate_index],
                    )
                )
                rate_index += 1
            self.stages.append(nn.Sequential(*blocks))
        self.downsample = nn.ModuleList(
            [
                nn.Sequential(
                    FieldNorm("layer"),
                    C4Conv(
                        dims[index],
                        dims[index + 1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        spatial_std=spatial_std,
                    ),
                )
                for index in range(len(dims) - 1)
            ]
        )

        self.readout = C4Readout(
            dims[-1],
            kind=readout,
        )
        self.embed_dim = self.readout.out_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.cache_weights = cache_weights
        self.register_buffer("inference_stem", None, persistent=False)
        self.register_load_state_dict_post_hook(ConvNextC4.reset_inference)

    def reset_inference(self, _incompatible_keys=None) -> None:
        """Refresh transient weights after mode changes or checkpoint loads."""
        cached = self.cache_weights and not self.training
        self.cache_inference(self.head.weight.dtype if cached else None)

    @torch.no_grad()
    def cache_inference(self, dtype: torch.dtype | None) -> None:
        # The lifted image is identical in all orientations, so the stem sums them.
        self.stem.prepare_inference(None)
        self.inference_stem = None
        if dtype is not None:
            weight = self.stem.dense_weight().unflatten(1, (C4_ORDER, -1)).sum(1)
            self.inference_stem = weight.to(dtype)
        for module in self.modules():
            if isinstance(module, (C4Spatial, C4Map, C4Conv)):
                module.prepare_inference(dtype)

    def train(self, mode: bool = True) -> "ConvNextC4":
        super().train(mode)
        self.reset_inference()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_stem is None or torch.is_grad_enabled():
            x = self.stem(x.unsqueeze(1).repeat(1, C4_ORDER, 1, 1, 1))
        else:
            x = F.conv2d(
                x,
                self.inference_stem,
                self.stem.orient.expanded_bias(),
                stride=self.stem.spatial.stride,
                padding=self.stem.spatial.padding,
            ).unflatten(1, (C4_ORDER, -1))
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsample):
                x = self.downsample[index](x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.readout(x))

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
