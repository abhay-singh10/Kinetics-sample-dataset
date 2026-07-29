import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules.utils import _triple, _pair
from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
from fvcore.common.config import CfgNode as CN


# ==========================================
# Activation Layers & Utility Functions
# ==========================================

class Hardsigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (0.2 * x + 0.5).clamp(min=0.0, max=1.0)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def same_padding(
    x: torch.Tensor,
    in_height: int,
    in_width: int,
    stride_h: int,
    stride_w: int,
    filter_height: int,
    filter_width: int,
) -> torch.Tensor:
    if in_height % stride_h == 0:
        pad_along_height = max(filter_height - stride_h, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_h), 0)
    if in_width % stride_w == 0:
        pad_along_width = max(filter_width - stride_w, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_w), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


# ==========================================
# Streaming Buffer Core Modules
# ==========================================

class CausalModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = None

    def reset_activation(self) -> None:
        self.activation = None


class TemporalCGAvgPool3D(CausalModule):
    def __init__(self) -> None:
        super().__init__()
        self.n_cumulated_values = 0
        self.register_forward_hook(self._detach_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        device = x.device
        cumulative_sum = torch.cumsum(x, dim=2)
        if self.activation is None:
            self.activation = cumulative_sum[:, :, -1:].clone()
        else:
            cumulative_sum += self.activation
            self.activation = cumulative_sum[:, :, -1:].clone()

        divisor = torch.arange(
            1, input_shape[2] + 1, device=device
        )[None, None, :, None, None].expand(x.shape)
        x = cumulative_sum / (self.n_cumulated_values + divisor)
        self.n_cumulated_values += input_shape[2]
        return x

    @staticmethod
    def _detach_activation(module: "CausalModule", input: torch.Tensor, output: torch.Tensor) -> None:
        if module.activation is not None:
            module.activation.detach_()

    def reset_activation(self) -> None:
        super().reset_activation()
        self.n_cumulated_values = 0


# ==========================================
# Convolutional Blocks
# ==========================================

class Conv2dBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        norm_layer = norm_layer or nn.Identity
        activation_layer = activation_layer or nn.Identity

        dict_layers = OrderedDict({
            "conv2d": nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                **kwargs,
            ),
            "norm": norm_layer(out_planes, eps=0.001),
            "act": activation_layer(),
        })
        super().__init__(dict_layers)


class Conv3DBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        norm_layer = norm_layer or nn.Identity
        activation_layer = activation_layer or nn.Identity

        dict_layers = OrderedDict({
            "conv3d": nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                **kwargs,
            ),
            "norm": norm_layer(out_planes, eps=0.001),
            "act": activation_layer(),
        })
        super().__init__(dict_layers)


class ConvBlock3D(CausalModule):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int, int]],
        tf_like: bool,
        causal: bool,
        conv_type: str,
        padding: Union[int, Tuple[int, int, int]] = 0,
        stride: Union[int, Tuple[int, int, int]] = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.conv_2 = None

        if tf_like:
            if kernel_size[0] % 2 == 0:
                raise ValueError("tf_like supports only odd kernels for temporal dimension")
            padding = ((kernel_size[0] - 1) // 2, 0, 0)
            if stride[0] != 1:
                raise ValueError("illegal stride value, tf_like supports only stride == 1 for temporal dimension")

        if causal:
            padding = (0, padding[1], padding[2])

        if conv_type == "2plus1d":
            self.conv_1 = Conv2dBNActivation(
                in_planes,
                out_planes,
                kernel_size=(kernel_size[1], kernel_size[2]),
                padding=(padding[1], padding[2]),
                stride=(stride[1], stride[2]),
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                bias=bias,
                **kwargs,
            )
            if kernel_size[0] > 1:
                self.conv_2 = Conv2dBNActivation(
                    in_planes,
                    out_planes,
                    kernel_size=(kernel_size[0], 1),
                    padding=(padding[0], 0),
                    stride=(stride[0], 1),
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    bias=bias,
                    **kwargs,
                )
        elif conv_type == "3d":
            self.conv_1 = Conv3DBNActivation(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=padding,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                stride=stride,
                bias=bias,
                **kwargs,
            )

        self.padding = padding
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0] - 1
        self.stride = stride
        self.causal = causal
        self.conv_type = conv_type
        self.tf_like = tf_like

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if self.dim_pad > 0 and self.conv_2 is None and self.causal:
            x = self._cat_stream_buffer(x, device)
        shape_with_buffer = x.shape
        if self.conv_type == "2plus1d":
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.conv_1(x)
        if self.conv_type == "2plus1d":
            x = rearrange(x, "(b t) c h w -> b c t h w", t=shape_with_buffer[2])

            if self.conv_2 is not None:
                if self.dim_pad > 0 and self.causal:
                    x = self._cat_stream_buffer(x, device)
                w = x.shape[-1]
                x = rearrange(x, "b c t h w -> b c t (h w)")
                x = self.conv_2(x)
                x = rearrange(x, "b c t (h w) -> b c t h w", w=w)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tf_like:
            x = same_padding(
                x, x.shape[-2], x.shape[-1],
                self.stride[-2], self.stride[-1],
                self.kernel_size[-2], self.kernel_size[-1]
            )
        return self._forward(x)

    def _cat_stream_buffer(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.activation is None:
            self._setup_activation(x.shape)
        x = torch.cat((self.activation.to(device), x), 2)
        self._save_in_activation(x)
        return x

    def _save_in_activation(self, x: torch.Tensor) -> None:
        assert self.dim_pad > 0
        self.activation = x[:, :, -self.dim_pad:, ...].clone().detach()

    def _setup_activation(self, input_shape: Tuple[int, ...]) -> None:
        assert self.dim_pad > 0
        self.activation = torch.zeros(*input_shape[:2], self.dim_pad, *input_shape[3:])


# ==========================================
# Bottleneck & Squeeze-Excitation
# ==========================================

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels: int,
        activation_2: Callable[..., nn.Module],
        activation_1: Callable[..., nn.Module],
        conv_type: str,
        causal: bool,
        squeeze_factor: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.causal = causal
        se_multiplier = 2 if causal else 1
        squeeze_channels = _make_divisible(
            input_channels // squeeze_factor * se_multiplier, 8
        )
        self.temporal_cumualtive_GAvg3D = TemporalCGAvgPool3D()
        self.fc1 = ConvBlock3D(
            input_channels * se_multiplier,
            squeeze_channels,
            kernel_size=(1, 1, 1),
            padding=0,
            tf_like=False,
            causal=causal,
            conv_type=conv_type,
            bias=bias,
        )
        self.activation_1 = activation_1()
        self.activation_2 = activation_2()
        self.fc2 = ConvBlock3D(
            squeeze_channels,
            input_channels,
            kernel_size=(1, 1, 1),
            padding=0,
            tf_like=False,
            causal=causal,
            conv_type=conv_type,
            bias=bias,
        )

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x_space = torch.mean(input, dim=[3, 4], keepdim=True)
            scale = self.temporal_cumualtive_GAvg3D(x_space)
            scale = torch.cat((scale, x_space), dim=1)
        else:
            scale = F.adaptive_avg_pool3d(input, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._scale(input) * input


class tfAvgPool3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgf = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError("only same shape for h and w are supported by avg with tf_like")
        f1 = x.shape[-1] % 2 != 0
        padding_pad = (0, 0, 0, 0) if f1 else (0, 1, 0, 1)
        x = F.pad(x, padding_pad)
        if f1:
            x = F.avg_pool3d(x, (1, 3, 3), stride=(1, 2, 2), count_include_pad=False, padding=(0, 1, 1))
        else:
            x = self.avgf(x)
            x[..., -1] = x[..., -1] * 9 / 6
            x[..., -1, :] = x[..., -1, :] * 9 / 6
        return x


class BasicBneck(nn.Module):
    def __init__(
        self,
        cfg: CN,
        causal: bool,
        tf_like: bool,
        conv_type: str,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.res = None
        self.expand = None

        layers = []
        if cfg.expanded_channels != cfg.input_channels:
            self.expand = ConvBlock3D(
                in_planes=cfg.input_channels,
                out_planes=cfg.expanded_channels,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
                causal=causal,
                conv_type=conv_type,
                tf_like=tf_like,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

        self.deep = ConvBlock3D(
            in_planes=cfg.expanded_channels,
            out_planes=cfg.expanded_channels,
            kernel_size=cfg.kernel_size,
            padding=cfg.padding,
            stride=cfg.stride,
            groups=cfg.expanded_channels,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.se = SqueezeExcitation(
            cfg.expanded_channels,
            causal=causal,
            activation_1=activation_layer,
            activation_2=(nn.Sigmoid if conv_type == "3d" else Hardsigmoid),
            conv_type=conv_type,
        )

        self.project = ConvBlock3D(
            cfg.expanded_channels,
            cfg.out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=nn.Identity,
        )

        if not (cfg.stride == (1, 1, 1) and cfg.input_channels == cfg.out_channels):
            if cfg.stride != (1, 1, 1):
                if tf_like:
                    layers.append(tfAvgPool3D())
                else:
                    layers.append(nn.AvgPool3d((1, 3, 3), stride=cfg.stride, padding=cfg.padding_avg))
            layers.append(
                ConvBlock3D(
                    in_planes=cfg.input_channels,
                    out_planes=cfg.out_channels,
                    kernel_size=(1, 1, 1),
                    padding=(0, 0, 0),
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity,
                    causal=causal,
                    conv_type=conv_type,
                    tf_like=tf_like,
                )
            )
            self.res = nn.Sequential(*layers)

        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = self.res(input) if self.res is not None else input
        x = self.expand(input) if self.expand is not None else input
        x = self.deep(x)
        x = self.se(x)
        x = self.project(x)
        return residual + self.alpha * x


# ==========================================
# Main MoViNet & Local Weight Loader
# ==========================================

class MoViNet(nn.Module):
    def __init__(
        self,
        cfg: CN,
        causal: bool = True,
        load_weights_from_dir: bool = True,
        num_classes: int = 600,
        conv_type: str = "3d",
        tf_like: bool = False,
    ) -> None:
        super().__init__()
        
        # When loading pre-trained causal models, force tf_like and 2plus1d as expected
        if load_weights_from_dir:
            tf_like = True
            num_classes = 600
            conv_type = "2plus1d" if causal else "3d"

        blocks_dic = OrderedDict()
        norm_layer = nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d
        activation_layer = Swish if conv_type == "3d" else nn.Hardswish

        self.conv1 = ConvBlock3D(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            padding=cfg.conv1.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic[f"b{i}_l{j}"] = BasicBneck(
                    basicblock,
                    causal=causal,
                    conv_type=conv_type,
                    tf_like=tf_like,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
        self.blocks = nn.Sequential(blocks_dic)

        self.conv7 = ConvBlock3D(
            in_planes=cfg.conv7.input_channels,
            out_planes=cfg.conv7.out_channels,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.classifier = nn.Sequential(
            ConvBlock3D(
                cfg.conv7.out_channels,
                cfg.dense9.hidden_dim,
                kernel_size=(1, 1, 1),
                tf_like=tf_like,
                causal=causal,
                conv_type=conv_type,
                bias=True,
            ),
            Swish(),
            nn.Dropout(p=0.2, inplace=True),
            ConvBlock3D(
                cfg.dense9.hidden_dim,
                num_classes,
                kernel_size=(1, 1, 1),
                tf_like=tf_like,
                causal=causal,
                conv_type=conv_type,
                bias=True,
            ),
        )

        self.cgap = TemporalCGAvgPool3D() if causal else None
        self.causal = causal

        # Load weights from the current directory
        if load_weights_from_dir:
            weight_file = cfg.stream_filename if causal else cfg.filename
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(current_dir, weight_file)

            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"\n[Error] Weight file '{weight_file}' not found in directory:\n"
                    f"-> {current_dir}\n"
                    f"Please download it and save it as '{weight_file}' in the same directory."
                )

            print(f"Loading weights locally from: {weight_path}")
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def avg(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = F.adaptive_avg_pool3d(x, 1)
        return avg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.avg(x)
        x = self.classifier(x)
        return x.flatten(1)

    @staticmethod
    def _clean_activation_buffers(m: nn.Module) -> None:
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)


def get_movinet_a2_config() -> CN:
    """Helper to generate MoViNet-A2 configuration tree."""
    def fill_SE_config(conf, in_c, out_c, exp_c, k_size, stride, pad, pad_avg):
        conf.expanded_channels = exp_c
        conf.padding_avg = pad_avg
        conf.input_channels = in_c
        conf.out_channels = out_c
        conf.kernel_size = k_size
        conf.stride = stride
        conf.padding = pad

    def fill_conv(conf, in_c, out_c, k_size, stride, pad):
        conf.input_channels = in_c
        conf.out_channels = out_c
        conf.kernel_size = k_size
        conf.stride = stride
        conf.padding = pad

    A2 = CN()
    A2.name = "A2"
    # Local filenames expected in the directory
    A2.filename = "modelA2_statedict_v3.pth"
    A2.stream_filename = "modelA2_stream_statedict_v3.pth"

    A2.conv1 = CN()
    fill_conv(A2.conv1, 3, 16, (1, 3, 3), (1, 2, 2), (0, 1, 1))

    A2.blocks = [
        [CN() for _ in range(3)],
        [CN() for _ in range(5)],
        [CN() for _ in range(5)],
        [CN() for _ in range(6)],
        [CN() for _ in range(7)],
    ]

    # Block 1
    fill_SE_config(A2.blocks[0][0], 16, 16, 40, (1, 5, 5), (1, 2, 2), (0, 2, 2), (0, 1, 1))
    fill_SE_config(A2.blocks[0][1], 16, 16, 40, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[0][2], 16, 16, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

    # Block 2
    fill_SE_config(A2.blocks[1][0], 16, 40, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[1][1], 40, 40, 120, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[1][2], 40, 40, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[1][3], 40, 40, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[1][4], 40, 40, 120, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

    # Block 3
    fill_SE_config(A2.blocks[2][0], 40, 72, 240, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[2][1], 72, 72, 160, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[2][2], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[2][3], 72, 72, 192, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[2][4], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

    # Block 4
    fill_SE_config(A2.blocks[3][0], 72, 72, 240, (5, 3, 3), (1, 1, 1), (2, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[3][1], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[3][2], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[3][3], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[3][4], 72, 72, 144, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
    fill_SE_config(A2.blocks[3][5], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

    # Block 5
    fill_SE_config(A2.blocks[4][0], 72, 144, 480, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[4][1], 144, 144, 384, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
    fill_SE_config(A2.blocks[4][2], 144, 144, 384, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
    fill_SE_config(A2.blocks[4][3], 144, 144, 480, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
    fill_SE_config(A2.blocks[4][4], 144, 144, 480, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
    fill_SE_config(A2.blocks[4][5], 144, 144, 480, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
    fill_SE_config(A2.blocks[4][6], 144, 144, 576, (1, 3, 3), (1, 1, 1), (0, 1, 1), (0, 1, 1))

    A2.conv7 = CN()
    fill_conv(A2.conv7, 144, 640, (1, 1, 1), (1, 1, 1), (0, 0, 0))

    A2.dense9 = CN()
    A2.dense9.hidden_dim = 2048

    return A2


def build_movinet_a2_stream(load_weights: bool = True) -> MoViNet:
    """Instantiate MoViNet-A2 in Causal/Stream mode loading local weights."""
    cfg = get_movinet_a2_config()
    model = MoViNet(cfg, causal=True, load_weights_from_dir=load_weights)
    return model