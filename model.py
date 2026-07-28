import os
import torch
import torch.nn as nn
import torch.nn.functional as F

LOCAL_BASE_WEIGHTS = "weights/movinet_a2_kinetics600.pt"

# =====================================================================
# 1. CAUSAL CONVOLUTION WITH STREAM BUFFERING
# =====================================================================
class CausalConv3d(nn.Module):
    """
    Causal 3D convolution that applies padding ONLY to past frames
    and stores temporal hidden buffers for real-time frame streaming.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(0, padding[1], padding[2]),
            groups=groups, bias=bias
        )
        self.buffer = None

    def clean_activation_buffers(self):
        self.buffer = None

    def forward(self, x):
        pad_len = self.kernel_size[0] - 1

        if self.buffer is not None and self.buffer.shape[0] == x.shape[0]:
            x = torch.cat([self.buffer, x], dim=2)
        else:
            pad = torch.zeros(
                x.shape[0], x.shape[1], pad_len, x.shape[3], x.shape[4],
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([pad, x], dim=2)

        if pad_len > 0:
            self.buffer = x[:, :, -pad_len:].detach()

        return self.conv(x)


# =====================================================================
# 2. SQUEEZE-AND-EXCITATION (SE) MODULE
# =====================================================================
class TemporalSqueezeExcite(nn.Module):
    def __init__(self, in_channels, squeezed_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Conv3d(in_channels, squeezed_channels, kernel_size=1)
        self.fc2 = nn.Conv3d(squeezed_channels, in_channels, kernel_size=1)

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# =====================================================================
# 3. MOVINET INVERTED BOTTLENECK BLOCK
# =====================================================================
class MoViNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_channels, kernel_size, stride, se_ratio=0.25):
        super().__init__()
        stride_tuple = (stride, stride, stride) if isinstance(stride, int) else stride
        self.use_residual = (in_channels == out_channels and stride_tuple == (1, 1, 1))

        self.expand = nn.Sequential(
            nn.Conv3d(in_channels, exp_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(exp_channels),
            nn.SiLU()
        )

        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.depthwise = CausalConv3d(
            exp_channels, exp_channels, kernel_size,
            stride=stride, padding=padding, groups=exp_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm3d(exp_channels)
        self.act_dw = nn.SiLU()

        se_channels = max(1, int(in_channels * se_ratio))
        self.se = TemporalSqueezeExcite(exp_channels, se_channels)

        self.project = nn.Sequential(
            nn.Conv3d(exp_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        res = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)
        x = self.se(x)
        x = self.project(x)

        if self.use_residual:
            x = x + res
        return x


# =====================================================================
# 4. MOVINET-A2 FULL STREAM ARCHITECTURE
# =====================================================================
class MoViNetA2Stream(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            CausalConv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            # Stage 1
            MoViNetBlock(16, 16, exp_channels=40, kernel_size=(1, 3, 3), stride=1),
            MoViNetBlock(16, 16, exp_channels=40, kernel_size=(3, 3, 3), stride=1),
            # Stage 2
            MoViNetBlock(16, 40, exp_channels=96, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            MoViNetBlock(40, 40, exp_channels=120, kernel_size=(3, 3, 3), stride=1),
            MoViNetBlock(40, 40, exp_channels=120, kernel_size=(3, 3, 3), stride=1),
            # Stage 3
            MoViNetBlock(40, 72, exp_channels=240, kernel_size=(5, 3, 3), stride=(1, 2, 2)),
            MoViNetBlock(72, 72, exp_channels=240, kernel_size=(3, 3, 3), stride=1),
            MoViNetBlock(72, 72, exp_channels=240, kernel_size=(3, 3, 3), stride=1),
            # Stage 4
            MoViNetBlock(72, 140, exp_channels=480, kernel_size=(5, 3, 3), stride=(1, 2, 2)),
            MoViNetBlock(140, 140, exp_channels=480, kernel_size=(3, 3, 3), stride=1),
            MoViNetBlock(140, 140, exp_channels=480, kernel_size=(3, 3, 3), stride=1),
            # Stage 5
            MoViNetBlock(140, 320, exp_channels=960, kernel_size=(3, 3, 3), stride=1)
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(320, 640, kernel_size=1, bias=False),
            nn.BatchNorm3d(640),
            nn.SiLU()
        )

        self.dense = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Conv3d(640, 640, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(640, num_classes, kernel_size=1)
        )

    def clean_activation_buffers(self):
        for m in self.modules():
            if isinstance(m, CausalConv3d):
                m.clean_activation_buffers()

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.dense(x)
        x = self.classifier(x)
        return torch.flatten(x, start_dim=1)


# =====================================================================
# 5. BUILDER & LOADER HELPERS
# =====================================================================
def build_movinet_a2_stream(num_classes=3, weights_path=LOCAL_BASE_WEIGHTS, freeze_early_stages=True):
    model = MoViNetA2Stream(num_classes=600)

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded base Kinetics weights from '{weights_path}'")
    else:
        print("Notice: Base weights not found. Model initialized with random weights.")

    # Swap head to custom class count
    model.classifier[3] = nn.Conv3d(640, num_classes, kernel_size=1)

    # Freeze Stem and early stages
    if freeze_early_stages:
        for name, param in model.named_parameters():
            if "conv1" in name or "blocks.0" in name or "blocks.1" in name or "blocks.2" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model


def load_trained_weights(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    model.clean_activation_buffers()
    return model
