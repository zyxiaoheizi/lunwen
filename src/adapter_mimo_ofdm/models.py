from __future__ import annotations

import torch
from torch import nn


class ResidualBlock2D(nn.Module):
    """轻量残差块，用于二维时频信道图像去噪/补全。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResidualGridCNN(nn.Module):
    """Offline CNN baseline: H_hat = H_LS + CNN(H_LS).

    输入/输出 shape: [batch, 2*rx*tx, ofdm_symbol, subcarrier]。
    对当前 2x2 MIMO 数据来说就是 [B, 8, 14, 72]。
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 64, depth: int = 6) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock2D(hidden_channels) for _ in range(depth)])
        self.tail = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.tail(self.body(self.head(x)))
        return x + residual


class SimpleGridCNN(nn.Module):
    """简易 CNN baseline：多层 3x3 卷积直接学习 LS 插值误差。

    这是论文里常见的 plain CNN / denoising CNN 对照组，比 ChannelNet
    简单很多，用来说明“不是随便一个 CNN 都能达到 ChannelNet/残差网络效果”。
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 64, depth: int = 5) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("simplecnn depth must be at least 3.")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ChannelNetStyleCNN(nn.Module):
    """ChannelNet-style baseline: SRCNN reconstruction + DnCNN denoising.

    参考 ChannelNet 论文代码中的思想：
    1) SRCNN 用大卷积核恢复完整信道图像；
    2) DnCNN 学习噪声/误差残差，再从 SRCNN 输出中减掉。

    这里将原论文的单通道实部/虚部图像扩展为 8 通道 MIMO 实虚拆分图像。
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 64, denoise_depth: int = 8) -> None:
        super().__init__()
        self.srcnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=5, padding=2),
        )

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(denoise_depth):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = self.srcnn(x)
        estimated_noise = self.dncnn(coarse)
        return coarse - estimated_noise


class SRCNNGrid(nn.Module):
    """ChannelNet/DeepPilotDesign 使用的 SRCNN 超分辨率模块。

    原始开源代码是单通道 Keras Conv2D: 9x9 -> 1x1 -> 5x5。
    这里扩展到 8 通道 MIMO 实虚拆分输入，用作论文级开源对齐 baseline。
    """

    def __init__(self, in_channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DnCNNGrid(nn.Module):
    """ChannelNet 使用的 DnCNN 图像恢复模块。

    DnCNN 学习残差噪声，然后输出 x - noise。单独训练时可作为去噪 baseline。
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 64, depth: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        estimated_noise = self.net(x)
        return x - estimated_noise


class ErrorRefinerCNN(nn.Module):
    """Residual error-field predictor used after a frozen base estimator.

    The refiner sees the interpolated LS grid, the base estimate, and the
    pilot mask. It only predicts the remaining structured error field.
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 64, depth: int = 6) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("ErrorRefinerCNN depth must be at least 1.")
        refiner_in_channels = 2 * in_channels + 1
        self.head = nn.Sequential(
            nn.Conv2d(refiner_in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock2D(hidden_channels) for _ in range(depth)])
        self.tail = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, ls_grid: torch.Tensor, base_estimate: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
        if pilot_mask.ndim == 2:
            pilot_mask = pilot_mask.view(1, 1, pilot_mask.shape[0], pilot_mask.shape[1])
        if pilot_mask.shape[0] == 1:
            pilot_mask = pilot_mask.expand(ls_grid.shape[0], -1, -1, -1)
        features = torch.cat((ls_grid, base_estimate, pilot_mask.to(dtype=ls_grid.dtype)), dim=1)
        return self.tail(self.body(self.head(features)))


class PilotLockedErrorRefinementNet(nn.Module):
    """Pilot-locked error refinement model.

    A pretrained base estimator first produces H_base. A small refiner then
    predicts E_hat, and the final estimate is H_base + E_hat.
    """

    def __init__(
        self,
        base_model: nn.Module,
        in_channels: int = 8,
        hidden_channels: int = 64,
        depth: int = 6,
        pilot_mask: torch.Tensor | None = None,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.refiner = ErrorRefinerCNN(in_channels=in_channels, hidden_channels=hidden_channels, depth=depth)
        if pilot_mask is None:
            pilot_mask = torch.zeros(1, 1, 1, 1)
        if pilot_mask.ndim == 2:
            pilot_mask = pilot_mask.view(1, 1, pilot_mask.shape[0], pilot_mask.shape[1])
        self.register_buffer("pilot_mask", pilot_mask.float())
        self.freeze_base = freeze_base
        if freeze_base:
            for parameter in self.base_model.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_base:
            with torch.no_grad():
                base_estimate = self.base_model(x)
        else:
            base_estimate = self.base_model(x)
        residual = self.refiner(x, base_estimate.detach() if self.freeze_base else base_estimate, self.pilot_mask)
        return base_estimate + residual


def build_model(name: str, in_channels: int = 8, hidden_channels: int = 64, depth: int = 6) -> nn.Module:
    normalized = name.lower()
    if normalized in {"simplecnn", "simple"}:
        return SimpleGridCNN(in_channels=in_channels, hidden_channels=hidden_channels, depth=depth)
    if normalized in {"rescnn", "reesnet"}:
        return ResidualGridCNN(in_channels=in_channels, hidden_channels=hidden_channels, depth=depth)
    if normalized == "srcnn":
        return SRCNNGrid(in_channels=in_channels)
    if normalized == "dncnn":
        return DnCNNGrid(in_channels=in_channels, hidden_channels=hidden_channels, depth=depth)
    if normalized == "channelnet":
        return ChannelNetStyleCNN(in_channels=in_channels, hidden_channels=hidden_channels, denoise_depth=depth)
    raise ValueError(
        f"Unknown model: {name}. Choose 'simplecnn', 'reesnet', 'rescnn', 'srcnn', 'dncnn', or 'channelnet'."
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
