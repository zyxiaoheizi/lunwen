from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class RelativePilotAttentionLayer(nn.Module):
    """Self-attention over pilot tokens with learned relative time-frequency bias."""

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        relative_feature_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads.")
        self.hidden_channels = int(hidden_channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_channels // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, 3 * hidden_channels)
        self.relative_bias = nn.Sequential(
            nn.Linear(relative_feature_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_heads),
        )
        self.out = nn.Linear(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_channels, hidden_channels),
        )
        self.last_attention_entropy: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, relative_features: torch.Tensor) -> torch.Tensor:
        batch, num_pilots, _ = tokens.shape
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized)
        qkv = qkv.view(batch, num_pilots, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        rel_bias = self.relative_bias(relative_features).permute(2, 0, 1).unsqueeze(0)
        scores = scores + rel_bias.to(device=tokens.device, dtype=tokens.dtype)
        attn = torch.softmax(scores, dim=-1)
        self.last_attention_entropy = -torch.sum(attn * torch.log(attn.clamp_min(1e-8)), dim=-1).mean()
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch, num_pilots, self.hidden_channels)
        tokens = tokens + self.out(context)
        return tokens + self.ffn(self.norm2(tokens))


class BasisPilotCrossAttention(nn.Module):
    """Let learned basis queries read the current pilot set before gating bases."""

    def __init__(self, hidden_channels: int, num_heads: int, num_basis: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads.")
        self.hidden_channels = int(hidden_channels)
        self.num_heads = int(num_heads)
        self.num_basis = int(num_basis)
        self.head_dim = self.hidden_channels // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)

        self.basis_queries = nn.Parameter(torch.empty(num_basis, hidden_channels))
        self.token_norm = nn.LayerNorm(hidden_channels)
        self.query_proj = nn.Linear(hidden_channels, hidden_channels)
        self.key_proj = nn.Linear(hidden_channels, hidden_channels)
        self.value_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, hidden_channels)
        self.gate_head = nn.Linear(hidden_channels, 1)
        self.last_attention_entropy: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.basis_queries, mean=0.0, std=self.hidden_channels**-0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_pilots, _ = tokens.shape
        tokens = self.token_norm(tokens)
        q = self.query_proj(self.basis_queries)
        q = q.view(self.num_basis, self.num_heads, self.head_dim).permute(1, 0, 2)
        q = q.unsqueeze(0).expand(batch, -1, -1, -1)

        k = self.key_proj(tokens).view(batch, num_pilots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(tokens).view(batch, num_pilots, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        self.last_attention_entropy = -torch.sum(attn * torch.log(attn.clamp_min(1e-8)), dim=-1).mean()
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch, self.num_basis, self.hidden_channels)
        context = self.out(context)
        return self.gate_head(context).squeeze(-1)


class MIMOLinkAttentionLayer(nn.Module):
    """Self-attention over Tx/Rx link tokens inside one pilot observation.

    The relative bias is generated from the pairwise Rx/Tx relation instead of
    from a generic token index. This keeps the token encoder aware that H11,
    H12, H21, and H22 are MIMO links sharing the same physical scatterers.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        pair_feature_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads.")
        self.hidden_channels = int(hidden_channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_channels // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, 3 * hidden_channels)
        self.link_bias = nn.Sequential(
            nn.Linear(pair_feature_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_heads),
        )
        self.out = nn.Linear(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_channels, hidden_channels),
        )
        self.last_attention_entropy: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, link_pair_features: torch.Tensor) -> torch.Tensor:
        batch_pilots, num_links, _ = tokens.shape
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized)
        qkv = qkv.view(batch_pilots, num_links, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        link_bias = self.link_bias(link_pair_features).permute(2, 0, 1).unsqueeze(0)
        scores = scores + link_bias.to(device=tokens.device, dtype=tokens.dtype)
        attn = torch.softmax(scores, dim=-1)
        self.last_attention_entropy = -torch.sum(attn * torch.log(attn.clamp_min(1e-8)), dim=-1).mean()
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_pilots, num_links, self.hidden_channels)
        tokens = tokens + self.out(context)
        return tokens + self.ffn(self.norm2(tokens))


class MIMOStructuredPilotTokenFormerEncoder(nn.Module):
    """Encode each pilot through MIMO link tokens before pilot-level attention."""

    def __init__(
        self,
        n_rx: int,
        n_tx: int,
        pos_dim: int,
        hidden_channels: int,
        num_heads: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.n_rx = int(n_rx)
        self.n_tx = int(n_tx)
        self.n_links = self.n_rx * self.n_tx
        self.hidden_channels = int(hidden_channels)

        self.value_projection = nn.Sequential(
            nn.Linear(5 + int(pos_dim), hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.rx_embedding = nn.Embedding(self.n_rx, hidden_channels)
        self.tx_embedding = nn.Embedding(self.n_tx, hidden_channels)
        self.link_layers = nn.ModuleList(
            [
                MIMOLinkAttentionLayer(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    pair_feature_dim=6,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.link_pool = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1),
        )
        self.out_norm = nn.LayerNorm(hidden_channels)

        rx_ids = torch.arange(self.n_rx).repeat_interleave(self.n_tx)
        tx_ids = torch.arange(self.n_tx).repeat(self.n_rx)
        self.register_buffer("link_rx_ids", rx_ids.to(dtype=torch.long), persistent=False)
        self.register_buffer("link_tx_ids", tx_ids.to(dtype=torch.long), persistent=False)
        self.register_buffer("link_pair_features", self._link_pair_features(rx_ids, tx_ids), persistent=False)

    def _link_pair_features(self, rx_ids: torch.Tensor, tx_ids: torch.Tensor) -> torch.Tensor:
        rx = rx_ids.to(dtype=torch.float32)
        tx = tx_ids.to(dtype=torch.float32)
        rx_scale = float(max(self.n_rx - 1, 1))
        tx_scale = float(max(self.n_tx - 1, 1))
        drx = (rx[:, None] - rx[None, :]) / rx_scale
        dtx = (tx[:, None] - tx[None, :]) / tx_scale
        same_rx = (rx[:, None] == rx[None, :]).to(dtype=torch.float32)
        same_tx = (tx[:, None] == tx[None, :]).to(dtype=torch.float32)
        return torch.stack((drx, dtx, torch.abs(drx), torch.abs(dtx), same_rx, same_tx), dim=-1)

    def forward(self, h_pilot: torch.Tensor, pos_features: torch.Tensor) -> torch.Tensor:
        batch, num_pilots = h_pilot.shape[:2]
        links = h_pilot.reshape(batch, num_pilots, self.n_links)
        magnitude = torch.abs(links).clamp_min(1e-8)
        value_features = torch.stack(
            (
                links.real,
                links.imag,
                magnitude,
                links.imag / magnitude,
                links.real / magnitude,
            ),
            dim=-1,
        ).to(dtype=pos_features.dtype)
        pos = pos_features.unsqueeze(0).unsqueeze(2).expand(batch, num_pilots, self.n_links, -1)
        link_tokens = self.value_projection(torch.cat((value_features, pos), dim=-1))

        rx_embed = self.rx_embedding(self.link_rx_ids.to(device=h_pilot.device))
        tx_embed = self.tx_embedding(self.link_tx_ids.to(device=h_pilot.device))
        link_tokens = link_tokens + (rx_embed + tx_embed).view(1, 1, self.n_links, self.hidden_channels)

        link_tokens = link_tokens.reshape(batch * num_pilots, self.n_links, self.hidden_channels)
        pair_features = self.link_pair_features.to(device=h_pilot.device, dtype=link_tokens.dtype)
        for layer in self.link_layers:
            link_tokens = layer(link_tokens, pair_features)

        pool_logits = self.link_pool(link_tokens).squeeze(-1)
        pool = torch.softmax(pool_logits, dim=-1)
        pilot_tokens = torch.sum(link_tokens * pool.unsqueeze(-1), dim=1)
        pilot_tokens = pilot_tokens.reshape(batch, num_pilots, self.hidden_channels)
        return self.out_norm(pilot_tokens)

    def attention_entropies(self) -> list[torch.Tensor]:
        entropies: list[torch.Tensor] = []
        for layer in self.link_layers:
            if layer.last_attention_entropy is not None:
                entropies.append(layer.last_attention_entropy)
        return entropies


class PilotFittedMIMOSharedBasisNet(nn.Module):
    """Pilot-fitted MIMO shared-basis estimator.

    The model does not use the interpolated LS grid as input. It learns a set of
    shared time-frequency basis maps, then fits the per-frame MIMO link
    coefficients from the raw pilot LS observations through a differentiable
    weighted ridge step.

    Input:
        h_pilot: complex tensor [batch, pilot, rx, tx].

    Output:
        real/imag tensor [batch, 2*rx*tx, n_symbols, n_subcarriers].
    """

    def __init__(
        self,
        pilot_positions: torch.Tensor,
        n_symbols: int = 14,
        n_subcarriers: int = 72,
        n_rx: int = 2,
        n_tx: int = 2,
        num_basis: int = 32,
        hidden_channels: int = 128,
        pos_bands: int = 6,
        min_gate: float = 0.05,
        min_regularization: float = 1e-4,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_layers: int = 1,
        token_encoder: str = "flat",
        link_attention_heads: int | None = None,
        link_attention_layers: int = 1,
        link_attention_dropout: float | None = None,
        use_learned_pilot_weights: bool = True,
        use_learned_regularization: bool = True,
        use_basis_gate: bool = True,
        gate_temperature: float = 1.0,
        attention_dropout: float = 0.0,
        gate_dropout: float = 0.0,
        basis_dropout: float = 0.0,
        pilot_noise_std: float = 0.0,
        active_basis_topk: int = 0,
    ) -> None:
        super().__init__()
        if pilot_positions.ndim != 2 or pilot_positions.shape[1] != 2:
            raise ValueError("pilot_positions must have shape [pilot, 2].")

        self.n_symbols = int(n_symbols)
        self.n_subcarriers = int(n_subcarriers)
        self.n_rx = int(n_rx)
        self.n_tx = int(n_tx)
        self.n_links = self.n_rx * self.n_tx
        self.num_basis = int(num_basis)
        self.hidden_channels = int(hidden_channels)
        self.pos_bands = int(pos_bands)
        self.min_gate = float(min_gate)
        self.min_regularization = float(min_regularization)
        self.use_attention = bool(use_attention)
        self.attention_heads = int(attention_heads)
        self.num_attention_layers = int(attention_layers)
        self.token_encoder_type = token_encoder.lower().replace("-", "_")
        if self.token_encoder_type not in {"flat", "mimo_tokenformer"}:
            raise ValueError("token_encoder must be 'flat' or 'mimo_tokenformer'.")
        self.link_attention_heads = int(link_attention_heads or attention_heads)
        self.link_attention_layers = int(link_attention_layers)
        self.link_attention_dropout = float(attention_dropout if link_attention_dropout is None else link_attention_dropout)
        self.use_learned_pilot_weights = bool(use_learned_pilot_weights)
        self.use_learned_regularization = bool(use_learned_regularization)
        self.use_basis_gate = bool(use_basis_gate)
        self.gate_temperature = max(float(gate_temperature), 1e-3)
        self.gate_dropout = nn.Dropout(float(gate_dropout))
        self.basis_dropout_probability = float(basis_dropout)
        self.pilot_noise_std = float(pilot_noise_std)
        self.active_basis_topk = int(active_basis_topk)

        positions = pilot_positions.to(dtype=torch.long)
        self.register_buffer("pilot_positions", positions)
        pilot_flat = positions[:, 0] * self.n_subcarriers + positions[:, 1]
        self.register_buffer("pilot_flat_indices", pilot_flat.to(dtype=torch.long))
        self.register_buffer("pilot_position_features", self._position_features(positions))
        self.register_buffer("relative_position_features", self._relative_position_features(positions), persistent=False)

        pos_dim = int(self.pilot_position_features.shape[1])
        token_dim = 2 * self.n_links + pos_dim
        if self.token_encoder_type == "flat":
            self.token_mlp = nn.Sequential(
                nn.Linear(token_dim, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.mimo_token_encoder = MIMOStructuredPilotTokenFormerEncoder(
                n_rx=self.n_rx,
                n_tx=self.n_tx,
                pos_dim=pos_dim,
                hidden_channels=hidden_channels,
                num_heads=self.link_attention_heads,
                num_layers=self.link_attention_layers,
                dropout=self.link_attention_dropout,
            )
        self.weight_head = nn.Linear(hidden_channels, 1)
        if self.use_attention:
            relative_dim = int(self.relative_position_features.shape[-1])
            self.attention_encoder = nn.ModuleList(
                [
                    RelativePilotAttentionLayer(
                        hidden_channels,
                        self.attention_heads,
                        relative_dim,
                        dropout=float(attention_dropout),
                    )
                    for _ in range(self.num_attention_layers)
                ]
            )
            self.basis_attention = BasisPilotCrossAttention(
                hidden_channels,
                self.attention_heads,
                num_basis,
                dropout=float(attention_dropout),
            )
        self.summary_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.gate_head = nn.Linear(hidden_channels, num_basis)
        self.regularization_head = nn.Linear(hidden_channels, 1)

        self.basis = nn.Parameter(torch.empty(num_basis, 2, self.n_symbols, self.n_subcarriers))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.basis, mean=0.0, std=1.0 / (self.n_symbols * self.n_subcarriers) ** 0.5)
        nn.init.constant_(self.regularization_head.bias, -7.0)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.weight_head.bias)

    def _position_features(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.to(dtype=torch.float32)
        if self.n_symbols > 1:
            t = pos[:, 0] / float(self.n_symbols - 1)
        else:
            t = pos[:, 0]
        if self.n_subcarriers > 1:
            f = pos[:, 1] / float(self.n_subcarriers - 1)
        else:
            f = pos[:, 1]

        features = [t, f]
        for band in range(self.pos_bands):
            freq = float(2 ** band)
            features.extend(
                [
                    torch.sin(2.0 * torch.pi * freq * t),
                    torch.cos(2.0 * torch.pi * freq * t),
                    torch.sin(2.0 * torch.pi * freq * f),
                    torch.cos(2.0 * torch.pi * freq * f),
                ]
            )
        return torch.stack(features, dim=1)

    def _relative_position_features(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.to(dtype=torch.float32)
        if self.n_symbols > 1:
            t = pos[:, 0] / float(self.n_symbols - 1)
        else:
            t = pos[:, 0]
        if self.n_subcarriers > 1:
            f = pos[:, 1] / float(self.n_subcarriers - 1)
        else:
            f = pos[:, 1]

        dt = t[:, None] - t[None, :]
        df = f[:, None] - f[None, :]
        features = [dt, df, torch.abs(dt), torch.abs(df)]
        for band in range(self.pos_bands):
            freq = float(2 ** band)
            features.extend(
                [
                    torch.sin(2.0 * torch.pi * freq * dt),
                    torch.cos(2.0 * torch.pi * freq * dt),
                    torch.sin(2.0 * torch.pi * freq * df),
                    torch.cos(2.0 * torch.pi * freq * df),
                ]
            )
        return torch.stack(features, dim=-1)

    @torch.no_grad()
    def set_complex_basis(self, basis: torch.Tensor) -> None:
        """Initialize learned basis from a complex tensor [basis, symbol, subcarrier]."""

        if basis.ndim != 3:
            raise ValueError("basis must have shape [basis, symbol, subcarrier].")
        if basis.shape[0] != self.num_basis:
            raise ValueError(f"basis count mismatch: expected {self.num_basis}, got {basis.shape[0]}.")
        if basis.shape[1:] != (self.n_symbols, self.n_subcarriers):
            raise ValueError("basis grid shape mismatch.")

        if not torch.is_complex(basis):
            raise ValueError("basis must be complex.")
        target = torch.stack((basis.real, basis.imag), dim=1).to(device=self.basis.device, dtype=self.basis.dtype)
        self.basis.copy_(target)

    def complex_basis(self) -> torch.Tensor:
        return torch.complex(self.basis[:, 0], self.basis[:, 1])

    def encode_pilots(self, h_pilot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_pilots = h_pilot.shape[:2]
        links = h_pilot.reshape(batch, num_pilots, self.n_links)
        if self.training and self.pilot_noise_std > 0:
            noise_scale = torch.as_tensor(self.pilot_noise_std, device=h_pilot.device, dtype=links.real.dtype)
            noise = torch.randn_like(links.real) * noise_scale
            links = torch.complex(links.real + noise, links.imag + torch.randn_like(links.imag) * noise_scale)
        pos_features = self.pilot_position_features.to(device=h_pilot.device, dtype=self.basis.dtype)
        if self.token_encoder_type == "flat":
            value_features = torch.cat((links.real, links.imag), dim=-1).to(dtype=self.basis.dtype)
            expanded_pos = pos_features.unsqueeze(0).expand(batch, -1, -1)
            tokens = self.token_mlp(torch.cat((value_features, expanded_pos), dim=-1))
        else:
            h_for_encoder = links.reshape(batch, num_pilots, self.n_rx, self.n_tx)
            tokens = self.mimo_token_encoder(h_for_encoder, pos_features)
        if self.use_attention:
            relative_features = self.relative_position_features.to(device=h_pilot.device, dtype=self.basis.dtype)
            for layer in self.attention_encoder:
                tokens = layer(tokens, relative_features)

        if self.use_learned_pilot_weights:
            pilot_weights = F.softplus(self.weight_head(tokens)).squeeze(-1) + 1e-4
        else:
            pilot_weights = torch.ones(batch, num_pilots, device=h_pilot.device, dtype=self.basis.dtype)
        alpha = pilot_weights / pilot_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        summary = torch.sum(tokens * alpha.unsqueeze(-1), dim=1)
        summary = self.summary_mlp(summary)

        if self.use_basis_gate:
            if self.use_attention:
                gate_logits = self.basis_attention(tokens)
            else:
                gate_logits = self.gate_head(summary)
            gate = self.min_gate + (1.0 - self.min_gate) * torch.sigmoid(gate_logits / self.gate_temperature)
            if self.training:
                gate = self.gate_dropout(gate)
        else:
            gate = torch.ones(batch, self.num_basis, device=h_pilot.device, dtype=self.basis.dtype)

        if self.use_learned_regularization:
            regularization = self.min_regularization + F.softplus(self.regularization_head(summary)).squeeze(-1)
        else:
            regularization = torch.full(
                (batch,),
                self.min_regularization,
                device=h_pilot.device,
                dtype=self.basis.dtype,
            )
        return pilot_weights, gate, regularization

    def forward(self, h_pilot: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(h_pilot):
            raise ValueError("PilotFittedMIMOSharedBasisNet expects complex h_pilot input.")
        if h_pilot.shape[2] != self.n_rx or h_pilot.shape[3] != self.n_tx:
            raise ValueError("h_pilot rx/tx dimensions do not match model metadata.")

        batch = h_pilot.shape[0]
        pilot_weights, gate, regularization = self.encode_pilots(h_pilot)
        if self.training:
            if self.basis_dropout_probability > 0:
                keep_probability = 1.0 - self.basis_dropout_probability
                keep = (torch.rand_like(gate) < keep_probability).to(dtype=gate.dtype)
                if self.num_basis > 0 and torch.any(torch.sum(keep, dim=1) == 0):
                    fallback = torch.argmax(gate, dim=1)
                    keep[torch.arange(gate.shape[0], device=gate.device), fallback] = 1.0
                gate = gate * keep / max(keep_probability, 1e-6)
        basis = self.complex_basis().to(device=h_pilot.device)
        basis_flat = basis.reshape(self.num_basis, self.n_symbols * self.n_subcarriers)
        basis_pilot = basis_flat[:, self.pilot_flat_indices.to(device=h_pilot.device)].transpose(0, 1)

        if 0 < self.active_basis_topk < self.num_basis:
            active_count = min(int(self.active_basis_topk), self.num_basis)
            topk_indices = torch.topk(gate, k=active_count, dim=1).indices
            active_gate = torch.gather(gate, dim=1, index=topk_indices)

            basis_pilot_batch = basis_pilot.unsqueeze(0).expand(batch, -1, -1)
            pilot_gather_index = topk_indices.unsqueeze(1).expand(-1, basis_pilot.shape[0], -1)
            active_basis_pilot = torch.gather(basis_pilot_batch, dim=2, index=pilot_gather_index)

            basis_flat_batch = basis_flat.unsqueeze(0).expand(batch, -1, -1)
            full_gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, basis_flat.shape[1])
            active_basis_full = torch.gather(basis_flat_batch, dim=1, index=full_gather_index)
        else:
            active_count = self.num_basis
            active_gate = gate
            active_basis_pilot = basis_pilot.unsqueeze(0).expand(batch, -1, -1)
            active_basis_full = basis_flat.unsqueeze(0).expand(batch, -1, -1)

        sqrt_gate = torch.sqrt(active_gate).to(dtype=basis.dtype)
        gated_basis_pilot = active_basis_pilot * sqrt_gate.unsqueeze(1)
        gated_basis_full = active_basis_full * sqrt_gate.unsqueeze(-1)

        sqrt_weight = torch.sqrt(pilot_weights).to(dtype=basis.dtype).unsqueeze(-1)
        weighted_pilot_basis = gated_basis_pilot * sqrt_weight
        gram = torch.matmul(weighted_pilot_basis.conj().transpose(-2, -1), weighted_pilot_basis)
        eye = torch.eye(active_count, device=h_pilot.device, dtype=basis.dtype).unsqueeze(0)
        gram = gram + regularization.to(dtype=basis.real.dtype).view(batch, 1, 1) * eye

        h_links = h_pilot.reshape(batch, h_pilot.shape[1], self.n_links)
        rhs = torch.matmul(gated_basis_pilot.conj().transpose(-2, -1), h_links * pilot_weights.unsqueeze(-1))
        coeff = torch.linalg.solve(gram, rhs)
        estimate = torch.einsum("bml,bmg->blg", coeff, gated_basis_full)
        estimate = estimate.reshape(batch, self.n_links, self.n_symbols, self.n_subcarriers)
        return torch.cat((estimate.real, estimate.imag), dim=1).to(dtype=self.basis.dtype)

    def basis_orthogonality_loss(self) -> torch.Tensor:
        basis = self.complex_basis().reshape(self.num_basis, -1)
        basis = basis / torch.linalg.vector_norm(basis, dim=1, keepdim=True).clamp_min(1e-8)
        gram = basis @ basis.conj().transpose(0, 1)
        eye = torch.eye(self.num_basis, device=basis.device, dtype=basis.dtype)
        return torch.mean(torch.abs(gram - eye) ** 2).real

    def attention_entropy_loss(self) -> torch.Tensor:
        entropies: list[torch.Tensor] = []
        if self.token_encoder_type == "mimo_tokenformer":
            entropies.extend(self.mimo_token_encoder.attention_entropies())
        if self.use_attention:
            for layer in self.attention_encoder:
                if layer.last_attention_entropy is not None:
                    entropies.append(layer.last_attention_entropy)
            if self.basis_attention.last_attention_entropy is not None:
                entropies.append(self.basis_attention.last_attention_entropy)
        if not entropies:
            return self.basis.sum() * 0.0
        return -torch.stack(entropies).mean()


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
