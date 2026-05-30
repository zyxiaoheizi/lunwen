from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.special import j0


Array = np.ndarray


@dataclass(frozen=True)
class SimConfig:
    """Core simulation settings for the first-stage baseline."""

    n_frames: int = 2000
    n_tx: int = 2
    n_rx: int = 2
    n_subcarriers: int = 64
    n_taps: int = 4
    channel_profile: str = "tdl-a"
    lmmse_profile: str | None = "tdl-a"
    subcarrier_spacing_hz: float = 15_000.0
    delay_spread_ns: float = 300.0
    lmmse_delay_spread_ns: float | None = None
    pilot_ratio: float = 0.25
    snr_db: tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30)
    channel_model: str = "rayleigh"
    rician_k: float = 5.0
    pdp_decay: float = 1.5
    seed: int = 1234


@dataclass(frozen=True)
class DelayProfile:
    name: str
    delays_sec: Array
    powers_linear: Array


@dataclass(frozen=True)
class PilotObservation:
    """显式导频观测结果，后续在线损失会直接用 x_p 和 y_p。"""

    x_p: Array
    y_p: Array
    h_pilot_ls: Array
    noise_var: Array
    pilot_mask: Array


@dataclass(frozen=True)
class BaselineResult:
    snr_db: float
    ls_linear_nmse: float
    lmmse_oracle_nmse: float
    lmmse_mismatched_nmse: float | None = None

    @property
    def ls_linear_nmse_db(self) -> float:
        return linear_to_db(self.ls_linear_nmse)

    @property
    def lmmse_oracle_nmse_db(self) -> float:
        return linear_to_db(self.lmmse_oracle_nmse)

    @property
    def lmmse_mismatched_nmse_db(self) -> float | None:
        if self.lmmse_mismatched_nmse is None:
            return None
        return linear_to_db(self.lmmse_mismatched_nmse)


def db_to_linear(db_value: float) -> float:
    return float(10.0 ** (db_value / 10.0))


def linear_to_db(value: float) -> float:
    value = max(float(value), np.finfo(np.float64).tiny)
    return float(10.0 * np.log10(value))


def db_powers_to_linear(powers_db: Iterable[float]) -> Array:
    powers = 10.0 ** (np.asarray(list(powers_db), dtype=np.float64) / 10.0)
    return powers / np.sum(powers)


def complex_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    variance: float | Array = 1.0,
) -> Array:
    std = np.sqrt(np.asarray(variance) / 2.0)
    return std * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def exponential_pdp(n_taps: int, decay: float = 1.5) -> Array:
    tap_ids = np.arange(n_taps, dtype=np.float64)
    powers = np.exp(-tap_ids / decay)
    return powers / np.sum(powers)


def resolve_delay_profile(
    name: str,
    n_taps: int,
    n_subcarriers: int,
    subcarrier_spacing_hz: float,
    delay_spread_ns: float,
    pdp_decay: float = 1.5,
) -> DelayProfile:
    """Resolve a named tapped-delay-line power-delay profile.

    EPA/EVA/ETU are LTE-style tapped delay profiles with delays in ns.
    TDL-A/B/C use normalized 3GPP TR 38.901 delays scaled by delay_spread_ns.
    """

    normalized_name = name.lower().replace("_", "-")
    if normalized_name in {"exp", "exponential", "rayleigh-l"}:
        sample_period = 1.0 / (n_subcarriers * subcarrier_spacing_hz)
        delays = np.arange(n_taps, dtype=np.float64) * sample_period
        powers = exponential_pdp(n_taps, pdp_decay)
        return DelayProfile(f"exponential-l{n_taps}", delays, powers)

    fixed_profiles_ns: dict[str, tuple[list[float], list[float]]] = {
        "epa": (
            [0, 30, 70, 90, 110, 190, 410],
            [0, -1, -2, -3, -8, -17.2, -20.8],
        ),
        "eva": (
            [0, 30, 150, 310, 370, 710, 1090, 1730, 2510],
            [0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9],
        ),
        "etu": (
            [0, 50, 120, 200, 230, 500, 1600, 2300, 5000],
            [-1, -1, -1, 0, 0, 0, -3, -5, -7],
        ),
    }
    if normalized_name in fixed_profiles_ns:
        delays_ns, powers_db = fixed_profiles_ns[normalized_name]
        return DelayProfile(
            normalized_name,
            np.asarray(delays_ns, dtype=np.float64) * 1e-9,
            db_powers_to_linear(powers_db),
        )

    tdl_profiles: dict[str, tuple[list[float], list[float]]] = {
        "tdl-a": (
            [
                0,
                0.3819,
                0.4025,
                0.5868,
                0.4610,
                0.5375,
                0.6708,
                0.5750,
                0.7618,
                1.5375,
                1.8978,
                2.2242,
                2.1718,
                2.4942,
                2.5119,
                3.0582,
                4.0810,
                4.4579,
                4.5695,
                4.7966,
                5.0066,
                5.3043,
                9.6586,
            ],
            [
                -13.4,
                0,
                -2.2,
                -4,
                -6,
                -8.2,
                -9.9,
                -10.5,
                -7.5,
                -15.9,
                -6.6,
                -16.7,
                -12.4,
                -15.2,
                -10.8,
                -11.3,
                -12.7,
                -16.2,
                -18.3,
                -18.9,
                -16.6,
                -19.9,
                -29.7,
            ],
        ),
        "tdl-b": (
            [
                0,
                0.1072,
                0.2155,
                0.2095,
                0.2870,
                0.2986,
                0.3752,
                0.5055,
                0.3681,
                0.3697,
                0.5700,
                0.5283,
                1.1021,
                1.2756,
                1.5474,
                1.7842,
                2.0169,
                2.8294,
                3.0219,
                3.6187,
                4.1067,
                4.2790,
                4.7834,
            ],
            [
                0,
                -2.2,
                -4,
                -3.2,
                -9.8,
                -1.2,
                -3.4,
                -5.2,
                -7.6,
                -3,
                -8.9,
                -9,
                -4.8,
                -5.7,
                -7.5,
                -1.9,
                -7.6,
                -12.2,
                -9.8,
                -11.4,
                -14.9,
                -9.2,
                -11.3,
            ],
        ),
        "tdl-c": (
            [
                0,
                0.2099,
                0.2219,
                0.2329,
                0.2176,
                0.6366,
                0.6448,
                0.6560,
                0.6584,
                0.7935,
                0.8213,
                0.9336,
                1.2285,
                1.3083,
                2.1704,
                2.7105,
                4.2589,
                4.6003,
                5.4902,
                5.6077,
                6.3065,
                6.6374,
                7.0427,
                8.6523,
            ],
            [
                -4.4,
                -1.2,
                -3.5,
                -5.2,
                -2.5,
                0,
                -2.2,
                -3.9,
                -7.4,
                -7.1,
                -10.7,
                -11.1,
                -5.1,
                -6.8,
                -8.7,
                -13.2,
                -13.9,
                -13.9,
                -15.8,
                -17.1,
                -16,
                -15.7,
                -21.6,
                -22.8,
            ],
        ),
    }
    if normalized_name in tdl_profiles:
        normalized_delays, powers_db = tdl_profiles[normalized_name]
        delays = np.asarray(normalized_delays, dtype=np.float64) * delay_spread_ns * 1e-9
        return DelayProfile(normalized_name, delays, db_powers_to_linear(powers_db))

    choices = ["exponential", "epa", "eva", "etu", "tdl-a", "tdl-b", "tdl-c"]
    raise ValueError(f"Unsupported channel_profile: {name}. Choose one of {choices}.")


def pilot_indices(n_subcarriers: int, pilot_ratio: float) -> Array:
    if not 0.0 < pilot_ratio <= 1.0:
        raise ValueError("pilot_ratio must be in (0, 1].")
    if pilot_ratio >= 1.0:
        return np.arange(n_subcarriers, dtype=np.int64)

    spacing = max(1, int(round(1.0 / pilot_ratio)))
    indices = np.arange(0, n_subcarriers, spacing, dtype=np.int64)
    if len(indices) < 2:
        indices = np.array([0, n_subcarriers // 2], dtype=np.int64)
    return indices


def pilot_mask(n_subcarriers: int, indices: Array) -> Array:
    """生成长度为 N_subcarriers 的导频位置 mask。"""

    mask = np.zeros(n_subcarriers, dtype=np.float32)
    mask[indices] = 1.0
    return mask


def channelnet_uniform_pilot_positions(
    n_symbols: int = 14,
    n_subcarriers: int = 72,
    num_pilots: int = 48,
) -> Array:
    """生成 ChannelNet/DeepPilotDesign 风格的二维均匀导频位置。

    ChannelNet 相关代码把信道看成 [72 subcarriers, 14 OFDM symbols]
    的二维图像，并使用 8/16/24/32/36/48 个导频点。这里返回的位置格式是
    [pilot, 2]，每行是 [ofdm_symbol, subcarrier]，方便在我们的
    [batch, symbol, subcarrier, rx, tx] 张量上索引。
    """

    if n_symbols == 14 and n_subcarriers == 72:
        if num_pilots == 48:
            flat = (
                [14 * i for i in range(1, 72, 6)]
                + [4 + 14 * i for i in range(4, 72, 6)]
                + [7 + 14 * i for i in range(1, 72, 6)]
                + [11 + 14 * i for i in range(4, 72, 6)]
            )
        elif num_pilots == 36:
            flat = (
                [14 * i for i in range(1, 72, 6)]
                + [6 + 14 * i for i in range(4, 72, 6)]
                + [11 + 14 * i for i in range(1, 72, 6)]
            )
        elif num_pilots == 24:
            flat = (
                [14 * i for i in range(1, 72, 9)]
                + [6 + 14 * i for i in range(4, 72, 9)]
                + [11 + 14 * i for i in range(1, 72, 9)]
            )
        elif num_pilots == 32:
            flat = (
                [2 + 14 * i for i in range(1, 72, 9)]
                + [5 + 14 * i for i in range(5, 72, 9)]
                + [8 + 14 * i for i in range(1, 72, 9)]
                + [12 + 14 * i for i in range(5, 72, 9)]
            )
        elif num_pilots == 16:
            flat = [4 + 14 * i for i in range(1, 72, 9)] + [9 + 14 * i for i in range(4, 72, 9)]
        elif num_pilots == 8:
            flat = [4 + 14 * i for i in range(5, 72, 18)] + [9 + 14 * i for i in range(8, 72, 18)]
        elif num_pilots == 4:
            flat = [4 + 14 * 8, 9 + 14 * 26, 4 + 14 * 44, 9 + 14 * 62]
        else:
            raise ValueError("ChannelNet-style pilot count must be one of 4, 8, 16, 24, 32, 36, 48.")
        return np.asarray([(idx % 14, idx // 14) for idx in flat], dtype=np.int64)

    # 非 72x14 时退化为规则网格，保证函数也能用于小规模 sanity check。
    n_time = max(1, int(round(np.sqrt(num_pilots * n_symbols / n_subcarriers))))
    n_freq = max(1, int(np.ceil(num_pilots / n_time)))
    time_ids = np.linspace(0, n_symbols - 1, n_time, dtype=np.int64)
    freq_ids = np.linspace(0, n_subcarriers - 1, n_freq, dtype=np.int64)
    positions = np.array([(t, f) for f in freq_ids for t in time_ids], dtype=np.int64)
    return positions[:num_pilots]


def pilot_mask_2d(n_symbols: int, n_subcarriers: int, positions: Array) -> Array:
    """生成二维时频导频 mask，shape 为 [ofdm_symbol, subcarrier]。"""

    mask = np.zeros((n_symbols, n_subcarriers), dtype=np.float32)
    mask[positions[:, 0], positions[:, 1]] = 1.0
    return mask


def generate_frequency_channel(
    rng: np.random.Generator,
    n_frames: int,
    n_subcarriers: int,
    n_rx: int,
    n_tx: int,
    profile: DelayProfile,
    subcarrier_spacing_hz: float,
    channel_model: str = "rayleigh",
    rician_k: float = 5.0,
    center_subcarriers: bool = True,
) -> Array:
    """Generate H[k] with shape [batch, subcarrier, rx, tx]."""

    n_taps = len(profile.powers_linear)
    tap_shape = (n_frames, n_rx, n_tx, n_taps)
    nlos_taps = complex_normal(rng, tap_shape, profile.powers_linear.reshape(1, 1, 1, n_taps))

    model = channel_model.lower()
    if model == "rayleigh":
        taps = nlos_taps
    elif model == "rician":
        k_linear = float(rician_k)
        los = np.zeros(tap_shape, dtype=np.complex128)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_frames, n_rx, n_tx))
        los[..., 0] = np.exp(1j * phase)
        taps = np.sqrt(1.0 / (k_linear + 1.0)) * nlos_taps
        taps += np.sqrt(k_linear / (k_linear + 1.0)) * los
    else:
        raise ValueError(f"Unsupported channel_model: {channel_model}")

    subcarrier_ids = np.arange(n_subcarriers, dtype=np.float64)
    if center_subcarriers:
        subcarrier_ids = subcarrier_ids - n_subcarriers // 2
    freqs = subcarrier_ids * subcarrier_spacing_hz
    phase = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * profile.delays_sec[None, :])
    freq = np.einsum("brtl,kl->bkrt", taps, phase)
    return freq.astype(np.complex64)


def generate_time_frequency_channel(
    rng: np.random.Generator,
    n_frames: int,
    n_symbols: int,
    n_subcarriers: int,
    n_rx: int,
    n_tx: int,
    profile: DelayProfile,
    subcarrier_spacing_hz: float,
    channel_model: str = "rayleigh",
    rician_k: float = 5.0,
    max_doppler_hz: float = 0.0,
    center_subcarriers: bool = True,
) -> Array:
    """生成二维时频 MIMO 信道，shape 为 [batch, symbol, subcarrier, rx, tx]。

    高水平 OFDM 信道估计论文常把信道响应看作二维时频图像，例如
    ChannelNet 使用 [72 subcarriers, 14 OFDM symbols]。这里用一个
    轻量 AR(1) 近似描述 tap 随 OFDM symbol 的时间相关性；当
    max_doppler_hz=0 时，信道在一个 frame 内保持静态。
    """

    n_taps = len(profile.powers_linear)
    tap_shape = (n_frames, n_rx, n_tx, n_taps)
    variance = profile.powers_linear.reshape(1, 1, 1, n_taps)
    first_taps = complex_normal(rng, tap_shape, variance)

    if max_doppler_hz <= 0:
        nlos_taps = np.repeat(first_taps[:, None, :, :, :], n_symbols, axis=1)
    else:
        symbol_duration = 1.0 / subcarrier_spacing_hz
        rho = float(j0(2.0 * np.pi * max_doppler_hz * symbol_duration))
        rho = float(np.clip(rho, -0.999, 0.999))
        innovation_scale = np.sqrt(max(0.0, 1.0 - rho**2))
        nlos_taps = np.empty((n_frames, n_symbols, n_rx, n_tx, n_taps), dtype=np.complex128)
        nlos_taps[:, 0, :, :, :] = first_taps
        for symbol_idx in range(1, n_symbols):
            innovation = complex_normal(rng, tap_shape, variance)
            nlos_taps[:, symbol_idx, :, :, :] = rho * nlos_taps[:, symbol_idx - 1, :, :, :] + innovation_scale * innovation

    model = channel_model.lower()
    if model == "rayleigh":
        taps = nlos_taps
    elif model == "rician":
        k_linear = float(rician_k)
        los = np.zeros_like(nlos_taps)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_frames, n_rx, n_tx))
        los[:, :, :, :, 0] = np.exp(1j * phase)[:, None, :, :]
        taps = np.sqrt(1.0 / (k_linear + 1.0)) * nlos_taps
        taps += np.sqrt(k_linear / (k_linear + 1.0)) * los
    else:
        raise ValueError(f"Unsupported channel_model: {channel_model}")

    subcarrier_ids = np.arange(n_subcarriers, dtype=np.float64)
    if center_subcarriers:
        subcarrier_ids = subcarrier_ids - n_subcarriers // 2
    freqs = subcarrier_ids * subcarrier_spacing_hz
    phase = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * profile.delays_sec[None, :])
    freq = np.einsum("bsrtl,kl->bskrt", taps, phase)
    return freq.astype(np.complex64)


def orthogonal_pilot_matrix(n_pilot_subcarriers: int, n_tx: int) -> Array:
    """生成正交导频矩阵。

    输出形状为 [pilot_subcarrier, pilot_symbol, tx]。对 2 发射天线来说，
    每个导频子载波使用 2 个正交导频符号：第 1 个符号只让 tx0 发，第
    2 个符号只让 tx1 发，因此接收端可以把不同发射天线的信道分开估计。
    """

    eye = np.eye(n_tx, dtype=np.complex64)
    return np.broadcast_to(eye[None, :, :], (n_pilot_subcarriers, n_tx, n_tx)).copy()


def ls_from_explicit_pilots(y_p: Array, x_p: Array) -> Array:
    """由显式 Xp/Yp 计算导频位置 LS 信道估计。

    y_p: [batch, pilot_subcarrier, pilot_symbol, rx]
    x_p: [pilot_subcarrier, pilot_symbol, tx]
    return: [batch, pilot_subcarrier, rx, tx]
    """

    pinv_x = np.linalg.pinv(x_p).astype(np.complex64)
    h_tx_rx = np.einsum("pts,bpsr->bptr", pinv_x, y_p)
    return np.transpose(h_tx_rx, (0, 1, 3, 2)).astype(np.complex64)


def observe_explicit_orthogonal_pilots(
    rng: np.random.Generator,
    h_true: Array,
    indices: Array,
    noise_var: float | Array,
) -> PilotObservation:
    """显式生成正交导频 Xp、接收导频 Yp，并由 LS 反推 H_LS,p。

    系统模型是 y = H x + n。这里用单位阵作为正交导频，所以 LS 结果
    与早期简化的 H_true + noise 等价，但现在保留了论文在线损失需要的
    Xp/Yp 原始量。
    """

    h_pilot = h_true[:, indices, :, :]
    x_p = orthogonal_pilot_matrix(len(indices), h_true.shape[3])
    clean_y = np.einsum("pst,bprt->bpsr", x_p, h_pilot)

    variance = np.asarray(noise_var, dtype=np.float32)
    if variance.ndim == 1:
        variance = variance[:, None, None, None]
    noise = complex_normal(rng, clean_y.shape, variance)
    y_p = (clean_y + noise).astype(np.complex64)
    h_pilot_ls = ls_from_explicit_pilots(y_p, x_p)
    return PilotObservation(
        x_p=x_p.astype(np.complex64),
        y_p=y_p,
        h_pilot_ls=h_pilot_ls,
        noise_var=np.asarray(noise_var, dtype=np.float32),
        pilot_mask=pilot_mask(h_true.shape[1], indices),
    )


def observe_explicit_orthogonal_grid_pilots(
    rng: np.random.Generator,
    h_true_grid: Array,
    positions: Array,
    noise_var: float | Array,
) -> PilotObservation:
    """二维时频网格上的显式正交导频观测。

    h_true_grid: [batch, symbol, subcarrier, rx, tx]
    positions: [pilot, 2]，每行 [symbol, subcarrier]
    """

    h_pilot = h_true_grid[:, positions[:, 0], positions[:, 1], :, :]
    x_p = orthogonal_pilot_matrix(len(positions), h_true_grid.shape[4])
    clean_y = np.einsum("pst,bprt->bpsr", x_p, h_pilot)
    variance = np.asarray(noise_var, dtype=np.float32)
    if variance.ndim == 1:
        variance = variance[:, None, None, None]
    noise = complex_normal(rng, clean_y.shape, variance)
    y_p = (clean_y + noise).astype(np.complex64)
    h_pilot_ls = ls_from_explicit_pilots(y_p, x_p)
    return PilotObservation(
        x_p=x_p.astype(np.complex64),
        y_p=y_p,
        h_pilot_ls=h_pilot_ls,
        noise_var=np.asarray(noise_var, dtype=np.float32),
        pilot_mask=pilot_mask_2d(h_true_grid.shape[1], h_true_grid.shape[2], positions),
    )


def observe_orthogonal_pilots(
    rng: np.random.Generator,
    h_true: Array,
    indices: Array,
    noise_var: float | Array,
) -> Array:
    """兼容旧接口：只返回导频位置 LS 信道估计。"""

    return observe_explicit_orthogonal_pilots(rng, h_true, indices, noise_var).h_pilot_ls


def interpolate_pilots_linear(h_pilot_ls: Array, indices: Array, n_subcarriers: int) -> Array:
    """Linear periodic interpolation from pilot subcarriers to all subcarriers."""

    if h_pilot_ls.shape[1] == n_subcarriers and np.array_equal(indices, np.arange(n_subcarriers)):
        return h_pilot_ls.astype(np.complex64, copy=True)

    x = indices.astype(np.float64)
    x_ext = np.concatenate(([x[-1] - n_subcarriers], x, [x[0] + n_subcarriers]))

    h_by_link = np.transpose(h_pilot_ls, (0, 2, 3, 1))
    original_shape = h_by_link.shape[:-1]
    flat = h_by_link.reshape(-1, h_pilot_ls.shape[1])
    flat_ext = np.concatenate((flat[:, -1:], flat, flat[:, :1]), axis=1)

    grid = np.arange(n_subcarriers, dtype=np.float64)
    real_interp = interp1d(x_ext, flat_ext.real, axis=1, kind="linear", assume_sorted=True)(grid)
    imag_interp = interp1d(x_ext, flat_ext.imag, axis=1, kind="linear", assume_sorted=True)(grid)
    out = (real_interp + 1j * imag_interp).reshape(*original_shape, n_subcarriers)
    return np.transpose(out, (0, 3, 1, 2)).astype(np.complex64)


def interpolate_pilots_2d_linear_nearest(
    h_pilot_ls: Array,
    positions: Array,
    n_symbols: int,
    n_subcarriers: int,
) -> Array:
    """把二维导频位置 LS 信道插值到完整时频网格。

    先用二维线性插值，边缘线性插不到的位置再用最近邻填充。这对应
    论文里常见的 "LS + interpolation" 输入构造方式，只是我们保留了
    MIMO 的 rx/tx 维度。
    """

    sym_grid, sub_grid = np.meshgrid(np.arange(n_symbols), np.arange(n_subcarriers), indexing="ij")
    grid_points = np.stack((sym_grid.ravel(), sub_grid.ravel()), axis=1)
    values = np.transpose(h_pilot_ls, (1, 0, 2, 3)).reshape(len(positions), -1)

    linear_real = LinearNDInterpolator(positions, values.real, fill_value=np.nan)(grid_points)
    linear_imag = LinearNDInterpolator(positions, values.imag, fill_value=np.nan)(grid_points)
    nearest_real = NearestNDInterpolator(positions, values.real)(grid_points)
    nearest_imag = NearestNDInterpolator(positions, values.imag)(grid_points)
    real = np.where(np.isnan(linear_real), nearest_real, linear_real)
    imag = np.where(np.isnan(linear_imag), nearest_imag, linear_imag)

    out = (real + 1j * imag).reshape(n_symbols, n_subcarriers, *h_pilot_ls.shape[0:1], *h_pilot_ls.shape[2:])
    return np.transpose(out, (2, 0, 1, 3, 4)).astype(np.complex64)


def frequency_covariance(
    n_subcarriers: int,
    profile: DelayProfile,
    subcarrier_spacing_hz: float,
) -> Array:
    subcarrier_ids = np.arange(n_subcarriers, dtype=np.float64)
    lags = subcarrier_ids[:, None] - subcarrier_ids[None, :]
    phase = np.exp(
        -1j * 2.0 * np.pi * lags[..., None] * subcarrier_spacing_hz * profile.delays_sec[None, None, :]
    )
    return np.sum(profile.powers_linear.reshape(1, 1, -1) * phase, axis=-1).astype(np.complex128)


def lmmse_frequency_estimate(
    h_pilot_ls: Array,
    indices: Array,
    n_subcarriers: int,
    profile: DelayProfile,
    subcarrier_spacing_hz: float,
    noise_var: float,
) -> Array:
    """Oracle LMMSE frequency interpolation using the true PDP and noise."""

    r_hh = frequency_covariance(n_subcarriers, profile, subcarrier_spacing_hz)
    r_hp = r_hh[:, indices]
    r_pp = r_hh[np.ix_(indices, indices)]
    regularized = r_pp + noise_var * np.eye(len(indices), dtype=np.complex128)
    weights = np.linalg.solve(regularized.T, r_hp.T).T
    estimate = np.einsum("kp,bprt->bkrt", weights, h_pilot_ls.astype(np.complex128))
    return estimate.astype(np.complex64)


def nmse(h_true: Array, h_hat: Array) -> float:
    err_power = np.sum(np.abs(h_true - h_hat) ** 2, dtype=np.float64)
    ref_power = np.sum(np.abs(h_true) ** 2, dtype=np.float64)
    return float(err_power / ref_power)


def simulate_baselines(config: SimConfig) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    indices = pilot_indices(config.n_subcarriers, config.pilot_ratio)
    profile = resolve_delay_profile(
        name=config.channel_profile,
        n_taps=config.n_taps,
        n_subcarriers=config.n_subcarriers,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
        delay_spread_ns=config.delay_spread_ns,
        pdp_decay=config.pdp_decay,
    )
    lmmse_profile = None
    if config.lmmse_profile is not None:
        lmmse_profile = resolve_delay_profile(
            name=config.lmmse_profile,
            n_taps=config.n_taps,
            n_subcarriers=config.n_subcarriers,
            subcarrier_spacing_hz=config.subcarrier_spacing_hz,
            delay_spread_ns=config.lmmse_delay_spread_ns or config.delay_spread_ns,
            pdp_decay=config.pdp_decay,
        )

    for snr_idx, snr in enumerate(config.snr_db):
        rng = np.random.default_rng(config.seed + 1009 * snr_idx)
        h_true = generate_frequency_channel(
            rng=rng,
            n_frames=config.n_frames,
            n_subcarriers=config.n_subcarriers,
            n_rx=config.n_rx,
            n_tx=config.n_tx,
            profile=profile,
            subcarrier_spacing_hz=config.subcarrier_spacing_hz,
            channel_model=config.channel_model,
            rician_k=config.rician_k,
        )
        noise_var = 1.0 / db_to_linear(float(snr))
        pilot_obs = observe_explicit_orthogonal_pilots(rng, h_true, indices, noise_var)
        h_pilot_ls = pilot_obs.h_pilot_ls
        h_ls_linear = interpolate_pilots_linear(h_pilot_ls, indices, config.n_subcarriers)
        h_lmmse = lmmse_frequency_estimate(
            h_pilot_ls=h_pilot_ls,
            indices=indices,
            n_subcarriers=config.n_subcarriers,
            profile=profile,
            subcarrier_spacing_hz=config.subcarrier_spacing_hz,
            noise_var=noise_var,
        )
        h_lmmse_mismatched = None
        if lmmse_profile is not None:
            # mismatched LMMSE 使用“假设 profile”而不是真实测试 profile，
            # 用于模拟实际系统里信道统计信息不准确的情况。
            h_lmmse_mismatched = lmmse_frequency_estimate(
                h_pilot_ls=h_pilot_ls,
                indices=indices,
                n_subcarriers=config.n_subcarriers,
                profile=lmmse_profile,
                subcarrier_spacing_hz=config.subcarrier_spacing_hz,
                noise_var=noise_var,
            )
        results.append(
            BaselineResult(
                snr_db=float(snr),
                ls_linear_nmse=nmse(h_true, h_ls_linear),
                lmmse_oracle_nmse=nmse(h_true, h_lmmse),
                lmmse_mismatched_nmse=(
                    nmse(h_true, h_lmmse_mismatched) if h_lmmse_mismatched is not None else None
                ),
            )
        )

    return results


def complex_mimo_to_ri_channels(h: Array) -> Array:
    """Convert [batch, subcarrier, rx, tx] complex H to [batch, 2*rx*tx, subcarrier]."""

    by_link = np.transpose(h, (0, 2, 3, 1)).reshape(h.shape[0], h.shape[2] * h.shape[3], h.shape[1])
    return np.concatenate((by_link.real, by_link.imag), axis=1).astype(np.float32)


def complex_grid_to_ri_channels(h: Array) -> Array:
    """Convert [batch, symbol, subcarrier, rx, tx] to [batch, 2*rx*tx, symbol, subcarrier]."""

    by_link = np.transpose(h, (0, 3, 4, 1, 2)).reshape(h.shape[0], h.shape[3] * h.shape[4], h.shape[1], h.shape[2])
    return np.concatenate((by_link.real, by_link.imag), axis=1).astype(np.float32)


def generate_dataset(
    out_path: str | Path,
    samples: int = 5000,
    n_tx: int = 2,
    n_rx: int = 2,
    n_subcarriers: int = 64,
    n_taps: int = 4,
    channel_profile: str = "tdl-a",
    subcarrier_spacing_hz: float = 15_000.0,
    delay_spread_ns: float = 300.0,
    pilot_ratio: float = 0.25,
    snr_min_db: float = 0.0,
    snr_max_db: float = 30.0,
    channel_model: str = "rayleigh",
    rician_k: float = 5.0,
    pdp_decay: float = 1.5,
    seed: int = 2026,
) -> Path:
    rng = np.random.default_rng(seed)
    indices = pilot_indices(n_subcarriers, pilot_ratio)
    profile = resolve_delay_profile(
        name=channel_profile,
        n_taps=n_taps,
        n_subcarriers=n_subcarriers,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        delay_spread_ns=delay_spread_ns,
        pdp_decay=pdp_decay,
    )
    h_true = generate_frequency_channel(
        rng=rng,
        n_frames=samples,
        n_subcarriers=n_subcarriers,
        n_rx=n_rx,
        n_tx=n_tx,
        profile=profile,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        channel_model=channel_model,
        rician_k=rician_k,
    )

    snr_db = rng.uniform(snr_min_db, snr_max_db, size=samples).astype(np.float32)
    noise_var = (1.0 / (10.0 ** (snr_db / 10.0))).astype(np.float32)
    pilot_obs = observe_explicit_orthogonal_pilots(rng, h_true, indices, noise_var)
    h_pilot_ls = pilot_obs.h_pilot_ls
    h_ls_linear = interpolate_pilots_linear(h_pilot_ls, indices, n_subcarriers)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        h_true=h_true.astype(np.complex64),
        h_ls=h_ls_linear.astype(np.complex64),
        h_pilot_ls=h_pilot_ls.astype(np.complex64),
        x_p=pilot_obs.x_p.astype(np.complex64),
        y_p=pilot_obs.y_p.astype(np.complex64),
        pilot_mask=pilot_obs.pilot_mask.astype(np.float32),
        noise_var=noise_var.astype(np.float32),
        h_true_ri=complex_mimo_to_ri_channels(h_true),
        h_ls_ri=complex_mimo_to_ri_channels(h_ls_linear),
        pilot_indices=indices.astype(np.int64),
        snr_db=snr_db,
        delays_sec=profile.delays_sec.astype(np.float64),
        powers_linear=profile.powers_linear.astype(np.float64),
        n_tx=np.int64(n_tx),
        n_rx=np.int64(n_rx),
        n_subcarriers=np.int64(n_subcarriers),
        n_taps=np.int64(len(profile.powers_linear)),
        subcarrier_spacing_hz=np.float64(subcarrier_spacing_hz),
        delay_spread_ns=np.float64(delay_spread_ns),
        pilot_ratio=np.float32(pilot_ratio),
        channel_profile=np.array(profile.name),
        channel_model=np.array(channel_model),
    )
    return out


def generate_grid_dataset(
    out_path: str | Path,
    samples: int = 5000,
    n_tx: int = 2,
    n_rx: int = 2,
    n_symbols: int = 14,
    n_subcarriers: int = 72,
    num_pilots: int = 48,
    n_taps: int = 4,
    channel_profile: str = "tdl-a",
    subcarrier_spacing_hz: float = 15_000.0,
    delay_spread_ns: float = 300.0,
    snr_min_db: float = 0.0,
    snr_max_db: float = 30.0,
    channel_model: str = "rayleigh",
    rician_k: float = 5.0,
    max_doppler_hz: float = 70.0,
    pdp_decay: float = 1.5,
    seed: int = 2027,
) -> Path:
    """生成 ChannelNet-style 二维时频信道估计数据集。

    输出既包含复数信道，也包含 CNN 常用的实虚拆分格式。默认
    [14 symbols, 72 subcarriers, 48 pilots] 对齐 ChannelNet/
    DeepPilotDesign 的数据处理范式。
    """

    rng = np.random.default_rng(seed)
    positions = channelnet_uniform_pilot_positions(n_symbols, n_subcarriers, num_pilots)
    profile = resolve_delay_profile(
        name=channel_profile,
        n_taps=n_taps,
        n_subcarriers=n_subcarriers,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        delay_spread_ns=delay_spread_ns,
        pdp_decay=pdp_decay,
    )
    h_true = generate_time_frequency_channel(
        rng=rng,
        n_frames=samples,
        n_symbols=n_symbols,
        n_subcarriers=n_subcarriers,
        n_rx=n_rx,
        n_tx=n_tx,
        profile=profile,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        channel_model=channel_model,
        rician_k=rician_k,
        max_doppler_hz=max_doppler_hz,
    )

    snr_db = rng.uniform(snr_min_db, snr_max_db, size=samples).astype(np.float32)
    noise_var = (1.0 / (10.0 ** (snr_db / 10.0))).astype(np.float32)
    pilot_obs = observe_explicit_orthogonal_grid_pilots(rng, h_true, positions, noise_var)
    h_ls_grid = interpolate_pilots_2d_linear_nearest(pilot_obs.h_pilot_ls, positions, n_symbols, n_subcarriers)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        h_true_grid=h_true.astype(np.complex64),
        h_ls_grid=h_ls_grid.astype(np.complex64),
        h_pilot_ls=pilot_obs.h_pilot_ls.astype(np.complex64),
        x_p=pilot_obs.x_p.astype(np.complex64),
        y_p=pilot_obs.y_p.astype(np.complex64),
        pilot_mask=pilot_obs.pilot_mask.astype(np.float32),
        pilot_positions=positions.astype(np.int64),
        noise_var=noise_var.astype(np.float32),
        snr_db=snr_db,
        h_true_grid_ri=complex_grid_to_ri_channels(h_true),
        h_ls_grid_ri=complex_grid_to_ri_channels(h_ls_grid),
        delays_sec=profile.delays_sec.astype(np.float64),
        powers_linear=profile.powers_linear.astype(np.float64),
        n_tx=np.int64(n_tx),
        n_rx=np.int64(n_rx),
        n_symbols=np.int64(n_symbols),
        n_subcarriers=np.int64(n_subcarriers),
        num_pilots=np.int64(len(positions)),
        n_taps=np.int64(len(profile.powers_linear)),
        subcarrier_spacing_hz=np.float64(subcarrier_spacing_hz),
        delay_spread_ns=np.float64(delay_spread_ns),
        max_doppler_hz=np.float64(max_doppler_hz),
        channel_profile=np.array(profile.name),
        channel_model=np.array(channel_model),
    )
    return out


def results_to_rows(results: Iterable[BaselineResult]) -> list[dict[str, float]]:
    return [
        {
            "snr_db": result.snr_db,
            "ls_linear_nmse": result.ls_linear_nmse,
            "ls_linear_nmse_db": result.ls_linear_nmse_db,
            "lmmse_oracle_nmse": result.lmmse_oracle_nmse,
            "lmmse_oracle_nmse_db": result.lmmse_oracle_nmse_db,
            "lmmse_mismatched_nmse": result.lmmse_mismatched_nmse,
            "lmmse_mismatched_nmse_db": result.lmmse_mismatched_nmse_db,
        }
        for result in results
    ]
