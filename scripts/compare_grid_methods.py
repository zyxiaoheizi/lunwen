from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import j0
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapter_mimo_ofdm.models import (  # noqa: E402
    PilotFittedMIMOSharedBasisNet,
    PilotLockedErrorRefinementNet,
    build_model,
    count_parameters,
)
from adapter_mimo_ofdm.sim import resolve_delay_profile  # noqa: E402


class GridEvalDataset(Dataset):
    def __init__(self, path: str | Path, input_key: str, target_key: str, scale: float) -> None:
        data = np.load(path)
        self.x = data[input_key].astype(np.float32) / scale
        self.y = data[target_key].astype(np.float32) / scale
        self.path = Path(path)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])


class PilotFittedEvalDataset(Dataset):
    def __init__(self, path: str | Path, scale: float, target_key: str = "h_true_grid_ri") -> None:
        data = np.load(path)
        self.h_pilot = data["h_pilot_ls"].astype(np.complex64) / scale
        self.y = data[target_key].astype(np.float32) / scale
        self.pilot_positions = data["pilot_positions"].astype(np.int64)
        self.path = Path(path)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.h_pilot[index]), torch.from_numpy(self.y[index])


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--checkpoint must be in NAME=PATH format, for example srcnn=outputs/.../srcnn_grid_best.pt")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path)


def nmse_db(value: float) -> float:
    return float(10.0 * np.log10(max(value, np.finfo(np.float64).tiny)))


def np_nmse(input_value: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    err = np.sum((input_value - target) ** 2, axis=(1, 2, 3))
    ref = np.sum(target**2, axis=(1, 2, 3)).clip(min=1e-12)
    nmse = float(np.mean(err / ref))
    return nmse, nmse_db(nmse)


def complex_grid_to_ri(h: np.ndarray) -> np.ndarray:
    """[N, symbol, subcarrier, rx, tx] complex -> [N, 2*rx*tx, symbol, subcarrier]."""

    by_link = np.transpose(h, (0, 3, 4, 1, 2)).reshape(h.shape[0], h.shape[3] * h.shape[4], h.shape[1], h.shape[2])
    return np.concatenate((by_link.real, by_link.imag), axis=1).astype(np.float32)


def grid_covariance(
    n_symbols: int,
    n_subcarriers: int,
    delays_sec: np.ndarray,
    powers_linear: np.ndarray,
    subcarrier_spacing_hz: float,
    max_doppler_hz: float,
) -> np.ndarray:
    """二维时频信道相关矩阵，用于 Oracle LMMSE。

    频域相关来自 PDP；时域相关用 Jakes/J0 近似。当前数据生成器也用这个
    Doppler 相关近似，因此这是一个 oracle 传统强基线。
    """

    sym_ids = np.arange(n_symbols, dtype=np.float64)
    time_lags = sym_ids[:, None] - sym_ids[None, :]
    symbol_duration = 1.0 / subcarrier_spacing_hz
    r_time = j0(2.0 * np.pi * max_doppler_hz * symbol_duration * time_lags)

    sub_ids = np.arange(n_subcarriers, dtype=np.float64)
    freq_lags = sub_ids[:, None] - sub_ids[None, :]
    phase = np.exp(
        -1j * 2.0 * np.pi * freq_lags[..., None] * subcarrier_spacing_hz * delays_sec[None, None, :]
    )
    r_freq = np.sum(powers_linear.reshape(1, 1, -1) * phase, axis=-1)

    return np.kron(r_time, r_freq).astype(np.complex128)


def lmmse_grid_estimate_from_covariance(
    data: np.lib.npyio.NpzFile,
    cov: np.ndarray,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """从二维导频 LS 估计恢复完整时频网格。

    输出 shape: [N, symbol, subcarrier, rx, tx] complex64。
    """

    h_pilot_ls = data["h_pilot_ls"].astype(np.complex128)
    positions = data["pilot_positions"].astype(np.int64)
    noise_var = data["noise_var"].astype(np.float64)
    n_symbols = int(data["n_symbols"])
    n_subcarriers = int(data["n_subcarriers"])
    grid_size = n_symbols * n_subcarriers
    pilot_flat = positions[:, 0] * n_subcarriers + positions[:, 1]
    r_hp = cov[:, pilot_flat]
    r_pp = cov[np.ix_(pilot_flat, pilot_flat)]
    eye = np.eye(len(pilot_flat), dtype=np.complex128)
    if mean is None:
        mean_grid = np.zeros(grid_size, dtype=np.complex128)
    else:
        mean_grid = mean.astype(np.complex128).reshape(grid_size)
    mean_pilot = mean_grid[pilot_flat]

    estimates = np.empty((h_pilot_ls.shape[0], grid_size, h_pilot_ls.shape[2], h_pilot_ls.shape[3]), dtype=np.complex64)
    for idx in range(h_pilot_ls.shape[0]):
        # Solve (R_pp + sigma^2 I) A = H_pilot for all rx/tx links at once.
        rhs = h_pilot_ls[idx].reshape(len(pilot_flat), -1) - mean_pilot[:, None]
        coeff = np.linalg.solve(r_pp + noise_var[idx] * eye, rhs)
        estimates[idx] = (mean_grid[:, None] + r_hp @ coeff).reshape(
            grid_size,
            h_pilot_ls.shape[2],
            h_pilot_ls.shape[3],
        )

    return estimates.reshape(h_pilot_ls.shape[0], n_symbols, n_subcarriers, h_pilot_ls.shape[2], h_pilot_ls.shape[3])


def lmmse_grid_estimate(
    data: np.lib.npyio.NpzFile,
    delays_sec: np.ndarray,
    powers_linear: np.ndarray,
    max_doppler_hz: float,
) -> np.ndarray:
    """使用给定 PDP/Doppler 先验构造协方差，然后做二维 LMMSE。"""

    cov = grid_covariance(
        n_symbols=int(data["n_symbols"]),
        n_subcarriers=int(data["n_subcarriers"]),
        delays_sec=delays_sec.astype(np.float64),
        powers_linear=powers_linear.astype(np.float64),
        subcarrier_spacing_hz=float(data["subcarrier_spacing_hz"]),
        max_doppler_hz=max_doppler_hz,
    )
    return lmmse_grid_estimate_from_covariance(data, cov)


def oracle_lmmse_grid_estimate(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return lmmse_grid_estimate(
        data,
        delays_sec=data["delays_sec"].astype(np.float64),
        powers_linear=data["powers_linear"].astype(np.float64),
        max_doppler_hz=float(data["max_doppler_hz"]),
    )


def mismatched_lmmse_grid_estimate(
    data: np.lib.npyio.NpzFile,
    profile_name: str,
    delay_spread_ns: float,
    max_doppler_hz: float | None,
) -> np.ndarray:
    profile = resolve_delay_profile(
        name=profile_name,
        n_taps=int(data["n_taps"]),
        n_subcarriers=int(data["n_subcarriers"]),
        subcarrier_spacing_hz=float(data["subcarrier_spacing_hz"]),
        delay_spread_ns=delay_spread_ns,
    )
    return lmmse_grid_estimate(
        data,
        delays_sec=profile.delays_sec,
        powers_linear=profile.powers_linear,
        max_doppler_hz=float(data["max_doppler_hz"]) if max_doppler_hz is None else float(max_doppler_hz),
    )


def estimate_empirical_grid_statistics(
    train_path: Path,
    key: str = "h_true_grid",
    batch_frames: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """用训练集估计二维时频信道均值和协方差，作为非 oracle LMMSE 先验。

    这个 baseline 不读取测试集真实 PDP，只使用训练集统计量。协方差在 4 个
    MIMO 收发链路之间共享，对应常见的 i.i.d. link 仿真假设。
    """

    with np.load(train_path) as train_data:
        h_grid = train_data[key]
        n_samples, n_symbols, n_subcarriers, n_rx, n_tx = h_grid.shape
        grid_size = n_symbols * n_subcarriers
        n_observations = n_samples * n_rx * n_tx
        mean = np.zeros(grid_size, dtype=np.complex128)

        for start in range(0, n_samples, batch_frames):
            chunk = h_grid[start : start + batch_frames]
            links = np.transpose(chunk, (0, 3, 4, 1, 2)).reshape(-1, grid_size).astype(np.complex128)
            mean += np.sum(links, axis=0)
        mean /= float(n_observations)

        cov = np.zeros((grid_size, grid_size), dtype=np.complex128)
        for start in range(0, n_samples, batch_frames):
            chunk = h_grid[start : start + batch_frames]
            links = np.transpose(chunk, (0, 3, 4, 1, 2)).reshape(-1, grid_size).astype(np.complex128)
            centered = links - mean[None, :]
            cov += centered.T @ centered.conj()
        cov /= float(max(n_observations - 1, 1))

    return mean, cov


def empirical_lmmse_grid_estimate(
    data: np.lib.npyio.NpzFile,
    mean: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    return lmmse_grid_estimate_from_covariance(data, cov, mean)


@torch.no_grad()
def torch_nmse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        err = torch.sum((pred - y) ** 2, dim=(1, 2, 3))
        ref = torch.sum(y**2, dim=(1, 2, 3)).clamp_min(1e-12)
        total += float(torch.sum(err / ref).item())
        count += int(x.shape[0])
    nmse = total / max(count, 1)
    return nmse, nmse_db(nmse)


@torch.no_grad()
def torch_nmse_pf_msbnet(
    model: PilotFittedMIMOSharedBasisNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total = 0.0
    count = 0
    for h_pilot, y in loader:
        h_pilot = h_pilot.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(h_pilot)
        err = torch.sum((pred - y) ** 2, dim=(1, 2, 3))
        ref = torch.sum(y**2, dim=(1, 2, 3)).clamp_min(1e-12)
        total += float(torch.sum(err / ref).item())
        count += int(y.shape[0])
    nmse = total / max(count, 1)
    return nmse, nmse_db(nmse)


def load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[str, torch.nn.Module, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") == "pf_msbnet":
        pilot_positions = checkpoint["pilot_positions"]
        model = PilotFittedMIMOSharedBasisNet(
            pilot_positions=pilot_positions,
            n_symbols=int(checkpoint["n_symbols"]),
            n_subcarriers=int(checkpoint["n_subcarriers"]),
            n_rx=int(checkpoint.get("n_rx", 2)),
            n_tx=int(checkpoint.get("n_tx", 2)),
            num_basis=int(checkpoint.get("num_basis", 32)),
            hidden_channels=int(checkpoint.get("hidden_channels", 128)),
            pos_bands=int(checkpoint.get("pos_bands", 6)),
            min_regularization=float(checkpoint.get("min_regularization", 1e-4)),
            use_attention=bool(checkpoint.get("use_attention", False)),
            attention_heads=int(checkpoint.get("attention_heads", 4)),
            attention_layers=int(checkpoint.get("attention_layers", 1)),
            use_learned_pilot_weights=bool(checkpoint.get("use_learned_pilot_weights", True)),
            use_learned_regularization=bool(checkpoint.get("use_learned_regularization", True)),
            use_basis_gate=bool(checkpoint.get("use_basis_gate", True)),
            gate_temperature=float(checkpoint.get("gate_temperature", 1.0)),
            attention_dropout=float(checkpoint.get("attention_dropout", 0.0)),
            gate_dropout=float(checkpoint.get("gate_dropout", 0.0)),
            basis_dropout=float(checkpoint.get("basis_dropout", 0.0)),
            pilot_noise_std=float(checkpoint.get("pilot_noise_std", 0.0)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        scale = float(checkpoint.get("scale", 1.0))
        params = int(checkpoint.get("param_count", count_parameters(model)))
        return "pf_msbnet", model, scale, params

    if checkpoint.get("model_type") == "plern":
        base_meta = checkpoint["base_model"]
        base_model_name = str(base_meta["model"])
        in_channels = int(checkpoint.get("in_channels", 8))
        base_hidden_channels = int(base_meta.get("hidden_channels", 64))
        base_depth = int(base_meta.get("depth", 6))
        refiner_hidden_channels = int(checkpoint.get("refiner_hidden_channels", 64))
        refiner_depth = int(checkpoint.get("refiner_depth", 4))
        pilot_mask = checkpoint.get("pilot_mask")
        if pilot_mask is None:
            pilot_mask = checkpoint["model"].get("pilot_mask")
        base_model = build_model(base_model_name, in_channels, base_hidden_channels, base_depth).to(device)
        model = PilotLockedErrorRefinementNet(
            base_model=base_model,
            in_channels=in_channels,
            hidden_channels=refiner_hidden_channels,
            depth=refiner_depth,
            pilot_mask=pilot_mask,
            freeze_base=True,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        scale = float(checkpoint.get("scale", 1.0))
        params = int(checkpoint.get("param_count", count_parameters(model)))
        return "plern", model, scale, params

    args = checkpoint.get("args", {})
    model_name = str(args.get("model", path.stem.split("_")[0]))
    in_channels = int(checkpoint.get("in_channels", 8))
    hidden_channels = int(args.get("hidden_channels", 64))
    depth = int(args.get("depth", 6))
    scale = float(checkpoint.get("scale", 1.0))
    model = build_model(model_name, in_channels, hidden_channels, depth).to(device)
    model.load_state_dict(checkpoint["model"])
    return model_name, model, scale, count_parameters(model)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(path: Path, rows: list[dict[str, object]]) -> None:
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    values = {
        (str(row["dataset"]), str(row["method"])): float(row["nmse_db"])
        for row in rows
    }

    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    plt.figure(figsize=(max(8, 1.6 * len(datasets)), 4.8))
    for idx, method in enumerate(methods):
        y = [values.get((dataset, method), np.nan) for dataset in datasets]
        offset = (idx - (len(methods) - 1) / 2.0) * width
        plt.bar(x + offset, y, width=width, label=method)

    plt.ylabel("NMSE (dB)")
    plt.xlabel("Test dataset")
    plt.title("Grid MIMO-OFDM Channel Estimation Comparison")
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LS, SRCNN, ChannelNet, and other grid CNN methods.")
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Model checkpoint in NAME=PATH format. Can be repeated.",
    )
    parser.add_argument("--input-key", default="h_ls_grid_ri")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "comparison")
    parser.add_argument("--include-empirical-lmmse", action="store_true")
    parser.add_argument("--empirical-lmmse-train", type=Path, default=None)
    parser.add_argument("--empirical-lmmse-key", default="h_true_grid")
    parser.add_argument("--empirical-lmmse-batch-frames", type=int, default=512)
    parser.add_argument("--include-oracle-lmmse", action="store_true")
    parser.add_argument("--include-mismatched-lmmse", action="store_true")
    parser.add_argument("--lmmse-profile", default="tdl-a")
    parser.add_argument("--lmmse-delay-spread-ns", type=float, default=300.0)
    parser.add_argument("--lmmse-max-doppler-hz", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []

    empirical_stats: tuple[np.ndarray, np.ndarray] | None = None
    if args.include_empirical_lmmse or args.empirical_lmmse_train is not None:
        if args.empirical_lmmse_train is None:
            raise ValueError("--include-empirical-lmmse requires --empirical-lmmse-train")
        print(f"estimating empirical LMMSE covariance from: {args.empirical_lmmse_train}")
        empirical_stats = estimate_empirical_grid_statistics(
            train_path=args.empirical_lmmse_train,
            key=args.empirical_lmmse_key,
            batch_frames=args.empirical_lmmse_batch_frames,
        )

    model_specs = [parse_model_spec(spec) for spec in args.checkpoint]
    loaded_models: list[tuple[str, str, torch.nn.Module, float, int, Path]] = []
    for display_name, checkpoint_path in model_specs:
        if not checkpoint_path.exists():
            print(f"skip missing checkpoint: {checkpoint_path}")
            continue
        model_name, model, scale, params = load_model_from_checkpoint(checkpoint_path, device)
        loaded_models.append((display_name, model_name, model, scale, params, checkpoint_path))
        print(f"loaded {display_name}: model={model_name}, params={params}, scale={scale:.6g}")

    for test_path in args.test:
        data = np.load(test_path)
        dataset_name = test_path.stem
        x = data[args.input_key].astype(np.float32)
        y = data[args.target_key].astype(np.float32)
        ls_nmse, ls_nmse_db = np_nmse(x, y)
        rows.append(
            {
                "dataset": dataset_name,
                "method": "LS + 2D interp.",
                "nmse": ls_nmse,
                "nmse_db": ls_nmse_db,
                "params": 0,
                "checkpoint": "",
            }
        )
        print(f"{dataset_name} | LS + 2D interp.: {ls_nmse_db:.3f} dB")

        if empirical_stats is not None:
            empirical_mean, empirical_cov = empirical_stats
            h_empirical = empirical_lmmse_grid_estimate(data, empirical_mean, empirical_cov)
            empirical_ri = complex_grid_to_ri(h_empirical)
            empirical_nmse, empirical_nmse_db = np_nmse(empirical_ri, y)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": "Empirical 2D LMMSE (train covariance)",
                    "nmse": empirical_nmse,
                    "nmse_db": empirical_nmse_db,
                    "params": 0,
                    "checkpoint": str(args.empirical_lmmse_train),
                }
            )
            print(f"{dataset_name} | Empirical 2D LMMSE (train covariance): {empirical_nmse_db:.3f} dB")

        if args.include_oracle_lmmse:
            h_lmmse = oracle_lmmse_grid_estimate(data)
            lmmse_ri = complex_grid_to_ri(h_lmmse)
            lmmse_nmse, lmmse_nmse_db = np_nmse(lmmse_ri, y)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": "Oracle 2D LMMSE",
                    "nmse": lmmse_nmse,
                    "nmse_db": lmmse_nmse_db,
                    "params": 0,
                    "checkpoint": "",
                }
            )
            print(f"{dataset_name} | Oracle 2D LMMSE: {lmmse_nmse_db:.3f} dB")

        if args.include_mismatched_lmmse:
            h_mismatch = mismatched_lmmse_grid_estimate(
                data,
                profile_name=args.lmmse_profile,
                delay_spread_ns=args.lmmse_delay_spread_ns,
                max_doppler_hz=args.lmmse_max_doppler_hz,
            )
            mismatch_ri = complex_grid_to_ri(h_mismatch)
            mismatch_nmse, mismatch_nmse_db = np_nmse(mismatch_ri, y)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": f"Mismatched 2D LMMSE ({args.lmmse_profile})",
                    "nmse": mismatch_nmse,
                    "nmse_db": mismatch_nmse_db,
                    "params": 0,
                    "checkpoint": "",
                }
            )
            print(f"{dataset_name} | Mismatched 2D LMMSE ({args.lmmse_profile}): {mismatch_nmse_db:.3f} dB")

        for display_name, model_name, model, scale, params, checkpoint_path in loaded_models:
            if model_name == "pf_msbnet":
                eval_set = PilotFittedEvalDataset(test_path, scale, args.target_key)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                checkpoint_positions = checkpoint["pilot_positions"].cpu().numpy()
                if not np.array_equal(checkpoint_positions, eval_set.pilot_positions):
                    print(f"skip {display_name} on {dataset_name}: pilot_positions differ")
                    continue
                loader = DataLoader(
                    eval_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )
                model_nmse, model_nmse_db = torch_nmse_pf_msbnet(model, loader, device)
            else:
                eval_set = GridEvalDataset(test_path, args.input_key, args.target_key, scale)
                loader = DataLoader(
                    eval_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )
                model_nmse, model_nmse_db = torch_nmse(model, loader, device)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": display_name,
                    "nmse": model_nmse,
                    "nmse_db": model_nmse_db,
                    "params": params,
                    "checkpoint": str(checkpoint_path),
                }
            )
            print(f"{dataset_name} | {display_name}: {model_nmse_db:.3f} dB")

    csv_path = args.outdir / "grid_method_comparison.csv"
    json_path = args.outdir / "grid_method_comparison.json"
    fig_path = args.outdir / "grid_method_comparison.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_grouped_bars(fig_path, rows)
    print(f"saved csv: {csv_path}")
    print(f"saved json: {json_path}")
    print(f"saved figure: {fig_path}")


if __name__ == "__main__":
    main()
