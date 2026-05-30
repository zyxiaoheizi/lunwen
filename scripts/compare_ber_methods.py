from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from adapter_mimo_ofdm.models import PilotFittedMIMOSharedBasisNet  # noqa: E402
from compare_grid_methods import (  # noqa: E402
    empirical_lmmse_grid_estimate,
    estimate_separable_lmmse_statistics,
    estimate_empirical_grid_statistics,
    load_model_from_checkpoint,
    low_rank_lmmse_grid_estimate,
    parse_model_spec,
)


METHOD_LABELS = {
    "LS + 2D interp.": "LS",
    "Paper LMMSE (sample covariance)": "Ideal LMMSE",
    "ALMMSE": "ALMMSE",
    "ChannelNet": "ChannelNet",
    "ReEsNet": "ReEsNet",
    "PF-MSBNet-A tuned": "Proposed",
    "PF-MSBNet-A": "Proposed",
}

COLORS = {
    "LS": "#7f7f7f",
    "Ideal LMMSE": "#1f77b4",
    "ALMMSE": "#4c78a8",
    "ChannelNet": "#ff7f0e",
    "ReEsNet": "#2ca02c",
    "Proposed": "#d62728",
}

MARKERS = {
    "LS": "o",
    "Ideal LMMSE": "o",
    "ALMMSE": "o",
    "ChannelNet": "o",
    "ReEsNet": "o",
    "Proposed": "o",
}

LINESTYLES = {
    "LS": "-",
    "Ideal LMMSE": "-",
    "ALMMSE": "-",
    "ChannelNet": "-",
    "ReEsNet": "-",
    "Proposed": "-",
}

HOLLOW_MARKERS = {"ChannelNet", "ALMMSE"}


def parse_snr(dataset: str) -> float:
    match = re.search(r"_snr(m?\d+(?:p\d+)?)_test_", dataset)
    if not match:
        raise ValueError(f"Cannot parse SNR from dataset name: {dataset}")
    return float(match.group(1).replace("m", "-").replace("p", "."))


def ri_to_complex_grid(value: np.ndarray, n_rx: int = 2, n_tx: int = 2) -> np.ndarray:
    n_links = n_rx * n_tx
    real = value[:, :n_links]
    imag = value[:, n_links : 2 * n_links]
    links = real + 1j * imag
    return links.reshape(value.shape[0], n_rx, n_tx, value.shape[2], value.shape[3]).transpose(0, 3, 4, 1, 2)


def predict_model_grid(
    model: torch.nn.Module,
    model_name: str,
    data: np.lib.npyio.NpzFile,
    scale: float,
    device: torch.device,
    batch_size: int,
    input_key: str,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    if model_name == "pf_msbnet" and isinstance(model, PilotFittedMIMOSharedBasisNet):
        source = data["h_pilot_ls"].astype(np.complex64) / scale
        with torch.no_grad():
            for start in range(0, source.shape[0], batch_size):
                batch = torch.from_numpy(source[start : start + batch_size]).to(device)
                pred = model(batch).detach().cpu().numpy() * scale
                preds.append(pred)
    else:
        source = data[input_key].astype(np.float32) / scale
        with torch.no_grad():
            for start in range(0, source.shape[0], batch_size):
                batch = torch.from_numpy(source[start : start + batch_size]).to(device)
                pred = model(batch).detach().cpu().numpy() * scale
                preds.append(pred)
    pred_ri = np.concatenate(preds, axis=0)
    return ri_to_complex_grid(pred_ri, n_rx=int(data["n_rx"]), n_tx=int(data["n_tx"]))


def qpsk_ber_mmse(
    h_true: np.ndarray,
    h_est: np.ndarray,
    noise_var: np.ndarray,
    pilot_positions: np.ndarray,
    seed: int,
    batch_frames: int,
    exclude_pilots: bool,
) -> tuple[float, int, int]:
    rng = np.random.default_rng(seed)
    n_frames, n_symbols, n_subcarriers, n_rx, n_tx = h_true.shape
    data_mask = np.ones((n_symbols, n_subcarriers), dtype=bool)
    if exclude_pilots:
        data_mask[pilot_positions[:, 0], pilot_positions[:, 1]] = False
    eye = np.eye(n_tx, dtype=np.complex128).reshape(1, 1, 1, n_tx, n_tx)
    bit_errors = 0
    bit_count = 0

    for start in range(0, n_frames, batch_frames):
        end = min(start + batch_frames, n_frames)
        h_chunk = h_true[start:end].astype(np.complex128)
        h_est_chunk = h_est[start:end].astype(np.complex128)
        sigma2 = noise_var[start:end].astype(np.float64).reshape(-1, 1, 1, 1, 1)
        bits = rng.integers(0, 2, size=(end - start, n_symbols, n_subcarriers, n_tx, 2), dtype=np.int8)
        real = 1.0 - 2.0 * bits[..., 0].astype(np.float64)
        imag = 1.0 - 2.0 * bits[..., 1].astype(np.float64)
        symbols = (real + 1j * imag) / np.sqrt(2.0)
        y = np.einsum("bsfrt,bsft->bsfr", h_chunk, symbols)
        noise_scale = np.sqrt(noise_var[start:end].reshape(-1, 1, 1, 1) / 2.0)
        noise = noise_scale * (
            rng.standard_normal(size=y.shape) + 1j * rng.standard_normal(size=y.shape)
        )
        y = y + noise

        h_herm = np.conj(np.swapaxes(h_est_chunk, -1, -2))
        gram = np.einsum("bsftr,bsfru->bsftu", h_herm, h_est_chunk)
        rhs = np.einsum("bsftr,bsfr->bsft", h_herm, y)
        system = gram + sigma2 * eye + 1e-8 * eye
        solved = np.linalg.solve(
            system.reshape(-1, n_tx, n_tx),
            rhs.reshape(-1, n_tx, 1),
        ).reshape(end - start, n_symbols, n_subcarriers, n_tx)

        detected = np.empty(bits.shape, dtype=np.int8)
        detected[..., 0] = (solved.real < 0).astype(np.int8)
        detected[..., 1] = (solved.imag < 0).astype(np.int8)
        bit_errors += int(np.count_nonzero(detected[:, data_mask, :, :] != bits[:, data_mask, :, :]))
        bit_count += int(bits[:, data_mask, :, :].size)

    return bit_errors / max(bit_count, 1), bit_errors, bit_count


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_ber(path: Path, rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.9,
            "legend.fontsize": 5.9,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.50,
            "grid.linewidth": 0.20,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    methods = []
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            methods.append(method)
    plt.figure(figsize=(3.35, 2.12))
    ax = plt.gca()
    for method in methods:
        curve = sorted((row for row in rows if row["method"] == method), key=lambda item: float(item["snr_db"]))
        x = [float(row["snr_db"]) for row in curve]
        y = [
            float(row["ber_db"]) if row.get("ber_db") not in (None, "") else 10.0 * np.log10(max(float(row["ber"]), 1e-9))
            for row in curve
        ]
        ax.plot(
            x,
            y,
            marker=MARKERS.get(method, "o"),
            markersize=1.85 if method == "Proposed" else 1.55,
            color=COLORS.get(method),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=0.82 if method == "Proposed" else 0.72 if method in {"Ideal LMMSE", "ALMMSE"} else 0.68,
            alpha=1.0 if method in {"Proposed", "Ideal LMMSE", "ALMMSE"} else 0.88,
            markerfacecolor="white" if method in HOLLOW_MARKERS else COLORS.get(method),
            markeredgecolor=COLORS.get(method),
            markeredgewidth=0.30,
            label=method,
            zorder=5 if method == "Proposed" else 4 if method in {"Ideal LMMSE", "ALMMSE"} else 3,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER (dB)")
    ax.grid(True, linestyle="--", alpha=0.30)
    legend = ax.legend(
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#555555",
        loc="upper right",
        borderpad=0.22,
        columnspacing=0.52,
        handlelength=1.05,
        handletextpad=0.32,
    )
    legend.get_frame().set_linewidth(0.32)
    plt.tight_layout(pad=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QPSK BER using estimated MIMO-OFDM channels.")
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="Model checkpoint in NAME=PATH format.")
    parser.add_argument("--input-key", default="h_ls_grid_ri")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ber-batch-frames", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "ber_curve")
    parser.add_argument("--include-paper-lmmse", action="store_true")
    parser.add_argument("--paper-lmmse-train", type=Path, default=None)
    parser.add_argument("--paper-lmmse-key", default="h_true_grid")
    parser.add_argument("--paper-lmmse-batch-frames", type=int, default=512)
    parser.add_argument("--include-almmse", action="store_true")
    parser.add_argument("--almmse-train", type=Path, default=None)
    parser.add_argument("--almmse-key", default="h_true_grid")
    parser.add_argument("--almmse-batch-frames", type=int, default=512)
    parser.add_argument("--almmse-time-rank", type=int, default=1)
    parser.add_argument("--almmse-freq-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--include-pilots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    paper_stats = None
    if args.include_paper_lmmse:
        if args.paper_lmmse_train is None:
            raise ValueError("--include-paper-lmmse requires --paper-lmmse-train")
        print(f"estimating paper-style LMMSE second moment from: {args.paper_lmmse_train}")
        paper_stats = estimate_empirical_grid_statistics(
            train_path=args.paper_lmmse_train,
            key=args.paper_lmmse_key,
            batch_frames=args.paper_lmmse_batch_frames,
            center=False,
        )

    almmse_stats = None
    if args.include_almmse:
        if args.almmse_train is None:
            raise ValueError("--include-almmse requires --almmse-train")
        print(
            "estimating ALMMSE low-rank separable prior from: "
            f"{args.almmse_train} (time_rank={args.almmse_time_rank}, freq_rank={args.almmse_freq_rank})"
        )
        almmse_stats = estimate_separable_lmmse_statistics(
            train_path=args.almmse_train,
            key=args.almmse_key,
            time_rank=args.almmse_time_rank,
            freq_rank=args.almmse_freq_rank,
            batch_frames=args.almmse_batch_frames,
            center=True,
        )

    loaded_models = []
    for spec in args.checkpoint:
        display_name, checkpoint_path = parse_model_spec(spec)
        if not checkpoint_path.exists():
            print(f"skip missing checkpoint: {checkpoint_path}")
            continue
        model_name, model, scale, params = load_model_from_checkpoint(checkpoint_path, device)
        loaded_models.append((METHOD_LABELS.get(display_name, display_name), model_name, model, scale, params, checkpoint_path))
        print(f"loaded {display_name}: model={model_name}, params={params}, scale={scale:.6g}")

    rows: list[dict[str, object]] = []
    for dataset_index, test_path in enumerate(args.test):
        data = np.load(test_path)
        dataset = test_path.stem
        snr = parse_snr(dataset)
        h_true = data["h_true_grid"].astype(np.complex64)
        noise_var = data["noise_var"].astype(np.float64)
        pilot_positions = data["pilot_positions"].astype(np.int64)
        estimates: list[tuple[str, np.ndarray, int, str]] = [
            ("LS", ri_to_complex_grid(data[args.input_key].astype(np.float32), int(data["n_rx"]), int(data["n_tx"])), 0, "")
        ]
        if paper_stats is not None:
            paper_mean, paper_cov = paper_stats
            estimates.append(
                (
                    "Ideal LMMSE",
                    empirical_lmmse_grid_estimate(data, paper_mean, paper_cov),
                    0,
                    str(args.paper_lmmse_train),
                )
            )
        if almmse_stats is not None:
            almmse_mean, almmse_basis, almmse_eigvals = almmse_stats
            estimates.append(
                (
                    "ALMMSE",
                    low_rank_lmmse_grid_estimate(data, almmse_mean, almmse_basis, almmse_eigvals),
                    0,
                    str(args.almmse_train),
                )
            )
        for display_name, model_name, model, scale, params, checkpoint_path in loaded_models:
            estimates.append(
                (
                    display_name,
                    predict_model_grid(model, model_name, data, scale, device, args.batch_size, args.input_key),
                    params,
                    str(checkpoint_path),
                )
            )

        for method, h_est, params, checkpoint in estimates:
            ber, errors, total = qpsk_ber_mmse(
                h_true=h_true,
                h_est=h_est,
                noise_var=noise_var,
                pilot_positions=pilot_positions,
                seed=args.seed + dataset_index,
                batch_frames=args.ber_batch_frames,
                exclude_pilots=not args.include_pilots,
            )
            row = {
                "dataset": dataset,
                "snr_db": snr,
                "method": method,
                "ber": ber,
                "ber_db": 10.0 * np.log10(max(ber, 1e-9)),
                "bit_errors": errors,
                "total_bits": total,
                "params": params,
                "checkpoint": checkpoint,
            }
            rows.append(row)
            print(f"{dataset} | {method}: BER={ber:.4e}, {row['ber_db']:.3f} dB ({errors}/{total})")

    csv_path = args.outdir / "ber_curve.csv"
    json_path = args.outdir / "ber_curve.json"
    fig_path = args.outdir / "ber_curve"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_ber(fig_path, rows)
    print(f"saved csv: {csv_path}")
    print(f"saved json: {json_path}")
    print(f"saved figure: {fig_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
