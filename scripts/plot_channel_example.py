from __future__ import annotations

import argparse
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
from compare_grid_methods import load_model_from_checkpoint  # noqa: E402


def ri_to_complex_grid(value: np.ndarray, n_rx: int = 2, n_tx: int = 2) -> np.ndarray:
    n_links = n_rx * n_tx
    real = value[:, :n_links]
    imag = value[:, n_links : 2 * n_links]
    links = real + 1j * imag
    return links.reshape(value.shape[0], n_rx, n_tx, value.shape[2], value.shape[3]).transpose(0, 3, 4, 1, 2)


def predict_grid(
    model: torch.nn.Module,
    model_name: str,
    data: np.lib.npyio.NpzFile,
    scale: float,
    device: torch.device,
    sample_index: int,
) -> np.ndarray:
    model.eval()
    if model_name == "pf_msbnet" and isinstance(model, PilotFittedMIMOSharedBasisNet):
        source = data["h_pilot_ls"][sample_index : sample_index + 1].astype(np.complex64) / scale
        batch = torch.from_numpy(source).to(device)
    else:
        source = data["h_ls_grid_ri"][sample_index : sample_index + 1].astype(np.float32) / scale
        batch = torch.from_numpy(source).to(device)
    with torch.no_grad():
        pred = model(batch).detach().cpu().numpy() * scale
    return ri_to_complex_grid(pred, int(data["n_rx"]), int(data["n_tx"]))[0]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a compact channel reconstruction heatmap example.")
    parser.add_argument("--test", type=Path, default=ROOT / "data/grid/grid_p8_tdl_a_test_4000.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/p8/pf_msbnet_a_best/pf_msbnet_grid_best.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--rx", type=int, default=0)
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/letter_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = np.load(args.test)
    model_name, model, scale, _ = load_model_from_checkpoint(args.checkpoint, device)

    h_true = data["h_true_grid"][args.sample_index]
    h_ls = ri_to_complex_grid(
        data["h_ls_grid_ri"][args.sample_index : args.sample_index + 1].astype(np.float32),
        int(data["n_rx"]),
        int(data["n_tx"]),
    )[0]
    h_pred = predict_grid(model, model_name, data, scale, device, args.sample_index)
    pilot_positions = data["pilot_positions"].astype(np.int64)

    true_map = np.abs(h_true[:, :, args.rx, args.tx])
    ls_map = np.abs(h_ls[:, :, args.rx, args.tx])
    pred_map = np.abs(h_pred[:, :, args.rx, args.tx])
    err_map = np.abs(h_true[:, :, args.rx, args.tx] - h_pred[:, :, args.rx, args.tx])

    vmax = float(np.percentile(true_map, 99))
    fig, axes = plt.subplots(1, 4, figsize=(6.95, 1.85), constrained_layout=True)
    panels = [
        (true_map, r"$|H|$ true", 0.0, vmax, "viridis"),
        (ls_map, r"$|H|$ LS", 0.0, vmax, "viridis"),
        (pred_map, r"$|H|$ proposed", 0.0, vmax, "viridis"),
        (err_map, r"$|H-\hat{H}|$", 0.0, float(np.percentile(err_map, 99)), "magma"),
    ]
    for ax, (image, title, vmin, vmax_panel, cmap) in zip(axes, panels):
        im = ax.imshow(image, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_panel)
        ax.scatter(pilot_positions[:, 1], pilot_positions[:, 0], s=5, c="white", edgecolors="black", linewidths=0.25)
        ax.set_title(title)
        ax.set_xlabel("Subcarrier")
        ax.set_xticks([0, image.shape[1] - 1])
        ax.set_yticks([0, image.shape[0] - 1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("OFDM symbol")
    for ax in axes[1:]:
        ax.set_ylabel("")

    out = args.outdir / "letter_channel_example"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved channel example: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
