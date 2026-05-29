from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_mimo_ofdm.models import FixedSharedBasisRidgeNet, count_parameters  # noqa: E402
from train_pf_msbnet import (  # noqa: E402
    PilotFittedGridDataset,
    compute_scale,
    evaluate,
    pca_basis_from_training_file,
    serializable_args,
)


def dft_basis(n_symbols: int, n_subcarriers: int, num_basis: int) -> torch.Tensor:
    """Low-frequency 2D complex exponential basis for a BEM-style baseline."""

    time = torch.arange(n_symbols, dtype=torch.float32)
    freq = torch.arange(n_subcarriers, dtype=torch.float32)
    time_bins = list(range(n_symbols))
    freq_bins = list(range(n_subcarriers))
    time_bins.sort(key=lambda value: min(value, n_symbols - value))
    freq_bins.sort(key=lambda value: min(value, n_subcarriers - value))

    candidates: list[tuple[int, int, int]] = []
    for kt in time_bins:
        for kf in freq_bins:
            radius = min(kt, n_symbols - kt) ** 2 + min(kf, n_subcarriers - kf) ** 2
            candidates.append((radius, kt, kf))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    basis = []
    norm = float(np.sqrt(n_symbols * n_subcarriers))
    for _, kt, kf in candidates[:num_basis]:
        phase_t = 2.0 * torch.pi * float(kt) * time / float(n_symbols)
        phase_f = 2.0 * torch.pi * float(kf) * freq / float(n_subcarriers)
        atom = torch.exp(1j * (phase_t[:, None] + phase_f[None, :])) / norm
        basis.append(atom)
    return torch.stack(basis, dim=0).to(dtype=torch.complex64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/evaluate a fixed shared-basis ridge baseline.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument("--num-basis", type=int, default=32)
    parser.add_argument("--basis-init", choices=["pca", "dft"], default="pca")
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--pca-max-observations", type=int, default=8192)
    parser.add_argument("--pca-seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "fixed_basis_ridge")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--pilot-key", default="h_pilot_ls")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    scale = 1.0 if args.no_normalize else compute_scale(args.train, args.target_key)

    train_set = PilotFittedGridDataset(args.train, scale, args.pilot_key, args.target_key)
    val_set = PilotFittedGridDataset(args.val, scale, args.pilot_key, args.target_key)
    if not np.array_equal(train_set.pilot_positions, val_set.pilot_positions):
        raise ValueError("train and val pilot_positions differ; fixed-basis baseline expects fixed pilots.")

    n_rx = int(train_set.h_pilot.shape[2])
    n_tx = int(train_set.h_pilot.shape[3])
    n_symbols = int(train_set.target.shape[2])
    n_subcarriers = int(train_set.target.shape[3])
    pilot_positions = torch.from_numpy(train_set.pilot_positions)

    if args.basis_init == "pca":
        basis = pca_basis_from_training_file(
            args.train,
            scale=scale,
            num_basis=args.num_basis,
            max_observations=args.pca_max_observations,
            seed=args.pca_seed,
        )
    else:
        basis = dft_basis(n_symbols, n_subcarriers, args.num_basis)

    model = FixedSharedBasisRidgeNet(
        pilot_positions=pilot_positions,
        basis=basis,
        n_rx=n_rx,
        n_tx=n_tx,
        regularization=args.regularization,
    ).to(device)

    criterion = torch.nn.MSELoss()
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_metrics = evaluate(model, val_loader, device, criterion)
    print(f"device={device}")
    print(
        f"model=fixed_basis_ridge, basis={args.basis_init}, num_basis={args.num_basis}, "
        f"lambda={args.regularization}, scale={scale:.6g}"
    )
    print(f"val_nmse={val_metrics['nmse_db']:.3f} dB, loss={val_metrics['loss']:.6e}")

    test_results: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = [
        {
            "dataset": str(args.val),
            "split": "val",
            "nmse": val_metrics["nmse"],
            "nmse_db": val_metrics["nmse_db"],
            "loss": val_metrics["loss"],
        }
    ]
    for test_path in args.test:
        test_set = PilotFittedGridDataset(test_path, scale, args.pilot_key, args.target_key)
        if not np.array_equal(train_set.pilot_positions, test_set.pilot_positions):
            raise ValueError(f"test pilot_positions differ: {test_path}")
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        result = evaluate(model, test_loader, device, criterion)
        test_results[str(test_path)] = result
        rows.append(
            {
                "dataset": str(test_path),
                "split": "test",
                "nmse": result["nmse"],
                "nmse_db": result["nmse_db"],
                "loss": result["loss"],
            }
        )
        print(f"test {test_path}: nmse={result['nmse_db']:.3f} dB, loss={result['loss']:.6e}")

    checkpoint_path = args.outdir / "fixed_basis_ridge_grid_best.pt"
    metrics_path = args.outdir / "fixed_basis_ridge_grid_metrics.csv"
    summary_path = args.outdir / "fixed_basis_ridge_grid_summary.json"
    state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
    torch.save(
        {
            "model_type": "fixed_basis_ridge",
            "model": state_dict,
            "args": serializable_args(args),
            "scale": scale,
            "n_rx": n_rx,
            "n_tx": n_tx,
            "n_symbols": n_symbols,
            "n_subcarriers": n_subcarriers,
            "pilot_positions": pilot_positions,
            "num_basis": args.num_basis,
            "basis_init": args.basis_init,
            "regularization": args.regularization,
            "basis": basis.detach().clone(),
            "best_val_nmse": val_metrics["nmse"],
            "param_count": count_parameters(model),
        },
        checkpoint_path,
    )
    write_metrics(metrics_path, rows)
    summary = {
        "model": "fixed_basis_ridge",
        "param_count": count_parameters(model),
        "scale": scale,
        "num_basis": args.num_basis,
        "basis_init": args.basis_init,
        "regularization": args.regularization,
        "best_val_nmse": val_metrics["nmse"],
        "best_val_nmse_db": val_metrics["nmse_db"],
        "best_checkpoint": str(checkpoint_path),
        "metrics_csv": str(metrics_path),
        "test_results": test_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
