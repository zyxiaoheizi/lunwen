from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_mimo_ofdm.models import AdaptiveMMSELinearFilterNet, count_parameters  # noqa: E402
from train_pf_msbnet import (  # noqa: E402
    PilotFittedGridDataset,
    compute_scale,
    evaluate,
    nmse_db_from_linear,
    serializable_args,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an A-MMSE-like learned linear filter baseline.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--rank", type=int, default=0, help="0 means full learned W; positive values use low-rank W.")
    parser.add_argument("--init-lmmse", action="store_true", help="Initialize full W from the training-set LMMSE solution.")
    parser.add_argument("--lmmse-regularization", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "ammse_filter")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--pilot-key", default="h_pilot_ls")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def initialize_full_filter_from_lmmse(
    model: AdaptiveMMSELinearFilterNet,
    train_path: Path,
    scale: float,
    lmmse_regularization: float,
    pilot_key: str,
) -> None:
    if model.rank > 0:
        raise ValueError("--init-lmmse currently supports only full-rank A-MMSE-like filters with rank=0.")
    data = np.load(train_path)
    h_pilot = data[pilot_key].astype(np.complex64) / np.float32(scale)
    h_grid = data["h_true_grid"].astype(np.complex64) / np.float32(scale)
    n_samples, n_symbols, n_subcarriers, n_rx, n_tx = h_grid.shape
    num_pilots = h_pilot.shape[1]
    grid_size = n_symbols * n_subcarriers

    x = np.transpose(h_pilot, (0, 2, 3, 1)).reshape(-1, num_pilots).astype(np.complex128)
    y = np.transpose(h_grid, (0, 3, 4, 1, 2)).reshape(-1, grid_size).astype(np.complex128)
    gram = x.conj().T @ x
    gram = gram + float(lmmse_regularization) * np.eye(num_pilots, dtype=np.complex128)
    rhs = x.conj().T @ y
    coefficients = np.linalg.solve(gram, rhs)
    weight = coefficients.T.astype(np.complex64)
    with torch.no_grad():
        model.filter[0].copy_(torch.from_numpy(weight.real).to(device=model.filter.device, dtype=model.filter.dtype))
        model.filter[1].copy_(torch.from_numpy(weight.imag).to(device=model.filter.device, dtype=model.filter.dtype))


def save_checkpoint(
    path: Path,
    model: AdaptiveMMSELinearFilterNet,
    args: argparse.Namespace,
    scale: float,
    n_rx: int,
    n_tx: int,
    n_symbols: int,
    n_subcarriers: int,
    pilot_positions: torch.Tensor,
    best_nmse: float,
) -> None:
    torch.save(
        {
            "model_type": "ammse_filter",
            "model": model.state_dict(),
            "args": serializable_args(args),
            "scale": scale,
            "n_rx": n_rx,
            "n_tx": n_tx,
            "n_symbols": n_symbols,
            "n_subcarriers": n_subcarriers,
            "pilot_positions": pilot_positions,
            "rank": args.rank,
            "best_val_nmse": best_nmse,
            "param_count": count_parameters(model),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    scale = 1.0 if args.no_normalize else compute_scale(args.train, args.target_key)

    train_set = PilotFittedGridDataset(args.train, scale, args.pilot_key, args.target_key)
    val_set = PilotFittedGridDataset(args.val, scale, args.pilot_key, args.target_key)
    if not np.array_equal(train_set.pilot_positions, val_set.pilot_positions):
        raise ValueError("train and val pilot_positions differ; A-MMSE-like baseline expects fixed pilots.")

    n_rx = int(train_set.h_pilot.shape[2])
    n_tx = int(train_set.h_pilot.shape[3])
    n_symbols = int(train_set.target.shape[2])
    n_subcarriers = int(train_set.target.shape[3])
    pilot_positions = torch.from_numpy(train_set.pilot_positions)

    model = AdaptiveMMSELinearFilterNet(
        pilot_positions=pilot_positions,
        n_symbols=n_symbols,
        n_subcarriers=n_subcarriers,
        n_rx=n_rx,
        n_tx=n_tx,
        rank=args.rank,
    ).to(device)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_name = "ammse_filter_grid"
    best_path = args.outdir / f"{run_name}_best.pt"
    last_path = args.outdir / f"{run_name}_last.pt"
    metrics_path = args.outdir / f"{run_name}_metrics.csv"
    summary_path = args.outdir / f"{run_name}_summary.json"

    print(f"device={device}")
    print(f"model=ammse_filter, params={count_parameters(model)}, rank={'full' if args.rank <= 0 else args.rank}")
    print(f"train={args.train}, val={args.val}, scale={scale:.6g}")
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}")
    if args.init_lmmse:
        print(f"initializing A-MMSE-like filter from sample LMMSE: regularization={args.lmmse_regularization}")
        initialize_full_filter_from_lmmse(model, args.train, scale, args.lmmse_regularization, args.pilot_key)

    rows: list[dict[str, float | int | str]] = []
    val_metrics = evaluate(model, val_loader, device, criterion)
    best_nmse = val_metrics["nmse"]
    best_epoch = 0
    epochs_without_improvement = 0
    rows.append(
        {
            "epoch": 0,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": 0.0,
            "val_loss": val_metrics["loss"],
            "val_nmse": val_metrics["nmse"],
            "val_nmse_db": val_metrics["nmse_db"],
        }
    )
    print(f"epoch 000 | val_nmse={val_metrics['nmse_db']:.3f} dB")
    save_checkpoint(
        best_path,
        model,
        args,
        scale,
        n_rx,
        n_tx,
        n_symbols,
        n_subcarriers,
        pilot_positions,
        best_nmse,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for h_pilot, target in train_loader:
            h_pilot = h_pilot.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(h_pilot)
            loss = criterion(pred, target)
            if torch.isnan(loss):
                raise FloatingPointError("Training loss became NaN. Try smaller LR.")
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            batch = target.shape[0]
            train_loss += float(loss.item()) * batch
            train_count += batch
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_metrics["loss"],
            "val_nmse": val_metrics["nmse"],
            "val_nmse_db": val_metrics["nmse_db"],
        }
        rows.append(row)
        print(
            f"epoch {epoch:03d} | train_loss={row['train_loss']:.6e} | "
            f"val_nmse={row['val_nmse_db']:.3f} dB"
        )

        improved = val_metrics["nmse"] < best_nmse - args.early_stopping_min_delta
        if improved:
            best_nmse = val_metrics["nmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                args,
                scale,
                n_rx,
                n_tx,
                n_symbols,
                n_subcarriers,
                pilot_positions,
                best_nmse,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"early stop at epoch {epoch}: best_epoch={best_epoch}, "
                f"best_val_nmse={nmse_db_from_linear(best_nmse):.3f} dB"
            )
            break

    save_checkpoint(
        last_path,
        model,
        args,
        scale,
        n_rx,
        n_tx,
        n_symbols,
        n_subcarriers,
        pilot_positions,
        best_nmse,
    )
    write_metrics(metrics_path, rows)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    test_results: dict[str, dict[str, float]] = {}
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
        print(f"test {test_path}: nmse={result['nmse_db']:.3f} dB, loss={result['loss']:.6e}")

    summary = {
        "model": "ammse_filter",
        "param_count": count_parameters(model),
        "scale": scale,
        "rank": args.rank,
        "best_val_nmse": best_nmse,
        "best_val_nmse_db": nmse_db_from_linear(best_nmse),
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "metrics_csv": str(metrics_path),
        "test_results": test_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
