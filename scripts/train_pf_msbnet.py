from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapter_mimo_ofdm.models import PilotFittedMIMOSharedBasisNet, count_parameters  # noqa: E402


class PilotFittedGridDataset(Dataset):
    """Read the same grid .npz files through raw pilot observations.

    Existing CNN baselines use h_ls_grid_ri -> h_true_grid_ri. This dataset
    avoids the interpolated LS image and exposes h_pilot_ls + pilot_positions.
    """

    def __init__(
        self,
        path: str | Path,
        scale: float = 1.0,
        pilot_key: str = "h_pilot_ls",
        target_key: str = "h_true_grid_ri",
    ) -> None:
        data = np.load(path)
        self.h_pilot = data[pilot_key].astype(np.complex64) / scale
        self.target = data[target_key].astype(np.float32) / scale
        self.pilot_positions = data["pilot_positions"].astype(np.int64)
        self.path = Path(path)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.h_pilot[index]), torch.from_numpy(self.target[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PF-MSBNet on raw pilot observations.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-basis", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--pos-bands", type=int, default=6)
    parser.add_argument("--min-regularization", type=float, default=1e-4)
    parser.add_argument("--lambda-pilot", type=float, default=0.0)
    parser.add_argument("--lambda-orth", type=float, default=1e-4)
    parser.add_argument("--lambda-gate", type=float, default=1e-5)
    parser.add_argument("--basis-init", choices=["pca", "random"], default="pca")
    parser.add_argument("--pca-max-observations", type=int, default=8192)
    parser.add_argument("--pca-seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "pf_msbnet")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--pilot-key", default="h_pilot_ls")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    clean: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            clean[key] = str(value)
        elif isinstance(value, list):
            clean[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            clean[key] = value
    return clean


def compute_scale(path: Path, target_key: str) -> float:
    data = np.load(path)
    target = data[target_key].astype(np.float32)
    scale = float(np.sqrt(np.mean(target**2)))
    return max(scale, 1e-8)


def nmse_linear(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    ref = torch.sum(target**2, dim=(1, 2, 3)).clamp_min(1e-12)
    return err / ref


def nmse_db_from_linear(value: float) -> float:
    return float(10.0 * np.log10(max(value, np.finfo(np.float64).tiny)))


def ri_to_complex_links(value: torch.Tensor, n_rx: int, n_tx: int) -> torch.Tensor:
    n_links = n_rx * n_tx
    real = value[:, :n_links]
    imag = value[:, n_links : 2 * n_links]
    return torch.complex(real, imag)


def pilot_consistency_loss(
    pred: torch.Tensor,
    h_pilot: torch.Tensor,
    pilot_positions: torch.Tensor,
    n_rx: int,
    n_tx: int,
) -> torch.Tensor:
    pred_links = ri_to_complex_links(pred, n_rx, n_tx)
    pred_pilot = pred_links[:, :, pilot_positions[:, 0], pilot_positions[:, 1]]
    pred_pilot = pred_pilot.permute(0, 2, 1).reshape(h_pilot.shape)
    return torch.mean(torch.abs(pred_pilot - h_pilot) ** 2)


def pca_basis_from_training_file(
    path: Path,
    scale: float,
    num_basis: int,
    max_observations: int,
    seed: int,
) -> torch.Tensor:
    data = np.load(path)
    h_grid = data["h_true_grid"].astype(np.complex64)
    n_samples, n_symbols, n_subcarriers, n_rx, n_tx = h_grid.shape
    observations = np.transpose(h_grid, (0, 3, 4, 1, 2)).reshape(-1, n_symbols * n_subcarriers)
    observations = observations / np.float32(scale)
    if observations.shape[0] > max_observations:
        rng = np.random.default_rng(seed)
        indices = rng.choice(observations.shape[0], size=max_observations, replace=False)
        observations = observations[indices]
    if observations.shape[0] < num_basis:
        raise ValueError("pca-max-observations must be at least num_basis.")

    print(f"initializing basis with PCA: observations={observations.shape[0]}, grid={n_symbols}x{n_subcarriers}")
    _, _, vh = np.linalg.svd(observations, full_matrices=False)
    basis = vh[:num_basis].reshape(num_basis, n_symbols, n_subcarriers).astype(np.complex64)
    return torch.from_numpy(basis)


@torch.no_grad()
def evaluate(
    model: PilotFittedMIMOSharedBasisNet,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nmse = 0.0
    total_count = 0
    for h_pilot, target in loader:
        h_pilot = h_pilot.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(h_pilot)
        batch = target.shape[0]
        loss = criterion(pred, target)
        total_loss += float(loss.item()) * batch
        total_nmse += float(torch.sum(nmse_linear(pred, target)).item())
        total_count += batch
    mean_loss = total_loss / max(total_count, 1)
    mean_nmse = total_nmse / max(total_count, 1)
    return {"loss": mean_loss, "nmse": mean_nmse, "nmse_db": nmse_db_from_linear(mean_nmse)}


def write_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
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
        raise ValueError("train and val pilot_positions differ; this first PF-MSBNet version expects fixed pilots.")

    n_rx = int(train_set.h_pilot.shape[2])
    n_tx = int(train_set.h_pilot.shape[3])
    n_symbols = int(train_set.target.shape[2])
    n_subcarriers = int(train_set.target.shape[3])
    pilot_positions = torch.from_numpy(train_set.pilot_positions)

    model = PilotFittedMIMOSharedBasisNet(
        pilot_positions=pilot_positions,
        n_symbols=n_symbols,
        n_subcarriers=n_subcarriers,
        n_rx=n_rx,
        n_tx=n_tx,
        num_basis=args.num_basis,
        hidden_channels=args.hidden_channels,
        pos_bands=args.pos_bands,
        min_regularization=args.min_regularization,
    ).to(device)
    if args.basis_init == "pca":
        basis = pca_basis_from_training_file(
            args.train,
            scale=scale,
            num_basis=args.num_basis,
            max_observations=args.pca_max_observations,
            seed=args.pca_seed,
        )
        model.set_complex_basis(basis.to(device))

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

    run_name = "pf_msbnet_grid"
    best_path = args.outdir / f"{run_name}_best.pt"
    last_path = args.outdir / f"{run_name}_last.pt"
    metrics_path = args.outdir / f"{run_name}_metrics.csv"
    summary_path = args.outdir / f"{run_name}_summary.json"

    print(f"device={device}")
    print(f"model=pf_msbnet, params={count_parameters(model)}")
    print(f"train={args.train}, val={args.val}, scale={scale:.6g}")
    print(
        f"basis={args.num_basis}, hidden={args.hidden_channels}, basis_init={args.basis_init}, "
        f"lambda_pilot={args.lambda_pilot}, lambda_orth={args.lambda_orth}, lambda_gate={args.lambda_gate}"
    )

    rows: list[dict[str, float | int | str]] = []
    best_nmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        train_gate = 0.0
        for h_pilot, target in train_loader:
            h_pilot = h_pilot.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(h_pilot)
            mse_loss = criterion(pred, target)
            loss = mse_loss
            if args.lambda_pilot > 0:
                loss = loss + args.lambda_pilot * pilot_consistency_loss(
                    pred,
                    h_pilot,
                    model.pilot_positions,
                    n_rx=n_rx,
                    n_tx=n_tx,
                )
            if args.lambda_orth > 0:
                loss = loss + args.lambda_orth * model.basis_orthogonality_loss()
            if args.lambda_gate > 0:
                _, gate, _ = model.encode_pilots(h_pilot)
                gate_loss = torch.mean(gate)
                loss = loss + args.lambda_gate * gate_loss
                train_gate += float(gate_loss.item()) * target.shape[0]

            loss.backward()
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
            "train_gate_mean": train_gate / max(train_count, 1) if args.lambda_gate > 0 else 0.0,
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
            torch.save(
                {
                    "model_type": "pf_msbnet",
                    "model": model.state_dict(),
                    "args": serializable_args(args),
                    "scale": scale,
                    "n_rx": n_rx,
                    "n_tx": n_tx,
                    "n_symbols": n_symbols,
                    "n_subcarriers": n_subcarriers,
                    "pilot_positions": pilot_positions,
                    "num_basis": args.num_basis,
                    "hidden_channels": args.hidden_channels,
                    "pos_bands": args.pos_bands,
                    "min_regularization": args.min_regularization,
                    "best_val_nmse": best_nmse,
                    "param_count": count_parameters(model),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"early stop at epoch {epoch}: best_epoch={best_epoch}, "
                f"best_val_nmse={nmse_db_from_linear(best_nmse):.3f} dB"
            )
            break

    torch.save({"model_type": "pf_msbnet", "model": model.state_dict(), "args": serializable_args(args)}, last_path)
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
        "model": "pf_msbnet",
        "param_count": count_parameters(model),
        "scale": scale,
        "best_val_nmse": best_nmse,
        "best_val_nmse_db": nmse_db_from_linear(best_nmse),
        "best_epoch": best_epoch,
        "basis_init": args.basis_init,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "metrics_csv": str(metrics_path),
        "test_results": test_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
