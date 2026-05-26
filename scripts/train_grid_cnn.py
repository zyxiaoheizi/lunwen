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

from adapter_mimo_ofdm.models import build_model, count_parameters  # noqa: E402


class GridChannelDataset(Dataset):
    """读取二维时频信道数据，默认使用 h_ls_grid_ri -> h_true_grid_ri。"""

    def __init__(
        self,
        path: str | Path,
        input_key: str = "h_ls_grid_ri",
        target_key: str = "h_true_grid_ri",
        scale: float = 1.0,
    ) -> None:
        data = np.load(path)
        self.x = data[input_key].astype(np.float32) / scale
        self.y = data[target_key].astype(np.float32) / scale
        self.path = Path(path)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline CNN baselines on 2D MIMO-OFDM grids.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument("--model", choices=["rescnn", "srcnn", "dncnn", "channelnet"], default="channelnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "cnn")
    parser.add_argument("--input-key", default="h_ls_grid_ri")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nmse = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        batch = x.shape[0]
        loss = criterion(pred, y)
        total_loss += float(loss.item()) * batch
        total_nmse += float(torch.sum(nmse_linear(pred, y)).item())
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


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    scale = 1.0 if args.no_normalize else compute_scale(args.train, args.target_key)

    train_set = GridChannelDataset(args.train, args.input_key, args.target_key, scale)
    val_set = GridChannelDataset(args.val, args.input_key, args.target_key, scale)
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

    in_channels = int(train_set.x.shape[1])
    model = build_model(args.model, in_channels, args.hidden_channels, args.depth).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_name = f"{args.model}_grid"
    best_path = args.outdir / f"{run_name}_best.pt"
    last_path = args.outdir / f"{run_name}_last.pt"
    metrics_path = args.outdir / f"{run_name}_metrics.csv"
    summary_path = args.outdir / f"{run_name}_summary.json"

    print(f"device={device}")
    print(f"model={args.model}, params={count_parameters(model)}")
    print(f"train={args.train}, val={args.val}, scale={scale:.6g}")

    rows: list[dict[str, float | int | str]] = []
    best_nmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            batch = x.shape[0]
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

        if val_metrics["nmse"] < best_nmse:
            best_nmse = val_metrics["nmse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": serializable_args(args),
                    "scale": scale,
                    "in_channels": in_channels,
                    "best_val_nmse": best_nmse,
                    "param_count": count_parameters(model),
                },
                best_path,
            )

    torch.save({"model": model.state_dict(), "args": serializable_args(args), "scale": scale}, last_path)
    write_metrics(metrics_path, rows)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    test_results: dict[str, dict[str, float]] = {}
    for test_path in args.test:
        test_set = GridChannelDataset(test_path, args.input_key, args.target_key, scale)
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
        "model": args.model,
        "param_count": count_parameters(model),
        "scale": scale,
        "best_val_nmse": best_nmse,
        "best_val_nmse_db": nmse_db_from_linear(best_nmse),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "metrics_csv": str(metrics_path),
        "test_results": test_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
