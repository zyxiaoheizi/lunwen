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

from adapter_mimo_ofdm.models import ChannelNetStyleCNN, count_parameters  # noqa: E402


def total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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
    parser = argparse.ArgumentParser(
        description="Train ChannelNet as a paper-style SRCNN -> DnCNN two-stage pipeline."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument("--srcnn-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "cnn_channelnet")
    parser.add_argument("--input-key", default="h_ls_grid_ri")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


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


def load_srcnn_into_channelnet(model: ChannelNetStyleCNN, checkpoint_path: Path, device: torch.device) -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    srcnn_state = checkpoint["model"]
    converted = {}
    for key, value in srcnn_state.items():
        if key.startswith("net."):
            converted[key.removeprefix("net.")] = value
        else:
            converted[key] = value
    model.srcnn.load_state_dict(converted)
    return float(checkpoint.get("scale", 1.0))


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
    clean["model"] = "channelnet"
    return clean


def save_checkpoint(
    path: Path,
    model: ChannelNetStyleCNN,
    args: argparse.Namespace,
    scale: float,
    in_channels: int,
    best_nmse: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "args": serializable_args(args),
            "scale": scale,
            "in_channels": in_channels,
            "best_val_nmse": best_nmse,
            "param_count": total_parameters(model),
        },
        path,
    )


def set_srcnn_trainable(model: ChannelNetStyleCNN, trainable: bool) -> None:
    for parameter in model.srcnn.parameters():
        parameter.requires_grad = trainable


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    srcnn_meta = torch.load(args.srcnn_checkpoint, map_location=device, weights_only=True)
    in_channels = int(srcnn_meta.get("in_channels", 8))
    model = ChannelNetStyleCNN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        denoise_depth=args.depth,
    ).to(device)
    scale = load_srcnn_into_channelnet(model, args.srcnn_checkpoint, device)

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

    criterion = nn.MSELoss()
    set_srcnn_trainable(model, False)
    optimizer = torch.optim.AdamW(model.dncnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_name = "channelnet_grid"
    best_path = args.outdir / f"{run_name}_best.pt"
    last_path = args.outdir / f"{run_name}_last.pt"
    metrics_path = args.outdir / f"{run_name}_metrics.csv"
    summary_path = args.outdir / f"{run_name}_summary.json"

    print(f"device={device}")
    print(f"model=channelnet_pipeline, total_params={total_parameters(model)}, trainable_params={count_parameters(model)}")
    print(f"srcnn_checkpoint={args.srcnn_checkpoint}, scale={scale:.6g}")
    print(f"train={args.train}, val={args.val}")

    rows: list[dict[str, float | int | str]] = []
    best_nmse = float("inf")
    best_epoch = 0
    best_stage = ""
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.srcnn.eval()
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
            "stage": "dncnn",
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_metrics["loss"],
            "val_nmse": val_metrics["nmse"],
            "val_nmse_db": val_metrics["nmse_db"],
        }
        rows.append(row)
        print(
            f"dncnn epoch {epoch:03d} | train_loss={row['train_loss']:.6e} | "
            f"val_nmse={row['val_nmse_db']:.3f} dB"
        )
        improved = val_metrics["nmse"] < best_nmse - args.early_stopping_min_delta
        if improved:
            best_nmse = val_metrics["nmse"]
            best_epoch = epoch
            best_stage = "dncnn"
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args, scale, in_channels, best_nmse)
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"dncnn early stop at epoch {epoch}: best_stage={best_stage}, "
                f"best_epoch={best_epoch}, best_val_nmse={nmse_db_from_linear(best_nmse):.3f} dB"
            )
            break

    if args.fine_tune_epochs > 0:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        set_srcnn_trainable(model, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.fine_tune_epochs, 1))
        epochs_without_improvement = 0
        for epoch in range(1, args.fine_tune_epochs + 1):
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
                "stage": "finetune",
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss / max(train_count, 1),
                "val_loss": val_metrics["loss"],
                "val_nmse": val_metrics["nmse"],
                "val_nmse_db": val_metrics["nmse_db"],
            }
            rows.append(row)
            print(
                f"finetune epoch {epoch:03d} | train_loss={row['train_loss']:.6e} | "
                f"val_nmse={row['val_nmse_db']:.3f} dB"
            )
            improved = val_metrics["nmse"] < best_nmse - args.early_stopping_min_delta
            if improved:
                best_nmse = val_metrics["nmse"]
                best_epoch = epoch
                best_stage = "finetune"
                epochs_without_improvement = 0
                save_checkpoint(best_path, model, args, scale, in_channels, best_nmse)
            else:
                epochs_without_improvement += 1

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"finetune early stop at epoch {epoch}: best_stage={best_stage}, "
                    f"best_epoch={best_epoch}, best_val_nmse={nmse_db_from_linear(best_nmse):.3f} dB"
                )
                break

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
        "model": "channelnet",
        "training_style": "srcnn_pretrain_then_dncnn",
        "param_count": total_parameters(model),
        "scale": scale,
        "srcnn_checkpoint": str(args.srcnn_checkpoint),
        "best_val_nmse": best_nmse,
        "best_val_nmse_db": nmse_db_from_linear(best_nmse),
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "metrics_csv": str(metrics_path),
        "test_results": test_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
