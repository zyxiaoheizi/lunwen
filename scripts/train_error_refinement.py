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

from adapter_mimo_ofdm.models import (  # noqa: E402
    PilotLockedErrorRefinementNet,
    build_model,
    count_parameters,
)


class GridRefinementDataset(Dataset):
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
        description="Train a pilot-locked error refinement network on top of a frozen channel estimator."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lambda-pilot", type=float, default=0.0)
    parser.add_argument("--lambda-residual", type=float, default=1e-4)
    parser.add_argument("--unfreeze-base", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "pl_ern")
    parser.add_argument("--input-key", default="h_ls_grid_ri")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def nmse_linear(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    ref = torch.sum(target**2, dim=(1, 2, 3)).clamp_min(1e-12)
    return err / ref


def nmse_db_from_linear(value: float) -> float:
    return float(10.0 * np.log10(max(value, np.finfo(np.float64).tiny)))


def load_pilot_mask(path: Path) -> torch.Tensor:
    data = np.load(path)
    mask = data["pilot_mask"].astype(np.float32)
    return torch.from_numpy(mask).view(1, 1, mask.shape[0], mask.shape[1])


def load_base_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, object], float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    args = checkpoint.get("args", {})
    model_name = str(args.get("model", path.stem.split("_")[0]))
    in_channels = int(checkpoint.get("in_channels", 8))
    hidden_channels = int(args.get("hidden_channels", 64))
    depth = int(args.get("depth", 6))
    model = build_model(model_name, in_channels, hidden_channels, depth).to(device)
    model.load_state_dict(checkpoint["model"])
    scale = float(checkpoint.get("scale", 1.0))
    meta = {
        "model": model_name,
        "hidden_channels": hidden_channels,
        "depth": depth,
        "checkpoint": str(path),
    }
    return model, meta, scale, in_channels


def pilot_lock_loss(pred: torch.Tensor, ls_grid: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
    mask = pilot_mask.to(device=pred.device, dtype=pred.dtype)
    denom = (mask.sum() * pred.shape[0] * pred.shape[1]).clamp_min(1.0)
    return torch.sum(((pred - ls_grid) ** 2) * mask) / denom


def forward_parts(
    model: PilotLockedErrorRefinementNet,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.freeze_base:
        with torch.no_grad():
            base = model.base_model(x)
    else:
        base = model.base_model(x)
    residual = model.refiner(x, base.detach() if model.freeze_base else base, model.pilot_mask)
    return base, residual, base + residual


@torch.no_grad()
def evaluate(
    model: PilotLockedErrorRefinementNet,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nmse = 0.0
    total_base_nmse = 0.0
    total_pilot_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        base, _, pred = forward_parts(model, x)
        batch = x.shape[0]
        total_loss += float(criterion(pred, y).item()) * batch
        total_nmse += float(torch.sum(nmse_linear(pred, y)).item())
        total_base_nmse += float(torch.sum(nmse_linear(base, y)).item())
        total_pilot_loss += float(pilot_lock_loss(pred, x, model.pilot_mask).item()) * batch
        total_count += batch
    mean_nmse = total_nmse / max(total_count, 1)
    mean_base_nmse = total_base_nmse / max(total_count, 1)
    return {
        "loss": total_loss / max(total_count, 1),
        "nmse": mean_nmse,
        "nmse_db": nmse_db_from_linear(mean_nmse),
        "base_nmse": mean_base_nmse,
        "base_nmse_db": nmse_db_from_linear(mean_base_nmse),
        "pilot_loss": total_pilot_loss / max(total_count, 1),
    }


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
    clean["model"] = "plern"
    return clean


def save_checkpoint(
    path: Path,
    model: PilotLockedErrorRefinementNet,
    args: argparse.Namespace,
    base_meta: dict[str, object],
    scale: float,
    in_channels: int,
    best_nmse: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "args": serializable_args(args),
            "model_type": "plern",
            "scale": scale,
            "in_channels": in_channels,
            "base_model": base_meta,
            "refiner_hidden_channels": args.hidden_channels,
            "refiner_depth": args.depth,
            "best_val_nmse": best_nmse,
            "param_count": total_parameters(model),
            "trainable_param_count": count_parameters(model),
            "pilot_mask": model.pilot_mask.detach().cpu(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    base_model, base_meta, scale, in_channels = load_base_model(args.base_checkpoint, device)
    pilot_mask = load_pilot_mask(args.train).to(device)
    model = PilotLockedErrorRefinementNet(
        base_model=base_model,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        pilot_mask=pilot_mask,
        freeze_base=not args.unfreeze_base,
    ).to(device)

    train_set = GridRefinementDataset(args.train, args.input_key, args.target_key, scale)
    val_set = GridRefinementDataset(args.val, args.input_key, args.target_key, scale)
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
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    run_name = "plern_grid"
    best_path = args.outdir / f"{run_name}_best.pt"
    last_path = args.outdir / f"{run_name}_last.pt"
    metrics_path = args.outdir / f"{run_name}_metrics.csv"
    summary_path = args.outdir / f"{run_name}_summary.json"

    print(f"device={device}")
    print(
        f"model=plern, total_params={total_parameters(model)}, "
        f"trainable_params={count_parameters(model)}"
    )
    print(f"base={base_meta}")
    print(f"train={args.train}, val={args.val}, scale={scale:.6g}")
    print(f"lambda_pilot={args.lambda_pilot}, lambda_residual={args.lambda_residual}")

    rows: list[dict[str, float | int | str]] = []
    initial_metrics = evaluate(model, val_loader, device, criterion)
    best_nmse = initial_metrics["nmse"]
    best_epoch = 0
    epochs_without_improvement = 0
    rows.append(
        {
            "epoch": 0,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": float("nan"),
            "train_fit_loss": float("nan"),
            "train_pilot_loss": float("nan"),
            "train_residual_loss": float("nan"),
            "val_loss": initial_metrics["loss"],
            "val_nmse": initial_metrics["nmse"],
            "val_nmse_db": initial_metrics["nmse_db"],
            "val_base_nmse_db": initial_metrics["base_nmse_db"],
            "val_pilot_loss": initial_metrics["pilot_loss"],
        }
    )
    save_checkpoint(best_path, model, args, base_meta, scale, in_channels, best_nmse)
    print(
        f"epoch 000 | val_nmse={initial_metrics['nmse_db']:.3f} dB | "
        f"base={initial_metrics['base_nmse_db']:.3f} dB"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if model.freeze_base:
            model.base_model.eval()
        train_loss = 0.0
        train_fit_loss = 0.0
        train_pilot_loss = 0.0
        train_residual_loss = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, residual, pred = forward_parts(model, x)
            fit_loss = criterion(pred, y)
            p_loss = pilot_lock_loss(pred, x, model.pilot_mask)
            r_loss = torch.mean(residual**2)
            loss = fit_loss + args.lambda_pilot * p_loss + args.lambda_residual * r_loss
            loss.backward()
            optimizer.step()

            batch = x.shape[0]
            train_loss += float(loss.item()) * batch
            train_fit_loss += float(fit_loss.item()) * batch
            train_pilot_loss += float(p_loss.item()) * batch
            train_residual_loss += float(r_loss.item()) * batch
            train_count += batch
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(train_count, 1),
            "train_fit_loss": train_fit_loss / max(train_count, 1),
            "train_pilot_loss": train_pilot_loss / max(train_count, 1),
            "train_residual_loss": train_residual_loss / max(train_count, 1),
            "val_loss": val_metrics["loss"],
            "val_nmse": val_metrics["nmse"],
            "val_nmse_db": val_metrics["nmse_db"],
            "val_base_nmse_db": val_metrics["base_nmse_db"],
            "val_pilot_loss": val_metrics["pilot_loss"],
        }
        rows.append(row)
        print(
            f"epoch {epoch:03d} | train_loss={row['train_loss']:.6e} | "
            f"val_nmse={row['val_nmse_db']:.3f} dB | "
            f"base={row['val_base_nmse_db']:.3f} dB"
        )

        improved = val_metrics["nmse"] < best_nmse - args.early_stopping_min_delta
        if improved:
            best_nmse = val_metrics["nmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args, base_meta, scale, in_channels, best_nmse)
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"early stop at epoch {epoch}: best_epoch={best_epoch}, "
                f"best_val_nmse={nmse_db_from_linear(best_nmse):.3f} dB"
            )
            break

    save_checkpoint(last_path, model, args, base_meta, scale, in_channels, best_nmse)
    write_metrics(metrics_path, rows)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    test_results: dict[str, dict[str, float]] = {}
    for test_path in args.test:
        test_set = GridRefinementDataset(test_path, args.input_key, args.target_key, scale)
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        result = evaluate(model, test_loader, device, criterion)
        test_results[str(test_path)] = result
        print(
            f"test {test_path}: nmse={result['nmse_db']:.3f} dB, "
            f"base={result['base_nmse_db']:.3f} dB, loss={result['loss']:.6e}"
        )

    summary = {
        "model": "plern",
        "base_model": base_meta,
        "param_count": total_parameters(model),
        "trainable_param_count": count_parameters(model),
        "scale": scale,
        "best_val_nmse": best_nmse,
        "best_val_nmse_db": nmse_db_from_linear(best_nmse),
        "best_epoch": best_epoch,
        "lambda_pilot": args.lambda_pilot,
        "lambda_residual": args.lambda_residual,
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
