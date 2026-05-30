from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from adapter_mimo_ofdm.models import PilotFittedMIMOSharedBasisNet  # noqa: E402
from compare_grid_methods import GridEvalDataset, PilotFittedEvalDataset, load_model_from_checkpoint, parse_model_spec  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def benchmark_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, repeats: int, warmup: int) -> float:
    model.eval()
    total_frames = 0
    total_seconds = 0.0
    for repeat in range(max(1, repeats)):
        for batch_index, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if repeat > 0 or batch_index >= warmup:
                total_seconds += elapsed
                total_frames += int(x.shape[0])
    return 1000.0 * total_seconds / max(total_frames, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark neural channel estimators for the paper complexity table.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH, repeated.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "p8" / "complexity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for spec in args.checkpoint:
        display_name, checkpoint_path = parse_model_spec(spec)
        if not checkpoint_path.exists():
            print(f"skip missing checkpoint: {checkpoint_path}")
            continue
        model_name, model, scale, params = load_model_from_checkpoint(checkpoint_path, device)
        if model_name == "pf_msbnet" and isinstance(model, PilotFittedMIMOSharedBasisNet):
            dataset = PilotFittedEvalDataset(args.test, scale)
            checkpoint_positions = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["pilot_positions"].cpu().numpy()
            if not np.array_equal(checkpoint_positions, dataset.pilot_positions):
                print(f"skip {display_name}: pilot positions differ")
                continue
        else:
            dataset = GridEvalDataset(args.test, "h_ls_grid_ri", "h_true_grid_ri", scale)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        ms = benchmark_loader(model, loader, device, args.repeats, args.warmup_batches)
        row = {"method": display_name, "params": params, "ms_per_frame": ms, "checkpoint": str(checkpoint_path)}
        rows.append(row)
        print(f"{display_name}: params={params}, {ms:.5f} ms/frame")
    if rows:
        write_csv(args.outdir / "neural_complexity.csv", rows)
        print(f"saved csv: {args.outdir / 'neural_complexity.csv'}")


if __name__ == "__main__":
    main()
