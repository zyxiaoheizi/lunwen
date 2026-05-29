from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from adapter_mimo_ofdm.models import PilotFittedMIMOSharedBasisNet  # noqa: E402
from compare_grid_methods import PilotFittedEvalDataset, load_model_from_checkpoint, nmse_db  # noqa: E402


def parse_topk(value: str) -> list[int]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    topks: list[int] = []
    for item in items:
        if item in {"full", "all", "0"}:
            topks.append(0)
        else:
            topks.append(int(item))
    return topks


def dataset_label(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^grid_p\d+_", "", name)
    name = re.sub(r"_test_\d+$", "", name)
    return name


@torch.no_grad()
def evaluate_nmse_and_time(
    model: PilotFittedMIMOSharedBasisNet,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
    timing_repeats: int,
) -> tuple[float, float, float]:
    model.eval()
    total = 0.0
    count = 0
    timed_frames = 0
    timed_seconds = 0.0

    for repeat in range(max(timing_repeats, 1)):
        for batch_index, (h_pilot, target) in enumerate(loader):
            h_pilot = h_pilot.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred = model(h_pilot)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if repeat == 0:
                err = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
                ref = torch.sum(target**2, dim=(1, 2, 3)).clamp_min(1e-12)
                total += float(torch.sum(err / ref).item())
                count += int(target.shape[0])

            if repeat > 0 or batch_index >= warmup_batches:
                timed_seconds += elapsed
                timed_frames += int(target.shape[0])

    nmse = total / max(count, 1)
    ms_per_frame = 1000.0 * timed_seconds / max(timed_frames, 1)
    return nmse, nmse_db(nmse), ms_per_frame


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_nmse(path: Path, rows: list[dict[str, object]]) -> None:
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    topks = list(dict.fromkeys(str(row["active_basis"]) for row in rows))
    values = {(str(row["dataset"]), str(row["active_basis"])): float(row["nmse_db"]) for row in rows}

    x = np.arange(len(topks))
    plt.figure(figsize=(8.5, 5.0))
    for dataset in datasets:
        y = [values.get((dataset, topk), np.nan) for topk in topks]
        plt.plot(x, y, marker="o", linewidth=1.8, label=dataset)
    plt.xticks(x, topks)
    plt.xlabel("Active basis count")
    plt.ylabel("NMSE (dB)")
    plt.title("Active-Basis Sparse Inference: NMSE")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_latency(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["active_basis"]), []).append(float(row["ms_per_frame"]))
    topks = list(grouped.keys())
    values = [float(np.mean(grouped[topk])) for topk in topks]

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(topks, values, marker="o", linewidth=2.0)
    plt.xlabel("Active basis count")
    plt.ylabel("Mean inference time (ms/frame)")
    plt.title("Active-Basis Sparse Inference: Latency")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PF-MSBNet active-basis top-K inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--topk", default="4,8,12,16,24,32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timing-repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-key", default="h_true_grid_ri")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "active_basis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_name, model, scale, params = load_model_from_checkpoint(args.checkpoint, device)
    if model_name != "pf_msbnet" or not isinstance(model, PilotFittedMIMOSharedBasisNet):
        raise ValueError("active-basis evaluation requires a PF-MSBNet checkpoint.")

    topks = parse_topk(args.topk)
    rows: list[dict[str, object]] = []
    for topk in topks:
        model.active_basis_topk = int(topk)
        active_label = "full" if topk <= 0 or topk >= model.num_basis else str(topk)
        for test_path in args.test:
            dataset = PilotFittedEvalDataset(test_path, scale, args.target_key)
            checkpoint_positions = torch.load(args.checkpoint, map_location="cpu", weights_only=True)[
                "pilot_positions"
            ].cpu().numpy()
            if not np.array_equal(checkpoint_positions, dataset.pilot_positions):
                print(f"skip {test_path}: pilot_positions differ")
                continue
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            value, value_db, ms_per_frame = evaluate_nmse_and_time(
                model,
                loader,
                device,
                warmup_batches=args.warmup_batches,
                timing_repeats=args.timing_repeats,
            )
            row = {
                "dataset": dataset_label(test_path),
                "active_basis": active_label,
                "active_basis_numeric": model.num_basis if active_label == "full" else int(active_label),
                "nmse": value,
                "nmse_db": value_db,
                "ms_per_frame": ms_per_frame,
                "params": params,
                "checkpoint": str(args.checkpoint),
            }
            rows.append(row)
            print(
                f"{row['dataset']} | active_basis={active_label}: "
                f"nmse={value_db:.3f} dB, {ms_per_frame:.4f} ms/frame"
            )

    csv_path = args.outdir / "active_basis_results.csv"
    json_path = args.outdir / "active_basis_results.json"
    nmse_path = args.outdir / "active_basis_nmse.png"
    latency_path = args.outdir / "active_basis_latency.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_nmse(nmse_path, rows)
    plot_latency(latency_path, rows)
    print(f"saved csv: {csv_path}")
    print(f"saved json: {json_path}")
    print(f"saved figure: {nmse_path}")
    print(f"saved figure: {latency_path}")


if __name__ == "__main__":
    main()
