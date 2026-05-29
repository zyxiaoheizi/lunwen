from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_snr(dataset: str) -> float:
    match = re.search(r"_snr(m?\d+(?:p\d+)?)_test_", dataset)
    if not match:
        raise ValueError(f"Cannot parse SNR from dataset name: {dataset}")
    token = match.group(1).replace("m", "-").replace("p", ".")
    return float(token)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SNR-NMSE curves from compare_grid_methods CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "snr_curve")
    parser.add_argument("--methods", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv)
    parsed: list[dict[str, object]] = []
    for row in rows:
        method = row["method"]
        if args.methods and method not in args.methods:
            continue
        parsed.append(
            {
                "snr_db": parse_snr(row["dataset"]),
                "method": method,
                "nmse": float(row["nmse"]),
                "nmse_db": float(row["nmse_db"]),
            }
        )

    methods = list(dict.fromkeys(str(row["method"]) for row in parsed))
    plt.figure(figsize=(8.0, 5.0))
    for method in methods:
        method_rows = sorted((row for row in parsed if row["method"] == method), key=lambda item: float(item["snr_db"]))
        x = [float(row["snr_db"]) for row in method_rows]
        y = [float(row["nmse_db"]) for row in method_rows]
        plt.plot(x, y, marker="o", linewidth=1.8, label=method)

    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("SNR-NMSE Curve")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = args.outdir / "snr_nmse_curve.png"
    csv_path = args.outdir / "snr_nmse_curve.csv"
    plt.savefig(fig_path, dpi=180)
    plt.close()
    write_csv(csv_path, parsed)
    print(f"saved csv: {csv_path}")
    print(f"saved figure: {fig_path}")


if __name__ == "__main__":
    main()
