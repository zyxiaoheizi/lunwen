from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapter_mimo_ofdm.sim import SimConfig, results_to_rows, simulate_baselines  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LS/LMMSE MIMO-OFDM NMSE baselines.")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--n-subcarriers", type=int, default=64)
    parser.add_argument("--n-tx", type=int, default=2)
    parser.add_argument("--n-rx", type=int, default=2)
    parser.add_argument("--taps", type=int, default=4)
    parser.add_argument(
        "--profile",
        choices=["exponential", "epa", "eva", "etu", "tdl-a", "tdl-b", "tdl-c"],
        default="tdl-a",
    )
    parser.add_argument(
        "--lmmse-profile",
        choices=["none", "exponential", "epa", "eva", "etu", "tdl-a", "tdl-b", "tdl-c"],
        default="tdl-a",
        help="Assumed profile used by mismatched LMMSE. Use 'none' to disable it.",
    )
    parser.add_argument("--subcarrier-spacing", type=float, default=15_000.0)
    parser.add_argument("--delay-spread-ns", type=float, default=300.0)
    parser.add_argument("--lmmse-delay-spread-ns", type=float, default=None)
    parser.add_argument("--pilot-ratio", type=float, default=0.25)
    parser.add_argument("--channel", choices=["rayleigh", "rician"], default="rayleigh")
    parser.add_argument("--rician-k", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--snr", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "baselines")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_nmse(path: Path, rows: list[dict[str, float]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snr = [row["snr_db"] for row in rows]
    ls = [row["ls_linear_nmse_db"] for row in rows]
    lmmse = [row["lmmse_oracle_nmse_db"] for row in rows]
    mismatched = [row["lmmse_mismatched_nmse_db"] for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(snr, ls, "o-", label="LS + linear interp.")
    plt.plot(snr, lmmse, "s-", label="Oracle LMMSE")
    if any(value is not None for value in mismatched):
        plt.plot(snr, mismatched, "^-", label="Mismatched LMMSE")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    config = SimConfig(
        n_frames=args.frames,
        n_tx=args.n_tx,
        n_rx=args.n_rx,
        n_subcarriers=args.n_subcarriers,
        n_taps=args.taps,
        channel_profile=args.profile,
        lmmse_profile=None if args.lmmse_profile == "none" else args.lmmse_profile,
        subcarrier_spacing_hz=args.subcarrier_spacing,
        delay_spread_ns=args.delay_spread_ns,
        lmmse_delay_spread_ns=args.lmmse_delay_spread_ns,
        pilot_ratio=args.pilot_ratio,
        snr_db=tuple(args.snr),
        channel_model=args.channel,
        rician_k=args.rician_k,
        seed=args.seed,
    )
    results = simulate_baselines(config)
    rows = results_to_rows(results)

    lmmse_tag = f"_assume-{args.lmmse_profile}" if args.lmmse_profile != "none" else ""
    stem = f"{args.channel}_{args.profile}{lmmse_tag}_p{args.pilot_ratio:g}_n{args.frames}"
    csv_path = args.outdir / f"{stem}.csv"
    fig_path = args.outdir / f"{stem}.png"
    write_csv(csv_path, rows)
    plot_nmse(
        fig_path,
        rows,
        title=(
            f"2x2 MIMO-OFDM, {args.channel}, {args.profile}, "
            f"pilot ratio={args.pilot_ratio:g}"
        ),
    )

    print("SNR(dB) | LS+linear NMSE(dB) | Oracle LMMSE NMSE(dB) | Mismatched LMMSE NMSE(dB)")
    print("--------+--------------------+----------------------+---------------------------")
    for row in rows:
        mismatched_db = row["lmmse_mismatched_nmse_db"]
        mismatched_text = "disabled" if mismatched_db is None else f"{mismatched_db:25.3f}"
        print(
            f"{row['snr_db']:7.1f} | "
            f"{row['ls_linear_nmse_db']:18.3f} | "
            f"{row['lmmse_oracle_nmse_db']:20.3f} | "
            f"{mismatched_text}"
        )
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
