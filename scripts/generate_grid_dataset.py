from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapter_mimo_ofdm.sim import generate_grid_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ChannelNet-style 2D time-frequency MIMO-OFDM data."
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "grid_tdl_a_72x14_p48_5k.npz")
    parser.add_argument("--n-symbols", type=int, default=14)
    parser.add_argument("--n-subcarriers", type=int, default=72)
    parser.add_argument("--n-tx", type=int, default=2)
    parser.add_argument("--n-rx", type=int, default=2)
    parser.add_argument("--num-pilots", type=int, default=48)
    parser.add_argument("--taps", type=int, default=4)
    parser.add_argument(
        "--profile",
        choices=["exponential", "epa", "eva", "etu", "tdl-a", "tdl-b", "tdl-c"],
        default="tdl-a",
    )
    parser.add_argument("--subcarrier-spacing", type=float, default=15_000.0)
    parser.add_argument("--delay-spread-ns", type=float, default=300.0)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument("--channel", choices=["rayleigh", "rician"], default="rayleigh")
    parser.add_argument("--rician-k", type=float, default=5.0)
    parser.add_argument("--max-doppler-hz", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = generate_grid_dataset(
        out_path=args.out,
        samples=args.samples,
        n_tx=args.n_tx,
        n_rx=args.n_rx,
        n_symbols=args.n_symbols,
        n_subcarriers=args.n_subcarriers,
        num_pilots=args.num_pilots,
        n_taps=args.taps,
        channel_profile=args.profile,
        subcarrier_spacing_hz=args.subcarrier_spacing,
        delay_spread_ns=args.delay_spread_ns,
        snr_min_db=args.snr_min,
        snr_max_db=args.snr_max,
        channel_model=args.channel,
        rician_k=args.rician_k,
        max_doppler_hz=args.max_doppler_hz,
        seed=args.seed,
    )
    print(f"Saved 2D grid dataset: {out}")


if __name__ == "__main__":
    main()
