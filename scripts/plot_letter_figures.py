from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "LS + 2D interp.": "LS",
    "Paper LMMSE (sample covariance)": "Ideal LMMSE",
    "Empirical 2D LMMSE (train covariance)": "Full LMMSE",
    "Oracle 2D LMMSE": "Oracle LMMSE",
    "ALMMSE": "ALMMSE",
    "ChannelNet": "ChannelNet",
    "ReEsNet": "ReEsNet",
    "PF-MSBNet": "PF-SBNet",
    "PF-MSBNet base": "PF-SBNet",
    "PF-MSBNet-A": "Proposed",
    "PF-MSBNet-A tuned": "Proposed",
    "PF-MSBNet-A-best": "Proposed",
    "PF-MSBNet-A2": "PF-MSBNet-A2",
    "w/o pilot attention": "w/o attention",
    "w/o basis gate": "w/o gate",
    "w/o learned W": "w/o weight",
    "w/o learned lambda": "w/o lambda",
}

METHOD_ORDER = [
    "LS",
    "Ideal LMMSE",
    "Full LMMSE",
    "ALMMSE",
    "Oracle LMMSE",
    "ChannelNet",
    "ReEsNet",
    "PF-SBNet",
    "Proposed",
]

COLORS = {
    "LS": "#7f7f7f",
    "Ideal LMMSE": "#1f77b4",
    "Full LMMSE": "#1f77b4",
    "ALMMSE": "#4c78a8",
    "Oracle LMMSE": "#1f77b4",
    "ChannelNet": "#ff7f0e",
    "ReEsNet": "#2ca02c",
    "PF-SBNet": "#9467bd",
    "Proposed": "#d62728",
    "w/o attention": "#8c564b",
    "w/o gate": "#e377c2",
    "w/o weight": "#17becf",
    "w/o lambda": "#bcbd22",
}

HOLLOW_MARKERS = {"ChannelNet", "ALMMSE", "PF-SBNet", "w/o gate", "w/o lambda"}


def marker_face(label: str) -> str:
    return "white" if label in HOLLOW_MARKERS else COLORS.get(label, "#444444")


def line_alpha(label: str) -> float:
    return 1.0 if label in {"Proposed", "Ideal LMMSE", "Full LMMSE", "ALMMSE"} else 0.88


def line_width(label: str) -> float:
    if label == "Proposed":
        return 0.95
    if label in {"Ideal LMMSE", "Full LMMSE", "ALMMSE"}:
        return 0.82
    return 0.78


def marker_size(label: str) -> float:
    return 2.25 if label == "Proposed" else 1.9


def draw_method_curve(ax, x, y, label: str, *, semilogy: bool = False) -> None:
    plot_fn = ax.semilogy if semilogy else ax.plot
    plot_fn(
        x,
        y,
        marker="o",
        markersize=marker_size(label),
        color=COLORS.get(label),
        linestyle="-",
        linewidth=line_width(label),
        alpha=line_alpha(label),
        markerfacecolor=marker_face(label),
        markeredgecolor=COLORS.get(label),
        markeredgewidth=0.38,
        label=label,
        zorder=5 if label == "Proposed" else 4 if label in {"Ideal LMMSE", "Full LMMSE", "ALMMSE"} else 3,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.9,
            "axes.titlesize": 6.9,
            "legend.fontsize": 5.9,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.50,
            "grid.linewidth": 0.20,
            "lines.linewidth": 0.78,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def compact_legend(ax, *, ncol: int = 2) -> None:
    legend = ax.legend(
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#555555",
        loc="upper right",
        borderpad=0.22,
        columnspacing=0.52,
        handlelength=1.05,
        handletextpad=0.32,
    )
    legend.get_frame().set_linewidth(0.32)


def scenario_label(dataset: str) -> str:
    name = Path(dataset).stem
    name = re.sub(r"^grid_p\d+_", "", name)
    name = re.sub(r"_snr.*$", "", name)
    name = re.sub(r"_test_\d+$", "", name)
    mapping = {
        "tdl_a": "TDL-A",
        "tdl_b": "TDL-B",
        "tdl_c": "TDL-C",
        "rician_tdl_a": "Rician",
    }
    return mapping.get(name, name.replace("_", "-").upper())


def pilot_count(dataset: str, fallback: int | None = None) -> int | None:
    match = re.search(r"grid_p(\d+)_", dataset)
    if match:
        return int(match.group(1))
    return fallback


def parse_snr(dataset: str) -> float | None:
    match = re.search(r"_snr(m?\d+(?:p\d+)?)_test_", dataset)
    if not match:
        return None
    return float(match.group(1).replace("m", "-").replace("p", "."))


def ordered_labels(labels: set[str]) -> list[str]:
    ordered = [label for label in METHOD_ORDER if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def normalize_rows(rows: list[dict[str, str]], keep: set[str] | None = None) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        label = method_label(row["method"])
        if keep is not None and label not in keep:
            continue
        key = (row["dataset"], label)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "dataset": row["dataset"],
                "scenario": scenario_label(row["dataset"]),
                "method": label,
                "nmse": float(row["nmse"]),
                "nmse_db": float(row["nmse_db"]),
                "params": int(float(row.get("params", 0) or 0)),
                "checkpoint": row.get("checkpoint", ""),
            }
        )
    return out


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.18)
    plt.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close()


def plot_snr_curve(csv_path: Path, outdir: Path, keep: set[str]) -> None:
    parsed = []
    for row in read_csv(csv_path):
        label = method_label(row["method"])
        if label not in keep:
            continue
        if "snr_db" in row and row["snr_db"] != "":
            snr = float(row["snr_db"])
        else:
            snr = parse_snr(str(row.get("dataset", "")))
            if snr is None:
                continue
        parsed.append(
            {
                "snr_db": snr,
                "method": label,
                "nmse": float(row["nmse"]),
                "nmse_db": float(row["nmse_db"]),
            }
        )
    if not parsed:
        return

    labels = ordered_labels({str(row["method"]) for row in parsed})
    plt.figure(figsize=(3.35, 2.12))
    ax = plt.gca()
    for label in labels:
        curve = sorted((row for row in parsed if row["method"] == label), key=lambda item: float(item["snr_db"]))
        x = [float(row["snr_db"]) for row in curve]
        y = [float(row["nmse_db"]) for row in curve]
        draw_method_curve(ax, x, y, label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.grid(True, linestyle="--", alpha=0.30)
    compact_legend(ax, ncol=2)
    write_csv(outdir / "letter_snr_nmse.csv", parsed)
    save_figure(outdir / "letter_snr_nmse")


def plot_ber_curve(csv_path: Path, outdir: Path, keep: set[str]) -> None:
    parsed = []
    for row in read_csv(csv_path):
        label = method_label(row["method"])
        if label not in keep:
            continue
        if "snr_db" in row and row["snr_db"] != "":
            snr = float(row["snr_db"])
        else:
            snr = parse_snr(str(row.get("dataset", "")))
            if snr is None:
                continue
        parsed.append(
            {
                "snr_db": snr,
                "method": label,
                "ber": max(float(row["ber"]), 1e-7),
            }
        )
    if not parsed:
        return

    labels = ordered_labels({str(row["method"]) for row in parsed})
    plt.figure(figsize=(3.35, 2.12))
    ax = plt.gca()
    for label in labels:
        curve = sorted((row for row in parsed if row["method"] == label), key=lambda item: float(item["snr_db"]))
        x = [float(row["snr_db"]) for row in curve]
        y = [float(row["ber"]) for row in curve]
        draw_method_curve(ax, x, y, label, semilogy=True)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which="both", linestyle="--", alpha=0.30)
    compact_legend(ax, ncol=2)
    write_csv(outdir / "letter_snr_ber.csv", parsed)
    save_figure(outdir / "letter_snr_ber")


def plot_cross_profile(csv_path: Path, outdir: Path, keep: set[str]) -> None:
    rows = normalize_rows(read_csv(csv_path), keep=keep)
    if not rows:
        return
    scenarios = ["TDL-A", "TDL-B", "TDL-C", "Rician"]
    labels = ordered_labels({str(row["method"]) for row in rows})
    values = {(str(row["scenario"]), str(row["method"])): float(row["nmse_db"]) for row in rows}

    x = np.arange(len(scenarios))
    plt.figure(figsize=(3.35, 2.05))
    ax = plt.gca()
    for label in labels:
        y = [values.get((scenario, label), np.nan) for scenario in scenarios]
        draw_method_curve(ax, x, y, label)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("NMSE (dB)")
    ax.grid(True, linestyle="--", alpha=0.30)
    compact_legend(ax, ncol=2)
    write_csv(outdir / "letter_cross_profile_nmse.csv", rows)
    table_rows = []
    for label in labels:
        table_rows.append([label] + [f"{values.get((scenario, label), np.nan):.2f}" for scenario in scenarios])
    write_latex_table(outdir / "letter_cross_profile_table.tex", ["Method", *scenarios], table_rows)
    save_figure(outdir / "letter_cross_profile_nmse")


def plot_pilot_overhead(csv_paths: list[Path], outdir: Path, keep: set[str]) -> None:
    summary: list[dict[str, object]] = []
    for path in csv_paths:
        rows = normalize_rows(read_csv(path), keep=keep)
        grouped: dict[str, list[float]] = defaultdict(list)
        pilot = None
        for row in rows:
            pilot = pilot_count(str(row["dataset"]), fallback=pilot)
            grouped[str(row["method"])].append(float(row["nmse_db"]))
        if pilot is None:
            continue
        for method, values in grouped.items():
            summary.append({"pilots": pilot, "method": method, "mean_nmse_db": float(np.mean(values))})
    if not summary:
        return
    labels = ordered_labels({str(row["method"]) for row in summary})
    plt.figure(figsize=(3.35, 2.05))
    ax = plt.gca()
    for label in labels:
        curve = sorted((row for row in summary if row["method"] == label), key=lambda item: int(item["pilots"]))
        x = [int(row["pilots"]) for row in curve]
        y = [float(row["mean_nmse_db"]) for row in curve]
        draw_method_curve(ax, x, y, label)
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Average NMSE (dB)")
    ax.set_xticks(sorted({int(row["pilots"]) for row in summary}))
    ax.grid(True, linestyle="--", alpha=0.30)
    compact_legend(ax, ncol=2)
    write_csv(outdir / "letter_pilot_overhead_nmse.csv", summary)
    save_figure(outdir / "letter_pilot_overhead_nmse")


def plot_ablation(csv_path: Path, outdir: Path) -> None:
    rows = normalize_rows(read_csv(csv_path), keep=None)
    keep = {"PF-SBNet", "Proposed", "w/o attention", "w/o gate", "w/o weight", "w/o lambda"}
    rows = [row for row in rows if row["method"] in keep]
    if not rows:
        return
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["nmse_db"]))
    order = ["PF-SBNet", "w/o attention", "w/o gate", "w/o weight", "w/o lambda", "Proposed"]
    proposed_value = float(np.mean(grouped["Proposed"])) if "Proposed" in grouped else None
    summary = [
        {
            "method": method,
            "mean_nmse_db": float(np.mean(grouped[method])),
            "penalty_db": float(np.mean(grouped[method]) - proposed_value) if proposed_value is not None else np.nan,
        }
        for method in order
        if method in grouped
    ]

    plt.figure(figsize=(3.35, 2.12))
    ax = plt.gca()
    labels = [str(row["method"]) for row in summary]
    values = [float(row["penalty_db"]) for row in summary]
    x = np.arange(len(labels))
    ax.axhline(0.0, color="#222222", linewidth=0.55)
    for xi, yi, label in zip(x, values, labels):
        ax.vlines(xi, 0.0, yi, color=COLORS.get(label, "#999999"), linewidth=0.65, alpha=0.75)
        ax.scatter(
            [xi],
            [yi],
            s=20 if label == "Proposed" else 15,
            color=COLORS.get(label, "#999999"),
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("NMSE loss vs. full (dB)")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    write_csv(outdir / "letter_ablation_summary.csv", summary)
    write_latex_table(
        outdir / "letter_ablation_table.tex",
        ["Variant", "Avg. NMSE", "Loss"],
        [
            [str(row["method"]), f"{float(row['mean_nmse_db']):.2f}", f"{float(row['penalty_db']):+.2f}"]
            for row in summary
        ],
    )
    save_figure(outdir / "letter_ablation_nmse")


def summarize_latency(csv_path: Path, outdir: Path) -> None:
    rows = read_csv(csv_path)
    if not rows:
        return
    summary: list[dict[str, object]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    grouped_nmse: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = row["active_basis"]
        grouped[key].append(float(row["ms_per_frame"]))
        grouped_nmse[key].append(float(row["nmse_db"]))
    for key in grouped:
        summary.append(
            {
                "active_basis": key,
                "mean_ms_per_frame": float(np.mean(grouped[key])),
                "mean_nmse_db": float(np.mean(grouped_nmse[key])),
            }
        )
    write_csv(outdir / "letter_latency_summary.csv", summary)


def write_active_basis_table(csv_path: Path, outdir: Path) -> None:
    rows = read_csv(csv_path)
    if not rows:
        return
    for stale in [
        outdir / "letter_active_basis_tradeoff.png",
        outdir / "letter_active_basis_tradeoff.pdf",
    ]:
        if stale.exists():
            stale.unlink()
    grouped_nmse: dict[str, list[float]] = defaultdict(list)
    grouped_latency: dict[str, list[float]] = defaultdict(list)
    grouped_numeric: dict[str, int] = {}
    for row in rows:
        key = row["active_basis"]
        grouped_nmse[key].append(float(row["nmse_db"]))
        grouped_latency[key].append(float(row["ms_per_frame"]))
        grouped_numeric[key] = int(float(row.get("active_basis_numeric", 0) or 0))

    summary = []
    for key in grouped_nmse:
        summary.append(
            {
                "active_basis": key,
                "active_basis_numeric": grouped_numeric.get(key, 0),
                "mean_nmse_db": float(np.mean(grouped_nmse[key])),
                "mean_ms_per_frame": float(np.mean(grouped_latency[key])),
            }
        )
    summary.sort(key=lambda row: int(row["active_basis_numeric"]))
    if not summary:
        return

    write_csv(outdir / "letter_active_basis_tradeoff.csv", summary)
    write_latex_table(
        outdir / "letter_active_basis_table.tex",
        ["Active basis", "Avg. NMSE", "Time"],
        [
            [
                str(row["active_basis"]),
                f"{float(row['mean_nmse_db']):.2f}",
                f"{float(row['mean_ms_per_frame']):.4f}",
            ]
            for row in summary
        ],
    )


def write_complexity_table(
    comparison_csv: Path,
    active_csv: Path,
    complexity_csv: Path,
    outdir: Path,
    keep: set[str],
) -> None:
    rows = normalize_rows(read_csv(comparison_csv), keep=keep)
    if not rows:
        return
    params_by_method: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        params_by_method[str(row["method"])].append(int(row["params"]))

    active_rows = read_csv(active_csv)
    active_full_times = [
        float(row["ms_per_frame"])
        for row in active_rows
        if str(row.get("active_basis", "")).lower() == "full"
    ]
    active_best_times = [float(row["ms_per_frame"]) for row in active_rows]
    proposed_time = float(np.mean(active_full_times or active_best_times)) if active_rows else np.nan
    benchmark_times: dict[str, float] = {}
    for row in read_csv(complexity_csv):
        benchmark_times[method_label(row["method"])] = float(row["ms_per_frame"])

    table_rows = []
    for method in ordered_labels(set(params_by_method)):
        params = int(max(params_by_method[method]))
        if params == 0:
            params_text = "0"
        elif params >= 1_000_000:
            params_text = f"{params / 1_000_000:.2f}M"
        else:
            params_text = f"{params / 1_000:.1f}k"
        time_value = benchmark_times.get(method)
        if time_value is None and method == "Proposed" and np.isfinite(proposed_time):
            time_value = proposed_time
        time_text = f"{time_value:.4f}" if time_value is not None and np.isfinite(time_value) else "-"
        table_rows.append([method, params_text, time_text])

    write_latex_table(outdir / "letter_complexity_table.tex", ["Method", "Params", "Time"], table_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact publication figures for the letter version.")
    parser.add_argument("--p8-comparison", type=Path, default=ROOT / "outputs/p8/comparison/grid_method_comparison.csv")
    parser.add_argument("--pilot-comparisons", type=Path, nargs="*", default=[])
    parser.add_argument("--snr-csv", type=Path, default=ROOT / "outputs/p8/snr_curve/snr_nmse_curve.csv")
    parser.add_argument("--ber-csv", type=Path, default=ROOT / "outputs/p8/ber_curve/ber_curve.csv")
    parser.add_argument("--ablation-csv", type=Path, default=ROOT / "outputs/p8/ablations/comparison/grid_method_comparison.csv")
    parser.add_argument("--active-csv", type=Path, default=ROOT / "outputs/p8/active_basis/active_basis_results.csv")
    parser.add_argument("--complexity-csv", type=Path, default=ROOT / "outputs/p8/complexity/neural_complexity.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/letter_figures")
    parser.add_argument(
        "--main-methods",
        nargs="*",
        default=["LS", "Ideal LMMSE", "ALMMSE", "ChannelNet", "ReEsNet", "Proposed"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    keep = set(args.main_methods)

    plot_snr_curve(args.snr_csv, args.outdir, keep)
    plot_ber_curve(args.ber_csv, args.outdir, keep)
    plot_cross_profile(args.p8_comparison, args.outdir, keep)
    pilot_paths = args.pilot_comparisons or [
        ROOT / "outputs/p8/comparison/grid_method_comparison.csv",
        ROOT / "outputs/p24/comparison/grid_method_comparison.csv",
        ROOT / "outputs/p48/comparison/grid_method_comparison.csv",
    ]
    plot_pilot_overhead(pilot_paths, args.outdir, keep)
    plot_ablation(args.ablation_csv, args.outdir)
    summarize_latency(args.active_csv, args.outdir)
    write_active_basis_table(args.active_csv, args.outdir)
    write_complexity_table(args.p8_comparison, args.active_csv, args.complexity_csv, args.outdir, keep)
    print(f"saved letter figures and tables under: {args.outdir}")


if __name__ == "__main__":
    main()
