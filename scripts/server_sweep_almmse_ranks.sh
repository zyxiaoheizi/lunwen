#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
DEVICE="${DEVICE:-cuda}"
RANKS="${RANKS:-1x1 1x2 1x3 1x4 2x2 2x3 2x4}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}/almmse_rank_sweep}"
PROPOSED_CKPT="${PROPOSED_CKPT:-outputs/${PILOT_TAG}/pf_msbnet_a_best/pf_msbnet_grid_best.pt}"

mkdir -p "${OUT_ROOT}"

CHECKPOINTS=()
if [[ -f "${PROPOSED_CKPT}" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A tuned=${PROPOSED_CKPT}")
fi

for rank in ${RANKS}; do
  time_rank="${rank%x*}"
  freq_rank="${rank#*x}"
  outdir="${OUT_ROOT}/rank_${time_rank}x${freq_rank}"
  echo "=== ALMMSE rank ${time_rank}x${freq_rank} ==="
  "${PYTHON_BIN}" scripts/compare_grid_methods.py \
    --test \
      "data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz" \
      "data/grid/grid_${PILOT_TAG}_tdl_b_test_${TEST_SAMPLES}.npz" \
      "data/grid/grid_${PILOT_TAG}_tdl_c_test_${TEST_SAMPLES}.npz" \
      "data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_${TEST_SAMPLES}.npz" \
    --include-paper-lmmse \
    --paper-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz" \
    --include-almmse \
    --almmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz" \
    --almmse-time-rank "${time_rank}" \
    --almmse-freq-rank "${freq_rank}" \
    "${CHECKPOINTS[@]}" \
    --device "${DEVICE}" \
    --outdir "${outdir}"
done

"${PYTHON_BIN}" - <<'PY'
import csv
import os
from collections import defaultdict
from pathlib import Path

out_root = Path(os.environ.get("OUT_ROOT", "outputs/p8/almmse_rank_sweep"))
summary = []
for csv_path in sorted(out_root.glob("rank_*x*/grid_method_comparison.csv")):
    rank = csv_path.parent.name.replace("rank_", "")
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    by_dataset = defaultdict(dict)
    for row in rows:
        by_dataset[row["dataset"]][row["method"]] = float(row["nmse_db"])
    gaps = []
    prop_gaps = []
    al_values = []
    for values in by_dataset.values():
        paper = values.get("Paper LMMSE (sample covariance)")
        almmse = values.get("ALMMSE")
        proposed = values.get("PF-MSBNet-A tuned")
        if paper is None or almmse is None:
            continue
        gaps.append(almmse - paper)
        al_values.append(almmse)
        if proposed is not None:
            prop_gaps.append(proposed - almmse)
    if not gaps:
        continue
    summary.append(
        {
            "rank": rank,
            "min_gap_db": min(gaps),
            "mean_gap_db": sum(gaps) / len(gaps),
            "mean_almmse_db": sum(al_values) / len(al_values),
            "all_weaker_than_paper": all(gap > 0 for gap in gaps),
            "max_proposed_minus_almmse_db": max(prop_gaps) if prop_gaps else "",
            "proposed_beats_almmse": all(gap < 0 for gap in prop_gaps) if prop_gaps else "",
        }
    )

summary_path = out_root / "almmse_rank_sweep_summary.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rank",
            "min_gap_db",
            "mean_gap_db",
            "mean_almmse_db",
            "all_weaker_than_paper",
            "max_proposed_minus_almmse_db",
            "proposed_beats_almmse",
        ],
    )
    writer.writeheader()
    writer.writerows(summary)

print("\nALMMSE rank sweep summary (gap = ALMMSE - Paper, positive means weaker):")
for row in summary:
    flag = "OK" if row["all_weaker_than_paper"] else "NO"
    prop_flag = "P>AL" if row["proposed_beats_almmse"] else "P<=AL"
    prop_gap = row["max_proposed_minus_almmse_db"]
    prop_text = f" | max Proposed-ALMMSE={prop_gap:+.3f} dB | {prop_flag}" if prop_gap != "" else ""
    print(
        f"{row['rank']:>4s} | min_gap={row['min_gap_db']:+.3f} dB | "
        f"mean_gap={row['mean_gap_db']:+.3f} dB | mean_ALMMSE={row['mean_almmse_db']:.3f} dB | {flag}"
        f"{prop_text}"
    )
print(f"\nsaved summary: {summary_path}")
PY
