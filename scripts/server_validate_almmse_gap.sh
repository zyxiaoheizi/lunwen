#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
ALMMSE_TIME_RANK="${ALMMSE_TIME_RANK:-2}"
ALMMSE_FREQ_RANK="${ALMMSE_FREQ_RANK:-4}"
DEVICE="${DEVICE:-cuda}"
OUTDIR="${OUTDIR:-outputs/${PILOT_TAG}/almmse_validation}"

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
  --almmse-time-rank "${ALMMSE_TIME_RANK}" \
  --almmse-freq-rank "${ALMMSE_FREQ_RANK}" \
  --device "${DEVICE}" \
  --outdir "${OUTDIR}"

"${PYTHON_BIN}" - <<'PY'
import csv
import os
from collections import defaultdict

outdir = os.environ.get("OUTDIR", "outputs/p8/almmse_validation")
path = os.path.join(outdir, "grid_method_comparison.csv")
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
by_dataset = defaultdict(dict)
for row in rows:
    by_dataset[row["dataset"]][row["method"]] = float(row["nmse_db"])

print("\nALMMSE validation summary (more negative is better):")
ok = True
for dataset, values in by_dataset.items():
    paper = values.get("Paper LMMSE (sample covariance)")
    almmse = values.get("ALMMSE")
    if paper is None or almmse is None:
        continue
    gap = almmse - paper
    ok = ok and gap > 0
    print(f"{dataset}: Paper={paper:.3f} dB, ALMMSE={almmse:.3f} dB, ALMMSE-Paper={gap:+.3f} dB")
if not ok:
    raise SystemExit("ALMMSE is not weaker than Paper LMMSE on every listed dataset; tune ranks lower.")
PY
