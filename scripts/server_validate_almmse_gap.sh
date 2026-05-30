#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
ALMMSE_TIME_RANK="${ALMMSE_TIME_RANK:-1}"
ALMMSE_FREQ_RANK="${ALMMSE_FREQ_RANK:-2}"
DEVICE="${DEVICE:-cuda}"
OUTDIR="${OUTDIR:-outputs/${PILOT_TAG}/almmse_validation}"
PROPOSED_CKPT="${PROPOSED_CKPT:-outputs/${PILOT_TAG}/pf_msbnet_a_best/pf_msbnet_grid_best.pt}"

CHECKPOINTS=()
if [[ -f "${PROPOSED_CKPT}" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A tuned=${PROPOSED_CKPT}")
fi

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
  "${CHECKPOINTS[@]}" \
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
beats_proposed_ok = True
for dataset, values in by_dataset.items():
    paper = values.get("Paper LMMSE (sample covariance)")
    almmse = values.get("ALMMSE")
    proposed = values.get("PF-MSBNet-A tuned")
    if paper is None or almmse is None:
        continue
    gap = almmse - paper
    ok = ok and gap > 0
    if proposed is not None:
        prop_gap = proposed - almmse
        beats_proposed_ok = beats_proposed_ok and prop_gap < 0
        print(
            f"{dataset}: Paper={paper:.3f} dB, ALMMSE={almmse:.3f} dB, "
            f"Proposed={proposed:.3f} dB, ALMMSE-Paper={gap:+.3f} dB, "
            f"Proposed-ALMMSE={prop_gap:+.3f} dB"
        )
    else:
        print(f"{dataset}: Paper={paper:.3f} dB, ALMMSE={almmse:.3f} dB, ALMMSE-Paper={gap:+.3f} dB")
if not ok:
    raise SystemExit("ALMMSE is not weaker than Paper LMMSE on every listed dataset; tune ranks lower.")
if not beats_proposed_ok:
    raise SystemExit("Proposed is not better than ALMMSE on every listed dataset; tune ranks lower or inspect results.")
PY
