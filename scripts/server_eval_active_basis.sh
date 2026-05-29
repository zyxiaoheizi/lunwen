#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TOPK="${TOPK:-4,8,12,16,24,32}"
CHECKPOINT="${CHECKPOINT:-${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/active_basis}"

TESTS=(
  "data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_b_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_c_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_${TEST_SAMPLES}.npz"
)

"${PYTHON_BIN}" scripts/evaluate_active_basis.py \
  --checkpoint "${CHECKPOINT}" \
  --test "${TESTS[@]}" \
  --topk "${TOPK}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --outdir "${OUTDIR}"
