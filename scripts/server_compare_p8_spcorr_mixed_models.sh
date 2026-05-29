#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}_spcorr_mixed}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"

TESTS=(
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho00_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho03_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho07_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho09_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_b_rho07_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_c_rho07_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_rician_tdl_a_rho07_test_${TEST_SAMPLES}.npz"
)

CHECKPOINTS=()
if [[ -f "${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A=${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt")
fi
if [[ -f "${OUT_ROOT}/pf_msbnet_a2/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A2=${OUT_ROOT}/pf_msbnet_a2/pf_msbnet_grid_best.pt")
fi

"${PYTHON_BIN}" scripts/compare_grid_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --outdir "${OUT_ROOT}/comparison"
