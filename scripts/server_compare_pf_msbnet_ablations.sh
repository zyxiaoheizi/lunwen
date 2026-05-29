#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}/ablations}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"

TESTS=(
  "data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_b_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_c_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_${TEST_SAMPLES}.npz"
)

CHECKPOINTS=()
for spec in \
  "Full-A=${OUT_ROOT}/full_a/pf_msbnet_grid_best.pt" \
  "w/o pilot attention=${OUT_ROOT}/no_pilot_attention/pf_msbnet_grid_best.pt" \
  "w/o basis gate=${OUT_ROOT}/no_basis_gate/pf_msbnet_grid_best.pt" \
  "w/o learned W=${OUT_ROOT}/no_learned_w/pf_msbnet_grid_best.pt" \
  "w/o learned lambda=${OUT_ROOT}/no_learned_lambda/pf_msbnet_grid_best.pt"; do
  path="${spec#*=}"
  if [[ -f "${path}" ]]; then
    CHECKPOINTS+=(--checkpoint "${spec}")
  fi
done

"${PYTHON_BIN}" scripts/compare_grid_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --outdir "${OUT_ROOT}/comparison"
