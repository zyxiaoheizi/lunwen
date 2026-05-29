#!/usr/bin/env bash
set -euo pipefail

# Build/evaluate a fixed shared-basis ridge baseline.
# This reproduces the classical BEM-style idea: fixed basis + per-frame pilot
# coefficient fitting, without trainable attention/gates.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
NUM_BASIS="${NUM_BASIS:-32}"
BASIS_INIT="${BASIS_INIT:-pca}"
REGULARIZATION="${REGULARIZATION:-1e-3}"
PCA_MAX_OBSERVATIONS="${PCA_MAX_OBSERVATIONS:-8192}"

TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/fixed_basis_ridge}"

COMMON_TESTS=("${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}")

echo "Building fixed shared-basis ridge baseline for ${PILOT_TAG}"
echo "basis=${BASIS_INIT}, num_basis=${NUM_BASIS}, lambda=${REGULARIZATION}, device=${DEVICE}"
echo "outdir=${OUTDIR}"

"${PYTHON_BIN}" scripts/build_fixed_basis_baseline.py \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --num-basis "${NUM_BASIS}" \
  --basis-init "${BASIS_INIT}" \
  --regularization "${REGULARIZATION}" \
  --pca-max-observations "${PCA_MAX_OBSERVATIONS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --outdir "${OUTDIR}"
