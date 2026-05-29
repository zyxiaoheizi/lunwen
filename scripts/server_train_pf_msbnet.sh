#!/usr/bin/env bash
set -euo pipefail

# Train PF-MSBNet: Pilot-Fitted MIMO Shared-Basis Network.
# It reuses the existing grid .npz files but reads raw h_pilot_ls instead of
# the interpolated h_ls_grid_ri image used by CNN baselines.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
NUM_BASIS="${NUM_BASIS:-32}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-128}"
POS_BANDS="${POS_BANDS:-6}"
BASIS_INIT="${BASIS_INIT:-pca}"
PCA_MAX_OBSERVATIONS="${PCA_MAX_OBSERVATIONS:-8192}"
LAMBDA_PILOT="${LAMBDA_PILOT:-0}"
LAMBDA_ORTH="${LAMBDA_ORTH:-1e-4}"
LAMBDA_GATE="${LAMBDA_GATE:-1e-5}"
MIN_REGULARIZATION="${MIN_REGULARIZATION:-1e-4}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0}"

TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/pf_msbnet}"

COMMON_TESTS=("${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}")

echo "Training PF-MSBNet for ${PILOT_TAG}"
echo "epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LR}, device=${DEVICE}"
echo "basis=${NUM_BASIS}, hidden=${HIDDEN_CHANNELS}, pos_bands=${POS_BANDS}, basis_init=${BASIS_INIT}"
echo "lambda_pilot=${LAMBDA_PILOT}, lambda_orth=${LAMBDA_ORTH}, lambda_gate=${LAMBDA_GATE}, min_reg=${MIN_REGULARIZATION}"
echo "outdir=${OUTDIR}"

"${PYTHON_BIN}" scripts/train_pf_msbnet.py \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-basis "${NUM_BASIS}" \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --pos-bands "${POS_BANDS}" \
  --basis-init "${BASIS_INIT}" \
  --pca-max-observations "${PCA_MAX_OBSERVATIONS}" \
  --lambda-pilot "${LAMBDA_PILOT}" \
  --lambda-orth "${LAMBDA_ORTH}" \
  --lambda-gate "${LAMBDA_GATE}" \
  --min-regularization "${MIN_REGULARIZATION}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --outdir "${OUTDIR}"
