#!/usr/bin/env bash
set -euo pipefail

# Train an A-MMSE-like learned linear-filter baseline.
# The model directly learns a complex filter W and estimates h_hat = W h_p.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
RANK="${RANK:-0}"
INIT_LMMSE="${INIT_LMMSE:-1}"
LMMSE_REGULARIZATION="${LMMSE_REGULARIZATION:-1e-4}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/ammse_filter}"

COMMON_TESTS=("${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}")

echo "Training A-MMSE-like linear filter for ${PILOT_TAG}"
echo "epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LR}, weight_decay=${WEIGHT_DECAY}, rank=${RANK}, init_lmmse=${INIT_LMMSE}, device=${DEVICE}"
echo "outdir=${OUTDIR}"

EXTRA_ARGS=()
if [[ "${INIT_LMMSE}" == "1" ]]; then
  EXTRA_ARGS+=(--init-lmmse --lmmse-regularization "${LMMSE_REGULARIZATION}")
fi

"${PYTHON_BIN}" scripts/train_ammse_filter.py \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --rank "${RANK}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --outdir "${OUTDIR}" \
  "${EXTRA_ARGS[@]}"
