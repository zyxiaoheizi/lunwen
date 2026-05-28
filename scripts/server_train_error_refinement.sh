#!/usr/bin/env bash
set -euo pipefail

# Train PL-ERN: a pilot-locked error refinement model on top of an existing
# ReEsNet/ChannelNet checkpoint. Run this after server_train_open_cnn_baselines.sh.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
BASE_NAME="${BASE_NAME:-reesnet}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
DEPTH="${DEPTH:-4}"
LAMBDA_PILOT="${LAMBDA_PILOT:-0}"
LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-1e-4}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0}"

TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"

case "${BASE_NAME}" in
  reesnet)
    BASE_CHECKPOINT="${BASE_CHECKPOINT:-${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt}"
    OUTDIR="${OUTDIR:-${OUT_ROOT}/pl_ern_reesnet}"
    ;;
  channelnet)
    BASE_CHECKPOINT="${BASE_CHECKPOINT:-${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt}"
    OUTDIR="${OUTDIR:-${OUT_ROOT}/pl_ern_channelnet}"
    ;;
  *)
    echo "Unsupported BASE_NAME=${BASE_NAME}. Use BASE_NAME=reesnet or BASE_NAME=channelnet." >&2
    exit 2
    ;;
esac

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Missing base checkpoint: ${BASE_CHECKPOINT}" >&2
  echo "Run scripts/server_train_open_cnn_baselines.sh first, or set BASE_CHECKPOINT=..." >&2
  exit 2
fi

COMMON_TESTS=("${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}")

echo "Training PL-ERN for ${PILOT_TAG}"
echo "base=${BASE_NAME}, base_checkpoint=${BASE_CHECKPOINT}"
echo "epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LR}, device=${DEVICE}"
echo "depth/hidden=${DEPTH}/${HIDDEN_CHANNELS}, lambda_pilot=${LAMBDA_PILOT}, lambda_residual=${LAMBDA_RESIDUAL}"
echo "outdir=${OUTDIR}"

"${PYTHON_BIN}" scripts/train_error_refinement.py \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --depth "${DEPTH}" \
  --lambda-pilot "${LAMBDA_PILOT}" \
  --lambda-residual "${LAMBDA_RESIDUAL}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --outdir "${OUTDIR}"
