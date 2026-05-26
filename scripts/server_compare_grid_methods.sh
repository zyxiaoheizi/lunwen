#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LMMSE_PROFILE="${LMMSE_PROFILE:-tdl-a}"
LMMSE_DELAY_SPREAD_NS="${LMMSE_DELAY_SPREAD_NS:-300}"

TESTS=(
  "data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_b_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_c_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_${TEST_SAMPLES}.npz"
)

CHECKPOINTS=()
if [[ -f "${OUT_ROOT}/cnn_srcnn/srcnn_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "SRCNN=${OUT_ROOT}/cnn_srcnn/srcnn_grid_best.pt")
fi
if [[ -f "${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ChannelNet=${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt")
fi
if [[ -f "${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ReEsNet=${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt")
fi

LMMSE_ARGS=(
  --include-oracle-lmmse
  --include-mismatched-lmmse
  --lmmse-profile "${LMMSE_PROFILE}"
  --lmmse-delay-spread-ns "${LMMSE_DELAY_SPREAD_NS}"
)
if [[ -f "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz" ]]; then
  LMMSE_ARGS+=(
    --include-empirical-lmmse
    --empirical-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
  )
fi

echo "Comparing ${PILOT_TAG}: out_root=${OUT_ROOT}, train_samples=${TRAIN_SAMPLES}, test_samples=${TEST_SAMPLES}"

"${PYTHON_BIN}" scripts/compare_grid_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  "${LMMSE_ARGS[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --outdir "${OUT_ROOT}/comparison"
