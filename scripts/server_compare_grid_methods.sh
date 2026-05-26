#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"

TESTS=(
  "data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz"
  "data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz"
)

CHECKPOINTS=()
if [[ -f "outputs/${PILOT_TAG}/cnn_srcnn/srcnn_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "SRCNN=outputs/${PILOT_TAG}/cnn_srcnn/srcnn_grid_best.pt")
fi
if [[ -f "outputs/${PILOT_TAG}/cnn_channelnet/channelnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ChannelNet=outputs/${PILOT_TAG}/cnn_channelnet/channelnet_grid_best.pt")
fi
if [[ -f "outputs/${PILOT_TAG}/cnn_reesnet/reesnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ReEsNet=outputs/${PILOT_TAG}/cnn_reesnet/reesnet_grid_best.pt")
fi

LMMSE_ARGS=(--include-oracle-lmmse --include-mismatched-lmmse --lmmse-profile tdl-a)
if [[ -f "data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz" ]]; then
  LMMSE_ARGS+=(
    --include-empirical-lmmse
    --empirical-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz"
  )
fi

"${PYTHON_BIN}" scripts/compare_grid_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  "${LMMSE_ARGS[@]}" \
  --outdir "outputs/${PILOT_TAG}/comparison"
