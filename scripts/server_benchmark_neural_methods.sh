#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/complexity}"

CHECKPOINTS=()
if [[ -f "${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ChannelNet=${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt")
fi
if [[ -f "${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ReEsNet=${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt")
fi
if [[ -f "${OUT_ROOT}/pf_msbnet_a_best/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "Proposed=${OUT_ROOT}/pf_msbnet_a_best/pf_msbnet_grid_best.pt")
elif [[ -f "${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "Proposed=${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt")
fi

"${PYTHON_BIN}" scripts/benchmark_neural_methods.py \
  --test "data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz" \
  "${CHECKPOINTS[@]}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --outdir "${OUTDIR}"
