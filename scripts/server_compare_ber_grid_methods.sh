#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
SNR_LIST="${SNR_LIST:-0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30}"
PROFILE="${PROFILE:-tdl-a}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BER_BATCH_FRAMES="${BER_BATCH_FRAMES:-128}"
INCLUDE_PAPER_LMMSE="${INCLUDE_PAPER_LMMSE:-1}"
INCLUDE_ALMMSE="${INCLUDE_ALMMSE:-1}"
ALMMSE_TIME_RANK="${ALMMSE_TIME_RANK:-1}"
ALMMSE_FREQ_RANK="${ALMMSE_FREQ_RANK:-2}"
INCLUDE_CHANNELNET="${INCLUDE_CHANNELNET:-1}"
INCLUDE_REESNET="${INCLUDE_REESNET:-1}"
INCLUDE_PROPOSED="${INCLUDE_PROPOSED:-1}"

TESTS=()
for snr in ${SNR_LIST}; do
  tag="${snr}"
  tag="${tag/-/m}"
  tag="${tag/./p}"
  TESTS+=("data/grid/grid_${PILOT_TAG}_${PROFILE}_snr${tag}_test_${TEST_SAMPLES}.npz")
done

CHECKPOINTS=()
if [[ "${INCLUDE_CHANNELNET}" == "1" && -f "${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ChannelNet=${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt")
fi
if [[ "${INCLUDE_REESNET}" == "1" && -f "${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ReEsNet=${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt")
fi
if [[ "${INCLUDE_PROPOSED}" == "1" && -f "${OUT_ROOT}/pf_msbnet_a_best/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A tuned=${OUT_ROOT}/pf_msbnet_a_best/pf_msbnet_grid_best.pt")
elif [[ "${INCLUDE_PROPOSED}" == "1" && -f "${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A=${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt")
fi

LMMSE_ARGS=()
if [[ "${INCLUDE_PAPER_LMMSE}" == "1" && -f "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz" ]]; then
  LMMSE_ARGS+=(
    --include-paper-lmmse
    --paper-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
  )
fi
if [[ "${INCLUDE_ALMMSE}" == "1" && -f "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz" ]]; then
  LMMSE_ARGS+=(
    --include-almmse
    --almmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
    --almmse-time-rank "${ALMMSE_TIME_RANK}"
    --almmse-freq-rank "${ALMMSE_FREQ_RANK}"
  )
fi

OUTDIR="${OUTDIR:-${OUT_ROOT}/ber_curve}"
echo "Comparing BER curve for ${PILOT_TAG}: profile=${PROFILE}, outdir=${OUTDIR}"

"${PYTHON_BIN}" scripts/compare_ber_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  "${LMMSE_ARGS[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --ber-batch-frames "${BER_BATCH_FRAMES}" \
  --outdir "${OUTDIR}"
