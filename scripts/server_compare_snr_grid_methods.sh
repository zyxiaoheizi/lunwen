#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
SNR_LIST="${SNR_LIST:-0 5 10 15 20 25 30}"
PROFILE="${PROFILE:-tdl-a}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LMMSE_PROFILE="${LMMSE_PROFILE:-tdl-a}"
LMMSE_DELAY_SPREAD_NS="${LMMSE_DELAY_SPREAD_NS:-300}"
INCLUDE_PAPER_LMMSE="${INCLUDE_PAPER_LMMSE:-1}"
INCLUDE_EMPIRICAL_LMMSE="${INCLUDE_EMPIRICAL_LMMSE:-1}"

TESTS=()
for snr in ${SNR_LIST}; do
  tag="${snr}"
  tag="${tag/-/m}"
  tag="${tag/./p}"
  TESTS+=("data/grid/grid_${PILOT_TAG}_${PROFILE}_snr${tag}_test_${TEST_SAMPLES}.npz")
done

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
if [[ -f "${OUT_ROOT}/pf_msbnet/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet=${OUT_ROOT}/pf_msbnet/pf_msbnet_grid_best.pt")
fi
if [[ -f "${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A=${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt")
fi
if [[ -f "${OUT_ROOT}/pf_msbnet_a2/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A2=${OUT_ROOT}/pf_msbnet_a2/pf_msbnet_grid_best.pt")
fi

LMMSE_ARGS=(
  --include-oracle-lmmse
  --include-mismatched-lmmse
  --lmmse-profile "${LMMSE_PROFILE}"
  --lmmse-delay-spread-ns "${LMMSE_DELAY_SPREAD_NS}"
)
if [[ -f "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz" ]]; then
  if [[ "${INCLUDE_PAPER_LMMSE}" == "1" ]]; then
    LMMSE_ARGS+=(
      --include-paper-lmmse
      --paper-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
    )
  fi
  if [[ "${INCLUDE_EMPIRICAL_LMMSE}" == "1" ]]; then
    LMMSE_ARGS+=(
      --include-empirical-lmmse
      --empirical-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
    )
  fi
fi

OUTDIR="${OUTDIR:-${OUT_ROOT}/snr_curve}"
echo "Comparing SNR curve for ${PILOT_TAG}: profile=${PROFILE}, outdir=${OUTDIR}"

"${PYTHON_BIN}" scripts/compare_grid_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  "${LMMSE_ARGS[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --outdir "${OUTDIR}/raw_comparison"

"${PYTHON_BIN}" scripts/plot_snr_curve.py \
  --csv "${OUTDIR}/raw_comparison/grid_method_comparison.csv" \
  --outdir "${OUTDIR}"
