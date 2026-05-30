#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
SNR_LIST="${SNR_LIST:-0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30}"
N_SYMBOLS="${N_SYMBOLS:-14}"
N_SUBCARRIERS="${N_SUBCARRIERS:-72}"
N_TX="${N_TX:-2}"
N_RX="${N_RX:-2}"
N_TAPS="${N_TAPS:-4}"
PROFILE="${PROFILE:-tdl-a}"
CHANNEL="${CHANNEL:-rayleigh}"
MAX_DOPPLER_HZ="${MAX_DOPPLER_HZ:-70}"
DELAY_SPREAD_NS="${DELAY_SPREAD_NS:-300}"

mkdir -p data/grid

for snr in ${SNR_LIST}; do
  tag="${snr}"
  tag="${tag/-/m}"
  tag="${tag/./p}"
  out="data/grid/grid_${PILOT_TAG}_${PROFILE}_snr${tag}_test_${TEST_SAMPLES}.npz"
  echo "Generating SNR test: ${out}"
  "${PYTHON_BIN}" scripts/generate_grid_dataset.py \
    --samples "${TEST_SAMPLES}" \
    --n-symbols "${N_SYMBOLS}" \
    --n-subcarriers "${N_SUBCARRIERS}" \
    --n-tx "${N_TX}" \
    --n-rx "${N_RX}" \
    --taps "${N_TAPS}" \
    --profile "${PROFILE}" \
    --channel "${CHANNEL}" \
    --num-pilots "${PILOTS}" \
    --snr-min "${snr}" \
    --snr-max "${snr}" \
    --max-doppler-hz "${MAX_DOPPLER_HZ}" \
    --delay-spread-ns "${DELAY_SPREAD_NS}" \
    --seed "$((204000 + PILOTS * 100 + ${snr%.*}))" \
    --out "${out}"
done
