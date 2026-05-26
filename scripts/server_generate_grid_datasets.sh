#!/usr/bin/env bash
set -euo pipefail

# Paper-facing 2D time-frequency datasets.
# Defaults follow the ChannelNet/DeepPilotDesign style:
# 72 subcarriers x 14 OFDM symbols with 8/16/24/36/48 pilot positions.
# This round defaults to a moderately sparse setting: 16 pilots.

TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
VAL_SAMPLES="${VAL_SAMPLES:-4000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_SYMBOLS="${N_SYMBOLS:-14}"
N_SUBCARRIERS="${N_SUBCARRIERS:-72}"
N_TX="${N_TX:-2}"
N_RX="${N_RX:-2}"
N_TAPS="${N_TAPS:-4}"
SNR_MIN="${SNR_MIN:-0}"
SNR_MAX="${SNR_MAX:-30}"
MAX_DOPPLER_HZ="${MAX_DOPPLER_HZ:-70}"
DELAY_SPREAD_NS="${DELAY_SPREAD_NS:-300}"
RICIAN_K="${RICIAN_K:-5}"

mkdir -p data/grid

echo "Generating ${PILOT_TAG} datasets: ${TRAIN_SAMPLES}/${VAL_SAMPLES}/${TEST_SAMPLES} samples"
echo "grid=${N_SYMBOLS}x${N_SUBCARRIERS}, mimo=${N_TX}x${N_RX}, taps=${N_TAPS}, snr=[${SNR_MIN}, ${SNR_MAX}] dB, doppler=${MAX_DOPPLER_HZ} Hz"

echo "[1/6] Train: Rayleigh TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TRAIN_SAMPLES}" \
  --n-symbols "${N_SYMBOLS}" \
  --n-subcarriers "${N_SUBCARRIERS}" \
  --n-tx "${N_TX}" \
  --n-rx "${N_RX}" \
  --taps "${N_TAPS}" \
  --profile tdl-a \
  --num-pilots "${PILOTS}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --max-doppler-hz "${MAX_DOPPLER_HZ}" \
  --delay-spread-ns "${DELAY_SPREAD_NS}" \
  --seed 202700 \
  --out data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz

echo "[2/6] Val: Rayleigh TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${VAL_SAMPLES}" \
  --n-symbols "${N_SYMBOLS}" \
  --n-subcarriers "${N_SUBCARRIERS}" \
  --n-tx "${N_TX}" \
  --n-rx "${N_RX}" \
  --taps "${N_TAPS}" \
  --profile tdl-a \
  --num-pilots "${PILOTS}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --max-doppler-hz "${MAX_DOPPLER_HZ}" \
  --delay-spread-ns "${DELAY_SPREAD_NS}" \
  --seed 202701 \
  --out data/grid/grid_${PILOT_TAG}_tdl_a_val_${VAL_SAMPLES}.npz

echo "[3/6] Test: Rayleigh TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --n-symbols "${N_SYMBOLS}" \
  --n-subcarriers "${N_SUBCARRIERS}" \
  --n-tx "${N_TX}" \
  --n-rx "${N_RX}" \
  --taps "${N_TAPS}" \
  --profile tdl-a \
  --num-pilots "${PILOTS}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --max-doppler-hz "${MAX_DOPPLER_HZ}" \
  --delay-spread-ns "${DELAY_SPREAD_NS}" \
  --seed 202702 \
  --out data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz

echo "[4/6] Shift test: Rayleigh TDL-B"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --n-symbols "${N_SYMBOLS}" \
  --n-subcarriers "${N_SUBCARRIERS}" \
  --n-tx "${N_TX}" \
  --n-rx "${N_RX}" \
  --taps "${N_TAPS}" \
  --profile tdl-b \
  --num-pilots "${PILOTS}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --max-doppler-hz "${MAX_DOPPLER_HZ}" \
  --delay-spread-ns "${DELAY_SPREAD_NS}" \
  --seed 202703 \
  --out data/grid/grid_${PILOT_TAG}_tdl_b_test_${TEST_SAMPLES}.npz

echo "[5/6] Shift test: Rayleigh TDL-C"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --n-symbols "${N_SYMBOLS}" \
  --n-subcarriers "${N_SUBCARRIERS}" \
  --n-tx "${N_TX}" \
  --n-rx "${N_RX}" \
  --taps "${N_TAPS}" \
  --profile tdl-c \
  --num-pilots "${PILOTS}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --max-doppler-hz "${MAX_DOPPLER_HZ}" \
  --delay-spread-ns "${DELAY_SPREAD_NS}" \
  --seed 202704 \
  --out data/grid/grid_${PILOT_TAG}_tdl_c_test_${TEST_SAMPLES}.npz

echo "[6/6] Shift test: Rician TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --n-symbols "${N_SYMBOLS}" \
  --n-subcarriers "${N_SUBCARRIERS}" \
  --n-tx "${N_TX}" \
  --n-rx "${N_RX}" \
  --taps "${N_TAPS}" \
  --profile tdl-a \
  --channel rician \
  --rician-k "${RICIAN_K}" \
  --num-pilots "${PILOTS}" \
  --snr-min "${SNR_MIN}" \
  --snr-max "${SNR_MAX}" \
  --max-doppler-hz "${MAX_DOPPLER_HZ}" \
  --delay-spread-ns "${DELAY_SPREAD_NS}" \
  --seed 202705 \
  --out data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_${TEST_SAMPLES}.npz

echo "Done. Datasets are under data/grid"
