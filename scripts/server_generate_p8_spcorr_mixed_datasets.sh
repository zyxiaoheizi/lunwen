#!/usr/bin/env bash
set -euo pipefail

# Generate p8 datasets for testing whether MIMO-structured token encoding helps.
# Training/validation are mixed-profile and spatially correlated; tests are kept
# separate so A vs A2 can be inspected per scenario.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
TRAIN_PER_PROFILE="${TRAIN_PER_PROFILE:-8000}"
VAL_PER_PROFILE="${VAL_PER_PROFILE:-1000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
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
TRAIN_RHO="${TRAIN_RHO:-0.7}"

mkdir -p data/grid

run_one() {
  local samples="$1"
  local profile="$2"
  local channel="$3"
  local rho="$4"
  local seed="$5"
  local out="$6"
  local extra=()
  if [[ "${channel}" == "rician" ]]; then
    extra+=(--channel rician --rician-k "${RICIAN_K}")
  fi

  echo "Generating ${out}"
  "${PYTHON_BIN}" scripts/generate_grid_dataset.py \
    --samples "${samples}" \
    --n-symbols "${N_SYMBOLS}" \
    --n-subcarriers "${N_SUBCARRIERS}" \
    --n-tx "${N_TX}" \
    --n-rx "${N_RX}" \
    --taps "${N_TAPS}" \
    --profile "${profile}" \
    --num-pilots "${PILOTS}" \
    --snr-min "${SNR_MIN}" \
    --snr-max "${SNR_MAX}" \
    --max-doppler-hz "${MAX_DOPPLER_HZ}" \
    --delay-spread-ns "${DELAY_SPREAD_NS}" \
    --tx-corr-rho "${rho}" \
    --rx-corr-rho "${rho}" \
    --seed "${seed}" \
    "${extra[@]}" \
    --out "${out}"
}

echo "Generating mixed-profile spatially-correlated ${PILOT_TAG} data"
echo "train_per_profile=${TRAIN_PER_PROFILE}, val_per_profile=${VAL_PER_PROFILE}, test=${TEST_SAMPLES}, train_rho=${TRAIN_RHO}"

run_one "${TRAIN_PER_PROFILE}" tdl-a rayleigh "${TRAIN_RHO}" 203000 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_train_${TRAIN_PER_PROFILE}.npz"
run_one "${TRAIN_PER_PROFILE}" tdl-b rayleigh "${TRAIN_RHO}" 203001 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_b_train_${TRAIN_PER_PROFILE}.npz"
run_one "${TRAIN_PER_PROFILE}" tdl-c rayleigh "${TRAIN_RHO}" 203002 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_c_train_${TRAIN_PER_PROFILE}.npz"
run_one "${TRAIN_PER_PROFILE}" tdl-a rician "${TRAIN_RHO}" 203003 "data/grid/grid_${PILOT_TAG}_spcorr_rician_tdl_a_train_${TRAIN_PER_PROFILE}.npz"

run_one "${VAL_PER_PROFILE}" tdl-a rayleigh "${TRAIN_RHO}" 203100 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_val_${VAL_PER_PROFILE}.npz"
run_one "${VAL_PER_PROFILE}" tdl-b rayleigh "${TRAIN_RHO}" 203101 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_b_val_${VAL_PER_PROFILE}.npz"
run_one "${VAL_PER_PROFILE}" tdl-c rayleigh "${TRAIN_RHO}" 203102 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_c_val_${VAL_PER_PROFILE}.npz"
run_one "${VAL_PER_PROFILE}" tdl-a rician "${TRAIN_RHO}" 203103 "data/grid/grid_${PILOT_TAG}_spcorr_rician_tdl_a_val_${VAL_PER_PROFILE}.npz"

run_one "${TEST_SAMPLES}" tdl-a rayleigh 0.0 203200 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho00_test_${TEST_SAMPLES}.npz"
run_one "${TEST_SAMPLES}" tdl-a rayleigh 0.3 203201 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho03_test_${TEST_SAMPLES}.npz"
run_one "${TEST_SAMPLES}" tdl-a rayleigh 0.7 203202 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho07_test_${TEST_SAMPLES}.npz"
run_one "${TEST_SAMPLES}" tdl-a rayleigh 0.9 203203 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho09_test_${TEST_SAMPLES}.npz"
run_one "${TEST_SAMPLES}" tdl-b rayleigh "${TRAIN_RHO}" 203204 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_b_rho07_test_${TEST_SAMPLES}.npz"
run_one "${TEST_SAMPLES}" tdl-c rayleigh "${TRAIN_RHO}" 203205 "data/grid/grid_${PILOT_TAG}_spcorr_tdl_c_rho07_test_${TEST_SAMPLES}.npz"
run_one "${TEST_SAMPLES}" tdl-a rician "${TRAIN_RHO}" 203206 "data/grid/grid_${PILOT_TAG}_spcorr_rician_tdl_a_rho07_test_${TEST_SAMPLES}.npz"

echo "Done. Mixed spatial-correlation datasets are under data/grid"
