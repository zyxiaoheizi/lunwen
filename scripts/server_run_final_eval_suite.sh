#!/usr/bin/env bash
set -euo pipefail

# Final evaluation suite for the paper stage. This script assumes the model
# checkpoints already exist. It does not retrain models by default.

PILOT_LIST="${PILOT_LIST:-8 24 48}"
RUN_MAIN_COMPARE="${RUN_MAIN_COMPARE:-1}"
RUN_ACTIVE_BASIS="${RUN_ACTIVE_BASIS:-1}"
RUN_SNR="${RUN_SNR:-1}"
RUN_P4_STRESS="${RUN_P4_STRESS:-0}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"

for pilots in ${PILOT_LIST}; do
  echo "==== Final compare: p${pilots} ===="
  if [[ "${RUN_MAIN_COMPARE}" == "1" ]]; then
    PILOTS="${pilots}" TEST_SAMPLES="${TEST_SAMPLES}" bash scripts/server_compare_grid_methods.sh
  fi
done

if [[ "${RUN_ACTIVE_BASIS}" == "1" ]]; then
  echo "==== Active-basis sparse inference: p8 ===="
  PILOTS=8 TEST_SAMPLES="${TEST_SAMPLES}" bash scripts/server_eval_active_basis.sh
fi

if [[ "${RUN_SNR}" == "1" ]]; then
  echo "==== SNR-NMSE curve: p8 ===="
  PILOTS=8 TEST_SAMPLES="${TEST_SAMPLES}" bash scripts/server_generate_snr_grid_tests.sh
  PILOTS=8 TEST_SAMPLES="${TEST_SAMPLES}" bash scripts/server_compare_snr_grid_methods.sh
fi

if [[ "${RUN_P4_STRESS}" == "1" ]]; then
  echo "==== Optional p4 extreme few-pilot stress test ===="
  PILOTS=4 TEST_SAMPLES="${TEST_SAMPLES}" TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}" VAL_SAMPLES="${VAL_SAMPLES:-4000}" bash scripts/server_generate_grid_datasets.sh
  PILOTS=4 EPOCHS="${P4_EPOCHS:-80}" EARLY_STOPPING_PATIENCE="${P4_PATIENCE:-10}" bash scripts/server_train_pf_msbnet.sh
  PILOTS=4 EPOCHS="${P4_EPOCHS:-80}" EARLY_STOPPING_PATIENCE="${P4_PATIENCE:-10}" bash scripts/server_train_pf_msbnet_a.sh
  PILOTS=4 TEST_SAMPLES="${TEST_SAMPLES}" bash scripts/server_compare_grid_methods.sh
fi

echo "Final evaluation suite complete."
