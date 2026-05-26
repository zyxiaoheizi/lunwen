#!/usr/bin/env bash
set -euo pipefail

# One-command entry for a pilot-count experiment.
# All settings can be overridden from the shell, for example:
# PILOTS=24 EPOCHS=80 bash scripts/server_run_pilot_experiment.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-16}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
VAL_SAMPLES="${VAL_SAMPLES:-4000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
EPOCHS="${EPOCHS:-120}"
SRCNN_EPOCHS="${SRCNN_EPOCHS:-${EPOCHS}}"
DNCNN_EPOCHS="${DNCNN_EPOCHS:-${EPOCHS}}"
REESNET_EPOCHS="${REESNET_EPOCHS:-${EPOCHS}}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"

export PYTHON_BIN
export PILOTS
export TRAIN_SAMPLES
export VAL_SAMPLES
export TEST_SAMPLES
export EPOCHS
export SRCNN_EPOCHS
export DNCNN_EPOCHS
export REESNET_EPOCHS
export EARLY_STOPPING_PATIENCE
export EARLY_STOPPING_MIN_DELTA
export BATCH_SIZE
export NUM_WORKERS
export DEVICE

echo "==== Pilot experiment ===="
echo "PILOTS=${PILOTS}"
echo "TRAIN/VAL/TEST=${TRAIN_SAMPLES}/${VAL_SAMPLES}/${TEST_SAMPLES}"
echo "EPOCHS SRCNN/DNCNN/ReEsNet=${SRCNN_EPOCHS}/${DNCNN_EPOCHS}/${REESNET_EPOCHS}"
echo "EARLY_STOPPING patience/min_delta=${EARLY_STOPPING_PATIENCE}/${EARLY_STOPPING_MIN_DELTA}"
echo "BATCH_SIZE=${BATCH_SIZE}, NUM_WORKERS=${NUM_WORKERS}, DEVICE=${DEVICE}"
echo "=========================="

bash scripts/server_generate_grid_datasets.sh
bash scripts/server_train_open_cnn_baselines.sh
bash scripts/server_compare_grid_methods.sh
